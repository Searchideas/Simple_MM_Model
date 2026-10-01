"""No-hedge SUIUSDC maker. Skew inventory flat. Pull toxic sides.

Trains on event-level fills already logged in the placement study
(days before 2026-09-08). Replays later days on the Binance level-2 book.
Fair is the Binance microprice. There is no Hyperliquid hedge.
"""

from __future__ import annotations

import gc
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from src.loaders import (
    available_dates,
    load_bbo_range,
    load_incremental_l2_day,
    load_trades_day,
    merge_l2_and_trades,
)
from src.sim.event_l2_queue import EventL2Queue
from src.sim.markout import MarkoutTracker
from src.sim.mexc_fair_bt import _ceil_tick, _floor_tick
from src.sim.multi_book_backtest import align_cross_books
from src.sim.run_event_l2_fair_bt import _flow_against
from src.sim.toxicity import TradeOFI, quote_features, signed_trade_qty

TICK = 0.0001
ORDER_SIZE = 10.0
MAX_INV = 10.0
OPEN_TICKS = 5.0
HOT_TICKS = 5.0
TP_TICKS = 1.0
CUT_TICKS = 8.0
PULL_P = 0.70
TRAIN_BEFORE = "2026-09-08"
FEATURES = (
    "side",
    "micro_gap_signed",
    "ofi_100_signed",
    "ofi_500_signed",
    "move_2s_ticks",
)


def train_adverse_model() -> tuple[XGBClassifier | None, dict]:
    path = Path("results") / "placement_study.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["day"] >= TRAIN_BEFORE:
            continue
        for fill in rec["fill_rows"]:
            if fill.get("markout_1s") is None:
                continue
            if any(fill.get(k) is None for k in FEATURES if k != "side"):
                continue
            rows.append(fill)
    report = {"n_train": len(rows)}
    if len(rows) < 40:
        report["error"] = "too few training fills"
        return None, report
    x = np.asarray([[float(r[k]) for k in FEATURES] for r in rows], dtype=float)
    y = np.asarray([1 if r["markout_1s"] < 0 else 0 for r in rows], dtype=int)
    report["train_adverse_rate"] = float(y.mean())
    if y.min() == y.max():
        report["error"] = "one class"
        return None, report
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    model = XGBClassifier(
        max_depth=3,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        n_jobs=1,
        random_state=0,
    )
    model.fit(x, y)
    report["importance"] = {
        name: float(score)
        for name, score in zip(FEATURES, model.feature_importances_, strict=True)
    }
    return model, report


def _toxic(model: XGBClassifier | None, feat: dict) -> bool:
    if model is None:
        return False
    x = np.asarray([[float(feat[k]) for k in FEATURES]], dtype=float)
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return False
    return float(proba[0, classes.index(1)]) > PULL_P


def run_day(day: str, model: XGBClassifier | None) -> dict:
    print(f"=== {day} skew-mm ===", flush=True)
    fair = load_bbo_range("suiusdt", day, day, every="1s", exchange="hyperliquid")
    trade = load_bbo_range("suiusdc", day, day, every="1s", exchange="binance-futures")
    fx = load_bbo_range("usdcusdt", day, day, every="1s", exchange="binance")
    timeline = align_cross_books(fair, trade, fx, include_l25=False, hedge_bbo=fair)
    del fair, trade, fx
    gc.collect()
    l2 = load_incremental_l2_day("suiusdc", day, exchange="binance-futures")
    trades = load_trades_day("suiusdc", day, exchange="binance-futures")
    events = merge_l2_and_trades(l2, trades)
    print(f"  events={events.height}", flush=True)
    del l2, trades
    gc.collect()

    queue = EventL2Queue(
        tick=TICK, order_size=ORDER_SIZE, max_inventory=MAX_INV, maker_fee_rate=0.0
    )
    markout = MarkoutTracker(tick=TICK, horizon_seconds=1.0)
    ev = events.with_columns(
        pl.when(pl.col("kind") == "trade").then(0).otherwise(1).alias("_ord")
    ).sort(["ts", "_ord"])
    ev_ts = ev["ts"].to_list()
    ev_kind = ev["kind"].to_list()
    ev_snap = ev["is_snapshot"].to_list()
    ev_side = ev["side"].to_list()
    ev_price = ev["price"].to_list()
    ev_amt = ev["amount"].to_list()
    n_ev = len(ev_ts)
    rows = list(timeline.sort("ts").iter_rows(named=True))
    del timeline, events, ev
    gc.collect()

    ev_i = 0
    ofi_100 = TradeOFI(100_000)
    ofi_500 = TradeOFI(500_000)
    mid_hist: deque[tuple[datetime, float]] = deque()
    last_mid = 0.0
    pulls = 0

    def drain(ts: datetime) -> None:
        nonlocal ev_i
        while ev_i < n_ev and ev_ts[ev_i] <= ts:
            if ev_kind[ev_i] == "book":
                queue.on_book_row(
                    side=str(ev_side[ev_i]),
                    price=float(ev_price[ev_i]),
                    amount=float(ev_amt[ev_i]),
                    is_snapshot=bool(ev_snap[ev_i]),
                )
            else:
                mid = queue.book.mid()
                amt = float(ev_amt[ev_i])
                fills = queue.on_trade(
                    timestamp=ev_ts[ev_i],
                    price=float(ev_price[ev_i]),
                    amount=amt,
                    aggressor_side=str(ev_side[ev_i]),
                    mid=mid,
                )
                for fill in fills:
                    markout.on_fill(
                        side=fill.side,
                        mid=fill.mid_at_fill or mid,
                        price=fill.price,
                        quantity=fill.quantity,
                        ts=fill.timestamp,
                    )
                signed = signed_trade_qty(str(ev_side[ev_i]), amt)
                ofi_100.update(ev_ts[ev_i], signed)
                ofi_500.update(ev_ts[ev_i], signed)
            ev_i += 1

    for row in rows:
        ts = row["ts"]
        drain(ts)
        bb, bq = queue.book.best_bid()
        ba, aq = queue.book.best_ask()
        mid = queue.book.mid()
        if mid <= 0:
            mid = float(row["trade_mid"])
        last_mid = mid
        mid_hist.append((ts, mid))
        cutoff = ts.timestamp() - 2.0
        while len(mid_hist) > 1 and mid_hist[0][0].timestamp() < cutoff:
            mid_hist.popleft()
        move_2s = 0.0
        if mid_hist and mid_hist[0][1] > 0 and mid > 0:
            move_2s = abs(mid - mid_hist[0][1]) / TICK
        markout.on_mid(mid, ts)

        inv = queue.position.inventory
        entry = queue.position.avg_entry_price
        flat = abs(inv) <= 1e-9
        enable_bid = flat
        enable_ask = flat
        if flat and move_2s >= HOT_TICKS:
            enable_bid = False
            enable_ask = False
            pulls += 1

        bid = ask = 0.0
        if bb > 0 and ba > bb and bq + aq > 0:
            micro = (bb * aq + ba * bq) / (bq + aq)
            common = dict(
                tick=TICK,
                bid=bb,
                ask=ba,
                bid_qty=bq,
                ask_qty=aq,
                order_qty=ORDER_SIZE,
                ofi_100=ofi_100.value(ts),
                ofi_500=ofi_500.value(ts),
                hl_bid=bb,
                hl_ask=ba,
                fx=1.0,
                buffer_ticks=OPEN_TICKS,
            )
            bid_feat = quote_features(
                side=1,
                queue_ahead=0.0,
                **common,
            )
            ask_feat = quote_features(
                side=-1,
                queue_ahead=0.0,
                **common,
            )
            bid_feat["move_2s_ticks"] = move_2s
            ask_feat["move_2s_ticks"] = move_2s
            if enable_bid and (
                _flow_against(bid_feat, "either") or _toxic(model, bid_feat)
            ):
                enable_bid = False
                pulls += 1
            if enable_ask and (
                _flow_against(ask_feat, "either") or _toxic(model, ask_feat)
            ):
                enable_ask = False
                pulls += 1
            if inv > 1e-9:
                enable_bid = False
                enable_ask = True
                loss_ticks = (entry - mid) / TICK if entry > 0 and mid > 0 else 0.0
                if loss_ticks >= CUT_TICKS:
                    ask = _ceil_tick(ba, TICK)
                else:
                    ask = max(
                        _ceil_tick(entry + TP_TICKS * TICK, TICK),
                        _ceil_tick(ba, TICK),
                    )
                bid = 0.0
            elif inv < -1e-9:
                enable_ask = False
                enable_bid = True
                loss_ticks = (mid - entry) / TICK if entry > 0 and mid > 0 else 0.0
                if loss_ticks >= CUT_TICKS:
                    bid = _floor_tick(bb, TICK)
                else:
                    bid = min(
                        _floor_tick(entry - TP_TICKS * TICK, TICK),
                        _floor_tick(bb, TICK),
                    )
                ask = 0.0
            else:
                bid = min(_floor_tick(micro - OPEN_TICKS * TICK, TICK), _floor_tick(bb, TICK))
                ask = max(_ceil_tick(micro + OPEN_TICKS * TICK, TICK), _ceil_tick(ba, TICK))

        if bid <= 0 or ask <= bid or (not enable_bid and not enable_ask):
            if enable_bid and bid > 0 and (ask <= bid or not enable_ask):
                queue.post(bid=bid, ask=0.0, bid_enabled=True, ask_enabled=False, size=ORDER_SIZE)
            elif enable_ask and ask > 0 and (bid <= 0 or not enable_bid):
                queue.post(bid=0.0, ask=ask, bid_enabled=False, ask_enabled=True, size=ORDER_SIZE)
            else:
                queue.post(bid=0.0, ask=0.0, bid_enabled=False, ask_enabled=False)
        else:
            queue.post(
                bid=bid,
                ask=ask,
                bid_enabled=enable_bid,
                ask_enabled=enable_ask,
                size=ORDER_SIZE,
            )

    while ev_i < n_ev:
        if ev_kind[ev_i] == "book":
            queue.on_book_row(
                side=str(ev_side[ev_i]),
                price=float(ev_price[ev_i]),
                amount=float(ev_amt[ev_i]),
                is_snapshot=bool(ev_snap[ev_i]),
            )
        else:
            mid = queue.book.mid()
            fills = queue.on_trade(
                timestamp=ev_ts[ev_i],
                price=float(ev_price[ev_i]),
                amount=float(ev_amt[ev_i]),
                aggressor_side=str(ev_side[ev_i]),
                mid=mid,
            )
            for fill in fills:
                markout.on_fill(
                    side=fill.side,
                    mid=fill.mid_at_fill or mid,
                    price=fill.price,
                    quantity=fill.quantity,
                    ts=fill.timestamp,
                )
        ev_i += 1

    pnl = queue.position.marked_pnl(last_mid)
    report = markout.report()
    print(
        f"  fills={len(queue.fills)} inv={queue.position.inventory:.1f} "
        f"PnL={pnl:+.4f} adv={100 * report.adverse_rate:.1f}% "
        f"mean_markout={report.mean_markout_ticks:+.2f} pulls={pulls}",
        flush=True,
    )
    return {
        "day": day,
        "fills": len(queue.fills),
        "pnl": pnl,
        "inv": queue.position.inventory,
        "adverse": report.adverse_rate,
        "markout": report.mean_markout_ticks,
    }


def main() -> None:
    model, report = train_adverse_model()
    print(f"model {report}", flush=True)
    have = set(
        available_dates("suiusdc", data_type="incremental_book_L2", exchange="binance-futures")
    )
    days = [
        d
        for d in (
            "2026-09-08",
            "2026-09-09",
            "2026-09-10",
            "2026-09-11",
            "2026-09-12",
            "2026-09-13",
        )
        if d in have
    ]
    print(
        f"Skew MM no hedge  open={OPEN_TICKS:g}t tp={TP_TICKS:g}t cut={CUT_TICKS:g}t "
        f"hot={HOT_TICKS:g}t pull_p={PULL_P} size={ORDER_SIZE:g} max_inv={MAX_INV:g} days={days}",
        flush=True,
    )
    totals = [run_day(day, model) for day in days]
    fills = sum(r["fills"] for r in totals)
    pnl = sum(r["pnl"] for r in totals)
    adv = sum(r["adverse"] * r["fills"] for r in totals) / fills if fills else 0.0
    print(f"TOTAL fills={fills} PnL={pnl:+.4f} adverse={100 * adv:.1f}%", flush=True)


if __name__ == "__main__":
    main()
