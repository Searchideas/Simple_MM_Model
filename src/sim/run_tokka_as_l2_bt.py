"""Event-L2 replay of the Tokka SUI Avellaneda-Stoikov inside quote.

Parameters are the sodex-testnet-perp-sui bot: reference fair, spread
1/k_eff + gamma*sigma^2/2, clamped to 1.25–2.5 bp, inventory skew.
Sodex has no tape here, so the quote rests on Binance SUIUSDC and the
reference mid is the Hyperliquid book (already USDC, no FX).
Only the inside level is queued. Deeper ladder levels sit behind it.
"""

from __future__ import annotations

import gc
from collections import deque
from datetime import datetime

import polars as pl

from src.loaders import (
    available_dates,
    load_bbo_range,
    load_incremental_l2_day,
    load_trades_day,
    merge_l2_and_trades,
    resolve_date_range,
)
from src.sim.event_l2_queue import EventL2Queue
from src.sim.markout import MarkoutTracker
from src.sim.mexc_fair_bt import _ceil_tick, _floor_tick
from src.sim.multi_book_backtest import align_cross_books

# sodex-testnet-perp-sui-mm-uat.yaml
K = 4400.0
ALPHA = 0.264
BETA = 2.2
GAMMA = 0.016
MIN_SPREAD = 0.000125
MAX_SPREAD = 0.00025
MAX_VOL = 0.08
INV_MULT = 0.5
MAX_INV_FACTOR = 2.0
TICK = 0.0001
ORDER_SIZE = 10.0
MAX_INV = 50.0


def _k_eff() -> float:
    ratio = min(ALPHA / BETA, 0.95)
    return min(100000.0, max(1.0, K / (1.0 - ratio)))


def _spreads(sigma: float, inventory: float) -> tuple[float, float, float]:
    inventory_ratio = inventory / MAX_INV if MAX_INV else 0.0
    inventory_factor = min(MAX_INV_FACTOR, 1.0 + INV_MULT * abs(inventory_ratio))
    base = 1.0 / _k_eff() + GAMMA * sigma * sigma / 2.0
    if inventory > 0:
        bid_spread, ask_spread = base * inventory_factor, base / inventory_factor
    elif inventory < 0:
        bid_spread, ask_spread = base / inventory_factor, base * inventory_factor
    else:
        bid_spread = ask_spread = base
    skew = (GAMMA * sigma * sigma / 2.0) * inventory_ratio
    skew = max(-MIN_SPREAD / 4.0, min(MIN_SPREAD / 4.0, skew))
    return (
        min(MAX_SPREAD, max(MIN_SPREAD, bid_spread)),
        min(MAX_SPREAD, max(MIN_SPREAD, ask_spread)),
        skew,
    )


def run_day(day: str) -> dict:
    print(f"=== {day} tokka-as ===", flush=True)
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
    rets: deque[float] = deque(maxlen=60)
    prev_mid = 0.0
    last_mid = 0.0

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

    for row in rows:
        ts = row["ts"]
        hl_bid = float(row["ref_bid"])
        hl_ask = float(row["ref_ask"])
        drain(ts)
        bb, _ = queue.book.best_bid()
        ba, _ = queue.book.best_ask()
        bn_mid = queue.book.mid()
        if bn_mid <= 0:
            bn_mid = float(row["trade_mid"])
        last_mid = bn_mid
        if prev_mid > 0 and bn_mid > 0:
            rets.append((bn_mid - prev_mid) / prev_mid)
        prev_mid = bn_mid
        if hl_bid <= 0 or hl_ask <= hl_bid:
            queue.post(bid=0, ask=0, bid_enabled=False, ask_enabled=False)
            continue
        fair_px = 0.5 * (hl_bid + hl_ask)
        var = 0.0
        if len(rets) >= 5:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / len(rets)
        sigma = min(MAX_VOL, var ** 0.5)
        bid_spread, ask_spread, skew = _spreads(sigma, queue.position.inventory)
        raw_bid = _floor_tick(fair_px * (1.0 - bid_spread - skew), TICK)
        raw_ask = _ceil_tick(fair_px * (1.0 + ask_spread - skew), TICK)
        if bb > 0:
            raw_bid = min(raw_bid, _floor_tick(bb, TICK))
        if ba > 0:
            raw_ask = max(raw_ask, _ceil_tick(ba, TICK))
        inv = queue.position.inventory
        enable_bid = inv + ORDER_SIZE <= MAX_INV
        enable_ask = inv - ORDER_SIZE >= -MAX_INV
        if inv >= MAX_INV - 1e-12:
            enable_bid, enable_ask = False, True
        elif inv <= -MAX_INV + 1e-12:
            enable_bid, enable_ask = True, False
        if raw_bid <= 0 or raw_ask <= raw_bid or (not enable_bid and not enable_ask):
            queue.post(bid=0, ask=0, bid_enabled=False, ask_enabled=False)
        else:
            queue.post(
                bid=raw_bid,
                ask=raw_ask,
                bid_enabled=enable_bid,
                ask_enabled=enable_ask,
                size=ORDER_SIZE,
            )
        markout.on_mid(bn_mid, ts)

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
        f"PnL={pnl:+.4f} adv={100*report.adverse_rate:.1f}% "
        f"mean_markout={report.mean_markout_ticks:+.2f}",
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
    _, _, days = resolve_date_range(
        "suiusdc",
        from_date="2026-08-26",
        to_date="2026-09-13",
        exchange="binance-futures",
    )
    l2_days = set(
        available_dates("suiusdc", data_type="incremental_book_L2", exchange="binance-futures")
    )
    want = [d for d in ("2026-08-26", "2026-08-28", "2026-09-02", "2026-09-08", "2026-09-11") if d in days and d in l2_days]
    print(
        f"Tokka AS inside quote  k_eff={_k_eff():.0f} spread=[{MIN_SPREAD*1e4:.2f},{MAX_SPREAD*1e4:.2f}]bp "
        f"size={ORDER_SIZE:g} max_inv={MAX_INV:g} days={len(want)}",
        flush=True,
    )
    totals = []
    for day in want:
        totals.append(run_day(day))
    fills = sum(r["fills"] for r in totals)
    pnl = sum(r["pnl"] for r in totals)
    adv = sum(r["adverse"] * r["fills"] for r in totals) / fills if fills else 0.0
    print(
        f"TOTAL fills={fills} PnL={pnl:+.4f} adverse={100*adv:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
