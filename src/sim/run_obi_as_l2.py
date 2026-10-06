"""Unhedged Avellaneda-Stoikov on the Binance SUIUSDC level-2 queue.

The only extra gate is the touch imbalance pull: OBI >= +0.8 turns the
ask off, OBI <= -0.8 turns the bid off. There is no hedge.
"""

from __future__ import annotations

import gc
from datetime import datetime

import polars as pl

from src.loaders import (
    available_dates,
    load_bbo_range,
    load_incremental_l2_day,
    load_trades_day,
    merge_l2_and_trades,
)
from src.sim.event_l2_queue import EventL2Queue
from src.sim.markout import MarkoutTracker
from src.sim.market_state import MarketState, PositionState
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters

TICK = 0.0001
ORDER_SIZE = 10.0
MAX_INV = 50.0
VOL_WINDOW = 120


def run_day(day: str, model: AvellanedaStoikovModel) -> dict:
    print(f"=== {day} obi={model.params.obi_pull_level:g} ===", flush=True)
    bbo = load_bbo_range("suiusdc", day, day, every="1s", exchange="binance-futures")
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
    rows = list(bbo.sort("ts").iter_rows(named=True))
    del bbo, events, ev
    gc.collect()

    ev_i = 0
    mids: list[float] = []
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
        drain(ts)
        bb, bq = queue.book.best_bid()
        ba, aq = queue.book.best_ask()
        mid = queue.book.mid()
        if mid <= 0:
            mid = float(row["mid"])
        if mid <= 0:
            continue
        last_mid = mid
        mids.append(mid)
        recent = mids[-VOL_WINDOW:]
        vol = 0.0
        if len(recent) >= 3 and min(recent) > 0:
            vol = MarketState.calculate_volatility(recent, 1.0)
        markout.on_mid(mid, ts)

        market = MarketState(
            timestamp=ts,
            best_bid=bb if bb > 0 else float(row["bid_price"]),
            best_bid_volume=bq,
            best_ask=ba if ba > 0 else float(row["ask_price"]),
            best_ask_volume=aq,
            volatility=vol,
        )
        pos = PositionState(
            inventory=queue.position.inventory,
            cash=queue.position.cash,
            avg_entry_price=queue.position.avg_entry_price,
        )
        quote = model.calculate_quotes(market, pos)
        if quote.action != "QUOTE":
            pulls += 1
        queue.post(
            bid=quote.bid,
            ask=quote.ask,
            bid_enabled=quote.bid_enabled,
            ask_enabled=quote.ask_enabled,
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
        "fills": len(queue.fills),
        "pnl": pnl,
        "adverse": report.adverse_rate,
        "markout": report.mean_markout_ticks,
    }


def main() -> None:
    have = set(
        available_dates("suiusdc", data_type="incremental_book_L2", exchange="binance-futures")
    )
    days = [d for d in sorted(have) if "2026-08-26" <= d <= "2026-09-13"]
    print(f"Unhedged AS L2  days={len(days)} size={ORDER_SIZE:g} max_inv={MAX_INV:g}", flush=True)
    for pull in (0.8, 0.0):
        model = AvellanedaStoikovModel(
            QuoteParameters(
                base_gamma=0.01,
                kappa=0.25,
                time_horizon=300.0,
                min_spread=0.0004032,
                tick_size=TICK,
                max_inventory=MAX_INV,
                max_spread_ticks=10.0,
                maker_fee_rate=0.0,
                obi_pull_level=pull,
            )
        )
        print(f"\n## obi_pull_level={pull:g}", flush=True)
        totals = [run_day(day, model) for day in days]
        fills = sum(r["fills"] for r in totals)
        pnl = sum(r["pnl"] for r in totals)
        adv = sum(r["adverse"] * r["fills"] for r in totals) / fills if fills else 0.0
        mk = sum(r["markout"] * r["fills"] for r in totals) / fills if fills else 0.0
        print(
            f"TOTAL obi={pull:g} fills={fills} PnL={pnl:+.4f} "
            f"adverse={100 * adv:.1f}% mean_markout={mk:+.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
