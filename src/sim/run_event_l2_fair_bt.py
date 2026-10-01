"""MEXC-fair → SUIUSDC maker with **event-L2 queue** fills + MEXC B2B hedge.

Uses Tardis ``incremental_book_L2`` + ``trades`` on SUIUSDC for queue/partial
fills. Fair still from MEXC÷USDCUSDT; requote on a 1s grid.
"""

from __future__ import annotations

import argparse
import gc
from collections import Counter
from datetime import datetime, timedelta

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
from src.sim.metrics import build_metrics
from src.sim.mexc_fair_bt import MexcFairParams, _ceil_tick, _floor_tick
from src.sim.mexc_hedge import MexcTakerHedge
from src.sim.multi_book_backtest import align_cross_books
from config import FEE_BY_EXCHANGE
from src.sim.toxicity import (
    TradeOFI,
    attach_markouts,
    locked_edge_ticks,
    quote_features,
    signed_trade_qty,
)


def _intersect_days(
    exchanges_symbols: list[tuple[str, str]],
    date: str,
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str, list[str]]:
    primary_ex, primary_sym = exchanges_symbols[0]
    _, _, ref_days = resolve_date_range(
        primary_sym,
        date=date,
        from_date=from_date,
        to_date=to_date,
        exchange=primary_ex,
    )
    picked = list(ref_days)
    for ex, sym in exchanges_symbols[1:]:
        have = set(available_dates(sym, exchange=ex))
        # Prefer L2 dates for trade symbol when available.
        if ex == exchanges_symbols[1][0]:
            l2 = set(
                available_dates(sym, data_type="incremental_book_L2", exchange=ex)
            )
            if l2:
                have &= l2
        picked = [d for d in picked if d in have]
    if not picked:
        raise FileNotFoundError(
            "No overlapping dates (need incremental_book_L2 on trade symbol). "
            + ", ".join(f"{s}@{e}" for e, s in exchanges_symbols)
        )
    return picked[0], picked[-1], picked


def _build_quote_prices(
    *,
    mx_bid: float,
    mx_ask: float,
    fx_mid: float,
    trade_bid: float,
    trade_ask: float,
    tick: float,
    buffer_ticks: float,
    bid_extra_ticks: float = 0.0,
    ask_extra_ticks: float = 0.0,
) -> tuple[float, float] | None:
    if not (0 < mx_bid < mx_ask) or fx_mid <= 0:
        return None
    if not (0 < trade_bid < trade_ask):
        return None
    bid_off = (buffer_ticks + max(0.0, bid_extra_ticks)) * tick
    ask_off = (buffer_ticks + max(0.0, ask_extra_ticks)) * tick
    fair_bid = mx_bid / fx_mid
    fair_ask = mx_ask / fx_mid
    bid = min(_floor_tick(fair_bid - bid_off, tick), _floor_tick(trade_bid, tick))
    ask = max(_ceil_tick(fair_ask + ask_off, tick), _ceil_tick(trade_ask, tick))
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return bid, ask


def _flow_against(feat: dict[str, float], mode: str) -> bool:
    """True when this side should be pulled. Signed features are negative against the quote."""
    if mode in ("", "off"):
        return False
    micro_against = feat["micro_gap_signed"] < 0.0
    ofi_against = feat["ofi_100_signed"] < 0.0 or feat["ofi_500_signed"] < 0.0
    if mode == "micro":
        return micro_against
    if mode == "ofi":
        return ofi_against
    if mode == "both":
        return micro_against and ofi_against
    return micro_against or ofi_against


def _append_fill_rows(
    fill_rows: list[dict] | None,
    fills: list,
    queue: EventL2Queue,
    *,
    day: str,
    tick: float,
    mx_bid: float,
    mx_ask: float,
    fx: float,
) -> None:
    if fill_rows is None or not fills:
        return
    contexts = queue.fill_context[-len(fills) :]
    for fill, ctx in zip(fills, contexts, strict=True):
        if not ctx:
            continue
        edge = locked_edge_ticks(
            side=fill.side,
            fill_price=fill.price,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            fx=fx,
            tick=tick,
        )
        fill_rows.append(
            {
                **ctx,
                "day": day,
                "ts": fill.timestamp,
                "price": fill.price,
                "qty": fill.quantity,
                "mid0": fill.mid_at_fill,
                "edge_ticks": edge,
            }
        )


def run_day(
    *,
    day: str,
    every: str,
    fair_symbol: str,
    trade_symbol: str,
    fx_symbol: str,
    fair_exchange: str,
    trade_exchange: str,
    fx_exchange: str,
    params: MexcFairParams,
    maker_fee: float,
    hedge_taker_fee: float,
    markout_horizon: float,
    policy: object | None = None,
    fill_rows: list[dict] | None = None,
) -> dict:
    print(f"=== {day} event-L2 ===", flush=True)
    mexc_bbo = load_bbo_range(
        fair_symbol, day, day, every=every, exchange=fair_exchange
    )
    trade_bbo = load_bbo_range(
        trade_symbol, day, day, every=every, exchange=trade_exchange
    )
    fx_bbo = load_bbo_range(fx_symbol, day, day, every=every, exchange=fx_exchange)
    timeline = align_cross_books(
        mexc_bbo, trade_bbo, fx_bbo, include_l25=False, hedge_bbo=mexc_bbo
    )
    del mexc_bbo, trade_bbo, fx_bbo
    gc.collect()

    l2 = load_incremental_l2_day(trade_symbol, day, exchange=trade_exchange)
    trades = load_trades_day(trade_symbol, day, exchange=trade_exchange)
    events = merge_l2_and_trades(l2, trades)
    print(
        f"  timeline={timeline.height} l2={l2.height} trades={trades.height} "
        f"events={events.height}",
        flush=True,
    )
    del l2, trades
    gc.collect()

    queue = EventL2Queue(
        tick=params.trade_tick,
        order_size=params.order_size,
        max_inventory=params.max_inventory,
        maker_fee_rate=maker_fee,
    )
    hedge = MexcTakerHedge(
        taker_fee_rate=hedge_taker_fee,
        max_basis_bps=0.0,
        favorable_only=False,
    )
    markout = MarkoutTracker(
        tick=params.trade_tick, horizon_seconds=markout_horizon
    )
    action_steps: Counter[str] = Counter()
    equity_curve: list[tuple[datetime, float]] = []
    max_abs_inv = 0.0

    timeline_rows = list(timeline.sort("ts").iter_rows(named=True))
    # Columnar event arrays (avoid 2.7M named dicts).
    ev = events.sort("ts")
    # Ensure trade-before-book on ties.
    ev = ev.with_columns(
        pl.when(pl.col("kind") == "trade").then(0).otherwise(1).alias("_ord")
    ).sort(["ts", "_ord"])
    ev_ts = ev["ts"].to_list()
    ev_kind = ev["kind"].to_list()
    ev_snap = ev["is_snapshot"].to_list()
    ev_side = ev["side"].to_list()
    ev_price = ev["price"].to_list()
    ev_amt = ev["amount"].to_list()
    n_ev = len(ev_ts)
    del timeline, events, ev
    gc.collect()

    ev_i = 0
    last_trade_mid = 0.0
    last_fx = 0.0
    last_mx_mid = 0.0
    last_mx_bid = 0.0
    last_mx_ask = 0.0
    last_eq_ts: datetime | None = None
    mid_hist: list[tuple[datetime, float]] = []
    pause_until: datetime | None = None
    pause_steps = 0
    pull_bid = 0
    pull_ask = 0
    ofi_100 = TradeOFI(100_000)
    ofi_500 = TradeOFI(500_000)
    mid_ts: list[datetime] = []
    mid_px: list[float] = []
    row_start = len(fill_rows) if fill_rows is not None else 0

    def note_mid(ts: datetime, mid: float) -> None:
        if mid > 0:
            mid_ts.append(ts)
            mid_px.append(float(mid))

    def handle_fills(fills: list) -> None:
        for fill in fills:
            markout.on_fill(
                side=fill.side,
                mid=fill.mid_at_fill or fill.price,
                price=fill.price,
                quantity=fill.quantity,
                ts=fill.timestamp,
            )
            hedge.on_mm_fill(
                fill,
                mx_bid=last_mx_bid,
                mx_ask=last_mx_ask,
                mx_mid=last_mx_mid,
                bn_usdt_mid=last_mx_mid,
                fx_mid=last_fx,
            )
        _append_fill_rows(
            fill_rows,
            fills,
            queue,
            day=day,
            tick=params.trade_tick,
            mx_bid=last_mx_bid,
            mx_ask=last_mx_ask,
            fx=last_fx,
        )

    def drain_until(ts: datetime) -> None:
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
                note_mid(ev_ts[ev_i], mid)
                trade_amt = float(ev_amt[ev_i])
                fills = queue.on_trade(
                    timestamp=ev_ts[ev_i],
                    price=float(ev_price[ev_i]),
                    amount=trade_amt,
                    aggressor_side=str(ev_side[ev_i]),
                    mid=mid,
                    action="QUOTE",
                )
                handle_fills(fills)
                signed = signed_trade_qty(str(ev_side[ev_i]), trade_amt)
                ofi_100.update(ev_ts[ev_i], signed)
                ofi_500.update(ev_ts[ev_i], signed)
            ev_i += 1

    for row in timeline_rows:
        ts = row["ts"]
        mx_bid = float(row["ref_bid"])
        mx_ask = float(row["ref_ask"])
        mx_mid = float(row.get("hedge_mid") or row["ref_mid"])
        trade_bid = float(row["trade_bid"])
        trade_ask = float(row["trade_ask"])
        trade_mid = float(row["trade_mid"])
        fx_mid = float(row["fx_mid"])
        last_trade_mid = trade_mid
        last_fx = fx_mid
        last_mx_mid = mx_mid
        last_mx_bid = mx_bid
        last_mx_ask = mx_ask

        drain_until(ts)
        markout.on_mid(trade_mid, ts)
        note_mid(ts, trade_mid)
        mid_hist.append((ts, trade_mid))
        lookback = params.fast_move_lookback_seconds
        if lookback > 0 and params.fast_move_ticks > 0:
            cutoff = ts.timestamp() - lookback
            while len(mid_hist) > 1 and mid_hist[0][0].timestamp() < cutoff:
                mid_hist.pop(0)
            if mid_hist and mid_hist[0][1] > 0 and trade_mid > 0:
                move_ticks = abs(trade_mid - mid_hist[0][1]) / params.trade_tick
                if move_ticks >= params.fast_move_ticks:
                    pause_until = ts + timedelta(
                        seconds=max(0.0, params.fast_move_pause_seconds)
                    )

        bb, bq = queue.book.best_bid()
        ba, aq = queue.book.best_ask()
        if bb > 0 and ba > 0:
            trade_bid, trade_ask = bb, ba

        inv = queue.position.inventory
        soft = max(0.0, params.soft_inventory_lots) * params.order_size
        enable_bid = (
            inv <= soft and inv + params.order_size <= params.max_inventory
        )
        enable_ask = (
            inv >= -soft and inv - params.order_size >= -params.max_inventory
        )
        if inv >= params.max_inventory - 1e-12:
            enable_bid, enable_ask = False, True
        elif inv <= -params.max_inventory + 1e-12:
            enable_bid, enable_ask = True, False

        paused = pause_until is not None and ts < pause_until
        if paused:
            enable_bid = False
            enable_ask = False
            pause_steps += 1
            action_steps["PAUSE"] += 1

        bid_extra = params.bid_skew_ticks
        ask_extra = params.ask_skew_ticks
        if params.inv_skew_ticks > 0 and params.order_size > 0:
            lots = abs(inv) / params.order_size
            if inv > 0:
                bid_extra += params.inv_skew_ticks * lots
            elif inv < 0:
                ask_extra += params.inv_skew_ticks * lots

        def _prices(bid_off: float, ask_off: float) -> tuple[float, float] | None:
            return _build_quote_prices(
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                fx_mid=fx_mid,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                tick=params.trade_tick,
                buffer_ticks=params.maker_buffer_ticks,
                bid_extra_ticks=bid_off,
                ask_extra_ticks=ask_off,
            )

        base = _prices(bid_extra, ask_extra)
        bid_feat = None
        ask_feat = None
        if base is not None and trade_bid > 0 and trade_ask > trade_bid:
            bpx, apx = base
            ofi100 = ofi_100.value(ts)
            ofi500 = ofi_500.value(ts)
            common = dict(
                tick=params.trade_tick,
                bid=trade_bid,
                ask=trade_ask,
                bid_qty=bq,
                ask_qty=aq,
                order_qty=params.order_size,
                ofi_100=ofi100,
                ofi_500=ofi500,
                hl_bid=mx_bid,
                hl_ask=mx_ask,
                fx=fx_mid,
                buffer_ticks=params.maker_buffer_ticks,
            )
            bid_feat = quote_features(
                side=1,
                queue_ahead=queue.book.size_ahead(side="bid", our_price=bpx),
                **common,
            )
            ask_feat = quote_features(
                side=-1,
                queue_ahead=queue.book.size_ahead(side="ask", our_price=apx),
                **common,
            )
            move_2s = 0.0
            if mid_hist and mid_hist[0][1] > 0 and trade_mid > 0 and params.trade_tick > 0:
                move_2s = abs(trade_mid - mid_hist[0][1]) / params.trade_tick
            if bid_feat is not None:
                bid_feat["move_2s_ticks"] = move_2s
            if ask_feat is not None:
                ask_feat["move_2s_ticks"] = move_2s
            if params.flow_filter == "bid":
                enable_ask = False
            elif params.flow_filter == "ask":
                enable_bid = False
            if params.flow_filter not in ("", "off"):
                if enable_bid and bid_feat is not None and _flow_against(bid_feat, params.flow_filter):
                    enable_bid = False
                    pull_bid += 1
                    action_steps["FLOW_PULL"] += 1
                if enable_ask and ask_feat is not None and _flow_against(ask_feat, params.flow_filter):
                    enable_ask = False
                    pull_ask += 1
                    action_steps["FLOW_PULL"] += 1
            if policy is not None:
                if enable_bid:
                    keep, extra = policy.decide(bid_feat)
                    if not keep:
                        enable_bid = False
                        pull_bid += 1
                        action_steps["TOXIC_PULL"] += 1
                    else:
                        bid_extra += extra
                if enable_ask:
                    keep, extra = policy.decide(ask_feat)
                    if not keep:
                        enable_ask = False
                        pull_ask += 1
                        action_steps["TOXIC_PULL"] += 1
                    else:
                        ask_extra += extra

        prices = base if policy is None else _prices(bid_extra, ask_extra)
        if prices is None or (not enable_bid and not enable_ask):
            queue.post(bid=0.0, ask=0.0, bid_enabled=False, ask_enabled=False)
            if not paused:
                action_steps["REJECT"] += 1
        else:
            bid, ask = prices
            queue.post(
                bid=bid,
                ask=ask,
                bid_enabled=enable_bid,
                ask_enabled=enable_ask,
                size=params.order_size,
            )
            if enable_bid and bid_feat is not None:
                queue.bid.context = bid_feat
            if enable_ask and ask_feat is not None:
                queue.ask.context = ask_feat
            action_steps["QUOTE"] += 1

        max_abs_inv = max(max_abs_inv, abs(queue.position.inventory))
        if last_eq_ts is None or (ts - last_eq_ts).total_seconds() >= 60.0:
            bn_m = queue.position.marked_pnl(trade_mid)
            hx = hedge.marked_usdc(fx_mid=fx_mid, mx_mid_usdt=mx_mid)
            equity_curve.append((ts, bn_m + hx))
            last_eq_ts = ts

    # Drain remaining events.
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
            note_mid(ev_ts[ev_i], mid)
            trade_amt = float(ev_amt[ev_i])
            fills = queue.on_trade(
                timestamp=ev_ts[ev_i],
                price=float(ev_price[ev_i]),
                amount=trade_amt,
                aggressor_side=str(ev_side[ev_i]),
                mid=mid,
            )
            handle_fills(fills)
            signed = signed_trade_qty(str(ev_side[ev_i]), trade_amt)
            ofi_100.update(ev_ts[ev_i], signed)
            ofi_500.update(ev_ts[ev_i], signed)
        ev_i += 1

    if fill_rows is not None:
        attach_markouts(
            fill_rows[row_start:], mid_ts, mid_px, params.trade_tick
        )

    bn_pnl = queue.position.marked_pnl(last_trade_mid)
    mx_pnl = hedge.marked_usdc(fx_mid=last_fx, mx_mid_usdt=last_mx_mid)
    combined = bn_pnl + mx_pnl
    report = markout.report()
    # Build a thin daily_pnl for metrics
    daily = [(day, combined, combined)]
    metrics = build_metrics(
        position=queue.position,
        marked_pnl=bn_pnl,
        last_mid=last_trade_mid,
        fills=queue.fills,
        equity_curve=equity_curve,
        daily_pnl=daily,
        markout=report,
        action_steps=action_steps,
        max_abs_inventory=max_abs_inv,
    )
    return {
        "day": day,
        "bn_fills": len(queue.fills),
        "bn_inv": queue.position.inventory,
        "bn_pnl": bn_pnl,
        "mx_fills": len(hedge.fills),
        "mx_inv": hedge.inventory,
        "mx_pnl": mx_pnl,
        "net": hedge.net_sui(queue.position.inventory),
        "combined": combined,
        "adverse": report.adverse_rate,
        "pause_steps": pause_steps,
        "pull_bid": pull_bid,
        "pull_ask": pull_ask,
        "metrics": metrics,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Event-L2 queue fill BT (MEXC fair → SUIUSDC → MEXC hedge)"
    )
    p.add_argument("--fair-symbol", default="suiusdt")
    p.add_argument("--trade-symbol", default="suiusdc")
    p.add_argument("--fx-symbol", default="usdcusdt")
    p.add_argument("--fair-exchange", default="hyperliquid")
    p.add_argument("--trade-exchange", default="binance-futures")
    p.add_argument("--fx-exchange", default="binance")
    p.add_argument("--date", default="2026-09-01")
    p.add_argument("--from-date", default=None)
    p.add_argument("--to-date", default=None)
    p.add_argument("--every", default="1s")
    p.add_argument("--maker-buffer-ticks", type=float, default=5.0)
    p.add_argument("--bid-skew-ticks", type=float, default=0.0)
    p.add_argument("--ask-skew-ticks", type=float, default=0.0)
    p.add_argument(
        "--inv-skew-ticks",
        type=float,
        default=0.0,
        help="Extra passive ticks per inventory lot on the increasing side",
    )
    p.add_argument(
        "--fast-move-ticks",
        type=float,
        default=8.0,
        help="Pause both sides if |Δmid| over lookback ≥ this (0=off)",
    )
    p.add_argument("--fast-move-lookback-seconds", type=float, default=2.0)
    p.add_argument("--fast-move-pause-seconds", type=float, default=3.0)
    p.add_argument(
        "--flow-filter",
        choices=["off", "micro", "ofi", "both", "either", "bid", "ask"],
        default="either",
        help="Pull a side when microprice and/or trade flow is against it. bid/ask = that side only",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Run buffer×{pause on/off}×inv-skew compare on the date range",
    )
    p.add_argument(
        "--size-grid",
        action="store_true",
        help="Sweep order_size × max_inventory (uses buffer/fast knobs from flags)",
    )
    p.add_argument(
        "--sizes",
        default="5,10,20,50,100",
        help="Comma order sizes for --size-grid",
    )
    p.add_argument(
        "--max-inventories",
        default="25,50,100,250,500",
        help="Comma max_inventory values for --size-grid (only pairs with max>=size)",
    )
    p.add_argument("--trade-tick", type=float, default=0.0001)
    p.add_argument("--order-size", type=float, default=10.0)
    p.add_argument("--max-inventory", type=float, default=50.0)
    p.add_argument("--maker-fee", type=float, default=0.0)
    p.add_argument(
        "--hedge-taker-fee",
        type=float,
        default=float(FEE_BY_EXCHANGE["hyperliquid"]["target_taker"]),
    )
    p.add_argument("--markout-horizon", type=float, default=1.0)
    args = p.parse_args()

    from_date, to_date, days = _intersect_days(
        [
            (args.fair_exchange, args.fair_symbol),
            (args.trade_exchange, args.trade_symbol),
            (args.fx_exchange, args.fx_symbol),
        ],
        args.date,
        args.from_date,
        args.to_date,
    )
    l2_days = set(
        available_dates(
            args.trade_symbol,
            data_type="incremental_book_L2",
            exchange=args.trade_exchange,
        )
    )
    days = [d for d in days if d in l2_days]
    if not days:
        raise SystemExit(
            "No days with incremental_book_L2. Download e.g.\n"
            "  python -c \"from src.download import download_jobs; "
            "download_jobs(jobs=[('binance-futures',['suiusdc'])], "
            "from_date='2026-08-26', to_date='2026-09-14', "
            "data_types=['incremental_book_L2'])\""
        )

    def run_params(label: str, params: MexcFairParams) -> dict:
        print(
            f"\n## {label}  days={len(days)} buffer={params.maker_buffer_ticks} "
            f"inv_skew={params.inv_skew_ticks} fast={params.fast_move_ticks}",
            flush=True,
        )
        totals = {
            "bn_fills": 0,
            "mx_fills": 0,
            "bn_pnl": 0.0,
            "mx_pnl": 0.0,
            "combined": 0.0,
            "pause_steps": 0,
            "adv_w": 0.0,
            "n_fill_mark": 0,
        }
        for day in days:
            r = run_day(
                day=day,
                every=args.every,
                fair_symbol=args.fair_symbol,
                trade_symbol=args.trade_symbol,
                fx_symbol=args.fx_symbol,
                fair_exchange=args.fair_exchange,
                trade_exchange=args.trade_exchange,
                fx_exchange=args.fx_exchange,
                params=params,
                maker_fee=args.maker_fee,
                hedge_taker_fee=args.hedge_taker_fee,
                markout_horizon=args.markout_horizon,
            )
            print(
                f"  {day} fills={r['bn_fills']} BN={r['bn_pnl']:+.4f} "
                f"MX={r['mx_pnl']:+.4f} COMBINED={r['combined']:+.4f} "
                f"adv={100*r['adverse']:.1f}% pause={r['pause_steps']} "
                f"flow_pull={r['pull_bid'] + r['pull_ask']}",
                flush=True,
            )
            totals["bn_fills"] += r["bn_fills"]
            totals["mx_fills"] += r["mx_fills"]
            totals["bn_pnl"] += r["bn_pnl"]
            totals["mx_pnl"] += r["mx_pnl"]
            totals["combined"] += r["combined"]
            totals["pause_steps"] += r["pause_steps"]
            if r["bn_fills"] > 0:
                totals["adv_w"] += r["adverse"] * r["bn_fills"]
                totals["n_fill_mark"] += r["bn_fills"]
        adv = (
            totals["adv_w"] / totals["n_fill_mark"]
            if totals["n_fill_mark"]
            else 0.0
        )
        print(
            f"  TOTAL fills={totals['bn_fills']} "
            f"BN={totals['bn_pnl']:+.6f} MX={totals['mx_pnl']:+.6f} "
            f"COMBINED={totals['combined']:+.6f} "
            f"adverse={100*adv:.1f}% pause_steps={totals['pause_steps']}",
            flush=True,
        )
        return {**totals, "adverse": adv, "label": label, "order_size": params.order_size, "max_inventory": params.max_inventory}

    if args.size_grid:
        sizes = [float(x) for x in args.sizes.split(",") if x.strip()]
        maxes = [float(x) for x in args.max_inventories.split(",") if x.strip()]
        # Default to best quote knobs from prior grid if user left fast off.
        buf = args.maker_buffer_ticks
        fast = args.fast_move_ticks if args.fast_move_ticks > 0 else 8.0
        print(
            f"Size grid {from_date}..{to_date} ({len(days)}d) "
            f"sizes={sizes} maxes={maxes} buffer={buf} fast={fast}",
            flush=True,
        )
        rows = []
        for oz in sizes:
            for mx in maxes:
                if mx + 1e-12 < oz:
                    continue
                label = f"oz{oz:g}_max{mx:g}"
                par = MexcFairParams(
                    trade_tick=args.trade_tick,
                    maker_buffer_ticks=buf,
                    order_size=oz,
                    max_inventory=mx,
                    bid_skew_ticks=args.bid_skew_ticks,
                    ask_skew_ticks=args.ask_skew_ticks,
                    inv_skew_ticks=args.inv_skew_ticks,
                    fast_move_ticks=fast,
                    fast_move_lookback_seconds=args.fast_move_lookback_seconds,
                    fast_move_pause_seconds=args.fast_move_pause_seconds,
                    flow_filter=args.flow_filter,
                )
                rows.append(run_params(label, par))
        best = max(rows, key=lambda r: r["combined"])
        # Also rank by PnL per unit inventory capacity (efficiency).
        def eff(r: dict) -> float:
            return r["combined"] / max(r["max_inventory"], 1e-9)

        best_eff = max(rows, key=eff)
        print(
            f"\nBEST_PnL {best['label']} COMBINED={best['combined']:+.6f} "
            f"fills={best['bn_fills']} adverse={100*best['adverse']:.1f}%",
            flush=True,
        )
        print(
            f"BEST_eff {best_eff['label']} COMBINED={best_eff['combined']:+.6f} "
            f"pnl/max_inv={eff(best_eff):+.6f} fills={best_eff['bn_fills']}",
            flush=True,
        )
        print(
            f"Live-ish: ORDER_SIZE={best['order_size']:g} "
            f"MAX_INVENTORY={best['max_inventory']:g} "
            f"(buffer={buf:g} fast_move_ticks={fast:g})",
            flush=True,
        )
        return

    if args.grid:
        combos: list[tuple[str, MexcFairParams]] = []
        for buf in (1.0, 3.0, 5.0):
            for inv_sk in (0.0, 2.0):
                for fast in (0.0, 8.0):
                    label = f"buf{buf:g}_inv{inv_sk:g}_fast{fast:g}"
                    combos.append(
                        (
                            label,
                            MexcFairParams(
                                trade_tick=args.trade_tick,
                                maker_buffer_ticks=buf,
                                order_size=args.order_size,
                                max_inventory=args.max_inventory,
                                bid_skew_ticks=args.bid_skew_ticks,
                                ask_skew_ticks=args.ask_skew_ticks,
                                inv_skew_ticks=inv_sk,
                                fast_move_ticks=fast,
                                fast_move_lookback_seconds=args.fast_move_lookback_seconds,
                                fast_move_pause_seconds=args.fast_move_pause_seconds,
                                flow_filter=args.flow_filter,
                            ),
                        )
                    )
        rows = [run_params(lab, par) for lab, par in combos]
        best = max(rows, key=lambda r: r["combined"])
        print(
            f"\nBEST {best['label']} COMBINED={best['combined']:+.6f} "
            f"fills={best['bn_fills']} adverse={100*best['adverse']:.1f}%",
            flush=True,
        )
        return

    print(
        f"Event-L2 fair BT {from_date}..{to_date} ({len(days)}d) "
        f"buffer={args.maker_buffer_ticks} inv_skew={args.inv_skew_ticks} "
        f"fast={args.fast_move_ticks} flow={args.flow_filter} size={args.order_size}",
        flush=True,
    )
    params = MexcFairParams(
        trade_tick=args.trade_tick,
        maker_buffer_ticks=args.maker_buffer_ticks,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        bid_skew_ticks=args.bid_skew_ticks,
        ask_skew_ticks=args.ask_skew_ticks,
        inv_skew_ticks=args.inv_skew_ticks,
        fast_move_ticks=args.fast_move_ticks,
        fast_move_lookback_seconds=args.fast_move_lookback_seconds,
        fast_move_pause_seconds=args.fast_move_pause_seconds,
        flow_filter=args.flow_filter,
    )
    run_params("single", params)


if __name__ == "__main__":
    main()
