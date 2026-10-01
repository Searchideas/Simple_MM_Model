"""Sweep maker JOIN_TP% × CUT% × hold on pure-BN cross-quote BT (load once).

Example:
  python -m src.sim.sweep_cover_cut --from-date 2026-08-26 --to-date 2026-09-13 ^
      --every 5s --fill-mode queue --cover-join-leverage 10 ^
      --joins 3,5,8,10 --cuts 5,8,10,12,15,20
"""

from __future__ import annotations

import argparse
import gc
import time

import polars as pl

from src.sim.cross_quote import CrossQuoteParams
from src.sim.execution import ExecutionSimulator
from src.sim.fill_probabilty import estimate_day_kappa
from src.sim.multi_book_backtest import MultiBookBacktest, align_cross_books
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.run_cross_quote_bt import _intersect_days
from src.loaders import load_bbo_range, load_trades_range


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep COVER_JOIN / COVER_CUT / hold for max marked PnL"
    )
    p.add_argument("--date", default=None)
    p.add_argument("--from-date", default="2026-08-26")
    p.add_argument("--to-date", default="2026-09-13")
    p.add_argument("--every", default="5s", help="Coarse grid default; use 1s to confirm")
    p.add_argument("--fill-mode", choices=["touch", "through", "queue"], default="touch")
    p.add_argument("--cuts", default="0,1,2,3,4,5,6,7,8,9,10,12,15")
    p.add_argument(
        "--joins",
        default=None,
        help="Comma COVER_JOIN_ROI_PCT values; default = --cover-join-roi-pct alone",
    )
    p.add_argument("--holds", default="0", help="Comma seconds; 0=off")
    p.add_argument("--cover-join-roi-pct", type=float, default=5.0)
    p.add_argument("--cover-trail-roi-pct", type=float, default=1.0)
    p.add_argument("--cover-join-leverage", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--horizon", type=float, default=300.0)
    p.add_argument("--min-spread", type=float, default=0.0004032)
    p.add_argument("--ref-tick", type=float, default=0.0001)
    p.add_argument("--trade-tick", type=float, default=0.0001)
    p.add_argument("--maker-fee", type=float, default=0.0)
    p.add_argument("--max-inventory", type=float, default=50.0)
    p.add_argument("--order-size", type=float, default=10.0)
    p.add_argument("--max-spread-ticks", type=float, default=10.0)
    p.add_argument("--volatility-window", type=int, default=120)
    p.add_argument("--basis-bps", type=float, default=4.0)
    p.add_argument("--maker-safety-ticks", type=float, default=2.0)
    p.add_argument("--soft-inventory-lots", type=float, default=0.0)
    p.add_argument("--vol-floor", type=float, default=0.0001)
    p.add_argument("--kappa-min", type=float, default=0.05)
    p.add_argument("--kappa-max", type=float, default=0.5)
    p.add_argument("--no-hard-flatten", action="store_true")
    p.add_argument("--no-fit-kappa", action="store_true")
    p.add_argument("--markout-horizon", type=float, default=1.0)
    p.add_argument("--ref-symbol", default="suiusdt")
    p.add_argument("--trade-symbol", default="suiusdc")
    p.add_argument("--fx-symbol", default="usdcusdt")
    p.add_argument("--ref-exchange", default="binance-futures")
    p.add_argument("--trade-exchange", default="binance-futures")
    p.add_argument("--fx-exchange", default="binance")
    args = p.parse_args()

    date = args.date or "all"
    from_date, to_date, days = _intersect_days(
        args.ref_exchange,
        args.trade_exchange,
        args.fx_exchange,
        args.ref_symbol,
        args.trade_symbol,
        args.fx_symbol,
        date,
        args.from_date,
        args.to_date,
    )
    if args.from_date:
        from_date = max(from_date, args.from_date)
    if args.to_date:
        to_date = min(to_date, args.to_date)
    days = [d for d in days if from_date <= d <= to_date]
    if not days:
        raise SystemExit("No days in requested window")

    cuts = _parse_floats(args.cuts)
    holds = _parse_floats(args.holds)
    joins = (
        _parse_floats(args.joins)
        if args.joins
        else [float(args.cover_join_roi_pct)]
    )

    print(
        f"Sweep cover join×cut {from_date}..{to_date} ({len(days)}d) "
        f"every={args.every} fill={args.fill_mode} "
        f"joins={joins} cuts={cuts} holds={holds} lev={args.cover_join_leverage}",
        flush=True,
    )

    print("Loading books...", flush=True)
    ref_bbo = load_bbo_range(
        args.ref_symbol, from_date, to_date, every=args.every, exchange=args.ref_exchange
    )
    trade_bbo = load_bbo_range(
        args.trade_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.trade_exchange,
    )
    fx_bbo = load_bbo_range(
        args.fx_symbol, from_date, to_date, every=args.every, exchange=args.fx_exchange
    )
    include_l25 = args.fill_mode == "queue"
    timeline = align_cross_books(ref_bbo, trade_bbo, fx_bbo, include_l25=include_l25)
    del ref_bbo, trade_bbo, fx_bbo
    gc.collect()

    print("Loading trades...", flush=True)
    trades = load_trades_range(
        args.trade_symbol, from_date, to_date, exchange=args.trade_exchange
    )
    print(f"timeline={timeline.height} trades={trades.height}", flush=True)

    kappa = args.kappa
    kappa_path: pl.DataFrame | None = None
    if not args.no_fit_kappa:
        paths: list[pl.DataFrame] = []
        for day in days:
            try:
                _A, k, path = estimate_day_kappa(
                    args.ref_symbol, day, tick=args.ref_tick, exchange=args.ref_exchange
                )
                paths.append(path)
                kappa = k
                print(f"kappa {day}: {k:.6g}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"kappa {day} failed: {exc}", flush=True)
        if paths:
            kappa_path = pl.concat(paths).sort("win")

    print("Materializing rows...", flush=True)
    timeline_rows = list(timeline.sort("ts").iter_rows(named=True))
    trade_rows = list(trades.sort("ts").iter_rows(named=True))
    del timeline, trades
    gc.collect()

    rows_out: list[dict] = []
    n_combo = len(holds) * len(joins) * len(cuts)
    done = 0
    print(
        f"{'join%':>6} {'cut%':>6} {'hold_s':>8} {'fills':>7} {'inv':>10} "
        f"{'marked_pnl':>12} {'adverse%':>9} {'sec':>7}",
        flush=True,
    )
    for hold in holds:
        for join_pct in joins:
            for cut in cuts:
                done += 1
                t0 = time.perf_counter()
                parameters = QuoteParameters(
                    base_gamma=args.gamma,
                    kappa=kappa,
                    time_horizon=args.horizon,
                    min_spread=args.min_spread,
                    tick_size=args.ref_tick,
                    max_inventory=args.max_inventory,
                    max_spread_ticks=args.max_spread_ticks,
                    maker_fee_rate=args.maker_fee,
                    include_kappa_spread=True,
                    kappa_min=args.kappa_min,
                    kappa_max=args.kappa_max,
                )
                model = AvellanedaStoikovModel(parameters)
                execution = ExecutionSimulator(
                    maker_fee_rate=args.maker_fee,
                    max_inventory=args.max_inventory,
                    order_size=args.order_size,
                    fill_mode=args.fill_mode,
                    tick_size=args.trade_tick,
                )
                cross = CrossQuoteParams(
                    reference_basis_bps=args.basis_bps,
                    maker_safety_ticks=args.maker_safety_ticks,
                    soft_inventory_lots=args.soft_inventory_lots,
                    cover_join_roi_pct=join_pct,
                    cover_cut_roi_pct=cut,
                    cover_trail_roi_pct=args.cover_trail_roi_pct,
                    cover_join_leverage=args.cover_join_leverage,
                    cover_max_hold_seconds=hold,
                    trade_tick=args.trade_tick,
                    vol_floor=args.vol_floor,
                    order_size=args.order_size,
                    max_inventory=args.max_inventory,
                    hard_flatten_at_max=not args.no_hard_flatten,
                )
                bt = MultiBookBacktest(
                    model=model,
                    execution=execution,
                    cross_params=cross,
                    volatility_window=args.volatility_window,
                    mid_sample_seconds=1.0,
                    kappa_path=kappa_path,
                    kappa_fallback=kappa,
                    markout_horizon_seconds=args.markout_horizon,
                )
                result = bt.run_rows(timeline_rows, trade_rows)
                elapsed = time.perf_counter() - t0
                adv = 100.0 * result.markout.adverse_rate
                row = {
                    "join_pct": join_pct,
                    "cut_pct": cut,
                    "hold_s": hold,
                    "fills": result.fills,
                    "inv": result.final_inventory,
                    "marked_pnl": result.marked_pnl,
                    "adverse_pct": adv,
                    "sec": elapsed,
                }
                rows_out.append(row)
                print(
                    f"{join_pct:6.1f} {cut:6.1f} {hold:8.0f} {result.fills:7d} "
                    f"{result.final_inventory:10.4f} {result.marked_pnl:12.6f} "
                    f"{adv:8.1f}% {elapsed:7.1f}  [{done}/{n_combo}]",
                    flush=True,
                )

    best = max(rows_out, key=lambda r: r["marked_pnl"])
    print(
        f"\nBEST marked_pnl={best['marked_pnl']:.6f} "
        f"join={best['join_pct']:g}% cut={best['cut_pct']:g}% "
        f"hold={best['hold_s']:g}s inv={best['inv']:.4f} fills={best['fills']} "
        f"adverse={best['adverse_pct']:.1f}%",
        flush=True,
    )
    print(
        f"Live-ish:\n"
        f"  COVER_JOIN_ROI_PCT={best['join_pct']:g}\n"
        f"  COVER_CUT_ROI_PCT={best['cut_pct']:g}\n"
        f"  COVER_JOIN_LEVERAGE={args.cover_join_leverage:g}\n"
        f"  COVER_TRAIL_ROI_PCT={args.cover_trail_roi_pct:g}\n"
        f"  COVER_MAX_HOLD_SECONDS={best['hold_s']:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
