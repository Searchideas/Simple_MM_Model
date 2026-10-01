"""Sweep hybrid JOIN_TP% × CUT% (load books once).

Optimizes COVER_JOIN_ROI_PCT (+TP) and residual/cut COVER_CUT_ROI_PCT (−loss)
for ``--hedge-hybrid`` race (MEXC when good + pure-BN cover on BN).

Example (coarse then confirm):
  python -m src.sim.sweep_hybrid_roi --from-date 2026-08-26 --to-date 2026-09-13 ^
      --every 5s --fill-mode queue
  python -m src.sim.sweep_hybrid_roi --from-date 2026-08-26 --to-date 2026-09-13 ^
      --every 1s --fill-mode queue --joins 5,8 --cuts 10,12,15
"""

from __future__ import annotations

import argparse
import gc
import time

import polars as pl

from src.loaders import available_dates, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.cross_quote import CrossQuoteParams
from src.sim.execution import ExecutionSimulator
from src.sim.fill_probabilty import estimate_day_kappa
from src.sim.mexc_hedge import MexcTakerHedge
from src.sim.multi_book_backtest import MultiBookBacktest, align_cross_books
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters


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
        picked = [d for d in picked if d in have]
    if not picked:
        raise FileNotFoundError(
            "No overlapping dates for "
            + ", ".join(f"{s}@{e}" for e, s in exchanges_symbols)
        )
    return picked[0], picked[-1], picked


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep hybrid JOIN_TP% × CUT% for max combined PnL"
    )
    p.add_argument("--date", default="all")
    p.add_argument("--from-date", default="2026-08-26")
    p.add_argument("--to-date", default="2026-09-13")
    p.add_argument("--every", default="5s")
    p.add_argument("--fill-mode", choices=["touch", "through", "queue"], default="queue")
    p.add_argument(
        "--joins",
        default="3,5,8,10",
        help="Comma COVER_JOIN_ROI_PCT values (take-profit join)",
    )
    p.add_argument(
        "--cuts",
        default="5,8,10,12,15,20",
        help="Comma residual/cut COVER_CUT_ROI_PCT values",
    )
    p.add_argument("--cover-trail-roi-pct", type=float, default=1.0)
    p.add_argument("--cover-join-leverage", type=float, default=10.0)
    p.add_argument("--hedge-taker-fee", type=float, default=0.0)
    p.add_argument("--hedge-max-basis-bps", type=float, default=0.0)
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
    p.add_argument("--hedge-symbol", default="suiusdt")
    p.add_argument("--ref-exchange", default="binance-futures")
    p.add_argument("--trade-exchange", default="binance-futures")
    p.add_argument("--fx-exchange", default="binance")
    p.add_argument("--hedge-exchange", default="hyperliquid")
    args = p.parse_args()

    joins = _parse_floats(args.joins)
    cuts = _parse_floats(args.cuts)
    from_date, to_date, days = _intersect_days(
        [
            (args.ref_exchange, args.ref_symbol),
            (args.trade_exchange, args.trade_symbol),
            (args.fx_exchange, args.fx_symbol),
            (args.hedge_exchange, args.hedge_symbol),
        ],
        args.date,
        args.from_date,
        args.to_date,
    )
    print(
        f"Sweep hybrid ROI {from_date}..{to_date} ({len(days)}d) "
        f"every={args.every} fill={args.fill_mode} "
        f"joins={joins} cuts={cuts} lev={args.cover_join_leverage}",
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
    hedge_bbo = load_bbo_range(
        args.hedge_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.hedge_exchange,
    )
    include_l25 = True  # MEXC L25 + optional queue fills
    timeline = align_cross_books(
        ref_bbo,
        trade_bbo,
        fx_bbo,
        include_l25=include_l25,
        hedge_bbo=hedge_bbo,
    )
    del ref_bbo, trade_bbo, fx_bbo, hedge_bbo
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
    n_combo = len(joins) * len(cuts)
    print(
        f"{'join%':>6} {'cut%':>6} {'fills':>7} {'net':>8} "
        f"{'combined':>12} {'bn_pnl':>10} {'res_clr':>7} {'adv%':>7} {'sec':>7}",
        flush=True,
    )
    done = 0
    for join_pct in joins:
        for cut_pct in cuts:
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
            # Hedge CLI uses cut=0 base; hybrid enables residual cut dynamically.
            cross = CrossQuoteParams(
                reference_basis_bps=args.basis_bps,
                maker_safety_ticks=args.maker_safety_ticks,
                soft_inventory_lots=args.soft_inventory_lots,
                cover_join_roi_pct=join_pct,
                cover_cut_roi_pct=0.0,
                cover_trail_roi_pct=args.cover_trail_roi_pct,
                cover_join_leverage=args.cover_join_leverage,
                cover_max_hold_seconds=0.0,
                trade_tick=args.trade_tick,
                vol_floor=args.vol_floor,
                order_size=args.order_size,
                max_inventory=args.max_inventory,
                hard_flatten_at_max=not args.no_hard_flatten,
            )
            hedge = MexcTakerHedge(
                taker_fee_rate=args.hedge_taker_fee,
                max_basis_bps=args.hedge_max_basis_bps,
                favorable_only=True,
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
                hedge=hedge,
                hedge_hybrid=True,
                residual_cut_roi_pct=cut_pct,
            )
            result = bt.run_rows(timeline_rows, trade_rows)
            elapsed = time.perf_counter() - t0
            adv = 100.0 * result.markout.adverse_rate
            row = {
                "join_pct": join_pct,
                "cut_pct": cut_pct,
                "fills": result.fills,
                "net": result.net_sui,
                "combined": result.combined_marked_pnl,
                "bn_pnl": result.bn_marked_pnl,
                "residual_clears": result.hedge_residual_clears,
                "adverse_pct": adv,
                "sec": elapsed,
            }
            rows_out.append(row)
            print(
                f"{join_pct:6.1f} {cut_pct:6.1f} {result.fills:7d} "
                f"{result.net_sui:8.2f} {result.combined_marked_pnl:12.6f} "
                f"{result.bn_marked_pnl:10.4f} {result.hedge_residual_clears:7d} "
                f"{adv:6.1f}% {elapsed:7.1f}  [{done}/{n_combo}]",
                flush=True,
            )

    best = max(rows_out, key=lambda r: r["combined"])
    print(
        f"\nBEST combined={best['combined']:.6f} "
        f"join={best['join_pct']:g}% cut={best['cut_pct']:g}% "
        f"net={best['net']:.4f} fills={best['fills']} "
        f"adverse={best['adverse_pct']:.1f}%",
        flush=True,
    )
    print(
        f"Live / BT knobs:\n"
        f"  COVER_JOIN_ROI_PCT={best['join_pct']:g}\n"
        f"  COVER_CUT_ROI_PCT={best['cut_pct']:g}\n"
        f"  COVER_JOIN_LEVERAGE={args.cover_join_leverage:g}\n"
        f"  COVER_TRAIL_ROI_PCT={args.cover_trail_roi_pct:g}\n"
        f"Hybrid CLI:\n"
        f"  --hedge-hybrid --cover-join-roi-pct {best['join_pct']:g} "
        f"--residual-cut-roi-pct {best['cut_pct']:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
