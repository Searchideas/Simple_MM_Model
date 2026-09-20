"""CLI: multi-book cross-quote backtest (SUIUSDT → USDCUSDT → SUIUSDC).

Default mode (production-aligned guards):
  queue fills (L25 size-ahead), fitted κ clamped to [0.05, 0.5], hard flatten
  at max inventory, maker_fee=0, min_spread≈live, vol_floor=1e-4,
  cover TP 5%, cut 12%@10x then trail 1%@10x.

See docs/CROSS_QUOTE_ROADMAP.md.
"""

from __future__ import annotations

import argparse
import gc

import polars as pl

from src.loaders import (
    available_dates,
    load_bbo_range,
    load_trades_range,
    resolve_date_range,
)
from src.sim.cross_quote import CrossQuoteParams
from src.sim.execution import ExecutionSimulator
from src.sim.fill_probabilty import estimate_day_kappa
from src.sim.multi_book_backtest import MultiBookBacktest, align_cross_books
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.sweep import interval_seconds


def _intersect_days(
    ref_ex: str,
    trade_ex: str,
    fx_ex: str,
    ref_sym: str,
    trade_sym: str,
    fx_sym: str,
    date: str,
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str, list[str]]:
    ref_from, ref_to, ref_days = resolve_date_range(
        ref_sym, date=date, from_date=from_date, to_date=to_date, exchange=ref_ex
    )
    trade_days = set(
        available_dates(trade_sym, exchange=trade_ex)
    )
    fx_days = set(available_dates(fx_sym, exchange=fx_ex))
    picked = [d for d in ref_days if d in trade_days and d in fx_days]
    if not picked:
        raise FileNotFoundError(
            f"No overlapping dates for {ref_sym}/{trade_sym}/{fx_sym} "
            f"in [{ref_from}, {ref_to}]"
        )
    return picked[0], picked[-1], picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-quote BT: AS on ref, FX convert, fill on trade symbol"
    )
    parser.add_argument("--ref-symbol", default="suiusdt")
    parser.add_argument("--trade-symbol", default="suiusdc")
    parser.add_argument("--fx-symbol", default="usdcusdt")
    parser.add_argument("--ref-exchange", default="binance-futures")
    parser.add_argument("--trade-exchange", default="binance-futures")
    parser.add_argument("--fx-exchange", default="binance")
    parser.add_argument("--date", default="all")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.0004032,
        help="AS min spread floor (live default)",
    )
    parser.add_argument(
        "--ref-tick",
        type=float,
        default=0.0001,
        help="SUIUSDT tick (AS / κ space)",
    )
    parser.add_argument(
        "--trade-tick",
        type=float,
        default=0.0001,
        help="SUIUSDC tick (posting / clamp)",
    )
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=0.0,
        help="Maker fee in AS fee floor + sim fills (SUIUSDC promo default 0)",
    )
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument("--order-size", type=float, default=10.0)
    parser.add_argument("--max-spread-ticks", type=float, default=10.0)
    parser.add_argument("--volatility-window", type=int, default=120)
    parser.add_argument("--every", default="1s")
    parser.add_argument("--basis-bps", type=float, default=4.0)
    parser.add_argument("--maker-safety-ticks", type=float, default=2.0)
    parser.add_argument("--soft-inventory-lots", type=float, default=0.0)
    parser.add_argument("--cover-join-roi-pct", type=float, default=5.0)
    parser.add_argument(
        "--cover-cut-roi-pct",
        type=float,
        default=12.0,
        help="Cut threshold %% @ cover leverage: join BBO at breach; "
        "then trail (see --cover-trail-roi-pct). 0=disable",
    )
    parser.add_argument(
        "--cover-trail-roi-pct",
        type=float,
        default=1.0,
        help="After cut breach, trail mid±(trail%%/lev) each cycle (live: 1)",
    )
    parser.add_argument("--cover-join-leverage", type=float, default=10.0)
    parser.add_argument("--cover-max-hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--vol-floor",
        type=float,
        default=0.0001,
        help="Floor on σ (1/sqrt(s)), live default",
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=0.05,
        help="Clamp fitted κ_ticks lower bound",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=0.5,
        help="Clamp fitted κ_ticks upper bound",
    )
    parser.add_argument(
        "--no-hard-flatten",
        action="store_true",
        help="Disable join-BBO flatten when |q|>=max_inventory",
    )
    parser.add_argument(
        "--fill-mode",
        choices=["touch", "through", "queue"],
        default="queue",
        help="queue=L25 size-ahead (Part 3B default); touch=at BBO only; through=optimistic",
    )
    parser.add_argument(
        "--markout-horizon",
        type=float,
        default=1.0,
        help="Seconds after fill to score mid mark-out (default 1s)",
    )
    parser.add_argument(
        "--no-fit-kappa",
        action="store_true",
        help="Use --kappa only; skip fill_probability fit on ref symbol",
    )
    args = parser.parse_args()

    from_date, to_date, days = _intersect_days(
        args.ref_exchange,
        args.trade_exchange,
        args.fx_exchange,
        args.ref_symbol,
        args.trade_symbol,
        args.fx_symbol,
        args.date,
        args.from_date,
        args.to_date,
    )
    label = days[0] if len(days) == 1 else f"{days[0]}..{days[-1]} ({len(days)}d)"
    print(
        f"Cross-quote BT ref={args.ref_symbol}@{args.ref_exchange} "
        f"fx={args.fx_symbol}@{args.fx_exchange} "
        f"trade={args.trade_symbol}@{args.trade_exchange} "
        f"{label} every={args.every} fill_mode={args.fill_mode} "
        f"markout_h={args.markout_horizon}s",
        flush=True,
    )

    print("Loading books...", flush=True)
    ref_bbo = load_bbo_range(
        args.ref_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.ref_exchange,
    )
    trade_bbo = load_bbo_range(
        args.trade_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.trade_exchange,
    )
    fx_bbo = load_bbo_range(
        args.fx_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.fx_exchange,
    )
    print("Aligning timeline...", flush=True)
    include_l25 = args.fill_mode == "queue"
    timeline = align_cross_books(
        ref_bbo, trade_bbo, fx_bbo, include_l25=include_l25
    )
    del ref_bbo, trade_bbo, fx_bbo
    gc.collect()

    print("Loading trade-symbol trades...", flush=True)
    trades = load_trades_range(
        args.trade_symbol,
        from_date,
        to_date,
        exchange=args.trade_exchange,
    )
    interval = interval_seconds(args.every)
    print(
        f"timeline={timeline.height} trades={trades.height} interval_s={interval}",
        flush=True,
    )

    kappa = args.kappa
    kappa_path: pl.DataFrame | None = None
    if not args.no_fit_kappa:
        paths: list[pl.DataFrame] = []
        for day in days:
            try:
                A, k, path = estimate_day_kappa(
                    args.ref_symbol,
                    day,
                    tick=args.ref_tick,
                    exchange=args.ref_exchange,
                )
                paths.append(path)
                kappa = k
                print(
                    f"fill_probability {day}: A={A:.6g} kappa_ticks={k:.6g}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"fill_probability {day} failed: {exc}", flush=True)
        if paths:
            kappa_path = pl.concat(paths).sort("win")

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
        cover_join_roi_pct=args.cover_join_roi_pct,
        cover_cut_roi_pct=args.cover_cut_roi_pct,
        cover_trail_roi_pct=args.cover_trail_roi_pct,
        cover_join_leverage=args.cover_join_leverage,
        cover_max_hold_seconds=args.cover_max_hold_seconds,
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
    result = bt.run(timeline, trades)
    m = result.markout
    print(
        f"fills={result.fills} inv={result.final_inventory:.4f} "
        f"cash={result.cash:.6f} marked_pnl={result.marked_pnl:.6f} "
        f"fill_mode={result.fill_mode} reject_cross={result.reject_cross} "
        f"fx_stale={result.fx_stale} mean_basis_resid_bps={result.mean_basis_residual_bps:.2f} "
        f"last_action={result.last_action} kappa={result.last_kappa:.6g}",
        flush=True,
    )
    print(
        f"markout@{args.markout_horizon}s: scored={m.n_scored} "
        f"adverse={m.n_adverse} ({100.0 * m.adverse_rate:.1f}%) "
        f"mean_ticks={m.mean_markout_ticks:+.2f} "
        f"mean_adverse_ticks={m.mean_adverse_ticks:.2f} "
        f"markout_pnl_proxy={m.total_markout_pnl_proxy:+.6f} "
        f"adverse_pnl_proxy={m.total_adverse_pnl_proxy:+.6f}",
        flush=True,
    )
    for line in result.metrics.format_lines():
        print(line, flush=True)
    if result.daily_pnl:
        print("daily_pnl:", flush=True)
        for day, dpnl, cum in result.daily_pnl:
            print(f"  {day}  day={dpnl:+.6f}  cum={cum:+.6f}", flush=True)


if __name__ == "__main__":
    main()
