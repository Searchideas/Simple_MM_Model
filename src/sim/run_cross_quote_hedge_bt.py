"""CLI: SUIUSDC cross-quote BT + back-to-back MEXC SUIUSDT taker hedge.

Same quoting logic as ``run_cross_quote_bt`` (AS on Binance SUIUSDT → FX →
maker on SUIUSDC). Every MM fill is immediately hedged opposite on MEXC
(hit bid / lift ask) at ``--hedge-taker-fee`` (default 0).

Equity marked in USDC:
  BN cash + BN inv · SUIUSDC mid
  + (MEXC cash_usdt + MEXC inv · MEXC mid) / USDCUSDT

Quoting inventory stays the Binance book (same AS / cover logic as unhedged).
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
from src.sim.mexc_hedge import MexcTakerHedge
from src.sim.multi_book_backtest import MultiBookBacktest, align_cross_books
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.sweep import interval_seconds


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-quote SUIUSDC + immediate MEXC SUIUSDT taker hedge"
    )
    parser.add_argument("--ref-symbol", default="suiusdt")
    parser.add_argument("--trade-symbol", default="suiusdc")
    parser.add_argument("--fx-symbol", default="usdcusdt")
    parser.add_argument("--hedge-symbol", default="suiusdt")
    parser.add_argument("--ref-exchange", default="binance-futures")
    parser.add_argument("--trade-exchange", default="binance-futures")
    parser.add_argument("--fx-exchange", default="binance")
    parser.add_argument("--hedge-exchange", default="hyperliquid")
    parser.add_argument("--date", default="all")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--min-spread", type=float, default=0.0004032)
    parser.add_argument("--ref-tick", type=float, default=0.0001)
    parser.add_argument("--trade-tick", type=float, default=0.0001)
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument(
        "--hedge-taker-fee",
        type=float,
        default=0.0,
        help="MEXC taker fee rate (default 0)",
    )
    parser.add_argument(
        "--hedge-max-basis-bps",
        type=float,
        default=0.0,
        help="Optional abs mid-basis skip rail (0=off; directional filter is primary)",
    )
    parser.add_argument(
        "--no-hedge-favorable",
        action="store_true",
        help="Disable directional L25 filter (always hit bid/lift ask)",
    )
    parser.add_argument(
        "--hedge-hybrid",
        action="store_true",
        help=(
            "Race: favorable MEXC hedge (keep monitoring) + pure-BN cover "
            "ladder on BN to close; requote both sides when net flat"
        ),
    )
    parser.add_argument(
        "--residual-cut-roi-pct",
        type=float,
        default=12.0,
        help="While hybrid residual open: same cut%% as pure BN (default 12 @ lev)",
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
        default=0.0,
        help="Hedged mode default 0 (ROI cut not needed when B2B hedging)",
    )
    parser.add_argument("--cover-trail-roi-pct", type=float, default=1.0)
    parser.add_argument("--cover-join-leverage", type=float, default=10.0)
    parser.add_argument("--cover-max-hold-seconds", type=float, default=0.0)
    parser.add_argument("--vol-floor", type=float, default=0.0001)
    parser.add_argument("--kappa-min", type=float, default=0.05)
    parser.add_argument("--kappa-max", type=float, default=0.5)
    parser.add_argument("--no-hard-flatten", action="store_true")
    parser.add_argument(
        "--fill-mode",
        choices=["touch", "through", "queue"],
        default="touch",
        help="MM fill model on SUIUSDC (default touch for faster hedge sweeps)",
    )
    parser.add_argument("--markout-horizon", type=float, default=1.0)
    parser.add_argument("--no-fit-kappa", action="store_true")
    args = parser.parse_args()

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
    label = days[0] if len(days) == 1 else f"{days[0]}..{days[-1]} ({len(days)}d)"
    print(
        f"Hedged cross-quote BT ref={args.ref_symbol}@{args.ref_exchange} "
        f"trade={args.trade_symbol}@{args.trade_exchange} "
        f"fx={args.fx_symbol}@{args.fx_exchange} "
        f"hedge={args.hedge_symbol}@{args.hedge_exchange} "
        f"{label} every={args.every} fill_mode={args.fill_mode} "
        f"hedge_taker={args.hedge_taker_fee} "
        f"max_basis_bps={args.hedge_max_basis_bps} "
        f"favorable={not args.no_hedge_favorable} hybrid={args.hedge_hybrid} "
        f"residual_cut={args.residual_cut_roi_pct} cut={args.cover_cut_roi_pct}",
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
    include_l25 = True  # need MEXC L25 for favorable-level walk
    timeline = align_cross_books(
        ref_bbo,
        trade_bbo,
        fx_bbo,
        include_l25=include_l25 or args.fill_mode == "queue",
        hedge_bbo=hedge_bbo,
    )
    del ref_bbo, trade_bbo, fx_bbo, hedge_bbo
    gc.collect()

    print("Loading trade-symbol trades...", flush=True)
    trades = load_trades_range(
        args.trade_symbol, from_date, to_date, exchange=args.trade_exchange
    )
    print(
        f"timeline={timeline.height} trades={trades.height} "
        f"interval_s={interval_seconds(args.every)}",
        flush=True,
    )

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
                print(f"fill_probability {day}: kappa_ticks={k:.6g}", flush=True)
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
    # Hybrid: favorable MEXC + pure-BN cover race on BN inventory.
    favorable = not args.no_hedge_favorable
    if args.hedge_hybrid:
        favorable = True
    hedge = MexcTakerHedge(
        taker_fee_rate=args.hedge_taker_fee,
        max_basis_bps=args.hedge_max_basis_bps,
        favorable_only=favorable,
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
        hedge_hybrid=args.hedge_hybrid,
        residual_cut_roi_pct=args.residual_cut_roi_pct,
    )
    result = bt.run(timeline, trades)

    print(
        f"bn_fills={result.fills} bn_inv={result.final_inventory:.4f} "
        f"bn_pnl={result.bn_marked_pnl:+.6f} "
        f"hedge_fills={result.hedge_fills} hedge_inv={result.hedge_inventory:.4f} "
        f"hedge_cash_usdt={result.hedge_cash_usdt:+.6f} "
        f"skipped_book={result.hedge_skipped} skipped_basis={result.hedge_skipped_basis} "
        f"skipped_unfav={result.hedge_skipped_unfavorable} "
        f"partial_qty={result.hedge_partial_qty:.4f} "
        f"residual_clears={result.hedge_residual_clears} "
        f"net_sui={result.net_sui:.4f} "
        f"COMBINED_pnl_usdc={result.combined_marked_pnl:+.6f} "
        f"fill_mode={result.fill_mode}",
        flush=True,
    )
    m = result.markout
    print(
        f"markout@{args.markout_horizon}s: scored={m.n_scored} "
        f"adverse={m.n_adverse} ({100.0 * m.adverse_rate:.1f}%) "
        f"mean_ticks={m.mean_markout_ticks:+.2f}",
        flush=True,
    )
    for line in result.metrics.format_lines():
        print(line, flush=True)
    if result.daily_pnl:
        print("daily_combined_pnl:", flush=True)
        for day, dpnl, cum in result.daily_pnl:
            print(f"  {day}  day={dpnl:+.6f}  cum={cum:+.6f}", flush=True)


if __name__ == "__main__":
    main()
