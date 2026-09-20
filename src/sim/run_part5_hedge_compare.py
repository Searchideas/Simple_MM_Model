"""Part 5 finish: load once, compare unhedged vs MEXC B2B hedge variants.

Example:
  python -m src.sim.run_part5_hedge_compare --from-date 2026-08-26 --to-date 2026-09-13 \\
      --every 5s --fill-mode touch
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


def _overlap(
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str, list[str]]:
    jobs = [
        ("binance-futures", "suiusdt"),
        ("binance-futures", "suiusdc"),
        ("binance", "usdcusdt"),
        ("mexc-futures", "suiusdt"),
    ]
    _, _, days = resolve_date_range(
        "suiusdt",
        date="all",
        from_date=from_date,
        to_date=to_date,
        exchange="binance-futures",
    )
    picked = list(days)
    for ex, sym in jobs[1:]:
        have = set(available_dates(sym, exchange=ex))
        picked = [d for d in picked if d in have]
    if from_date:
        picked = [d for d in picked if d >= from_date]
    if to_date:
        picked = [d for d in picked if d <= to_date]
    if not picked:
        raise SystemExit("No overlapping days")
    return picked[0], picked[-1], picked


def _run(
    *,
    label: str,
    timeline_rows: list[dict],
    trade_rows: list[dict],
    kappa: float,
    kappa_path: pl.DataFrame | None,
    fill_mode: str,
    cut: float,
    hedge: MexcTakerHedge | None,
    args: argparse.Namespace,
) -> dict:
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
        fill_mode=fill_mode,
        tick_size=args.trade_tick,
    )
    cross = CrossQuoteParams(
        reference_basis_bps=args.basis_bps,
        maker_safety_ticks=args.maker_safety_ticks,
        soft_inventory_lots=args.soft_inventory_lots,
        cover_join_roi_pct=args.cover_join_roi_pct,
        cover_cut_roi_pct=cut,
        cover_trail_roi_pct=args.cover_trail_roi_pct,
        cover_join_leverage=args.cover_join_leverage,
        cover_max_hold_seconds=0.0,
        trade_tick=args.trade_tick,
        vol_floor=args.vol_floor,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        hard_flatten_at_max=True,
    )
    bt = MultiBookBacktest(
        model=model,
        execution=execution,
        cross_params=cross,
        volatility_window=args.volatility_window,
        kappa_path=kappa_path,
        kappa_fallback=kappa,
        markout_horizon_seconds=1.0,
        hedge=hedge,
    )
    result = bt.run_rows(timeline_rows, trade_rows)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "cut": cut,
        "fills": result.fills,
        "bn_inv": result.final_inventory,
        "bn_pnl": result.bn_marked_pnl if hedge else result.marked_pnl,
        "hedge_fills": result.hedge_fills,
        "hedge_inv": result.hedge_inventory,
        "skipped_basis": result.hedge_skipped_basis,
        "net_sui": result.net_sui if hedge else result.final_inventory,
        "combined_pnl": result.combined_marked_pnl if hedge else result.marked_pnl,
        "adverse": 100.0 * result.markout.adverse_rate,
        "sec": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Part 5: unhedged vs MEXC B2B compare")
    p.add_argument("--from-date", default="2026-08-26")
    p.add_argument("--to-date", default="2026-09-13")
    p.add_argument("--every", default="5s")
    p.add_argument("--fill-mode", choices=["touch", "through", "queue"], default="touch")
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
    p.add_argument("--cover-join-roi-pct", type=float, default=5.0)
    p.add_argument("--cover-trail-roi-pct", type=float, default=1.0)
    p.add_argument("--cover-join-leverage", type=float, default=10.0)
    p.add_argument("--vol-floor", type=float, default=0.0001)
    p.add_argument("--kappa-min", type=float, default=0.05)
    p.add_argument("--kappa-max", type=float, default=0.5)
    p.add_argument("--hedge-taker-fee", type=float, default=0.0)
    p.add_argument("--no-fit-kappa", action="store_true")
    args = p.parse_args()

    from_d, to_d, days = _overlap(args.from_date, args.to_date)
    print(
        f"Part5 compare {from_d}..{to_d} ({len(days)}d) every={args.every} "
        f"fill={args.fill_mode}",
        flush=True,
    )

    print("Loading books...", flush=True)
    ref = load_bbo_range(
        "suiusdt", from_d, to_d, every=args.every, exchange="binance-futures"
    )
    trade = load_bbo_range(
        "suiusdc", from_d, to_d, every=args.every, exchange="binance-futures"
    )
    fx = load_bbo_range("usdcusdt", from_d, to_d, every=args.every, exchange="binance")
    hedge_bbo = load_bbo_range(
        "suiusdt", from_d, to_d, every=args.every, exchange="mexc-futures"
    )
    include_l25 = args.fill_mode == "queue"
    timeline = align_cross_books(
        ref, trade, fx, include_l25=include_l25, hedge_bbo=hedge_bbo
    )
    del ref, trade, fx, hedge_bbo
    gc.collect()

    trades = load_trades_range(
        "suiusdc", from_d, to_d, exchange="binance-futures"
    )
    print(f"timeline={timeline.height} trades={trades.height}", flush=True)

    kappa = args.kappa
    kappa_path: pl.DataFrame | None = None
    if not args.no_fit_kappa:
        paths: list[pl.DataFrame] = []
        for day in days:
            try:
                _A, k, path = estimate_day_kappa(
                    "suiusdt", day, tick=args.ref_tick, exchange="binance-futures"
                )
                paths.append(path)
                kappa = k
                print(f"kappa {day}: {k:.6g}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"kappa {day} failed: {exc}", flush=True)
        if paths:
            kappa_path = pl.concat(paths).sort("win")

    print("Materializing...", flush=True)
    timeline_rows = list(timeline.sort("ts").iter_rows(named=True))
    trade_rows = list(trades.sort("ts").iter_rows(named=True))
    del timeline, trades
    gc.collect()

    variants = [
        ("unhedged_cut12", 12.0, None, None),
        ("unhedged_cut0", 0.0, None, None),
        ("hedge_b2b_cut0_rail5", 0.0, 0.0, 5.0),
        ("hedge_b2b_cut0_rail0", 0.0, 0.0, 0.0),
        ("hedge_b2b_cut0_rail10", 0.0, 0.0, 10.0),
    ]

    print(
        f"{'label':<28} {'cut':>5} {'fills':>7} {'bn_inv':>9} {'bn_pnl':>11} "
        f"{'h_fills':>7} {'net':>8} {'comb_pnl':>11} {'skip_b':>7} {'adv%':>6} {'sec':>6}",
        flush=True,
    )
    rows: list[dict] = []
    for label, cut, _fee, rail in variants:
        hedge = None
        if rail is not None:
            hedge = MexcTakerHedge(
                taker_fee_rate=args.hedge_taker_fee,
                max_basis_bps=float(rail),
            )
        r = _run(
            label=label,
            timeline_rows=timeline_rows,
            trade_rows=trade_rows,
            kappa=kappa,
            kappa_path=kappa_path,
            fill_mode=args.fill_mode,
            cut=cut,
            hedge=hedge,
            args=args,
        )
        rows.append(r)
        print(
            f"{r['label']:<28} {r['cut']:5.1f} {r['fills']:7d} {r['bn_inv']:9.4f} "
            f"{r['bn_pnl']:11.6f} {r['hedge_fills']:7d} {r['net_sui']:8.4f} "
            f"{r['combined_pnl']:11.6f} {r['skipped_basis']:7d} "
            f"{r['adverse']:5.1f}% {r['sec']:6.1f}",
            flush=True,
        )

    best = max(rows, key=lambda x: x["combined_pnl"])
    print(
        f"\nBEST combined_pnl={best['combined_pnl']:+.6f}  [{best['label']}]",
        flush=True,
    )
    print(
        "Note: hedged PnL is BN+MEXC in USDC; unhedged is BN SUIUSDC only.",
        flush=True,
    )


if __name__ == "__main__":
    main()
