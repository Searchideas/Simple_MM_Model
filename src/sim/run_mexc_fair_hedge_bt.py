"""CLI: MEXC fair → USDCUSDT → SUIUSDC maker (N ticks off) → hit MEXC B2B.

Reports Binance-side PnL, MEXC-side PnL (USDC), and combined.
"""

from __future__ import annotations

import argparse
import gc

from src.loaders import available_dates, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.execution import ExecutionSimulator
from src.sim.mexc_fair_bt import MexcFairHedgeBacktest, MexcFairParams
from src.sim.mexc_hedge import MexcTakerHedge
from src.sim.multi_book_backtest import align_cross_books


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
    p = argparse.ArgumentParser(
        description="MEXC fair → FX → SUIUSDC maker + immediate MEXC hedge"
    )
    p.add_argument("--fair-symbol", default="suiusdt", help="MEXC fair source")
    p.add_argument("--trade-symbol", default="suiusdc")
    p.add_argument("--fx-symbol", default="usdcusdt")
    p.add_argument("--fair-exchange", default="hyperliquid")
    p.add_argument("--trade-exchange", default="binance-futures")
    p.add_argument("--fx-exchange", default="binance")
    p.add_argument("--date", default="all")
    p.add_argument("--from-date", default=None)
    p.add_argument("--to-date", default=None)
    p.add_argument("--every", default="1s")
    p.add_argument(
        "--fill-mode",
        choices=["touch", "through", "queue"],
        default="queue",
        help="SUIUSDC fill model (default queue)",
    )
    p.add_argument(
        "--maker-buffer-ticks",
        type=float,
        default=1.0,
        help="Ticks off MEXC fair before maker clamp (default 1)",
    )
    p.add_argument("--trade-tick", type=float, default=0.0001)
    p.add_argument("--order-size", type=float, default=10.0)
    p.add_argument("--max-inventory", type=float, default=50.0)
    p.add_argument("--soft-inventory-lots", type=float, default=0.0)
    p.add_argument("--maker-fee", type=float, default=0.0)
    p.add_argument("--hedge-taker-fee", type=float, default=0.0)
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
    print(
        f"MEXC-fair hedge BT fair={args.fair_symbol}@{args.fair_exchange} "
        f"trade={args.trade_symbol}@{args.trade_exchange} "
        f"fx={args.fx_symbol}@{args.fx_exchange} "
        f"{from_date}..{to_date} ({len(days)}d) every={args.every} "
        f"fill={args.fill_mode} buffer_ticks={args.maker_buffer_ticks} "
        f"size={args.order_size} max_inv={args.max_inventory}",
        flush=True,
    )

    print("Loading books...", flush=True)
    # ref = MEXC fair; trade = BN SUIUSDC; fx = USDCUSDT; hedge = MEXC again
    mexc_bbo = load_bbo_range(
        args.fair_symbol,
        from_date,
        to_date,
        every=args.every,
        exchange=args.fair_exchange,
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
    include_l25 = args.fill_mode == "queue"
    timeline = align_cross_books(
        mexc_bbo,
        trade_bbo,
        fx_bbo,
        include_l25=include_l25,
        hedge_bbo=mexc_bbo,
    )
    del mexc_bbo, trade_bbo, fx_bbo
    gc.collect()

    print("Loading SUIUSDC trades...", flush=True)
    trades = load_trades_range(
        args.trade_symbol, from_date, to_date, exchange=args.trade_exchange
    )
    print(
        f"timeline={timeline.height} trades={trades.height}",
        flush=True,
    )

    timeline_rows = list(timeline.sort("ts").iter_rows(named=True))
    trade_rows = list(trades.sort("ts").iter_rows(named=True))
    del timeline, trades
    gc.collect()

    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=args.max_inventory,
        order_size=args.order_size,
        fill_mode=args.fill_mode,
        tick_size=args.trade_tick,
    )
    hedge = MexcTakerHedge(
        taker_fee_rate=args.hedge_taker_fee,
        max_basis_bps=0.0,
        favorable_only=False,
    )
    params = MexcFairParams(
        trade_tick=args.trade_tick,
        maker_buffer_ticks=args.maker_buffer_ticks,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        soft_inventory_lots=args.soft_inventory_lots,
    )
    bt = MexcFairHedgeBacktest(
        execution=execution,
        hedge=hedge,
        params=params,
        markout_horizon_seconds=args.markout_horizon,
    )
    result = bt.run_rows(timeline_rows, trade_rows)

    print(
        f"BN_fills={result.fills} BN_inv={result.final_inventory:.4f} "
        f"BN_pnl_usdc={result.bn_marked_pnl:+.6f}",
        flush=True,
    )
    print(
        f"MEXC_fills={result.hedge_fills} MEXC_inv={result.hedge_inventory:.4f} "
        f"MEXC_cash_usdt={result.hedge_cash_usdt:+.6f} "
        f"MEXC_pnl_usdc={result.hedge_marked_usdc:+.6f}",
        flush=True,
    )
    print(
        f"net_sui={result.net_sui:.4f} "
        f"COMBINED_pnl_usdc={result.combined_marked_pnl:+.6f} "
        f"reject={result.reject_cross} fx_stale={result.fx_stale} "
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
    print(
        f"last fair_bid/ask={result.last_fair_bid:.6f}/{result.last_fair_ask:.6f} "
        f"quote={result.last_quote_bid:.6f}/{result.last_quote_ask:.6f}",
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
