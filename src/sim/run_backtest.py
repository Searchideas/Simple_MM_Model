"""Command-line entry point for the pure market-making backtest."""

import argparse

import polars as pl

from src.loaders import load_bbo_day, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.backtest import Backtest
from src.sim.execution import ExecutionSimulator
from src.sim.fill_probabilty import estimate_day_kappa
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.sweep import interval_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="suiusdt")
    parser.add_argument(
        "--date",
        default="all",
        help="YYYY-MM-DD or 'all' for every downloaded day (default: all).",
    )
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.25,
        help="Fallback κ if fill_probability fit fails.",
    )
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--min-spread", type=float, default=0.0004032)
    parser.add_argument("--tick-size", type=float, default=0.0001)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument("--order-size", type=float, default=10.0)
    parser.add_argument("--max-spread-ticks", type=float, default=10.0)
    parser.add_argument("--volatility-window", type=int, default=120)
    parser.add_argument(
        "--every",
        default="5s",
        help="BBO resample for multi-day (and single-day if set). Default 5s.",
    )
    parser.add_argument(
        "--no-fit-kappa",
        action="store_true",
        help="Use --kappa only; skip fill_probability 3m fit.",
    )
    args = parser.parse_args()

    from_date, to_date, days = resolve_date_range(
        args.symbol,
        date=args.date,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    date_label = days[0] if len(days) == 1 else f"{days[0]}..{days[-1]} ({len(days)}d)"
    print(f"Loading {args.symbol} {date_label} every={args.every}", flush=True)
    trades = load_trades_range(args.symbol, from_date, to_date)
    if len(days) == 1 and args.every in {None, "", "full", "raw"}:
        bbo = load_bbo_day(args.symbol, days[0])
        interval = 1
    else:
        bbo = load_bbo_range(args.symbol, from_date, to_date, every=args.every)
        interval = interval_seconds(args.every)
    print(f"bbo_rows={bbo.height} trades={len(trades)} interval_s={interval}", flush=True)

    kappa = args.kappa
    kappa_path: pl.DataFrame | None = None
    if not args.no_fit_kappa:
        paths: list[pl.DataFrame] = []
        for day in days:
            try:
                A, k, path = estimate_day_kappa(
                    args.symbol, day, tick=args.tick_size
                )
                paths.append(path)
                kappa = k
                print(f"fill_probability {day}: A={A:.6g} kappa_ticks={k:.6g}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"fill_probability {day} failed: {exc}", flush=True)
        if paths:
            kappa_path = pl.concat(paths).sort("win")

    parameters = QuoteParameters(
        base_gamma=args.gamma,
        kappa=kappa,
        time_horizon=args.horizon,
        min_spread=args.min_spread,
        tick_size=args.tick_size,
        max_inventory=args.max_inventory,
        max_spread_ticks=args.max_spread_ticks,
        maker_fee_rate=args.maker_fee,
        include_kappa_spread=True,
    )
    model = AvellanedaStoikovModel(parameters)
    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=args.max_inventory,
        order_size=args.order_size,
    )
    print(
        f"Running backtest gamma={args.gamma} max_spread_ticks={args.max_spread_ticks} ...",
        flush=True,
    )
    result = Backtest(
        model=model,
        execution=execution,
        volatility_window=args.volatility_window,
        interval_seconds=interval,
        kappa_path=kappa_path,
        kappa_fallback=kappa,
    ).run(bbo, trades)

    fees = sum(f.fee for f in execution.fills)
    print(f"symbol={args.symbol} date={date_label}")
    print(f"gamma={args.gamma} max_spread_ticks={args.max_spread_ticks} every={args.every}")
    print(f"kappa_ticks={kappa}")
    print(f"fills={result.fills}")
    print(f"fees={fees}")
    print(f"final_inventory={result.final_inventory}")
    print(f"cash={result.cash}")
    print(f"marked_pnl={result.marked_pnl}")
    print(f"realized_pnl={execution.position.realized_pnl}")


if __name__ == "__main__":
    main()
