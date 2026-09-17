"""Command-line entry point for the pure market-making backtest."""

import argparse

from src.loaders import load_bbo_day, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.backtest import Backtest
from src.sim.execution import ExecutionSimulator
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters


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
    # Defaults match live SUI mainnet quoter.
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--min-spread", type=float, default=0.0004032)
    parser.add_argument("--tick-size", type=float, default=0.0001)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument("--order-size", type=float, default=10.0)
    parser.add_argument("--skew-ticks", type=float, default=10.0)
    parser.add_argument("--max-spread-ticks", type=float, default=10.0)
    parser.add_argument("--book-skew-ticks", type=float, default=8.0)
    parser.add_argument("--book-levels", type=int, default=5)
    parser.add_argument("--book-skew-size-weight", type=float, default=0.75)
    parser.add_argument("--ma-fast", type=int, default=7)
    parser.add_argument("--ma-slow", type=int, default=25)
    parser.add_argument("--ma-skew-ticks", type=float, default=6.0)
    parser.add_argument("--ma-signal-ticks", type=float, default=10.0)
    parser.add_argument("--volatility-window", type=int, default=120)
    args = parser.parse_args()

    from_date, to_date, days = resolve_date_range(
        args.symbol,
        date=args.date,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    date_label = days[0] if len(days) == 1 else f"{days[0]}..{days[-1]} ({len(days)}d)"
    trades = load_trades_range(args.symbol, from_date, to_date)
    if len(days) == 1:
        bbo = load_bbo_day(args.symbol, days[0])
    else:
        bbo = load_bbo_range(args.symbol, from_date, to_date, every="5s")

    parameters = QuoteParameters(
        base_gamma=args.gamma,
        kappa=args.kappa,
        symmetrical_bid=0.0,
        symmetrical_ask=0.0,
        order_book_liquidity=0.0,
        time_horizon=args.horizon,
        min_spread=args.min_spread,
        tick_size=args.tick_size,
        max_inventory=args.max_inventory,
        max_skew_ticks=args.skew_ticks,
        max_spread_ticks=args.max_spread_ticks,
        book_skew_ticks=args.book_skew_ticks,
        book_skew_size_weight=args.book_skew_size_weight,
        book_levels=args.book_levels,
        maker_fee_rate=args.maker_fee,
        ma_fast=args.ma_fast,
        ma_slow=args.ma_slow,
        ma_skew_ticks=args.ma_skew_ticks,
        ma_signal_ticks=args.ma_signal_ticks,
    )
    model = AvellanedaStoikovModel(parameters)
    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=args.max_inventory,
        order_size=args.order_size,
    )
    result = Backtest(
        model=model,
        execution=execution,
        volatility_window=args.volatility_window,
    ).run(bbo, trades)

    print(f"symbol={args.symbol} date={date_label}")
    print(f"fills={result.fills}")
    print(f"final_inventory={result.final_inventory}")
    print(f"cash={result.cash}")
    print(f"marked_pnl={result.marked_pnl}")


if __name__ == "__main__":
    main()
