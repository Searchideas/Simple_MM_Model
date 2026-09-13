"""Command-line entry point for the pure market-making backtest."""

import argparse

from src.loaders import load_bbo_day, load_trades_day
from src.sim.backtest import Backtest
from src.sim.execution import ExecutionSimulator
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="xautusdt")
    parser.add_argument("--date", required=True)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=1.5)
    parser.add_argument("--horizon", type=float, default=3600.0)
    parser.add_argument("--min-spread", type=float, default=0.50)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--max-inventory", type=float, default=1.0)
    parser.add_argument("--volatility-window", type=int, default=300)
    args = parser.parse_args()

    bbo = load_bbo_day(args.symbol, args.date)
    trades = load_trades_day(args.symbol, args.date)

    parameters = QuoteParameters(
        base_gamma=args.gamma,
        kappa=args.kappa,
        symmetrical_bid=0.0,
        symmetrical_ask=0.0,
        order_book_liquidity=0.0,
        time_horizon=args.horizon,
        min_spread=args.min_spread,
        tick_size=args.tick_size,
    )
    model = AvellanedaStoikovModel(parameters)
    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=args.max_inventory,
    )
    result = Backtest(
        model=model,
        execution=execution,
        volatility_window=args.volatility_window,
    ).run(bbo, trades)

    print(f"symbol={args.symbol} date={args.date}")
    print(f"fills={result.fills}")
    print(f"final_inventory={result.final_inventory}")
    print(f"cash={result.cash}")


if __name__ == "__main__":
    main()
