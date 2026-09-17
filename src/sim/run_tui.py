"""Run the pure market-making backtest with a terminal dashboard."""

import argparse

from rich.live import Live

from src.loaders import load_bbo_day, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.backtest import Backtest
from src.sim.execution import ExecutionSimulator
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.sweep import (
    build_combos,
    default_all_book_skew_size_weights,
    default_all_book_skew_ticks,
    default_all_gammas,
    default_all_kappas,
    default_all_max_inventories,
    default_all_max_spread_ticks,
    default_all_min_spreads,
    default_all_skew_ticks,
    default_book_skew_size_weights,
    default_book_skew_ticks,
    default_gammas,
    default_max_inventories,
    default_max_spread_ticks,
    default_min_spreads,
    default_skew_ticks,
    interval_seconds,
    resample_bbo,
    run_combo,
)
from src.sim.tui import BacktestTUI, SweepPicker


def _quote_parameters(
    gamma: float,
    kappa: float,
    min_spread: float,
    args,
    max_inventory: float,
    *,
    skew_ticks: float | None = None,
    book_skew_ticks: float | None = None,
    book_skew_size_weight: float | None = None,
    max_spread_ticks: float | None = None,
) -> QuoteParameters:
    return QuoteParameters(
        base_gamma=gamma,
        kappa=kappa,
        symmetrical_bid=0.0,
        symmetrical_ask=0.0,
        order_book_liquidity=0.0,
        time_horizon=args.horizon,
        min_spread=min_spread,
        tick_size=args.tick_size,
        max_inventory=max_inventory,
        max_skew_ticks=skew_ticks if skew_ticks is not None else args.skew_ticks,
        max_spread_ticks=(
            max_spread_ticks if max_spread_ticks is not None else args.max_spread_ticks
        ),
        book_skew_ticks=book_skew_ticks if book_skew_ticks is not None else args.book_skew_ticks,
        book_skew_size_weight=(
            book_skew_size_weight
            if book_skew_size_weight is not None
            else args.book_skew_size_weight
        ),
        book_levels=args.book_levels,
        maker_fee_rate=args.maker_fee,
        ma_fast=args.ma_fast,
        ma_slow=args.ma_slow,
        ma_skew_ticks=args.ma_skew_ticks,
        ma_signal_ticks=args.ma_signal_ticks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="suiusdt")
    parser.add_argument(
        "--date",
        default="all",
        help="YYYY-MM-DD or 'all' for every downloaded day (default: all).",
    )
    parser.add_argument("--from-date", default=None, help="Inclusive start date (overrides --date).")
    parser.add_argument("--to-date", default=None, help="Inclusive end date (overrides --date).")
    # Defaults match live SUI mainnet quoter.
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--min-spread", type=float, default=0.0004032)
    parser.add_argument("--tick-size", type=float, default=0.0001)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument("--order-size", type=float, default=10.0, help="Max fill per trade; 0 = 2% of max inventory.")
    parser.add_argument("--skew-ticks", type=float, default=10.0, help="Inventory shift in ticks at full position.")
    parser.add_argument(
        "--max-spread-ticks",
        type=float,
        default=10.0,
        help="Hard cap on quoted spread in ticks (AS+vol clipped to this).",
    )
    parser.add_argument(
        "--book-skew-ticks",
        type=float,
        default=8.0,
        help="Reservation shift in ticks at full top-N book imbalance.",
    )
    parser.add_argument(
        "--book-levels",
        type=int,
        default=5,
        help="Number of book levels used for imbalance (default: 5).",
    )
    parser.add_argument(
        "--book-skew-size-weight",
        type=float,
        default=0.75,
        help="OBI blend weight on size imbalance I (0=VWAP only, 1=size only).",
    )
    parser.add_argument("--ma-fast", type=int, default=7, help="Fast MA length on 1m closes.")
    parser.add_argument("--ma-slow", type=int, default=25, help="Slow MA length on 1m closes.")
    parser.add_argument("--ma-skew-ticks", type=float, default=6.0, help="Max MA reservation shift in ticks.")
    parser.add_argument(
        "--ma-signal-ticks",
        type=float,
        default=10.0,
        help="MA diff (in ticks) that maps to |ma_frac|=1.",
    )
    parser.add_argument("--volatility-window", type=int, default=120)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Replay speed multiplier; 0 runs as fast as possible, 1 is real-time.",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip the parameter grid page and replay with CLI values only.",
    )
    parser.add_argument(
        "--sweep-every",
        default="5s",
        help="Resample interval for the grid (and for multi-day replay).",
    )
    parser.add_argument(
        "--sweep-rest",
        action="store_true",
        help=(
            "Pin gamma/kappa to --gammas/--kappas (or CLI), and sweep "
            "min_spread, max_inventory, skew, book skew, and OBI weight grids."
        ),
    )
    parser.add_argument(
        "--sweep-spreads",
        action="store_true",
        help=(
            "Pin all knobs except min_spread and max_spread_ticks; "
            "search fillable spread bands (default min 2..10 ticks, max 2..10 ticks)."
        ),
    )
    parser.add_argument(
        "--sweep-skew",
        action="store_true",
        help=(
            "Pin all knobs except inventory skew, book skew, and OBI weight; "
            "default grids: skew 1..10, book 1..10, w 0/0.25/0.5/0.75/1."
        ),
    )
    parser.add_argument(
        "--sweep-all",
        action="store_true",
        help=(
            "Sweep every axis with a coarse full grid "
            "(γ, κ, min/max spread, inventory, skew, book, w). "
            "Override any axis with --gammas / --kappas / etc."
        ),
    )
    parser.add_argument("--gammas", nargs="+", type=float, default=None)
    parser.add_argument("--kappas", nargs="+", type=float, default=None)
    parser.add_argument("--min-spreads", nargs="+", type=float, default=None)
    parser.add_argument(
        "--max-spread-ticks-grid",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for max spread ticks (default with --sweep-spreads: 2..10).",
    )
    parser.add_argument(
        "--max-inventories",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for max inventory (default with --sweep-rest: 10,20,…,500).",
    )
    parser.add_argument(
        "--skew-ticks-grid",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for inventory skew ticks (default with --sweep-rest: 1..10).",
    )
    parser.add_argument(
        "--book-skew-ticks-grid",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for book skew ticks (default with --sweep-rest: 1..10).",
    )
    parser.add_argument(
        "--book-skew-weights",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for OBI size weight (default with --sweep-rest: 0..1 step 0.25).",
    )
    args = parser.parse_args()

    from_date, to_date, days = resolve_date_range(
        args.symbol,
        date=args.date,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    date_label = days[0] if len(days) == 1 else f"{days[0]}..{days[-1]} ({len(days)}d)"
    print(f"Loading {args.symbol} {date_label}", flush=True)
    trades = load_trades_range(args.symbol, from_date, to_date)
    if len(days) == 1:
        bbo = load_bbo_day(args.symbol, days[0])
        sweep_bbo = resample_bbo(bbo, args.sweep_every)
        replay_bbo = bbo
    else:
        # Multi-day full books are too large; sweep and replay share a resampled grid.
        replay_bbo = load_bbo_range(args.symbol, from_date, to_date, every=args.sweep_every)
        sweep_bbo = replay_bbo

    gamma = args.gamma
    kappa = args.kappa
    min_spread = args.min_spread
    max_inventory = args.max_inventory
    skew_ticks: float | None = None
    book_skew_ticks: float | None = None
    book_skew_size_weight: float | None = None
    max_spread_ticks: float | None = None

    if not args.no_sweep:
        mid = float((sweep_bbo["bid_price"][0] + sweep_bbo["ask_price"][0]) / 2.0)
        if args.sweep_all:
            gammas = args.gammas or default_all_gammas()
            kappas = args.kappas or default_all_kappas()
            min_spreads = args.min_spreads or default_all_min_spreads(
                args.tick_size, args.min_spread, mid, args.maker_fee
            )
            max_inventories = args.max_inventories or default_all_max_inventories()
            skew_grid = args.skew_ticks_grid or default_all_skew_ticks()
            book_grid = args.book_skew_ticks_grid or default_all_book_skew_ticks()
            weight_grid = args.book_skew_weights or default_all_book_skew_size_weights()
            max_spread_grid = args.max_spread_ticks_grid or default_all_max_spread_ticks()
        elif args.sweep_spreads:
            gammas = args.gammas or [args.gamma]
            kappas = args.kappas or [args.kappa]
            min_spreads = args.min_spreads or default_min_spreads(
                args.tick_size, args.min_spread, mid, args.maker_fee
            )
            max_inventories = args.max_inventories or [args.max_inventory]
            skew_grid = args.skew_ticks_grid or [args.skew_ticks]
            book_grid = args.book_skew_ticks_grid or [args.book_skew_ticks]
            weight_grid = args.book_skew_weights or [args.book_skew_size_weight]
            max_spread_grid = args.max_spread_ticks_grid or default_max_spread_ticks()
        elif args.sweep_skew:
            # Inventory skew + book skew + OBI weight only.
            gammas = args.gammas or [args.gamma]
            kappas = args.kappas or [args.kappa]
            min_spreads = args.min_spreads or [args.min_spread]
            max_inventories = args.max_inventories or [args.max_inventory]
            skew_grid = args.skew_ticks_grid or default_skew_ticks()
            book_grid = args.book_skew_ticks_grid or default_book_skew_ticks()
            weight_grid = args.book_skew_weights or default_book_skew_size_weights()
            max_spread_grid = args.max_spread_ticks_grid or [args.max_spread_ticks]
        elif args.sweep_rest:
            gammas = args.gammas or [args.gamma]
            kappas = args.kappas or [args.kappa]
            min_spreads = args.min_spreads or default_min_spreads(
                args.tick_size, args.min_spread, mid, args.maker_fee
            )
            max_inventories = args.max_inventories or default_max_inventories()
            skew_grid = args.skew_ticks_grid or default_skew_ticks()
            book_grid = args.book_skew_ticks_grid or default_book_skew_ticks()
            weight_grid = args.book_skew_weights or default_book_skew_size_weights()
            max_spread_grid = args.max_spread_ticks_grid or [args.max_spread_ticks]
        else:
            gammas = args.gammas or default_gammas(args.gamma)
            kappas = args.kappas or [args.kappa]
            min_spreads = args.min_spreads or [args.min_spread]
            max_inventories = args.max_inventories or [args.max_inventory]
            skew_grid = args.skew_ticks_grid or [args.skew_ticks]
            book_grid = args.book_skew_ticks_grid or [args.book_skew_ticks]
            weight_grid = args.book_skew_weights or [args.book_skew_size_weight]
            max_spread_grid = args.max_spread_ticks_grid or [args.max_spread_ticks]

        combos = build_combos(
            gammas,
            kappas,
            min_spreads,
            max_inventories,
            skew_ticks=skew_grid,
            book_skew_ticks=book_grid,
            book_skew_size_weights=weight_grid,
            max_spread_ticks=max_spread_grid,
            tick_size=args.tick_size,
        )
        print(
            f"Sweep: {len(combos)} combinations "
            f"(gamma={list(gammas)} kappa={list(kappas)} "
            f"skew={list(skew_grid)} book={list(book_grid)} w={list(weight_grid)})",
            flush=True,
        )
        interval = interval_seconds(args.sweep_every)
        bbo_rows = list(sweep_bbo.sort("ts").iter_rows(named=True))
        trade_rows = list(trades.sort("ts").iter_rows(named=True))
        picker = SweepPicker(
            symbol=args.symbol,
            date=date_label,
            every=args.sweep_every,
            tick_size=args.tick_size,
            horizon=args.horizon,
            maker_fee=args.maker_fee,
            volatility_window=args.volatility_window,
        )
        finished = []
        with Live(refresh_per_second=8) as live:
            total = len(combos)
            for index, combo in enumerate(combos, start=1):
                live.update(picker.render_progress(index, total, combo, finished))
                finished.append(
                    run_combo(
                        combo,
                        bbo_rows,
                        trade_rows,
                        horizon=args.horizon,
                        tick_size=args.tick_size,
                        maker_fee=args.maker_fee,
                        volatility_window=args.volatility_window,
                        interval=interval,
                        book_levels=args.book_levels,
                        order_size=args.order_size,
                        ma_fast=args.ma_fast,
                        ma_slow=args.ma_slow,
                        ma_skew_ticks=args.ma_skew_ticks,
                        ma_signal_ticks=args.ma_signal_ticks,
                    )
                )
                live.update(picker.render_progress(index, total, combo, finished))
        rows = sorted(finished, key=lambda item: item.marked_pnl, reverse=True)
        while True:
            selected = picker.choose(rows)
            if selected is None:
                print("Sweep cancelled.")
                return
            action = picker.show_report(selected)
            if action == "back":
                continue
            if action == "quit":
                print("Done.")
                return
            gamma = selected.combo.gamma
            kappa = selected.combo.kappa
            min_spread = selected.combo.min_spread
            max_inventory = selected.combo.max_inventory
            skew_ticks = selected.combo.skew_ticks
            book_skew_ticks = selected.combo.book_skew_ticks
            book_skew_size_weight = selected.combo.book_skew_size_weight
            max_spread_ticks = selected.combo.max_spread_ticks
            break

    parameters = _quote_parameters(
        gamma,
        kappa,
        min_spread,
        args,
        max_inventory,
        skew_ticks=skew_ticks,
        book_skew_ticks=book_skew_ticks,
        book_skew_size_weight=book_skew_size_weight,
        max_spread_ticks=max_spread_ticks,
    )
    model = AvellanedaStoikovModel(parameters)
    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=max_inventory,
        order_size=args.order_size,
    )
    backtest = Backtest(
        model=model,
        execution=execution,
        volatility_window=args.volatility_window,
    )

    with BacktestTUI(
        speed=args.speed,
        symbol=args.symbol,
        date=date_label,
        parameters=parameters,
        maker_fee=args.maker_fee,
        max_inventory=max_inventory,
        volatility_window=args.volatility_window,
    ) as dashboard:
        result = backtest.run(replay_bbo, trades, on_update=dashboard.update)

    print(
        f"Finished: fills={result.fills} "
        f"inventory={result.final_inventory} cash={result.cash}"
    )


if __name__ == "__main__":
    main()
