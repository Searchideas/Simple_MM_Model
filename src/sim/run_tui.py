"""Run the pure market-making backtest with a terminal dashboard."""

import argparse

from rich.live import Live

from src.loaders import load_bbo_day, load_bbo_range, load_trades_range, resolve_date_range
from src.sim.backtest import Backtest
from src.sim.execution import ExecutionSimulator
from src.sim.fill_probabilty import estimate_day_kappa
from src.sim.quoting_logic import AvellanedaStoikovModel, QuoteParameters
from src.sim.sweep import (
    build_combos,
    default_all_gammas,
    default_all_max_inventories,
    default_all_max_spread_ticks,
    default_all_min_spreads,
    default_gammas,
    default_max_inventories,
    default_max_spread_ticks,
    default_min_spreads,
    interval_seconds,
    resample_bbo,
    run_combo,
)
from src.sim.tui import BacktestTUI, ResultReport, SweepPicker, show_result_report

import polars as pl


def _quote_parameters(
    gamma: float,
    kappa: float,
    min_spread: float,
    args,
    max_inventory: float,
    *,
    max_spread_ticks: float | None = None,
) -> QuoteParameters:
    return QuoteParameters(
        base_gamma=gamma,
        kappa=kappa,
        time_horizon=args.horizon,
        min_spread=min_spread,
        tick_size=args.tick_size,
        max_inventory=max_inventory,
        max_spread_ticks=(
            max_spread_ticks if max_spread_ticks is not None else args.max_spread_ticks
        ),
        maker_fee_rate=args.maker_fee,
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
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=300.0)
    parser.add_argument("--min-spread", type=float, default=0.0004032)
    parser.add_argument("--tick-size", type=float, default=0.0001)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--max-inventory", type=float, default=50.0)
    parser.add_argument(
        "--order-size",
        type=float,
        default=10.0,
        help="Max fill per trade; 0 = 2% of max inventory.",
    )
    parser.add_argument(
        "--max-spread-ticks",
        type=float,
        default=10.0,
        help="Hard cap on quoted spread in ticks (AS clipped to this).",
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
        "--sweep-spreads",
        action="store_true",
        help="Pin γ/κ/inv; sweep min_spread and max_spread_ticks.",
    )
    parser.add_argument(
        "--sweep-all",
        action="store_true",
        help="Sweep γ, κ, min/max spread, inventory (override with --gammas etc.).",
    )
    parser.add_argument("--gammas", nargs="+", type=float, default=None)
    parser.add_argument("--kappas", nargs="+", type=float, default=None)
    parser.add_argument("--min-spreads", nargs="+", type=float, default=None)
    parser.add_argument(
        "--max-spread-ticks-grid",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for max spread ticks.",
    )
    parser.add_argument(
        "--max-inventories",
        nargs="+",
        type=float,
        default=None,
        help="Sweep grid for max inventory.",
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
        replay_bbo = load_bbo_range(args.symbol, from_date, to_date, every=args.sweep_every)
        sweep_bbo = replay_bbo

    fitted_kappa = args.kappa
    kappa_path: pl.DataFrame | None = None
    paths: list[pl.DataFrame] = []
    for day in days:
        try:
            A, k, path = estimate_day_kappa(args.symbol, day, tick=args.tick_size)
            paths.append(path)
            fitted_kappa = k
            print(f"fill_probability {day}: A={A:.6g} kappa_ticks={k:.6g}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"fill_probability {day} failed: {exc}", flush=True)
    if paths:
        kappa_path = pl.concat(paths).sort("win")
    args.kappa = fitted_kappa

    gamma = args.gamma
    kappa = fitted_kappa
    min_spread = args.min_spread
    max_inventory = args.max_inventory
    max_spread_ticks: float | None = None

    if not args.no_sweep:
        mid = float((sweep_bbo["bid_price"][0] + sweep_bbo["ask_price"][0]) / 2.0)
        if args.sweep_all:
            gammas = args.gammas or default_all_gammas()
            kappas = args.kappas or [fitted_kappa]
            min_spreads = args.min_spreads or default_all_min_spreads(
                args.tick_size, args.min_spread, mid, args.maker_fee
            )
            max_inventories = args.max_inventories or default_all_max_inventories()
            max_spread_grid = args.max_spread_ticks_grid or default_all_max_spread_ticks()
        elif args.sweep_spreads:
            gammas = args.gammas or [args.gamma]
            kappas = args.kappas or [fitted_kappa]
            min_spreads = args.min_spreads or default_min_spreads(
                args.tick_size, args.min_spread, mid, args.maker_fee
            )
            max_inventories = args.max_inventories or [args.max_inventory]
            max_spread_grid = args.max_spread_ticks_grid or default_max_spread_ticks()
        else:
            gammas = args.gammas or default_gammas(args.gamma)
            kappas = args.kappas or [fitted_kappa]
            min_spreads = args.min_spreads or [args.min_spread]
            max_inventories = args.max_inventories or [args.max_inventory]
            max_spread_grid = args.max_spread_ticks_grid or [args.max_spread_ticks]

        combos = build_combos(
            gammas,
            kappas,
            min_spreads,
            max_inventories,
            max_spread_ticks=max_spread_grid,
            tick_size=args.tick_size,
        )
        print(
            f"Sweep: {len(combos)} combinations "
            f"(gamma={list(gammas)} kappa={list(kappas)} from fill_probability)",
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
                        order_size=args.order_size,
                    )
                )
                live.update(picker.render_progress(index, total, combo, finished))
        rows = sorted(finished, key=lambda item: item.marked_pnl, reverse=True)
        selected = picker.choose(rows)
        if selected is None:
            print("Sweep cancelled.")
            return
        gamma = selected.combo.gamma
        kappa = selected.combo.kappa
        min_spread = selected.combo.min_spread
        max_inventory = selected.combo.max_inventory
        max_spread_ticks = selected.combo.max_spread_ticks
        from_sweep = True
    else:
        from_sweep = False

    while True:
        parameters = _quote_parameters(
            gamma,
            kappa,
            min_spread,
            args,
            max_inventory,
            max_spread_ticks=max_spread_ticks,
        )

        print("Computing backtest summary...", flush=True)
        summary_model = AvellanedaStoikovModel(parameters)
        if kappa_path is not None:
            summary_model.estimated_kappa = fitted_kappa
        summary_result = Backtest(
            model=summary_model,
            execution=ExecutionSimulator(
                maker_fee_rate=args.maker_fee,
                max_inventory=max_inventory,
                order_size=args.order_size,
            ),
            volatility_window=args.volatility_window,
            kappa_path=kappa_path,
            kappa_fallback=fitted_kappa,
        ).run(replay_bbo, trades)

        report_action = show_result_report(
            ResultReport(
                args.symbol,
                date_label,
                summary_result,
                gamma=parameters.base_gamma,
                kappa=summary_result.last_kappa or parameters.kappa,
                min_spread=parameters.min_spread,
                max_spread_ticks=parameters.max_spread_ticks,
                max_inventory=max_inventory,
                allow_back=from_sweep,
            )
        )
        if report_action == "back" and from_sweep:
            selected = picker.choose(rows)
            if selected is None:
                print("Sweep cancelled.")
                return
            gamma = selected.combo.gamma
            kappa = selected.combo.kappa
            min_spread = selected.combo.min_spread
            max_inventory = selected.combo.max_inventory
            max_spread_ticks = selected.combo.max_spread_ticks
            continue
        if report_action != "replay":
            print("Done.")
            return
        break

    model = AvellanedaStoikovModel(parameters)
    if kappa_path is not None:
        model.estimated_kappa = fitted_kappa
    execution = ExecutionSimulator(
        maker_fee_rate=args.maker_fee,
        max_inventory=max_inventory,
        order_size=args.order_size,
    )
    backtest = Backtest(
        model=model,
        execution=execution,
        volatility_window=args.volatility_window,
        kappa_path=kappa_path,
        kappa_fallback=fitted_kappa,
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
