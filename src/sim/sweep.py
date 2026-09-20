"""Parameter-grid sweep used by the TUI selection page."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import polars as pl

from .backtest import Backtest, BacktestResult
from .execution import ExecutionSimulator
from .quoting_logic import AvellanedaStoikovModel, QuoteParameters


@dataclass(frozen=True)
class SweepCombo:
    gamma: float
    kappa: float
    min_spread: float
    max_inventory: float
    max_spread_ticks: float = 10.0


@dataclass
class SweepRow:
    combo: SweepCombo
    result: BacktestResult
    error: str | None = None

    @property
    def marked_pnl(self) -> float:
        return 0.0 if self.error else self.result.marked_pnl


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right))


def unique_sorted(values: Iterable[float]) -> list[float]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values if item > 0):
        if not any(_close(value, seen) for seen in unique):
            unique.append(value)
    return unique


def around(center: float) -> list[float]:
    """Three values around a CLI default: 0.5x, 1x, 2x."""
    return unique_sorted([center * 0.5, center, center * 2.0])


def _nice(value: float) -> float:
    if value >= 100:
        exp = int(math.floor(math.log10(value)))
        return float(round(value / 10 ** (exp - 1)) * 10 ** (exp - 1))
    if value >= 1:
        return float(round(value, 2))
    return float(round(value, 6))


def fee_floor_spread(mid: float, maker_fee: float, tick: float) -> float:
    """Round-trip maker fees plus one tick — below this, capturing the spread loses money."""
    return 2.0 * maker_fee * mid + tick


def default_kappas(mid: float, tick: float, requested: float) -> list[float]:
    """Wide κ steps in tick-space so spreads are not all fee-floor clamped."""
    del mid, tick, requested
    return unique_sorted([0.05, 0.25, 1.0, 10.0])


def default_min_spreads(tick: float, requested: float, mid: float, maker_fee: float) -> list[float]:
    """Min-spread candidates from fee floor / requested up to 10 ticks."""
    floor = fee_floor_spread(mid, maker_fee, tick)
    cap = 10.0 * tick
    start = max(floor, min(requested, cap))
    start_ticks = max(1, int(math.ceil(start / tick - 1e-12)))
    values = [max(floor, n * tick) for n in range(start_ticks, 11)]
    values.append(max(requested, floor))
    return unique_sorted(v for v in values if v <= cap + 1e-15)


def default_max_spread_ticks() -> list[float]:
    """Max quoted spread in ticks: 2 … 10."""
    return [float(value) for value in range(2, 11)]


def default_gammas(requested: float) -> list[float]:
    """Ten log-spaced gammas around the CLI default (risk aversion sweep)."""
    center = max(float(requested), 1e-6)
    lo = center / 10.0
    hi = center * 10.0
    steps = 10
    values = [
        _nice(lo * (hi / lo) ** (i / (steps - 1)))
        for i in range(steps)
    ]
    return unique_sorted(values)


def default_max_inventories() -> list[float]:
    """10, 20, …, 500."""
    return [float(value) for value in range(10, 501, 10)]


def default_all_gammas() -> list[float]:
    return unique_sorted([0.01, 0.1, 0.2])


def default_all_kappas() -> list[float]:
    return unique_sorted([0.01, 0.25, 1.0])


def default_all_min_spreads(
    tick: float, requested: float, mid: float, maker_fee: float
) -> list[float]:
    """Coarse min-spread band for --sweep-all: floor, mid, 10 ticks."""
    floor = fee_floor_spread(mid, maker_fee, tick)
    cap = 10.0 * tick
    mid_spread = max(floor, min(requested, cap), 0.5 * (floor + cap))
    return unique_sorted([floor, mid_spread, cap])


def default_all_max_spread_ticks() -> list[float]:
    return [4.0, 8.0, 10.0]


def default_all_max_inventories() -> list[float]:
    return [50.0, 200.0, 500.0]


def build_combos(
    gammas: Iterable[float],
    kappas: Iterable[float],
    min_spreads: Iterable[float],
    max_inventories: Iterable[float],
    max_spread_ticks: Iterable[float] | None = None,
    *,
    tick_size: float = 0.0001,
) -> list[SweepCombo]:
    max_spread_values = list(max_spread_ticks or default_max_spread_ticks())
    combos = [
        SweepCombo(
            gamma=g,
            kappa=k,
            min_spread=s,
            max_inventory=inv,
            max_spread_ticks=max_sp,
        )
        for g, k, s, inv, max_sp in product(
            gammas,
            kappas,
            min_spreads,
            max_inventories,
            max_spread_values,
        )
        if max_sp * tick_size + 1e-15 >= s
    ]
    if not combos:
        raise ValueError("parameter grid is empty")
    return combos


def resample_bbo(bbo: pl.DataFrame, every: str) -> pl.DataFrame:
    """Downsample book snapshots so a full grid can finish in the TUI."""
    keep = [
        column
        for column in (
            "bid_price",
            "bid_amount",
            "ask_price",
            "ask_amount",
            "bid_prices",
            "bid_amounts",
            "ask_prices",
            "ask_amounts",
            "mid",
        )
        if column in bbo.columns
    ]
    return (
        bbo.sort("ts")
        .group_by_dynamic("ts", every=every)
        .agg(*[pl.col(column).last() for column in keep])
    )


def interval_seconds(every: str) -> float:
    token = every.strip().lower()
    if token.endswith("ms"):
        return max(float(token[:-2]) / 1000.0, 1e-6)
    if token.endswith("s"):
        return max(float(token[:-1]), 1e-6)
    if token.endswith("m"):
        return max(float(token[:-1]) * 60.0, 1e-6)
    raise ValueError(f"unsupported resample interval {every!r}")


def run_combo(
    combo: SweepCombo,
    bbo_rows: list[dict],
    trade_rows: list[dict],
    *,
    horizon: float,
    tick_size: float,
    maker_fee: float,
    volatility_window: int,
    interval: int,
    order_size: float = 0.0,
) -> SweepRow:
    parameters = QuoteParameters(
        base_gamma=combo.gamma,
        kappa=combo.kappa,
        time_horizon=horizon,
        min_spread=combo.min_spread,
        tick_size=tick_size,
        max_inventory=combo.max_inventory,
        max_spread_ticks=combo.max_spread_ticks,
        maker_fee_rate=maker_fee,
    )
    try:
        result = Backtest(
            model=AvellanedaStoikovModel(parameters),
            execution=ExecutionSimulator(
                maker_fee_rate=maker_fee,
                max_inventory=combo.max_inventory,
                order_size=order_size,
            ),
            volatility_window=max(3, volatility_window),
            interval_seconds=interval,
        ).run_rows(bbo_rows, trade_rows)
    except Exception as exc:  # noqa: BLE001 - surface any combo failure in the table
        empty = BacktestResult(
            fills=0,
            final_inventory=0.0,
            cash=0.0,
            execution=ExecutionSimulator(max_inventory=combo.max_inventory),
        )
        return SweepRow(combo=combo, result=empty, error=str(exc))
    return SweepRow(combo=combo, result=result)


def run_grid(
    combos: list[SweepCombo],
    bbo_rows: list[dict],
    trade_rows: list[dict],
    *,
    horizon: float,
    tick_size: float,
    maker_fee: float,
    volatility_window: int,
    interval: int,
    on_progress: Callable[[int, int, SweepCombo], None] | None = None,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    total = len(combos)
    for index, combo in enumerate(combos, start=1):
        if on_progress is not None:
            on_progress(index, total, combo)
        rows.append(
            run_combo(
                combo,
                bbo_rows,
                trade_rows,
                horizon=horizon,
                tick_size=tick_size,
                maker_fee=maker_fee,
                volatility_window=volatility_window,
                interval=interval,
            )
        )
    rows.sort(key=lambda row: row.marked_pnl, reverse=True)
    return rows
