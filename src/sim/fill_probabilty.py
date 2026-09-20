"""Fit fill intensity λ(δ)=A·e^{-κδ} from book + trades (3m tumbling windows).

δ is in ticks, so fitted κ_ticks has units 1/tick.
AS quoting converts to price units: κ_$ = κ_ticks / tick_size.

Note on hits: counts trades that reach/cross mid±δ, not whether a
hypothetical resting maker order at that level would have filled.
Empty 3m windows (no trades) are omitted, so λ is conditional on
trade-containing windows with fixed 180s exposure each.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from src.loaders import load_bbo_day, load_trades_day


def load_date_data(
    symbol: str,
    date: str,
    exchange: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        load_bbo_day(symbol, date, exchange=exchange),
        load_trades_day(symbol, date, exchange=exchange),
    )


def fill_rates_tumbling_3m(
    bbo: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    tick: float = 0.0001,
    max_ticks: int = 10,
    window: str = "3m",
    window_seconds: float = 180.0,
) -> pl.DataFrame:
    """Per tumbling window: touch intensity λ(δ) for δ = 1..max_ticks ticks.

    A hit is a trade that reaches at least ``δ`` away from contemporaneous mid
    (sell ≤ mid−d or buy ≥ mid+d). That is a price-touch proxy, not a true
    queue-position maker fill.

    Columns: win, delta_ticks, delta, hits, exposure, lam
    """
    t = (
        trades.sort("ts")
        .join_asof(bbo.select("ts", "mid").sort("ts"), on="ts")
        .drop_nulls("mid")
        .with_columns(pl.col("ts").dt.truncate(window).alias("win"))
    )

    rows: list[dict] = []
    for win_df in t.partition_by("win", maintain_order=True):
        exposure = float(window_seconds)  # fixed 3m; silent windows omitted
        mid = win_df["mid"].to_numpy()
        price = win_df["price"].to_numpy()
        side = win_df["side"].to_numpy()
        win = win_df["win"][0]

        for k in range(1, max_ticks + 1):
            d = k * tick
            hits = int(
                (
                    ((side == "sell") & (price <= mid - d))
                    | ((side == "buy") & (price >= mid + d))
                ).sum()
            )
            rows.append(
                {
                    "win": win,
                    "delta_ticks": k,
                    "delta": d,
                    "hits": hits,
                    "exposure": exposure,
                    "lam": hits / exposure,
                }
            )
    return pl.DataFrame(rows)


def fit_A_kappa(
    delta_ticks: np.ndarray,
    lam: np.ndarray,
) -> tuple[float, float]:
    """ln(λ)=ln(A)-κ_ticks·δ_ticks → (A, kappa_ticks). δ in ticks."""
    mask = np.asarray(lam) > 0
    x = np.asarray(delta_ticks, dtype=float)[mask]
    y = np.log(np.asarray(lam, dtype=float)[mask])
    if x.size < 2:
        raise ValueError("need at least 2 positive λ points")
    slope, intercept = np.polyfit(x, y, 1)
    kappa_ticks = float(-slope)
    if kappa_ticks <= 0:
        raise ValueError(f"kappa_ticks={kappa_ticks} <= 0")
    return float(np.exp(intercept)), kappa_ticks


def fit_A_kappa_pooled(rates: pl.DataFrame) -> tuple[float, float]:
    """Day-level baseline (A, kappa_ticks); assumes constant κ across windows."""
    return fit_A_kappa(rates["delta_ticks"].to_numpy(), rates["lam"].to_numpy())


def fit_A_kappa_by_window(rates: pl.DataFrame) -> pl.DataFrame:
    """Per-3m path with columns win, A, kappa_ticks."""
    out: list[dict] = []
    for key, wdf in rates.group_by("win", maintain_order=True):
        win = key[0] if isinstance(key, tuple) else key
        try:
            A, kappa_ticks = fit_A_kappa(
                wdf["delta_ticks"].to_numpy(),
                wdf["lam"].to_numpy(),
            )
        except ValueError:
            continue
        out.append({"win": win, "A": A, "kappa_ticks": kappa_ticks})
    return pl.DataFrame(out)


def estimate_day_kappa(
    symbol: str,
    date: str,
    *,
    tick: float = 0.0001,
    max_ticks: int = 10,
    exchange: str | None = None,
) -> tuple[float, float, pl.DataFrame]:
    """Fit pooled (A, κ_ticks) and per-3m path for one parquet day.

    Returns ``(A, kappa_ticks, path_df)`` where path has win, A, kappa_ticks.
    """
    bbo, trades = load_date_data(symbol, date, exchange=exchange)
    rates = fill_rates_tumbling_3m(bbo, trades, tick=tick, max_ticks=max_ticks)
    A, kappa_ticks = fit_A_kappa_pooled(rates)
    path = fit_A_kappa_by_window(rates)
    return A, kappa_ticks, path


def kappa_at_time(path: pl.DataFrame, timestamp, fallback: float) -> float:
    """Latest 3m-window kappa_ticks at or before ``timestamp``; else ``fallback``."""
    if path is None or path.is_empty():
        return float(fallback)
    col = "kappa_ticks" if "kappa_ticks" in path.columns else "kappa"
    prior = path.filter(pl.col("win") <= timestamp)
    if prior.is_empty():
        return float(fallback)
    return float(prior[col][-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="suiusdt")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--tick-size", type=float, default=0.0001)
    p.add_argument("--max-ticks", type=int, default=10)
    args = p.parse_args()

    bbo, trades = load_date_data(args.symbol, args.date)
    rates = fill_rates_tumbling_3m(
        bbo, trades, tick=args.tick_size, max_ticks=args.max_ticks
    )
    A, kappa_ticks = fit_A_kappa_pooled(rates)
    path = fit_A_kappa_by_window(rates)
    tick = args.tick_size
    print(f"symbol={args.symbol} date={args.date}")
    print(f"pooled A={A:.6g}  kappa_ticks={kappa_ticks:.6g}  (1/tick)")
    print(f"kappa_$={kappa_ticks / tick:.6g}  (1/price, tick={tick})")
    print(f"windows fitted={path.height} / unique={rates['win'].n_unique()}")
    if path.height:
        print(path.head(5))


if __name__ == "__main__":
    main()
