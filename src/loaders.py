from __future__ import annotations

import gc
import re
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

BOOK_LEVELS = 25
BID_PRICE_COLS = [f"bids[{level}].price" for level in range(BOOK_LEVELS)]
BID_AMOUNT_COLS = [f"bids[{level}].amount" for level in range(BOOK_LEVELS)]
ASK_PRICE_COLS = [f"asks[{level}].price" for level in range(BOOK_LEVELS)]
ASK_AMOUNT_COLS = [f"asks[{level}].amount" for level in range(BOOK_LEVELS)]
BBO_PARQUET_COLS = [
    "local_timestamp",
    *BID_PRICE_COLS,
    *BID_AMOUNT_COLS,
    *ASK_PRICE_COLS,
    *ASK_AMOUNT_COLS,
]

TRADES_PARQUET_COLS = [
    "local_timestamp",
    "price",
    "amount",
    "side",
]


def parquet_root(exchange: str | None = None) -> Path:
    """Parquet root for an exchange (default: config.Exchange)."""
    if exchange is None:
        return config.PARQUET_DIR
    return config.parquet_dir_for(exchange)


def daily_files(
    symbol: str,
    data_type: str,
    from_date: str,
    to_date: str,
    exchange: str | None = None,
) -> list[Path]:
    folder = parquet_root(exchange) / symbol.lower() / data_type
    files = []
    for f in sorted(folder.glob("*.parquet")):
        m = DATE_RE.search(f.name)
        if m and from_date <= m.group(1) <= to_date:
            files.append(f)
    return files


def load(
    symbol: str,
    data_type: str,
    from_date: str,
    to_date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    files = daily_files(symbol, data_type, from_date, to_date, exchange=exchange)
    if not files:
        raise FileNotFoundError(
            f"No parquet for {symbol}/{data_type} in [{from_date}, {to_date}]"
            + (f" exchange={exchange}" if exchange else "")
        )
    return pl.read_parquet(files).sort("local_timestamp")


def _bbo_from_parquet(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path, columns=BBO_PARQUET_COLS)
        .select(
            "local_timestamp",
            pl.col(BID_PRICE_COLS[0]).alias("bid_price"),
            pl.col(BID_AMOUNT_COLS[0]).alias("bid_amount"),
            pl.col(ASK_PRICE_COLS[0]).alias("ask_price"),
            pl.col(ASK_AMOUNT_COLS[0]).alias("ask_amount"),
            pl.concat_list(BID_PRICE_COLS).alias("bid_prices"),
            pl.concat_list(BID_AMOUNT_COLS).alias("bid_amounts"),
            pl.concat_list(ASK_PRICE_COLS).alias("ask_prices"),
            pl.concat_list(ASK_AMOUNT_COLS).alias("ask_amounts"),
        )
        .with_columns(
            mid=((pl.col("bid_price") + pl.col("ask_price")) / 2),
            microprice=(
                (pl.col("bid_price") * pl.col("ask_amount") + pl.col("ask_price") * pl.col("bid_amount"))
                / (pl.col("bid_amount") + pl.col("ask_amount"))
            ),
        )
    )


def load_bbo(
    symbol: str,
    from_date: str,
    to_date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    files = daily_files(
        symbol, "book_snapshot_25", from_date, to_date, exchange=exchange
    )
    if not files:
        raise FileNotFoundError(
            f"No parquet for {symbol}/book_snapshot_25 in [{from_date}, {to_date}]"
        )

    frames: list[pl.DataFrame] = []
    for i, path in enumerate(files, start=1):
        print(f"  {symbol}: load BBO {i}/{len(files)} {path.name}", flush=True)
        frames.append(_bbo_from_parquet(path))
    return pl.concat(frames).sort("local_timestamp")


def bbo_grid(
    symbol: str,
    from_date: str,
    to_date: str,
    every: str = "1s",
    exchange: str | None = None,
) -> pl.DataFrame:
    """Build time grid one day at a time to keep memory low."""
    files = daily_files(
        symbol, "book_snapshot_25", from_date, to_date, exchange=exchange
    )
    if not files:
        raise FileNotFoundError(
            f"No parquet for {symbol}/book_snapshot_25 in [{from_date}, {to_date}]"
        )

    grids: list[pl.DataFrame] = []
    for i, path in enumerate(files, start=1):
        print(f"  {symbol}: grid {i}/{len(files)} {path.name} (every={every})", flush=True)
        bbo = _bbo_from_parquet(path)
        grid = (
            bbo.group_by_dynamic("local_timestamp", every=every)
            .agg(
                pl.col(
                    "bid_price",
                    "bid_amount",
                    "ask_price",
                    "ask_amount",
                    "bid_prices",
                    "bid_amounts",
                    "ask_prices",
                    "ask_amounts",
                    "mid",
                    "microprice",
                ).last()
            )
            .rename({"local_timestamp": "ts"})
        )
        grids.append(grid)
        del bbo

    return pl.concat(grids).sort("ts")


def aligned_mids(symbols: list[str], from_date: str, to_date: str, every: str = "1s") -> pl.DataFrame:
    out = None
    for sym in symbols:
        print(f"Building grid for {sym}...", flush=True)
        grid = bbo_grid(sym, from_date, to_date, every=every).select(
            "ts",
            pl.col("mid").alias(f"mid_{sym}"),
            pl.col("bid_price").alias(f"bid_{sym}"),
            pl.col("ask_price").alias(f"ask_{sym}"),
        )
        out = grid if out is None else out.join(grid, on="ts", how="full", coalesce=True)
        gc.collect()

    return out.sort("ts").with_columns(pl.exclude("ts").forward_fill()).drop_nulls()


def available_dates(
    symbol: str,
    data_type: str = "book_snapshot_25",
    exchange: str | None = None,
) -> list[str]:
    """Return sorted YYYY-MM-DD dates that have parquet for this symbol."""
    folder = parquet_root(exchange) / symbol.lower() / data_type
    dates: list[str] = []
    if not folder.exists():
        return dates
    for path in sorted(folder.glob("*.parquet")):
        match = DATE_RE.search(path.name)
        if match and match.group(1) not in dates:
            dates.append(match.group(1))
    return dates


def resolve_date_range(
    symbol: str,
    date: str = "all",
    from_date: str | None = None,
    to_date: str | None = None,
    exchange: str | None = None,
) -> tuple[str, str, list[str]]:
    """Resolve CLI dates to an inclusive [from, to] range and the list of days."""
    days = available_dates(symbol, exchange=exchange)
    if not days:
        raise FileNotFoundError(
            f"No parquet dates for {symbol}"
            + (f" exchange={exchange}" if exchange else "")
        )
    start = from_date or (days[0] if date == "all" else date)
    end = to_date or (days[-1] if date == "all" else date)
    picked = [day for day in days if start <= day <= end]
    if not picked:
        raise FileNotFoundError(f"No parquet for {symbol} in [{start}, {end}]")
    return picked[0], picked[-1], picked


def load_trades_range(
    symbol: str,
    from_date: str,
    to_date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    files = daily_files(symbol, "trades", from_date, to_date, exchange=exchange)
    if not files:
        raise FileNotFoundError(f"No trades for {symbol} in [{from_date}, {to_date}]")
    frames = [
        pl.read_parquet(path, columns=TRADES_PARQUET_COLS).select(
            pl.col("local_timestamp").alias("ts"),
            "price",
            "amount",
            "side",
        )
        for path in files
    ]
    return pl.concat(frames).sort("ts")


def load_bbo_range(
    symbol: str,
    from_date: str,
    to_date: str,
    every: str | None = None,
    exchange: str | None = None,
) -> pl.DataFrame:
    """Load books for a date range. ``every`` resamples (needed for multi-day)."""
    if every:
        return bbo_grid(symbol, from_date, to_date, every=every, exchange=exchange)
    days = [
        day
        for day in available_dates(symbol, exchange=exchange)
        if from_date <= day <= to_date
    ]
    if not days:
        raise FileNotFoundError(f"No book for {symbol} in [{from_date}, {to_date}]")
    if len(days) == 1:
        return load_bbo_day(symbol, days[0], exchange=exchange)
    return bbo_grid(symbol, from_date, to_date, every="1s", exchange=exchange)


def load_trades_day(
    symbol: str,
    date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    files = daily_files(symbol, "trades", date, date, exchange=exchange)
    if not files:
        raise FileNotFoundError(f"No trades for {symbol} on {date}")
    return (
        pl.read_parquet(files[0], columns=TRADES_PARQUET_COLS)
        .select(
            pl.col("local_timestamp").alias("ts"),
            "price",
            "amount",
            "side",
        )
        .sort("ts")
    )


def load_bbo_day(
    symbol: str,
    date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    files = daily_files(symbol, "book_snapshot_25", date, date, exchange=exchange)
    if not files:
        raise FileNotFoundError(f"No book for {symbol} on {date}")
    return (
        _bbo_from_parquet(files[0])
        .rename({"local_timestamp": "ts"})
        .select(
            "ts",
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
        .sort("ts")
    )


L2_PARQUET_COLS = [
    "local_timestamp",
    "is_snapshot",
    "side",
    "price",
    "amount",
]


def load_incremental_l2_day(
    symbol: str,
    date: str,
    exchange: str | None = None,
) -> pl.DataFrame:
    """One day of Tardis incremental_book_L2 updates."""
    files = daily_files(
        symbol, "incremental_book_L2", date, date, exchange=exchange
    )
    if not files:
        raise FileNotFoundError(
            f"No incremental_book_L2 for {symbol} on {date}"
            + (f" exchange={exchange}" if exchange else "")
        )
    return (
        pl.read_parquet(files[0], columns=L2_PARQUET_COLS)
        .select(
            pl.col("local_timestamp").alias("ts"),
            "is_snapshot",
            "side",
            "price",
            "amount",
        )
        .sort("ts")
    )


def merge_l2_and_trades(
    l2: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Merge book updates + trades; trades first on equal timestamps."""
    book_ev = l2.select(
        "ts",
        pl.lit("book").alias("kind"),
        pl.col("is_snapshot").cast(pl.Boolean),
        "side",
        "price",
        "amount",
    )
    trade_ev = trades.select(
        "ts",
        pl.lit("trade").alias("kind"),
        pl.lit(False).alias("is_snapshot"),
        "side",
        "price",
        "amount",
    )
    # kind sort: trade < book so trades process first on ties.
    return (
        pl.concat([trade_ev, book_ev], how="diagonal_relaxed")
        .sort(["ts", "kind"])
        .with_columns(
            pl.when(pl.col("kind") == "trade")
            .then(0)
            .otherwise(1)
            .alias("_ord")
        )
        .sort(["ts", "_ord"])
        .drop("_ord")
    )
