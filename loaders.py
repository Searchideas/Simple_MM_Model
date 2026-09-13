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


def daily_files(symbol: str, data_type: str, from_date: str, to_date: str) -> list[Path]:
    folder = config.PARQUET_DIR / symbol.lower() / data_type
    files = []
    for f in sorted(folder.glob("*.parquet")):
        m = DATE_RE.search(f.name)
        if m and from_date <= m.group(1) <= to_date:
            files.append(f)
    return files


def load(symbol: str, data_type: str, from_date: str, to_date: str) -> pl.DataFrame:
    files = daily_files(symbol, data_type, from_date, to_date)
    if not files:
        raise FileNotFoundError(f"No parquet for {symbol}/{data_type} in [{from_date}, {to_date}]")
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


def load_bbo(symbol: str, from_date: str, to_date: str) -> pl.DataFrame:
    files = daily_files(symbol, "book_snapshot_25", from_date, to_date)
    if not files:
        raise FileNotFoundError(f"No parquet for {symbol}/book_snapshot_25 in [{from_date}, {to_date}]")

    frames: list[pl.DataFrame] = []
    for i, path in enumerate(files, start=1):
        print(f"  {symbol}: load BBO {i}/{len(files)} {path.name}", flush=True)
        frames.append(_bbo_from_parquet(path))
    return pl.concat(frames).sort("local_timestamp")


def bbo_grid(symbol: str, from_date: str, to_date: str, every: str = "1s") -> pl.DataFrame:
    """Build time grid one day at a time to keep memory low."""
    files = daily_files(symbol, "book_snapshot_25", from_date, to_date)
    if not files:
        raise FileNotFoundError(f"No parquet for {symbol}/book_snapshot_25 in [{from_date}, {to_date}]")

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


def load_trades_day(symbol: str, date: str) -> pl.DataFrame:
    files = daily_files(symbol, "trades", date, date)
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


def load_bbo_day(symbol: str, date: str) -> pl.DataFrame:
    files = daily_files(symbol, "book_snapshot_25", date, date)
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
