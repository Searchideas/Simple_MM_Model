"""Convert raw Tardis csv.gz files to parquet.

Layout:
    data/parquet/{symbol}/{data_type}/{original_name}.parquet

Run:
    python -m src.convert
    python -m src.convert --force
    python -m src.convert --keep-raw   # keep csv.gz after conversion
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

TIMESTAMP_COLS = ("timestamp", "local_timestamp")

# This is a function to 
def _out_path(output_dir: Path, symbol: str, data_type: str, raw_path: Path) -> Path:
    return output_dir / symbol.lower() / data_type / raw_path.name.replace(".csv.gz", ".parquet")


def _delete_raw_if_converted(raw_path: Path, out_path: Path) -> bool:
    """Delete csv.gz only when parquet exists and is non-empty."""
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return False
    raw_path.unlink()
    return True


def convert_file(
    raw_path: Path,
    output_dir: Path,
    symbol: str,
    data_type: str,
    force: bool = False,
) -> tuple[str, Path]:
    """Convert one csv.gz to parquet. Returns (status, out_path)."""
    out_path = _out_path(output_dir, symbol, data_type, raw_path)
    if out_path.exists() and not force:
        return "skipped", out_path

    df = pl.read_csv(raw_path, infer_schema_length=100_000)

    ts_cols = [c for c in TIMESTAMP_COLS if c in df.columns]
    if ts_cols:
        df = df.with_columns(pl.from_epoch(pl.col(c), time_unit="us").alias(c) for c in ts_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    return "converted", out_path


def convert_datasets(
    input_dir: Path,
    output_dir: Path,
    data_types: list[str] | None = None,
    symbols: list[str] | None = None,
    force: bool = False,
    delete_raw: bool = True,
) -> dict[str, int]:
    """Convert matching csv.gz files under input_dir into parquet."""
    data_types = data_types or config.data_types
    symbols = symbols or config.Symbols

    counts = {"converted": 0, "skipped": 0, "failed": 0, "deleted": 0}

    for symbol in symbols:
        # Tardis file uses exchange dataset id (e.g. XAU_USDT); parquet folder uses local name
        ds_id = config.dataset_id(symbol)
        for data_type in data_types:
            pattern = f"{config.Exchange}_{data_type}_*_{ds_id}.csv.gz"
            for raw_path in sorted(input_dir.glob(pattern)):
                try:
                    result, out_path = convert_file(
                        raw_path, output_dir, symbol, data_type, force=force
                    )
                except Exception as exc:
                    counts["failed"] += 1
                    print(f"FAILED {raw_path.name}: {exc}")
                    continue

                counts[result] += 1
                if result == "converted":
                    print(f"converted -> {out_path.relative_to(output_dir)}")

                if delete_raw and _delete_raw_if_converted(raw_path, out_path):
                    counts["deleted"] += 1
                    print(f"deleted  -> {raw_path.name}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tardis csv.gz to parquet")
    parser.add_argument("--input-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.PARQUET_DIR)
    parser.add_argument("--symbols", nargs="+", default=config.Symbols)
    parser.add_argument("--data-types", nargs="+", default=config.data_types)
    parser.add_argument("--force", action="store_true", help="re-convert even if parquet exists")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="keep csv.gz files after successful conversion (default: delete them)",
    )
    args = parser.parse_args()

    raw_files = list(args.input_dir.glob("*.csv.gz"))
    if not raw_files:
        print(f"No csv.gz files in {args.input_dir}. Run python -m src.download first.")
        return

    print(f"Converting {args.input_dir} -> {args.output_dir}")
    counts = convert_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        data_types=args.data_types,
        symbols=args.symbols,
        force=args.force,
        delete_raw=not args.keep_raw,
    )
    print(
        f"Done. converted={counts['converted']} skipped={counts['skipped']} "
        f"deleted={counts['deleted']} failed={counts['failed']}"
    )


if __name__ == "__main__":
    main()
