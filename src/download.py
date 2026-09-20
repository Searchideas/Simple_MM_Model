from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys

import aiohttp.connector
import aiohttp.resolver
from tardis_dev import download_datasets

# Ensure project root is on sys.path for both `python src/download.py` and `python -m src.download`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# DNS fix (Windows: aiodns resolver can fail with "Could not contact DNS servers")
aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver
aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver

exclusive_end = (date.fromisoformat(config.EndDate) + timedelta(days=1)).isoformat()


@dataclass
class DownloadParams:
    exchange: str
    symbols: list[str]
    data_types: list[str]
    from_date: str
    to_date: str
    api_key: str
    download_dir: str


def download_data(params: DownloadParams) -> None:
    download_datasets(
        exchange=params.exchange,
        data_types=params.data_types,
        symbols=params.symbols,
        from_date=params.from_date,
        to_date=params.to_date,
        api_key=params.api_key,
        download_dir=params.download_dir,
        skip_if_exists=True,
    )


def download_jobs(
    jobs: list[tuple[str, list[str]]] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    data_types: list[str] | None = None,
) -> None:
    """Download each (exchange, local_symbols) job into data/raw/{exchange}/."""
    jobs = jobs if jobs is not None else list(config.DOWNLOAD_JOBS)
    from_date = from_date or config.StartDate
    to_date = to_date or exclusive_end
    data_types = data_types or config.data_types

    for exchange, local_symbols in jobs:
        dataset_ids = [config.dataset_id(s, exchange) for s in local_symbols]
        out_dir = config.raw_dir_for(exchange)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {exchange}  symbols={dataset_ids}  "
            f"{from_date} .. {config.EndDate}  -> {out_dir}"
        )
        download_data(
            DownloadParams(
                exchange=exchange,
                symbols=dataset_ids,
                data_types=data_types,
                from_date=from_date,
                to_date=to_date,
                api_key=config.Tardis_API_Key,
                download_dir=str(out_dir),
            )
        )
        n = len(list(out_dir.glob("*.csv.gz")))
        print(f"  done ({n} csv.gz files in {out_dir})")


if __name__ == "__main__":
    download_jobs()
    print("Data downloaded successfully")
