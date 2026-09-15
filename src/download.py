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


if __name__ == "__main__":
    symbols = config.dataset_symbols()
    print(
        f"Downloading {config.Exchange}  symbols={symbols}  "
        f"{config.StartDate} .. {config.EndDate}  -> {config.RAW_DIR}"
    )
    params = DownloadParams(
        exchange=config.Exchange,
        symbols=symbols,
        data_types=config.data_types,
        from_date=config.StartDate,
        to_date=exclusive_end,
        api_key=config.Tardis_API_Key,
        download_dir=str(config.RAW_DIR),
    )
    download_data(params)
    n = len(list(config.RAW_DIR.glob("*.csv.gz")))
    print(f"Data downloaded successfully ({n} csv.gz files in {config.RAW_DIR})")
