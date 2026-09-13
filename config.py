from datetime import date, timedelta
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

### Configuration Parameters
# Tardis exchange id: "binance-futures" | "gate-io-futures" | "bybit" (Bybit Derivatives = perps + futures)
Exchange = "binance-futures"

# Local symbols used by download and convert
Symbols = ["xauusdt", "xautusdt", "paxgusdt", "btcusdt", "suiusdt"]

# Tardis dataset symbol ids per exchange
SYMBOL_DATASET_IDS: dict[str, dict[str, str]] = {
    "binance-futures": {
        "xauusdt": "XAUUSDT",
        "xautusdt": "XAUTUSDT",
        "paxgusdt": "PAXGUSDT",
        "btcusdt": "BTCUSDT",
        "suiusdt": "SUIUSDT",
    },
    "gate-io-futures": {
        "xauusdt": "XAU_USDT",
        "xautusdt": "XAUT_USDT",
        "paxgusdt": "PAXG_USDT",
        "btcusdt": "BTC_USDT",
        "suiusdt": "SUI_USDT",
    },
    # Bybit linear USDT perpetuals (not inverse *PERP, not dated futures)
    "bybit": {
        "xauusdt": "XAUUSDT",
        "xautusdt": "XAUTUSDT",
        "paxgusdt": "PAXGUSDT",
        "btcusdt": "BTCUSDT",
        "suiusdt": "SUIUSDT",
    },
}


def dataset_id(symbol: str, exchange: str | None = None) -> str:
    ex = exchange or Exchange
    mapping = SYMBOL_DATASET_IDS.get(ex)
    if mapping is None:
        raise KeyError(f"No symbol map for exchange={ex!r}")
    key = symbol.lower()
    if key not in mapping:
        raise KeyError(f"No dataset id for {key!r} on {ex}")
    return mapping[key]


def dataset_symbols(exchange: str | None = None) -> list[str]:
    return [dataset_id(s, exchange) for s in Symbols]


StartDate = "2026-08-26"
# Tardis daily exports lag ~1 day; use 2 days ago to avoid "dataset not available yet" errors
EndDate = (date.today() - timedelta(days=2)).isoformat()

# API key (loaded from .env — never print this)
Tardis_API_Key = os.getenv("TARDIS_API_KEY", "").strip()

# Output paths (per-exchange so venues can coexist)
Output_Path = PROJECT_ROOT / "data"
RAW_DIR = Output_Path / "raw" / Exchange
PARQUET_DIR = Output_Path / "parquet" / Exchange
RESULTS_DIR = PROJECT_ROOT / "results"

# Legacy Binance layout (pre–per-exchange folders)
if Exchange == "binance-futures":
    _legacy_parquet = Output_Path / "parquet"
    _legacy_raw = Output_Path / "raw"
    if not (PARQUET_DIR / "xauusdt").exists() and (_legacy_parquet / "xauusdt").exists():
        PARQUET_DIR = _legacy_parquet
    if not any(RAW_DIR.glob("*.csv.gz")) and any(_legacy_raw.glob("binance-futures_*.csv.gz")):
        RAW_DIR = _legacy_raw


# Data types
data_types = ["trades", "book_snapshot_25"]

# Default fees used by quoter_sim (override per venue / your VIP tier)
FEE_BY_EXCHANGE = {
    "binance-futures": {
        "target_maker": 0.0002,
        "target_taker": 0.0005,
        "xau_maker": 0.0000,
        "xau_taker": 0.0004,
    },
    "gate-io-futures": {
        "target_maker": 0.00015,
        "target_taker": 0.0005,
        "xau_maker": 0.00015,
        "xau_taker": 0.0005,
    },
    # Bybit USDT perpetual non-VIP-ish defaults
    "bybit": {
        "target_maker": 0.0002,
        "target_taker": 0.00055,
        "xau_maker": 0.0002,
        "xau_taker": 0.00055,
    },
}

for _dir in (Output_Path, RAW_DIR, PARQUET_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
