from datetime import timedelta
from pathlib import Path

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia rejects the default urllib/requests user agent (403); it doesn't
# require a personal contact, just a descriptive, non-default one.
_USER_AGENT = "quantpulse-data-ingestion/0.1 (contact via project README)"


def _fetch_raw() -> pd.DataFrame:
    tables = pd.read_html(_SP500_URL, storage_options={"User-Agent": _USER_AGENT})
    return tables[0]


def fetch_sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 constituents with GICS sector/sub-industry.

    Columns are normalized to match the `tickers` table (Section 13). This is
    today's membership only — Section 5's survivorship-bias note applies:
    point-in-time historical membership is reconstructed separately, during
    the cold-start backfill.
    """
    cache_dir = Path(get_settings().ingestion_cache_dir) / "wikipedia"
    raw = cached_dataframe("sp500_constituents", _fetch_raw, cache_dir, ttl=timedelta(days=1))

    df = raw.rename(
        columns={
            "Symbol": "symbol",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )[["symbol", "name", "sector", "industry"]].copy()

    # Data providers (yfinance, Finnhub) use '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B).
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df["exchange"] = None
    df["asset_type"] = "equity"
    df["is_active"] = True
    return df.reset_index(drop=True)


def fetch_sp500_date_added() -> pd.DataFrame:
    """Current constituents with the date each was added, as [symbol, added_date].

    Used only by the seed script's fallback path (Section 5): when no
    point-in-time historical dataset is available, this at least gives real
    add-dates for today's survivors. Rows whose date can't be parsed are
    dropped rather than guessed.
    """
    cache_dir = Path(get_settings().ingestion_cache_dir) / "wikipedia"
    raw = cached_dataframe("sp500_date_added", _fetch_raw, cache_dir, ttl=timedelta(days=1))

    df = raw.rename(columns={"Symbol": "symbol", "Date added": "added_date"})[
        ["symbol", "added_date"]
    ].copy()
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df["added_date"] = pd.to_datetime(df["added_date"], errors="coerce").dt.date
    return df.dropna(subset=["added_date"]).reset_index(drop=True)
