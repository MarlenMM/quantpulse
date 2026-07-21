from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_json
from quantpulse.ingestion.rate_limit import TokenBucketRateLimiter

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_SOURCE = "fred"

# Section 5: ~120 req/min free tier. A token bucket (capacity 100, refilled
# over 60s) allows bursting through a batch of series then throttling.
_rate_limiter = TokenBucketRateLimiter(capacity=100, per_seconds=60.0)

# Named series used elsewhere in the plan (Section 5, 28); fetch_series()
# works with any other FRED series id too.
FED_FUNDS_RATE = "FEDFUNDS"
CPI = "CPIAUCSL"
UNEMPLOYMENT_RATE = "UNRATE"
GDP = "GDP"
TREASURY_YIELD_10Y = "DGS10"
TREASURY_YIELD_2Y = "DGS2"


def _cache_dir() -> Path:
    return Path(get_settings().ingestion_cache_dir) / "fred"


def fetch_series(
    series_id: str, start_date: date | None = None, end_date: date | None = None
) -> pd.DataFrame:
    """Raw observations for a FRED series, normalized to (date, indicator_name, value).

    No derived math (e.g. the 10Y-2Y spread) happens here -- that's an
    analysis-layer concern (Section 28), not ingestion.
    """
    settings = get_settings()
    if not settings.fred_api_key:
        raise ValueError("FRED_API_KEY is not set")

    def _fetch() -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": settings.fred_api_key,
            "file_type": "json",
        }
        if start_date is not None:
            params["observation_start"] = start_date.isoformat()
        if end_date is not None:
            params["observation_end"] = end_date.isoformat()

        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            result = get_json(_BASE_URL, params=params)
        df = pd.DataFrame(result["observations"])
        # FRED marks missing observations with "."; drop them rather than
        # silently coercing to NaN-as-zero downstream.
        df = df[df["value"] != "."].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = df["value"].astype(float)
        df["indicator_name"] = series_id
        return df[["date", "indicator_name", "value"]].reset_index(drop=True)

    key = f"{series_id}_{start_date or 'all'}_{end_date or 'all'}"
    return cached_dataframe(key, _fetch, _cache_dir(), ttl=timedelta(days=7))


def fetch_fed_funds_rate(
    start_date: date | None = None, end_date: date | None = None
) -> pd.DataFrame:
    return fetch_series(FED_FUNDS_RATE, start_date, end_date)


def fetch_cpi(start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    return fetch_series(CPI, start_date, end_date)


def fetch_unemployment_rate(
    start_date: date | None = None, end_date: date | None = None
) -> pd.DataFrame:
    return fetch_series(UNEMPLOYMENT_RATE, start_date, end_date)


def fetch_gdp(start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    return fetch_series(GDP, start_date, end_date)


def fetch_treasury_yield_10y(
    start_date: date | None = None, end_date: date | None = None
) -> pd.DataFrame:
    return fetch_series(TREASURY_YIELD_10Y, start_date, end_date)


def fetch_treasury_yield_2y(
    start_date: date | None = None, end_date: date | None = None
) -> pd.DataFrame:
    return fetch_series(TREASURY_YIELD_2Y, start_date, end_date)
