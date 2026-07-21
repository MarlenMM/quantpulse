from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe, cached_json
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_json
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SOURCE = "edgar"

# Section 5: "free, no key, generous fair-use rate". No documented number,
# but SEC's own guidance is to stay well under ~10 req/sec -- min-interval,
# not a burst-allowing token bucket, is the polite fit for a fair-use source.
_rate_limiter = SimpleRateLimiter(min_interval_seconds=0.2)


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "edgar" / subdir


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_edgar_user_agent}


def fetch_cik_lookup() -> pd.DataFrame:
    """Ticker -> CIK mapping (SEC's own file, not company-facts specific)."""

    def _fetch() -> pd.DataFrame:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            result = get_json(_TICKERS_URL, headers=_headers())
        df = pd.DataFrame(result.values())
        return df.rename(columns={"cik_str": "cik", "title": "name"})[["ticker", "cik", "name"]]

    return cached_dataframe("cik_lookup", _fetch, _cache_dir("cik_lookup"), ttl=timedelta(days=30))


def get_cik_for_ticker(symbol: str) -> str:
    """10-digit, zero-padded CIK for `symbol`, as required by the company-facts URL."""
    lookup = fetch_cik_lookup()
    matches = lookup[lookup["ticker"].str.upper() == symbol.upper()]
    if matches.empty:
        raise ValueError(f"No CIK found for ticker {symbol!r}")
    return f"{int(matches.iloc[0]['cik']):010d}"


def fetch_company_facts(symbol: str) -> dict[str, Any]:
    """Raw XBRL company-facts payload (all reported financial-statement concepts) for `symbol`."""
    cik = get_cik_for_ticker(symbol)

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            return dict(get_json(_COMPANY_FACTS_URL.format(cik=cik), headers=_headers()))

    return dict(
        cached_json(
            f"company_facts_{symbol}", _fetch, _cache_dir("company_facts"), ttl=timedelta(days=7)
        )
    )
