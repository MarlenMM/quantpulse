from datetime import date, timedelta
from pathlib import Path
from typing import Any

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_json
from quantpulse.ingestion.http import get_json
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_BASE_URL = "https://finnhub.io/api/v1"

# Section 5: ~60 calls/min free tier. 1.1s keeps a comfortable margin.
_rate_limiter = SimpleRateLimiter(min_interval_seconds=1.1)


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "finnhub" / subdir


def _get(path: str, params: dict[str, Any]) -> Any:
    settings = get_settings()
    if not settings.finnhub_api_key:
        raise ValueError("FINNHUB_API_KEY is not set")
    _rate_limiter.wait()
    return get_json(f"{_BASE_URL}{path}", params={**params, "token": settings.finnhub_api_key})


def fetch_quote(symbol: str) -> dict[str, Any]:
    """Real-time quote: current/high/low/open/previous-close price."""

    def _fetch() -> dict[str, Any]:
        return dict(_get("/quote", {"symbol": symbol}))

    return cached_json(f"quote_{symbol}", _fetch, _cache_dir("quote"), ttl=timedelta(minutes=15))


def fetch_company_profile(symbol: str) -> dict[str, Any]:
    """Company profile: name, exchange, industry, market cap, shares outstanding."""

    def _fetch() -> dict[str, Any]:
        return dict(_get("/stock/profile2", {"symbol": symbol}))

    return cached_json(f"profile_{symbol}", _fetch, _cache_dir("profile"), ttl=timedelta(days=7))


def fetch_company_news(symbol: str, from_date: date, to_date: date) -> list[dict[str, Any]]:
    """Ticker-tagged news headlines between `from_date` and `to_date` (Tier 1, Section 7.3)."""

    def _fetch() -> list[dict[str, Any]]:
        result = _get(
            "/company-news",
            {"symbol": symbol, "from": from_date.isoformat(), "to": to_date.isoformat()},
        )
        return list(result)

    key = f"news_{symbol}_{from_date.isoformat()}_{to_date.isoformat()}"
    return list(cached_json(key, _fetch, _cache_dir("news"), ttl=timedelta(hours=6)))


def fetch_analyst_recommendation_trends(symbol: str) -> list[dict[str, Any]]:
    """Monthly analyst rating counts for the trailing several months."""

    def _fetch() -> list[dict[str, Any]]:
        return list(_get("/stock/recommendation", {"symbol": symbol}))

    return list(
        cached_json(
            f"recommendation_trends_{symbol}",
            _fetch,
            _cache_dir("recommendation_trends"),
            ttl=timedelta(days=1),
        )
    )


def fetch_earnings_calendar(from_date: date, to_date: date) -> list[dict[str, Any]]:
    """Scheduled earnings releases between `from_date` and `to_date`."""

    def _fetch() -> list[dict[str, Any]]:
        result = _get(
            "/calendar/earnings", {"from": from_date.isoformat(), "to": to_date.isoformat()}
        )
        return list(result.get("earningsCalendar", []))

    key = f"earnings_calendar_{from_date.isoformat()}_{to_date.isoformat()}"
    return list(cached_json(key, _fetch, _cache_dir("earnings_calendar"), ttl=timedelta(hours=12)))


def fetch_basic_financials(symbol: str) -> dict[str, Any]:
    """Fallback fundamentals when yfinance's `info` is sparse (Section 5)."""

    def _fetch() -> dict[str, Any]:
        result = _get("/stock/metric", {"symbol": symbol, "metric": "all"})
        return dict(result.get("metric", {}))

    return cached_json(
        f"basic_financials_{symbol}", _fetch, _cache_dir("basic_financials"), ttl=timedelta(days=7)
    )
