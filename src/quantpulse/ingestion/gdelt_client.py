"""GDELT DOC 2.0 API ingestion — Tier 2 (industry/thematic) and Tier 3
(macro/market-wide) news (Section 7.3).

Free, no API key. Two query shapes matter here: a matching article list
(Tier 2 — "what coverage exists right now for this theme/keyword search?")
and a daily tone/volume timeline (Tier 3 — "how is broad sentiment on this
theme trending?", feeding the Market Regime Index, Sections 5 & 28). No theme
classification or entity matching happens here — event_classifier.py and
entity_extraction.py own that; this module only fetches and normalizes.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_json
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_SOURCE = "gdelt"
_MAX_RECORDS = 250  # GDELT DOC API's own per-request ceiling

_ARTICLE_COLUMNS = [
    "title",
    "url",
    "domain",
    "published_at",
    "source_country",
    "language",
    "query",
]
_TONE_COLUMNS = ["date", "tone", "query"]

# Section 5: free, no key, "generous fair-use" with no published per-minute
# number -- the same conservative min-interval treatment as SEC EDGAR (Section 19).
_rate_limiter = SimpleRateLimiter(min_interval_seconds=1.0)

# GDELT's article `seendate` looks like "20260722T120000Z"; timeline `date`
# points have been observed both with and without that trailing "Z" -- try
# both rather than assuming one.
_DATETIME_FORMATS = ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S")


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "gdelt" / subdir


def _slug(query: str) -> str:
    """Collapse `query` into a filesystem-safe cache-key fragment."""
    return "".join(c if c.isalnum() else "_" for c in query.lower())[:80]


def _get(params: dict[str, Any]) -> dict[str, Any]:
    _rate_limiter.wait()
    with get_breaker(_SOURCE).guard():
        result = get_json(_BASE_URL, params={**params, "format": "json"}, timeout=30.0)
    return result if isinstance(result, dict) else {}


def _parse_gdelt_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def fetch_articles(
    query: str, *, timespan: str = "1d", max_records: int = _MAX_RECORDS
) -> pd.DataFrame:
    """Articles matching `query` within the trailing `timespan` (e.g. "1d", "7d", "3m").

    `query` uses GDELT's own search syntax — plain keywords
    ("semiconductor export controls") or a theme operator
    (`theme:WB_678_ARTIFICIAL_INTELLIGENCE`); either way this is Tier-2
    industry/thematic coverage, not tied to any single ticker.
    """

    def _fetch() -> pd.DataFrame:
        result = _get(
            {
                "query": query,
                "mode": "artlist",
                "maxrecords": str(min(max_records, _MAX_RECORDS)),
                "sort": "DateDesc",
                "timespan": timespan,
            }
        )
        rows = [
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "domain": article.get("domain", ""),
                "published_at": _parse_gdelt_datetime(article.get("seendate")),
                "source_country": article.get("sourcecountry"),
                "language": article.get("language"),
                "query": query,
            }
            for article in result.get("articles", [])
        ]
        return pd.DataFrame(rows, columns=_ARTICLE_COLUMNS)

    key = f"articles_{_slug(query)}_{timespan}"
    return cached_dataframe(key, _fetch, _cache_dir("articles"), ttl=timedelta(hours=1))


def fetch_tone_timeline(query: str, *, timespan: str = "7d") -> pd.DataFrame:
    """Daily average tone for `query` over `timespan` — Tier-3 macro tone tracking.

    Feeds the Market Regime Index's macro-news-tone input (Sections 5 & 28);
    this is a market-wide dampening signal, not a per-article sentiment score.
    """

    def _fetch() -> pd.DataFrame:
        result = _get({"query": query, "mode": "timelinetone", "timespan": timespan})
        series = result.get("timeline", [])
        points = series[0].get("data", []) if series else []
        rows = [
            {"date": parsed.date(), "tone": point.get("value"), "query": query}
            for point in points
            if (parsed := _parse_gdelt_datetime(point.get("date"))) is not None
        ]
        return pd.DataFrame(rows, columns=_TONE_COLUMNS)

    key = f"tone_{_slug(query)}_{timespan}"
    return cached_dataframe(key, _fetch, _cache_dir("tone_timeline"), ttl=timedelta(hours=6))
