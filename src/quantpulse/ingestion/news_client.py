"""RSS feed ingestion — Tier 1 (company-specific) news (Section 7.3).

Yahoo Finance, Google News, and Seeking Alpha each publish a free, no-key RSS
feed keyed by ticker/query. This module only fetches and normalizes the raw
headlines; entity matching beyond "this is the feed we asked for `symbol`",
event-type classification, and sentiment scoring are
entity_extraction.py / event_classifier.py / sentiment.py's job
(Section 7.3 steps 1-3), not this ingestion layer's.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from time import struct_time

import feedparser
import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import get_text
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

logger = logging.getLogger(__name__)

_YAHOO_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
_GOOGLE_NEWS_URL = "https://news.google.com/rss/search"
_SEEKING_ALPHA_URL = "https://seekingalpha.com/api/sa/combined/{symbol}.xml"

# A descriptive UA, not a personal contact -- unlike EDGAR/Wikipedia, none of
# these three publish a UA policy requiring one, but some feed hosts reject
# the bare default `requests` UA outright.
_USER_AGENT = "quantpulse-news-ingestion/0.1 (contact via project README)"

_COLUMNS = ["title", "link", "summary", "published_at", "source", "symbol", "tier"]

# Section 5/19: none of the three publishes a per-minute limit -- the same
# conservative min-interval treatment as SEC EDGAR, kept per-source so a slow
# Seeking Alpha response doesn't throttle unrelated Yahoo/Google calls.
_yahoo_rate_limiter = SimpleRateLimiter(min_interval_seconds=1.0)
_google_rate_limiter = SimpleRateLimiter(min_interval_seconds=1.0)
_seeking_alpha_rate_limiter = SimpleRateLimiter(min_interval_seconds=1.0)


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "news_rss" / subdir


def _fetch_feed(
    url: str, params: dict[str, str] | None, *, source: str, rate_limiter: SimpleRateLimiter
) -> str:
    rate_limiter.wait()
    with get_breaker(f"news_rss_{source}").guard():
        return get_text(url, params=params, headers={"User-Agent": _USER_AGENT})


def _parsed_time_to_datetime(parsed: struct_time | None) -> datetime | None:
    if parsed is None:
        return None
    return datetime(*parsed[:6])


def _entries_to_frame(raw_text: str, *, source: str, symbol: str) -> pd.DataFrame:
    parsed = feedparser.parse(raw_text)
    rows = [
        {
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", ""),
            "summary": entry.get("summary", "").strip(),
            "published_at": _parsed_time_to_datetime(
                entry.get("published_parsed") or entry.get("updated_parsed")
            ),
            "source": source,
            "symbol": symbol,
            "tier": 1,
        }
        for entry in parsed.entries
    ]
    return pd.DataFrame(rows, columns=_COLUMNS)


def fetch_yahoo_finance_news(symbol: str) -> pd.DataFrame:
    """Yahoo Finance's per-ticker headline RSS feed."""

    def _fetch() -> pd.DataFrame:
        raw = _fetch_feed(
            _YAHOO_URL,
            {"s": symbol, "region": "US", "lang": "en-US"},
            source="yahoo",
            rate_limiter=_yahoo_rate_limiter,
        )
        return _entries_to_frame(raw, source="yahoo", symbol=symbol)

    return cached_dataframe(f"yahoo_{symbol}", _fetch, _cache_dir("yahoo"), ttl=timedelta(hours=1))


def fetch_google_news(symbol: str, company_name: str | None = None) -> pd.DataFrame:
    """Google News RSS search for `symbol`, optionally narrowed by `company_name`."""
    query = f"{company_name} stock" if company_name else f"{symbol} stock"

    def _fetch() -> pd.DataFrame:
        raw = _fetch_feed(
            _GOOGLE_NEWS_URL,
            {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            source="google_news",
            rate_limiter=_google_rate_limiter,
        )
        return _entries_to_frame(raw, source="google_news", symbol=symbol)

    key = f"google_{symbol}_{query}"
    return cached_dataframe(key, _fetch, _cache_dir("google_news"), ttl=timedelta(hours=1))


def fetch_seeking_alpha_news(symbol: str) -> pd.DataFrame:
    """Seeking Alpha's per-ticker combined RSS feed."""

    def _fetch() -> pd.DataFrame:
        raw = _fetch_feed(
            _SEEKING_ALPHA_URL.format(symbol=symbol.lower()),
            None,
            source="seeking_alpha",
            rate_limiter=_seeking_alpha_rate_limiter,
        )
        return _entries_to_frame(raw, source="seeking_alpha", symbol=symbol)

    return cached_dataframe(
        f"seeking_alpha_{symbol}", _fetch, _cache_dir("seeking_alpha"), ttl=timedelta(hours=1)
    )


def fetch_all_tier1_news(symbol: str, company_name: str | None = None) -> pd.DataFrame:
    """All three Tier-1 RSS sources for `symbol`, concatenated.

    One source failing (feed down, ticker not covered by that provider)
    doesn't take down the other two -- each fetch is isolated and logged
    rather than allowed to raise out of the whole batch.
    """
    fetchers = (
        ("yahoo", lambda: fetch_yahoo_finance_news(symbol)),
        ("google_news", lambda: fetch_google_news(symbol, company_name)),
        ("seeking_alpha", lambda: fetch_seeking_alpha_news(symbol)),
    )
    frames = []
    for source_name, fetch in fetchers:
        try:
            frames.append(fetch())
        except Exception:
            logger.warning("Tier-1 RSS source %s failed for %s", source_name, symbol, exc_info=True)
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.concat(frames, ignore_index=True)
