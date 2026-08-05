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

# Newest-N headlines kept per symbol by `fetch_all_tier1_news`. The three feeds
# return ~150 between them (Google News ~100, Seeking Alpha ~30, Yahoo ~20),
# which across a 500-name universe is ~75,000 articles for the local NLI and
# sentiment models to chew through in one batch. 40 keeps roughly a fortnight of
# real coverage for a widely-covered mega-cap and essentially everything for the
# rest, at about a quarter of the model cost. See `fetch_all_tier1_news`.
MAX_ARTICLES_PER_SYMBOL = 40

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


def fetch_all_tier1_news(
    symbol: str,
    company_name: str | None = None,
    *,
    max_articles: int | None = MAX_ARTICLES_PER_SYMBOL,
) -> pd.DataFrame:
    """All three Tier-1 RSS sources for `symbol`, newest first, capped.

    One source failing (feed down, ticker not covered by that provider)
    doesn't take down the other two -- each fetch is isolated and logged
    rather than allowed to raise out of the whole batch.

    **The cap is not a nicety.** The three feeds together return about 150
    headlines per ticker, so an unbounded sweep of a 500-name universe hands
    the downstream models ~75,000 articles in a single batch. Every one of
    those is eight NLI forward passes through `bart-large-mnli` plus a FinBERT
    pass, which on a 2-vCPU CI runner is many hours of work -- the nightly's
    weekly branch ran 5h38m without emitting a line and was killed at GitHub's
    6-hour job limit, so *nothing* from that run was ever committed. Sorted
    newest-first before truncating, because the recency-decay step downstream
    weights a three-week-old headline to near nothing anyway: the articles
    dropped here are the ones that would have contributed least.

    Pass `max_articles=None` for the genuinely unbounded feed (a backfill or an
    ad-hoc query), never the nightly.
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
    combined = pd.concat(frames, ignore_index=True)
    if max_articles is None or len(combined) <= max_articles:
        return combined
    # `na_position="last"` keeps undated entries as the first to be dropped --
    # an article with no timestamp can't be recency-weighted meaningfully.
    ordered = combined.sort_values("published_at", ascending=False, na_position="last")
    return ordered.head(max_articles).reset_index(drop=True)
