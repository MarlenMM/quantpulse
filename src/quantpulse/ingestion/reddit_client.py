"""Reddit ingestion via PRAW — Tier 1 social sentiment source (Sections 5 & 7.3).

Section 19: Reddit's API terms restrict what you can store/redistribute from
user posts, so this only pulls post titles and metadata (score, comment
count, subreddit, permalink) — never `selftext` or comment bodies. Downstream
consumers should persist aggregated sentiment scores derived from this, not
these raw rows, if the app is deployed publicly.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import praw

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe
from quantpulse.ingestion.circuit_breaker import get_breaker

_SOURCE = "reddit"
_DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")
_COLUMNS = [
    "post_id",
    "title",
    "created_at",
    "score",
    "num_comments",
    "subreddit",
    "permalink",
    "symbol",
    "tier",
]


def _cache_dir() -> Path:
    return Path(get_settings().ingestion_cache_dir) / "reddit"


def _client() -> praw.Reddit:
    settings = get_settings()
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        raise ValueError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET are not set")
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent or "quantpulse/0.1",
        read_only=True,
    )


def _normalize_submission(submission: Any, *, subreddit_name: str, symbol: str) -> dict[str, Any]:
    return {
        "post_id": submission.id,
        "title": submission.title,
        "created_at": datetime.fromtimestamp(submission.created_utc, tz=UTC).replace(tzinfo=None),
        "score": submission.score,
        "num_comments": submission.num_comments,
        "subreddit": subreddit_name,
        "permalink": f"https://reddit.com{submission.permalink}",
        "symbol": symbol,
        "tier": 1,
    }


def fetch_mentions(
    symbol: str,
    *,
    subreddits: tuple[str, ...] = _DEFAULT_SUBREDDITS,
    limit: int = 100,
) -> pd.DataFrame:
    """Recent posts mentioning `symbol` across `subreddits`.

    PRAW manages Reddit's own rate limit internally (it sleeps based on the
    response headers Reddit sends back), so no extra limiter wraps this --
    only the circuit breaker, so a sustained Reddit outage short-circuits the
    nightly job instead of hanging it (Section 6).
    """

    def _fetch() -> pd.DataFrame:
        reddit = _client()
        rows = []
        with get_breaker(_SOURCE).guard():
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                for submission in subreddit.search(symbol, sort="new", limit=limit):
                    rows.append(
                        _normalize_submission(
                            submission, subreddit_name=subreddit_name, symbol=symbol
                        )
                    )
        return pd.DataFrame(rows, columns=_COLUMNS)

    key = f"mentions_{symbol}_{'-'.join(subreddits)}_{limit}"
    return cached_dataframe(key, _fetch, _cache_dir(), ttl=timedelta(hours=2))
