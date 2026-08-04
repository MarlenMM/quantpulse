import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def _is_fresh(path: Path, ttl: timedelta | None) -> bool:
    if not path.exists():
        return False
    if ttl is None:
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < ttl


def cached_dataframe(
    key: str,
    fetch: Callable[[], pd.DataFrame],
    cache_dir: Path,
    ttl: timedelta | None = None,
) -> pd.DataFrame:
    """Return `fetch()`, cached as Parquet under `cache_dir/{key}.parquet`.

    **An empty result is never cached.** Emptiness from these sources is almost
    always transient -- a throttled request, a 404 during a listing change --
    not a durable fact about the symbol, and persisting it converts a momentary
    failure into a lasting one for the whole TTL. That is not hypothetical: the
    first full cold-start backfill cached an empty frame for `AAPL`, and every
    later call read that back for 12 hours without touching the network, so 625
    of 1,067 symbols -- including Apple, Amazon and Adobe -- silently ended up
    with no price history at all.

    Re-fetching a genuinely dataless symbol costs one request; caching a
    throttled one costs that symbol for the rest of the TTL. The asymmetry is
    the whole argument.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.parquet"
    if _is_fresh(path, ttl):
        cached = pd.read_parquet(path)
        # Tolerate empties already on disk from before this rule existed.
        if not cached.empty:
            return cached
    df = fetch()
    if not df.empty:
        df.to_parquet(path)
    return df


def cached_json(
    key: str,
    fetch: Callable[[], Any],
    cache_dir: Path,
    ttl: timedelta | None = None,
) -> Any:
    """Return `fetch()`, cached as JSON under `cache_dir/{key}.json`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if _is_fresh(path, ttl):
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data
