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
    """Return `fetch()`, cached as Parquet under `cache_dir/{key}.parquet`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.parquet"
    if _is_fresh(path, ttl):
        return pd.read_parquet(path)
    df = fetch()
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
