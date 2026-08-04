import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from quantpulse.ingestion.cache import cached_dataframe, cached_json


def test_cached_dataframe_writes_and_reuses_cache(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    fetch = Mock(return_value=df)

    result1 = cached_dataframe("key", fetch, tmp_path)
    result2 = cached_dataframe("key", fetch, tmp_path)

    pd.testing.assert_frame_equal(result1, df)
    pd.testing.assert_frame_equal(result2, df)
    fetch.assert_called_once()


def test_cached_dataframe_refetches_after_ttl_expiry(tmp_path: Path) -> None:
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    fetch = Mock(side_effect=[df1, df2])

    cached_dataframe("key", fetch, tmp_path, ttl=timedelta(seconds=60))
    stale_time = time.time() - 120
    os.utime(tmp_path / "key.parquet", (stale_time, stale_time))

    result = cached_dataframe("key", fetch, tmp_path, ttl=timedelta(seconds=60))

    pd.testing.assert_frame_equal(result, df2)
    assert fetch.call_count == 2


def test_cached_json_writes_and_reuses_cache(tmp_path: Path) -> None:
    data = {"a": 1, "b": [1, 2, 3]}
    fetch = Mock(return_value=data)

    result1 = cached_json("key", fetch, tmp_path)
    result2 = cached_json("key", fetch, tmp_path)

    assert result1 == data
    assert result2 == data
    fetch.assert_called_once()


def test_cached_json_refetches_after_ttl_expiry(tmp_path: Path) -> None:
    fetch = Mock(side_effect=[{"v": 1}, {"v": 2}])

    cached_json("key", fetch, tmp_path, ttl=timedelta(seconds=60))
    stale_time = time.time() - 120
    os.utime(tmp_path / "key.json", (stale_time, stale_time))

    result = cached_json("key", fetch, tmp_path, ttl=timedelta(seconds=60))

    assert result == {"v": 2}
    assert fetch.call_count == 2


class TestEmptyResultsAreNotCached:
    """A throttled fetch must not become a lasting fact about the symbol.

    The first full cold-start backfill cached a zero-row frame for AAPL; every
    later call read it back for the 12-hour TTL without touching the network,
    so 625 of 1,067 symbols -- Apple, Amazon and Adobe among them -- silently
    finished with no price history at all.
    """

    def test_an_empty_frame_is_never_written_to_disk(self, tmp_path: Path) -> None:
        calls = []

        def fetch() -> pd.DataFrame:
            calls.append(1)
            return pd.DataFrame()

        first = cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12))
        assert first.empty
        assert not (tmp_path / "sym.parquet").exists(), "an empty result must not be cached"

        # ...so the next call actually retries rather than replaying the failure.
        cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12))
        assert len(calls) == 2

    def test_a_recovered_fetch_replaces_the_empty_result(self, tmp_path: Path) -> None:
        results = [pd.DataFrame(), pd.DataFrame({"close": [1.0, 2.0]})]

        def fetch() -> pd.DataFrame:
            return results.pop(0)

        assert cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12)).empty
        recovered = cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12))
        assert len(recovered) == 2
        assert (tmp_path / "sym.parquet").exists()

    def test_a_pre_existing_empty_cache_file_is_ignored_not_trusted(self, tmp_path: Path) -> None:
        # Files written before this rule existed must not keep poisoning reads.
        pd.DataFrame().to_parquet(tmp_path / "sym.parquet")

        result = cached_dataframe(
            "sym", lambda: pd.DataFrame({"close": [3.0]}), tmp_path, ttl=timedelta(hours=12)
        )
        assert len(result) == 1

    def test_a_non_empty_result_still_caches_and_is_reused(self, tmp_path: Path) -> None:
        calls = []

        def fetch() -> pd.DataFrame:
            calls.append(1)
            return pd.DataFrame({"close": [1.0]})

        cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12))
        cached_dataframe("sym", fetch, tmp_path, ttl=timedelta(hours=12))
        assert len(calls) == 1, "the normal caching path must be unaffected"
