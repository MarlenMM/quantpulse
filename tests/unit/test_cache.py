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
