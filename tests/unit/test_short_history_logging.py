"""A short price history is handled, so it should not be announced 429 times a run.

The 2026-10-05 weekly log carried 429 lines of `pandas_ta_classic.utils._core:
[X] Series has 121 rows but indicator requires at least 200. Returning None.`
-- an SMA-200 asked of the forecasting walk-forward's growing windows (20 names,
121 to 196 rows) and of a few newly listed tickers. Both callers already turn
that `None` into NaN on purpose (`technical._assign_indicator`,
`forecasting.build_features`), so the line reports a designed outcome as a
warning, and it was the second-loudest thing in the log. That logger emits
nothing else.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import forecasting, technical


def _prices(rows: int) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, rows), index=index)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=index,
    )


def test_a_short_history_is_quiet_and_still_nan(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        indicators = technical.compute_indicators(_prices(121))
        features, _ = forecasting.build_features(_prices(121))

    noisy = [r for r in caplog.records if r.name.startswith("pandas_ta")]
    assert noisy == [], [r.getMessage() for r in noisy]
    # Behaviour unchanged: what cannot be computed is NaN, not an error.
    assert indicators["sma_200"].isna().all()
    assert features["dist_sma_200"].isna().all()
    assert indicators["sma_50"].notna().any()


def test_only_that_logger_is_quietened() -> None:
    assert logging.getLogger("pandas_ta_classic.utils._core").getEffectiveLevel() > logging.WARNING
    assert logging.getLogger("pandas_ta_classic").getEffectiveLevel() <= logging.WARNING
