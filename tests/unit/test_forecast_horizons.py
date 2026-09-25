"""Every published forecast horizon can be graded on the history the job reads.

Point 35. A hit rate needs `backtest.MIN_GRADED_WINDOWS` non-overlapping
out-of-sample windows. The 63- and 252-day horizons never reached it on the
~3.5-year read window (9 and 0 windows in the 2026-09-21 log), so every one of
their rows was published ungraded, carried the largest number on the page, and
cost about five minutes of each weekly run. This pins the published set to the
horizons the read window can actually grade, so one cannot come back unnoticed.
"""

import pandas as pd
import pytest

import refresh_data
from quantpulse.analysis import backtest, forecasting

TRADING_DAYS_PER_CALENDAR_DAY = 252 / 365


def windows_in_read_window(horizon: int) -> int:
    """The most non-overlapping walk-forward windows the weekly read window allows.

    An upper bound: the walk-forward starts at `_MIN_ACCURACY_TRAIN` bars and steps
    by the horizon, as `backtest.walk_forward_accuracy` does. GBR's longer training
    floor only lowers it.
    """
    bars = int(refresh_data._FORECAST_PRICE_LOOKBACK_DAYS * TRADING_DAYS_PER_CALENDAR_DAY)
    last_eval = bars - horizon - 1
    return len(range(backtest._MIN_ACCURACY_TRAIN, last_eval + 1, horizon))


@pytest.mark.parametrize("horizon", forecasting.DEFAULT_HORIZONS)
def test_every_published_horizon_can_reach_a_hit_rate(horizon: int) -> None:
    assert windows_in_read_window(horizon) >= backtest.MIN_GRADED_WINDOWS, (
        f"h={horizon} can be graded over at most {windows_in_read_window(horizon)} windows, "
        f"below the {backtest.MIN_GRADED_WINDOWS} a hit rate needs -- every row at it "
        "would be published ungraded (point 35)"
    )


def test_the_bound_matches_what_the_walk_forward_really_grades() -> None:
    """The count above is the real one, not a formula that merely looks right."""
    horizon = 20
    bars = int(refresh_data._FORECAST_PRICE_LOOKBACK_DAYS * TRADING_DAYS_PER_CALENDAR_DAY)
    dates = pd.bdate_range("2023-01-02", periods=bars)
    close = pd.Series([100.0 * 1.0005**i for i in range(bars)], index=dates)
    prices = pd.DataFrame({"close": close, "adj_close": close})
    result = backtest.walk_forward_accuracy(
        prices, model_fn=forecasting.baseline_forecast, horizon_days=horizon, model_name="baseline"
    )
    assert result is not None
    assert len(result.as_of) == windows_in_read_window(horizon)


def test_the_dropped_horizons_really_cannot_be_graded() -> None:
    """The reason given on the page, checked, so the sentence stays true."""
    assert windows_in_read_window(63) < backtest.MIN_GRADED_WINDOWS
    assert windows_in_read_window(252) < backtest.MIN_GRADED_WINDOWS
    assert "20 trading days" in forecasting.HORIZON_SCOPE_NOTE
    assert max(forecasting.DEFAULT_HORIZONS) == 20
