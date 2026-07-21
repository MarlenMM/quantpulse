from datetime import date

from quantpulse.utils.market_calendar import (
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)


def test_is_trading_day_true_for_a_weekday() -> None:
    assert is_trading_day(date(2026, 7, 21)) is True  # Tuesday


def test_is_trading_day_false_for_a_weekend() -> None:
    assert is_trading_day(date(2026, 7, 18)) is False  # Saturday


def test_is_trading_day_false_for_a_market_holiday() -> None:
    assert is_trading_day(date(2026, 1, 1)) is False  # New Year's Day


def test_trading_days_between_excludes_the_weekend() -> None:
    days = trading_days_between(date(2026, 7, 17), date(2026, 7, 20))  # Fri..Mon
    assert days == [date(2026, 7, 17), date(2026, 7, 20)]


def test_previous_trading_day_skips_the_weekend() -> None:
    assert previous_trading_day(date(2026, 7, 20)) == date(2026, 7, 17)  # Mon -> Fri


def test_next_trading_day_skips_the_weekend() -> None:
    assert next_trading_day(date(2026, 7, 17)) == date(2026, 7, 20)  # Fri -> Mon
