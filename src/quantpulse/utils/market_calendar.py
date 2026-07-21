from datetime import date, timedelta
from functools import lru_cache

import pandas_market_calendars as mcal
from pandas_market_calendars.market_calendar import MarketCalendar


@lru_cache
def _nyse() -> MarketCalendar:
    return mcal.get_calendar("NYSE")


def is_trading_day(day: date) -> bool:
    return len(_nyse().valid_days(start_date=day, end_date=day)) > 0


def trading_days_between(start: date, end: date) -> list[date]:
    return [d.date() for d in _nyse().valid_days(start_date=start, end_date=end)]


def previous_trading_day(day: date, lookback_days: int = 10) -> date:
    """The most recent trading day strictly before `day`."""
    start = day - timedelta(days=lookback_days)
    valid_days = _nyse().valid_days(start_date=start, end_date=day - timedelta(days=1))
    if len(valid_days) == 0:
        raise ValueError(f"No trading day found in the {lookback_days} days before {day}")
    return valid_days[-1].date()


def next_trading_day(day: date, lookahead_days: int = 10) -> date:
    """The soonest trading day strictly after `day`."""
    end = day + timedelta(days=lookahead_days)
    valid_days = _nyse().valid_days(start_date=day + timedelta(days=1), end_date=end)
    if len(valid_days) == 0:
        raise ValueError(f"No trading day found in the {lookahead_days} days after {day}")
    return valid_days[0].date()
