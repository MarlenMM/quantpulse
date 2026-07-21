from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from quantpulse.ingestion import economic_calendar


def test_first_friday_is_always_a_friday_in_the_first_week() -> None:
    for year in (2026, 2027):
        for month in range(1, 13):
            friday = economic_calendar._first_friday(year, month)
            assert friday.weekday() == 4  # Friday
            assert friday.day <= 7
            assert friday.month == month
            assert friday.year == year


def test_jobs_report_dates_returns_twelve_fridays() -> None:
    dates = economic_calendar._jobs_report_dates(2026)
    assert len(dates) == 12
    assert all(d.weekday() == 4 for d in dates)
    assert [d.month for d in dates] == list(range(1, 13))


def test_upcoming_events_includes_all_three_event_types_within_window() -> None:
    events = economic_calendar.upcoming_events(date(2026, 1, 1), lookahead_days=45)
    names = {e.event_name for e in events}
    assert names == {
        economic_calendar.FOMC_RATE_DECISION,
        economic_calendar.CPI_RELEASE,
        economic_calendar.JOBS_REPORT,
    }


def test_upcoming_events_excludes_events_outside_the_window() -> None:
    events = economic_calendar.upcoming_events(date(2026, 1, 1), lookahead_days=5)
    for event in events:
        assert date(2026, 1, 1) <= event.event_date <= date(2026, 1, 6)


def test_upcoming_events_is_sorted_by_date() -> None:
    events = economic_calendar.upcoming_events(date(2026, 1, 1), lookahead_days=90)
    dates = [e.event_date for e in events]
    assert dates == sorted(dates)


def test_upcoming_events_handles_year_boundary_gracefully() -> None:
    # Dec 31 + a 14-day lookahead crosses into a year with no seeded
    # FOMC/CPI dates -- should still return the rule-based jobs report entry
    # rather than raising or silently dropping the whole window.
    events = economic_calendar.upcoming_events(date(2026, 12, 31), lookahead_days=14)
    assert any(e.event_name == economic_calendar.JOBS_REPORT for e in events)
    for event in events:
        assert date(2026, 12, 31) <= event.event_date <= date(2027, 1, 14)


def test_economic_event_is_immutable() -> None:
    event = economic_calendar.EconomicEvent(date(2026, 1, 1), "Test Event")
    with pytest.raises(FrozenInstanceError):
        event.event_name = "Changed"  # type: ignore[misc]
