"""Static economic event calendar — FOMC/CPI/jobs-report schedule (Section 28).

Unlike every other module in `ingestion/`, this isn't a live API call: the
Fed publishes its FOMC meeting calendar, and BLS its CPI/jobs-report release
dates, months to a year ahead of time. Section 28 explicitly calls for "a
small static calendar (updated a few times a year)" rather than reaching for
a paid economic-calendar API just to avoid maintaining a short date list.

PLACEHOLDER DATA: `_FOMC_DECISION_DATES` and `_CPI_RELEASE_DATES` below are
illustrative seed values (a plausible ~6-8-week FOMC cadence and a mid-month
CPI cadence), *not* verified against the Fed/BLS calendars. Replace them with
the real published dates from federalreserve.gov/monetarypolicy/fomccalendars
.htm and bls.gov/schedule/news_release before relying on the "elevated
uncertainty ahead" signal (Section 28) this feeds -- shipping an unverified
date as if it were confirmed would be exactly the kind of false-confidence
Section 22 warns against.
"""

from dataclasses import dataclass
from datetime import date, timedelta

FOMC_RATE_DECISION = "FOMC Rate Decision"
CPI_RELEASE = "CPI Release"
JOBS_REPORT = "Jobs Report"


@dataclass(frozen=True)
class EconomicEvent:
    event_date: date
    event_name: str


# Decision day (the second day of each two-day meeting) -- PLACEHOLDER, see
# module docstring.
_FOMC_DECISION_DATES: dict[int, list[date]] = {
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
}

# BLS Consumer Price Index release dates -- PLACEHOLDER, see module docstring.
_CPI_RELEASE_DATES: dict[int, list[date]] = {
    2026: [
        date(2026, 1, 14),
        date(2026, 2, 12),
        date(2026, 3, 12),
        date(2026, 4, 14),
        date(2026, 5, 13),
        date(2026, 6, 10),
        date(2026, 7, 14),
        date(2026, 8, 12),
        date(2026, 9, 11),
        date(2026, 10, 14),
        date(2026, 11, 13),
        date(2026, 12, 10),
    ],
}


def _first_friday(year: int, month: int) -> date:
    """First Friday of `year`/`month`.

    The BLS Employment Situation ("jobs report") is released the first
    Friday of the month with rare BLS-announced exceptions this simple rule
    doesn't capture -- the same "small static calendar" caveat applies.
    """
    first_of_month = date(year, month, 1)
    days_until_friday = (4 - first_of_month.weekday()) % 7  # Monday=0 ... Friday=4
    return first_of_month + timedelta(days=days_until_friday)


def _jobs_report_dates(year: int) -> list[date]:
    return [_first_friday(year, month) for month in range(1, 13)]


def _calendar_for_year(year: int) -> list[EconomicEvent]:
    events = [EconomicEvent(d, FOMC_RATE_DECISION) for d in _FOMC_DECISION_DATES.get(year, [])]
    events += [EconomicEvent(d, CPI_RELEASE) for d in _CPI_RELEASE_DATES.get(year, [])]
    events += [EconomicEvent(d, JOBS_REPORT) for d in _jobs_report_dates(year)]
    return sorted(events, key=lambda e: e.event_date)


def upcoming_events(as_of: date, *, lookahead_days: int = 14) -> list[EconomicEvent]:
    """Scheduled macro releases in `[as_of, as_of + lookahead_days]`.

    The "elevated uncertainty ahead" flag's input (Section 28) -- knowing a
    market-moving release is imminent, not just its value after the fact.
    """
    horizon = as_of + timedelta(days=lookahead_days)
    years = sorted({as_of.year, horizon.year})
    events = [event for year in years for event in _calendar_for_year(year)]
    return [event for event in events if as_of <= event.event_date <= horizon]
