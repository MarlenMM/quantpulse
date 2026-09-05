"""The refresh workflow's trigger is a methodology decision, so it is asserted.

Two facts about the schedule are load-bearing and neither is visible from the
Python side, where every other rule in this project is checked:

* **It has to exist.** Without it nothing updates the repo-committed demo
  database, and both public deployments age in place. That is not theoretical:
  the schedule was removed on 2026-08-27 and by 2026-09-06 the demo's own
  freshness strip read "24 days ago" for sentiment, on the third screenful of
  the landing page.
* **It has to run after the New York close.** Before it, the day's closing
  prices and option chain are not published, and the database once stored an
  implied volatility of 0.46% for Apple -- sub-penny values for 96% of names --
  from a run that went too early. A cron edited to "0 14" would keep every other
  gate in this repo green and quietly poison a column.

The comments in the workflow say both things already. A comment does not fail.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "refresh_data.yml"

#: 16:00 ET is 20:00 UTC in summer and 21:00 in winter, so anything from 21:00
#: onward clears the close year-round. Kept as a bound rather than an equality:
#: the requirement is "after the close", not one particular hour.
EARLIEST_SAFE_UTC_HOUR = 21


@pytest.fixture(scope="module")
def workflow() -> dict:
    # `on:` is the YAML 1.1 boolean `True` once parsed, which is a well-known
    # trap for workflow files and the reason this reads the key both ways rather
    # than assuming one.
    loaded = yaml.safe_load(WORKFLOW.read_text())
    return loaded


def _triggers(workflow: dict) -> dict:
    return workflow.get("on") or workflow[True]


def _cron_entries(workflow: dict) -> list[str]:
    schedule = _triggers(workflow).get("schedule") or []
    return [entry["cron"] for entry in schedule]


def test_the_refresh_is_scheduled(workflow: dict) -> None:
    assert _cron_entries(workflow), (
        "the data refresh has no `schedule:`, so nothing updates the committed demo "
        "database and both public deployments age in place -- which is exactly what "
        "happened between 2026-08-27 and 2026-09-06"
    )


def test_it_runs_after_the_new_york_close(workflow: dict) -> None:
    """A run before the close stores prices and an option chain that do not exist yet."""
    for entry in _cron_entries(workflow):
        minute, hour, *_ = entry.split()
        assert hour.isdigit(), f"cron {entry!r} does not name a single hour"
        assert int(hour) >= EARLIEST_SAFE_UTC_HOUR, (
            f"cron {entry!r} fires at {hour}:00 UTC, before the 16:00 ET close in at "
            f"least one half of the year. Running early is how the database ended up "
            f"with a 0.46% implied volatility for Apple."
        )
        assert minute.isdigit()


def test_it_runs_on_weekdays_only(workflow: dict) -> None:
    """A weekend run is a no-op the script declines; scheduling one just burns a runner."""
    for entry in _cron_entries(workflow):
        day_of_week = entry.split()[4]
        assert day_of_week == "1-5", (
            f"cron {entry!r} runs on {day_of_week}; the exchange is shut at the weekend "
            f"and `is_trading_day` turns those runs into a no-op"
        )


def test_monday_is_included_so_the_weekly_branch_comes_round(workflow: dict) -> None:
    """The slow-moving datasets have no other way of being refreshed.

    Fundamentals, analyst consensus, 13F, forecasts, the backtest, news and
    sentiment all key off `_WEEKLY_REFRESH_WEEKDAY`. Scheduling Tue-Fri only --
    a tempting way to avoid the long Monday run -- would leave every one of them
    to manual dispatch forever, which is the state the demo was already in.
    """
    for entry in _cron_entries(workflow):
        days = entry.split()[4]
        start, _, end = days.partition("-")
        assert int(start) <= 1 <= int(end or start), (
            f"cron {entry!r} skips Monday, so the weekly branch never runs unattended"
        )


def test_manual_dispatch_survives_alongside_the_schedule(workflow: dict) -> None:
    """The two answer different questions and the schedule must not displace dispatch.

    A timer keeps the demo current; dispatch is how a database gets caught up out
    of band, and how a fix to the weekly branch is tested without waiting a week
    for Monday.
    """
    triggers = _triggers(workflow)
    assert "workflow_dispatch" in triggers
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert {"force_weekly", "ignore_market_calendar"} <= set(inputs)
