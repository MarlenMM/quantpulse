"""Alerts whose subject is the pipeline itself (finding 27).

Section 10's alerting is about the *data* -- rating changes, new formations on
held names. Nothing told anyone when the job that produces the data broke: the
2026-09-15 run published an empty database and the 2026-09-24 publish froze the
demo on the previous day, and both were findable only in the Actions tab.

Two notices, both through the same webhook and `discord.send`:

* **A failed run** -- which job, and which step in it, with the run's URL.
* **A published site that has stopped moving** -- judged from the live
  ``health.json``, in *trading sessions*, because a run can be green while the
  site stands still, and a failure notice cannot fire for a run that never ran.

With no webhook configured -- this repository today, and every fork -- each logs
one line and sends nothing. That is not an error (``Settings.alerting_configured``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from quantpulse.alerting import discord
from quantpulse.config import get_settings
from quantpulse.utils.market_calendar import is_trading_day, trading_days_between

__all__ = [
    "STALE_AFTER_SESSIONS",
    "deliver",
    "failed_jobs",
    "failure_message",
    "is_stale",
    "last_completed_session",
    "sessions_missing",
    "staleness_message",
]

logger = logging.getLogger(__name__)

_MARKET_TZ = ZoneInfo("America/New_York")
_CLOSE = time(16, 0)

#: Completed sessions the live site may be missing before it is called stale.
#:
#: Counted when the nightly starts, after the close, so a healthy site is
#: always missing exactly one -- tonight's, which the run is about to add. Two
#: therefore means one missed night, and that already has its own notice: the
#: failed run's. The same rule as the front ends' ``STALE_AFTER_DAYS`` --
#: "roughly twice the cadence" -- but in sessions, so Labor Day or Thanksgiving
#: is not an outage.
#:
#: Chosen against the record rather than guessed. Replaying every scheduled run
#: from 2026-07-27 to 2026-09-24, the site was one missed night behind five
#: times: two isolated (08-03, 09-24) and three the first night of a real outage
#: (the longest ran thirteen sessions and went unnoticed for over two weeks).
#: Alerting past two fires on the second night of all three outages and on
#: neither isolated miss.
STALE_AFTER_SESSIONS = 2


def last_completed_session(now: datetime) -> date:
    """The newest trading session whose close has passed at ``now``.

    Judged in New York time, not the runner's: the nightly starts around
    midnight UTC, which is still the same trading day in New York.
    """
    local = now.astimezone(_MARKET_TZ)
    day = local.date()
    if is_trading_day(day) and local.time() >= _CLOSE:
        return day
    start = day - timedelta(days=10)
    return trading_days_between(start, day - timedelta(days=1))[-1]


def sessions_missing(newest: date, now: datetime) -> int:
    """How many completed trading sessions are newer than ``newest``."""
    last = last_completed_session(now)
    if newest >= last:
        return 0
    return len(trading_days_between(newest + timedelta(days=1), last))


def is_stale(newest: date, now: datetime) -> bool:
    return sessions_missing(newest, now) > STALE_AFTER_SESSIONS


def staleness_message(newest: date, missing: int, *, site_url: str) -> str:
    return (
        f"QuantPulse: the public demo is stale. Its newest prices are from {newest.isoformat()}, "
        f"{missing} trading sessions behind (more than {STALE_AFTER_SESSIONS} is not an ordinary "
        f"missed night). The nightly refresh or the Pages publish has stopped updating it.\n"
        f"{site_url}"
    )


def failed_jobs(jobs: Iterable[Mapping[str, Any]]) -> list[str]:
    """Each failed job, with its first failed step when there is one.

    ``jobs`` is the ``jobs`` array of ``gh run view --json jobs``. A job that
    timed out or never started has no failed step, and is named alone.
    """
    described = []
    for job in jobs:
        if job.get("conclusion") != "failure":
            continue
        step = next(
            (s.get("name") for s in job.get("steps") or [] if s.get("conclusion") == "failure"),
            None,
        )
        name = str(job.get("name", "unnamed job"))
        described.append(f"{name} — step “{step}”" if step else name)
    return described


def failure_message(workflow: str, run_url: str, failures: list[str]) -> str:
    lines = [f"QuantPulse: {workflow} failed."]
    if failures:
        lines += [f"• {failure}" for failure in failures]
    else:
        lines.append("No failed job was reported; the run itself is marked failed.")
    lines.append(run_url)
    return "\n".join(lines)


def deliver(text: str) -> int:
    """Post ``text`` to the configured webhook; with none, log one line and send nothing.

    Returns the number of messages sent. ``discord.WebhookError`` propagates --
    a configured webhook that refuses is a fault worth a red job, because
    otherwise every later notice would vanish without a trace. Its message is
    already scrubbed of the URL.
    """
    settings = get_settings()
    if not settings.alerting_configured():
        logger.info(
            "Alerting is not configured (no webhook URL); not sending: %s", text.splitlines()[0]
        )
        return 0
    return discord.send(str(settings.alert_discord_webhook_url), text)
