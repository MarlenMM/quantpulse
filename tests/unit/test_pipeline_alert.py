"""Telling someone when the pipeline breaks (finding 27).

On 2026-09-15 the nightly went red, published an empty database and left the
demo frozen; on 2026-09-24 a publish gate failed and the demo stayed on the
previous day's data. Both were findable only by opening the Actions tab. These
tests cover the two notices added for that -- a failed run, and a published site
that has stopped moving -- without the network: the rule, the message, the path
taken when no webhook is configured (the state of this repository and of every
fork), and the workflow wiring that makes either one run at all.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

import pipeline_alert
from quantpulse.alerting import pipeline
from quantpulse.config import get_settings

NY = ZoneInfo("America/New_York")
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
SECRET = "https://discord.com/api/webhooks/1234567890/sEcReT-ToKeN"


def evening(day: date) -> datetime:
    """When the nightly actually starts: after the close, New York time."""
    return datetime(day.year, day.month, day.day, 20, 15, tzinfo=NY)


# --------------------------------------------------------------------------- rule


class TestSessionsMissing:
    def test_a_healthy_site_is_missing_only_tonights_session(self) -> None:
        # The refresh that is about to run is the one that adds today.
        assert pipeline.sessions_missing(date(2026, 9, 21), evening(date(2026, 9, 22))) == 1

    def test_before_the_close_today_does_not_count(self) -> None:
        during = datetime(2026, 9, 22, 11, 0, tzinfo=NY)
        assert pipeline.sessions_missing(date(2026, 9, 21), during) == 0

    def test_a_weekend_adds_nothing(self) -> None:
        saturday = datetime(2026, 9, 26, 12, 0, tzinfo=NY)
        assert pipeline.sessions_missing(date(2026, 9, 25), saturday) == 0

    def test_labor_day_is_not_a_missed_session(self) -> None:
        # Friday 4 Sep's data, checked on Wednesday 9 Sep. Monday was Labor Day,
        # so two sessions are missing (Tue, Wed) -- not three, which is what
        # counting weekdays says, and three is the alarm.
        friday = date(2026, 9, 4)
        assert pipeline.sessions_missing(friday, evening(date(2026, 9, 8))) == 1
        assert pipeline.sessions_missing(friday, evening(date(2026, 9, 9))) == 2
        assert not pipeline.is_stale(friday, evening(date(2026, 9, 9)))
        assert pipeline.is_stale(friday, evening(date(2026, 9, 10)))

    def test_thanksgiving_week_raises_no_false_alarm(self) -> None:
        # Wednesday's data checked the following Monday: Thursday was closed,
        # Friday was a (half) session, Monday is tonight's.
        wednesday = date(2026, 11, 25)
        assert pipeline.sessions_missing(wednesday, evening(date(2026, 11, 30))) == 2
        assert not pipeline.is_stale(wednesday, evening(date(2026, 11, 30)))

    def test_the_august_outage_is_caught_on_its_second_night(self) -> None:
        # Measured, not invented: the site stood on 2026-08-07 data while the
        # nightly failed thirteen sessions running, and nobody noticed for two
        # weeks. Tolerating one missed night and alerting on the second fires on
        # 08-12 -- and on the second night of each of the other two outages in
        # the record -- while the isolated misses (08-03, 09-24) stay quiet.
        newest = date(2026, 8, 7)
        assert not pipeline.is_stale(newest, evening(date(2026, 8, 11)))
        assert pipeline.is_stale(newest, evening(date(2026, 8, 12)))

    def test_the_tolerance_is_twice_the_cadence(self) -> None:
        # The same rule as the front ends' STALE_AFTER_DAYS ("roughly twice the
        # cadence"), counted in sessions so a long weekend is not an outage.
        assert pipeline.STALE_AFTER_SESSIONS == 2


class TestStalenessMessage:
    def test_it_names_the_date_the_count_and_where_to_look(self) -> None:
        text = pipeline.staleness_message(
            date(2026, 8, 7), 3, site_url="https://marlenmm.github.io/quantpulse/"
        )
        assert "2026-08-07" in text
        assert "3 trading sessions" in text
        assert "marlenmm.github.io/quantpulse" in text


# ------------------------------------------------------------------ failed runs

#: The shape `gh run view --json jobs` returned for run 35937083808, trimmed.
JOBS_2026_09_24 = [
    {
        "name": "refresh",
        "conclusion": "success",
        "steps": [{"name": "Run refresh", "conclusion": "success"}],
    },
    {
        "name": "publish / build",
        "conclusion": "failure",
        "steps": [
            {"name": "Build the client", "conclusion": "success"},
            {"name": "Check the built site actually serves its data", "conclusion": "failure"},
            {"name": "Run actions/configure-pages@v6", "conclusion": "skipped"},
        ],
    },
    {"name": "publish / deploy", "conclusion": "skipped", "steps": []},
    {"name": "notify", "conclusion": "", "steps": []},
]


class TestFailureSummary:
    def test_it_names_the_failed_job_and_its_first_failed_step(self) -> None:
        assert pipeline.failed_jobs(JOBS_2026_09_24) == [
            "publish / build — step “Check the built site actually serves its data”"
        ]

    def test_a_job_that_failed_without_a_failed_step_is_still_named(self) -> None:
        # A job that timed out or could not start has no failed step to point at.
        jobs = [{"name": "refresh", "conclusion": "failure", "steps": []}]
        assert pipeline.failed_jobs(jobs) == ["refresh"]

    def test_the_message_carries_the_run_url(self) -> None:
        text = pipeline.failure_message(
            "Data Refresh",
            "https://github.com/MarlenMM/quantpulse/actions/runs/35937083808",
            pipeline.failed_jobs(JOBS_2026_09_24),
        )
        assert text.startswith("QuantPulse: Data Refresh failed")
        assert "publish / build" in text
        assert "actions/runs/35937083808" in text

    def test_no_failed_job_found_still_says_the_run_failed(self) -> None:
        text = pipeline.failure_message("Data Refresh", "u", [])
        assert "failed" in text and "u" in text


# ------------------------------------------------- delivery, and the unset path


@pytest.fixture
def webhook(monkeypatch: pytest.MonkeyPatch):
    """Set the webhook for one test, with the settings cache cleared both ways."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv("ALERT_DISCORD_WEBHOOK_URL", raising=False)
        else:
            monkeypatch.setenv("ALERT_DISCORD_WEBHOOK_URL", value)
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Every call to Discord, recorded instead of made."""
    calls: list[tuple[str, str]] = []

    def fake_send(url: str, text: str) -> int:
        calls.append((url, text))
        return 1

    monkeypatch.setattr(pipeline.discord, "send", fake_send)
    return calls


class TestUnconfigured:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_an_unset_or_blank_webhook_logs_one_line_and_succeeds(
        self, value, webhook, sent, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        # An unset Actions secret arrives as "" -- the state of this repository
        # today and of every fork. That must never be an error.
        webhook(value)
        jobs = tmp_path / "jobs.json"
        jobs.write_text(json.dumps(JOBS_2026_09_24))
        with caplog.at_level(logging.INFO, logger="quantpulse.alerting.pipeline"):
            code = pipeline_alert.main(
                [
                    "failure",
                    "--workflow",
                    "Data Refresh",
                    "--run-url",
                    "u",
                    "--jobs-json",
                    str(jobs),
                ]
            )
        assert code == 0
        assert sent == []
        records = [r for r in caplog.records if r.name == "quantpulse.alerting.pipeline"]
        assert len(records) == 1, [r.getMessage() for r in records]
        assert "not configured" in records[0].getMessage()


class TestConfigured:
    def test_a_failure_is_posted_and_the_url_is_never_printed(
        self, webhook, sent, capsys: pytest.CaptureFixture[str], caplog, tmp_path: Path
    ) -> None:
        webhook(SECRET)
        jobs = tmp_path / "jobs.json"
        jobs.write_text(json.dumps(JOBS_2026_09_24))
        with caplog.at_level(logging.DEBUG):
            code = pipeline_alert.main(
                [
                    "failure",
                    "--workflow",
                    "Data Refresh",
                    "--run-url",
                    "u",
                    "--jobs-json",
                    str(jobs),
                ]
            )
        assert code == 0
        assert len(sent) == 1
        assert sent[0][0] == SECRET
        assert "publish / build" in sent[0][1]
        out = capsys.readouterr()
        for stream in (out.out, out.err, caplog.text):
            assert "sEcReT" not in stream
            assert "1234567890" not in stream

    def test_a_delivery_failure_fails_the_job_without_the_url(
        self, webhook, monkeypatch, capsys, caplog, tmp_path: Path
    ) -> None:
        # Configured but broken is a real fault -- a revoked webhook would
        # otherwise make every future notice vanish silently.
        webhook(SECRET)

        def refuse(url: str, text: str) -> int:
            raise pipeline.discord.WebhookError(
                "posting message 1/1 to https://discord.com/api/webhooks/… failed: "
                "HTTPError (HTTP 401)"
            )

        monkeypatch.setattr(pipeline.discord, "send", refuse)
        jobs = tmp_path / "jobs.json"
        jobs.write_text("[]")
        code = pipeline_alert.main(
            ["failure", "--workflow", "Data Refresh", "--run-url", "u", "--jobs-json", str(jobs)]
        )
        assert code == 1
        out = capsys.readouterr()
        assert "HTTP 401" in out.err + caplog.text
        assert "sEcReT" not in out.out + out.err + caplog.text


class TestStalenessCommand:
    def _run(self, monkeypatch, health: dict | Exception, now: datetime) -> int:
        def fake_fetch(url: str) -> dict:
            if isinstance(health, Exception):
                raise health
            return health

        # Patched where it is used, not where it is defined.
        monkeypatch.setattr(pipeline_alert, "fetch_health", fake_fetch)
        monkeypatch.setattr(pipeline_alert, "_now", lambda: now)
        return pipeline_alert.main(["staleness", "--site-url", "https://example.invalid/qp/"])

    def test_a_current_site_sends_nothing(self, monkeypatch, webhook, sent) -> None:
        webhook(SECRET)
        health = {"freshness": {"prices": "2026-09-21"}}
        assert self._run(monkeypatch, health, evening(date(2026, 9, 22))) == 0
        assert sent == []

    def test_a_stale_site_is_reported_and_fails_the_job(self, monkeypatch, webhook, sent) -> None:
        webhook(SECRET)
        health = {"freshness": {"prices": "2026-08-07"}}
        assert self._run(monkeypatch, health, evening(date(2026, 8, 12))) == 1
        assert len(sent) == 1
        assert "2026-08-07" in sent[0][1]

    def test_stale_with_no_webhook_still_fails_the_job(
        self, monkeypatch, webhook, sent, caplog
    ) -> None:
        # The missing secret is not an error; the frozen site is. A red run is
        # also what makes GitHub's own failure email fire.
        webhook("")
        health = {"freshness": {"prices": "2026-08-07"}}
        assert self._run(monkeypatch, health, evening(date(2026, 8, 12))) == 1
        assert sent == []

    def test_an_unreadable_health_file_is_reported(self, monkeypatch, webhook, sent) -> None:
        webhook(SECRET)
        assert self._run(monkeypatch, OSError("boom"), evening(date(2026, 9, 22))) == 1
        assert "health.json" in sent[0][1]

    def test_a_health_file_without_prices_is_reported(self, monkeypatch, webhook, sent) -> None:
        webhook(SECRET)
        assert self._run(monkeypatch, {"freshness": {}}, evening(date(2026, 9, 22))) == 1
        assert len(sent) == 1


# ------------------------------------------------------------------- the wiring


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _runs(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def _webhook_from_secret(job: dict) -> None:
    envs = [step.get("env", {}) for step in job["steps"]]
    values = [env.get("ALERT_DISCORD_WEBHOOK_URL") for env in envs if env]
    assert "${{ secrets.ALERT_DISCORD_WEBHOOK_URL }}" in values


class TestTheRefreshWiring:
    def test_a_failed_night_is_announced(self) -> None:
        jobs = _load("refresh_data.yml")["jobs"]
        (notify,) = [j for j in jobs.values() if "pipeline_alert.py failure" in _runs(j)]
        assert notify["if"] == "failure()"
        # Every job whose failure matters, including the publish that failed on
        # 2026-09-24 while the refresh succeeded.
        assert {"refresh", "publish", "keepalive"} <= set(notify["needs"])
        assert notify["permissions"]["actions"] == "read"
        _webhook_from_secret(notify)

    def test_the_staleness_check_does_not_wait_for_the_refresh(self) -> None:
        # It has to run on the nights the refresh fails, which is the point.
        jobs = _load("refresh_data.yml")["jobs"]
        (stale,) = [j for j in jobs.values() if "pipeline_alert.py staleness" in _runs(j)]
        assert "needs" not in stale
        assert "https://marlenmm.github.io/quantpulse/" in _runs(stale)
        _webhook_from_secret(stale)

    def test_a_stale_site_is_not_announced_twice(self) -> None:
        # The staleness job posts its own notice; if the failure notice also
        # depended on it, one frozen site would produce two messages.
        jobs = _load("refresh_data.yml")["jobs"]
        (name,) = [n for n, j in jobs.items() if "pipeline_alert.py staleness" in _runs(j)]
        (notify,) = [j for j in jobs.values() if "pipeline_alert.py failure" in _runs(j)]
        assert name not in notify["needs"]

    def test_the_called_publish_leaves_the_notice_to_the_refresh(self) -> None:
        jobs = _load("refresh_data.yml")["jobs"]
        assert jobs["publish"]["with"]["failure_notice"] == "caller"


class TestThePagesWiring:
    def test_a_failed_publish_on_push_is_announced_once(self) -> None:
        pages = _load("pages.yml")
        (notify,) = [j for j in pages["jobs"].values() if "pipeline_alert.py failure" in _runs(j)]
        # Skipped when the refresh called this workflow -- the refresh's own
        # notice already covers that run -- and a string, not a boolean:
        # GitHub's `==` is loose, and null == false is true there.
        assert notify["if"] == "failure() && inputs.failure_notice != 'caller'"
        assert {"build", "deploy"} <= set(notify["needs"])
        assert notify["permissions"]["actions"] == "read"
        _webhook_from_secret(notify)

    def test_the_input_exists_for_the_refresh_to_pass(self) -> None:
        pages = _load("pages.yml")
        triggers = pages.get("on") or pages[True]
        spec = triggers["workflow_call"]["inputs"]["failure_notice"]
        assert spec["type"] == "string"


def test_no_workflow_writes_the_webhook_into_a_command() -> None:
    # Secrets reach the script through env only; a `run:` line that
    # interpolates one puts it in the rendered script GitHub logs on failure.
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        for job in _load(workflow.name)["jobs"].values():
            assert "secrets.ALERT_DISCORD_WEBHOOK_URL" not in _runs(job), workflow.name
