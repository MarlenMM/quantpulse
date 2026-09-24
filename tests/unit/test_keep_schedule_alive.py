"""The nightly schedule must not switch itself off (finding 26).

GitHub: *"In a public repository, scheduled workflows are automatically
disabled when no repository activity has occurred in 60 days."* Until point 18
the refresh committed the demo database five nights a week, so the repository
was never idle. Moving the database to a release asset removed that, and the
workflow's own comment said the rule "no longer applies" -- written while the
schedule was switched off, and left behind when it came back.

GitHub does not define "repository activity". The one thing every account
agrees counts is a commit on the default branch, so that is the mechanism: the
refresh calls ``keepalive.yml``, which commits a short status file only when
``main`` has been idle for ``DEFAULT_MAX_IDLE_DAYS``. Two things are asserted
here -- the decision and the file (the script), and the wiring that makes it
actually run (the workflows) -- because either alone passes with the other
broken.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import keep_schedule_alive as ksa

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
NOW = datetime(2026, 11, 20, 0, 30, tzinfo=UTC)


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow: dict) -> dict:
    # `on:` parses as the YAML 1.1 boolean True.
    return workflow.get("on") or workflow[True]


class TestTheDecision:
    def test_a_branch_idle_past_the_threshold_gets_a_heartbeat(self) -> None:
        last = NOW - timedelta(days=ksa.DEFAULT_MAX_IDLE_DAYS, minutes=1)
        assert ksa.needs_heartbeat(last, NOW, ksa.DEFAULT_MAX_IDLE_DAYS)

    def test_a_recently_active_branch_is_left_alone(self) -> None:
        # One commit a month when nobody else commits; none at all otherwise.
        # Point 18 existed to stop data commits bloating the history.
        last = NOW - timedelta(days=ksa.DEFAULT_MAX_IDLE_DAYS - 1)
        assert not ksa.needs_heartbeat(last, NOW, ksa.DEFAULT_MAX_IDLE_DAYS)

    def test_the_threshold_leaves_weeks_of_retries_before_github_acts(self) -> None:
        # The refresh runs on weekdays, so every day of margin is roughly one
        # more nightly attempt at the commit. Two weeks is about ten attempts --
        # enough to ride out a run of failed nights or a holiday week.
        margin = ksa.GITHUB_INACTIVITY_LIMIT_DAYS - ksa.DEFAULT_MAX_IDLE_DAYS
        assert margin >= 14

    @pytest.mark.parametrize("days", [ksa.GITHUB_INACTIVITY_LIMIT_DAYS, 90])
    def test_a_threshold_at_or_past_githubs_limit_is_refused(self, days: int) -> None:
        with pytest.raises(ValueError, match="60"):
            ksa.needs_heartbeat(NOW, NOW, days)

    def test_zero_forces_a_heartbeat_so_the_commit_path_can_be_proven_by_hand(self) -> None:
        assert ksa.needs_heartbeat(NOW, NOW, 0)


class TestTheStatusFile:
    def test_it_records_what_a_reader_would_want_to_know(self) -> None:
        text = ksa.render_status(
            now=NOW,
            published_at="2026-11-19T00:19:30Z",
            published_bytes=82_796_544,
            refresh_result="success",
            run_url="https://github.com/MarlenMM/quantpulse/actions/runs/1",
        )
        assert "2026-11-19T00:19:30Z" in text
        assert "82.8 MB" in text
        assert "success" in text
        assert "actions/runs/1" in text
        # And why it exists, so nobody deletes it as noise.
        assert "60 days" in text

    def test_a_manual_heartbeat_does_not_claim_a_refresh_result(self) -> None:
        text = ksa.render_status(
            now=NOW, published_at="", published_bytes=0, refresh_result="", run_url="u"
        )
        assert "not part of a refresh run" in text
        assert "unknown" in text


class TestTheScript:
    def _run(self, tmp_path: Path, last: datetime, max_idle: int) -> tuple[int, Path, Path]:
        status = tmp_path / "status.md"
        output = tmp_path / "github_output"
        code = ksa.main(
            [
                "--last-commit-epoch",
                str(int(last.timestamp())),
                "--max-idle-days",
                str(max_idle),
                "--status-file",
                str(status),
                "--github-output",
                str(output),
                "--now",
                NOW.isoformat(),
                "--published-at",
                "2026-11-19T00:19:30Z",
                "--published-bytes",
                "82796544",
                "--refresh-result",
                "success",
                "--run-url",
                "https://example.invalid/run",
            ]
        )
        return code, status, output

    def test_an_idle_branch_writes_the_file_and_asks_for_a_commit(self, tmp_path: Path) -> None:
        code, status, output = self._run(tmp_path, NOW - timedelta(days=45), 30)
        assert code == 0
        assert status.exists()
        assert output.read_text().strip() == "commit=true"

    def test_an_active_branch_touches_nothing(self, tmp_path: Path) -> None:
        code, status, output = self._run(tmp_path, NOW - timedelta(days=3), 30)
        assert code == 0
        assert not status.exists()
        assert output.read_text().strip() == "commit=false"


class TestTheWiring:
    """The script is worthless unless something scheduled actually calls it."""

    def test_the_scheduled_refresh_calls_the_keepalive(self) -> None:
        refresh = _load("refresh_data.yml")
        assert _triggers(refresh).get("schedule"), "the refresh is no longer scheduled"
        callers = [
            job
            for job in refresh["jobs"].values()
            if job.get("uses") == "./.github/workflows/keepalive.yml"
        ]
        assert len(callers) == 1, "refresh_data.yml does not call keepalive.yml"
        job = callers[0]
        # A failed refresh night is exactly when the branch may be idle; the
        # heartbeat must not depend on the data being good.
        assert job.get("if") == "always()"
        assert job["permissions"]["contents"] == "write"
        assert job["with"]["refresh_result"] == "${{ needs.refresh.result }}"

    def test_the_keepalive_has_no_schedule_of_its_own(self) -> None:
        # The 60-day rule disables every scheduled workflow in the repository at
        # once, so a separate timer would switch off at the same moment as the
        # thing it is keeping alive. It rides on the refresh's schedule instead.
        triggers = _triggers(_load("keepalive.yml"))
        assert "schedule" not in triggers
        assert "workflow_call" in triggers
        assert "workflow_dispatch" in triggers

    def test_both_entry_points_default_to_the_scripts_threshold(self) -> None:
        triggers = _triggers(_load("keepalive.yml"))
        for event in ("workflow_call", "workflow_dispatch"):
            default = triggers[event]["inputs"]["max_idle_days"]["default"]
            assert int(default) == ksa.DEFAULT_MAX_IDLE_DAYS, event

    def test_it_commits_to_main_with_write_access(self) -> None:
        keepalive = _load("keepalive.yml")
        (job,) = keepalive["jobs"].values()
        assert job["permissions"]["contents"] == "write"
        checkout = next(
            s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
        )
        # The commit lands on main whatever triggered the run.
        assert checkout["with"]["ref"] == "main"
        runs = "\n".join(str(s.get("run", "")) for s in job["steps"])
        assert "scripts/keep_schedule_alive.py" in runs
        assert "git push origin HEAD:main" in runs
        commit = next(s for s in job["steps"] if "git commit" in str(s.get("run", "")))
        assert commit["if"] == "steps.decide.outputs.commit == 'true'"


class TestTheCommentsTellTheTruth:
    """Finding 41: the comment claiming the 60-day rule no longer applied is what hid 26."""

    def test_no_workflow_still_describes_a_committed_database(self) -> None:
        # Every file, not a named one: pages.yml carried the same `[skip ci]`
        # explanation, and a literal list stops covering whatever is added next.
        stale_phrases = (
            "none left to keep alive",
            "[skip ci]",
            "repo-committed file",
            "committed demo database",
            "has committed fresh data",
        )
        for workflow in sorted(WORKFLOWS.glob("*.yml")):
            text = workflow.read_text()
            for stale in stale_phrases:
                assert stale not in text, f"{workflow.name} still says {stale!r}"

    def test_the_refresh_states_the_rule_it_is_exposed_to(self) -> None:
        text = (WORKFLOWS / "refresh_data.yml").read_text()
        assert "60 days" in text
        assert "keepalive.yml" in text
