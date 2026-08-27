"""The subprocess runner behind the Settings page's "Refresh now" button.

Worth real tests rather than a glance, because the button is now the *only*
trigger there is -- nothing runs on a schedule any more -- and the two ways
this can fail are both silent. If the runner reports a run as finished before
its output has been read, the page shows a truncated log and no outcome; if it
lets a second run start while the first is alive, two processes write the same
SQLite file at once.

Every test drives a trivial `python -c` child instead of the real refresh, so
the subject under test is the runner's own bookkeeping and not a nine-minute
data pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from lib.refresh import MAX_LOG_LINES, RefreshRunner


def _wait_until_idle(runner: RefreshRunner, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not runner.state().running:
            return
        time.sleep(0.02)
    raise AssertionError("the refresh never finished")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RefreshRunner:
    """A runner pointed at a stand-in script, so nothing real is refreshed."""
    script = tmp_path / "fake_refresh.py"
    script.write_text(
        "import sys\n"
        "print('step one')\n"
        "print('step two', file=sys.stderr)\n"
        "print(sys.executable)\n"
        "print('argv:', *sys.argv[1:])\n"
        "sys.exit(len(sys.argv) - 1)\n"
    )
    monkeypatch.setattr("lib.refresh.SCRIPT", script)
    monkeypatch.setattr("lib.refresh.REPO_ROOT", tmp_path)
    return RefreshRunner()


def test_a_fresh_runner_reports_nothing_has_run(runner: RefreshRunner) -> None:
    state = runner.state()
    assert state.never_run
    assert not state.running
    assert state.log == ()


def test_a_completed_run_records_its_exit_code_and_output(runner: RefreshRunner) -> None:
    assert runner.start()
    _wait_until_idle(runner)

    state = runner.state()
    assert not state.running
    assert state.succeeded
    assert state.returncode == 0
    assert state.started_at is not None and state.finished_at is not None
    # stderr is folded into stdout on purpose: `configure_logging` sends the
    # run's entire narration there, so a page tailing stdout alone would show
    # an empty log for a job that was talking the whole time.
    assert "step one" in state.log
    assert "step two" in state.log


def test_a_run_is_not_finished_until_its_exit_has_been_recorded(runner: RefreshRunner) -> None:
    """Why `running` is keyed on the recorded exit code and not on `Popen.poll()`.

    `poll()` goes non-None the instant the child exits -- which is *before* the
    reader thread has drained the tail of the pipe and recorded the outcome. A
    page rendering inside that window would report a run as finished while
    showing a truncated log and no success or failure at all, and it would do it
    rarely enough to look like a ghost rather than a bug.

    The window is a few milliseconds wide, so rather than race it, this puts the
    runner into exactly that state: a dead process object with nothing recorded
    yet. The invariant is that it still reads as running. (Nothing can latch on
    forever, because the drain thread records the exit from a `finally` -- every
    other test in this file depends on that and would hang otherwise.)
    """
    assert runner.start()
    _wait_until_idle(runner)
    assert not runner.state().running

    runner._returncode = None

    assert runner.state().running


def test_a_failing_run_is_reported_as_failed(runner: RefreshRunner) -> None:
    # The stand-in exits with its own argument count, so one flag means exit 1.
    assert runner.start(weekly=True)
    _wait_until_idle(runner)

    state = runner.state()
    assert not state.succeeded
    assert state.returncode == 1


def test_the_flags_reach_the_script_under_the_names_it_expects(runner: RefreshRunner) -> None:
    """The runner's two booleans have to arrive as `refresh_data.py`'s own flags.

    Asserted from inside the child rather than by reading the command line the
    runner built, because a flag renamed on only one side of that boundary is
    the failure worth catching -- and argparse would reject an unknown flag
    minutes after a click, in a place nobody is looking.
    """
    assert runner.start()
    _wait_until_idle(runner)
    assert "argv:" in runner.state().log

    assert runner.start(weekly=True, ignore_market_calendar=True)
    _wait_until_idle(runner)
    argv = next(line for line in runner.state().log if line.startswith("argv:"))
    assert argv == "argv: --force-weekly --ignore-market-calendar"


def test_a_second_run_cannot_start_while_one_is_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two browser tabs are still one SQLite file."""
    script = tmp_path / "slow_refresh.py"
    script.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr("lib.refresh.SCRIPT", script)
    monkeypatch.setattr("lib.refresh.REPO_ROOT", tmp_path)
    runner = RefreshRunner()

    assert runner.start()
    try:
        assert runner.state().running
        assert not runner.start()
    finally:
        assert runner.stop()
    _wait_until_idle(runner)

    assert not runner.state().running
    # ...and once it is over the runner is reusable, not latched shut. Started
    # and stopped again rather than run to completion: the stand-in sleeps 30s.
    assert runner.start()
    assert runner.stop()
    _wait_until_idle(runner)


def test_stopping_when_nothing_runs_is_a_no_op(runner: RefreshRunner) -> None:
    assert not runner.stop()


def test_the_log_buffer_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run that logs a line per ticker must not grow the app process."""
    script = tmp_path / "chatty_refresh.py"
    script.write_text(f"for i in range({MAX_LOG_LINES * 2}):\n    print(i)\n")
    monkeypatch.setattr("lib.refresh.SCRIPT", script)
    monkeypatch.setattr("lib.refresh.REPO_ROOT", tmp_path)
    runner = RefreshRunner()

    assert runner.start()
    _wait_until_idle(runner)

    log = runner.state().log
    assert len(log) == MAX_LOG_LINES
    # The tail is what's kept -- where a run got to matters more than where it began.
    assert log[-1] == str(MAX_LOG_LINES * 2 - 1)


def test_the_child_runs_under_this_interpreter(runner: RefreshRunner) -> None:
    """Not a bare `python`, which on a developer's PATH is rarely the app's venv.

    The refresh imports the whole `quantpulse` package plus torch and spaCy; a
    different interpreter would fail on the first import, minutes after the
    button was clicked and with a traceback that names none of this.
    """
    assert runner.start()
    _wait_until_idle(runner)
    assert sys.executable in runner.state().log
