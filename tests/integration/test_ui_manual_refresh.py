"""The Settings page's refresh control, driven through the real page script.

This is the only trigger a refresh has now -- the nightly cron is gone -- so
"the runner has unit tests" is not enough on its own. What matters here is what
someone opening the page actually gets: a button when they are allowed to press
it, no button when they are not, and a live log while a run is in flight.

Each test runs the real Streamlit script through `streamlit.testing.v1.AppTest`
with a stand-in runner, so nothing spawns a process or touches a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from lib.refresh import RefreshState
from quantpulse.config import Settings
from quantpulse.storage.models import Base

SETTINGS_PAGE = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "5_Settings.py")


class FakeRunner:
    """A runner that reports a fixed state and records what it was asked to do."""

    def __init__(self, state: RefreshState) -> None:
        self._state = state
        # Set to make the runner report a *different* state from the second
        # `state()` call onwards, which is how a run finishing mid-render looks.
        self.state_after_first_call: RefreshState | None = None
        self.started: list[tuple[bool, bool]] = []
        self.stopped = 0

    def state(self) -> RefreshState:
        state = self._state
        if self.state_after_first_call is not None:
            self._state = self.state_after_first_call
        return state

    def start(self, *, weekly: bool = False, ignore_market_calendar: bool = False) -> bool:
        self.started.append((weekly, ignore_market_calendar))
        return True

    def stop(self) -> bool:
        self.stopped += 1
        return True


def _idle() -> RefreshState:
    return RefreshState(running=False, started_at=None, finished_at=None, returncode=None, log=())


def _running() -> RefreshState:
    return RefreshState(
        running=True,
        started_at=datetime.now() - timedelta(seconds=90),
        finished_at=None,
        returncode=None,
        log=("fetching prices", "scoring universe"),
    )


def _finished(returncode: int) -> RefreshState:
    started = datetime.now() - timedelta(minutes=4)
    return RefreshState(
        running=False,
        started_at=started,
        finished_at=started + timedelta(minutes=3, seconds=20),
        returncode=returncode,
        log=("fetching prices", "Traceback (most recent call last):"),
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def _page(engine: Engine, runner: FakeRunner, *, allowed: bool = True) -> Iterator[AppTest]:
    factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_session() -> Iterator[object]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    from lib import data as lib_data

    for name in dir(lib_data):
        if name.startswith("_"):
            continue
        reader = getattr(lib_data, name)
        if callable(reader) and hasattr(reader, "clear"):
            reader.clear()

    settings = Settings(
        _env_file=None,
        portfolio_backend="sqlite" if allowed else "session",
    )
    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.refresh.get_runner", return_value=runner),
        # Patched in its defining module, not on the page: AppTest re-executes
        # the page source on every run, so its `from quantpulse.config import
        # get_settings` re-binds to whatever is patched here.
        patch("quantpulse.config.get_settings", return_value=settings),
        patch("quantpulse.llm.providers.get_provider", return_value=None),
    ):
        at = AppTest.from_file(SETTINGS_PAGE, default_timeout=60)
        at.run()
        yield at


def _button(at: AppTest, label_fragment: str):
    for button in at.button:
        if label_fragment in button.label:
            return button
    return None


def test_a_local_instance_is_offered_the_button(engine: Engine) -> None:
    runner = FakeRunner(_idle())
    with _page(engine, runner) as at:
        assert not at.exception
        assert _button(at, "Refresh now") is not None


def test_pressing_it_starts_a_run(engine: Engine) -> None:
    runner = FakeRunner(_idle())
    with _page(engine, runner) as at:
        _button(at, "Refresh now").click().run()
    assert runner.started == [(False, False)]


def test_both_checkboxes_are_passed_through(engine: Engine) -> None:
    """Each checkbox has to reach the subprocess, or it is decoration.

    Worth asserting end to end because each flag crosses three boundaries on the
    way down -- widget, runner, command line -- and both failures are invisible:
    a weekly run that silently isn't weekly just leaves fundamentals stale, and
    a forced run that silently isn't forced looks exactly like a weekend no-op.
    Checked one at a time so a crossed pair of arguments cannot pass.
    """
    weekly_box, calendar_box = 0, 1
    for index, expected in ((weekly_box, (True, False)), (calendar_box, (False, True))):
        runner = FakeRunner(_idle())
        with _page(engine, runner) as at:
            at.checkbox[index].check().run()
            _button(at, "Refresh now").click().run()
        assert runner.started == [expected]


def test_the_public_demo_is_not_offered_the_button(engine: Engine) -> None:
    """Otherwise any visitor could start a multi-minute job on shared API quota."""
    runner = FakeRunner(_idle())
    with _page(engine, runner, allowed=False) as at:
        assert not at.exception
        assert _button(at, "Refresh now") is None
        assert not runner.started


def test_a_run_in_flight_shows_its_log_and_a_way_out(engine: Engine) -> None:
    runner = FakeRunner(_running())
    with _page(engine, runner) as at:
        assert not at.exception
        # No way to start a second one while the first is alive...
        assert _button(at, "Refresh now") is None
        # ...but the run is visible rather than the page just sitting there.
        assert any("Refresh running" in info.value for info in at.info)
        assert any("scoring universe" in block.value for block in at.code)
        assert _button(at, "Stop this refresh") is not None


def test_a_run_finishing_flips_the_page_back_to_idle(engine: Engine) -> None:
    """The normal way a run ends, and the one path that is not a steady state.

    The poller lives in a fragment, so it is the fragment -- not the page -- that
    notices the run is over, and it has two jobs at that moment: drop
    `lib.data`'s five-minute read caches so the freshness table stops serving
    pre-refresh numbers, and rerun the whole app so the idle branch replaces the
    live log. Neither failure looks like a bug from the outside (the page just
    keeps polling, or keeps showing yesterday's data), so both are asserted.

    The runner reports "running" once and "finished" from then on, which is
    exactly the real race: the run ends between the page rendering and the
    fragment's next poll. Everything below therefore happens in ONE `at.run()` --
    a second one would reach the idle branch by itself and prove nothing.
    """
    runner = FakeRunner(_running())
    runner.state_after_first_call = _finished(returncode=0)

    with patch.object(st.cache_data, "clear") as cleared, _page(engine, runner) as at:
        pass

    assert not at.exception
    assert cleared.called, "stale reads would outlive the refresh by five minutes"
    assert _button(at, "Refresh now") is not None
    assert _button(at, "Stop this refresh") is None
    assert any("Refresh finished" in success.value for success in at.success)


def test_stopping_a_run_reaches_the_runner(engine: Engine) -> None:
    runner = FakeRunner(_running())
    with _page(engine, runner) as at:
        _button(at, "Stop this refresh").click().run()
    assert runner.stopped == 1


def test_a_failed_run_is_reported_as_failed(engine: Engine) -> None:
    runner = FakeRunner(_finished(returncode=1))
    with _page(engine, runner) as at:
        assert not at.exception
        assert any("Refresh failed" in error.value for error in at.error)
        # The traceback is one click away, not buried in a terminal the user
        # never sees -- the app is the only surface a manual run has.
        assert any("Traceback" in block.value for block in at.code)


def test_a_finished_run_does_not_claim_more_than_the_exit_code_knows(engine: Engine) -> None:
    """Exit 0 covers `partial` and `skipped_non_trading_day` too.

    Both of those mean "your data did not fully refresh", so the page points at
    the pipeline-health table for the real status instead of saying "success".
    """
    runner = FakeRunner(_finished(returncode=0))
    with _page(engine, runner) as at:
        assert not at.exception
        finished = [s.value for s in at.success if "Refresh finished" in s.value]
        assert finished
        assert "See its status below" in finished[0]
        assert not any("success" in message.lower() for message in finished)
