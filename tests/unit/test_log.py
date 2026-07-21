import logging

import pytest

import quantpulse.utils.log as log


@pytest.fixture(autouse=True)
def _reset_run_id() -> None:
    log._RUN_ID = None
    yield
    log._RUN_ID = None


def test_get_run_id_prefers_github_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "1234567890")
    assert log.get_run_id() == "1234567890"


def test_get_run_id_generates_and_caches_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    first = log.get_run_id()
    assert first
    assert log.get_run_id() == first  # cached for the life of the process


def test_configure_logging_tags_records_with_run_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    run_id = log.configure_logging("INFO", run_id="abc123")

    assert run_id == "abc123"
    logging.getLogger("quantpulse.test").warning("hello")

    err = capsys.readouterr().err
    assert "run=abc123" in err
    assert "hello" in err


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    log.configure_logging("INFO", run_id="one")
    log.configure_logging("INFO", run_id="two")

    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_quantpulse", False)]
    assert len(ours) == 1  # re-invocation replaces, never stacks
