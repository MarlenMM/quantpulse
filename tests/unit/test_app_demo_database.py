"""The hosted Streamlit app fetches its database whichever page is opened first (point 44).

Streamlit Community Cloud clones the repository -- no database, it is a release
asset -- and runs `app/Home.py`. Only Home called `ensure_demo_database()`. On a
cold container whose first visitor opened a deep link (`/Screener`, a shared
stock page), that page connected first, SQLite created an **empty**
`quantpulse_demo.db`, and from then on `ensure_demo_database()` saw a file and
never downloaded: every page showed `OperationalError: no such table` for the
life of the container, Home included. Reproduced on a clean checkout with a
venv built from `requirements.txt` alone, as the host builds it.

Two fixes, each tested here: every database read in the app goes through
`lib.data.get_session`, which ensures the database first; and a zero-byte file
-- the artefact SQLite leaves, holding nothing -- counts as missing.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib import data
from quantpulse import demo_data
from quantpulse.config import get_settings

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture
def database_at(monkeypatch: pytest.MonkeyPatch) -> Iterator:
    def _point(path: Path) -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
        get_settings.cache_clear()
        data.ensure_demo_database.clear()

    yield _point
    get_settings.cache_clear()
    data.ensure_demo_database.clear()


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every download attempt, recorded instead of made."""
    calls: list[Path] = []

    def fake_fetch(target: Path, **_: object) -> bool:
        calls.append(Path(target))
        return True

    # Patched where it is looked up: `ensure_demo_database` imports it from the
    # module at call time.
    monkeypatch.setattr(demo_data, "fetch", fake_fetch)
    return calls


class TestWhatCountsAsMissing:
    def test_a_missing_file_is_downloaded(self, tmp_path, database_at, downloads) -> None:
        database_at(tmp_path / "quantpulse_demo.db")
        assert data.ensure_demo_database()
        assert downloads == [tmp_path / "quantpulse_demo.db"]

    def test_the_empty_file_sqlite_leaves_behind_is_downloaded_over(
        self, tmp_path, database_at, downloads
    ) -> None:
        # The exact state a deep link on a cold container produced.
        target = tmp_path / "quantpulse_demo.db"
        target.touch()
        database_at(target)
        assert data.ensure_demo_database()
        assert downloads == [target]

    def test_a_real_database_is_never_replaced(self, tmp_path, database_at, downloads) -> None:
        # Any non-empty file is someone's data -- a local dev database may be
        # small -- so only the zero-byte artefact counts as missing.
        target = tmp_path / "quantpulse.db"
        target.write_bytes(b"SQLite format 3\x00" + b"\x00" * 4080)
        database_at(target)
        assert not data.ensure_demo_database()
        assert downloads == []


class TestEveryReadEnsuresTheDatabaseFirst:
    def test_the_apps_session_ensures_before_it_connects(
        self, tmp_path, database_at, downloads
    ) -> None:
        database_at(tmp_path / "quantpulse_demo.db")
        with data.get_session():
            pass
        assert downloads == [tmp_path / "quantpulse_demo.db"]

    def test_no_page_or_helper_opens_a_session_any_other_way(self) -> None:
        """A page that imports the engine's own `get_session` skips the download."""
        offenders = []
        for path in sorted(APP.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "quantpulse.storage.db":
                    if path != APP / "lib" / "data.py":
                        offenders.append(f"{path.relative_to(APP)}:{node.lineno}")
        assert not offenders, f"open sessions through lib.data.get_session instead: {offenders}"


def test_the_empty_file_is_really_replaced_end_to_end(tmp_path, database_at) -> None:
    """The caller's rule is worthless if the callee disagrees.

    The first version of this fix taught `ensure_demo_database` that zero bytes
    means missing and mocked `fetch` in its tests -- which hid `fetch`'s own
    `if target.exists(): return False`. The simulated host stayed broken. Here
    the real `fetch` runs, with only the HTTP response stubbed.
    """
    target = tmp_path / "quantpulse_demo.db"
    target.touch()
    database_at(target)
    body = b"SQLite format 3\x00" + b"\x00" * demo_data.MIN_BYTES
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [body]
    with patch("quantpulse.demo_data.requests.get", return_value=response):
        assert data.ensure_demo_database()
    assert target.stat().st_size == len(body)
