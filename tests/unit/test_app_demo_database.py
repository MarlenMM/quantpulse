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


class TestThePublishedDatabaseIsReadAtTheCurrentSchema:
    """Point 45: nothing that reads the published database migrated it first.

    The nightly migrates before it writes; the Pages build and the hosted app
    read the release asset as downloaded. With the first column added since the
    database became a release asset (finding 34), the pre-render would have
    failed and the hosted app's Stock Detail page would have raised -- and kept
    raising on a warm container, which never downloads the file again.
    """

    @staticmethod
    def _database_at(path: Path, revision: str) -> str:
        from alembic import command
        from alembic.config import Config

        url = f"sqlite:///{path}"
        config = Config()
        config.set_main_option(
            "script_location", str(APP.parent / "src" / "quantpulse" / "storage" / "migrations")
        )
        import os

        before = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = url
        get_settings.cache_clear()
        try:
            command.upgrade(config, revision)
        finally:
            if before is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = before
            get_settings.cache_clear()
        return url

    def test_a_database_behind_head_is_brought_up_to_it(self, tmp_path: Path) -> None:
        import sqlite3

        from alembic.config import Config
        from alembic.script import ScriptDirectory

        url = self._database_at(tmp_path / "old.db", "f5ad63651d6b")
        assert demo_data.ensure_schema_current(url) is True
        config = Config()
        config.set_main_option(
            "script_location", str(APP.parent / "src" / "quantpulse" / "storage" / "migrations")
        )
        head = ScriptDirectory.from_config(config).get_current_head()
        version = (
            sqlite3.connect(tmp_path / "old.db")
            .execute("select version_num from alembic_version")
            .fetchone()[0]
        )
        assert version == head
        # Idempotent: the second call finds nothing to do.
        assert demo_data.ensure_schema_current(url) is False

    def test_a_database_alembic_does_not_manage_is_left_alone(self, tmp_path: Path) -> None:
        import sqlite3

        path = tmp_path / "plain.db"
        sqlite3.connect(path).execute("create table t (x int)").connection.commit()
        assert demo_data.ensure_schema_current(f"sqlite:///{path}") is False
        tables = {r[0] for r in sqlite3.connect(path).execute("select name from sqlite_master")}
        assert tables == {"t"}

    def test_the_apps_session_brings_the_schema_current_before_reading(
        self, monkeypatch: pytest.MonkeyPatch, downloads
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            demo_data, "ensure_schema_current", lambda url: calls.append(url) or False
        )
        data.ensure_schema.clear()
        with data.get_session():
            pass
        data.ensure_schema.clear()
        assert calls == [get_settings().database_url]
