"""Fetching the demo database, which is a release asset rather than a commit.

Forty committed revisions of it were 309 MB of a 317 MB repository -- 97.5%,
against 8 MB for every source file, test and document together -- growing about
7.7 MB packed per refresh, five refreshes a week.

The download itself is never exercised here; `requests` is patched. What is
tested is everything around it, which is where a fetch like this goes wrong: a
half-written file left at the real path, an HTML error page saved as a database,
and re-downloading something already present.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# `scripts` is on pytest's `pythonpath`, so the CLI imports bare, the way
# `test_refresh_workflow.py` imports `refresh_data`.
import fetch_demo_db
from quantpulse import demo_data

_SQLITE = b"SQLite format 3\x00"


def _response(body: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [body]
    response.raise_for_status.return_value = None
    return response


def _payload(size: int = demo_data.MIN_BYTES + 1) -> bytes:
    return _SQLITE + b"\x00" * (size - len(_SQLITE))


class TestFetch:
    def test_it_writes_the_database_when_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "demo.db"
        with patch("quantpulse.demo_data.requests.get", return_value=_response(_payload())):
            assert demo_data.fetch(target) is True
        assert target.exists()
        assert target.read_bytes()[:15] == b"SQLite format 3"

    def test_an_existing_file_is_left_alone(self, tmp_path: Path) -> None:
        """Not merely "does not overwrite": it must not spend 66 MB of a
        reader's bandwidth to discover that."""
        target = tmp_path / "demo.db"
        target.write_bytes(b"mine")
        with patch("quantpulse.demo_data.requests.get") as get:
            assert demo_data.fetch(target) is False
        get.assert_not_called()
        assert target.read_bytes() == b"mine"

    def test_a_short_response_is_refused(self, tmp_path: Path) -> None:
        """GitHub answers a bad release URL with an HTML error page. Saved as a
        database it surfaces much later as a corrupt-database error from SQLite,
        pointing at everything except the download."""
        target = tmp_path / "demo.db"
        with patch(
            "quantpulse.demo_data.requests.get", return_value=_response(b"<html>Not Found</html>")
        ):
            with pytest.raises(RuntimeError, match="not the database"):
                demo_data.fetch(target)
        assert not target.exists()

    def test_a_large_non_sqlite_response_is_refused(self, tmp_path: Path) -> None:
        """Size alone is not the check: a big wrong file is still wrong."""
        target = tmp_path / "demo.db"
        body = b"\x89PNG\r\n" + b"\x00" * demo_data.MIN_BYTES
        with patch("quantpulse.demo_data.requests.get", return_value=_response(body)):
            with pytest.raises(RuntimeError, match="not a SQLite database"):
                demo_data.fetch(target)
        assert not target.exists()

    def test_a_download_that_dies_mid_stream_leaves_no_partial_file(self, tmp_path: Path) -> None:
        """The case that actually exercises the cleanup.

        A connection that fails before the first byte never creates the
        temporary file, so it cannot prove anything about removing it — that
        version of this test passed with the cleanup deleted. Here the stream
        yields a chunk and *then* fails, which is what a dropped connection
        partway through 66 MB looks like.
        """
        target = tmp_path / "demo.db"

        def dying_chunks(chunk_size: int = 0):
            yield _SQLITE + b"\x00" * 1024
            raise OSError("connection reset")

        response = MagicMock()
        response.__enter__.return_value = response
        response.raise_for_status.return_value = None
        response.iter_content.side_effect = dying_chunks

        with patch("quantpulse.demo_data.requests.get", return_value=response):
            with pytest.raises(OSError):
                demo_data.fetch(target)

        assert not target.exists()
        assert list(tmp_path.iterdir()) == [], "a .partial file was left behind"

    def test_a_failed_download_leaves_nothing_at_the_real_path(self, tmp_path: Path) -> None:
        """The failure that makes a download permanently broken rather than
        retryable: a partial file at the target path is read as "already here"
        by the next run and never repaired."""
        target = tmp_path / "demo.db"
        with patch("quantpulse.demo_data.requests.get", side_effect=OSError("connection reset")):
            with pytest.raises(OSError):
                demo_data.fetch(target)
        assert not target.exists()
        assert list(tmp_path.iterdir()) == [], "a .partial file was left behind"

    def test_the_url_points_at_the_rolling_release_tag(self) -> None:
        """A rolling tag, not a numbered series -- keeping 40 of these is the
        thing being fixed."""
        assert demo_data.RELEASE_URL.endswith("/releases/download/demo-data/quantpulse_demo.db")
        assert demo_data.RELEASE_URL.startswith("https://")


class TestPinnedFixture:
    """The pinned copy CI reads, which is a different file from the demo's.

    CI used to fetch the rolling asset, so when a refresh published a gutted
    database three unrelated frontend pull requests went red on a Streamlit test
    about effective weights -- failing on a file none of them had touched.
    """

    def test_a_digest_that_does_not_match_is_refused(self, tmp_path: Path) -> None:
        """Digest, not size: the database that caused this was 11.7 MB of
        perfectly well-formed SQLite and passed every other check here."""
        target = tmp_path / "demo.db"
        with patch("quantpulse.demo_data.requests.get", return_value=_response(_payload())):
            with pytest.raises(RuntimeError, match="not the pinned one"):
                demo_data.fetch(target, expected_sha256="0" * 64)
        assert not target.exists()
        assert list(tmp_path.iterdir()) == [], "a .partial file was left behind"

    def test_the_error_says_it_is_not_about_the_code(self, tmp_path: Path) -> None:
        """The whole point of pinning is attribution. A mismatch that reads like
        a test failure would leave the reader exactly where this started."""
        target = tmp_path / "demo.db"
        with patch("quantpulse.demo_data.requests.get", return_value=_response(_payload())):
            with pytest.raises(RuntimeError) as failure:
                demo_data.fetch(target, expected_sha256="0" * 64)
        assert "says nothing about the code being tested" in str(failure.value)

    def test_a_matching_digest_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "demo.db"
        body = _payload()
        expected = hashlib.sha256(body).hexdigest()
        with patch("quantpulse.demo_data.requests.get", return_value=_response(body)):
            assert demo_data.fetch(target, expected_sha256=expected) is True
        assert target.read_bytes() == body

    def test_the_fixture_is_a_different_asset_from_the_rolling_one(self) -> None:
        """Pointing both names at one file would undo the split silently, and
        every test here would still pass."""
        assert demo_data.CI_FIXTURE_URL != demo_data.RELEASE_URL
        assert demo_data.CI_FIXTURE_URL.endswith("/releases/download/ci-fixture/quantpulse_ci.db")
        assert len(demo_data.CI_FIXTURE_SHA256) == 64
        assert set(demo_data.CI_FIXTURE_SHA256) <= set("0123456789abcdef")


class TestCli:
    """`--ci-fixture` is what the CI workflow passes; nothing else may."""

    # Patched where it is *used*, not where it is defined: the CLI does
    # `from quantpulse.demo_data import fetch`, which binds the function into
    # its own namespace at import time. Patching `quantpulse.demo_data.fetch`
    # replaces a name the CLI never reads again, so both of these would have
    # passed while asserting nothing about the real call.
    def test_the_flag_selects_the_pinned_source(self, tmp_path: Path) -> None:
        target = tmp_path / "demo.db"
        with patch("fetch_demo_db.fetch") as fetch:
            fetch.side_effect = lambda path, **_: path.write_bytes(_payload(demo_data.MIN_BYTES))
            assert fetch_demo_db.main(["--ci-fixture", str(target)]) == 0
        assert fetch.call_args.kwargs["url"] == demo_data.CI_FIXTURE_URL
        assert fetch.call_args.kwargs["expected_sha256"] == demo_data.CI_FIXTURE_SHA256

    def test_without_the_flag_it_is_the_rolling_database_and_unpinned(self, tmp_path: Path) -> None:
        """The demo and `./run.sh` must keep getting the current data -- a digest
        here would pin the shop window to whatever was current on the day."""
        target = tmp_path / "demo.db"
        with patch("fetch_demo_db.fetch") as fetch:
            fetch.side_effect = lambda path, **_: path.write_bytes(_payload(demo_data.MIN_BYTES))
            assert fetch_demo_db.main([str(target)]) == 0
        assert fetch.call_args.kwargs["url"] == demo_data.RELEASE_URL
        assert fetch.call_args.kwargs["expected_sha256"] is None

    def test_an_existing_file_is_still_left_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "demo.db"
        target.write_bytes(b"mine")
        with patch("fetch_demo_db.fetch") as fetch:
            assert fetch_demo_db.main(["--ci-fixture", str(target)]) == 0
        fetch.assert_not_called()
