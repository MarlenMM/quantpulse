"""The guard that stands in front of `gh release upload --clobber`.

Written against the shape of the accident it exists to stop: on 2026-09-15 the
refresh job started from an empty database, because it had no step that fetched
the published one, and replaced an asset holding 417,993 price rows with one
holding 9,577. `pragma integrity_check` -- the only guard there was -- passed,
because the replacement was a perfectly well-formed database.

So the fixtures here are deliberately that shape rather than a toy one: a
"before" with history in it, an "after" that is smaller *and valid*, and the
assertion that valid-but-smaller is refused.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

# `scripts` is on pytest's `pythonpath`, so this is the spelling the rest of the
# suite uses for a script module (`import refresh_data`), not `scripts.x`.
from demo_db_guard import census, compare, main


def _database(path: Path, counts: dict[str, int]) -> Path:
    """A real SQLite file with `counts[table]` rows in each named table."""
    connection = sqlite3.connect(path)
    try:
        for table, rows in counts.items():
            connection.execute(f"create table {table} (n integer primary key)")
            connection.executemany(
                f"insert into {table} (n) values (?)", [(index,) for index in range(rows)]
            )
        connection.commit()
    finally:
        connection.close()
    return path


#: The real before/after, rounded to the tables that carry the history.
_PUBLISHED = {"price_history": 417, "forecasts": 333, "composite_scores": 392, "tickers": 133}
_REBUILT_FROM_EMPTY = {"price_history": 9, "forecasts": 0, "composite_scores": 15, "tickers": 5}


def test_census_counts_every_table(tmp_path: Path) -> None:
    snapshot = census(_database(tmp_path / "a.db", {"price_history": 3, "forecasts": 1}))
    assert snapshot["rows"] == {"price_history": 3, "forecasts": 1}
    assert snapshot["bytes"] > 0


def test_a_run_that_only_appends_is_allowed(tmp_path: Path) -> None:
    """The normal night: some tables grow, the rest stand still."""
    before = census(_database(tmp_path / "before.db", _PUBLISHED))
    after = census(
        _database(tmp_path / "after.db", {**_PUBLISHED, "price_history": 420, "forecasts": 340})
    )
    assert compare(before, after) == []


def test_the_2026_09_15_shape_is_refused(tmp_path: Path) -> None:
    before = census(_database(tmp_path / "before.db", _PUBLISHED))
    after = census(_database(tmp_path / "after.db", _REBUILT_FROM_EMPTY))
    losses = compare(before, after)
    # Worst loss first, so a log reader sees the biggest number immediately.
    assert losses[0].startswith("price_history: 417 rows before, 9 after")
    assert any(message.startswith("forecasts: 333 rows before, 0 after") for message in losses)
    assert len(losses) == 4


def test_a_table_that_vanished_counts_as_lost_not_skipped(tmp_path: Path) -> None:
    """A dropped table has no "after" row to compare against, and the naive
    loop that iterates the *current* census would silently pass it."""
    before = census(_database(tmp_path / "before.db", {"forecasts": 12}))
    after = census(_database(tmp_path / "after.db", {"price_history": 12}))
    assert compare(before, after) == ["forecasts: 12 rows before, 0 after (-12)"]


def test_visitor_local_tables_are_exempt(tmp_path: Path) -> None:
    """`watchlist` and the two portfolio tables are session-backend state on the
    public demo and legitimately empty; they say nothing about lost history."""
    before = census(_database(tmp_path / "before.db", {"watchlist": 5, "price_history": 9}))
    after = census(_database(tmp_path / "after.db", {"watchlist": 0, "price_history": 9}))
    assert compare(before, after) == []


def test_check_refuses_when_the_baseline_is_missing(tmp_path: Path, capsys) -> None:
    """A missing baseline means the fetch step did not run -- the original bug
    itself -- so this must fail rather than wave the upload through."""
    database = _database(tmp_path / "after.db", _REBUILT_FROM_EMPTY)
    assert main(["check", str(tmp_path / "nothing.json"), str(database)]) == 1
    assert "never fetched" in capsys.readouterr().err


def test_capture_then_check_round_trips(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    database = _database(tmp_path / "db.db", _PUBLISHED)
    assert main(["capture", str(baseline), str(database)]) == 0
    assert json.loads(baseline.read_text())["rows"] == _PUBLISHED
    assert main(["check", str(baseline), str(database)]) == 0


def test_check_fails_the_job_on_a_smaller_database(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    main(["capture", str(baseline), str(_database(tmp_path / "before.db", _PUBLISHED))])
    smaller = _database(tmp_path / "after.db", _REBUILT_FROM_EMPTY)
    assert main(["check", str(baseline), str(smaller)]) == 1
    errors = capsys.readouterr().err
    assert "::error::" in errors and "append-only" in errors


def test_an_unknown_command_explains_itself(capsys) -> None:
    assert main(["publish", "baseline.json"]) == 2
    assert "capture" in capsys.readouterr().err


def test_census_of_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        census(tmp_path / "absent.db")
