"""Refuse to publish a demo database that has lost rows since it was fetched.

**Why this exists, precisely.** `quantpulse_demo.db` is a rolling release asset
and the refresh workflow replaces it with `gh release upload --clobber`, which
is destructive by design. The only check standing in front of that upload was
`pragma integrity_check`, and on 2026-09-15 it passed while the published
database went from 417,993 price rows to 9,577, from 33,320 forecasts to none,
and from three years of history to four weeks: the refresh job had no step that
fetched the existing database, so `alembic upgrade head` created an empty one
and the run filled it in from nothing. **A well-formed database is not the same
as the right database**, and file size alone would not have caught it either --
the replacement was a perfectly reasonable 11.7 MB.

The invariant this leans on is the pipeline's own: every writer in
`storage/persistence.py` is append-only (Section 6.8), and recomputing a date
deletes-then-writes within one run. So across a whole run **no table may end
with fewer rows than it started with**. That is a stronger and much simpler
statement than any threshold on size, and it is exactly the property the
accident violated.

Two commands, because the two halves run in different steps of the same job:

    demo_db_guard.py capture BASELINE_JSON [DATABASE]
    demo_db_guard.py check   BASELINE_JSON [DATABASE]

`capture` runs immediately after the fetch, `check` immediately before the
upload. A **missing baseline is itself a failure**: it means the fetch step did
not run, which is the original bug wearing a different hat, so `check` refuses
rather than waving the upload through.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

DEFAULT_DATABASE = Path("quantpulse_demo.db")

#: Tables that hold no pipeline data and are legitimately empty or
#: visitor-local, so a change in them says nothing about a lost history. They
#: are still recorded in the census; they are only exempt from the comparison.
_NOT_PIPELINE_DATA = frozenset(
    {
        "alembic_version",
        "portfolio_holdings",
        "portfolio_transactions",
        "watchlist",
    }
)


def census(database: Path) -> dict[str, Any]:
    """Row counts per table, plus the file size, as a plain JSON-able dict."""
    if not database.exists():
        raise SystemExit(f"::error::{database} does not exist, so there is nothing to census")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table' "
                "and name not like 'sqlite_%' order by name"
            )
        ]
        rows = {
            table: int(connection.execute(f'select count(*) from "{table}"').fetchone()[0])
            for table in tables
        }
    finally:
        connection.close()
    return {"database": str(database), "bytes": database.stat().st_size, "rows": rows}


def compare(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Every table that lost rows, worst first. Empty means the upload is safe."""
    previous: dict[str, int] = before.get("rows", {})
    current: dict[str, int] = after.get("rows", {})
    losses: list[tuple[int, str]] = []
    for table, was in previous.items():
        if table in _NOT_PIPELINE_DATA:
            continue
        # A table that has disappeared entirely counts as losing all of it,
        # rather than being skipped for want of a row to compare against.
        now = current.get(table, 0)
        if now < was:
            losses.append((was - now, f"{table}: {was} rows before, {now} after (-{was - now})"))
    losses.sort(reverse=True)
    return [message for _, message in losses]


def _capture(baseline: Path, database: Path) -> int:
    snapshot = census(database)
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    total = sum(snapshot["rows"].values())
    tables = len(snapshot["rows"])
    print(f"census of {database}: {tables} tables, {total:,} rows -> {baseline}")
    return 0


def _check(baseline: Path, database: Path) -> int:
    if not baseline.exists():
        print(
            f"::error::no baseline at {baseline}. That means the database was never fetched, "
            "so this run started from an empty file -- refusing to publish it over the "
            "released one.",
            file=sys.stderr,
        )
        return 1
    before = json.loads(baseline.read_text())
    after = census(database)
    losses = compare(before, after)
    if losses:
        print(
            f"::error::{database} has fewer rows than the database this run started from; "
            "every writer is append-only, so this is data loss, not a refresh. "
            "Not publishing.",
            file=sys.stderr,
        )
        for message in losses:
            print(f"  {message}", file=sys.stderr)
        return 1
    grew = sum(after["rows"].values()) - sum(before["rows"].values())
    print(f"census check passed: no table shrank, {grew:+,} rows overall since the fetch")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2 or arguments[0] not in {"capture", "check"}:
        print(__doc__, file=sys.stderr)
        return 2
    command, baseline = arguments[0], Path(arguments[1])
    database = Path(arguments[2]) if len(arguments) > 2 else DEFAULT_DATABASE
    return _capture(baseline, database) if command == "capture" else _check(baseline, database)


if __name__ == "__main__":
    raise SystemExit(main())
