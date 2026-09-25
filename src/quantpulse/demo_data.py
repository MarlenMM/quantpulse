"""Fetch the demo database from its GitHub Release, unless it is already here.

The file is ~66 MB and used to be committed. Forty revisions of it accounted for
**309 MB of a 317 MB repository** -- 97.5%, against 8 MB for every source file,
test and document put together -- and it grew about 7.7 MB packed per refresh,
five refreshes a week. Git cannot delta-compress a SQLite file usefully, so each
night's copy is very nearly a whole new object. A release asset costs nothing
against repository size.

What a reader is promised is unchanged and paid for differently: one download on
first run, instead of every version of the file ever made arriving at
`git clone`.

**In the package rather than in `scripts/`, and that is the point.** Four
callers need this: `run.sh`, both GitHub workflows, and the Streamlit app.
Streamlit Community Cloud installs `requirements.txt` and runs
`streamlit run app/Home.py`, which puts only `app/` on `sys.path` -- `scripts/`
is not importable there, so a helper living in `scripts/` would have been
reachable by three of the four callers and silently unavailable to the one that
cannot fall back to a shell step. `app.lib` already appends `src/` to the path
for exactly this reason, so here it is importable everywhere.

**There are two sources, and the difference is deliberate.** `demo-data` is
rolling: the refresh replaces it five nights a week, and it is what the public
demo, the Pages build and `./run.sh` read, because those want the current data.
`ci-fixture` is pinned and verified by digest, and it is what the test job
reads. CI used to read the rolling one, so when the refresh published a gutted
database on 2026-09-15 three unrelated frontend pull requests went red on
`test_settings_reports_effective_weights_against_the_real_ranking` -- a
Streamlit test failing because of a file nobody in those pull requests had
touched, with nothing on screen to say so. A test job whose inputs change
underneath it is not testing the diff.

Current data is still exercised, where a failure means what it says: the publish
workflow's static-site gate runs against the rolling asset before the demo
updates, and it is what stopped the gutted database reaching the site.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

RELEASE_URL = (
    "https://github.com/MarlenMM/quantpulse/releases/download/demo-data/quantpulse_demo.db"
)
DEFAULT_TARGET = Path("quantpulse_demo.db")

#: The pinned copy the test job reads. Rolling it forward is a deliberate act:
#: upload a newer database to the `ci-fixture` release and put its digest below
#: in the same commit, so the change to what CI tests against is reviewable and
#: lands with a reason attached.
CI_FIXTURE_URL = (
    "https://github.com/MarlenMM/quantpulse/releases/download/ci-fixture/quantpulse_ci.db"
)
CI_FIXTURE_SHA256 = "58480632d3cf613b28fa626bd3799a31c4f2e26c95c324fc91dd7fd031f0fb28"

#: Anything smaller is an error page, not a database. Checked because a silent
#: HTML download surfaces much later as a corrupt-database error from SQLite,
#: pointing at everything except the download.
MIN_BYTES = 10 * 1024 * 1024

_SQLITE_MAGIC = b"SQLite format 3"


def fetch(
    target: Path = DEFAULT_TARGET,
    *,
    url: str = RELEASE_URL,
    expected_sha256: str | None = None,
    timeout: float = 120.0,
) -> bool:
    """Download the database to `target` if absent or empty. True if it downloaded.

    Written to a temporary file beside the target and moved into place only
    after every check passes. A half-written file at the real path would be
    picked up as "already here" by the next run and never repaired -- the
    failure mode that makes a download idempotent-looking and permanently
    broken.

    `expected_sha256` pins the content. Digest, not size: the database that
    caused this to be written was 11.7 MB of perfectly well-formed SQLite, and
    passed both checks below.
    """
    # Zero bytes counts as absent (point 44): it is what SQLite leaves when
    # something connects before the download, and it holds nothing to protect.
    # Anything larger is left alone -- it may be someone's real database.
    if target.exists() and target.stat().st_size > 0:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)

        size = partial.stat().st_size
        if size < MIN_BYTES:
            raise RuntimeError(
                f"downloaded only {size} bytes from {url} -- that is not the database"
            )
        with partial.open("rb") as handle:
            if handle.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
                raise RuntimeError(f"file from {url} is not a SQLite database")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise RuntimeError(
                f"the database at {url} is not the pinned one: expected sha256 "
                f"{expected_sha256}, got {digest.hexdigest()}. This says nothing about the "
                "code being tested -- either the pinned fixture was replaced without its "
                "digest being updated, or the download was corrupted."
            )

        partial.replace(target)
        return True
    finally:
        partial.unlink(missing_ok=True)


_MIGRATIONS = Path(__file__).resolve().parent / "storage" / "migrations"
_logger = logging.getLogger(__name__)


def ensure_schema_current(database_url: str) -> bool:
    """Upgrade a SQLite database Alembic manages to the current schema. True if it did.

    Point 45. The nightly migrates the demo database before it writes, but the
    Pages build and the hosted app read the release asset as downloaded -- so a
    new column (finding 34's were the first since the database became a release
    asset) made the reader select columns the file did not have yet. A warm
    Streamlit container never downloads again, so it would have kept failing
    after the nightly had published a migrated copy.

    Only a database with an `alembic_version` table is touched: one built some
    other way (a test's `create_all`) is its creator's business, and upgrading
    it would try to create tables that already exist. The Alembic `Config` is
    built without an ini file so `env.py` does not reconfigure the host's
    logging.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return False
    path = Path(database_url[len(prefix) :])
    if not path.exists() or path.stat().st_size == 0:
        return False

    import sqlite3

    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    with sqlite3.connect(path) as connection:
        managed = connection.execute(
            "select 1 from sqlite_master where type='table' and name='alembic_version'"
        ).fetchone()
        current = (
            connection.execute("select version_num from alembic_version").fetchone()
            if managed
            else None
        )
    if not managed:
        return False
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS))
    config.set_main_option("sqlalchemy.url", database_url)
    head = ScriptDirectory.from_config(config).get_current_head()
    if current is not None and current[0] == head:
        return False
    _logger.info("Migrating %s from %s to %s", path.name, current[0] if current else None, head)
    command.upgrade(config, "head")
    return True
