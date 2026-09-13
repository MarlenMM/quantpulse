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
"""

from __future__ import annotations

from pathlib import Path

import requests

RELEASE_URL = (
    "https://github.com/MarlenMM/quantpulse/releases/download/demo-data/quantpulse_demo.db"
)
DEFAULT_TARGET = Path("quantpulse_demo.db")

#: Anything smaller is an error page, not a database. Checked because a silent
#: HTML download surfaces much later as a corrupt-database error from SQLite,
#: pointing at everything except the download.
MIN_BYTES = 10 * 1024 * 1024

_SQLITE_MAGIC = b"SQLite format 3"


def fetch(target: Path = DEFAULT_TARGET, *, timeout: float = 120.0) -> bool:
    """Download the database to `target` if absent. True if it downloaded.

    Written to a temporary file beside the target and moved into place only
    after both checks pass. A half-written file at the real path would be
    picked up as "already here" by the next run and never repaired -- the
    failure mode that makes a download idempotent-looking and permanently
    broken.
    """
    if target.exists():
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with requests.get(RELEASE_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)

        size = partial.stat().st_size
        if size < MIN_BYTES:
            raise RuntimeError(
                f"downloaded only {size} bytes from {RELEASE_URL} -- that is not the database"
            )
        with partial.open("rb") as handle:
            if handle.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
                raise RuntimeError(f"file from {RELEASE_URL} is not a SQLite database")

        partial.replace(target)
        return True
    finally:
        partial.unlink(missing_ok=True)
