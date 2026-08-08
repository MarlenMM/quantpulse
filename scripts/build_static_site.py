"""Pre-render the read API to static JSON, so the SPA can be hosted for free.

GitHub Pages serves files. It cannot run Python, so it cannot run the FastAPI
app -- but the app is read-only by design and the data behind it changes once a
night, which makes every response it can produce a file waiting to be written.

**The files are the API's own output, not a second implementation.** This runs
the real `quantpulse.api.main:app` through Starlette's `TestClient` against the
committed demo database and writes the response bodies verbatim. There is no
place here for the two to disagree about a number, because there is only one
implementation of every number.

The naming rule -- parameters folded into the filename, sorted -- exists twice,
here and as `staticPath()` in `frontend/src/lib/api.ts`, because a query string
cannot address a static file and the two sides never meet. That duplication is
the one real risk in this design, so it is checked from both ends:
`tests/unit/test_static_site_layout.py` pins which requests get generated at
all, and `.github/workflows/pages.yml` loads the finished bundle in a browser
and reads real numbers off it before publishing. If the two spellings ever
disagreed, every request would 404 and that second check fails.

    python scripts/build_static_site.py --out frontend/public/data

Then build the client with `VITE_STATIC_API=1` so it reads the files instead of
calling a server. `.github/workflows/pages.yml` does both and publishes the
result.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = REPO / "quantpulse_demo.db"
DEFAULT_OUT = REPO / "frontend" / "public" / "data"

# Every request the SPA can make, as (path, params).
#
# The literal limits mirror the client's call sites -- `api.regime(90)`,
# `api.news(6)` and so on. A file that does not exist is a 404 on the live site,
# so `tests/unit/test_static_site_layout.py` parses those call sites out of the
# `.tsx` files and asserts each one is covered here. Change a limit in the
# client and that test tells you to change it here too.
#
# Per-symbol stock pages and the per-profile screener views are expanded at
# build time from the database, so adding an index constituent or an investor
# profile needs no edit to this list.
FIXED_REQUESTS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("/health", None),
    ("/glossary", None),
    ("/universe", None),
    ("/profiles", None),
    ("/screener/changes", {"limit": 8}),
    ("/regime", {"limit": 90}),
    ("/news", {"limit": 6}),
    ("/backtest", {"limit": 20}),
    ("/sectors/rotation", {"lookback_days": 21}),
)

PROFILE_REQUESTS: tuple[str, ...] = ("/screener", "/screener/absolute")


def static_path(path: str, params: dict[str, Any] | None = None) -> str:
    """Filename for a pre-rendered response.

    Mirrors `staticPath()` in `frontend/src/lib/api.ts`. Kept deliberately
    simple -- a query string cannot address a static file, so the parameters
    become part of the name, sorted so one request always names one file.
    """
    name = path.lstrip("/").replace("/", "__")
    suffix = "".join(f"__{key}-{params[key]}" for key in sorted(params)) if params else ""
    return f"{name}{suffix}.json"


def build(out_dir: Path, database: Path, *, quiet: bool = False) -> list[Path]:
    """Write every response to `out_dir` and return the files written."""
    # The API's engine is bound at import time from settings, so this has to
    # happen before `quantpulse.api.main` is imported.
    os.environ["DATABASE_URL"] = f"sqlite:///{database}"
    os.environ.setdefault("PORTFOLIO_BACKEND", "session")

    from quantpulse.config import get_settings

    get_settings.cache_clear()

    from starlette.testclient import TestClient

    from quantpulse.analysis.investor_profiles import profile_names
    from quantpulse.api.main import app

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    written: list[Path] = []

    with TestClient(app) as client:

        def emit(path: str, params: dict[str, Any] | None = None) -> Any:
            response = client.get(f"/api{path}", params=params)
            response.raise_for_status()
            target = out_dir / static_path(path, params)
            target.write_bytes(response.content)
            written.append(target)
            return response.json()

        for path, params in FIXED_REQUESTS:
            emit(path, params)

        symbols: list[str] = []
        for profile in profile_names():
            for path in PROFILE_REQUESTS:
                payload = emit(path, {"profile": profile})
                if path == "/screener":
                    symbols = sorted(
                        {row["symbol"] for row in payload.get("rows", [])} | set(symbols)
                    )

        # Every stock the screener can link to. A symbol with no page is a 404
        # in the browser, so this follows the table rather than a fixed list.
        for index, symbol in enumerate(symbols, start=1):
            emit(f"/stocks/{symbol}")
            if not quiet and index % 100 == 0:
                print(f"  ... {index}/{len(symbols)} stock pages")

    if not quiet:
        total = sum(path.stat().st_size for path in written)
        print(f"wrote {len(written)} files, {total / 1_000_000:.1f} MB, to {out_dir}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.database.exists():
        parser.error(f"no database at {args.database}")
    build(args.out, args.database.resolve(), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
