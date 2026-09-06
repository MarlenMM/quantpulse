"""Give every deep link a real file, so a static host answers it with 200.

The SPA routes on the History API, and GitHub Pages has no rewrite rule. The
standard workaround is to copy `index.html` to `404.html`, which the Pages build
already did: a visitor opening `/quantpulse/stocks/NVDA` gets the app and the
page renders correctly. What they *also* get is **HTTP 404**, because the file
being served is the not-found document.

That status is invisible to a person and highly visible to everything else. A
crawler will not index a 404. Slack, Discord, iMessage and X will not unfurl one,
so a shared link to a stock arrives as bare text rather than a preview card. An
uptime check reports the site down. For a project whose distribution model is
"send someone the link", that is the wrong answer to give.

The fix does not need a rewrite rule -- it needs the files to exist. A static
host serves `/screener/index.html` for `/screener` with a 200, and this writes
one per route: the four fixed pages, and one per stock the screener can link to,
which is the same set `build_static_site.py` wrote a JSON payload for. A route
page exists here exactly when its data does, so the two cannot drift.

Two details worth stating:

* **`404.html` stays.** It is now the fallback for genuinely unknown paths -- a
  typo, or a symbol dropped from the index between deploys -- rather than the
  mechanism every deep link relies on. Those *should* answer 404.
* **Each stock page carries its own `<title>` and description.** Copying
  `index.html` unchanged would give every one of them "QuantPulse — S&P 500
  research", so 503 shared links would preview identically. Nothing in the app
  sets `document.title`, so what is written here is what a crawler and an
  unfurler read.

Run after `npm run build`, against the built `dist/`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

DEFAULT_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

#: The app's fixed routes. `App.tsx`'s switch is the source of truth; a route
#: added there and not here still works (via `404.html`) but answers 404, which
#: is the whole bug this script exists to fix -- so `test_route_pages.py` pins
#: the two lists against each other rather than trusting this comment.
FIXED_ROUTES: tuple[str, ...] = ("dashboard", "screener", "track-record", "glossary")

_TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)
_DESCRIPTION_RE = re.compile(r'(<meta\s+name="description"\s+content=")(.*?)(")', re.DOTALL)


def _with_metadata(shell: str, *, title: str, description: str) -> str:
    """`shell` with its title and description replaced, both HTML-escaped."""
    page = _TITLE_RE.sub(lambda _: f"<title>{html.escape(title)}</title>", shell, count=1)
    return _DESCRIPTION_RE.sub(
        lambda match: f"{match.group(1)}{html.escape(description)}{match.group(3)}",
        page,
        count=1,
    )


def _stock_metadata(payload: dict) -> tuple[str, str]:
    """The `<title>` and description for one stock's page."""
    symbol = str(payload.get("symbol", "")).upper()
    summary = payload.get("summary") or {}
    name = summary.get("name")
    score = payload.get("score") or {}
    rating = str(score.get("rating") or "").replace("_", " ").strip()

    title = f"{symbol} — {name}" if name else symbol
    described = f"{name} ({symbol})" if name else symbol
    sector = summary.get("sector")
    parts = [
        f"{described} scored across seven categories of public data"
        + (f", in {sector}" if sector else "")
    ]
    if rating:
        parts.append(f"Currently rated {rating}")
    parts.append("Educational research, not financial advice.")
    return title, ". ".join(parts) + ""


def emit(dist: Path, *, quiet: bool = False) -> list[Path]:
    """Write `404.html` and one `index.html` per route. Returns the files written."""
    shell_path = dist / "index.html"
    if not shell_path.exists():
        raise FileNotFoundError(
            f"no built client at {shell_path} -- run `npm run build` before this script"
        )
    shell = shell_path.read_text()
    written: list[Path] = []

    # The fallback for paths nothing enumerated. Those genuinely are not found,
    # so 404 is the correct answer for them and only for them.
    fallback = dist / "404.html"
    fallback.write_text(shell)
    written.append(fallback)

    for route in FIXED_ROUTES:
        target = dist / route / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(shell)
        written.append(target)

    data_dir = dist / "data"
    for payload_path in sorted(data_dir.glob("stocks__*.json")):
        payload = json.loads(payload_path.read_text())
        symbol = str(payload.get("symbol") or payload_path.stem.removeprefix("stocks__"))
        title, description = _stock_metadata(payload)
        target = dist / "stocks" / symbol / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_with_metadata(shell, title=title, description=description))
        written.append(target)

    if not quiet:
        total = sum(path.stat().st_size for path in written)
        print(f"wrote {len(written)} route pages, {total / 1_000_000:.1f} MB, under {dist}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    emit(args.dist, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
