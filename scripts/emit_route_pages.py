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
  sets `document.title` on a hard load before the app has fetched anything, so
  what is written here is what a crawler and an unfurler read. The app sets
  the same strings on client-side navigation (`frontend/src/lib/title.ts`).

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
FIXED_ROUTES: tuple[str, ...] = ("dashboard", "screener", "portfolio", "track-record", "glossary")

#: Each fixed route's `<title>` (finding 30). The Dashboard is the landing page
#: and keeps the site's own title; the others name themselves, so four tabs are
#: not four identical tabs. `frontend/src/lib/title.ts` sets the same strings on
#: client-side navigation, and the static-site suite checks, in a browser, that
#: the two agree -- this and that file are the only two copies.
FIXED_TITLES: dict[str, str] = {
    "dashboard": "QuantPulse — S&P 500 research",
    "screener": "Screener — QuantPulse",
    "portfolio": "Portfolio — QuantPulse",
    "track-record": "Track Record — QuantPulse",
    "glossary": "Glossary — QuantPulse",
}

#: Where the published site lives. The Pages workflow passes the real one
#: (`actions/configure-pages`' `base_url`), so a fork advertises its own; this
#: default is for local runs.
DEFAULT_SITE_URL = "https://marlenmm.github.io/quantpulse/"

#: The one preview image every page's card uses (finding 32), in
#: `frontend/public/` so Vite copies it to the site root. PNG because link
#: unfurlers do not render SVG; 1200x630 because that is the size they crop to.
OG_IMAGE = "og-image.png"
_OG_IMAGE_ALT = "The QuantPulse mark — three candles stepping up — and the name"

_TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)
#: Tags this script writes, so a second run replaces rather than repeats them:
#: the root `index.html` is both the shell every page is read from and a page.
_OWN_TAGS_RE = re.compile(
    r'[ \t]*<(?:meta (?:property="og:[^"]*"|name="twitter:[^"]*")|link rel="canonical")[^>]*/>\n?'
)
_DESCRIPTION_RE = re.compile(r'(<meta\s+name="description"\s+content=")(.*?)(")', re.DOTALL)


def _description(shell: str) -> str:
    match = _DESCRIPTION_RE.search(shell)
    return html.unescape(match.group(2)) if match else ""


def _page(shell: str, *, site_url: str, path: str | None, title: str, description: str) -> str:
    """`shell` with its title, description and link-preview tags set.

    `path` is the page's location under the site ("stocks/NVDA/"), or `None`
    for `404.html`, which gets the card but claims no URL of its own.
    """
    page = _OWN_TAGS_RE.sub("", shell)
    page = _TITLE_RE.sub(lambda _: f"<title>{html.escape(title)}</title>", page, count=1)
    page = _DESCRIPTION_RE.sub(
        lambda match: f"{match.group(1)}{html.escape(description)}{match.group(3)}",
        page,
        count=1,
    )
    tags = {
        "og:type": "website",
        "og:site_name": "QuantPulse",
        "og:title": title,
        "og:description": description,
        "og:image": site_url + OG_IMAGE,
        "og:image:width": "1200",
        "og:image:height": "630",
        "og:image:alt": _OG_IMAGE_ALT,
    }
    if path is not None:
        tags["og:url"] = site_url + path
    lines = [
        f'<meta property="{key}" content="{html.escape(value)}" />' for key, value in tags.items()
    ]
    lines.append('<meta name="twitter:card" content="summary_large_image" />')
    if path is not None:
        lines.append(f'<link rel="canonical" href="{html.escape(site_url + path)}" />')
    block = "".join(f"    {line}\n" for line in lines)
    return re.sub(r"[ \t]*</head>", lambda _: f"{block}  </head>", page, count=1)


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


def _sitemap(site_url: str, paths: list[str], lastmod: str | None) -> str:
    entries = []
    for path in paths:
        stamp = f"<lastmod>{lastmod}</lastmod>" if lastmod else ""
        entries.append(f"  <url><loc>{html.escape(site_url + path)}</loc>{stamp}</url>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(entries)
        + "\n</urlset>\n"
    )


def emit(dist: Path, *, quiet: bool = False, site_url: str = DEFAULT_SITE_URL) -> list[Path]:
    """Write every route's page, `404.html` and `sitemap.xml`. Returns the files written.

    **No `robots.txt`** (finding 32 proposed one): crawlers read it only at the
    host root, and on a project site that is `<owner>.github.io/robots.txt` --
    another repository's URL. A file here would be read by nothing. Without one,
    crawlers allow everything already; the sitemap can be submitted directly.
    """
    if not site_url.startswith(("https://", "http://")):
        raise ValueError(f"site_url must be absolute (link previews need it), got {site_url!r}")
    site_url = site_url if site_url.endswith("/") else f"{site_url}/"
    shell_path = dist / "index.html"
    if not shell_path.exists():
        raise FileNotFoundError(
            f"no built client at {shell_path} -- run `npm run build` before this script"
        )
    shell = shell_path.read_text()
    site_description = _description(shell)
    written: list[Path] = []
    paths: list[str] = []

    def write(target: Path, path: str | None, title: str, description: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        page = _page(shell, site_url=site_url, path=path, title=title, description=description)
        target.write_text(page)
        written.append(target)
        if path is not None:
            paths.append(path)

    # The fallback for paths nothing enumerated. Those genuinely are not found,
    # so 404 is the correct answer for them and only for them.
    write(dist / "404.html", None, FIXED_TITLES["dashboard"], site_description)
    for route in FIXED_ROUTES:
        write(dist / route / "index.html", f"{route}/", FIXED_TITLES[route], site_description)

    data_dir = dist / "data"
    for payload_path in sorted(data_dir.glob("stocks__*.json")):
        payload = json.loads(payload_path.read_text())
        symbol = str(payload.get("symbol") or payload_path.stem.removeprefix("stocks__"))
        title, description = _stock_metadata(payload)
        write(dist / "stocks" / symbol / "index.html", f"stocks/{symbol}/", title, description)

    # The root last: it is the shell every page above was read from.
    write(shell_path, "", FIXED_TITLES["dashboard"], site_description)

    health_path = data_dir / "health.json"
    lastmod = None
    if health_path.exists():
        lastmod = (json.loads(health_path.read_text()).get("freshness") or {}).get("prices")
    sitemap = dist / "sitemap.xml"
    sitemap.write_text(_sitemap(site_url, ["", *sorted(p for p in paths if p)], lastmod))
    written.append(sitemap)

    if not quiet:
        total = sum(path.stat().st_size for path in written)
        print(
            f"wrote {len(paths)} pages, 404.html and sitemap.xml, "
            f"{total / 1_000_000:.1f} MB, under {dist}"
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--site-url",
        default=DEFAULT_SITE_URL,
        help="the published site's absolute URL; the Pages workflow passes its own",
    )
    args = parser.parse_args(argv)
    emit(args.dist, quiet=args.quiet, site_url=args.site_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
