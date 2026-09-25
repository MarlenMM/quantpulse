"""Every route the app answers must be a real file, or the host says 404.

The SPA routes on the History API and GitHub Pages has no rewrite rule, so the
build used to copy `index.html` to `404.html` and let every deep link fall
through it. The page rendered correctly and the *status* was 404, because the
file being served was the not-found document. Nobody clicking a link notices;
a crawler will not index it, an unfurler will not preview it, and an uptime
check reads the site as down.

`scripts/emit_route_pages.py` writes a real page per route instead. These tests
guard the two ways that quietly stops being true: a route added to the app and
not to the emitter, and a stock page that loses its own title.
"""

import html
import json
import re
import struct
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from emit_route_pages import FIXED_ROUTES, FIXED_TITLES, OG_IMAGE, emit

REPO = Path(__file__).resolve().parents[2]
APP_TSX = REPO / "frontend" / "src" / "App.tsx"

SHELL = (
    '<!doctype html>\n<html lang="en">\n  <head>\n'
    "    <title>QuantPulse — S&amp;P 500 research</title>\n"
    '    <meta name="description" content="The S&amp;P 500 scored across seven categories." />\n'
    '  </head>\n  <body><div id="root"></div></body>\n</html>\n'
)


def _dist(tmp_path: Path, stocks: dict[str, dict] | None = None) -> Path:
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "index.html").write_text(SHELL)
    for symbol, payload in (stocks or {}).items():
        (dist / "data" / f"stocks__{symbol}.json").write_text(json.dumps(payload))
    return dist


class TestEveryAppRouteGetsAPage:
    def test_the_fixed_routes_match_the_app_switch(self) -> None:
        """Cross-language, because a route added on one side alone answers 404.

        `App.tsx` is the source of truth for what the app renders; this emitter
        decides what the *host* will answer 200 for. A route in the first and
        not the second still works -- it falls through `404.html` -- which is
        exactly the failure being fixed, and it would show up nowhere else.
        """
        source = APP_TSX.read_text()
        cases = set(re.findall(r'case "/([a-z-]+)":', source))
        # "/" is the bare root, served by `index.html` itself rather than by a
        # directory page, so it is not in `FIXED_ROUTES`.
        assert cases == set(FIXED_ROUTES), (
            f"App.tsx routes {sorted(cases)} but the emitter writes {sorted(FIXED_ROUTES)}; "
            f"the difference will be served as 404 with the app in the body"
        )

    def test_each_fixed_route_becomes_a_directory_index(self, tmp_path: Path) -> None:
        dist = _dist(tmp_path)
        emit(dist, quiet=True)
        for route in FIXED_ROUTES:
            assert (dist / route / "index.html").is_file(), f"/{route} has no page"

    def test_each_fixed_route_has_its_own_title(self, tmp_path: Path) -> None:
        """Finding 30: four pages, one tab title.

        Every fixed route used to be a verbatim copy of the shell, so the
        Screener, Track Record and Glossary tabs all read "QuantPulse — S&P 500
        research". The Dashboard -- the landing page -- keeps the site's title;
        the others name themselves. The SPA sets the same strings on client-side
        navigation, and the static suite checks the two agree in a browser.
        """
        dist = _dist(tmp_path)
        emit(dist, quiet=True)
        assert set(FIXED_TITLES) == set(FIXED_ROUTES)
        assert len(set(FIXED_TITLES.values())) == len(FIXED_TITLES), "two routes share a title"
        for route, title in FIXED_TITLES.items():
            page = (dist / route / "index.html").read_text()
            assert f"<title>{title.replace('&', '&amp;')}</title>" in page, route
        assert FIXED_TITLES["dashboard"] == "QuantPulse — S&P 500 research"

    def test_a_stock_page_exists_for_every_stock_payload(self, tmp_path: Path) -> None:
        """The two sets are derived from one another, so they cannot drift."""
        dist = _dist(
            tmp_path,
            {
                "NVDA": {"symbol": "NVDA", "summary": {"name": "Nvidia", "sector": "Tech"}},
                "BRK-B": {"symbol": "BRK-B", "summary": {"name": "Berkshire Hathaway"}},
            },
        )
        emit(dist, quiet=True)
        assert (dist / "stocks" / "NVDA" / "index.html").is_file()
        # A symbol with punctuation must land on the path the router builds.
        assert (dist / "stocks" / "BRK-B" / "index.html").is_file()

    def test_the_fallback_is_still_written(self, tmp_path: Path) -> None:
        """404.html remains -- for paths that genuinely are not found.

        It stops being the mechanism every deep link depends on and becomes what
        it should always have been: the answer for a typo or a delisted symbol.
        """
        dist = _dist(tmp_path)
        emit(dist, quiet=True)
        fallback = (dist / "404.html").read_text()
        # The shell, plus the site-level preview tags every page now carries.
        assert "<title>QuantPulse — S&amp;P 500 research</title>" in fallback
        assert '<div id="root"></div>' in fallback

    def test_it_refuses_to_run_before_the_client_is_built(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        with pytest.raises(FileNotFoundError, match="npm run build"):
            emit(dist, quiet=True)


class TestStockPagesCarryTheirOwnMetadata:
    """Otherwise 503 shared links preview identically, which is barely a preview.

    Nothing in the app sets `document.title`, so what the emitter writes is what
    a crawler and an unfurler read.
    """

    @staticmethod
    def _page(tmp_path: Path, payload: dict) -> str:
        dist = _dist(tmp_path, {"NVDA": payload})
        emit(dist, quiet=True)
        return (dist / "stocks" / "NVDA" / "index.html").read_text()

    def test_the_title_names_the_company(self, tmp_path: Path) -> None:
        page = self._page(
            tmp_path, {"symbol": "NVDA", "summary": {"name": "Nvidia", "sector": "Tech"}}
        )
        assert "<title>NVDA — Nvidia</title>" in page
        assert "QuantPulse — S&amp;P 500 research" not in page

    def test_the_description_is_specific_to_the_stock(self, tmp_path: Path) -> None:
        page = self._page(
            tmp_path,
            {
                "symbol": "NVDA",
                "summary": {"name": "Nvidia", "sector": "Information Technology"},
                "score": {"rating": "strong_buy"},
            },
        )
        description = re.search(r'name="description" content="([^"]*)"', page).group(1)
        assert "Nvidia (NVDA)" in description
        assert "Information Technology" in description
        assert "strong buy" in description
        assert "not financial advice" in description.lower()

    def test_a_stock_with_no_stored_name_still_gets_a_page(self, tmp_path: Path) -> None:
        """A catalogue-only symbol has no company name; the page is still valid."""
        page = self._page(tmp_path, {"symbol": "NVDA", "summary": {}})
        assert "<title>NVDA</title>" in page

    def test_metadata_is_html_escaped(self, tmp_path: Path) -> None:
        """A company name with an ampersand must not break the document."""
        page = self._page(
            tmp_path, {"symbol": "NVDA", "summary": {"name": 'Procter & Gamble "Co"'}}
        )
        assert "&amp;" in page
        assert 'Gamble "Co"' not in page, "an unescaped quote would end the attribute early"


SITE = "https://example.github.io/quantpulse/"


def _meta(page: str, key: str) -> str | None:
    """The content of `<meta property|name="key" content="...">`, unescaped."""
    match = re.search(rf'<meta (?:property|name)="{re.escape(key)}" content="([^"]*)"', page)
    return html.unescape(match.group(1)) if match else None


class TestSharedLinksPreview:
    """Finding 32: 508 real pages, and a pasted link arrived as a bare URL.

    Every page answered 200 with its own title and description, but carried no
    Open Graph or Twitter tags, so Slack, LinkedIn, Discord and iMessage had
    nothing to build a card from. Each page now states its title, description,
    canonical absolute URL and one preview image.
    """

    @staticmethod
    def _emit(tmp_path: Path) -> Path:
        dist = _dist(
            tmp_path,
            {
                "NVDA": {
                    "symbol": "NVDA",
                    "summary": {"name": "Nvidia", "sector": "Information Technology"},
                    "score": {"rating": "buy"},
                }
            },
        )
        (dist / "data" / "health.json").write_text(
            json.dumps({"freshness": {"prices": "2026-09-24"}})
        )
        emit(dist, quiet=True, site_url=SITE)
        return dist

    def _pages(self, dist: Path) -> dict[str, str]:
        pages = {"": dist / "index.html", "404": dist / "404.html"}
        pages |= {route: dist / route / "index.html" for route in FIXED_ROUTES}
        pages["stocks/NVDA"] = dist / "stocks" / "NVDA" / "index.html"
        return {route: path.read_text() for route, path in pages.items()}

    def test_every_page_carries_a_complete_card(self, tmp_path: Path) -> None:
        for route, page in self._pages(self._emit(tmp_path)).items():
            title = html.unescape(re.search(r"<title>([^<]*)</title>", page).group(1))
            description = _meta(page, "description")
            assert _meta(page, "og:title") == title, route
            assert _meta(page, "og:description") == description, route
            assert _meta(page, "og:type") == "website", route
            assert _meta(page, "og:site_name") == "QuantPulse", route
            assert _meta(page, "og:image") == SITE + OG_IMAGE, route
            assert _meta(page, "og:image:width") == "1200", route
            assert _meta(page, "og:image:height") == "630", route
            assert _meta(page, "og:image:alt"), route
            assert _meta(page, "twitter:card") == "summary_large_image", route
            # Exactly one of each: a second og:title from a re-run would leave
            # an unfurler to pick one.
            assert page.count('property="og:title"') == 1, route

    def test_each_page_names_its_own_canonical_url(self, tmp_path: Path) -> None:
        pages = self._pages(self._emit(tmp_path))
        assert _meta(pages["stocks/NVDA"], "og:url") == SITE + "stocks/NVDA/"
        assert _meta(pages["screener"], "og:url") == SITE + "screener/"
        assert _meta(pages[""], "og:url") == SITE
        assert f'<link rel="canonical" href="{SITE}stocks/NVDA/" />' in pages["stocks/NVDA"]
        # 404 is not a place; it gets the card but claims no URL of its own.
        assert _meta(pages["404"], "og:url") is None
        assert 'rel="canonical"' not in pages["404"]

    def test_a_stock_card_is_that_stock(self, tmp_path: Path) -> None:
        page = self._pages(self._emit(tmp_path))["stocks/NVDA"]
        assert _meta(page, "og:title") == "NVDA — Nvidia"
        assert "Nvidia (NVDA)" in _meta(page, "og:description")

    def test_running_twice_changes_nothing(self, tmp_path: Path) -> None:
        # The root index.html is both the shell and a page, so a naive second
        # run would stack a second set of tags on every page.
        dist = self._emit(tmp_path)
        before = {route: page for route, page in self._pages(dist).items()}
        emit(dist, quiet=True, site_url=SITE)
        assert self._pages(dist) == before

    def test_the_site_url_must_be_absolute(self, tmp_path: Path) -> None:
        dist = _dist(tmp_path)
        with pytest.raises(ValueError, match="absolute"):
            emit(dist, quiet=True, site_url="/quantpulse/")

    def test_the_preview_image_is_committed_at_the_size_the_tags_claim(self) -> None:
        image = REPO / "frontend" / "public" / OG_IMAGE
        header = image.read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG (cards do not accept SVG)"
        width, height = struct.unpack(">II", header[16:24])
        assert (width, height) == (1200, 630)


class TestSitemap:
    """Written from the same route list as the pages, so the two cannot drift."""

    def test_it_lists_every_page_and_nothing_else(self, tmp_path: Path) -> None:
        dist = TestSharedLinksPreview._emit(tmp_path)
        tree = ElementTree.parse(dist / "sitemap.xml")
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in tree.findall("s:url/s:loc", ns)]
        expected = [SITE] + [f"{SITE}{route}/" for route in FIXED_ROUTES] + [f"{SITE}stocks/NVDA/"]
        assert sorted(urls) == sorted(expected)
        lastmods = {el.text for el in tree.findall("s:url/s:lastmod", ns)}
        assert lastmods == {"2026-09-24"}, "lastmod is the data's newest price date"

    def test_no_robots_txt_is_written(self, tmp_path: Path) -> None:
        # Crawlers read robots.txt only at the host root. On a project site that
        # is another repository's URL, so a file here would be read by nothing.
        dist = TestSharedLinksPreview._emit(tmp_path)
        assert not (dist / "robots.txt").exists()


def test_the_pages_build_tells_the_emitter_where_the_site_lives() -> None:
    """Link previews need absolute URLs, and a fork's site lives somewhere else.

    `actions/configure-pages` reports the real `base_url`, so it has to run
    before the emitter and hand it over -- otherwise every card would point at
    the default, which is this repository's site even in a fork.
    """
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "pages.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    configure = next(i for i, s in enumerate(steps) if "configure-pages" in str(s.get("uses", "")))
    emit_step = next(
        i for i, s in enumerate(steps) if "emit_route_pages.py" in str(s.get("run", ""))
    )
    assert configure < emit_step, "configure-pages must run before the emitter"
    assert steps[configure]["id"] == "pages"
    assert '--site-url "${{ steps.pages.outputs.base_url }}"' in steps[emit_step]["run"]
