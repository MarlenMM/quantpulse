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

import json
import re
from pathlib import Path

import pytest

from emit_route_pages import FIXED_ROUTES, emit

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
        assert (dist / "404.html").read_text() == SHELL

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
