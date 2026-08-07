"""Every Streamlit page renders against the *real* committed data, and against thin data.

`test_ui_pages_render.py` already renders all seven pages against an empty
database and a synthetic seed. This adds the two shapes that have actually
produced bugs in this project, neither of which a hand-built fixture reproduces:

* **The committed `quantpulse_demo.db`** -- the file the live demo serves, with
  503 real tickers, three years of prices, and *partial* category coverage
  (fundamentals, analyst and sentiment are empty until a weekly run completes).
  The Screener once crashed with `StreamlitDuplicateElementId` in exactly this
  state, and nothing caught it because every existing fixture seeded all seven
  categories.
* **A deliberately thin holding** -- a newly-added constituent with a handful of
  bars and no score at all, which is what a real index addition looks like the
  day it lands.

**One subprocess per page, and that is load-bearing.** Rendering several pages
against the full-size demo database inside one interpreter dies with SIGSEGV
(exit 139) and no traceback -- a native-level crash in this environment that is
indistinguishable from a page bug until you isolate it. `tests/support/
render_one_page.py` is the child; this file is the parent.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
RUNNER = REPO / "tests" / "support" / "render_one_page.py"
DEMO_DB = REPO / "quantpulse_demo.db"

PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]

pytestmark = pytest.mark.skipif(
    not DEMO_DB.exists(),
    reason="quantpulse_demo.db is not present; nothing real to render against",
)


def _render(
    page: Path, database: Path, holding: str | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUNNER), str(page), str(database)]
    if holding:
        command.append(holding)
    return subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=REPO)


@pytest.fixture(scope="module")
def real_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the committed demo database -- never the file itself."""
    target = tmp_path_factory.mktemp("real-data") / "demo.db"
    shutil.copy(DEMO_DB, target)
    return target


@pytest.fixture(scope="module")
def thin_database(real_database: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real database plus a newly-added ticker: priced, but not yet scored.

    This is the shape a real index addition takes on day one -- present in the
    universe, a few bars of history, no composite score, no forecast. Pages have
    to cope with a symbol that is in some reads and absent from others.
    """
    target = tmp_path_factory.mktemp("thin-data") / "demo.db"
    shutil.copy(real_database, target)

    connection = sqlite3.connect(target)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO tickers (symbol, name, sector, asset_type, is_active)"
            " VALUES ('NEWCO', 'NewCo Holdings', 'Industrials', 'equity', 1)"
        )
        last = date(2026, 8, 5)
        level = 42.0
        for offset in range(8, 0, -1):
            level *= 1.004
            connection.execute(
                "INSERT OR IGNORE INTO price_history"
                " (symbol, date, open, high, low, close, adj_close, volume)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "NEWCO",
                    (last - timedelta(days=offset)).isoformat(),
                    level,
                    level * 1.01,
                    level * 0.99,
                    level,
                    level,
                    500_000,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return target


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_against_the_committed_demo_database(page: Path, real_database: Path) -> None:
    result = _render(page, real_database)
    assert result.returncode == 0, (
        f"{page.name} failed against the real demo database\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )


def test_portfolio_renders_holding_only_a_newly_added_ticker(thin_database: Path) -> None:
    """The Portfolio page, holding nothing but an eight-bar unscored symbol.

    Only this page is parametrized on the thin shape, and that is a deliberate
    narrowing rather than an oversight. An unscored ticker is invisible to every
    other page -- they all read the scored universe -- so rendering them against
    the thin database produced byte-identical output to the real one. Seven
    duplicate tests that cannot fail differently are worse than one that can.

    Holding it is what forces the short-history path: volatility, Sharpe,
    Sortino, beta, VaR, all three optimizers and the correlation clustering each
    have to decline on eight observations and say why, rather than raising or
    publishing a number built from nothing.
    """
    page = APP / "pages" / "3_Portfolio.py"
    result = _render(page, thin_database, holding="NEWCO")
    assert result.returncode == 0, (
        "Portfolio failed while holding a newly-added, unscored, 8-bar ticker\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
