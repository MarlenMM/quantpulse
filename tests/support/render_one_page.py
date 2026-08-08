"""Render exactly one Streamlit page against one database, then exit.

Invoked as a subprocess by `tests/integration/test_ui_pages_real_data.py`, one
process per page, on purpose: rendering several pages against the full-size
committed demo database inside a single interpreter reliably dies with
**SIGSEGV (exit 139)** and no Python traceback. That is a native-level crash in
this environment, not a fault in any page -- and because it looks exactly like
a page bug, isolating each render is what makes the difference between a real
signal and a confusing one.

Exit code 0 means the page rendered with no Streamlit exception; 1 means it
raised or rendered nothing worth showing. Diagnostics go to stdout for the
parent to attach to a failure message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Enough content to tell a rendered page from an early return. Every page shows
# a title plus either data or an explanatory empty state.
MIN_RENDERED_CHARS = 120

# The machine-learning stack belongs to `scripts/refresh_data.py` -- spaCy for
# entity extraction, BART for event classification, FinBERT for sentiment. The
# app only ever reads what that job already wrote, so no page may import any of
# it, and `requirements.txt` (what Streamlit Community Cloud installs) leaves
# all three out. Together they are roughly 2.5 GB of wheels on Linux, which is
# more disk and memory than the free tier has: if a page ever grows an import
# of one, the deploy stops working, and it stops working at install time on a
# host nobody is watching rather than here.
#
# Checked after the render rather than before, and by absence from `sys.modules`
# rather than by making the import fail -- that way this runs inside the *full*
# development environment, where all three are installed and an accidental
# import would otherwise succeed silently.
FORBIDDEN_MODULES = ("torch", "transformers", "spacy", "thinc")


def main() -> int:
    page_path, database = sys.argv[1], sys.argv[2]
    # Optional: a symbol to hold before rendering. An unscored ticker is
    # invisible to every other page -- they all read the scored universe -- so
    # a portfolio holding is the only way to force a page down the
    # short-history path at all.
    holding = sys.argv[3] if len(sys.argv) > 3 else None

    # `AppTest` does not put the script's own directory on `sys.path` the way
    # `streamlit run` does, so without this every page dies on `import lib` --
    # which reads as a page bug rather than a harness one.
    sys.path.insert(0, str(REPO / "app"))
    sys.path.insert(0, str(REPO / "src"))

    os.environ["DATABASE_URL"] = f"sqlite:///{database}"
    # What the deployed demo runs, and the only backend that needs no writable
    # file of its own.
    os.environ["PORTFOLIO_BACKEND"] = "session"

    from quantpulse.config import get_settings

    get_settings.cache_clear()

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(page_path, default_timeout=180)
    if holding:
        from datetime import date

        from quantpulse.portfolio import holdings as holdings_lib
        from quantpulse.portfolio.transactions import Transaction

        app.session_state["quantpulse_portfolio"] = holdings_lib.PortfolioState(
            transactions=[
                Transaction(
                    symbol=holding,
                    action="buy",
                    shares=100.0,
                    price=42.0,
                    date=date(2026, 8, 1),
                )
            ],
            cash=500.0,
        )
    app.run()

    if app.exception:
        for exception in app.exception:
            print(f"EXCEPTION: {exception.value!r}")
        return 1

    rendered = "".join(
        str(element.value)
        for group in (app.title, app.header, app.subheader, app.markdown, app.caption, app.info)
        for element in group
    )
    if len(rendered) < MIN_RENDERED_CHARS:
        print(f"EMPTY: rendered only {len(rendered)} characters of text")
        return 1

    imported = sorted(
        name
        for name in FORBIDDEN_MODULES
        if name in sys.modules or any(m.startswith(f"{name}.") for m in sys.modules)
    )
    if imported:
        print(
            f"HEAVY IMPORT: rendering this page imported {', '.join(imported)}, which the "
            "deployed app's requirements.txt deliberately omits"
        )
        return 1

    print(f"OK: {len(rendered)} chars, {len(app.metric)} metrics, {len(app.dataframe)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
