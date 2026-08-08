"""Keep `requirements.txt` pinned to the same versions as `uv.lock`.

`requirements.txt` exists only because Streamlit Community Cloud installs from
it (see the comment at the top of that file for why the deployed app gets a
smaller dependency set than the project). That makes it a second, hand-written
copy of version numbers `uv.lock` already owns -- and a second copy of a
version number is how the deployed app quietly ends up running something the
test suite never ran.

So the file is generated, not maintained: the package *names* below are the
deliberate part, and every version is read out of the lockfile.

    python scripts/sync_requirements.py            # rewrite requirements.txt
    python scripts/sync_requirements.py --check     # fail if it is out of date

`--check` runs in the test suite (`tests/unit/test_deploy_requirements.py`), so
a dependency bump that forgets to regenerate is caught in CI rather than at the
next deploy.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCKFILE = REPO / "uv.lock"
REQUIREMENTS = REPO / "requirements.txt"

# What the Streamlit app actually needs at runtime, in the order it is written
# out. Grouped roughly by role so the file reads as an explanation rather than
# an alphabetical dump.
#
# Deliberately absent: torch, transformers, spacy (the nightly job's news and
# sentiment models -- no page imports them, which `tests/support/
# render_one_page.py` asserts on every render), and yfinance / feedparser /
# praw / pyarrow-backed ingestion clients that only the nightly job calls.
APP_PACKAGES = (
    # UI
    "streamlit",
    "plotly",
    # Data handling
    "pandas",
    "numpy",
    "pyarrow",
    # Analysis
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pandas-ta-classic",
    "pyportfolioopt",
    "pandas-market-calendars",
    # Storage and configuration
    "sqlalchemy",
    "alembic",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    # HTTP -- the optional LLM providers and the on-demand SEC filing reader
    "requests",
    "lxml",
)

HEADER = """\
# Runtime dependencies for the *deployed Streamlit app only*.
#
# Why this file exists at all, when the project's real dependency source of
# truth is `pyproject.toml` + `uv.lock`: Streamlit Community Cloud installs from
# `requirements.txt`, and installing the full project there would fail. The
# nightly data-refresh job (`scripts/refresh_data.py`) needs torch,
# transformers and spaCy for the news and sentiment pipeline -- roughly 2.5 GB
# of wheels on Linux -- and the free tier has neither the disk nor the memory
# for them.
#
# The app never needs them: it reads rows the nightly already wrote and never
# runs a model. That is verified rather than assumed -- every page render in
# `tests/integration/test_ui_pages_real_data.py` asserts none of the three ends
# up in `sys.modules`, and it runs inside the full environment where all three
# are installed.
#
# GENERATED FILE -- do not edit the versions by hand. Every pin is read from
# `uv.lock`, so the deployed app runs exactly what the test suite ran against.
# Regenerate with `python scripts/sync_requirements.py`; CI fails if it is stale.
"""


def locked_versions() -> dict[str, str]:
    with LOCKFILE.open("rb") as handle:
        lock = tomllib.load(handle)
    return {package["name"]: package["version"] for package in lock["package"]}


def render() -> str:
    versions = locked_versions()
    missing = [name for name in APP_PACKAGES if name not in versions]
    if missing:
        raise SystemExit(f"not in uv.lock: {', '.join(missing)}")
    pins = "\n".join(f"{name}=={versions[name]}" for name in APP_PACKAGES)
    return f"{HEADER}\n{pins}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if requirements.txt does not match uv.lock",
    )
    args = parser.parse_args(argv)

    expected = render()
    current = REQUIREMENTS.read_text() if REQUIREMENTS.exists() else ""

    if args.check:
        if current != expected:
            print(
                "requirements.txt is out of date with uv.lock -- "
                "run `python scripts/sync_requirements.py`",
                file=sys.stderr,
            )
            return 1
        print("requirements.txt matches uv.lock")
        return 0

    REQUIREMENTS.write_text(expected)
    print(f"wrote {REQUIREMENTS.relative_to(REPO)} ({len(APP_PACKAGES)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
