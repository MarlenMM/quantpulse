"""Shared helpers for the Streamlit app (Section 12).

Kept out of `pages/` because Streamlit turns every module in that directory
into a navigable page. Importable as `lib.<module>` from `Home.py` and from any
page, since Streamlit puts the entrypoint's directory on `sys.path`.

The split mirrors Section 14's layering rule in miniature: `data.py` is the only
module that touches the database (and only through `storage.persistence`, never
raw SQL), `charts.py` only arranges already-computed numbers into figures, and
`format.py` is pure strings. Nothing here computes analysis -- that all lives in
`src/quantpulse/`, which never imports from `app/`.

Importing this package also puts `src/` on `sys.path` if `quantpulse` is not
already importable -- see `_ensure_engine_importable` below for why the
deployed app needs that and a development checkout does not.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path


def _ensure_engine_importable() -> None:
    """Make `import quantpulse` work when the project itself was not installed.

    A development checkout runs through `uv run`, which installs this project
    into the environment, so `quantpulse` is a normal importable package and
    this function does nothing at all.

    A *deployed* app is different. Streamlit Community Cloud installs
    `requirements.txt` and then runs `streamlit run app/Home.py` -- it never
    installs the repository as a package, and `streamlit run` puts only the
    entrypoint's own directory (`app/`) on `sys.path`. The engine lives under
    `src/`, so without this every page dies on `ModuleNotFoundError: No module
    named 'quantpulse'` before rendering a single element. That is not
    hypothetical: it is exactly what the first deploy-shaped run produced, and
    nothing short of actually starting the server that way shows it -- the
    tests, the type checker and the linter all pass either way, because they
    all run inside an environment where the package *is* installed.

    Appended rather than inserted, so a real installation always wins and this
    can only ever act as a fallback.

    Every page reaches the engine through `lib`, and `lib` sorts before
    `quantpulse` in the import block that ruff's isort rule enforces, so this
    runs first. `tests/unit/test_app_bootstrap.py` asserts that ordering per
    page rather than leaving it to luck.
    """
    if find_spec("quantpulse") is not None:
        return
    source = Path(__file__).resolve().parents[2] / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.append(str(source))


_ensure_engine_importable()
