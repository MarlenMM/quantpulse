"""The deployed app can find the engine.

`app/lib/__init__.py` appends `src/` to `sys.path` when `quantpulse` is not
already installed, which is the situation on Streamlit Community Cloud: it
installs `requirements.txt` and runs `streamlit run app/Home.py` without ever
installing this repository as a package.

That bootstrap only helps if it runs *before* the first `quantpulse` import in
whichever page the visitor happens to land on, so this file pins the ordering
per page. It holds today because ruff's isort rule sorts `lib` before
`quantpulse` inside the first-party block -- but "the linter happens to sort it
that way" is a reason to assert it, not a reason to trust it. A page that
imported the engine first would fail only on the deployed host, only for a
visitor deep-linking to that page, and only after a green CI run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]


def _first_import_positions(source: str) -> tuple[int | None, int | None]:
    """Line numbers of the first top-level `lib` and `quantpulse` imports."""
    tree = ast.parse(source)
    lib = engine = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
        elif isinstance(node, ast.Import):
            root = node.names[0].name.split(".")[0]
        else:
            continue
        if root == "lib" and lib is None:
            lib = node.lineno
        elif root == "quantpulse" and engine is None:
            engine = node.lineno
    return lib, engine


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_pages_import_lib_before_the_engine(page: Path) -> None:
    lib, engine = _first_import_positions(page.read_text())
    assert lib is not None, (
        f"{page.name} does not import `lib` at all, so nothing puts `src/` on "
        "sys.path when the app is deployed without the package installed"
    )
    if engine is not None:
        assert lib < engine, (
            f"{page.name} imports `quantpulse` at line {engine}, before `lib` at "
            f"line {lib}. On a host that has not installed this project, that is a "
            "ModuleNotFoundError on page load."
        )


def test_bootstrap_is_a_no_op_when_the_package_is_installed() -> None:
    """It must never shadow a real installation -- only fill in for a missing one."""
    import sys

    from lib import _ensure_engine_importable

    before = list(sys.path)
    _ensure_engine_importable()
    assert sys.path == before
