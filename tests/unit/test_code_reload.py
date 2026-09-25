"""The hosted app serves the code that was pushed, without a reboot (point 46).

Streamlit Community Cloud updates the files of a running app on every push and
keeps the process. Streamlit drops changed modules only for sessions that are
connected when the files change; a push that lands between visits leaves the
old `lib.*` and `quantpulse.*` modules loaded. Measured on the hosted app on
2026-09-25: after the finding-34 push, Stock Detail raised `ImportError: cannot
import name 'format_edge_cell' from 'lib.format'` -- the page was new, the
module it imported was not -- until the app was rebooted.

`lib/code_reload.py` is every page's first import: it drops any repository
module whose source changed since the process loaded it, and removes itself
from `sys.modules` so it runs on every page load.
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A package `fakepkg` with one module, importable from tmp_path."""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield pkg
    for name in [n for n in sys.modules if n == "fakepkg" or n.startswith("fakepkg.")]:
        del sys.modules[name]


def test_a_changed_module_is_dropped_and_reloads_fresh(fake_tree: Path) -> None:
    import fakepkg.util

    from lib import code_reload

    snapshot = code_reload.snapshot([fake_tree])
    assert fakepkg.util.VALUE == 1

    (fake_tree / "util.py").write_text("VALUE = 2\n")
    later = time.time() + 5
    os.utime(fake_tree / "util.py", (later, later))

    dropped, snapshot = code_reload.drop_stale_modules([fake_tree], snapshot)
    assert dropped == ["fakepkg.util"]
    import fakepkg.util as reloaded

    assert reloaded.VALUE == 2
    # And the package no longer hands out the old module as an attribute --
    # `from fakepkg import util` reads the attribute before sys.modules.
    from fakepkg import util

    assert util.VALUE == 2


def test_nothing_changed_nothing_dropped(fake_tree: Path) -> None:
    import fakepkg.util  # noqa: F401

    from lib import code_reload

    snapshot = code_reload.snapshot([fake_tree])
    dropped, _ = code_reload.drop_stale_modules([fake_tree], snapshot)
    assert dropped == []
    assert "fakepkg.util" in sys.modules


def test_the_first_call_records_and_later_calls_compare(monkeypatch, fake_tree: Path) -> None:
    import fakepkg.util  # noqa: F401

    from lib import code_reload

    monkeypatch.setattr(code_reload, "ROOTS", (fake_tree,))
    monkeypatch.setattr(code_reload, "_last", None)
    assert code_reload.refresh_changed_code() == []
    later = time.time() + 5
    os.utime(fake_tree / "util.py", (later, later))
    assert code_reload.refresh_changed_code() == ["fakepkg.util"]
    assert code_reload.refresh_changed_code() == []


PAGES = sorted([APP / "Home.py", *APP.glob("pages/*.py")])


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_refreshes_before_anything_else_from_lib(page: Path) -> None:
    """The call has to come before the page's other lib imports, or they are stale."""
    body = ast.parse(page.read_text()).body
    lib_statements = [
        (i, node)
        for i, node in enumerate(body)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lib"))
        or (isinstance(node, ast.Import) and any(a.name.startswith("lib") for a in node.names))
    ]
    assert lib_statements, f"{page.name} imports nothing from lib"
    first_index, first = lib_statements[0]
    assert isinstance(first, ast.ImportFrom) and first.module == "lib.code_reload", (
        f"{page.name}: the first lib import must be `from lib.code_reload import "
        "refresh_changed_code`"
    )
    call = body[first_index + 1]
    assert (
        isinstance(call, ast.Expr)
        and isinstance(call.value, ast.Call)
        and getattr(call.value.func, "id", None) == "refresh_changed_code"
    ), f"{page.name}: refresh_changed_code() must be called right after importing it"
