"""The analysis engine never imports a front end, the API or the LLM layer.

ARCHITECTURE.md's one rule: `analysis/`, `portfolio/` and `news_intelligence/`
take DataFrames and dataclasses in and return them out, which is what lets the
Streamlit app and the React SPA share every number. A single `import streamlit`
in a scorer would tie the engine to one front end without any other test
noticing, so the rule is checked here on every module's import statements,
including the ones under `TYPE_CHECKING`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "quantpulse"

ENGINE = ("analysis", "portfolio", "news_intelligence")

#: Top-level packages and project modules the engine must not reach.
FORBIDDEN = (
    "streamlit",
    "fastapi",
    "starlette",
    "uvicorn",
    "app",
    "quantpulse.api",
    "quantpulse.llm",
)


def _engine_modules() -> list[Path]:
    return sorted(path for package in ENGINE for path in (SRC / package).rglob("*.py"))


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _is_forbidden(name: str) -> bool:
    return any(name == banned or name.startswith(banned + ".") for banned in FORBIDDEN)


def test_the_engine_has_modules_to_check() -> None:
    # Not vacuous: a moved package would otherwise pass with nothing scanned.
    assert len(_engine_modules()) > 20


@pytest.mark.parametrize("path", _engine_modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_engine_module_imports_no_front_end(path: Path) -> None:
    bad = sorted(name for name in _imports(path) if _is_forbidden(name))
    assert not bad, f"{path.relative_to(SRC)} imports {bad}"


def test_the_check_catches_a_front_end_import() -> None:
    assert _is_forbidden("streamlit")
    assert _is_forbidden("quantpulse.api.main")
    assert not _is_forbidden("quantpulse.analysis.scoring")
    assert not _is_forbidden("apply")  # a prefix, not the package
