"""Finding 38: every count the docs state is derived here and compared.

Four review passes re-counted these by hand and each time they had drifted --
migrations were 16 on disk while the README said 13, ARCHITECTURE 12 and the
backlog 14; by 2026-10-02 there were 18, against 23 documented tables for 26,
14 endpoints for 15, 71 glossary terms for 74 and 1,554 tests for 2,051.
Hand-counting is the wrong mechanism, so the counts are computed from the code
and every mention in the living docs is checked against them.

Two rules keep the guard honest:

- **Each pattern must match somewhere.** A regex that silently matches nothing
  passes forever; so every kind of count names the files it must be found in.
- **Fast-moving counts are lower bounds** ("2,000+"), checked against the
  hundred or thousand they sit in, so the doc changes when the number crosses a
  boundary rather than on every commit. The test count is only checked when the
  whole suite was collected -- a `-k` run sees a fraction of it.

The dated records (`docs/SESSION_HANDOFF_*.md`, the backlog's numbered points)
are history and are not checked: "1,895 passed" was true on the day it says.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The living docs. The backlog is only its header and §1 -- the rest is a
#: dated record.
LIVING = ("README.md", "ARCHITECTURE.md", "HOW_TO_USE.md", "docs/IMPROVEMENT_BACKLOG.md#1")


def _text(name: str) -> str:
    path, _, section = name.partition("#")
    text = (ROOT / path).read_text()
    if section == "1":
        text = text.split("\n## 2.", 1)[0]
    return text


def _tables() -> int:
    from quantpulse.storage.models import Base

    return len(Base.metadata.tables)


def _migrations() -> int:
    return len(list((ROOT / "src/quantpulse/storage/migrations/versions").glob("*.py")))


def _endpoints() -> int:
    from quantpulse.api.main import app

    return sum(1 for route in app.routes if getattr(route, "path", "").startswith("/api/"))


def _streamlit_pages() -> int:
    return 1 + len(list((ROOT / "app/pages").glob("*.py")))


def _react_pages() -> int:
    return len(list((ROOT / "frontend/src/pages").glob("*.tsx")))


def _glossary_terms() -> int:
    from quantpulse.glossary import TERMS

    return len(TERMS)


def _glossary_categories() -> int:
    from quantpulse.glossary import TERMS

    return len({category for category, _ in TERMS.values()})


def _engine_lines() -> int:
    return sum(
        len(path.read_text().splitlines()) for path in (ROOT / "src/quantpulse").rglob("*.py")
    )


def _number(text: str) -> int:
    return int(text.replace(",", ""))


#: kind -> (derive, regex whose group 1 is the documented figure, files it must appear in)
EXACT: dict[str, tuple[Callable[[], int], str, tuple[str, ...]]] = {
    "tables": (
        _tables,
        r"\b(\d+)\*{0,2} tables\b",
        ("README.md", "ARCHITECTURE.md", "docs/IMPROVEMENT_BACKLOG.md#1"),
    ),
    "migrations": (
        _migrations,
        r"\b(\d+)\*{0,2} (?:Alembic )?(?:migrations|revisions)\b",
        ("README.md", "ARCHITECTURE.md"),
    ),
    "endpoints": (_endpoints, r"\b(\d+)[- ]endpoints?\b", ("README.md", "ARCHITECTURE.md")),
    "Streamlit pages": (
        _streamlit_pages,
        r"\b(\d+)[- ]page Streamlit\b|Streamlit \(`app/`, (\d+) pages",
        ("README.md", "docs/IMPROVEMENT_BACKLOG.md#1"),
    ),
    "React pages": (
        _react_pages,
        r"\b(\d+)[- ]page React\b|React\+TS \(`frontend/`, (\d+) pages|SPA \(Vite\), (\d+) pages",
        ("README.md", "ARCHITECTURE.md", "docs/IMPROVEMENT_BACKLOG.md#1"),
    ),
    "glossary terms": (
        _glossary_terms,
        r"Glossary terms \| \*\*(\d+)\*\*|\b(\d+) (?:term )?definitions\b",
        ("README.md", "ARCHITECTURE.md", "HOW_TO_USE.md"),
    ),
    "glossary categories": (
        _glossary_categories,
        r"Glossary terms \| \*\*\d+\*\*, across (\d+) categories",
        ("README.md",),
    ),
}


def _mentions(pattern: str, text: str) -> list[int]:
    return [
        _number(next(group for group in match.groups() if group))
        for match in re.finditer(pattern, text)
    ]


@pytest.mark.parametrize("kind", list(EXACT))
def test_every_documented_count_matches_the_code(kind: str) -> None:
    derive, pattern, required = EXACT[kind]
    actual = derive()
    wrong: list[str] = []
    for name in LIVING:
        found = _mentions(pattern, _text(name))
        if name in required and not found:
            wrong.append(f"{name}: no mention of the {kind} count found -- the pattern is stale")
        wrong += [
            f"{name} says {value} {kind}; the code has {actual}"
            for value in found
            if value != actual
        ]
    assert not wrong, "\n".join(wrong)


def test_engine_size_is_stated_as_the_thousand_it_is_in() -> None:
    lines = _engine_lines()
    floor = lines // 1000 * 1000
    found = _mentions(r"Core engine code \| \*\*([\d,]+)\+\*\* lines", _text("README.md"))
    assert found, "README's 'Core engine code' row is missing or no longer reads **N+** lines"
    assert found == [floor], (
        f"README says {found[0]:,}+ lines; src/quantpulse has {lines:,}: write {floor:,}+"
    )


def test_test_count_is_stated_as_the_hundred_it_is_in(request: pytest.FixtureRequest) -> None:
    collected = getattr(request.config, "_quantpulse_whole_suite", None)
    if collected is None:
        pytest.skip("only meaningful when the whole suite is collected (CI and a bare `pytest`)")
    floor = collected // 100 * 100
    found = _mentions(r"Automated tests \| \*\*([\d,]+)\+\*\*", _text("README.md"))
    assert found, "README's 'Automated tests' row is missing or no longer reads **N+**"
    assert found == [floor], (
        f"README says {found[0]:,}+ tests; {collected:,} are collected: write {floor:,}+"
    )
