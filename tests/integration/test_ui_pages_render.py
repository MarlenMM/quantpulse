"""Every page renders without raising, against both an empty and a populated database.

The narrow safety net this file exists to provide: a page script is only
executed by its own feature test, so a page nobody added a feature to recently
(Settings, Glossary) had no page-level coverage at all, and an edit that broke
one would surface only when a human opened it. Streamlit swallows nothing --
an exception is a red traceback where the page should be -- so "it runs" is a
genuine assertion, not a tautology.

Both states matter. A freshly-cloned repo has an empty database and every page
is supposed to explain how to populate it rather than crash on an empty frame;
a populated one exercises the actual rendering paths. Most real page bugs found
in this project so far have been of exactly this shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage.models import (
    Base,
    CompositeScore,
    MarketRegime,
    PriceHistory,
    RefreshLog,
    Ticker,
)

APP_DIR = Path(__file__).resolve().parents[2] / "app"
PAGES = [
    str(APP_DIR / "Home.py"),
    *(str(path) for path in sorted((APP_DIR / "pages").glob("*.py"))),
]
AS_OF = date(2026, 7, 27)


def _seed(engine: Engine) -> None:
    rng = np.random.default_rng(0)
    with sessionmaker(bind=engine)() as session:
        for index, (symbol, sector) in enumerate(
            [
                ("AAA", "Information Technology"),
                ("BBB", "Energy"),
                ("CCC", "Financials"),
                ("DDD", "Health Care"),
            ]
        ):
            session.add(
                Ticker(
                    symbol=symbol,
                    name=f"{symbol} Inc",
                    sector=sector,
                    asset_type="equity",
                    is_active=True,
                )
            )
            level = 100.0
            for offset in range(300, 0, -1):
                level *= float(np.exp(rng.normal(0.0004, 0.014)))
                session.add(
                    PriceHistory(
                        symbol=symbol,
                        date=AS_OF - timedelta(days=offset),
                        open=level,
                        high=level * 1.01,
                        low=level * 0.99,
                        close=level,
                        adj_close=level,
                        volume=1_000_000,
                    )
                )
            session.add(
                CompositeScore(
                    symbol=symbol,
                    date=AS_OF,
                    profile="balanced",
                    composite_score=60.0 + index * 5,
                    percentile_rank=60.0 + index * 5,
                    rating="buy",
                    data_confidence=75.0,
                    **{f"{category}_score": 60.0 + index * 5 for category in CATEGORIES},
                    **{f"{category}_raw": 0.5 for category in CATEGORIES},
                )
            )
        session.add(
            MarketRegime(
                date=AS_OF,
                vix_level=18.0,
                breadth_pct_above_200dma=62.0,
                macro_news_tone=1.2,
                yield_curve_spread=0.4,
                regime_score=58.0,
                regime_label="neutral",
            )
        )
        session.add(
            RefreshLog(
                job_name="refresh_data",
                run_timestamp=__import__("datetime").datetime(2026, 7, 27, 22, 5),
                status="success",
                rows_updated=1234,
            )
        )
        session.commit()


@pytest.fixture
def empty_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def populated_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'full.db'}")
    Base.metadata.create_all(engine)
    _seed(engine)
    return engine


@contextmanager
def _wired(engine: Engine) -> Iterator[None]:
    factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_session() -> Iterator[object]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    from lib import data as lib_data

    # Every `@st.cache_data` reader, cleared so one test's rows aren't served to
    # the next. Dunder names are skipped deliberately: `dir()` includes
    # `__builtins__`, whose value is the builtins *dict* -- and a dict has a
    # `.clear`, so a naive "anything with .clear" sweep empties the interpreter's
    # builtins and the process dies with no output at all.
    for name in dir(lib_data):
        if name.startswith("_"):
            continue
        reader = getattr(lib_data, name)
        if callable(reader) and hasattr(reader, "clear"):
            reader.clear()
    with (
        patch("lib.data.get_session", fake_get_session),
        # The public demo runs in session mode; it is also the backend that
        # needs no filesystem, which keeps this test hermetic.
        patch("lib.data.portfolio_backend", return_value="session"),
    ):
        yield


def _run(page: str, engine: Engine) -> AppTest:
    with _wired(engine):
        at = AppTest.from_file(page, default_timeout=180)
        # No LLM configured: the optional narrative sections stay out of the way,
        # which is the configuration the deployed demo actually runs in.
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            at.run()
    return at


@pytest.mark.parametrize("page", PAGES, ids=lambda p: Path(p).stem)
def test_page_renders_against_an_empty_database(page: str, empty_engine: Engine) -> None:
    at = _run(page, empty_engine)
    assert not at.exception, f"{Path(page).name} raised on an empty database"
    # ...and says something, rather than rendering a blank screen that reads as
    # a broken app instead of an un-run pipeline.
    assert at.title or at.info or at.caption or at.markdown


@pytest.mark.parametrize("page", PAGES, ids=lambda p: Path(p).stem)
def test_page_renders_against_a_populated_database(page: str, populated_engine: Engine) -> None:
    at = _run(page, populated_engine)
    assert not at.exception, f"{Path(page).name} raised on a populated database"
