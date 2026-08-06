"""The Portfolio page's beta must be measured against the market, not the portfolio.

`render_risk` built its market proxy with
`risk.equal_weight_market_returns(panel)`, where `panel` held **only the
symbols the portfolio owns**. That regresses the portfolio against an
equal-weight version of itself. For an equal-weight portfolio the answer is
beta 1.0000 with R-squared 1.0000 *exactly*, and the page printed it under a
caption reading "measured against an equal-weight proxy for the market" beside
a glossary entry saying "Beta 1 tracks the market".

Measured on five real S&P 500 names from the committed demo database: 1.0000
(R-squared 1.0000) the old way, 0.448 (R-squared 0.238) against the actual
universe. An R-squared of exactly 1 against "the market" is not a number any
real regression produces, which is what makes this checkable rather than a
matter of taste.

The scenario below separates the two readings by construction: the portfolio
holds three names driven by one random factor, while the universe contains six
more driven by an independent one. Against its own holdings the portfolio is
explained almost perfectly; against the real universe it is barely explained at
all.
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
from sqlalchemy.orm import Session, sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.portfolio.transactions import Transaction
from quantpulse.storage.models import Base, CompositeScore, PriceHistory, Ticker

APP_DIR = Path(__file__).resolve().parents[2] / "app"
PORTFOLIO_PAGE = str(APP_DIR / "pages" / "3_Portfolio.py")
AS_OF = date(2026, 7, 27)

HELD = ["AAA", "BBB", "CCC"]
UNIVERSE_ONLY = ["DDD", "EEE", "FFF", "GGG", "HHH", "III"]


def _factor_series(session: Session, symbol: str, factor: np.ndarray, seed: int) -> None:
    """A price path that is mostly `factor` plus a little idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    level = 100.0
    for offset, shock in zip(range(len(factor), 0, -1), factor, strict=True):
        level *= float(np.exp(shock + rng.normal(0.0, 0.002)))
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


@pytest.fixture
def two_factor_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'beta.db'}")
    Base.metadata.create_all(engine)
    rng = np.random.default_rng(7)
    days = 400
    held_factor = rng.normal(0.0004, 0.014, days)
    # Independent of the first, so a proxy built from the *universe* explains
    # the portfolio poorly while one built from its own holdings explains it
    # almost perfectly.
    other_factor = rng.normal(0.0004, 0.014, days)

    with sessionmaker(bind=engine)() as session:
        for index, symbol in enumerate(HELD + UNIVERSE_ONLY):
            session.add(
                Ticker(
                    symbol=symbol,
                    name=f"{symbol} Inc",
                    sector="Information Technology",
                    asset_type="equity",
                    is_active=True,
                )
            )
            factor = held_factor if symbol in HELD else other_factor
            _factor_series(session, symbol, factor, seed=index)
            session.add(
                CompositeScore(
                    symbol=symbol,
                    date=AS_OF,
                    profile="balanced",
                    composite_score=50.0 + index,
                    percentile_rank=50.0 + index,
                    rating="hold",
                    data_confidence=80.0,
                    **{f"{category}_score": 50.0 + index for category in CATEGORIES},
                )
            )
        session.commit()
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

    for name in dir(lib_data):
        if name.startswith("_"):
            continue
        reader = getattr(lib_data, name)
        if callable(reader) and hasattr(reader, "clear"):
            reader.clear()
    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.data.portfolio_backend", return_value="session"),
        patch("quantpulse.llm.providers.get_provider", return_value=None),
    ):
        yield


def _run(engine: Engine) -> AppTest:
    from quantpulse.portfolio import holdings as holdings_lib

    with _wired(engine):
        at = AppTest.from_file(PORTFOLIO_PAGE, default_timeout=180)
        at.session_state["quantpulse_portfolio"] = holdings_lib.PortfolioState(
            # Equal share counts at one price, so the three holdings carry
            # near-equal weight and the old proxy is almost exactly the
            # portfolio itself.
            transactions=[
                Transaction(
                    symbol=symbol,
                    action="buy",
                    shares=10.0,
                    price=100.0,
                    date=AS_OF - timedelta(days=300),
                )
                for symbol in HELD
            ],
            cash=0.0,
        )
        at.run()
    return at


def _beta_caption(at: AppTest) -> str:
    captions = [str(element.value) for element in at.caption]
    matches = [c for c in captions if "Beta is measured against" in c]
    assert matches, f"the portfolio page never reported a beta; captions were {captions}"
    return matches[0]


def test_portfolio_beta_is_not_measured_against_the_portfolio_itself(
    two_factor_engine: Engine,
) -> None:
    at = _run(two_factor_engine)
    assert not at.exception

    caption = _beta_caption(at)
    r_squared = float(caption.split("R² = ")[1].split()[0])
    # Against its own holdings this portfolio is explained essentially
    # perfectly (R-squared ~1.00). Against a universe driven by an independent
    # factor it cannot be. Anything near 1 means the proxy is the portfolio.
    assert r_squared < 0.90, (
        "portfolio beta looks like it is still being regressed against the "
        f"portfolio's own holdings (R² = {r_squared:.4f}); caption was: {caption}"
    )
