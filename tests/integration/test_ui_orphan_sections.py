"""End-to-end checks for the sections wired up in the final review.

Four capabilities were fully built and reachable from no user interface at all:
short-interest readings, the Monte Carlo fan chart, correlation clustering and
Kelly position sizing. "The function exists and has unit tests" was true for
every one of them the whole time, which is exactly why the gap survived — so
these tests assert the thing that was actually missing: that a user running the
real page *sees* it.

Each runs the real Streamlit script through `streamlit.testing.v1.AppTest`
against a temporary database, following `test_ui_stock_detail_chat.py`.
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
    BacktestResult,
    Base,
    CompositeScore,
    PriceHistory,
    ShortInterest,
    Ticker,
)

APP_DIR = Path(__file__).resolve().parents[2] / "app"
STOCK_DETAIL = str(APP_DIR / "pages" / "2_Stock_Detail.py")
BACKTEST_PAGE = str(APP_DIR / "pages" / "4_Backtest.py")
AS_OF = date(2026, 7, 27)


def _price_series(session, symbol: str, *, days: int = 400, seed: int = 0) -> None:
    """A wiggly but plausible price path -- flat prices give zero variance."""
    rng = np.random.default_rng(seed)
    level = 100.0
    for offset in range(days, 0, -1):
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


@pytest.fixture
def seeded_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'orphans.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(
            Ticker(
                symbol="NVDA",
                name="NVIDIA Corporation",
                sector="Information Technology",
                asset_type="equity",
                is_active=True,
            )
        )
        _price_series(session, "NVDA", seed=7)
        session.add(
            CompositeScore(
                symbol="NVDA",
                date=AS_OF,
                profile="balanced",
                composite_score=87.3,
                percentile_rank=96.2,
                rating="strong_buy",
                data_confidence=84.0,
                **{f"{category}_score": 70.0 for category in CATEGORIES},
            )
        )
        # Elevated short interest -- the case Section 24 cares most about.
        session.add(
            ShortInterest(
                symbol="NVDA",
                as_of_date=AS_OF,
                pct_float_short=18.5,
                days_to_cover=4.2,
            )
        )
        session.add(
            BacktestResult(
                run_date=AS_OF,
                period_start=AS_OF - timedelta(days=1800),
                period_end=AS_OF,
                cadence="monthly",
                n_periods=60,
                sharpe=0.61,
                cagr=0.097,
                max_drawdown=-0.34,
                win_rate=0.58,
                payoff_ratio=1.6,
                benchmark_cagr=0.11,
                benchmark_sharpe=0.65,
                avg_turnover=0.67,
                assumed_txn_cost=0.001,
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

    for reader in (
        lib_data.screener_rows,
        lib_data.forecasts,
        lib_data.ohlcv,
        lib_data.short_interest,
        lib_data.backtest_history,
        lib_data.universe,
    ):
        reader.clear()
    with patch("lib.data.get_session", fake_get_session):
        yield


def _run(page: str, engine: Engine) -> AppTest:
    with _wired(engine):
        at = AppTest.from_file(page, default_timeout=120)
        # No LLM configured: the narrative/chat sections stay out of the way.
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            at.run()
    return at


def _text(at: AppTest) -> str:
    parts = [element.value for element in at.subheader]
    parts += [element.value for element in at.markdown]
    parts += [element.value for element in at.caption]
    parts += [element.value for element in at.warning]
    parts += [element.value for element in at.info]
    return "\n".join(str(p) for p in parts)


class TestShortInterestIsVisible:
    """Section 24 requires BOTH readings be shown, not one directional verdict."""

    def test_section_renders_with_both_readings(self, seeded_engine: Engine) -> None:
        at = _run(STOCK_DETAIL, seeded_engine)
        assert not at.exception
        assert "Short interest" in [element.value for element in at.subheader]
        labels = [metric.label for metric in at.metric]
        assert "% of float short" in labels
        assert "Days to cover" in labels

    def test_elevated_reading_presents_both_interpretations(self, seeded_engine: Engine) -> None:
        # The whole point of Section 24: not collapsed into bullish or bearish.
        at = _run(STOCK_DETAIL, seeded_engine)
        body = _text(at).lower()
        assert "squeeze" in body
        assert "betting against" in body


class TestMonteCarloIsVisible:
    def test_simulated_paths_section_renders(self, seeded_engine: Engine) -> None:
        at = _run(STOCK_DETAIL, seeded_engine)
        assert not at.exception
        assert "Simulated price paths" in [element.value for element in at.subheader]

    def test_it_is_labelled_a_range_not_a_prediction(self, seeded_engine: Engine) -> None:
        at = _run(STOCK_DETAIL, seeded_engine)
        body = _text(at).lower()
        assert "not a prediction" in body


class TestKellySizingIsVisible:
    def test_position_sizing_section_renders(self, seeded_engine: Engine) -> None:
        at = _run(BACKTEST_PAGE, seeded_engine)
        assert not at.exception
        assert "How much to bet" in [element.value for element in at.subheader]
        assert "Suggested position" in [metric.label for metric in at.metric]

    def test_it_states_the_inputs_it_used(self, seeded_engine: Engine) -> None:
        # A position size is only as good as the track record behind it, so the
        # page must show which win rate and payoff ratio produced it.
        at = _run(BACKTEST_PAGE, seeded_engine)
        labels = [metric.label for metric in at.metric]
        assert "Win rate used" in labels
        assert "Payoff ratio used" in labels

    def test_absent_when_the_run_has_no_payoff_ratio(self, tmp_path: Path) -> None:
        # A run with no losing period has an undefined payoff ratio; Kelly must
        # not be shown rather than sized off a bet that looks unlosable.
        engine = create_engine(f"sqlite:///{tmp_path / 'nopayoff.db'}")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        with factory() as session:
            session.add(
                BacktestResult(
                    run_date=AS_OF,
                    period_start=AS_OF - timedelta(days=900),
                    period_end=AS_OF,
                    cadence="monthly",
                    n_periods=30,
                    sharpe=0.5,
                    cagr=0.08,
                    max_drawdown=-0.2,
                    win_rate=1.0,
                    payoff_ratio=None,
                    benchmark_cagr=0.07,
                    benchmark_sharpe=0.4,
                    avg_turnover=0.5,
                    assumed_txn_cost=0.001,
                )
            )
            session.commit()

        at = _run(BACKTEST_PAGE, engine)
        assert not at.exception
        assert "How much to bet" not in [element.value for element in at.subheader]
