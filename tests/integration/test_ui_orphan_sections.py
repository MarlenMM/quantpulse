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
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.portfolio.transactions import Transaction
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
# Anchored to today, not to a literal date. Every window the Stock Detail page
# reads is measured from `date.today()` -- news over 21 days, the cross-asset
# macro series over 60 -- so fixture rows pinned to a fixed calendar date age
# out of those windows and the sections under test silently stop rendering.
# That is exactly what happened: these tests passed for a month and then began
# failing on a commit that touched none of them. Seeding relative to today
# keeps the fixtures inside the windows the code actually queries.
AS_OF = date.today()


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
        lib_data.patterns,
        lib_data.latest_prices,
        lib_data.adj_close_panel,
        lib_data.universe_panel,
        lib_data.macro_series,
        lib_data.options_signals,
        lib_data.market_regime,
        lib_data.rating_changes,
        lib_data.market_moving_news,
        lib_data.data_freshness,
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


class TestDetectedPatternsAreVisible:
    """Section 8's "price chart with indicators and detected patterns".

    The panel existed in both front ends the whole time; nothing ever wrote a
    `pattern_signals` row, so it could only ever say "No chart or candlestick
    patterns detected". This asserts the end a user sees: real detector output,
    stored by the nightly, rendered on the page.
    """

    @pytest.fixture
    def patterned_engine(self, seeded_engine: Engine) -> Engine:
        """The standard fixture, plus whatever the real detectors find in its prices."""
        import pandas as pd

        import refresh_data

        factory = sessionmaker(bind=seeded_engine)
        with factory() as session:
            universe = pd.DataFrame([{"symbol": "NVDA", "name": "NVIDIA", "sector": "Tech"}])
            refresh_data.refresh_pattern_signals(session, universe, AS_OF)
            session.commit()
        return seeded_engine

    def test_the_nightly_finds_patterns_in_the_seeded_prices(
        self, patterned_engine: Engine
    ) -> None:
        from quantpulse.storage.models import PatternSignal

        with sessionmaker(bind=patterned_engine)() as session:
            stored = session.scalars(select(PatternSignal)).all()
        assert stored, "the detectors found nothing in 400 bars of random-walk prices"

    def test_the_panel_lists_them_instead_of_saying_none_were_found(
        self, patterned_engine: Engine
    ) -> None:
        at = _run(STOCK_DETAIL, patterned_engine)
        assert not at.exception
        assert "Detected patterns" in [element.value for element in at.subheader]
        assert "No chart or candlestick patterns detected" not in _text(at)

    def test_the_panel_still_says_so_honestly_when_there_are_none(
        self, seeded_engine: Engine
    ) -> None:
        # The unpatterned fixture writes no pattern rows, so the empty state has
        # to remain reachable -- otherwise the test above proves nothing.
        at = _run(STOCK_DETAIL, seeded_engine)
        assert "No chart or candlestick patterns detected" in _text(at)


class TestTargetAllocationIsVisible:
    """Section 27's optimizers and trade list had no page at all.

    `mean_variance_optimize`, `hierarchical_risk_parity` and
    `black_litterman_optimize` -- plus `rebalancing.build_rebalance_plan` --
    were fully built and tested, and the README advertised "3 optimization
    methods" a user had no way to run. Only the Kelly helper was ever wired.
    """

    PORTFOLIO_PAGE = str(APP_DIR / "pages" / "3_Portfolio.py")

    @pytest.fixture
    def held_engine(self, tmp_path: Path) -> Engine:
        """Six priced holdings with 400 bars each and a session-backed portfolio."""
        engine = create_engine(f"sqlite:///{tmp_path / 'holdings.db'}")
        Base.metadata.create_all(engine)
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        with sessionmaker(bind=engine)() as session:
            for seed, symbol in enumerate(symbols):
                session.add(
                    Ticker(
                        symbol=symbol,
                        name=f"{symbol} Inc",
                        sector="Information Technology",
                        asset_type="equity",
                        is_active=True,
                    )
                )
                _price_series(session, symbol, seed=seed)
                session.add(
                    CompositeScore(
                        symbol=symbol,
                        date=AS_OF,
                        profile="balanced",
                        composite_score=50.0 + seed * 5,
                        percentile_rank=50.0 + seed * 5,
                        rating="hold",
                        data_confidence=80.0,
                        **{f"{category}_score": 50.0 + seed * 5 for category in CATEGORIES},
                    )
                )
            session.commit()
        return engine

    def _run_portfolio(self, engine: Engine, method_index: int = 0) -> AppTest:
        from quantpulse.portfolio import holdings as holdings_lib

        with _wired(engine):
            at = AppTest.from_file(self.PORTFOLIO_PAGE, default_timeout=180)
            at.session_state["quantpulse_portfolio"] = holdings_lib.PortfolioState(
                transactions=[
                    Transaction(
                        symbol=symbol,
                        action="buy",
                        shares=10.0 * (index + 1),
                        price=100.0,
                        date=AS_OF - timedelta(days=200),
                    )
                    for index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
                ],
                cash=1_000.0,
            )
            with patch("lib.data.portfolio_backend", return_value="session"):
                at.run()
                if method_index:
                    next(r for r in at.radio if r.label == "Method").set_value(
                        list(at.radio[0].options)[method_index]
                    )
                    at.run()
        return at

    def test_section_renders_with_a_trade_list(self, held_engine: Engine) -> None:
        at = self._run_portfolio(held_engine)
        assert not at.exception
        assert "Target allocation & trades" in [element.value for element in at.subheader]
        body = _text(at)
        assert "Sells first, then buys" in body or "Already at target" in body

    @pytest.mark.parametrize("method_index", [0, 1, 2])
    def test_every_optimizer_choice_runs(self, held_engine: Engine, method_index: int) -> None:
        # All three of the README's advertised methods have to be reachable, not
        # just the default one.
        at = self._run_portfolio(held_engine, method_index=method_index)
        assert not at.exception
        assert "Target allocation & trades" in [element.value for element in at.subheader]

    def test_it_is_labelled_as_beliefs_not_a_forecast(self, held_engine: Engine) -> None:
        at = self._run_portfolio(held_engine)
        body = _text(at)
        assert "not forecasts" in body
        assert "Not an instruction to trade" in body


class TestPerStockRiskAndMacroOverlayAreVisible:
    """Two more built-and-unreachable pieces, now on the Stock Detail page.

    `risk.stock_risk_profile` computed a name's volatility (realised and
    implied), beta, Sortino and VaR in one call and had no caller -- the
    Portfolio page shows portfolio-level risk, so a single stock's risk numbers
    were displayed nowhere. `macro.commodity_overlay_adjustment` is Section 28's
    targeted overlay; its inputs were ingested nightly from the start and
    nothing ever computed the adjustment.
    """

    HOME_PAGE = str(APP_DIR / "Home.py")

    @pytest.fixture
    def macro_engine(self, seeded_engine: Engine) -> Engine:
        """The standard fixture plus a second sector name and cross-asset history."""
        from quantpulse.analysis import macro
        from quantpulse.storage.models import MacroIndicator, OptionsSignal

        with sessionmaker(bind=seeded_engine)() as session:
            session.add(
                Ticker(
                    symbol="XOM",
                    name="Exxon Mobil",
                    sector="Energy",
                    asset_type="equity",
                    is_active=True,
                )
            )
            _price_series(session, "XOM", seed=11)
            # Oil down hard, dollar flat: Energy should read as a headwind.
            for offset, (oil, dollar) in enumerate([(90.0, 100.0), (80.0, 99.0)]):
                day = AS_OF - timedelta(days=30 - offset * 29)
                session.add(MacroIndicator(date=day, indicator_name=macro.OIL_WTI, value=oil))
                session.add(
                    MacroIndicator(date=day, indicator_name=macro.DOLLAR_INDEX, value=dollar)
                )
            session.add(
                OptionsSignal(
                    symbol="NVDA",
                    date=AS_OF,
                    put_call_ratio=0.9,
                    atm_implied_volatility=0.42,
                )
            )
            session.commit()
        return seeded_engine

    def test_risk_profile_renders_for_a_single_stock(self, macro_engine: Engine) -> None:
        at = _run(STOCK_DETAIL, macro_engine)
        assert not at.exception
        assert "Risk profile" in [element.value for element in at.subheader]
        labels = [metric.label for metric in at.metric]
        for expected in ("Volatility (ann.)", "Implied vol", "Beta", "Sortino", "Daily VaR 95%"):
            assert expected in labels

    def test_beta_is_shown_with_the_r_squared_that_qualifies_it(self, macro_engine: Engine) -> None:
        # A beta of 1.4 with an R^2 of 0.05 is not the number a reader thinks it
        # is, so the qualifier travels with it.
        at = _run(STOCK_DETAIL, macro_engine)
        assert "R²" in _text(at)

    def test_macro_overlay_is_targeted_not_universal(self, macro_engine: Engine) -> None:
        # NVDA is Information Technology, which is dollar-sensitive only; the
        # section renders. The wording has to make the targeting explicit,
        # because a universal overlay is exactly what Section 28 forbids.
        at = _run(STOCK_DETAIL, macro_engine)
        assert not at.exception
        body = _text(at)
        assert "Macro overlay" in [element.value for element in at.subheader]
        assert "Every other sector gets exactly zero" in body

    def test_sector_rotation_renders_on_the_dashboard(self, macro_engine: Engine) -> None:
        at = _run(self.HOME_PAGE, macro_engine)
        assert not at.exception
        assert "Sector rotation" in [element.value for element in at.subheader]
        assert "not a forecast" in _text(at)
