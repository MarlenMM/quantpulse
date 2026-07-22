"""End-to-end Phase 6: seed each category's raw data, then score the universe."""

from collections.abc import Iterator
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import refresh_data
from quantpulse.analysis import scoring
from quantpulse.storage import persistence
from quantpulse.storage.models import (
    AnalystConsensus,
    Base,
    CompositeScore,
    FundamentalsSnapshot,
    MarketRegime,
    NewsEvent,
    PriceHistory,
    SentimentScore,
    ThematicBasket,
    Ticker,
)

AS_OF = date(2026, 7, 22)


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        yield s


def _seed_prices(session: Session, symbol: str, closes: list[float]) -> None:
    for offset, close in enumerate(closes):
        d = AS_OF - timedelta(days=len(closes) - offset)
        session.add(
            PriceHistory(
                symbol=symbol,
                date=d,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                adj_close=close,
                volume=1_000_000,
            )
        )


def _seed_universe(session: Session) -> pd.DataFrame:
    """Three Tech names: AAA strong (uptrend + cheap), BBB middling, CCC weak."""
    specs = [
        ("AAA", "Alpha Inc.", list(np.linspace(100, 200, 260)), {"pe": 10.0, "roe": 0.30}),
        ("BBB", "Beta Inc.", [150.0] * 260, {"pe": 20.0, "roe": 0.15}),
        ("CCC", "Gamma Inc.", list(np.linspace(200, 100, 260)), {"pe": 40.0, "roe": 0.05}),
    ]
    for symbol, name, closes, funda in specs:
        session.add(
            Ticker(
                symbol=symbol,
                name=name,
                sector="Information Technology",
                asset_type="equity",
                is_active=True,
            )
        )
        _seed_prices(session, symbol, closes)
        session.add(
            FundamentalsSnapshot(symbol=symbol, as_of_date=AS_OF - timedelta(days=3), **funda)
        )
    session.flush()
    return pd.DataFrame(
        [{"symbol": s, "name": n, "sector": "Information Technology"} for s, n, _, _ in specs]
    )


def test_composite_ranks_and_persists(session: Session) -> None:
    universe = _seed_universe(session)
    session.add(MarketRegime(date=AS_OF, regime_score=55.0, regime_label="neutral"))
    session.flush()

    written = refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()

    scores = {
        c.symbol: c for c in session.scalars(select(CompositeScore).order_by(CompositeScore.symbol))
    }
    assert written == 3
    assert set(scores) == {"AAA", "BBB", "CCC"}
    # AAA (uptrend + cheap fundamentals) outranks CCC (downtrend + expensive).
    assert scores["AAA"].composite_score > scores["BBB"].composite_score
    assert scores["BBB"].composite_score > scores["CCC"].composite_score
    # Top of a 3-name universe is Strong Buy; ratings never improve as the
    # composite falls (the full relative cutoffs are unit-tested at scale).
    assert scores["AAA"].rating == "strong_buy"
    order = {rating: i for i, rating in enumerate(scoring.RATINGS)}
    assert order[scores["AAA"].rating] <= order[scores["BBB"].rating] <= order[scores["CCC"].rating]
    # Every row records the profile and a coverage-based confidence.
    assert all(c.profile == "balanced" for c in scores.values())
    assert all(0 < c.data_confidence <= 100 for c in scores.values())
    # Fundamental + technical + momentum all had data -> those sub-scores are set.
    assert scores["AAA"].fundamental_score is not None
    assert scores["AAA"].technical_score is not None
    assert scores["AAA"].momentum_score is not None
    # No sentiment/analyst/smart-money seeded -> those sub-scores are null.
    assert scores["AAA"].sentiment_score is None
    assert scores["AAA"].analyst_score is None


def test_sentiment_and_analyst_flow_into_the_composite(session: Session) -> None:
    universe = _seed_universe(session)
    # Add Tier-1 sentiment and an analyst consensus for AAA only.
    session.add(
        SentimentScore(
            symbol="AAA",
            date=AS_OF - timedelta(days=1),
            source="tier1_aggregate",
            sentiment_score=0.8,
            mention_volume=5,
        )
    )
    session.add(
        AnalystConsensus(
            symbol="AAA",
            as_of_date=AS_OF - timedelta(days=2),
            strong_buy=8,
            buy=2,
            hold=0,
            sell=0,
            strong_sell=0,
            mean_price_target=250.0,
        )
    )
    session.flush()

    refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()

    aaa = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "AAA")).one()
    assert aaa.sentiment_score is not None  # sentiment category now contributes
    assert aaa.analyst_score is not None
    # AAA has more categories covered than its peers -> higher data_confidence.
    bbb = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "BBB")).one()
    assert aaa.data_confidence > bbb.data_confidence


def test_tier2_industry_news_flows_in_via_baskets(session: Session) -> None:
    universe = _seed_universe(session)
    # AAA belongs to a basket that has a strongly positive Tier-2 story.
    session.add(ThematicBasket(theme_name="ai_theme", symbol="AAA"))
    session.add(
        NewsEvent(
            article_id="evt1",
            tier=2,
            matched_theme="ai_theme",
            sentiment_score=0.9,
            published_at=datetime(2026, 7, 21, 12, 0),
        )
    )
    session.flush()

    refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()

    aaa = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "AAA")).one()
    ccc = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "CCC")).one()
    assert aaa.industry_macro_score is not None  # AAA in the affected basket
    assert ccc.industry_macro_score is None  # CCC in no basket -> no industry signal


def test_scoring_is_point_in_time(session: Session) -> None:
    universe = _seed_universe(session)
    # A future sentiment row (dated after AS_OF) must not be read.
    session.add(
        SentimentScore(
            symbol="AAA",
            date=AS_OF + timedelta(days=1),
            source="tier1_aggregate",
            sentiment_score=0.9,
            mention_volume=3,
        )
    )
    session.flush()

    refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()

    aaa = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "AAA")).one()
    assert aaa.sentiment_score is None  # the future sentiment was correctly excluded


def test_composite_write_is_append_only(session: Session) -> None:
    universe = _seed_universe(session)
    session.flush()
    refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()
    first = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "AAA")).one()
    original = first.composite_score

    # A same-day re-run must not overwrite the first ranking (point-in-time).
    refresh_data.refresh_composite_scores(session, universe, AS_OF)
    session.flush()
    rows = session.scalars(select(CompositeScore).where(CompositeScore.symbol == "AAA")).all()
    assert len(rows) == 1
    assert rows[0].composite_score == original


def test_read_latest_fundamentals_is_point_in_time(session: Session) -> None:
    session.add(Ticker(symbol="AAA", name="Alpha", sector="Tech", asset_type="equity"))
    session.add(FundamentalsSnapshot(symbol="AAA", as_of_date=date(2026, 7, 20), pe=10.0))
    session.add(FundamentalsSnapshot(symbol="AAA", as_of_date=date(2026, 7, 25), pe=99.0))
    session.flush()
    latest = persistence.read_latest_fundamentals(session, as_of=AS_OF)
    # As of the 22nd, the future 25th snapshot must not be the "latest".
    assert latest.set_index("symbol").loc["AAA", "pe"] == 10.0
