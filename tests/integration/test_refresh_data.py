import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import refresh_data
from quantpulse.alerting import discord, rules
from quantpulse.analysis import backtest as bt
from quantpulse.analysis import risk, scoring
from quantpulse.execution import alpaca
from quantpulse.ingestion import historical_constituents_client as hist
from quantpulse.storage import persistence
from quantpulse.storage.models import (
    AnalystConsensus,
    BacktestResult,
    Base,
    CompositeScore,
    FundamentalsSnapshot,
    IndexMembershipHistory,
    MarketRegime,
    NewsEvent,
    OptionsSignal,
    PatternSignal,
    PriceHistory,
    RefreshLog,
    Ticker,
)
from quantpulse.utils.market_calendar import is_trading_day


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _stamp_migration_head(eng: Engine) -> None:
    """Record the current Alembic head on a `create_all`-built database.

    `run()` refuses to start against a database that isn't migrated up to head,
    and it now checks that through its *own* session -- i.e. the temporary
    database these tests actually use. A `create_all` schema has the right
    tables but no `alembic_version` row at all, so without this the guard sees
    "no revision" and (correctly) refuses. Stamping mirrors what
    `alembic upgrade head` leaves behind.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[2]
    head = ScriptDirectory.from_config(Config(str(root / "alembic.ini"))).get_current_head()
    with eng.begin() as connection:
        connection.execute(sa.text("CREATE TABLE IF NOT EXISTS alembic_version (version_num TEXT)"))
        connection.execute(sa.text("DELETE FROM alembic_version"))
        connection.execute(
            sa.text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head}
        )


@pytest.fixture(autouse=True)
def _no_live_listing_fetch():
    """No test in this file may reach Nasdaq's symbol directory over the network.

    `run()` syncs the searchable catalogue, so without this every test that
    drives it would make a live HTTP call -- slow, flaky, and (worse) it would
    insert 13,000 real symbols into a fixture built around three. The catalogue
    tests below patch this with their own rows.
    """
    with patch(
        "refresh_data.listing_client.fetch_us_listings",
        return_value=pd.DataFrame(columns=["symbol", "name", "exchange", "asset_type"]),
    ):
        yield


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(eng)
    _stamp_migration_head(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine)
    with factory() as s:
        yield s


def _price_df(symbol: str, start: str = "2026-07-13", rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "adj_close": 1.5,
            "volume": 1000,
        }
    )


def test_sync_universe_inserts_and_deactivates(session: Session) -> None:
    session.add(Ticker(symbol="OLD", name="Old Co", asset_type="equity", is_active=True))
    session.flush()

    constituents = pd.DataFrame(
        [
            {
                "symbol": "NEW",
                "name": "New Co",
                "sector": "Tech",
                "industry": "Software",
                "exchange": None,
                "asset_type": "equity",
                "is_active": True,
            }
        ]
    )
    with patch("refresh_data.wikipedia_client.fetch_sp500_constituents", return_value=constituents):
        count = refresh_data.sync_universe(session)

    assert count == 1
    tickers = {t.symbol: t for t in session.scalars(select(Ticker))}
    assert tickers["NEW"].is_active is True
    assert tickers["OLD"].is_active is False


_RECONCILE_DAY = date(2026, 7, 22)


def _member(session: Session, symbol: str, added: date, removed: date | None) -> None:
    session.add(
        IndexMembershipHistory(
            index_name="S&P 500", symbol=symbol, added_date=added, removed_date=removed
        )
    )


def test_reconcile_opens_interval_for_a_new_member(session: Session) -> None:
    session.add(Ticker(symbol="NEW", name="New", asset_type="equity", is_active=True))
    session.flush()
    changed = refresh_data.reconcile_index_membership(session, _RECONCILE_DAY)
    rows = session.scalars(
        select(IndexMembershipHistory).where(IndexMembershipHistory.symbol == "NEW")
    ).all()
    assert changed == 1
    assert len(rows) == 1 and rows[0].removed_date is None and rows[0].added_date == _RECONCILE_DAY


def test_reconcile_closes_interval_for_a_departed_member(session: Session) -> None:
    # A name still open in membership but no longer active in the index -> closed.
    session.add(Ticker(symbol="OLD", name="Old", asset_type="equity", is_active=False))
    _member(session, "OLD", date(2000, 1, 1), None)
    session.flush()
    changed = refresh_data.reconcile_index_membership(session, _RECONCILE_DAY)
    row = session.scalars(
        select(IndexMembershipHistory).where(IndexMembershipHistory.symbol == "OLD")
    ).one()
    assert changed == 1 and row.removed_date == _RECONCILE_DAY


def test_reconcile_is_idempotent_on_a_steady_index(session: Session) -> None:
    session.add(Ticker(symbol="AAA", name="A", asset_type="equity", is_active=True))
    _member(session, "AAA", date(2000, 1, 1), None)
    session.flush()
    assert refresh_data.reconcile_index_membership(session, _RECONCILE_DAY) == 0


def test_reconcile_reopens_a_readmitted_name_keeping_the_old_interval(session: Session) -> None:
    # AAA was a member, was removed, and is a current member again: a new open
    # interval opens while the old closed one is preserved (Section 6.9).
    session.add(Ticker(symbol="AAA", name="A", asset_type="equity", is_active=True))
    _member(session, "AAA", date(2000, 1, 1), date(2010, 1, 1))
    session.flush()
    changed = refresh_data.reconcile_index_membership(session, _RECONCILE_DAY)
    rows = session.scalars(
        select(IndexMembershipHistory)
        .where(IndexMembershipHistory.symbol == "AAA")
        .order_by(IndexMembershipHistory.added_date)
    ).all()
    assert changed == 1 and len(rows) == 2
    assert rows[0].removed_date == date(2010, 1, 1)  # old membership preserved
    assert rows[1].added_date == _RECONCILE_DAY and rows[1].removed_date is None  # reopened


def test_upsert_price_history_is_idempotent(session: Session) -> None:
    df = _price_df("AAPL", rows=3)

    first = refresh_data._upsert_price_history(session, df)
    second = refresh_data._upsert_price_history(session, df)
    session.flush()

    rows = session.scalars(select(PriceHistory).where(PriceHistory.symbol == "AAPL")).all()
    assert first == 3
    assert second == 3
    assert len(rows) == 3  # same (symbol, date) keys -> updated in place, not duplicated


def test_upsert_price_history_drops_bars_missing_a_required_field(session: Session) -> None:
    # The exact production failure (Actions run 30878926050): yfinance returned
    # a bar with a NaN close, the NOT NULL insert raised, and the whole nightly
    # run rolled back -- leaving the deployed demo four days stale while the
    # workflow still exited 0. An incomplete bar must cost that bar only.
    df = _price_df("AAPL", rows=3)
    df.loc[1, "close"] = float("nan")

    written = refresh_data._upsert_price_history(session, df)
    session.flush()

    rows = session.scalars(select(PriceHistory).where(PriceHistory.symbol == "AAPL")).all()
    assert written == 2
    assert len(rows) == 2
    assert all(row.close == 1.5 for row in rows)


def test_upsert_price_history_skips_a_frame_with_no_complete_bar(session: Session) -> None:
    df = _price_df("AAPL", rows=2)
    df["adj_close"] = float("nan")

    assert refresh_data._upsert_price_history(session, df) == 0
    session.flush()
    assert session.scalars(select(PriceHistory)).all() == []


def test_one_bad_ticker_does_not_discard_every_other_tickers_writes(session: Session) -> None:
    # Per-symbol SAVEPOINT isolation: the loop in `run()` writes ~500 tickers in
    # one session, so an exception on one used to take the whole batch with it.
    good_before = refresh_data._upsert_price_history(session, _price_df("AAPL", rows=2))

    try:
        with session.begin_nested():
            # A frame missing a NOT NULL column entirely still raises at insert
            # (the dropna guard can only drop rows, not invent a column) --
            # standing in for any per-symbol write failure.
            broken = _price_df("NOSUCH", rows=2).drop(columns=["adj_close"])
            refresh_data._upsert_price_history(session, broken)
            session.flush()
    except Exception:
        pass

    good_after = refresh_data._upsert_price_history(session, _price_df("MSFT", rows=2))
    session.flush()

    symbols = {row.symbol for row in session.scalars(select(PriceHistory)).all()}
    assert good_before == 2 and good_after == 2
    assert "AAPL" in symbols, "writes made before the failure must survive it"
    assert "MSFT" in symbols, "writes made after the failure must still land"
    assert "NOSUCH" not in symbols


def test_upsert_fundamentals_does_not_overwrite_same_day_snapshot(session: Session) -> None:
    today = date(2026, 7, 21)
    refresh_data._upsert_fundamentals(session, "AAPL", today, {"symbol": "AAPL", "pe": 10.0})
    refresh_data._upsert_fundamentals(session, "AAPL", today, {"symbol": "AAPL", "pe": 999.0})
    session.flush()

    row = session.scalars(
        select(FundamentalsSnapshot).where(
            FundamentalsSnapshot.symbol == "AAPL", FundamentalsSnapshot.as_of_date == today
        )
    ).one()
    assert row.pe == 10.0  # first write wins -- point-in-time data is append-only


def test_upsert_analyst_consensus_does_not_overwrite_same_day_snapshot(session: Session) -> None:
    today = date(2026, 7, 21)
    data = {
        "symbol": "AAPL",
        "strong_buy": 1,
        "buy": 2,
        "hold": 3,
        "sell": 0,
        "strong_sell": 0,
        "mean_price_target": 100.0,
    }
    refresh_data._upsert_analyst_consensus(session, "AAPL", today, data)
    refresh_data._upsert_analyst_consensus(
        session, "AAPL", today, {**data, "mean_price_target": 999.0}
    )
    session.flush()

    row = session.scalars(
        select(AnalystConsensus).where(
            AnalystConsensus.symbol == "AAPL", AnalystConsensus.as_of_date == today
        )
    ).one()
    assert row.mean_price_target == 100.0


def test_fetch_ticker_data_only_fetches_prices_on_non_weekly_days() -> None:
    with (
        patch(
            "refresh_data.yfinance_client.fetch_price_history", return_value=_price_df("AAPL")
        ) as mock_price,
        patch("refresh_data.yfinance_client.fetch_fundamentals") as mock_fundamentals,
        patch("refresh_data.yfinance_client.fetch_analyst_consensus") as mock_analyst,
    ):
        result = refresh_data.fetch_ticker_data("AAPL", last_price_date=None, is_weekly=False)

    mock_price.assert_called_once()
    mock_fundamentals.assert_not_called()
    mock_analyst.assert_not_called()
    assert result.errors == []
    assert result.price_df is not None


def test_fetch_ticker_data_filters_to_new_rows_only() -> None:
    df = _price_df("AAPL", rows=5)  # 2026-07-13 .. 2026-07-17
    with patch("refresh_data.yfinance_client.fetch_price_history", return_value=df):
        result = refresh_data.fetch_ticker_data(
            "AAPL", last_price_date=date(2026, 7, 15), is_weekly=False
        )

    assert result.price_df is not None
    assert (result.price_df["date"] > pd.Timestamp(date(2026, 7, 15))).all()


def test_fetch_ticker_data_records_errors_without_raising() -> None:
    with patch(
        "refresh_data.yfinance_client.fetch_price_history", side_effect=RuntimeError("boom")
    ):
        result = refresh_data.fetch_ticker_data("AAPL", last_price_date=None, is_weekly=False)

    assert result.price_df is None
    assert any("price_history" in e for e in result.errors)


def _fake_session_factory(engine: Engine):
    factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_session() -> Iterator[Session]:
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return fake_get_session, factory


def test_run_end_to_end_with_tiny_mocked_universe(engine: Engine) -> None:
    fake_get_session, factory = _fake_session_factory(engine)
    tiny_universe = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "exchange": None,
                "asset_type": "equity",
                "is_active": True,
            }
        ]
    )

    # Every external source the (possibly-weekly) run can reach is mocked to a
    # benign value so the test is deterministic regardless of the weekday, and
    # so a real network call never leaks into CI.
    options_snapshot = {
        "symbol": "AAPL",
        "expiration": "2026-08-21",
        "put_call_ratio": 0.8,
        "atm_implied_volatility": 0.25,
    }
    # The clock is pinned to a Tuesday, and that is load-bearing rather than
    # tidy. `run()` derives `is_weekly` from the real weekday, so this test
    # silently exercised a different code path depending on the day it ran --
    # and on a Monday it took the weekly branch, where `institutional_ownership`
    # is mocked to an empty frame, lands in `_STEPS_EXPECTED_TO_WRITE`, and
    # correctly downgrades the run to "partial". The assertion below then failed
    # for a reason that had nothing to do with what the test is about. It was
    # written on a weekday and went red the first Monday after (2026-09-07).
    # The weekly branch has its own tests; this one is the daily path.
    with (
        patch("refresh_data.get_session", fake_get_session),
        patch("refresh_data.datetime", wraps=datetime) as clock,
        patch("refresh_data.is_trading_day", return_value=True),
        patch("refresh_data.wikipedia_client.fetch_sp500_constituents", return_value=tiny_universe),
        patch(
            "refresh_data.yfinance_client.fetch_price_history",
            return_value=_price_df("AAPL", rows=2),
        ),
        patch("refresh_data.options_client.fetch_options_signals", return_value=options_snapshot),
        patch("refresh_data.yfinance_client.fetch_fundamentals", return_value={"symbol": "AAPL"}),
        patch(
            "refresh_data.yfinance_client.fetch_analyst_consensus",
            return_value={"symbol": "AAPL"},
        ),
        patch(
            "refresh_data.short_interest_client.fetch_short_interest",
            return_value={"symbol": "AAPL", "pct_float_short": None, "days_to_cover": None},
        ),
        patch(
            "refresh_data.edgar_client.fetch_insider_transactions",
            return_value=_empty_df(list(refresh_data._INSIDER_COLUMNS)),
        ),
        patch(
            "refresh_data.news_client.fetch_all_tier1_news",
            return_value=_empty_df(
                ["title", "link", "summary", "published_at", "source", "symbol"]
            ),
        ),
        patch(
            "refresh_data.edgar_13f_client.fetch_institutional_ownership_trend",
            return_value=_empty_df(list(refresh_data._INSTITUTIONAL_COLUMNS)),
        ),
        patch("refresh_data.gdelt_client.fetch_articles", return_value=_empty_df(["title", "url"])),
        patch(
            "refresh_data.gdelt_client.fetch_tone_timeline",
            return_value=_empty_df(["date", "tone", "query"]),
        ),
    ):
        clock.now.return_value = datetime(2026, 7, 21, 22, 0)  # a Tuesday
        refresh_data.run(job_name="test_run")

    with factory() as session:
        prices = session.scalars(select(PriceHistory)).all()
        logs = session.scalars(select(RefreshLog)).all()
        options = session.scalars(select(OptionsSignal)).all()
        regimes = session.scalars(select(MarketRegime)).all()

    assert len(prices) == 2
    assert len(logs) == 1
    assert logs[0].status == "success"
    # The new wiring actually persisted: a daily options snapshot and a regime row.
    assert len(options) == 1
    assert options[0].put_call_ratio == 0.8
    assert len(regimes) == 1  # computed even with sparse inputs (breadth None here)


def test_refresh_cross_asset_macro_writes_series(session: Session) -> None:
    from quantpulse.storage.models import MacroIndicator

    with patch(
        "refresh_data.yfinance_client.fetch_price_history", return_value=_price_df("X", rows=3)
    ):
        rows = refresh_data.refresh_cross_asset_macro(session, date(2026, 7, 22))
    session.flush()

    names = {m.indicator_name for m in session.scalars(select(MacroIndicator))}
    assert rows == 4
    assert names == {"vix", "oil_wti", "gold", "dollar_index"}


def test_persist_smart_money_writes_options_short_and_insider(session: Session) -> None:
    from quantpulse.storage.models import InsiderTransaction, OptionsSignal, ShortInterest

    session.add(Ticker(symbol="AAPL", name="Apple Inc.", asset_type="equity", is_active=True))
    session.flush()

    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "insider_name": "Jane Doe",
                "insider_title": "CEO",
                "filing_date": date(2026, 7, 1),
                "transaction_date": date(2026, 7, 1),
                "transaction_code": "P",
                "acquired_disposed_code": "A",
                "shares": 1000.0,
                "price_per_share": 150.0,
                "shares_owned_after": 5000.0,
            }
        ]
    )
    result = refresh_data.TickerFetchResult(
        symbol="AAPL",
        options_signals={
            "symbol": "AAPL",
            "expiration": "2026-08-21",
            "put_call_ratio": 1.2,
            "atm_implied_volatility": 0.3,
        },
        short_interest={"symbol": "AAPL", "pct_float_short": 4.5, "days_to_cover": 2.0},
        insider_df=insider,
    )

    refresh_data._persist_per_ticker_smart_money(session, result, date(2026, 7, 22))
    session.flush()

    options = session.scalars(select(OptionsSignal)).one()
    assert options.put_call_ratio == 1.2
    assert options.iv_rank is None  # no prior history to rank against yet
    assert session.scalars(select(ShortInterest)).one().pct_float_short == 4.5
    assert session.scalars(select(InsiderTransaction)).one().transaction_code == "P"


def test_iv_rank_uses_prior_history_point_in_time(session: Session) -> None:
    from quantpulse.storage.models import OptionsSignal

    session.add(Ticker(symbol="AAPL", name="Apple Inc.", asset_type="equity", is_active=True))
    session.flush()
    # Two prior daily IV snapshots, both below today's 0.30 -> today ranks at the top.
    for d, iv in [(date(2026, 7, 20), 0.10), (date(2026, 7, 21), 0.20)]:
        session.add(
            OptionsSignal(symbol="AAPL", date=d, atm_implied_volatility=iv, put_call_ratio=1.0)
        )
    session.flush()

    result = refresh_data.TickerFetchResult(
        symbol="AAPL",
        options_signals={
            "symbol": "AAPL",
            "expiration": None,
            "put_call_ratio": 1.0,
            "atm_implied_volatility": 0.30,
        },
    )
    refresh_data._persist_per_ticker_smart_money(session, result, date(2026, 7, 22))
    session.flush()

    today_row = session.scalars(
        select(OptionsSignal).where(OptionsSignal.date == date(2026, 7, 22))
    ).one()
    assert today_row.iv_rank == 100.0  # above both prior snapshots


def test_refresh_static_config_writes_baskets_and_calendar(session: Session) -> None:
    from quantpulse.storage.models import EconomicCalendarEvent, ThematicBasket

    rows = refresh_data.refresh_static_config(session, date(2026, 7, 22))
    session.flush()

    baskets = session.scalars(select(ThematicBasket)).all()
    assert rows > 0
    assert any(b.theme_name == "ai_theme" and b.symbol == "NVDA" for b in baskets)
    assert session.scalars(select(EconomicCalendarEvent)).all()  # upcoming events populated


def test_refresh_market_regime_computes_from_stored_inputs(session: Session) -> None:
    from quantpulse.storage.models import MacroIndicator, MarketRegime

    as_of = date(2026, 7, 22)
    session.add(Ticker(symbol="AAPL", name="Apple Inc.", asset_type="equity", is_active=True))
    # A VIX history + level, and a normal (non-inverted) yield curve.
    for i in range(40):
        session.add(
            MacroIndicator(
                date=date(2026, 6, 1) + timedelta(days=i), indicator_name="vix", value=15.0
            )
        )
    session.add(MacroIndicator(date=as_of, indicator_name="vix", value=15.0))
    session.add(MacroIndicator(date=as_of, indicator_name="DGS10", value=4.5))
    session.add(MacroIndicator(date=as_of, indicator_name="DGS2", value=4.0))
    # 210 rising bars so AAPL is above its 200-DMA -> breadth 100.
    for offset in range(210):
        session.add(
            PriceHistory(
                symbol="AAPL",
                date=as_of - timedelta(days=210 - offset),
                open=1.0,
                high=1.0,
                low=1.0,
                close=float(offset + 1),
                adj_close=float(offset + 1),
                volume=1,
            )
        )
    session.flush()

    with patch(
        "refresh_data.gdelt_client.fetch_tone_timeline",
        return_value=pd.DataFrame([{"date": as_of, "tone": 2.0, "query": "q"}]),
    ):
        refresh_data.refresh_market_regime(session, as_of)
    session.flush()

    regime = session.scalars(select(MarketRegime)).one()
    assert regime.breadth_pct_above_200dma == 100.0
    assert regime.yield_curve_spread == pytest.approx(0.5)
    assert regime.regime_score is not None
    assert regime.regime_label in {"risk_on", "neutral", "risk_off"}


def test_process_tier1_news_produces_sentiment_and_events() -> None:
    from quantpulse.news_intelligence.event_classifier import EventClassification, EventType
    from quantpulse.news_intelligence.sentiment import SentimentScore

    as_of = date(2026, 7, 22)
    universe = pd.DataFrame([{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"}])
    news = pd.DataFrame(
        [
            {
                "title": "Apple beats earnings",
                "link": "https://ex.com/a",
                "summary": "",
                "published_at": pd.Timestamp(as_of),
                "source": "yahoo",
                "symbol": "AAPL",
                "tier": 1,
            }
        ]
    )
    result = refresh_data.TickerFetchResult(symbol="AAPL", tier1_news_df=news)

    classification = EventClassification(EventType.EARNINGS, 0.9, {}, 5.0)
    with (
        patch(
            "refresh_data.entity_extraction.tag_articles",
            return_value=pd.Series([["AAPL"]], index=news.index),
        ),
        patch(
            "refresh_data.event_classifier.classify_articles",
            return_value=pd.Series([classification], index=news.index),
        ),
        patch(
            "refresh_data.sentiment.score_articles",
            return_value=pd.Series(
                [SentimentScore(polarity=0.6, positive=0.7, negative=0.1, neutral=0.2)],
                index=news.index,
            ),
        ),
    ):
        sentiment_records, news_records = refresh_data.process_tier1_news([result], universe, as_of)

    assert len(news_records) == 1
    assert news_records[0]["matched_symbols"] == ["AAPL"]
    assert news_records[0]["event_type"] == "earnings"
    assert len(sentiment_records) == 1
    assert sentiment_records[0]["symbol"] == "AAPL"
    assert sentiment_records[0]["sentiment_score"] == pytest.approx(0.6)


def test_refresh_tier2_news_writes_themed_events(session: Session) -> None:
    from quantpulse.news_intelligence.event_classifier import EventClassification, EventType
    from quantpulse.news_intelligence.sentiment import SentimentScore
    from quantpulse.storage.models import NewsEvent

    gdelt_articles = pd.DataFrame(
        [
            {
                "title": "New AI chip export controls announced",
                "url": "https://ex.com/ai",
                "domain": "ex.com",
                "published_at": pd.Timestamp(date(2026, 7, 22)),
                "source_country": "US",
                "language": "eng",
                "query": "ai",
            }
        ]
    )
    classification = EventClassification(EventType.REGULATORY_LEGAL, 0.8, {}, 14.0)
    with (
        patch("refresh_data.gdelt_client.fetch_articles", return_value=gdelt_articles),
        patch(
            "refresh_data.event_classifier.classify_articles",
            return_value=pd.Series([classification], index=gdelt_articles.index),
        ),
        patch(
            "refresh_data.sentiment.score_articles",
            return_value=pd.Series(
                [SentimentScore(polarity=-0.3, positive=0.1, negative=0.4, neutral=0.5)],
                index=gdelt_articles.index,
            ),
        ),
    ):
        rows = refresh_data.refresh_tier2_news(session, date(2026, 7, 22))
    session.flush()

    events = session.scalars(select(NewsEvent).where(NewsEvent.tier == 2)).all()
    assert rows >= 1
    assert any(e.matched_theme is not None and e.event_type == "regulatory/legal" for e in events)


def test_reit_ffo_is_computed_and_stored_in_snapshot(session: Session) -> None:
    from quantpulse.storage.models import FundamentalsSnapshot

    # A REIT's FFO inputs -> P/FFO = market_cap / (net_income + D&A).
    fundamentals = {"symbol": "O", "pe": 40.0, "div_yield": 0.05}
    ffo_inputs = {
        "symbol": "O",
        "market_cap": 1000.0,
        "net_income": 60.0,
        "depreciation_amortization": 40.0,
    }
    enriched = refresh_data._fundamentals_with_ffo(fundamentals, ffo_inputs)
    assert enriched["sector_specific_metrics"] == {"p_ffo": 10.0}  # 1000 / (60 + 40)

    session.add(Ticker(symbol="O", name="Realty Income", asset_type="equity", is_active=True))
    refresh_data._upsert_fundamentals(session, "O", date(2026, 7, 22), enriched)
    session.flush()
    stored = session.scalars(select(FundamentalsSnapshot)).one()
    assert stored.sector_specific_metrics == {"p_ffo": 10.0}


def test_ffo_inputs_only_fetched_for_reits() -> None:
    with (
        patch("refresh_data.yfinance_client.fetch_price_history", return_value=_price_df("AAPL")),
        patch("refresh_data.options_client.fetch_options_signals", return_value={}),
        patch("refresh_data.yfinance_client.fetch_fundamentals", return_value={"symbol": "AAPL"}),
        patch("refresh_data.yfinance_client.fetch_analyst_consensus", return_value={}),
        patch("refresh_data.short_interest_client.fetch_short_interest", return_value={}),
        patch("refresh_data.edgar_client.fetch_insider_transactions", return_value=pd.DataFrame()),
        patch("refresh_data.news_client.fetch_all_tier1_news", return_value=pd.DataFrame()),
        patch("refresh_data.yfinance_client.fetch_ffo_inputs") as mock_ffo,
    ):
        # A Technology name must not trigger the REIT-only FFO fetch.
        refresh_data.fetch_ticker_data(
            "AAPL", None, is_weekly=True, sector="Information Technology", today=date(2026, 7, 22)
        )
        mock_ffo.assert_not_called()
        # A Real Estate name must.
        refresh_data.fetch_ticker_data(
            "O", None, is_weekly=True, sector="Real Estate", today=date(2026, 7, 22)
        )
        mock_ffo.assert_called_once()


def test_run_skips_on_non_trading_day(engine: Engine) -> None:
    fake_get_session, factory = _fake_session_factory(engine)

    with (
        patch("refresh_data.get_session", fake_get_session),
        patch("refresh_data.is_trading_day", return_value=False),
    ):
        refresh_data.run(job_name="test_run_skip")

    with factory() as session:
        logs = session.scalars(select(RefreshLog)).all()

    assert len(logs) == 1
    assert logs[0].status == "skipped_non_trading_day"


def _one_name_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "SYM0",
                "name": "SYM0",
                "sector": "Technology",
                "industry": "Software",
                "exchange": None,
                "asset_type": "equity",
                "is_active": True,
            }
        ]
    )


def _run_recording_weekly_flag(engine: Engine, **kwargs: object) -> tuple[list[bool], list[str]]:
    """Run once and report the `is_weekly` each ticker fetch was handed."""
    fake_get_session, factory = _fake_session_factory(engine)
    seen: list[bool] = []

    def record(
        symbol: str, last_price_date: date | None, is_weekly: bool, **_: object
    ) -> refresh_data.TickerFetchResult:
        seen.append(is_weekly)
        return refresh_data.TickerFetchResult(symbol=symbol, price_df=_price_df(symbol, rows=1))

    with (
        patch("refresh_data.get_session", fake_get_session),
        patch(
            "refresh_data.wikipedia_client.fetch_sp500_constituents",
            return_value=_one_name_universe(),
        ),
        patch("refresh_data.fetch_ticker_data", side_effect=record),
        patch(
            "refresh_data.edgar_13f_client.fetch_institutional_ownership_trend",
            return_value=_empty_df(list(refresh_data._INSTITUTIONAL_COLUMNS)),
        ),
        patch("refresh_data.gdelt_client.fetch_articles", return_value=_empty_df(["title", "url"])),
        patch(
            "refresh_data.gdelt_client.fetch_tone_timeline",
            return_value=_empty_df(["date", "tone", "query"]),
        ),
    ):
        refresh_data.run(job_name="test_run_overrides", **kwargs)  # type: ignore[arg-type]

    with factory() as session:
        statuses = [log.status for log in session.scalars(select(RefreshLog)).all()]
    return seen, statuses


def test_ignore_market_calendar_runs_on_a_closed_day(engine: Engine) -> None:
    """The catch-up override.

    Without it a database that has missed a run cannot be brought up to date
    except by waiting for the calendar to agree -- which for the weekly branch
    means waiting a week.
    """
    with patch("refresh_data.is_trading_day", return_value=False):
        seen, statuses = _run_recording_weekly_flag(engine, ignore_market_calendar=True)

    assert seen, "the run skipped the closed day instead of overriding it"
    assert "skipped_non_trading_day" not in statuses


def test_a_closed_day_is_still_skipped_by_default(engine: Engine) -> None:
    """Mutation guard: the test above must be testing the override, not the day."""
    with patch("refresh_data.is_trading_day", return_value=False):
        seen, statuses = _run_recording_weekly_flag(engine)

    assert seen == []
    assert statuses == ["skipped_non_trading_day"]


def test_force_weekly_runs_the_weekly_branch_on_a_non_weekly_day(engine: Engine) -> None:
    """The weekly branch is where fundamentals, analyst consensus, news and
    sentiment come from, and it is the branch that has actually failed on a
    runner. Learning whether a fix worked should not take a week.

    The clock is pinned to an ordinary Thursday rather than the weekday constant
    being patched. The constant stopped deciding this when the weekly branch
    moved to "the week's first *trading* day" (US market holidays cluster on
    Mondays, so asking for Monday handed the branch a week off roughly one week
    in ten) -- after which patching it changed nothing, and the test failed the
    first time the real date happened to be a week's first trading day: Tuesday
    2026-09-08, the day after Labor Day.
    """
    ordinary_thursday = datetime(2026, 9, 17, 22, 0)
    assert ordinary_thursday.date().weekday() == 3

    with (
        patch("refresh_data.is_trading_day", return_value=True),
        patch("refresh_data.datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = ordinary_thursday
        forced, _ = _run_recording_weekly_flag(engine, force_weekly=True)
        default, _ = _run_recording_weekly_flag(engine)

    assert forced == [True]
    # The default is untouched: an ordinary day is still a daily run.
    assert default == [False]


# --------------------------------------------------------------------------- #
# Concurrency (Section 21: "worth testing carefully for race conditions").
# The module docstring's design claim -- every ticker's I/O fetch runs
# concurrently in a thread pool, and all DB writes happen afterwards,
# serially -- is exercised here with real threads and a real (if simulated)
# per-ticker failure, rather than just trusted from reading the code.
# --------------------------------------------------------------------------- #


def test_ticker_fetches_run_concurrently_and_survive_one_failure(engine: Engine) -> None:
    fake_get_session, factory = _fake_session_factory(engine)
    symbols = [f"SYM{i}" for i in range(6)]
    universe = pd.DataFrame(
        [
            {
                "symbol": s,
                "name": s,
                "sector": "Technology",
                "industry": "Software",
                "exchange": None,
                "asset_type": "equity",
                "is_active": True,
            }
            for s in symbols
        ]
    )

    active = 0
    max_active = 0
    lock = threading.Lock()
    seen_threads: set[int] = set()

    def fake_fetch_ticker_data(
        symbol: str, last_price_date: date | None, is_weekly: bool, **kwargs: object
    ) -> refresh_data.TickerFetchResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            seen_threads.add(threading.get_ident())
        try:
            if symbol == "SYM3":
                raise RuntimeError("simulated source outage")
            time.sleep(0.05)  # long enough for the pool to genuinely overlap calls
            return refresh_data.TickerFetchResult(symbol=symbol, price_df=_price_df(symbol, rows=1))
        finally:
            with lock:
                active -= 1

    with (
        patch("refresh_data.get_session", fake_get_session),
        patch("refresh_data.is_trading_day", return_value=True),
        patch("refresh_data.wikipedia_client.fetch_sp500_constituents", return_value=universe),
        patch("refresh_data.fetch_ticker_data", side_effect=fake_fetch_ticker_data),
        patch(
            "refresh_data.edgar_13f_client.fetch_institutional_ownership_trend",
            return_value=_empty_df(list(refresh_data._INSTITUTIONAL_COLUMNS)),
        ),
        patch("refresh_data.gdelt_client.fetch_articles", return_value=_empty_df(["title", "url"])),
        patch(
            "refresh_data.gdelt_client.fetch_tone_timeline",
            return_value=_empty_df(["date", "tone", "query"]),
        ),
        # The benchmark step calls the yfinance client directly rather than
        # going through `fetch_ticker_data`, so the patch above does not cover
        # it and an unpatched run reaches the network.
        patch(
            "refresh_data.yfinance_client.fetch_price_history",
            side_effect=lambda symbol, period="5y": _price_df(symbol, rows=3),
        ),
    ):
        refresh_data.run(job_name="test_run_concurrency")

    # Genuine parallelism: more than one fetch was in flight at once, and the
    # work landed on more than one thread -- not the pool silently
    # collapsing to sequential execution on the main thread.
    assert max_active > 1
    assert len(seen_threads) > 1

    with factory() as session:
        prices = session.scalars(select(PriceHistory)).all()
        logs = session.scalars(select(RefreshLog)).all()

    # SYM3's simulated failure didn't stop the other 5 tickers from being
    # fetched and written -- one bad future doesn't take the run down.
    #
    # The market index is written by its own step, not by the per-ticker pool
    # this test is about, so it is removed rather than added to the expectation:
    # the claim is still "exactly the five surviving constituents", not "at
    # least them".
    written_symbols = {p.symbol for p in prices} - {risk.MARKET_INDEX_SYMBOL}
    assert written_symbols == set(symbols) - {"SYM3"}
    assert len(logs) == 1
    assert logs[0].status == "partial"  # downgraded, not aborted (Section 6.12)


def test_equal_weight_benchmark_survives_a_zero_first_price() -> None:
    # One symbol whose first observation is 0.0 used to turn the whole
    # benchmark into `inf` (it is rebased by division), so every
    # strategy-vs-market comparison on the Track Record page was against
    # infinity. Real case: DEC, 732 bars of adj_close = 0.0.
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame(
        {
            "GOOD": [10.0, 11.0, 12.0, 13.0],
            "ALSOGOOD": [50.0, 50.0, 55.0, 60.0],
            "DEC": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    benchmark = refresh_data._equal_weight_benchmark(panel)

    assert np.isfinite(benchmark).all(), "a zero-priced symbol must not make the index infinite"
    assert benchmark.iloc[0] == pytest.approx(1.0)
    # Exactly the mean of the two usable names' rebased levels.
    assert benchmark.iloc[-1] == pytest.approx((13.0 / 10.0 + 60.0 / 50.0) / 2)


def test_equal_weight_benchmark_is_empty_when_no_symbol_is_usable() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    panel = pd.DataFrame({"DEC": [0.0, 0.0, 0.0]}, index=index)

    benchmark = refresh_data._equal_weight_benchmark(panel)

    assert len(benchmark) == 3
    assert benchmark.isna().all(), "no usable name -> no benchmark, not a fabricated one"


class TestCriticalStepsFailLoudly:
    """ "Partial" must not cover for a run that produced nothing usable.

    A real nightly reported `partial` and exited 0 while `composite_scores` --
    every rating in the Screener, on Home and on every Stock Detail page -- was
    completely empty. Prices and options had landed, so "partial" was literally
    true and entirely misleading. News, 13F and macro genuinely are optional:
    ratings still compute without them, at a lower `data_confidence` the UI
    already displays.
    """

    @staticmethod
    def _run_steps(failing: str) -> tuple[str, list[str]]:
        """Drive `run`'s `step` closure in isolation via a tiny stand-in."""
        status = "success"
        failed: list[str] = []

        def step(name: str, fn) -> int:
            nonlocal status
            try:
                return fn()
            except Exception:
                failed.append(name)
                if name in refresh_data._CRITICAL_STEPS:
                    status = "failed"
                elif status != "failed":
                    status = "partial"
                return 0

        def boom() -> int:
            raise RuntimeError("source down")

        for name in ("tier1_news", "composite_scores", "market_regime", "backtest"):
            step(name, boom if name == failing else (lambda: 1))
        return status, failed

    def test_an_optional_source_failing_is_only_partial(self) -> None:
        status, failed = self._run_steps("tier1_news")
        assert status == "partial"
        assert failed == ["tier1_news"]

    def test_composite_scores_failing_makes_the_whole_run_failed(self) -> None:
        status, failed = self._run_steps("composite_scores")
        assert status == "failed"
        assert failed == ["composite_scores"]

    def test_market_regime_is_critical_too(self) -> None:
        assert self._run_steps("market_regime")[0] == "failed"

    def test_a_later_optional_failure_cannot_downgrade_failed_to_partial(self) -> None:
        # Ordering matters: composite_scores fails first, then an optional step.
        # The run must stay "failed" rather than being talked back down.
        status = "success"

        def step(name: str, ok: bool) -> None:
            nonlocal status
            if ok:
                return
            if name in refresh_data._CRITICAL_STEPS:
                status = "failed"
            elif status != "failed":
                status = "partial"

        step("composite_scores", ok=False)
        step("tier1_news", ok=False)
        assert status == "failed"


class TestPooledHitRateWindows:
    """A pooled hit rate is only worth as much as its number of distinct windows.

    Pooling twenty symbols multiplies the graded-pair count twentyfold without
    adding a single window -- they all share one trading calendar, so they are
    twenty readings of the same history. Measured on real data that inflated the
    one-year horizon's apparent sample from 1-3 windows to 20-60 pairs, and the
    ML model's "60% hit rate vs 52% naive" there was twenty correlated readings
    of a single year.
    """

    @staticmethod
    def _frames(n_symbols: int, bars: int) -> dict[str, pd.DataFrame]:
        index = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=bars))
        out: dict[str, pd.DataFrame] = {}
        for i in range(n_symbols):
            rng = np.random.default_rng(100 + i)
            close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, bars)))
            out[f"S{i}"] = pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                },
                index=index,
            )
        return out

    def _rates(self, frames: dict[str, pd.DataFrame], horizons: tuple[int, ...]):
        with (
            patch.object(refresh_data, "_FORECAST_HORIZONS", horizons),
            patch.object(
                refresh_data,
                "_FORECAST_RUNNERS",
                {"baseline": refresh_data._FORECAST_RUNNERS["baseline"]},
            ),
        ):
            return refresh_data._pooled_hit_rates(frames)

    def test_windows_count_periods_not_pooled_pairs(self) -> None:
        # Ten symbols on one shared calendar: the pair count is ten times the
        # window count, and it is the window count that gets reported.
        frames = self._frames(n_symbols=10, bars=700)
        rates = self._rates(frames, (5,))
        assert ("baseline", 5) in rates
        _rate, windows = rates[("baseline", 5)]
        single = self._rates({"S0": frames["S0"]}, (5,))
        assert windows == single[("baseline", 5)][1], (
            "adding nine more symbols over the same dates must not add windows"
        )

    def test_a_thin_horizon_is_not_published_at_all(self) -> None:
        # 700 bars grades plenty of 5-day windows and far too few 252-day ones
        # (the horizon gate alone leaves almost no room), so the long horizon
        # must be absent rather than reported from a handful of periods.
        frames = self._frames(n_symbols=5, bars=700)
        rates = self._rates(frames, (5, 252))
        assert ("baseline", 5) in rates
        assert rates[("baseline", 5)][1] >= bt.MIN_GRADED_WINDOWS
        assert ("baseline", 252) not in rates

    def test_a_published_rate_always_clears_the_minimum(self) -> None:
        frames = self._frames(n_symbols=3, bars=900)
        for (_model, _h), (rate, windows) in self._rates(frames, (5, 20, 63, 252)).items():
            assert 0.0 <= rate <= 1.0
            assert windows >= bt.MIN_GRADED_WINDOWS


class TestPatternSignalsAreProduced:
    """`pattern_signals` had a table, a reader and a panel in both front ends -- and no writer.

    `analysis/patterns.py` (head-and-shoulders, double top/bottom, triangles/
    wedges/channels, cup-and-handle) was never called by anything, so the
    "Detected patterns" panel was permanently empty and the README's "4 pattern
    families detected" described a library rather than the app.
    """

    @staticmethod
    def _seed(session: Session, symbol: str = "AAA") -> pd.DataFrame:
        session.add(
            Ticker(symbol=symbol, name="Alpha", sector="Tech", asset_type="equity", is_active=True)
        )
        # A deliberate double top: rise, peak, dip, matching peak, break down.
        shape = (
            list(np.linspace(100, 140, 60))
            + list(np.linspace(140, 118, 30))
            + list(np.linspace(118, 139, 30))
            + list(np.linspace(139, 105, 40))
        )
        for offset, close in enumerate(shape):
            session.add(
                PriceHistory(
                    symbol=symbol,
                    date=date(2026, 7, 22) - timedelta(days=len(shape) - offset),
                    open=close,
                    high=close * 1.005,
                    low=close * 0.995,
                    close=close,
                    adj_close=close,
                    volume=1_000_000,
                )
            )
        session.flush()
        return pd.DataFrame([{"symbol": symbol, "name": "Alpha", "sector": "Tech"}])

    def test_detected_patterns_are_stored(self, session: Session) -> None:
        universe = self._seed(session)
        written = refresh_data.refresh_pattern_signals(session, universe, date(2026, 7, 22))
        session.flush()
        rows = session.scalars(select(PatternSignal)).all()
        assert written > 0
        assert rows
        assert {r.direction for r in rows} <= {"bullish", "bearish", "neutral"}
        assert all(0.0 <= r.confidence <= 100.0 for r in rows)

    def test_a_rerun_does_not_duplicate_the_same_formation(self, session: Session) -> None:
        # `pattern_signals` has an autoincrement id, so a bare ON CONFLICT DO
        # NOTHING would happily insert every formation again every night.
        universe = self._seed(session)
        refresh_data.refresh_pattern_signals(session, universe, date(2026, 7, 22))
        session.flush()
        before = session.scalars(select(PatternSignal)).all()
        refresh_data.refresh_pattern_signals(session, universe, date(2026, 7, 22))
        session.flush()
        after = session.scalars(select(PatternSignal)).all()
        assert len(after) == len(before)

    def test_no_price_history_writes_nothing_rather_than_raising(self, session: Session) -> None:
        session.add(
            Ticker(symbol="ZZZ", name="Zed", sector="Tech", asset_type="equity", is_active=True)
        )
        session.flush()
        universe = pd.DataFrame([{"symbol": "ZZZ", "name": "Zed", "sector": "Tech"}])
        assert refresh_data.refresh_pattern_signals(session, universe, date(2026, 7, 22)) == 0


class TestTrackRecordSuppression:
    """A run too short to bracket is not a track record.

    `cagr` raises a period's growth to the power of the periods per year, so two
    monthly periods over five weeks annualize into a headline. On the deployed
    demo database that produced a benchmark "CAGR" of 26.6% from returns of
    +1.6% and +2.4%, next to a strategy CAGR of 0.0% that came from a signal
    with too little history to rank anything -- so it held cash and never
    traded. The Track Record page said "the run was too short to bootstrap
    honestly" directly beneath both numbers.
    """

    @staticmethod
    def _seed(session: Session, *, days: int) -> None:
        rng = np.random.default_rng(4)
        for symbol in ("AAA", "BBB", "CCC"):
            session.add(
                Ticker(
                    symbol=symbol, name=symbol, sector="Tech", asset_type="equity", is_active=True
                )
            )
            session.add(
                IndexMembershipHistory(
                    index_name="S&P 500",
                    symbol=symbol,
                    added_date=date(2020, 1, 1),
                    removed_date=None,
                )
            )
            level = 100.0
            for offset in range(days, 0, -1):
                level *= float(np.exp(rng.normal(0.0004, 0.012)))
                session.add(
                    PriceHistory(
                        symbol=symbol,
                        date=date(2026, 7, 22) - timedelta(days=offset),
                        open=level,
                        high=level,
                        low=level,
                        close=level,
                        adj_close=level,
                        volume=1_000_000,
                    )
                )
        session.flush()

    def test_a_two_period_run_is_not_stored(self, session: Session) -> None:
        self._seed(session, days=40)
        assert refresh_data.refresh_backtest(session, date(2026, 7, 22)) == 0

    def test_a_run_where_the_strategy_never_traded_is_not_stored(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Long enough history for plenty of periods, but a signal that ranks
        # nothing -- so every period sits in cash. Storing that as a 0% track
        # record reads as "the strategy lost to the market"; it never ran.
        self._seed(session, days=900)
        monkeypatch.setattr(refresh_data, "_momentum_signal", lambda as_of, panel: {})
        assert refresh_data.refresh_backtest(session, date(2026, 7, 22)) == 0

    def test_a_run_that_sat_in_cash_for_most_periods_is_not_stored(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The half of this failure the "never traded" guard above misses.

        Real shape, found on real data: three years of prices behind an index
        membership history only days deep, so the point-in-time universe came
        back empty for 38 of 39 monthly periods. The strategy held cash for all
        of them, took one position at the end, and the stored run read Sharpe
        0.555 / CAGR 0.99% against a 29.3% benchmark -- which reads as "we
        underperformed the market" rather than "nothing was ever tested".
        Average turnover was 1/39, so it sailed past `avg_turnover <= 0`.
        """
        self._seed(session, days=900)
        real_signal = refresh_data._momentum_signal
        cutoff = date(2026, 5, 1)
        monkeypatch.setattr(
            refresh_data,
            "_momentum_signal",
            lambda as_of, panel: real_signal(as_of, panel) if as_of >= cutoff else {},
        )
        assert refresh_data.refresh_backtest(session, date(2026, 7, 22)) == 0

    def test_a_long_enough_run_that_traded_is_stored(self, session: Session) -> None:
        from quantpulse.storage.models import BacktestResult

        self._seed(session, days=900)
        written = refresh_data.refresh_backtest(session, date(2026, 7, 22))
        session.flush()
        assert written == 1
        stored = session.scalars(select(BacktestResult)).one()
        assert stored.n_periods >= bt.MIN_TRACK_RECORD_PERIODS
        assert stored.avg_turnover > 0


class TestStepBudget:
    """A stalled step must cost that step, not the whole night.

    The failure this guards against actually happened: the weekly news step
    classified every article it could fetch through a local NLI model, emitted
    no output for 5h38m, and was cancelled at GitHub's 6-hour job limit. Because
    the job was *killed* rather than failing, the workflow's "commit the
    refreshed database" step never ran either -- so a run that had already
    fetched every price and fundamental successfully committed nothing at all.
    A hang is not an exception, so `step()`'s `except` could never have caught
    it; only a wall-clock budget can.
    """

    def test_a_stalled_step_times_out_and_the_run_continues(self) -> None:
        ran_after = []

        def _stall() -> int:
            time.sleep(30)  # far beyond the 1s budget injected below
            return 99

        with (
            patch.dict(refresh_data._STEP_TIMEOUT_SECONDS, {"slow": 1}, clear=False),
            patch.object(refresh_data, "_DEFAULT_STEP_TIMEOUT_SECONDS", 1),
        ):
            started = time.monotonic()
            status, rows, failed = _drive_steps(
                [("slow", _stall), ("after", lambda: ran_after.append(True) or 7)]
            )
            elapsed = time.monotonic() - started

        assert elapsed < 20, "the budget did not interrupt the stall"
        assert failed == ["slow"]
        assert status == "partial"
        # The decisive assertion: work queued behind the stall still happened.
        assert ran_after == [True]
        assert rows == 7

    def test_a_timeout_in_a_critical_step_fails_the_run(self) -> None:
        # `_CRITICAL_STEPS` must keep meaning "the app is unusable without
        # this", whether the step raised or simply never returned.
        with (
            patch.dict(refresh_data._STEP_TIMEOUT_SECONDS, {"composite_scores": 1}, clear=False),
        ):
            status, _, failed = _drive_steps([("composite_scores", lambda: time.sleep(30))])

        assert failed == ["composite_scores"]
        assert status == "failed"

    def test_a_fast_step_is_unaffected(self) -> None:
        status, rows, failed = _drive_steps([("quick", lambda: 5)])
        assert (status, rows, failed) == ("success", 5, [])

    def test_the_alarm_is_disarmed_afterwards(self) -> None:
        import signal

        _drive_steps([("quick", lambda: 1)])
        # A leaked alarm would fire during some unrelated later step.
        assert signal.alarm(0) == 0


def _drive_steps(steps: list[tuple[str, object]]) -> tuple[str, int, list[str]]:
    """Run `step()`'s real body over `steps`, returning (status, rows, failed).

    `run()` itself needs a database, a universe and network mocks; this exercises
    the isolation/budget logic alone, which is what the tests above are about.
    Mirrors `run()`'s closure exactly.
    """
    status = "success"
    rows_updated = 0
    failed_steps: list[str] = []

    def step(name: str, fn: object) -> int:
        nonlocal status
        budget = refresh_data._STEP_TIMEOUT_SECONDS.get(
            name, refresh_data._DEFAULT_STEP_TIMEOUT_SECONDS
        )
        try:
            with refresh_data._step_timeout(budget, name):
                return fn() or 0  # type: ignore[operator]
        except Exception:
            failed_steps.append(name)
            if name in refresh_data._CRITICAL_STEPS:
                status = "failed"
            elif status != "failed":
                status = "partial"
            return 0

    for name, fn in steps:
        rows_updated += step(name, fn)
    return status, rows_updated, failed_steps


def test_news_runs_after_the_steps_a_visitor_actually_looks_at() -> None:
    """Ordering is load-bearing, so it is pinned rather than left to review.

    News is the most expensive step and the only one that has ever exhausted
    the job's budget. When it sat mid-run, everything behind it -- regime,
    patterns, composite scores, forecasts, backtest -- never ran at all.
    """
    source = Path(refresh_data.__file__).read_text()
    order = {
        name: source.index(f'step(\n                "{name}"')
        if f'step(\n                "{name}"' in source
        else source.index(f'"{name}"')
        for name in (
            "market_regime",
            "pattern_signals",
            "composite_scores",
            "forecasts",
            "backtest",
            "tier1_news",
            "tier2_news",
        )
    }
    for earlier in (
        "market_regime",
        "pattern_signals",
        "composite_scores",
        "forecasts",
        "backtest",
    ):
        assert order[earlier] < order["tier1_news"], (
            f"{earlier} must be dispatched before tier1_news -- a news stall "
            "otherwise takes the whole weekly branch down with it"
        )
    assert order["tier1_news"] < order["tier2_news"]


# --------------------------------------------------------------------------- #
# The searchable catalogue (Ticker.coverage).
#
# The nightly job can afford to score a few hundred names; the US market lists
# about 13,000. The catalogue records the rest as *names only* so they can be
# searched for and analysed on demand. Two properties make that safe, and both
# are the kind that fail silently if they break: a ranked symbol must never be
# demoted to a name, and 12,500 new rows must not leak into any fetch loop,
# ranking, or survivorship-aware backtest.
# --------------------------------------------------------------------------- #


def _listings(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "name", "exchange", "asset_type"])


def test_catalogue_records_symbols_the_ranked_universe_does_not_have(session: Session) -> None:
    session.add(Ticker(symbol="AAPL", name="Apple", asset_type="equity", is_active=True))
    session.flush()

    listings = _listings(
        ("AAPL", "Apple Inc.", "Nasdaq", "equity"),
        ("RKLB", "Rocket Lab", "Nasdaq", "equity"),
        ("SPY", "SPDR S&P 500 ETF", "NYSE Arca", "etf"),
    )
    with patch("refresh_data.listing_client.fetch_us_listings", return_value=listings):
        added = refresh_data.sync_catalogue(session)

    tickers = {t.symbol: t for t in session.scalars(select(Ticker))}
    assert added == 2
    assert tickers["RKLB"].coverage == Ticker.CATALOGUE
    assert tickers["SPY"].coverage == Ticker.CATALOGUE


def test_catalogue_never_demotes_a_ranked_symbol(session: Session) -> None:
    """The ranked row has three years of prices behind it; the listing row has a name.

    The two sources disagree constantly -- on share classes, on recent index
    changes, on how a company is spelled. Whenever they do, the row with data
    behind it has to win, or a nightly sync quietly drops a scored stock out of
    the Screener.
    """
    session.add(
        Ticker(
            symbol="AAPL",
            name="Apple",
            sector="Information Technology",
            asset_type="equity",
            is_active=True,
            coverage=Ticker.RANKED,
        )
    )
    session.flush()

    with patch(
        "refresh_data.listing_client.fetch_us_listings",
        return_value=_listings(("AAPL", "Apple Inc. - Common Stock", "Nasdaq", "equity")),
    ):
        added = refresh_data.sync_catalogue(session)

    apple = session.get(Ticker, "AAPL")
    assert added == 0
    assert apple is not None
    assert apple.coverage == Ticker.RANKED
    assert apple.is_active is True
    assert apple.sector == "Information Technology"  # not blanked by the listing row


def test_catalogue_rows_stay_out_of_the_active_universe(session: Session) -> None:
    """The one property that keeps a 13,000-row catalogue from becoming a
    13,000-ticker nightly job.

    Every existing reader and the whole fetch loop filter on `is_active`, so a
    catalogue row being active is not a cosmetic mistake -- it is thousands of
    price fetches, a ranking full of unscored names, and bogus S&P 500
    membership intervals.
    """
    session.add(Ticker(symbol="AAPL", name="Apple", asset_type="equity", is_active=True))
    session.flush()
    with patch(
        "refresh_data.listing_client.fetch_us_listings",
        return_value=_listings(*[(f"SYM{i}", f"Co {i}", "Nasdaq", "equity") for i in range(50)]),
    ):
        refresh_data.sync_catalogue(session)

    active = refresh_data._active_universe(session)
    assert list(active["symbol"]) == ["AAPL"]
    assert len(session.scalars(select(Ticker)).all()) == 51


def test_catalogue_sync_is_idempotent(session: Session) -> None:
    listings = _listings(("RKLB", "Rocket Lab", "Nasdaq", "equity"))
    with patch("refresh_data.listing_client.fetch_us_listings", return_value=listings):
        first = refresh_data.sync_catalogue(session)
        second = refresh_data.sync_catalogue(session)
    assert (first, second) == (1, 0)


class TestAStepThatWritesNothingIsNotASuccess:
    """ "Zero rows, no exception" was the shape the 13F bug wore for months.

    `refresh_institutional_ownership` caught its own 404, logged it and returned
    0. `step()` saw a clean return, so a source that had *never once* produced a
    row was indistinguishable from a source that simply had nothing new. The run
    said "partial" for unrelated reasons and nobody looked.

    These drive the real `run()` rather than a stand-in `step`, because the
    behaviour under test lives in that closure -- a reimplementation in the test
    would pass whatever the real one did.
    """

    @staticmethod
    def _run_with_institutional(engine: Engine, rows: int) -> tuple[str, list[str]]:
        fake_get_session, factory = _fake_session_factory(engine)
        records: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _Capture()
        logging.getLogger("refresh_data").addHandler(handler)
        try:
            with (
                patch("refresh_data.get_session", fake_get_session),
                patch(
                    "refresh_data.wikipedia_client.fetch_sp500_constituents",
                    return_value=_one_name_universe(),
                ),
                patch(
                    "refresh_data.fetch_ticker_data",
                    side_effect=lambda symbol, *a, **k: refresh_data.TickerFetchResult(
                        symbol=symbol, price_df=_price_df(symbol, rows=1)
                    ),
                ),
                patch("refresh_data.refresh_institutional_ownership", return_value=rows),
                patch(
                    "refresh_data.yfinance_client.fetch_price_history",
                    side_effect=lambda symbol, period="5y": _price_df(symbol, rows=3),
                ),
                patch(
                    "refresh_data.gdelt_client.fetch_articles",
                    return_value=_empty_df(["title", "url"]),
                ),
                patch(
                    "refresh_data.gdelt_client.fetch_tone_timeline",
                    return_value=_empty_df(["date", "tone", "query"]),
                ),
            ):
                # `ignore_market_calendar` so the assertion is about the step's
                # own emptiness on any day of the week, not about the calendar.
                refresh_data.run(
                    job_name="test_empty_step",
                    force_weekly=True,
                    ignore_market_calendar=True,
                )
        finally:
            logging.getLogger("refresh_data").removeHandler(handler)

        with factory() as session:
            log = session.scalars(select(RefreshLog)).all()[-1]
        return log.status, records

    def test_an_empty_13f_step_downgrades_the_run(self, engine: Engine) -> None:
        status, messages = self._run_with_institutional(engine, rows=0)
        assert status == "partial", (
            "a 13F step that wrote nothing reported a clean run -- which is exactly "
            "how six months of empty institutional ownership went unnoticed"
        )
        assert any("wrote no rows" in m for m in messages), (
            f"nothing in the log named the empty step; messages were {messages}"
        )
        assert any("institutional_ownership" in m and "wrote nothing" in m for m in messages), (
            "the run summary did not name which step wrote nothing"
        )

    def test_a_13f_step_that_writes_rows_does_not(self, engine: Engine) -> None:
        """The other half, or the assertion above is satisfied by any run at all."""
        status, messages = self._run_with_institutional(engine, rows=42)
        assert status == "success"
        assert not any("wrote no rows" in m for m in messages)

    def test_membership_is_narrow_on_purpose(self) -> None:
        """Steps with honest empty runs must stay out, or the signal becomes noise.

        `backtest` explicitly declines to store a run that would describe
        nothing, `tier2_news` finds no matching articles some days, and
        `macro_indicators` is empty without a FRED key. Listing any of them
        would mark a correct run degraded and train a reader to ignore the flag.
        """
        assert refresh_data._STEPS_EXPECTED_TO_WRITE == {
            "institutional_ownership",
            "benchmark_prices",
        }
        assert not (refresh_data._STEPS_EXPECTED_TO_WRITE & refresh_data._CRITICAL_STEPS)


class TestInstitutionalOwnershipAsksWhichWindowExists:
    """The pipeline must *use* the resolver, not merely have one available.

    Unit tests for `latest_published_window` pass whether or not
    `refresh_institutional_ownership` calls it -- reverting the call site to
    `quarter_window_for(today)` left every one of them green. The bug was
    always in the caller, so the assertion has to be there too.
    """

    @staticmethod
    def _fetch_window(engine: Engine, resolved: tuple[date, date] | None) -> tuple[int, list]:
        fake_get_session, factory = _fake_session_factory(engine)
        universe = _one_name_universe()[["symbol", "name", "sector"]]
        with (
            patch("refresh_data.get_session", fake_get_session),
            patch.object(
                refresh_data.edgar_13f_client, "latest_published_window", return_value=resolved
            ),
            patch.object(
                refresh_data.edgar_13f_client,
                "fetch_institutional_ownership_trend",
                return_value=_empty_df(list(refresh_data._INSTITUTIONAL_COLUMNS)),
            ) as fetch,
        ):
            with factory() as session:
                rows = refresh_data.refresh_institutional_ownership(
                    session, universe, date(2026, 9, 5)
                )
        return rows, fetch.call_args_list

    def test_it_fetches_the_resolved_window_not_the_computed_one(self, engine: Engine) -> None:
        today = date(2026, 9, 5)
        resolved = (date(2026, 3, 1), date(2026, 5, 31))
        _, calls = self._fetch_window(engine, resolved)

        assert len(calls) == 1
        assert calls[0].args[0] == resolved
        # The window the old code computed, named explicitly: on this date it is
        # Sep-Nov 2026, a quarter that has not finished and that SEC therefore
        # has not published. Fetching it is the whole bug.
        assert calls[0].args[0] != refresh_data.edgar_13f_client.quarter_window_for(today)

    def test_no_published_window_writes_nothing_and_downloads_nothing(self, engine: Engine) -> None:
        """`None` means give up, not guess -- and certainly not start a 100MB download."""
        rows, calls = self._fetch_window(engine, None)
        assert rows == 0
        assert calls == []


class TestIndustryMacroCoversTheUniverse:
    """`industry_macro` reached 24 of ~500 names; these pin both halves of the fix.

    The category carries 8-10% of the composite weight in every investor
    profile, but a symbol only gets a value if (a) it is in a basket and (b)
    Tier-2 articles are tagged to that basket. Only the six curated themes
    satisfied either, so for 95% of the index the weight was renormalized away
    and the advertised seven-category composite was a six-category one.
    """

    @staticmethod
    def _universe_frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"symbol": "AAA", "name": "AAA Inc", "sector": "Financials"},
                {"symbol": "BBB", "name": "BBB Inc", "sector": "Financials"},
                {"symbol": "CCC", "name": "CCC Inc", "sector": "Utilities"},
            ]
        )

    def test_static_config_writes_a_basket_for_every_sector(self, engine: Engine) -> None:
        fake_get_session, factory = _fake_session_factory(engine)
        with (
            patch("refresh_data.get_session", fake_get_session),
            patch("refresh_data._active_universe", return_value=self._universe_frame()),
        ):
            with factory() as session:
                refresh_data.refresh_static_config(session, date(2026, 9, 5))
                session.commit()

        with factory() as session:
            members = persistence.read_theme_members(session)

        covered = set().union(*members.values()) if members else set()
        assert {"Financials", "Utilities"} <= set(members)
        assert members["Financials"] == {"AAA", "BBB"}
        # The property that matters: nothing in the universe is left basketless.
        assert {"AAA", "BBB", "CCC"} <= covered

    def _run_tier2(self, engine: Engine, articles_per_basket: int) -> tuple[list, dict]:
        """Drive `refresh_tier2_news` with a fixed article count per basket."""
        fake_get_session, factory = _fake_session_factory(engine)

        def fake_articles(query: str, **_: object) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "title": f"story {i} for {query[:18]}",
                        "url": f"https://example.test/{abs(hash(query))}/{i}",
                        "published_at": datetime(2026, 9, 5, 12, 0)
                        - timedelta(minutes=i),  # newest first is i=0
                    }
                    for i in range(articles_per_basket)
                ]
            )

        def fake_classify(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"event_type": ["earnings"] * len(frame)})

        def fake_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"polarity": [0.25] * len(frame)})

        with (
            patch("refresh_data.get_session", fake_get_session),
            patch("refresh_data._active_universe", return_value=self._universe_frame()),
            patch("refresh_data.gdelt_client.fetch_articles", side_effect=fake_articles),
            patch(
                "refresh_data.event_classifier.classify_articles", side_effect=fake_classify
            ) as classify,
            patch("refresh_data.sentiment.score_articles", side_effect=fake_sentiment),
        ):
            with factory() as session:
                refresh_data.refresh_tier2_news(session, date(2026, 9, 5))
                session.commit()

        with factory() as session:
            rows = session.scalars(sa.select(NewsEvent).where(NewsEvent.tier == 2)).all()
        return classify.call_args_list, {
            "themes": {r.matched_theme for r in rows},
            "total": len(rows),
            "classified": sum(1 for r in rows if r.event_type is not None),
            "scored": sum(1 for r in rows if r.sentiment_score is not None),
        }

    def test_sector_baskets_are_queried_not_only_curated_themes(self, engine: Engine) -> None:
        _, seen = self._run_tier2(engine, articles_per_basket=3)
        assert {"Financials", "Utilities"} <= seen["themes"], (
            "no Tier-2 article was tagged to a sector, so every sector basket would "
            "cover its members with news that does not exist"
        )
        assert "ai_theme" in seen["themes"], "the curated themes must keep working too"

    def test_classification_is_capped_per_basket_but_sentiment_is_not(self, engine: Engine) -> None:
        """The trade that makes sector coverage affordable, asserted as a ratio.

        Measured on real articles: classifying costs 192 ms each against
        sentiment's 21 ms, and `event_type` is read only by the Dashboard's 8
        most recent stories while `sentiment_score` feeds every stock's tilt.
        """
        over = refresh_data._MAX_CLASSIFIED_TIER2_PER_BASKET + 15
        calls, seen = self._run_tier2(engine, articles_per_basket=over)

        assert seen["scored"] == seen["total"], "every article must carry a sentiment score"
        assert seen["classified"] < seen["total"], (
            "every article was classified -- the cap is not in effect, and the weekly "
            "news budget is back to being spent on labels nothing reads"
        )
        assert all(
            len(frame) <= refresh_data._MAX_CLASSIFIED_TIER2_PER_BASKET
            for (frame,) in (c.args for c in calls)
        )

    def test_the_classified_articles_are_the_newest_ones(self, engine: Engine) -> None:
        """The Dashboard shows the 8 most recent, so the cap must keep those labelled."""
        over = refresh_data._MAX_CLASSIFIED_TIER2_PER_BASKET + 15
        calls, _ = self._run_tier2(engine, articles_per_basket=over)
        for (frame,) in (c.args for c in calls):
            stamps = list(frame["published_at"])
            assert stamps == sorted(stamps, reverse=True)
            assert stamps[0] == datetime(2026, 9, 5, 12, 0), (
                "the newest article in the basket was not among those classified"
            )

    def test_news_themes_and_basket_membership_use_the_same_names(self, engine: Engine) -> None:
        """The join is by string, and a mismatch fails silently as "no industry signal".

        `refresh_static_config` writes basket membership and `refresh_tier2_news`
        writes `matched_theme`; `scoring.tier2_thematic_tilt` joins them by name.
        If the two ever disagree, every symbol keeps a basket and every article
        keeps a theme, and the category quietly goes back to None for everyone --
        the original bug wearing a full `thematic_baskets` table.
        """
        fake_get_session, factory = _fake_session_factory(engine)
        with (
            patch("refresh_data.get_session", fake_get_session),
            patch("refresh_data._active_universe", return_value=self._universe_frame()),
        ):
            with factory() as session:
                refresh_data.refresh_static_config(session, date(2026, 9, 5))
                session.commit()
        with factory() as session:
            membership_names = set(persistence.read_theme_members(session))

        _, seen = self._run_tier2(engine, articles_per_basket=2)
        assert seen["themes"] <= membership_names, (
            f"themes tagged on articles but absent from thematic_baskets: "
            f"{seen['themes'] - membership_names}"
        )


class TestTheBacktestRanksWhatThePageClaims:
    """The Track Record page said "followed the algorithm's ratings" and did not.

    Every stored run was ranked by a hand-rolled trailing return while the page
    described itself as a track record of the app's ratings, and nothing in the
    row or on the screen could have told a reader otherwise. Two things follow:
    the signal is now the app's own scorer, and the run records which one it was.
    """

    def test_the_signal_is_the_apps_own_momentum_scorer(self) -> None:
        """Not a proxy for `score_momentum` — the same function, same answer.

        Asserted against `scoring.score_momentum` itself rather than against a
        recomputed formula, so the backtest cannot drift away from the scorer the
        nightly composite calls without this failing.
        """
        rng = np.random.default_rng(7)
        n = scoring.MOMENTUM_WINDOW + 40
        panel = pd.DataFrame(
            {
                "AAA": 100.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.01, n))),
                "BBB": 100.0 * np.exp(np.cumsum(rng.normal(-0.0005, 0.02, n))),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )
        signal = refresh_data._momentum_signal(date(2026, 9, 5), panel)

        assert set(signal) == {"AAA", "BBB"}
        for symbol in ("AAA", "BBB"):
            expected = scoring.score_momentum(
                panel[symbol].tail(scoring.MOMENTUM_WINDOW + 1).to_frame(name="close")
            )
            assert signal[symbol] == pytest.approx(expected)

    def test_it_is_not_the_old_trailing_return(self) -> None:
        """A risk-adjusted score ranks differently from a raw return, or nothing changed.

        Two names built so the higher raw return is the *more volatile* one: a
        trailing-return ranking prefers it, a risk-adjusted one does not. If both
        rankings agreed here the switch would be cosmetic.
        """
        n = scoring.MOMENTUM_WINDOW + 5
        rng = np.random.default_rng(3)
        calm = 100.0 * np.exp(np.cumsum(rng.normal(0.0010, 0.004, n)))
        wild = 100.0 * np.exp(np.cumsum(rng.normal(0.0018, 0.040, n)))
        panel = pd.DataFrame(
            {"CALM": calm, "WILD": wild},
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )
        signal = refresh_data._momentum_signal(date(2026, 9, 5), panel)
        raw_return = {s: float(panel[s].iloc[-1] / panel[s].iloc[0] - 1.0) for s in panel.columns}

        assert raw_return["WILD"] > raw_return["CALM"], "fixture: WILD must win on raw return"
        assert signal["CALM"] > signal["WILD"], (
            "the risk-adjusted score ranked the same way a raw trailing return does, so "
            "this fixture cannot tell the two signals apart"
        )

    def test_it_uses_scorings_own_window_not_a_second_copy(self) -> None:
        """A shorter frame silently truncates `score_momentum`'s 126-bar window.

        The backtest would then publish a different momentum score than the
        nightly composite computes for the same prices -- the same shape of bug
        as the three copies of the beta window.
        """
        n = scoring.MOMENTUM_WINDOW + 60
        rng = np.random.default_rng(11)
        series = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.012, n)))
        panel = pd.DataFrame(
            {"AAA": series}, index=pd.date_range("2024-01-01", periods=n, freq="B")
        )
        from_backtest = refresh_data._momentum_signal(date(2026, 9, 5), panel)["AAA"]
        from_scorer = scoring.score_momentum(panel[["AAA"]].rename(columns={"AAA": "close"}))
        assert from_backtest == pytest.approx(from_scorer)

    def test_a_stored_run_records_which_signal_produced_it(self, engine: Engine) -> None:
        """Without it a row is uninterpretable after the signal changes -- as it just did."""
        fake_get_session, factory = _fake_session_factory(engine)
        rng = np.random.default_rng(5)
        n = 400
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        with factory() as session:
            for symbol in ("AAA", "BBB", "CCC"):
                session.add(Ticker(symbol=symbol, name=symbol, sector="Tech", asset_type="equity"))
                session.add(
                    IndexMembershipHistory(
                        index_name=hist.INDEX_NAME, symbol=symbol, added_date=dates[0].date()
                    )
                )
                level = 100.0
                for day, shock in zip(dates, rng.normal(0.0005, 0.012, n), strict=True):
                    level *= float(np.exp(shock))
                    session.add(
                        PriceHistory(
                            symbol=symbol,
                            date=day.date(),
                            open=level,
                            high=level,
                            low=level,
                            close=level,
                            adj_close=level,
                            volume=1_000_000,
                        )
                    )
            session.commit()

        with patch("refresh_data.get_session", fake_get_session):
            with factory() as session:
                refresh_data.refresh_backtest(session, dates[-1].date())
                session.commit()

        with factory() as session:
            runs = session.scalars(sa.select(BacktestResult)).all()
        assert runs, "the fixture produced no stored run to assert on"
        assert runs[-1].signal_name == refresh_data._BACKTEST_SIGNAL_NAME
        assert runs[-1].signal_name is not None


class TestAlerting:
    """Section 10's alert, asserted at the caller rather than at the helper.

    `send_alerts` has unit coverage through `alerting.rules`, `alerting.discord`
    and the three persistence reads. None of that says the nightly run ever
    calls it, which is the mistake this project has now shipped twice: a
    resolver with passing unit tests that the pipeline never invoked. So every
    test here drives `refresh_data.run()`.
    """

    _SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")

    @staticmethod
    def _sloped_prices(symbol: str, today: date, rows: int = 60) -> pd.DataFrame:
        """A distinct trend per symbol, because a tie ranks as no change.

        The first version of this fixture reused `_price_df`, which returns the
        same flat series for every symbol. Five identical inputs percentile-rank
        identically, so all five came out "hold" -- the same rating they were
        seeded with -- and the run correctly reported that nothing had changed.
        The test was measuring the fixture, not the alert.
        """
        index = TestAlerting._SYMBOLS.index(symbol)
        drift = 1.0 + (index - 2) * 0.01  # AAA falls hardest, EEE rises hardest
        closes = np.array([100.0 * drift**step for step in range(rows)])
        return pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp(today), periods=rows, freq="D"),
                "symbol": symbol,
                "open": closes,
                "high": closes * 1.02,
                "low": closes * 0.98,
                "close": closes,
                "adj_close": closes,
                "volume": 1_000_000,
            }
        )

    def _universe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "name": f"{symbol} Inc.",
                    "sector": "Technology",
                    "industry": "Software",
                    "exchange": None,
                    "asset_type": "equity",
                    "is_active": True,
                }
                for symbol in self._SYMBOLS
            ]
        )

    def _seed_snapshot(self, factory, day: date, rating: str, *, add_tickers: bool) -> None:
        """A stored scoring snapshot with every name on the same rating.

        Uniform on purpose. The composite ranks cross-sectionally, so five names
        come out spread across the five ratings whatever their inputs -- which
        means at least four must differ from a uniform "hold", and two of those
        land on strong_buy / strong_sell. That makes the universe-wide rule fire
        without the test having to predict a score.
        """
        with factory() as session:
            for index, symbol in enumerate(self._SYMBOLS):
                if add_tickers:
                    session.add(
                        Ticker(symbol=symbol, name=symbol, asset_type="equity", is_active=True)
                    )
                session.add(
                    CompositeScore(
                        symbol=symbol,
                        date=day,
                        profile="balanced",
                        composite_score=50.0 + index,
                        percentile_rank=50.0,
                        rating=rating,
                        data_confidence=80.0,
                    )
                )
            session.commit()

    @contextmanager
    def _driven_run(self, engine: Engine, *, settings, today: date, seed=("hold",)):
        """`run()` with every upstream source mocked and the clock pinned.

        `seed` is the ratings to store for the days before `today`, oldest
        first, so a test can arrange more than one prior snapshot. `()` seeds
        none.
        """
        fake_get_session, factory = _fake_session_factory(engine)
        for offset, rating in enumerate(reversed(seed), start=1):
            self._seed_snapshot(
                factory, today - timedelta(days=offset), rating, add_tickers=offset == len(seed)
            )
        options = {
            "symbol": "AAA",
            "expiration": "2026-08-21",
            "put_call_ratio": 0.8,
            "atm_implied_volatility": 0.25,
        }
        with (
            patch("refresh_data.get_session", fake_get_session),
            patch("refresh_data.get_settings", return_value=settings),
            patch("refresh_data.is_trading_day", return_value=True),
            patch("refresh_data.datetime", wraps=datetime) as clock,
            patch(
                "refresh_data.wikipedia_client.fetch_sp500_constituents",
                return_value=self._universe(),
            ),
            patch(
                "refresh_data.yfinance_client.fetch_price_history",
                side_effect=lambda symbol, **_: self._sloped_prices(symbol, today),
            ),
            patch("refresh_data.options_client.fetch_options_signals", return_value=options),
            patch("refresh_data.yfinance_client.fetch_fundamentals", return_value={}),
            patch("refresh_data.yfinance_client.fetch_analyst_consensus", return_value={}),
            patch(
                "refresh_data.short_interest_client.fetch_short_interest",
                return_value={"symbol": "AAA", "pct_float_short": None, "days_to_cover": None},
            ),
            patch(
                "refresh_data.edgar_client.fetch_insider_transactions",
                return_value=_empty_df(list(refresh_data._INSIDER_COLUMNS)),
            ),
            patch(
                "refresh_data.news_client.fetch_all_tier1_news",
                return_value=_empty_df(
                    ["title", "link", "summary", "published_at", "source", "symbol"]
                ),
            ),
            patch(
                "refresh_data.edgar_13f_client.fetch_institutional_ownership_trend",
                return_value=_empty_df(list(refresh_data._INSTITUTIONAL_COLUMNS)),
            ),
            patch(
                "refresh_data.gdelt_client.fetch_articles",
                return_value=_empty_df(["title", "url"]),
            ),
            patch(
                "refresh_data.gdelt_client.fetch_tone_timeline",
                return_value=_empty_df(["date", "tone", "query"]),
            ),
            patch("refresh_data.discord.send", return_value=1) as send,
        ):
            clock.now.return_value = datetime.combine(today, datetime.min.time())
            yield send, factory

    @staticmethod
    def _settings(**overrides):
        from quantpulse.config import Settings

        return Settings(
            _env_file=None,
            **{
                "alert_discord_webhook_url": "https://discord.test/api/webhooks/1/tok",
                **overrides,
            },
        )

    def test_the_nightly_run_sends_the_digest(self, engine: Engine) -> None:
        today = date(2026, 7, 22)
        with self._driven_run(engine, settings=self._settings(), today=today) as (send, _):
            refresh_data.run(job_name="test_alerts")

        send.assert_called_once()
        url, digest = send.call_args.args
        assert url == "https://discord.test/api/webhooks/1/tok"
        assert "2026-07-22" in digest and "2026-07-21" in digest
        assert rules.UNIVERSE_HEADING in digest

    def test_an_unconfigured_webhook_sends_nothing_and_does_not_degrade_the_run(
        self, engine: Engine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every fork of this public repo runs this workflow with no secret.

        The status is not asserted directly: this mocked universe has no
        `^GSPC`, so `benchmark_prices` writes nothing and the run is "partial"
        whatever alerting does. An assertion on the status alone would pass
        without alerting being involved at all, so what is asserted is that the
        step is not among the ones named as failed.
        """
        today = date(2026, 7, 22)
        settings = self._settings(alert_discord_webhook_url=None)
        with caplog.at_level(logging.INFO):
            with self._driven_run(engine, settings=settings, today=today) as (send, _):
                status = refresh_data.run(job_name="test_alerts")

        send.assert_not_called()
        assert status != "failed"
        assert "failed step(s)" not in caplog.text
        assert "Alerting is not configured" in caplog.text

    def test_the_kill_switch_stops_the_send(self, engine: Engine) -> None:
        today = date(2026, 7, 22)
        settings = self._settings(alerts_enabled=False)
        with self._driven_run(engine, settings=settings, today=today) as (send, _):
            refresh_data.run(job_name="test_alerts")
        send.assert_not_called()

    def test_a_webhook_failure_costs_the_message_and_nothing_else(
        self, engine: Engine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The night's data is already committed; a Discord outage must cost the
        message alone.

        `status == "partial"` is not enough on its own -- this fixture is
        already partial for an unrelated reason -- so the log has to name
        `alerts` as the failed step, and the day's scores have to be in the
        database regardless.
        """
        today = date(2026, 7, 22)
        with caplog.at_level(logging.INFO):
            with self._driven_run(engine, settings=self._settings(), today=today) as (
                send,
                factory,
            ):
                send.side_effect = discord.WebhookError("discord.test said no")
                status = refresh_data.run(job_name="test_alerts")

        assert status == "partial"
        assert "failed step(s): alerts" in caplog.text
        with factory() as session:
            scored = session.scalars(
                select(CompositeScore.symbol).where(CompositeScore.date == today)
            ).all()
        assert set(scored) == set(self._SYMBOLS)

    def test_no_alert_when_this_run_did_not_write_todays_scores(self, engine: Engine) -> None:
        """A failed composite step leaves the two newest snapshots unchanged, so
        an unguarded alert would re-send the previous run's digest as tonight's
        news -- and `step()`'s isolation means a red run still reaches this
        point.

        **Two** prior snapshots, and they differ. An earlier version of this
        test seeded one, so a failed composite left a single snapshot and the
        "needs two snapshots" guard fired first -- the `current != today` guard
        this test is named for was never reached, and deleting it changed
        nothing that any test could see. With hold -> strong_buy already
        stored, an unguarded run has a perfectly good digest to send, and only
        the date check stops it.
        """
        today = date(2026, 7, 22)
        with self._driven_run(
            engine, settings=self._settings(), today=today, seed=("hold", "strong_buy")
        ) as (send, _):
            with patch("refresh_data.refresh_composite_scores", side_effect=RuntimeError("boom")):
                refresh_data.run(job_name="test_alerts")
        send.assert_not_called()

    def test_a_fresh_database_says_why_it_cannot_compare_yet(
        self, engine: Engine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One stored snapshot is not a comparison.

        Behaviourally this is covered by the "scores are not today's" guard
        anyway, so the branch exists for its message: on a first run "needs two
        snapshots" and "the composite step failed" want different answers from
        whoever reads the log, and collapsing them would make a cold start look
        like a fault.
        """
        with caplog.at_level(logging.INFO):
            with self._driven_run(
                engine, settings=self._settings(), today=date(2026, 7, 21), seed=()
            ) as (send, _):
                refresh_data.run(job_name="test_alerts")
        send.assert_not_called()
        assert "needs two stored scoring snapshots" in caplog.text

    def test_a_quiet_night_sends_no_message_at_all(self, engine: Engine) -> None:
        """An alert that arrives every night regardless stops being read.

        Two consecutive runs over the same price shape: the second run's
        cross-sectional ranking is identical to the first's, so no rating moves
        and there is nothing to say. Driven through `run()` because "send
        nothing" is a decision the caller has to actually honour -- replacing
        the empty digest with a placeholder string is otherwise invisible.
        """
        with self._driven_run(
            engine, settings=self._settings(), today=date(2026, 7, 21), seed=()
        ) as (send, _):
            refresh_data.run(job_name="test_alerts")
        send.assert_not_called()  # only one snapshot exists yet

        with self._driven_run(
            engine, settings=self._settings(), today=date(2026, 7, 22), seed=()
        ) as (send, factory):
            refresh_data.run(job_name="test_alerts")

        with factory() as session:
            snapshots = set(session.scalars(select(CompositeScore.date).distinct()).all())
        assert len(snapshots) == 2, "the second run must have written a snapshot to compare"
        send.assert_not_called()

    def test_a_new_formation_on_a_watched_name_reaches_the_digest(self, engine: Engine) -> None:
        """The end-to-end path for the pattern trigger: the watermark taken
        before the sweep, the rows the sweep inserted, the watchlist scope, and
        the confidence floor -- all through `run()`."""
        from quantpulse.storage.models import WatchlistEntry

        today = date(2026, 7, 22)
        settings = self._settings()
        detected = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp(today - timedelta(days=30)),
                    "pattern_type": "cup_and_handle",
                    "direction": "bullish",
                    "confidence": 88.0,
                }
            ]
        )
        with self._driven_run(engine, settings=settings, today=today) as (send, factory):
            with factory() as session:
                session.add(WatchlistEntry(symbol="AAA", added_date=today))
                session.commit()
            with patch("refresh_data.patterns.detect_chart_patterns", return_value=detected):
                refresh_data.run(job_name="test_alerts")

        _, digest = send.call_args.args
        assert rules.TRACKED_HEADING in digest
        assert "cup and handle" in digest and "88" in digest

    def test_a_re_detected_formation_is_not_reported_as_new(self, engine: Engine) -> None:
        """The watermark's whole job. The sweep re-detects the same formation
        every night; only the run that first inserted it may say so."""
        from quantpulse.storage.models import WatchlistEntry

        today = date(2026, 7, 22)
        detected = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "date": pd.Timestamp(today - timedelta(days=30)),
                    "pattern_type": "cup_and_handle",
                    "direction": "bullish",
                    "confidence": 88.0,
                }
            ]
        )
        with self._driven_run(engine, settings=self._settings(), today=today) as (send, factory):
            with factory() as session:
                session.add(WatchlistEntry(symbol="AAA", added_date=today))
                session.commit()
            with patch("refresh_data.patterns.detect_chart_patterns", return_value=detected):
                refresh_data.run(job_name="test_alerts")
                send.reset_mock()
                refresh_data.run(job_name="test_alerts")

        if send.call_args is not None:
            assert "cup and handle" not in send.call_args.args[1]


class TestPaperTrading:
    """The forward test, asserted through `run()`.

    Same discipline as the alert: unit tests for `execution.alpaca` and
    `execution.strategy` say nothing about whether the nightly job ever calls
    them. Every test here drives `refresh_data.run()`, and `alpaca` is patched
    at the boundary that talks to the network -- so the paper-only guard, the
    credential check and the order validation all still execute.
    """

    _SYMBOLS = TestAlerting._SYMBOLS

    @staticmethod
    def _settings(**overrides):
        from quantpulse.config import Settings

        return Settings(
            _env_file=None,
            **{
                "alpaca_api_key_id": "PKTEST",
                "alpaca_api_secret_key": "shh",
                "alerts_enabled": False,
                **overrides,
            },
        )

    @contextmanager
    def _driven(self, engine: Engine, *, settings, today: date, account=None, positions=None):
        account = account or alpaca.Account(
            status="ACTIVE", equity=100_000.0, cash=100_000.0, buying_power=200_000.0
        )
        with TestAlerting()._driven_run(engine, settings=settings, today=today) as (_, factory):
            with (
                patch("refresh_data.alpaca.get_account", return_value=account) as get_account,
                patch("refresh_data.alpaca.list_positions", return_value=dict(positions or {})),
                patch("refresh_data.alpaca.submit_order", return_value={"id": "x"}) as submit,
            ):
                yield get_account, submit, factory

    def _snapshots(self, factory) -> list:
        from quantpulse.storage.models import PaperTradingSnapshot

        with factory() as session:
            return list(session.scalars(select(PaperTradingSnapshot)).all())

    def test_a_weekly_run_rebalances_and_records_the_account(self, engine: Engine) -> None:
        today = date(2026, 7, 20)  # a Monday -- the weekly branch
        assert today.weekday() == refresh_data._WEEKLY_REFRESH_WEEKDAY
        with self._driven(engine, settings=self._settings(), today=today) as (_, submit, factory):
            refresh_data.run(job_name="test_paper")

        rows = self._snapshots(factory)
        assert len(rows) == 1
        assert rows[0].equity == 100_000.0
        assert rows[0].rebalanced is True
        assert rows[0].signal_name == "composite_rating"
        assert submit.call_count > 0, "a weekly run with scores must actually place orders"
        assert rows[0].orders_submitted == submit.call_count

    def test_a_weekday_run_snapshots_without_trading(self, engine: Engine) -> None:
        """Section 7.6's cadence rule. A daily rebalance racks up turnover no
        real investor would accept, and the equity curve does not need trades
        to have daily resolution."""
        today = date(2026, 7, 21)  # a Tuesday
        with self._driven(engine, settings=self._settings(), today=today) as (_, submit, factory):
            refresh_data.run(job_name="test_paper")

        rows = self._snapshots(factory)
        assert len(rows) == 1 and rows[0].rebalanced is False
        submit.assert_not_called()

    def test_nothing_happens_without_credentials(self, engine: Engine) -> None:
        """The default in every fork of this public repo."""
        settings = self._settings(alpaca_api_key_id=None, alpaca_api_secret_key=None)
        with self._driven(engine, settings=settings, today=date(2026, 7, 20)) as (
            get_account,
            submit,
            factory,
        ):
            status = refresh_data.run(job_name="test_paper")

        get_account.assert_not_called()
        submit.assert_not_called()
        assert self._snapshots(factory) == []
        assert status != "failed"

    def test_the_kill_switch_stops_it_with_credentials_present(self, engine: Engine) -> None:
        with self._driven(
            engine, settings=self._settings(paper_trading_enabled=False), today=date(2026, 7, 20)
        ) as (get_account, submit, factory):
            refresh_data.run(job_name="test_paper")
        get_account.assert_not_called()
        assert self._snapshots(factory) == []

    def test_a_non_active_account_records_nothing(self, engine: Engine) -> None:
        """A restricted account still answers with a number, and recording it
        would flat-line the published curve at whatever it froze on."""
        frozen = alpaca.Account(status="ACCOUNT_CLOSED", equity=99.0, cash=0.0, buying_power=0.0)
        with self._driven(
            engine, settings=self._settings(), today=date(2026, 7, 20), account=frozen
        ) as (_, submit, factory):
            refresh_data.run(job_name="test_paper")
        assert self._snapshots(factory) == []
        submit.assert_not_called()

    def test_one_rejected_order_does_not_abandon_the_rest(self, engine: Engine) -> None:
        """A half-rebalanced book is worse than a fully rebalanced one, and a
        run that mostly failed must be visible rather than looking quiet."""
        today = date(2026, 7, 20)
        with self._driven(engine, settings=self._settings(), today=today) as (_, submit, factory):
            calls = {"n": 0}

            def flaky(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise alpaca.AlpacaError("rejected")
                return {"id": "x"}

            submit.side_effect = flaky
            refresh_data.run(job_name="test_paper")

        row = self._snapshots(factory)[0]
        assert row.orders_rejected == 1
        assert row.orders_submitted == submit.call_count - 1 > 0

    def test_an_alpaca_outage_does_not_fail_the_run(
        self, engine: Engine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The night's data is already committed; a broker being down costs the
        snapshot and nothing else."""
        with caplog.at_level(logging.INFO):
            with self._driven(engine, settings=self._settings(), today=date(2026, 7, 20)) as (
                get_account,
                _,
                factory,
            ):
                get_account.side_effect = alpaca.AlpacaError("paper-api unreachable")
                status = refresh_data.run(job_name="test_paper")

        assert status == "partial"
        assert "failed step(s): paper_trading" in caplog.text
        assert self._snapshots(factory) == []

    def test_re_running_the_same_day_corrects_rather_than_duplicates(self, engine: Engine) -> None:
        today = date(2026, 7, 20)
        with self._driven(engine, settings=self._settings(), today=today) as (_, __, factory):
            refresh_data.run(job_name="test_paper")
            refresh_data.run(job_name="test_paper")
        assert len(self._snapshots(factory)) == 1

    def test_the_benchmark_close_is_stored_beside_the_equity(self, engine: Engine) -> None:
        """So the comparison stays point-in-time rather than being re-joined
        against an index series that may later be re-ingested."""
        today = date(2026, 7, 20)
        with self._driven(engine, settings=self._settings(), today=today) as (_, __, factory):
            with factory() as session:
                session.add(
                    Ticker(
                        symbol=risk.MARKET_INDEX_SYMBOL,
                        name="S&P 500",
                        asset_type="index",
                        is_active=False,
                    )
                )
                session.add(
                    PriceHistory(
                        symbol=risk.MARKET_INDEX_SYMBOL,
                        date=today,
                        open=1.0,
                        high=1.0,
                        low=1.0,
                        close=5500.0,
                        adj_close=5500.0,
                        volume=1,
                    )
                )
                session.commit()
            refresh_data.run(job_name="test_paper")

        assert self._snapshots(factory)[0].benchmark_close == pytest.approx(5500.0)

    def test_the_orders_it_places_are_whole_share_buys_of_rated_names(self, engine: Engine) -> None:
        """The end-to-end shape: only names the app rates Buy/Strong Buy, whole
        positive quantities, sells before buys."""
        today = date(2026, 7, 20)
        with self._driven(
            engine, settings=self._settings(), today=today, positions={"ZZZZ": 4.0}
        ) as (_, submit, factory):
            refresh_data.run(job_name="test_paper")

        placed = [(c.args[0], c.kwargs["qty"], c.kwargs["side"]) for c in submit.call_args_list]
        assert placed, "expected orders"
        assert placed[0] == ("ZZZZ", 4, "sell"), "an unrated holding is sold, and sells go first"
        for _symbol, qty, side in placed:
            assert isinstance(qty, int) and qty > 0
            assert side in ("buy", "sell")


class TestWeeklyBranchCadence:
    """Which day carries the weekly branch, and why it is not simply "Monday".

    US market holidays cluster on Mondays by construction -- MLK, Presidents'
    Day, Memorial Day and Labor Day are each "the nth Monday of" a month -- so
    asking `today.weekday() == 0` hands the whole weekly branch (fundamentals,
    analyst consensus, 13F, forecasts, backtest, news and sentiment) a silent
    week off roughly one week in ten.

    That happened: the scheduled run for Monday 2026-09-07 logged
    `skipped_non_trading_day` because it was Labor Day, and the sector-basket
    work waiting for a weekly run since 2026-09-03 went untested for another
    week while the newest stored news article stayed three weeks old. Four of
    2026's fifty-two Mondays are closed.
    """

    def test_an_ordinary_monday_carries_it(self) -> None:
        assert refresh_data.is_weekly_run(date(2026, 9, 14)) is True

    def test_an_ordinary_tuesday_does_not(self) -> None:
        assert refresh_data.is_weekly_run(date(2026, 9, 15)) is False

    def test_the_tuesday_after_a_closed_monday_carries_it(self) -> None:
        """The whole point: the weekly work lands one day late rather than not
        at all."""
        assert not is_trading_day(date(2026, 9, 7)), "2026-09-07 is Labor Day"
        assert refresh_data.is_weekly_run(date(2026, 9, 8)) is True

    def test_a_closed_monday_is_not_itself_the_weeks_first_trading_day(self) -> None:
        assert refresh_data.is_weekly_run(date(2026, 9, 7)) is False

    def test_every_closed_monday_in_2026_hands_off_to_its_tuesday(self) -> None:
        """All four of them, not just the one that broke."""
        closed_mondays = [
            day
            for day in (date(2026, 1, 1) + timedelta(days=n) for n in range(365))
            if day.weekday() == 0 and not is_trading_day(day)
        ]
        assert len(closed_mondays) == 4, closed_mondays
        for monday in closed_mondays:
            assert refresh_data.is_weekly_run(monday) is False, monday
            assert refresh_data.is_weekly_run(monday + timedelta(days=1)) is True, monday

    def test_exactly_one_day_a_week_carries_it(self) -> None:
        """Two weekly runs in a week would double the heaviest job in the
        project; none is the outage this replaced."""
        for week in range(52):
            monday = date(2026, 1, 5) + timedelta(weeks=week)
            carriers = [
                monday + timedelta(days=n)
                for n in range(7)
                if refresh_data.is_weekly_run(monday + timedelta(days=n))
            ]
            assert len(carriers) == 1, (monday, carriers)

    def test_a_fully_closed_week_carries_it_on_no_day(self) -> None:
        """Not reachable from the NYSE calendar, but the arithmetic must not
        invent a trading day if it ever were."""
        with patch("refresh_data.is_trading_day", return_value=False):
            assert refresh_data.is_weekly_run(date(2026, 9, 14)) is False

    def test_the_run_itself_takes_the_weekly_branch_after_a_closed_monday(
        self, engine: Engine
    ) -> None:
        """Asserted through `run()`, not through the helper.

        The first version of these tests checked `is_weekly_run` alone, and
        reverting `run()`'s call site to `today.weekday() == 0` changed nothing
        any of them could see -- the same "assert the caller" mistake this
        project has now made three times.
        """
        settings = TestPaperTrading._settings(alpaca_api_key_id=None, alpaca_api_secret_key=None)
        labor_day_tuesday = date(2026, 9, 8)
        with TestAlerting()._driven_run(engine, settings=settings, today=labor_day_tuesday) as (
            _,
            __,
        ):
            with patch("refresh_data.refresh_forecasts", return_value=0) as forecasts:
                refresh_data.run(job_name="test_cadence")

        forecasts.assert_called(), "the Tuesday after Labor Day must carry the weekly branch"

    def test_the_run_skips_the_weekly_branch_on_an_ordinary_tuesday(self, engine: Engine) -> None:
        settings = TestPaperTrading._settings(alpaca_api_key_id=None, alpaca_api_secret_key=None)
        with TestAlerting()._driven_run(engine, settings=settings, today=date(2026, 9, 15)) as (
            _,
            __,
        ):
            with patch("refresh_data.refresh_forecasts", return_value=0) as forecasts:
                refresh_data.run(job_name="test_cadence")

        forecasts.assert_not_called()
