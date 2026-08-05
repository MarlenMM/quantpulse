import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import refresh_data
from quantpulse.analysis import backtest as bt
from quantpulse.storage.models import (
    AnalystConsensus,
    Base,
    FundamentalsSnapshot,
    IndexMembershipHistory,
    MarketRegime,
    OptionsSignal,
    PatternSignal,
    PriceHistory,
    RefreshLog,
    Ticker,
)


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
    with (
        patch("refresh_data.get_session", fake_get_session),
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
    written_symbols = {p.symbol for p in prices}
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
