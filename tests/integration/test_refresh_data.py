from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import refresh_data
from quantpulse.storage.models import (
    AnalystConsensus,
    Base,
    FundamentalsSnapshot,
    IndexMembershipHistory,
    MarketRegime,
    OptionsSignal,
    PriceHistory,
    RefreshLog,
    Ticker,
)


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(eng)
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
