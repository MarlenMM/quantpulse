from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.storage import persistence
from quantpulse.storage.models import (
    Base,
    InsiderTransaction,
    MacroIndicator,
    MarketRegime,
    NewsEvent,
    PriceHistory,
    SentimentScore,
    ThematicBasket,
    Ticker,
)


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        s.add(Ticker(symbol="AAPL", name="Apple Inc.", asset_type="equity", is_active=True))
        s.commit()
        yield s


class TestArticleId:
    def test_same_url_gives_same_id(self) -> None:
        a = persistence.article_id_for("https://x.com/a", fallback="t")
        b = persistence.article_id_for("https://x.com/a", fallback="different")
        assert a == b

    def test_falls_back_to_fallback_when_no_url(self) -> None:
        a = persistence.article_id_for(None, fallback="headline one")
        b = persistence.article_id_for("", fallback="headline two")
        assert a != b  # distinct fallbacks -> distinct ids, not a shared empty-string collision


class TestAppendOnly:
    def test_sentiment_scores_first_write_wins(self, session: Session) -> None:
        row = {
            "symbol": "AAPL",
            "date": date(2026, 7, 22),
            "source": "tier1_aggregate",
            "sentiment_score": 0.4,
            "mention_volume": 3,
            "total_weight": 2.1,
        }
        assert persistence.upsert_sentiment_scores(session, [row]) == 1
        persistence.upsert_sentiment_scores(session, [{**row, "sentiment_score": 0.9}])
        session.flush()

        stored = session.scalars(select(SentimentScore)).one()
        assert stored.sentiment_score == 0.4  # point-in-time: first write preserved

    def test_news_events_dedupe_on_article_id(self, session: Session) -> None:
        article = {
            "article_id": persistence.article_id_for("https://x.com/a", fallback="t"),
            "tier": 1,
            "title": "Apple beats",
            "matched_symbols": ["AAPL"],
            "sentiment_score": 0.5,
            "source_url": "https://x.com/a",
        }
        persistence.upsert_news_events(session, [article])
        persistence.upsert_news_events(session, [article])  # same URL -> same id
        session.flush()

        rows = session.scalars(select(NewsEvent)).all()
        assert len(rows) == 1
        assert rows[0].matched_symbols == ["AAPL"]

    def test_insider_transactions_dedupe_on_natural_key(self, session: Session) -> None:
        txn = {
            "symbol": "AAPL",
            "insider_name": "Jane Doe",
            "transaction_date": date(2026, 7, 1),
            "transaction_code": "P",
            "shares": 1000.0,
            "price_per_share": 150.0,
        }
        persistence.insert_insider_transactions(session, [txn])
        persistence.insert_insider_transactions(session, [txn])
        session.flush()
        assert len(session.scalars(select(InsiderTransaction)).all()) == 1

    def test_market_regime_first_write_wins(self, session: Session) -> None:
        base = {"date": date(2026, 7, 22), "regime_score": 60.0, "regime_label": "neutral"}
        persistence.upsert_market_regime(session, base)
        persistence.upsert_market_regime(session, {**base, "regime_score": 10.0})
        session.flush()
        assert session.scalars(select(MarketRegime)).one().regime_score == 60.0


class TestThematicBasketsReplace:
    def test_replace_swaps_the_whole_config(self, session: Session) -> None:
        persistence.replace_thematic_baskets(
            session, [{"theme_name": "ai_theme", "symbol": "NVDA"}]
        )
        session.flush()
        persistence.replace_thematic_baskets(
            session,
            [
                {"theme_name": "ai_theme", "symbol": "NVDA"},
                {"theme_name": "ai_theme", "symbol": "AMD"},
            ],
        )
        session.flush()
        rows = session.scalars(select(ThematicBasket)).all()
        assert {r.symbol for r in rows} == {"NVDA", "AMD"}  # stale set replaced, not appended

    def test_allows_members_outside_the_ticker_universe(self, session: Session) -> None:
        # TSM/ASML aren't in `tickers`; a thematic basket must still hold them (no FK).
        persistence.replace_thematic_baskets(session, [{"theme_name": "semis", "symbol": "TSM"}])
        session.flush()
        assert session.scalars(select(ThematicBasket)).one().symbol == "TSM"


class TestReaders:
    def test_read_recent_atm_iv_excludes_as_of_day(self, session: Session) -> None:
        for d, iv in [
            (date(2026, 7, 20), 0.20),
            (date(2026, 7, 21), 0.30),
            (date(2026, 7, 22), 0.99),  # the as-of day itself must be excluded
        ]:
            persistence.upsert_options_signals(
                session,
                [{"symbol": "AAPL", "date": d, "atm_implied_volatility": iv}],
            )
        session.flush()

        history = persistence.read_recent_atm_iv(session, "AAPL", before=date(2026, 7, 22))
        assert history == [0.20, 0.30]  # ordered, and 0.99 (as-of day) excluded

    def test_read_latest_macro_value_is_point_in_time(self, session: Session) -> None:
        for d, v in [(date(2026, 7, 20), 4.1), (date(2026, 7, 25), 4.9)]:
            session.add(MacroIndicator(date=d, indicator_name="DGS10", value=v))
        session.flush()
        # As of the 22nd, the future 25th value must not be read.
        assert persistence.read_latest_macro_value(session, "DGS10", as_of=date(2026, 7, 22)) == 4.1

    def test_read_latest_macro_value_missing_is_none(self, session: Session) -> None:
        assert persistence.read_latest_macro_value(session, "NOPE", as_of=date(2026, 7, 22)) is None

    def test_read_macro_series_windowed_oldest_first(self, session: Session) -> None:
        for d, v in [
            (date(2026, 7, 1), 15.0),
            (date(2026, 7, 20), 18.0),
            (date(2026, 7, 22), 22.0),
        ]:
            session.add(MacroIndicator(date=d, indicator_name="vix", value=v))
        session.flush()
        series = persistence.read_macro_series(
            session, "vix", as_of=date(2026, 7, 22), lookback_days=10
        )
        assert series == [18.0, 22.0]  # 07-01 falls outside the 10-day window

    def test_read_active_price_history_excludes_future_and_inactive(self, session: Session) -> None:
        session.add(Ticker(symbol="OLD", name="Old Co", asset_type="equity", is_active=False))
        session.flush()
        for symbol in ("AAPL", "OLD"):
            for d in (date(2026, 7, 20), date(2026, 7, 25)):
                session.add(
                    PriceHistory(
                        symbol=symbol,
                        date=d,
                        open=1.0,
                        high=1.0,
                        low=1.0,
                        close=1.0,
                        adj_close=1.0,
                        volume=1,
                    )
                )
        session.flush()

        frame = persistence.read_active_price_history(
            session, as_of=date(2026, 7, 22), lookback_days=30
        )
        assert set(frame["symbol"]) == {"AAPL"}  # inactive OLD excluded
        assert frame["date"].max() <= date(2026, 7, 22)  # future 07-25 bar excluded
