from collections.abc import Iterator
from datetime import date

import pandas as pd
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


# --------------------------------------------------------------------------- #
# Phase 10 — UI read helpers (latest-snapshot reads, Section 12)
# --------------------------------------------------------------------------- #


@pytest.fixture
def ui_session(tmp_path) -> Iterator[Session]:
    """A session seeded with two scoring snapshots plus prices/forecasts/news."""
    from datetime import datetime, timedelta

    from quantpulse.storage.models import (
        AnalystConsensus,
        BacktestResult,
        CompositeScore,
        Forecast,
        MarketRegime,
        PatternSignal,
        RefreshLog,
    )

    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'ui.db'}")
    Base.metadata.create_all(engine)
    today = date.today()
    yesterday = today - timedelta(days=1)

    with sessionmaker(bind=engine)() as s:
        s.add(Ticker(symbol="AAPL", name="Apple", sector="Tech", asset_type="equity"))
        s.add(Ticker(symbol="XOM", name="Exxon", sector="Energy", asset_type="equity"))
        s.add(Ticker(symbol="OLD", name="Delisted", sector="Tech", is_active=False))

        for day, (aapl_rating, aapl_score) in (
            (yesterday, ("hold", 50.0)),
            (today, ("buy", 75.0)),
        ):
            s.add(
                CompositeScore(
                    symbol="AAPL",
                    date=day,
                    profile="balanced",
                    composite_score=aapl_score,
                    technical_score=80.0,
                    fundamental_score=None,
                    rating=aapl_rating,
                    percentile_rank=aapl_score,
                    data_confidence=90.0,
                )
            )
            s.add(
                CompositeScore(
                    symbol="XOM",
                    date=day,
                    profile="balanced",
                    composite_score=40.0,
                    technical_score=30.0,
                    rating="sell",
                    percentile_rank=20.0,
                    data_confidence=55.0,
                )
            )
        for i in range(5):
            d = today - timedelta(days=i)
            s.add(
                PriceHistory(
                    symbol="AAPL",
                    date=d,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.5 + i,
                    adj_close=100.5 + i,
                    volume=1000,
                )
            )
        s.add(
            Forecast(
                symbol="AAPL",
                generated_date=yesterday,
                horizon_days=5,
                model_name="gbr",
                point_return=0.01,
                point_price=101.0,
            )
        )
        s.add(
            Forecast(
                symbol="AAPL",
                generated_date=today,
                horizon_days=5,
                model_name="gbr",
                point_return=0.02,
                point_price=103.0,
                lower_price=99.0,
                upper_price=107.0,
                historical_hit_rate=0.55,
            )
        )
        s.add(
            AnalystConsensus(
                symbol="AAPL",
                as_of_date=today,
                strong_buy=5,
                buy=3,
                hold=1,
                sell=0,
                strong_sell=0,
                mean_price_target=120.0,
            )
        )
        s.add(
            PatternSignal(
                symbol="AAPL",
                date=today,
                pattern_type="cup_and_handle",
                direction="bullish",
                confidence=0.8,
            )
        )
        s.add(
            MarketRegime(
                date=today,
                vix_level=18.0,
                breadth_pct_above_200dma=0.6,
                regime_score=62.0,
                regime_label="risk_on",
            )
        )
        s.add(
            NewsEvent(
                article_id="t3",
                tier=3,
                title="Fed holds",
                published_at=datetime.now(),
                event_type="macro",
                sentiment_score=0.1,
            )
        )
        s.add(
            NewsEvent(
                article_id="t1",
                tier=1,
                title="Apple ships",
                published_at=datetime.now(),
                matched_symbols=["AAPL"],
                sentiment_score=0.5,
            )
        )
        s.add(
            BacktestResult(
                run_date=today, cadence="monthly", n_periods=40, sharpe=0.8, assumed_txn_cost=0.001
            )
        )
        s.add(
            RefreshLog(
                job_name="refresh_data",
                run_timestamp=datetime.now(),
                status="success",
                rows_updated=10,
            )
        )
        s.commit()
        yield s


class TestScreenerReads:
    def test_returns_latest_snapshot_joined_to_ticker_metadata(self, ui_session) -> None:
        rows = persistence.read_screener_rows(ui_session)
        assert set(rows["symbol"]) == {"AAPL", "XOM"}
        aapl = rows[rows["symbol"] == "AAPL"].iloc[0]
        assert aapl["rating"] == "buy"  # today's row, not yesterday's
        assert aapl["name"] == "Apple"
        assert aapl["sector"] == "Tech"

    def test_sorted_by_composite_descending(self, ui_session) -> None:
        rows = persistence.read_screener_rows(ui_session)
        assert list(rows["symbol"]) == ["AAPL", "XOM"]

    def test_missing_subscores_stay_null_not_zero(self, ui_session) -> None:
        rows = persistence.read_screener_rows(ui_session)
        aapl = rows[rows["symbol"] == "AAPL"].iloc[0]
        assert pd.isna(aapl["fundamental_score"])

    def test_unknown_profile_is_empty(self, ui_session) -> None:
        assert persistence.read_screener_rows(ui_session, profile="growth").empty

    def test_empty_database_returns_empty_frame(self, session) -> None:
        assert persistence.read_screener_rows(session).empty

    def test_latest_score_date(self, ui_session) -> None:
        assert persistence.read_latest_score_date(ui_session) == date.today()


class TestRatingChanges:
    def test_reports_only_symbols_whose_rating_moved(self, ui_session) -> None:
        changes = persistence.read_rating_changes(ui_session)
        assert list(changes["symbol"]) == ["AAPL"]  # XOM stayed "sell"
        assert changes.iloc[0]["previous_rating"] == "hold"
        assert changes.iloc[0]["rating"] == "buy"
        assert changes.iloc[0]["score_change"] == pytest.approx(25.0)

    def test_needs_two_snapshots(self, session) -> None:
        from quantpulse.storage.models import CompositeScore

        session.add(
            CompositeScore(
                symbol="AAPL",
                date=date.today(),
                profile="balanced",
                composite_score=50.0,
                rating="hold",
                data_confidence=80.0,
            )
        )
        session.commit()
        assert persistence.read_rating_changes(session).empty


class TestSymbolReads:
    def test_ohlcv_is_oldest_first(self, ui_session) -> None:
        bars = persistence.read_symbol_ohlcv(ui_session, "AAPL")
        assert len(bars) == 5
        assert bars["date"].is_monotonic_increasing

    def test_forecasts_use_only_the_latest_generation(self, ui_session) -> None:
        rows = persistence.read_symbol_forecasts(ui_session, "AAPL")
        assert len(rows) == 1
        assert rows.iloc[0]["point_price"] == pytest.approx(103.0)
        assert rows.iloc[0]["historical_hit_rate"] == pytest.approx(0.55)

    def test_forecasts_absent_symbol_is_empty(self, ui_session) -> None:
        assert persistence.read_symbol_forecasts(ui_session, "NOPE").empty

    def test_patterns(self, ui_session) -> None:
        rows = persistence.read_symbol_patterns(ui_session, "AAPL")
        assert rows.iloc[0]["pattern_type"] == "cup_and_handle"

    def test_analyst_consensus(self, ui_session) -> None:
        consensus = persistence.read_latest_analyst_consensus(ui_session, "AAPL")
        assert consensus is not None
        assert consensus["strong_buy"] == 5
        assert persistence.read_latest_analyst_consensus(ui_session, "NOPE") is None

    def test_symbol_news_matches_on_the_stored_symbol_list(self, ui_session) -> None:
        rows = persistence.read_symbol_news(ui_session, "AAPL")
        assert list(rows["title"]) == ["Apple ships"]
        assert persistence.read_symbol_news(ui_session, "XOM").empty


class TestDashboardReads:
    def test_market_moving_news_is_tier_2_and_3_only(self, ui_session) -> None:
        rows = persistence.read_market_moving_news(ui_session)
        assert list(rows["title"]) == ["Fed holds"]

    def test_market_regime_oldest_first(self, ui_session) -> None:
        rows = persistence.read_recent_market_regime(ui_session)
        assert rows.iloc[-1]["regime_label"] == "risk_on"

    def test_backtest_history(self, ui_session) -> None:
        rows = persistence.read_backtest_history(ui_session)
        assert rows.iloc[0]["sharpe"] == pytest.approx(0.8)

    def test_refresh_log(self, ui_session) -> None:
        rows = persistence.read_refresh_log(ui_session)
        assert rows.iloc[0]["status"] == "success"


class TestFreshnessAndUniverse:
    def test_never_populated_tables_report_none_not_a_missing_key(self, ui_session) -> None:
        # The UI must be able to tell "stale" from "never ran"; omitting the key
        # would render both as a reassuring blank.
        freshness = persistence.read_data_freshness(ui_session)
        assert freshness["prices"] == date.today()
        assert freshness["fundamentals"] is None
        assert "sentiment" in freshness

    def test_universe_active_only_by_default(self, ui_session) -> None:
        active = persistence.read_ticker_universe(ui_session)
        assert set(active["symbol"]) == {"AAPL", "XOM"}
        everything = persistence.read_ticker_universe(ui_session, active_only=False)
        assert "OLD" in set(everything["symbol"])

    def test_latest_prices_omits_unpriced_symbols(self, ui_session) -> None:
        # An unpriced holding must be absent rather than mapped to 0.0, so the
        # Portfolio page can flag it stale instead of showing it as worthless.
        prices = persistence.read_latest_prices(ui_session, ["AAPL", "XOM"])
        assert "XOM" not in prices
        # The seed writes close=100.5+i at (today - i days), so today's bar is 100.5.
        assert prices["AAPL"] == pytest.approx(100.5)

    def test_latest_prices_empty_input(self, ui_session) -> None:
        assert persistence.read_latest_prices(ui_session, []) == {}


class TestAdjClosePanelExcludesUnusablePrices:
    """A non-positive adjusted close is not a price, and one poisons everything.

    Real example from the cold-start backfill: `DEC` carries 732 bars whose
    `adj_close` is exactly 0.0 while its raw `close` is $1.44. Left in the
    panel, `refresh_data._equal_weight_benchmark` rebases each name to its own
    first observation -- so dividing by that zero makes the ENTIRE benchmark
    index `inf`, and every strategy-vs-market number on the Track Record page
    is compared against infinity.
    """

    @staticmethod
    def _bar(session: Session, symbol: str, day: int, adj_close: float) -> None:
        session.add(
            PriceHistory(
                symbol=symbol,
                date=date(2024, 1, day),
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                adj_close=adj_close,
                volume=100,
            )
        )

    def test_zero_and_negative_adj_close_rows_are_excluded(self, session: Session) -> None:
        session.add(Ticker(symbol="DEC", name="Decoy", asset_type="equity", is_active=False))
        self._bar(session, "AAPL", 2, 10.0)
        self._bar(session, "AAPL", 3, 11.0)
        self._bar(session, "DEC", 2, 0.0)  # the real-world shape
        self._bar(session, "DEC", 3, -1.0)
        session.commit()

        panel = persistence.read_adj_close_panel(
            session, start=date(2024, 1, 1), end=date(2024, 1, 31)
        )

        assert "DEC" not in panel.columns, "an all-unusable symbol must not reach the panel"
        assert list(panel["AAPL"]) == [10.0, 11.0]

    def test_only_the_bad_bars_are_dropped_not_the_whole_symbol(self, session: Session) -> None:
        session.add(Ticker(symbol="DEC", name="Decoy", asset_type="equity", is_active=False))
        self._bar(session, "DEC", 2, 0.0)  # bad bar first -- the one that caused `inf`
        self._bar(session, "DEC", 3, 5.0)  # genuine price afterwards
        session.commit()

        panel = persistence.read_adj_close_panel(
            session, start=date(2024, 1, 1), end=date(2024, 1, 31)
        )

        assert panel["DEC"].dropna().tolist() == [5.0]


class TestBrokenAdjustmentSeriesAreDropped:
    """A broken split/dividend adjustment factor invents enormous fake returns.

    Real measurements from a 495-symbol backfill: CBE moves $0.005 -> $305.00 in
    a single bar (a 3,399,900% "return"), TNB reaches $31,080, COMS spans 16.7
    BILLION times end to end. Any momentum signal ranks those top, "buys" them,
    and books the fiction. Left in, the strategy backtest reported CAGR 64.77%
    against a 12.66% benchmark, with a bootstrap interval that EXCLUDED ZERO --
    a statistically significant edge made entirely of bad data. Sanitized, the
    same run reports 9.53% against a 13.01% benchmark: it loses to buy-and-hold.
    """

    @staticmethod
    def _bar(session: Session, symbol: str, day: int, adj_close: float) -> None:
        session.add(
            PriceHistory(
                symbol=symbol,
                date=date(2024, 1, day),
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                adj_close=adj_close,
                volume=100,
            )
        )

    def _panel(self, session: Session):
        return persistence.read_adj_close_panel(
            session, start=date(2024, 1, 1), end=date(2024, 1, 31)
        )

    def test_a_symbol_with_an_impossible_jump_is_removed_entirely(self, session: Session) -> None:
        session.add(Ticker(symbol="CBE", name="Broken", asset_type="equity", is_active=False))
        for day, px in ((2, 10.0), (3, 11.0), (4, 12.0)):
            self._bar(session, "AAPL", day, px)
        # The real CBE shape: a sub-penny price stepping to hundreds of dollars.
        for day, px in ((2, 0.005), (3, 305.0), (4, 300.0)):
            self._bar(session, "CBE", day, px)
        session.commit()

        panel = self._panel(session)

        assert "CBE" not in panel.columns
        # The whole series goes, not just the offending bar: one bad adjustment
        # factor scales every price it touches, so the rest isn't trustworthy.
        assert "AAPL" in panel.columns
        assert list(panel["AAPL"]) == [10.0, 11.0, 12.0]

    def test_a_large_but_believable_move_is_kept(self, session: Session) -> None:
        # A takeover pop or a biotech readout can double or triple a stock in a
        # day. The guard must not throw those away.
        for day, px in ((2, 10.0), (3, 28.0), (4, 26.0)):
            self._bar(session, "AAPL", day, px)
        session.commit()

        panel = self._panel(session)

        assert "AAPL" in panel.columns
        assert list(panel["AAPL"]) == [10.0, 28.0, 26.0]

    def test_a_crash_as_well_as_a_spike_is_caught(self, session: Session) -> None:
        session.add(Ticker(symbol="CBE", name="Broken", asset_type="equity", is_active=False))
        for day, px in ((2, 500.0), (3, 0.01), (4, 0.01)):
            self._bar(session, "CBE", day, px)
        session.commit()

        assert "CBE" not in self._panel(session).columns
