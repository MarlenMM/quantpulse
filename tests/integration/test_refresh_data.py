from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
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
    PriceHistory,
    RefreshLog,
    Ticker,
)


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

    with (
        patch("refresh_data.get_session", fake_get_session),
        patch("refresh_data.is_trading_day", return_value=True),
        patch("refresh_data.wikipedia_client.fetch_sp500_constituents", return_value=tiny_universe),
        patch(
            "refresh_data.yfinance_client.fetch_price_history",
            return_value=_price_df("AAPL", rows=2),
        ),
    ):
        refresh_data.run(job_name="test_run")

    with factory() as session:
        prices = session.scalars(select(PriceHistory)).all()
        logs = session.scalars(select(RefreshLog)).all()

    assert len(prices) == 2
    assert len(logs) == 1
    assert logs[0].status == "success"


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
