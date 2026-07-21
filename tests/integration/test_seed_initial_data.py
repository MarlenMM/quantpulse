from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

import seed_initial_data as seed
from quantpulse.storage.models import Base, IndexMembershipHistory, PriceHistory, RefreshLog, Ticker


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'seed.db'}")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine)


def _session_factory(factory: sessionmaker):
    @contextmanager
    def _get_session() -> Iterator[Session]:
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _get_session


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "index_name": "S&P 500",
                "symbol": "AAPL",
                "added_date": date(1996, 1, 2),
                "removed_date": None,
            },
            {
                "index_name": "S&P 500",
                "symbol": "AABA",
                "added_date": date(1999, 12, 8),
                "removed_date": date(2017, 6, 19),
            },
        ]
    )


def _price_df(symbol: str, start: str, rows: int = 3) -> pd.DataFrame:
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
            "volume": 100,
        }
    )


def test_seed_tickers_sets_active_flag_from_membership(factory: sessionmaker) -> None:
    current = pd.DataFrame(
        [{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Tech", "industry": "Devices"}]
    )
    with (
        patch.object(seed.wikipedia_client, "fetch_sp500_constituents", return_value=current),
        factory() as session,
    ):
        count = seed.seed_tickers(session, _membership())
        session.commit()

    with factory() as session:
        tickers = {t.symbol: t for t in session.scalars(select(Ticker))}

    assert count == 2
    assert tickers["AAPL"].is_active is True
    assert tickers["AAPL"].sector == "Tech"  # enriched from current metadata
    assert tickers["AABA"].is_active is False  # removed name -> inactive placeholder
    assert tickers["AABA"].name == "AABA"


def test_seed_tickers_does_not_clobber_existing_rows(factory: sessionmaker) -> None:
    with factory() as session:
        session.add(Ticker(symbol="AAPL", name="Existing Rich Name", sector="RealSector"))
        session.commit()

    empty = pd.DataFrame(columns=["symbol", "name", "sector", "industry"])
    with (
        patch.object(seed.wikipedia_client, "fetch_sp500_constituents", return_value=empty),
        factory() as session,
    ):
        seed.seed_tickers(session, _membership())
        session.commit()

    with factory() as session:
        aapl = session.get(Ticker, "AAPL")
    assert aapl.name == "Existing Rich Name"  # on-conflict-do-nothing preserved it


def test_seed_index_membership_replaces_not_duplicates(factory: sessionmaker) -> None:
    with (
        patch.object(
            seed.wikipedia_client,
            "fetch_sp500_constituents",
            return_value=pd.DataFrame(columns=["symbol", "name", "sector", "industry"]),
        ),
        factory() as session,
    ):
        seed.seed_tickers(session, _membership())
        seed.seed_index_membership(session, _membership())
        seed.seed_index_membership(session, _membership())  # re-seed
        session.commit()

    with factory() as session:
        rows = session.scalar(select(func.count()).select_from(IndexMembershipHistory))
    assert rows == 2  # replaced, not appended to 4


def test_needs_backfill_logic(factory: sessionmaker) -> None:
    with factory() as session:
        session.add(Ticker(symbol="OLD", name="Old"))
        session.add(
            PriceHistory(
                symbol="OLD",
                date=date(2005, 1, 3),
                open=1,
                high=1,
                low=1,
                close=1,
                adj_close=1,
                volume=1,
            )
        )
        session.commit()

    cutoff = date(2026, 1, 1)
    with factory() as session:
        assert seed._needs_backfill(session, "OLD", cutoff) is False  # has deep history
        assert seed._needs_backfill(session, "MISSING", cutoff) is True


def test_backfill_prices_skips_already_backfilled_and_fetches_the_rest(
    factory: sessionmaker,
) -> None:
    with factory() as session:
        for sym in ("AAPL", "AABA"):
            session.add(Ticker(symbol=sym, name=sym))
        # AAPL already has deep history (backfilled); AABA has none.
        session.add(
            PriceHistory(
                symbol="AAPL",
                date=date(2000, 1, 3),
                open=1,
                high=1,
                low=1,
                close=1,
                adj_close=1,
                volume=1,
            )
        )
        session.commit()

    with patch.object(
        seed.yfinance_client, "fetch_price_history", return_value=_price_df("AABA", "2015-01-05")
    ) as mock_fetch:
        written, skipped = seed.backfill_prices(
            ["AAPL", "AABA"],
            period="max",
            session_factory=_session_factory(factory),
            today=date(2026, 7, 22),
        )

    mock_fetch.assert_called_once_with("AABA", period="max")
    assert skipped == 1
    assert written == 3


def test_run_end_to_end_historical_mode(factory: sessionmaker) -> None:
    current = pd.DataFrame(
        [{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Tech", "industry": "Devices"}]
    )
    with (
        patch.object(seed, "resolve_membership", return_value=(_membership(), "historical")),
        patch.object(seed.wikipedia_client, "fetch_sp500_constituents", return_value=current),
        patch.object(
            seed.yfinance_client,
            "fetch_price_history",
            side_effect=lambda symbol, period: _price_df(symbol, "2010-01-04"),
        ),
    ):
        seed.run(period="max", session_factory=_session_factory(factory))

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Ticker)) == 2
        assert session.scalar(select(func.count()).select_from(IndexMembershipHistory)) == 2
        assert session.scalar(select(func.count()).select_from(PriceHistory)) == 6
        logs = session.scalars(select(RefreshLog)).all()
    assert len(logs) == 1
    assert logs[0].status == "success"


def test_run_flags_survivorship_bias_in_fallback_mode(factory: sessionmaker) -> None:
    with (
        patch.object(seed, "resolve_membership", return_value=(_membership(), "current_only")),
        patch.object(
            seed.wikipedia_client,
            "fetch_sp500_constituents",
            return_value=pd.DataFrame(columns=["symbol", "name", "sector", "industry"]),
        ),
    ):
        seed.run(skip_prices=True, session_factory=_session_factory(factory))

    with factory() as session:
        logs = session.scalars(select(RefreshLog)).all()
    assert logs[0].status == "partial_survivorship_biased"
