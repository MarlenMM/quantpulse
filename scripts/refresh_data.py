"""Nightly incremental data refresh (Section 6, Phase 1).

Only ever pulls small, recent increments for the current universe. The
one-time historical backfill and the survivorship-bias-aware population of
`index_membership_history` are a separate, not-yet-written script
(`scripts/seed_initial_data.py`) -- deliberately out of scope here.

I/O (API calls) and DB writes are kept in separate phases: every external
fetch happens concurrently in a thread pool with no DB access, and all
writes happen afterwards, serially, in the main thread. SQLite serializes
writes anyway, so mixing concurrent fetches with concurrent writes would
only add "database is locked" failures without buying any real parallelism.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.config import get_settings
from quantpulse.ingestion import fred_client, wikipedia_client, yfinance_client
from quantpulse.storage.db import get_session
from quantpulse.storage.models import (
    AnalystConsensus,
    FundamentalsSnapshot,
    MacroIndicator,
    PriceHistory,
    RefreshLog,
    Ticker,
)
from quantpulse.utils.market_calendar import is_trading_day

logger = logging.getLogger(__name__)

_MAX_WORKERS = 8
# Fundamentals/analyst consensus/macro don't change daily (Section 6.3) --
# refresh them once a week rather than on every nightly run.
_WEEKLY_REFRESH_WEEKDAY = 0  # Monday

_MACRO_SERIES_FETCHERS = (
    fred_client.fetch_fed_funds_rate,
    fred_client.fetch_cpi,
    fred_client.fetch_unemployment_rate,
    fred_client.fetch_gdp,
    fred_client.fetch_treasury_yield_10y,
    fred_client.fetch_treasury_yield_2y,
)


@dataclass
class TickerFetchResult:
    symbol: str
    price_df: pd.DataFrame | None = None
    fundamentals: dict[str, Any] | None = None
    analyst_consensus: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


def sync_universe(session: Session) -> int:
    """Upsert current S&P 500 constituents into `tickers`; mark removed ones inactive.

    This is ongoing universe maintenance, not the survivorship-bias-aware
    historical reconstruction -- that lives in `index_membership_history`
    and is populated by the (separate, not-yet-written) cold-start script.
    """
    constituents = wikipedia_client.fetch_sp500_constituents()
    current_symbols = set(constituents["symbol"])
    existing = {t.symbol: t for t in session.scalars(select(Ticker))}

    for row in constituents.itertuples(index=False):
        ticker = existing.get(row.symbol)
        if ticker is None:
            session.add(
                Ticker(
                    symbol=row.symbol,
                    name=row.name,
                    sector=row.sector,
                    industry=row.industry,
                    exchange=row.exchange,
                    asset_type=row.asset_type,
                    is_active=True,
                )
            )
        else:
            ticker.name = row.name
            ticker.sector = row.sector
            ticker.industry = row.industry
            ticker.is_active = True

    for symbol, ticker in existing.items():
        if symbol not in current_symbols and ticker.asset_type == "equity":
            ticker.is_active = False

    session.flush()
    return len(constituents)


def _last_price_date(session: Session, symbol: str) -> date | None:
    stmt = (
        select(PriceHistory.date)
        .where(PriceHistory.symbol == symbol)
        .order_by(PriceHistory.date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def fetch_ticker_data(
    symbol: str, last_price_date: date | None, is_weekly: bool
) -> TickerFetchResult:
    """Pure I/O: call external APIs for one ticker. No DB access -- safe to run concurrently."""
    result = TickerFetchResult(symbol=symbol)

    try:
        df = yfinance_client.fetch_price_history(symbol, period="5d")
        if last_price_date is not None:
            df = df[df["date"] > pd.Timestamp(last_price_date)]
        result.price_df = df
    except Exception as exc:
        result.errors.append(f"price_history: {exc}")

    if is_weekly:
        try:
            result.fundamentals = yfinance_client.fetch_fundamentals(symbol)
        except Exception as exc:
            result.errors.append(f"fundamentals: {exc}")
        try:
            result.analyst_consensus = yfinance_client.fetch_analyst_consensus(symbol)
        except Exception as exc:
            result.errors.append(f"analyst_consensus: {exc}")

    return result


def _upsert_price_history(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict("records")
    stmt = sqlite_insert(PriceHistory).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={c: stmt.excluded[c] for c in ("open", "high", "low", "close", "adj_close", "volume")},
    )
    session.execute(stmt)
    return len(records)


def _upsert_fundamentals(session: Session, symbol: str, as_of: date, data: dict[str, Any]) -> None:
    values = {k: v for k, v in data.items() if k != "symbol"}
    stmt = sqlite_insert(FundamentalsSnapshot).values(symbol=symbol, as_of_date=as_of, **values)
    # Point-in-time data is append-only (Section 6.8): a same-day re-run
    # leaves the first-written snapshot alone rather than overwriting it.
    stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "as_of_date"])
    session.execute(stmt)


def _upsert_analyst_consensus(
    session: Session, symbol: str, as_of: date, data: dict[str, Any]
) -> None:
    values = {k: v for k, v in data.items() if k != "symbol"}
    stmt = sqlite_insert(AnalystConsensus).values(symbol=symbol, as_of_date=as_of, **values)
    stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "as_of_date"])
    session.execute(stmt)


def refresh_macro_indicators(session: Session) -> int:
    today = date.today()
    lookback = today - timedelta(days=14)
    rows = 0
    for fetch in _MACRO_SERIES_FETCHERS:
        try:
            df = fetch(start_date=lookback, end_date=today)
        except ValueError:
            logger.warning("Skipping macro series %s: FRED_API_KEY not set", fetch.__name__)
            continue
        except Exception:
            logger.exception("Failed to fetch macro series %s", fetch.__name__)
            continue
        if df.empty:
            continue
        records = df.to_dict("records")
        stmt = sqlite_insert(MacroIndicator).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "indicator_name"], set_={"value": stmt.excluded["value"]}
        )
        session.execute(stmt)
        rows += len(records)
    return rows


def run(job_name: str = "refresh_data") -> None:
    logging.basicConfig(level=get_settings().log_level)
    started_at = datetime.now()
    today = started_at.date()
    status = "success"
    rows_updated = 0

    if not is_trading_day(today):
        logger.info("%s: market closed today (%s), skipping refresh", job_name, today)
        with get_session() as session:
            session.add(
                RefreshLog(
                    job_name=job_name,
                    run_timestamp=started_at,
                    status="skipped_non_trading_day",
                    rows_updated=0,
                )
            )
        return

    try:
        with get_session() as session:
            rows_updated += sync_universe(session)
            active = {
                t.symbol: _last_price_date(session, t.symbol)
                for t in session.scalars(select(Ticker).where(Ticker.is_active))
            }

        is_weekly = today.weekday() == _WEEKLY_REFRESH_WEEKDAY

        results: list[TickerFetchResult] = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(fetch_ticker_data, symbol, last_date, is_weekly): symbol
                for symbol, last_date in active.items()
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    logger.exception("Unhandled failure fetching %s", symbol)
                    status = "partial"

        with get_session() as session:
            for result in results:
                if result.errors:
                    logger.warning("%s: %s", result.symbol, "; ".join(result.errors))
                    status = "partial"
                if result.price_df is not None:
                    rows_updated += _upsert_price_history(session, result.price_df)
                if result.fundamentals is not None:
                    _upsert_fundamentals(session, result.symbol, today, result.fundamentals)
                    rows_updated += 1
                if result.analyst_consensus is not None:
                    _upsert_analyst_consensus(
                        session, result.symbol, today, result.analyst_consensus
                    )
                    rows_updated += 1

        if is_weekly:
            with get_session() as session:
                rows_updated += refresh_macro_indicators(session)

    except Exception:
        logger.exception("%s failed", job_name)
        status = "failed"

    with get_session() as session:
        session.add(
            RefreshLog(
                job_name=job_name,
                run_timestamp=started_at,
                status=status,
                rows_updated=rows_updated,
            )
        )


if __name__ == "__main__":
    run()
