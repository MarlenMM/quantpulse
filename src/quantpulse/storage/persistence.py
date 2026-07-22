"""Write/read helpers for the Phase 4/5 tables (Section 13).

The Phase-4/5 analysis and ingestion modules are pure functions over in-memory
frames; this module is the thin persistence seam between them and the database,
used by both the nightly refresh (writer) and Phase 6 composite scoring
(reader). It deliberately mirrors the conventions already established inline in
`scripts/refresh_data.py`:

- Point-in-time data (Section 6.8) is **append-only**: a same-day re-run uses
  `on_conflict_do_nothing`, leaving the first-written row untouched rather than
  overwriting history. That applies to every snapshot table here
  (`sentiment_scores`, `market_regime`, `options_signals`, `short_interest`,
  `institutional_ownership`, `insider_transactions`, `news_events`).
- Config-derived data (`thematic_baskets`, `economic_calendar`) is idempotently
  refreshed so the DB reflects the current config, since a curated basket's
  membership can legitimately change between runs.

All writers take plain ``list[dict]`` records (built by the caller from
DataFrames/dataclasses) and return the number of rows written, so the refresh
job's ``rows_updated`` accounting stays uniform.
"""

import hashlib
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.storage.models import (
    EconomicCalendarEvent,
    InsiderTransaction,
    InstitutionalOwnership,
    MacroIndicator,
    MarketRegime,
    NewsEvent,
    OptionsSignal,
    PriceHistory,
    SentimentScore,
    ShortInterest,
    ThematicBasket,
    Ticker,
)


def article_id_for(source_url: str | None, *, fallback: str) -> str:
    """Stable 64-char id for a news article, hashed from its source URL.

    Re-ingesting the same article (same URL) yields the same id, so
    `news_events` dedupes on it via the primary key. `fallback` (e.g. the
    title) is hashed instead when a row has no URL, so URL-less articles still
    get a deterministic id rather than colliding on the empty string.
    """
    basis = source_url if source_url else fallback
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _append_only(session: Session, model: type[Any], records: Sequence[dict[str, Any]]) -> int:
    """Insert `records`, skipping any whose primary key already exists (Section 6.8)."""
    if not records:
        return 0
    stmt = sqlite_insert(model).values(list(records)).on_conflict_do_nothing()
    session.execute(stmt)
    return len(records)


# --------------------------------------------------------------------------- #
# Writers — Phase 4 (news intelligence)
# --------------------------------------------------------------------------- #


def upsert_sentiment_scores(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append Tier-1 decay-weighted sentiment rows (append-only, Section 6.8)."""
    return _append_only(session, SentimentScore, records)


def upsert_news_events(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Insert news-event rows, deduped on `article_id` (a re-seen article is skipped)."""
    return _append_only(session, NewsEvent, records)


def upsert_market_regime(session: Session, record: dict[str, Any]) -> int:
    """Append one day's Market Regime Index row (append-only — never recompute history)."""
    return _append_only(session, MarketRegime, [record])


def replace_thematic_baskets(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Replace the whole `thematic_baskets` config table with `records`.

    A curated basket's membership can change between runs (a name added or
    dropped), so this is a full replace rather than an append -- the table is
    tiny and purely config-derived, so it should always reflect the current
    `thematic_mapping` config exactly.
    """
    session.execute(delete(ThematicBasket))
    if not records:
        return 0
    session.execute(sqlite_insert(ThematicBasket).values(list(records)))
    return len(records)


def upsert_economic_calendar(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Insert scheduled macro events, skipping any already present (idempotent)."""
    return _append_only(session, EconomicCalendarEvent, records)


# --------------------------------------------------------------------------- #
# Writers — Phase 5 (smart money)
# --------------------------------------------------------------------------- #


def insert_insider_transactions(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Insert Form-4 transaction rows, deduped by the natural unique key (Section 24)."""
    return _append_only(session, InsiderTransaction, records)


def upsert_institutional_ownership(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append per-symbol quarterly 13F ownership rows (append-only per quarter)."""
    return _append_only(session, InstitutionalOwnership, records)


def upsert_options_signals(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append daily options-positioning snapshots (append-only per (symbol, date))."""
    return _append_only(session, OptionsSignal, records)


def upsert_short_interest(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append short-interest snapshots (append-only per (symbol, as_of_date))."""
    return _append_only(session, ShortInterest, records)


# --------------------------------------------------------------------------- #
# Readers (used now by IV-rank + the Market Regime Index; extended in Phase 6)
# --------------------------------------------------------------------------- #


def read_recent_atm_iv(
    session: Session, symbol: str, *, before: date, lookback_days: int = 365
) -> list[float]:
    """Prior at-the-money IV snapshots for `symbol` in `[before - lookback, before)`.

    The trailing history `options_client.compute_iv_rank` ranks today's IV
    against. Deliberately excludes `before` itself (strictly `< before`) so a
    day's IV-rank is computed only from data that predates it -- point-in-time,
    no look-ahead (Section 7.5 step 5). Nulls are dropped.
    """
    start = before - timedelta(days=lookback_days)
    stmt = (
        select(OptionsSignal.atm_implied_volatility)
        .where(
            OptionsSignal.symbol == symbol,
            OptionsSignal.date >= start,
            OptionsSignal.date < before,
            OptionsSignal.atm_implied_volatility.is_not(None),
        )
        .order_by(OptionsSignal.date)
    )
    return [float(v) for v in session.scalars(stmt) if v is not None]


def read_latest_macro_value(session: Session, indicator_name: str, *, as_of: date) -> float | None:
    """Most recent value of a macro series at or before `as_of`, or None if none stored.

    Point-in-time: never reads a value dated after `as_of`. Used for the
    Market Regime Index's VIX level and the 10Y-2Y yield-curve spread inputs
    (Section 28), which are stored as rows in `macro_indicators`.
    """
    stmt = (
        select(MacroIndicator.value)
        .where(MacroIndicator.indicator_name == indicator_name, MacroIndicator.date <= as_of)
        .order_by(MacroIndicator.date.desc())
        .limit(1)
    )
    value = session.scalars(stmt).first()
    return float(value) if value is not None else None


def read_macro_series(
    session: Session, indicator_name: str, *, as_of: date, lookback_days: int
) -> list[float]:
    """Values of a macro series over `[as_of - lookback, as_of]`, oldest first.

    The history the VIX-percentile step of the Market Regime Index ranks the
    current level against (Section 5's "VIX level/percentile").
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(MacroIndicator.value)
        .where(
            MacroIndicator.indicator_name == indicator_name,
            MacroIndicator.date >= start,
            MacroIndicator.date <= as_of,
        )
        .order_by(MacroIndicator.date)
    )
    return [float(v) for v in session.scalars(stmt) if v is not None]


def read_active_price_history(session: Session, *, as_of: date, lookback_days: int) -> pd.DataFrame:
    """Adjusted-close history for all active equities over `[as_of - lookback, as_of]`.

    The raw material for the Market Regime Index's breadth input (% of the
    universe trading above its 200-DMA). Returns columns `symbol, date,
    adj_close`; never includes bars dated after `as_of` (point-in-time).
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(PriceHistory.symbol, PriceHistory.date, PriceHistory.adj_close)
        .join(Ticker, Ticker.symbol == PriceHistory.symbol)
        .where(
            Ticker.is_active,
            Ticker.asset_type == "equity",
            PriceHistory.date >= start,
            PriceHistory.date <= as_of,
        )
        .order_by(PriceHistory.symbol, PriceHistory.date)
    )
    rows: Iterable[Any] = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["symbol", "date", "adj_close"])
