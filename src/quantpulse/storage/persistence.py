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
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.storage.models import (
    AnalystConsensus,
    BacktestResult,
    CompositeScore,
    EconomicCalendarEvent,
    Forecast,
    FundamentalsSnapshot,
    IndexMembershipHistory,
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

# The plain metric columns of a fundamentals snapshot (the JSON sector-specific
# column is unpacked separately into `p_ffo`).
_FUNDAMENTAL_METRIC_COLUMNS = (
    "pe",
    "pb",
    "ps",
    "peg",
    "eps",
    "revenue_growth",
    "debt_equity",
    "roe",
    "roa",
    "fcf",
    "div_yield",
)
_ANALYST_HISTORY_COLUMNS = (
    "symbol",
    "as_of_date",
    "strong_buy",
    "buy",
    "hold",
    "sell",
    "strong_sell",
    "mean_price_target",
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


# --------------------------------------------------------------------------- #
# Phase 6 — composite-scoring gather (point-in-time reads) + writer
#
# Every reader below is strictly point-in-time: it never returns a row dated
# after `as_of`, and the "latest per symbol" readers take the most recent
# snapshot at or before `as_of`. That is what keeps the composite honest for a
# backtest (Section 7.5 step 5) -- scoring "as of March 3rd" sees only data that
# existed on March 3rd.
# --------------------------------------------------------------------------- #


def read_active_ohlcv(session: Session, *, as_of: date, lookback_days: int) -> pd.DataFrame:
    """OHLCV history for all active equities over `[as_of - lookback, as_of]`.

    The input to the technical and momentum category scorers. Returns columns
    `symbol, date, open, high, low, close, volume`, oldest first per symbol.
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(
            PriceHistory.symbol,
            PriceHistory.date,
            PriceHistory.open,
            PriceHistory.high,
            PriceHistory.low,
            PriceHistory.close,
            PriceHistory.volume,
        )
        .join(Ticker, Ticker.symbol == PriceHistory.symbol)
        .where(
            Ticker.is_active,
            Ticker.asset_type == "equity",
            PriceHistory.date >= start,
            PriceHistory.date <= as_of,
        )
        .order_by(PriceHistory.symbol, PriceHistory.date)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def read_latest_fundamentals(
    session: Session, *, as_of: date, lookback_days: int = 180
) -> pd.DataFrame:
    """Each symbol's most recent fundamentals snapshot at or before `as_of`.

    Shaped for `fundamental.score_fundamentals`: `symbol`, `sector` (from
    `tickers`), the plain metric columns, and `p_ffo` unpacked from the
    `sector_specific_metrics` JSON (REITs, Section 7.2). The lookback bounds the
    scan (fundamentals refresh weekly, so the latest is always recent).
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(FundamentalsSnapshot)
        .where(FundamentalsSnapshot.as_of_date >= start, FundamentalsSnapshot.as_of_date <= as_of)
        .order_by(FundamentalsSnapshot.as_of_date)
    )
    snapshots = session.scalars(stmt).all()
    columns = ["symbol", "sector", "as_of_date", *_FUNDAMENTAL_METRIC_COLUMNS, "p_ffo"]
    if not snapshots:
        return pd.DataFrame(columns=columns)

    records = []
    for snap in snapshots:
        record: dict[str, Any] = {"symbol": snap.symbol, "as_of_date": snap.as_of_date}
        for metric in _FUNDAMENTAL_METRIC_COLUMNS:
            record[metric] = getattr(snap, metric)
        record["p_ffo"] = (snap.sector_specific_metrics or {}).get("p_ffo")
        records.append(record)

    latest = (
        pd.DataFrame(records).sort_values("as_of_date").groupby("symbol", as_index=False).last()
    )
    sectors = {t.symbol: t.sector for t in session.scalars(select(Ticker))}
    latest["sector"] = latest["symbol"].map(sectors)
    return latest[columns]


def read_analyst_history(
    session: Session, *, as_of: date, lookback_days: int = 180
) -> pd.DataFrame:
    """Every analyst-consensus snapshot per symbol in `[as_of - lookback, as_of]`.

    The full point-in-time history `analyst_consensus.score_analyst_consensus`
    needs to fit its estimate-revision trend (Section 7.4). The lookback spans
    more than the trend window so the slope has data on both ends.
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(AnalystConsensus)
        .where(AnalystConsensus.as_of_date >= start, AnalystConsensus.as_of_date <= as_of)
        .order_by(AnalystConsensus.symbol, AnalystConsensus.as_of_date)
    )
    rows = session.scalars(stmt).all()
    records = [{c: getattr(r, c) for c in _ANALYST_HISTORY_COLUMNS} for r in rows]
    return pd.DataFrame(records, columns=list(_ANALYST_HISTORY_COLUMNS))


def read_latest_sentiment(
    session: Session, *, as_of: date, lookback_days: int = 30
) -> dict[str, float]:
    """Each symbol's most recent Tier-1 aggregate sentiment polarity at or before `as_of`."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(SentimentScore.symbol, SentimentScore.sentiment_score)
        .where(
            SentimentScore.source == "tier1_aggregate",
            SentimentScore.date >= start,
            SentimentScore.date <= as_of,
        )
        .order_by(SentimentScore.date)  # ascending -> last write per symbol wins
    )
    latest: dict[str, float] = {}
    for symbol, score in session.execute(stmt):
        if score is not None:
            latest[symbol] = float(score)
    return latest


def read_tier2_news(session: Session, *, as_of: date, lookback_days: int = 21) -> pd.DataFrame:
    """Tier-2 industry news (`matched_theme`, `sentiment_score`) over the trailing window.

    Feeds the per-stock industry tilt (`scoring.tier2_thematic_tilt`). Bounds
    `published_at` to `[as_of - lookback, end of as_of]`; undated articles are
    excluded (they can't be placed point-in-time).
    """
    start = datetime.combine(as_of - timedelta(days=lookback_days), time.min)
    end = datetime.combine(as_of, time.max)
    stmt = select(NewsEvent.matched_theme, NewsEvent.sentiment_score).where(
        NewsEvent.tier == 2,
        NewsEvent.published_at >= start,
        NewsEvent.published_at <= end,
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["matched_theme", "sentiment_score"])


def read_theme_members(session: Session) -> dict[str, set[str]]:
    """Theme/basket name -> its member symbols, from `thematic_baskets`."""
    members: dict[str, set[str]] = {}
    for theme, symbol in session.execute(select(ThematicBasket.theme_name, ThematicBasket.symbol)):
        members.setdefault(theme, set()).add(symbol)
    return members


def read_latest_regime_score(session: Session, *, as_of: date) -> float | None:
    """The most recent non-null Market Regime Index score at or before `as_of`."""
    stmt = (
        select(MarketRegime.regime_score)
        .where(MarketRegime.date <= as_of, MarketRegime.regime_score.is_not(None))
        .order_by(MarketRegime.date.desc())
        .limit(1)
    )
    value = session.scalars(stmt).first()
    return float(value) if value is not None else None


def read_recent_insider_transactions(
    session: Session, *, as_of: date, lookback_days: int = 180
) -> pd.DataFrame:
    """Insider transactions filed in `[as_of - lookback, as_of]` (Section 24).

    Shaped for `smart_money.score_insider_activity`: `symbol, insider_name,
    transaction_code, shares`. Keyed on `filing_date` so the point-in-time cut
    matches when the market actually learned of each trade.
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = select(
        InsiderTransaction.symbol,
        InsiderTransaction.insider_name,
        InsiderTransaction.transaction_code,
        InsiderTransaction.shares,
    ).where(InsiderTransaction.filing_date >= start, InsiderTransaction.filing_date <= as_of)
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["symbol", "insider_name", "transaction_code", "shares"])


def read_latest_institutional(session: Session, *, as_of: date) -> pd.DataFrame:
    """Each symbol's most recent 13F institutional-ownership row at or before `as_of`."""
    stmt = (
        select(
            InstitutionalOwnership.symbol,
            InstitutionalOwnership.total_shares_held,
            InstitutionalOwnership.change_from_prior_quarter,
            InstitutionalOwnership.num_filers,
        )
        .where(InstitutionalOwnership.quarter_end_date <= as_of)
        .order_by(InstitutionalOwnership.quarter_end_date)
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame(
        rows, columns=["symbol", "total_shares_held", "change_from_prior_quarter", "num_filers"]
    )
    return df if df.empty else df.groupby("symbol", as_index=False).last()


def read_latest_options(session: Session, *, as_of: date, lookback_days: int = 10) -> pd.DataFrame:
    """Each symbol's most recent options snapshot in the trailing window at or before `as_of`."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(
            OptionsSignal.symbol,
            OptionsSignal.put_call_ratio,
            OptionsSignal.atm_implied_volatility,
            OptionsSignal.iv_rank,
        )
        .where(OptionsSignal.date >= start, OptionsSignal.date <= as_of)
        .order_by(OptionsSignal.date)
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame(
        rows, columns=["symbol", "put_call_ratio", "atm_implied_volatility", "iv_rank"]
    )
    return df if df.empty else df.groupby("symbol", as_index=False).last()


def read_latest_short_interest(
    session: Session, *, as_of: date, lookback_days: int = 45
) -> pd.DataFrame:
    """Each symbol's most recent short-interest snapshot at or before `as_of`."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(ShortInterest.symbol, ShortInterest.pct_float_short, ShortInterest.days_to_cover)
        .where(ShortInterest.as_of_date >= start, ShortInterest.as_of_date <= as_of)
        .order_by(ShortInterest.as_of_date)
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame(rows, columns=["symbol", "pct_float_short", "days_to_cover"])
    return df if df.empty else df.groupby("symbol", as_index=False).last()


def upsert_composite_scores(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append composite-score rows (append-only, point-in-time -- never overwritten)."""
    return _append_only(session, CompositeScore, records)


# --------------------------------------------------------------------------- #
# Phase 7 — forecasts + backtest track record (Section 7.6, 13)
# --------------------------------------------------------------------------- #


def upsert_forecasts(session: Session, records: Sequence[dict[str, Any]]) -> int:
    """Append per-(symbol, date, horizon, model) forecast rows (append-only, Section 6.8)."""
    return _append_only(session, Forecast, records)


def insert_backtest_result(session: Session, record: dict[str, Any]) -> int:
    """Insert one strategy-backtest run's track record (append-only log; auto-id PK)."""
    session.add(BacktestResult(**record))
    return 1


def read_membership_intervals(session: Session, index_name: str) -> pd.DataFrame:
    """Point-in-time index membership intervals for the survivorship-aware backtest.

    Returns `symbol, added_date, removed_date` (removed_date null = still a
    member) for `index_name`, straight from `index_membership_history` (Section 5).
    The strategy backtest builds its `eligible(as_of)` universe from these so a
    company that was a member in the past -- then delisted -- is still traded on
    the dates it belonged, exactly what Section 22 demands.
    """
    stmt = select(
        IndexMembershipHistory.symbol,
        IndexMembershipHistory.added_date,
        IndexMembershipHistory.removed_date,
    ).where(IndexMembershipHistory.index_name == index_name)
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["symbol", "added_date", "removed_date"])


def read_adj_close_panel(
    session: Session, *, start: date, end: date, symbols: Sequence[str] | None = None
) -> pd.DataFrame:
    """Wide adjusted-close panel (index=date, columns=symbol) over `[start, end]`.

    Deliberately does **not** filter on `Ticker.is_active`: a survivorship-honest
    backtest needs the price history of names that were later removed, not just
    today's survivors (Section 22). `symbols`, if given, restricts the columns
    (e.g. to a single index's ever-members). An empty result yields an empty
    frame rather than raising.
    """
    conditions = [PriceHistory.date >= start, PriceHistory.date <= end]
    if symbols is not None:
        conditions.append(PriceHistory.symbol.in_(list(symbols)))
    stmt = (
        select(PriceHistory.date, PriceHistory.symbol, PriceHistory.adj_close)
        .where(*conditions)
        .order_by(PriceHistory.date)
    )
    rows = session.execute(stmt).all()
    long_df = pd.DataFrame(rows, columns=["date", "symbol", "adj_close"])
    if long_df.empty:
        return pd.DataFrame()
    panel = long_df.pivot_table(index="date", columns="symbol", values="adj_close")
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index))
    return panel.sort_index()
