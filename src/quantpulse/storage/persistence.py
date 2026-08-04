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
import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
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
    PatternSignal,
    PriceHistory,
    RefreshLog,
    SentimentScore,
    ShortInterest,
    ThematicBasket,
    Ticker,
)

logger = logging.getLogger(__name__)

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
    # A non-positive adjusted close is not a price. Free sources emit them for
    # some delisted names (a real example: DEC carries 732 bars with
    # adj_close = 0.0 while its raw close is $1.44). Left in, one such symbol
    # poisons everything computed from the panel: `_equal_weight_benchmark`
    # rebases each name to its first observation, so dividing by that zero
    # turns the ENTIRE benchmark index into `inf`, and any return computed
    # across a zero is `inf` or -100%. Excluded here, at the read boundary, so
    # already-stored bad rows are neutralized for every consumer rather than
    # each one needing its own guard -- matching `backtest._closes`, which
    # already keeps only strictly-positive closes.
    conditions = [
        PriceHistory.date >= start,
        PriceHistory.date <= end,
        PriceHistory.adj_close > 0,
    ]
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
    return _drop_broken_adjustment_series(panel.sort_index())


# A bar-to-bar price ratio beyond this is not a market move, it is a broken
# split/dividend adjustment factor. Real index constituents top out around a
# 2-3x single-day move (a takeover pop, a biotech readout); 10x is far above
# anything genuine and still catches the real artefacts by orders of magnitude.
_MAX_PLAUSIBLE_BAR_RATIO = 10.0


def _drop_broken_adjustment_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove symbols whose adjusted-close series contains an impossible jump.

    Free sources serve badly-adjusted history for long-delisted names, and the
    damage is not subtle. Measured on a real 495-symbol backfill: CBE moves
    $0.005 -> $305.00 in one bar (a 3,399,900% "return"), TNB reaches $31,080,
    and COMS spans 16.7 BILLION times end to end. 10,689 bars carry sub-penny
    adjusted closes.

    A backtest cannot survive this. Those names get ranked top by any momentum
    signal, "bought", and realize million-percent gains -- which is exactly what
    happened: the strategy reported Sharpe 0.310, CAGR 64.77% and a bootstrap
    interval that *excluded zero*, i.e. a statistically significant edge made
    entirely of broken data. That is the worst failure mode this project has,
    because every honesty mechanism downstream (cost model, survivorship
    handling, confidence intervals) faithfully described a fiction.

    The whole symbol is dropped, not the offending bar: a broken adjustment
    factor corrupts the entire series it scales, so the rest of that history
    cannot be trusted either. Dropping loses a name; keeping it loses the
    backtest.
    """
    if panel.empty:
        return panel
    ratio = panel / panel.shift(1)
    extreme = ((ratio > _MAX_PLAUSIBLE_BAR_RATIO) | (ratio < 1.0 / _MAX_PLAUSIBLE_BAR_RATIO)).any()
    broken = [str(symbol) for symbol, flagged in extreme.items() if bool(flagged)]
    if not broken:
        return panel
    logger.warning(
        "Dropping %d symbol(s) with an implausible price jump (>%gx in one bar), "
        "which indicates a broken adjustment factor rather than a market move: %s",
        len(broken),
        _MAX_PLAUSIBLE_BAR_RATIO,
        ", ".join(sorted(broken)[:20]),
    )
    return panel.drop(columns=broken)


# --------------------------------------------------------------------------- #
# Phase 10 — UI read helpers (Section 12)
#
# The readers above are point-in-time gathers for the nightly job: "what was
# knowable as of date X." These are the opposite question — "what does the app
# show right now" — so they read the LATEST stored row rather than an as-of
# slice. Both live here because both are plain SQL over the same tables and
# `app/` must never contain queries (Section 14's UI-agnostic engine); the
# Streamlit-side `@st.cache_data` wrappers around them live in `app/lib/data.py`.
# --------------------------------------------------------------------------- #


def read_latest_score_date(session: Session, *, profile: str = "balanced") -> date | None:
    """The most recent date `composite_scores` was written for `profile`."""
    stmt = (
        select(CompositeScore.date)
        .where(CompositeScore.profile == profile)
        .order_by(CompositeScore.date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def read_screener_rows(
    session: Session, *, profile: str = "balanced", as_of: date | None = None
) -> pd.DataFrame:
    """The ranked screener table: latest composite scores joined to ticker metadata.

    Returns one row per scored symbol with every sub-score, so the Screener's
    re-weighting sliders (Section 8) can recompute a composite client-side
    without re-running the pipeline — which is exactly why the stored
    sub-scores are weight-independent (Section 7.5).
    """
    resolved = as_of or read_latest_score_date(session, profile=profile)
    if resolved is None:
        return pd.DataFrame()
    stmt = (
        select(
            CompositeScore.symbol,
            Ticker.name,
            Ticker.sector,
            Ticker.asset_type,
            CompositeScore.date,
            CompositeScore.fundamental_score,
            CompositeScore.technical_score,
            CompositeScore.analyst_score,
            CompositeScore.sentiment_score,
            CompositeScore.momentum_score,
            CompositeScore.industry_macro_score,
            CompositeScore.smart_money_score,
            CompositeScore.composite_score,
            CompositeScore.percentile_rank,
            CompositeScore.rating,
            CompositeScore.data_confidence,
        )
        .join(Ticker, Ticker.symbol == CompositeScore.symbol)
        .where(CompositeScore.profile == profile, CompositeScore.date == resolved)
        .order_by(CompositeScore.composite_score.desc())
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "name",
            "sector",
            "asset_type",
            "date",
            "fundamental_score",
            "technical_score",
            "analyst_score",
            "sentiment_score",
            "momentum_score",
            "industry_macro_score",
            "smart_money_score",
            "composite_score",
            "percentile_rank",
            "rating",
            "data_confidence",
        ],
    )


def read_rating_changes(
    session: Session, *, profile: str = "balanced", limit: int = 25
) -> pd.DataFrame:
    """Symbols whose rating changed between the two most recent scoring dates.

    Section 10's "what changed since yesterday" view, which the append-only
    point-in-time schema (Section 6.8) makes almost free: compare the latest
    two stored snapshots instead of maintaining a separate change log. Empty
    frame when there aren't two snapshots yet.
    """
    dates = list(
        session.scalars(
            select(CompositeScore.date)
            .where(CompositeScore.profile == profile)
            .distinct()
            .order_by(CompositeScore.date.desc())
            .limit(2)
        )
    )
    if len(dates) < 2:
        return pd.DataFrame()
    current, previous = dates[0], dates[1]

    def _snapshot(target: date) -> dict[str, tuple[str, float]]:
        stmt = select(
            CompositeScore.symbol, CompositeScore.rating, CompositeScore.composite_score
        ).where(CompositeScore.profile == profile, CompositeScore.date == target)
        return {row.symbol: (row.rating, row.composite_score) for row in session.execute(stmt)}

    now, before = _snapshot(current), _snapshot(previous)
    changes = [
        {
            "symbol": symbol,
            "previous_rating": before[symbol][0],
            "rating": rating,
            "previous_score": before[symbol][1],
            "composite_score": score,
            "score_change": score - before[symbol][1],
        }
        for symbol, (rating, score) in now.items()
        if symbol in before and before[symbol][0] != rating
    ]
    frame = pd.DataFrame(changes)
    if frame.empty:
        return frame
    return frame.reindex(frame["score_change"].abs().sort_values(ascending=False).index).head(limit)


def read_symbol_ohlcv(session: Session, symbol: str, *, lookback_days: int = 400) -> pd.DataFrame:
    """One symbol's recent OHLCV bars, oldest first — the Stock Detail price chart."""
    start = date.today() - timedelta(days=lookback_days)
    stmt = (
        select(
            PriceHistory.date,
            PriceHistory.open,
            PriceHistory.high,
            PriceHistory.low,
            PriceHistory.close,
            PriceHistory.adj_close,
            PriceHistory.volume,
        )
        .where(PriceHistory.symbol == symbol, PriceHistory.date >= start)
        .order_by(PriceHistory.date)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
    )


def read_symbol_forecasts(session: Session, symbol: str) -> pd.DataFrame:
    """The most recent generated forecast set for one symbol, all models and horizons."""
    latest = session.scalars(
        select(Forecast.generated_date)
        .where(Forecast.symbol == symbol)
        .order_by(Forecast.generated_date.desc())
        .limit(1)
    ).first()
    if latest is None:
        return pd.DataFrame()
    stmt = (
        select(
            Forecast.model_name,
            Forecast.horizon_days,
            Forecast.point_return,
            Forecast.point_price,
            Forecast.lower_price,
            Forecast.upper_price,
            Forecast.historical_hit_rate,
            Forecast.baseline_hit_rate,
            Forecast.generated_date,
        )
        .where(Forecast.symbol == symbol, Forecast.generated_date == latest)
        .order_by(Forecast.horizon_days, Forecast.model_name)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows,
        columns=[
            "model_name",
            "horizon_days",
            "point_return",
            "point_price",
            "lower_price",
            "upper_price",
            "historical_hit_rate",
            "baseline_hit_rate",
            "generated_date",
        ],
    )


def read_symbol_patterns(
    session: Session, symbol: str, *, lookback_days: int = 120
) -> pd.DataFrame:
    """Detected chart/candlestick patterns for one symbol in the trailing window."""
    start = date.today() - timedelta(days=lookback_days)
    stmt = (
        select(
            PatternSignal.date,
            PatternSignal.pattern_type,
            PatternSignal.direction,
            PatternSignal.confidence,
        )
        .where(PatternSignal.symbol == symbol, PatternSignal.date >= start)
        .order_by(PatternSignal.date.desc())
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["date", "pattern_type", "direction", "confidence"])


def read_latest_analyst_consensus(session: Session, symbol: str) -> dict[str, Any] | None:
    """One symbol's most recent Wall Street rating counts + mean price target."""
    stmt = (
        select(AnalystConsensus)
        .where(AnalystConsensus.symbol == symbol)
        .order_by(AnalystConsensus.as_of_date.desc())
        .limit(1)
    )
    row = session.scalars(stmt).first()
    if row is None:
        return None
    return {
        "as_of_date": row.as_of_date,
        "strong_buy": row.strong_buy,
        "buy": row.buy,
        "hold": row.hold,
        "sell": row.sell,
        "strong_sell": row.strong_sell,
        "mean_price_target": row.mean_price_target,
    }


def read_backtest_history(session: Session, *, limit: int = 20) -> pd.DataFrame:
    """Stored strategy backtest runs, newest first — the Track Record page."""
    stmt = select(BacktestResult).order_by(BacktestResult.run_date.desc()).limit(limit)
    rows = session.scalars(stmt).all()
    return pd.DataFrame(
        [
            {
                "run_date": row.run_date,
                "period_start": row.period_start,
                "period_end": row.period_end,
                "cadence": row.cadence,
                "n_periods": row.n_periods,
                "sharpe": row.sharpe,
                "sharpe_ci_low": row.sharpe_ci_low,
                "sharpe_ci_high": row.sharpe_ci_high,
                "cagr": row.cagr,
                "cagr_ci_low": row.cagr_ci_low,
                "cagr_ci_high": row.cagr_ci_high,
                "ci_confidence_level": row.ci_confidence_level,
                "max_drawdown": row.max_drawdown,
                "win_rate": row.win_rate,
                "benchmark_cagr": row.benchmark_cagr,
                "benchmark_sharpe": row.benchmark_sharpe,
                "avg_turnover": row.avg_turnover,
                "assumed_txn_cost": row.assumed_txn_cost,
            }
            for row in rows
        ]
    )


def read_recent_market_regime(session: Session, *, limit: int = 90) -> pd.DataFrame:
    """The Market Regime Index history, oldest first — the Dashboard gauge + trend."""
    stmt = select(MarketRegime).order_by(MarketRegime.date.desc()).limit(limit)
    rows = list(session.scalars(stmt).all())[::-1]
    return pd.DataFrame(
        [
            {
                "date": row.date,
                "vix_level": row.vix_level,
                "breadth_pct_above_200dma": row.breadth_pct_above_200dma,
                "macro_news_tone": row.macro_news_tone,
                "yield_curve_spread": row.yield_curve_spread,
                "regime_score": row.regime_score,
                "regime_label": row.regime_label,
            }
            for row in rows
        ]
    )


def read_market_moving_news(
    session: Session, *, tiers: Sequence[int] = (2, 3), limit: int = 8, lookback_days: int = 3
) -> pd.DataFrame:
    """Recent Tier-2/3 industry & macro stories — the Dashboard's news panel (Section 12)."""
    cutoff = datetime.combine(date.today() - timedelta(days=lookback_days), time.min)
    stmt = (
        select(
            NewsEvent.title,
            NewsEvent.published_at,
            NewsEvent.tier,
            NewsEvent.matched_theme,
            NewsEvent.event_type,
            NewsEvent.sentiment_score,
            NewsEvent.source,
            NewsEvent.source_url,
        )
        .where(NewsEvent.tier.in_(list(tiers)), NewsEvent.published_at >= cutoff)
        .order_by(NewsEvent.published_at.desc())
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows,
        columns=[
            "title",
            "published_at",
            "tier",
            "matched_theme",
            "event_type",
            "sentiment_score",
            "source",
            "source_url",
        ],
    )


def read_symbol_news(
    session: Session, symbol: str, *, limit: int = 10, lookback_days: int = 21
) -> pd.DataFrame:
    """Tier-1 articles that named `symbol` — the Stock Detail "what's driving this" feed."""
    cutoff = datetime.combine(date.today() - timedelta(days=lookback_days), time.min)
    stmt = (
        select(
            NewsEvent.title,
            NewsEvent.published_at,
            NewsEvent.event_type,
            NewsEvent.sentiment_score,
            NewsEvent.source,
            NewsEvent.source_url,
            NewsEvent.matched_symbols,
        )
        .where(NewsEvent.tier == 1, NewsEvent.published_at >= cutoff)
        .order_by(NewsEvent.published_at.desc())
    )
    rows = session.execute(stmt).all()
    matched = [row for row in rows if row.matched_symbols and symbol in row.matched_symbols]
    return pd.DataFrame(
        [
            {
                "title": row.title,
                "published_at": row.published_at,
                "event_type": row.event_type,
                "sentiment_score": row.sentiment_score,
                "source": row.source,
                "source_url": row.source_url,
            }
            for row in matched[:limit]
        ]
    )


def read_refresh_log(session: Session, *, limit: int = 20) -> pd.DataFrame:
    """Recent nightly-job runs — the Settings page's pipeline-health table (Section 13)."""
    stmt = select(RefreshLog).order_by(RefreshLog.run_timestamp.desc()).limit(limit)
    rows = session.scalars(stmt).all()
    return pd.DataFrame(
        [
            {
                "job_name": row.job_name,
                "run_timestamp": row.run_timestamp,
                "status": row.status,
                "rows_updated": row.rows_updated,
            }
            for row in rows
        ]
    )


def read_data_freshness(session: Session) -> dict[str, date | None]:
    """Newest stored date per major table — Section 12's "show data freshness" rule.

    Powers the "fundamentals last updated 3 days ago" style badges. A table
    that has never been populated maps to `None` rather than being omitted, so
    the UI can distinguish "stale" from "never ran" instead of showing a
    reassuring blank for both.
    """
    return {
        "prices": session.scalars(select(func.max(PriceHistory.date))).first(),
        "fundamentals": session.scalars(select(func.max(FundamentalsSnapshot.as_of_date))).first(),
        "analyst_consensus": session.scalars(select(func.max(AnalystConsensus.as_of_date))).first(),
        "sentiment": session.scalars(select(func.max(SentimentScore.date))).first(),
        "composite_scores": session.scalars(select(func.max(CompositeScore.date))).first(),
        "forecasts": session.scalars(select(func.max(Forecast.generated_date))).first(),
        "market_regime": session.scalars(select(func.max(MarketRegime.date))).first(),
        "backtest": session.scalars(select(func.max(BacktestResult.run_date))).first(),
    }


def read_ticker_universe(session: Session, *, active_only: bool = True) -> pd.DataFrame:
    """`symbol, name, sector, asset_type` for the symbol pickers and sector filters."""
    stmt = select(Ticker.symbol, Ticker.name, Ticker.sector, Ticker.asset_type)
    if active_only:
        stmt = stmt.where(Ticker.is_active)
    rows = session.execute(stmt.order_by(Ticker.symbol)).all()
    return pd.DataFrame(rows, columns=["symbol", "name", "sector", "asset_type"])


def read_latest_prices(session: Session, symbols: Sequence[str]) -> dict[str, float]:
    """Each symbol's most recent close — for pricing a portfolio.

    A symbol with no stored price is simply absent from the result rather than
    mapped to 0.0, which is what lets the Portfolio page flag it as stale
    (Section 30's delisted-holding case) instead of showing a position that
    appears to have become worthless.
    """
    if not symbols:
        return {}
    subquery = (
        select(PriceHistory.symbol, func.max(PriceHistory.date).label("latest"))
        .where(PriceHistory.symbol.in_(list(symbols)))
        .group_by(PriceHistory.symbol)
        .subquery()
    )
    stmt = select(PriceHistory.symbol, PriceHistory.close).join(
        subquery,
        (PriceHistory.symbol == subquery.c.symbol) & (PriceHistory.date == subquery.c.latest),
    )
    return {row.symbol: float(row.close) for row in session.execute(stmt)}
