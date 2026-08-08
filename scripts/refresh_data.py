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

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from types import FrameType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.analysis import (
    analyst_consensus,
    backtest,
    forecasting,
    fundamental,
    macro,
    patterns,
    scoring,
    smart_money,
)
from quantpulse.analysis.investor_profiles import get_profile
from quantpulse.config import get_settings
from quantpulse.ingestion import (
    economic_calendar,
    edgar_13f_client,
    edgar_client,
    fred_client,
    gdelt_client,
    listing_client,
    news_client,
    options_client,
    short_interest_client,
    wikipedia_client,
    yfinance_client,
)
from quantpulse.ingestion import (
    historical_constituents_client as hist,
)
from quantpulse.news_intelligence import (
    entity_extraction,
    event_classifier,
    market_regime,
    sentiment,
    thematic_mapping,
)
from quantpulse.storage import persistence
from quantpulse.storage.db import assert_schema_current, get_session
from quantpulse.storage.models import (
    AnalystConsensus,
    FundamentalsSnapshot,
    IndexMembershipHistory,
    MacroIndicator,
    PriceHistory,
    RefreshLog,
    Ticker,
)
from quantpulse.utils.log import configure_logging
from quantpulse.utils.market_calendar import is_trading_day

logger = logging.getLogger(__name__)

# The exchange's own timezone. Every "what day is it" decision in this job is
# made here rather than in the runner's clock -- see `run()`.
_MARKET_TZ = ZoneInfo("America/New_York")

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

# Cross-asset series ingested daily into `macro_indicators` (Section 28): the
# VIX and the commodity/currency tickers the Market Regime Index and the
# sector overlay consume. `macro.<NAME>` is the stored series name; the value
# is the yfinance ticker fetched for it.
_CROSS_ASSET_TICKERS: dict[str, str] = {
    macro.VIX: "^VIX",
    macro.OIL_WTI: "CL=F",
    macro.GOLD: "GC=F",
    macro.DOLLAR_INDEX: "DX-Y.NYB",
}

# GDELT macro-tone query feeding the Market Regime Index's Tier-3 input
# (Sections 5, 7.3, 28) -- broad economic/policy themes, not any one ticker.
_MACRO_TONE_QUERY = "(economy OR inflation OR federal reserve OR recession OR interest rates)"

# How much trailing history to read for the point-in-time IV-rank and the
# 200-DMA breadth computation, in calendar days (generous vs. the trading-day
# windows they actually need, so weekends/holidays never starve them).
_IV_RANK_LOOKBACK_DAYS = 365
_BREADTH_LOOKBACK_DAYS = 420
_VIX_PERCENTILE_LOOKBACK_DAYS = 365
# Enough trailing price history to define the 200-DMA technical signal and the
# ~6-month momentum window with room for weekends/holidays.
_COMPOSITE_PRICE_LOOKBACK_DAYS = 420
# Trailing window the geometric chart-pattern detectors scan (Section 7.1). The
# same ~14 months as the composite read: long enough for a head-and-shoulders or
# a cup-and-handle to complete, short enough that the Stock Detail panel (which
# shows the last 120 days) is never scanning years of irrelevant history.
_PATTERN_LOOKBACK_DAYS = 420
# Which investor profiles get their own stored `composite_scores` rows nightly.
#
# Most profiles differ from `balanced` only in category WEIGHTS, and the stored
# sub-scores are weight-independent by design (Section 7.5) -- so the Screener
# re-weights to those client-side without a re-score (Section 8), and storing
# them would be storing the same seven numbers again.
#
# `income` and `conservative` are the exceptions, because Section 23 gives them
# a non-weight tilt that changes how a category is *scored*: income ranks
# fundamentals against a dividend-leaning sector config, and conservative scores
# the momentum category toward low volatility rather than high return. Neither
# can be recovered by re-weighting a finished sub-score -- an average of
# within-sector percentile ranks cannot be re-tilted afterwards, and negative
# volatility is not a monotone function of risk-adjusted return. Until these
# were stored, picking either profile in the Screener silently applied only the
# weight half of what it advertised.
_COMPOSITE_PROFILES = ("balanced", "income", "conservative")

# Section 6.3 calls for daily news; running three local ML models over the
# whole universe every night is the single heaviest workload here and needs the
# concurrency/model-cache work (Sections 6.10-6.11) to be daily-affordable on a
# free runner. Until then the news-intelligence and slow smart-money signals
# (short interest, insider filings, quarterly 13F) ride the weekly cadence --
# a documented, single-constant deviation, not a silent gap.
_NEWS_REFRESH_ON_WEEKLY_ONLY = True

# Two different ceilings, because the three news models do NOT cost the same.
# Measured on real headlines (title + summary), warm, on developer hardware:
#
#     spaCy NER          19 ms/article
#     FinBERT sentiment  14 ms/article
#     BART zero-shot    347 ms/article   <- 91% of the total
#
# The asymmetry is structural: eight candidate labels means eight entailment
# passes through a 400M-parameter model per article. So capping "articles" as
# one number would either starve sentiment (which is what actually feeds the
# composite score, and is nearly free) or blow the budget on event typing
# (which only sets a decay half-life and a display label).
#
# `_MAX_SENTIMENT_ARTICLES` bounds the cheap pass over everything; the
# per-symbol cap in `news_client` is what keeps coverage even across the
# universe rather than letting a few heavily-covered mega-caps crowd out the
# rest. `_MAX_CLASSIFIED_ARTICLES` bounds the expensive one; beyond it articles
# fall to `EventType.OTHER`, the same state a low-confidence classification
# already produces.
#
# Sizing, against a ~40 min target on a CI runner assumed ~3x slower than the
# numbers above: ~7,000 articles x 33 ms = ~4 min of NER+sentiment, plus 1,500
# x 347 ms = ~9 min of classification, so ~13 min locally and ~40 on the runner
# -- comfortably inside the 90-minute step budget below.
_MAX_SENTIMENT_ARTICLES = 7_000
_MAX_CLASSIFIED_ARTICLES = 1_500
# Time held back from the classifier's deadline so the rows it *did* produce
# still get written. A step that spends its whole budget classifying and is then
# killed before persisting has done the expensive part for nothing.
_NEWS_PERSIST_MARGIN_SECONDS = 120

# Per-step wall-clock budgets. `step()` enforces these so a stalled dependency
# costs one step instead of the entire run.
#
# This exists because of a real outage, not as defensive decoration: the weekly
# branch classified every article it could fetch (~75,000) in a single
# unbounded batch, produced no log output for 5h38m, and was cancelled at
# GitHub's 6-hour job limit -- which meant the "commit refreshed database" step
# never ran, so a night that had *already successfully fetched* every price and
# fundamental committed nothing at all. Capping the batch fixes the cause; this
# makes any future stall survivable rather than total.
_DEFAULT_STEP_TIMEOUT_SECONDS = 45 * 60
_STEP_TIMEOUT_SECONDS: dict[str, int] = {
    # The model-bound steps. Generous enough for a slow runner, far short of
    # the job limit.
    "tier1_news": 90 * 60,
    "tier2_news": 30 * 60,
    # Measured at ~42 min for 503 names on real history; the ceiling is a
    # backstop against a pathological series, not a target.
    "forecasts": 120 * 60,
}

# Phase 7 forecasting + backtest (Section 7.6). Both are among the heaviest steps
# in the job -- generating four horizons x three models per name, and a
# multi-year walk-forward -- so, like the news/13F workloads above, they ride the
# weekly cadence (a documented cost choice, not a silent gap; Sections 6.10-6.13
# on staged rollout and the model-cache work that would make this daily-affordable).
_FORECAST_HORIZONS = forecasting.DEFAULT_HORIZONS  # (5, 20, 63, 252) trading days
# runner name -> (model callable, the `model_name` its Forecast carries), so a
# forecast row's historical_hit_rate can be keyed to the pooled accuracy below.
_FORECAST_RUNNERS: dict[str, tuple[Any, str]] = {
    "baseline": (forecasting.baseline_forecast, "baseline"),
    "arima": (forecasting.statistical_forecast, "arima"),
    "ml": (forecasting.ml_forecast, "gbr"),
}
# Enough trailing history to fit the longest horizon's ML training window plus
# its forward-return target, with room for weekends/holidays (~3.5 years).
_FORECAST_PRICE_LOOKBACK_DAYS = 1280
# The model's own out-of-sample hit-rate (shown alongside every forecast,
# Section 7.6) is pooled over this many names -- a bounded sample keeps the
# walk-forward affordable while still being an honest per-model/horizon accuracy.
_ACCURACY_SAMPLE_SIZE = 20

# Strategy backtest ("followed the algorithm's ratings", Section 7.6): a
# survivorship-aware, monthly-rebalanced, cost-charged track record over a
# multi-year window. The wired signal is trailing price momentum -- a cheap,
# point-in-time stand-in for the composite rating (the engine is signal-agnostic,
# so the composite score can drive it once its stored history is deep enough).
_BACKTEST_LOOKBACK_DAYS = 1825  # ~5 years
_BACKTEST_CADENCE = "monthly"
_BACKTEST_TOP_FRACTION = 0.2
_BACKTEST_TXN_COST = 0.001  # 0.1% per unit turnover (Section 7.6's bid-ask stand-in)
_BACKTEST_MOMENTUM_LOOKBACK = 120  # ~6-month trailing return as the ranking signal

# The insider/13F table columns populated from their ingestion DataFrames.
_INSIDER_COLUMNS = (
    "symbol",
    "insider_name",
    "insider_title",
    "filing_date",
    "transaction_date",
    "transaction_code",
    "acquired_disposed_code",
    "shares",
    "price_per_share",
    "shares_owned_after",
)
_INSTITUTIONAL_COLUMNS = (
    "symbol",
    "quarter_end_date",
    "total_shares_held",
    "total_value",
    "num_filers",
    "change_from_prior_quarter",
)


# Steps the app is genuinely unusable without. A failure in one of these makes
# the whole run "failed" (red in CI), rather than the "partial" that every
# optional source failure produces.
#
# The distinction exists because it was missing: a run reported "partial" and
# exited 0 while `composite_scores` -- every rating in the Screener, on Home and
# on every Stock Detail page -- was completely empty. Prices and options had
# landed, so "partial" was literally true and entirely misleading. News, 13F and
# macro genuinely are optional: the ratings still compute without them, just
# with lower `data_confidence`, which the UI already shows.
_CRITICAL_STEPS = frozenset({"composite_scores", "market_regime"})

_REIT_SECTOR = "Real Estate"


class StepTimeout(Exception):
    """A refresh step exceeded its wall-clock budget."""


@contextmanager
def _step_timeout(seconds: int, name: str) -> "Iterator[None]":
    """Raise `StepTimeout` in the main thread if the body outlasts `seconds`.

    Uses `SIGALRM`, which is what makes this useful rather than decorative: the
    failure it exists for is a *stall*, not an exception, and a stalled step
    cannot be caught by `step()`'s `except`. The signal interrupts the running
    step and turns the stall into an ordinary step failure the surrounding
    handler already knows how to degrade.

    Two deliberate limits, stated because a timeout that silently does nothing
    is worse than none at all:

    * `SIGALRM` only exists on Unix and can only be armed from the main thread.
      Both hold for this job (it runs as `__main__` on Linux runners and macOS
      dev machines); anywhere else this degrades to a no-op rather than
      refusing to run, since the job is still correct without it.
    * The handler runs between bytecodes, so a single uninterruptible C call
      cannot be cut short. In practice the model steps loop in Python between
      batches, which is where the alarm lands.

    Nested use would clobber the outer alarm, so this is used at exactly one
    level -- inside `step()`, which is never re-entered.
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _on_alarm(signum: int, frame: "FrameType | None") -> None:
        raise StepTimeout(f"step {name} exceeded its {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class TickerFetchResult:
    symbol: str
    price_df: pd.DataFrame | None = None
    fundamentals: dict[str, Any] | None = None
    analyst_consensus: dict[str, Any] | None = None
    ffo_inputs: dict[str, Any] | None = None
    options_signals: dict[str, Any] | None = None
    short_interest: dict[str, Any] | None = None
    insider_df: pd.DataFrame | None = None
    tier1_news_df: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)


# yfinance's `period` argument only accepts named windows, so an incremental
# pull picks the smallest named window that comfortably covers the gap since
# the last stored bar (then filters to strictly-newer rows). A fixed "5d" would
# silently under-fetch after any outage longer than a few trading days
# (Section 6.7's "only fetch bars since the last stored date").
def _incremental_period(last_price_date: date | None, *, today: date) -> str:
    if last_price_date is None:
        return "1mo"
    gap_days = (today - last_price_date).days
    for threshold, period in ((5, "5d"), (25, "1mo"), (85, "3mo"), (330, "1y")):
        if gap_days <= threshold:
            return period
    return "2y"


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


def sync_catalogue(session: Session) -> int:
    """Record every US-listed symbol as searchable, without scoring any of them.

    The nightly job can afford to fetch and score a few hundred names; the US
    market lists about 13,000 securities. Rather than pretend the rest do not
    exist, they go in as `coverage="catalogue"` -- searchable, analysable on
    demand, and invisible to everything else.

    Two invariants make that safe, and both are asserted in the tests:

    * **A ranked symbol is never demoted.** This only ever inserts rows that are
      absent. If the S&P 500 sync and the listing directory disagree about a
      symbol -- and they will, on share classes and recent changes -- the ranked
      row wins, because it is the one with three years of prices behind it.
    * **Catalogue rows are `is_active=False`.** Every existing reader, the whole
      nightly pipeline and the index-membership reconstruction all filter on
      that column, so 12,500 new rows cannot leak into a fetch loop, a ranking,
      or a survivorship-aware backtest.

    Returns the number of newly catalogued symbols, which is large on the first
    run and near zero afterwards.
    """
    listings = listing_client.fetch_us_listings()
    known = set(session.scalars(select(Ticker.symbol)))

    added = 0
    for row in listings.itertuples(index=False):
        if row.symbol in known:
            continue
        session.add(
            Ticker(
                symbol=row.symbol,
                name=row.name,
                sector=None,
                industry=None,
                exchange=row.exchange,
                asset_type=row.asset_type,
                is_active=False,
                coverage=Ticker.CATALOGUE,
            )
        )
        added += 1

    session.flush()
    logger.info(
        "Catalogue: %d listed symbols, %d newly recorded (%d already known).",
        len(listings),
        added,
        len(listings) - added,
    )
    return added


def reconcile_index_membership(session: Session, today: date) -> int:
    """Keep `index_membership_history` current between cold-start re-seeds (Section 6.9).

    The cold-start script seeds authoritative *historical* intervals; this closes
    an interval the day a name drops out of the current index and opens one the
    day a name joins, using `today` as the boundary the app learned of the change.
    Without it, a name removed since the last seed would keep a null `removed_date`
    and quietly re-enter every survivorship-aware backtest as if still a member
    (Sections 6.9, 22) -- the incremental other half of the survivorship-bias
    handling the seed script owns for deep history.

    Must run right after `sync_universe`, which sets the `is_active` flags this
    reads. Idempotent: on a steady index it makes no changes. A re-seed later
    legitimately replaces everything (membership is authoritative reference data).
    """
    current = set(
        session.scalars(
            select(Ticker.symbol).where(Ticker.is_active, Ticker.asset_type == "equity")
        )
    )
    open_intervals = {
        row.symbol: row
        for row in session.scalars(
            select(IndexMembershipHistory).where(
                IndexMembershipHistory.index_name == hist.INDEX_NAME,
                IndexMembershipHistory.removed_date.is_(None),
            )
        )
    }

    changes = 0
    for symbol in current - set(open_intervals):  # joined the index -> open an interval
        session.add(
            IndexMembershipHistory(
                index_name=hist.INDEX_NAME, symbol=symbol, added_date=today, removed_date=None
            )
        )
        changes += 1
    for symbol, row in open_intervals.items():  # left the index -> close its interval
        if symbol not in current:
            row.removed_date = today
            changes += 1

    session.flush()
    return changes


def _last_price_date(session: Session, symbol: str) -> date | None:
    stmt = (
        select(PriceHistory.date)
        .where(PriceHistory.symbol == symbol)
        .order_by(PriceHistory.date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def fetch_ticker_data(
    symbol: str,
    last_price_date: date | None,
    is_weekly: bool,
    *,
    company_name: str | None = None,
    sector: str | None = None,
    today: date | None = None,
) -> TickerFetchResult:
    """Pure I/O: call external APIs for one ticker. No DB access -- safe to run concurrently.

    Daily: price history (incrementally) and the options-positioning snapshot.
    Weekly (and news only when `_NEWS_REFRESH_ON_WEEKLY_ONLY`): fundamentals,
    analyst consensus, short interest, insider filings, and Tier-1 news. Every
    fetch is isolated in its own try/except so one failing source degrades that
    one field to `None` rather than dropping the whole ticker.
    """
    result = TickerFetchResult(symbol=symbol)
    today = today or date.today()

    try:
        period = _incremental_period(last_price_date, today=today)
        df = yfinance_client.fetch_price_history(symbol, period=period)
        if last_price_date is not None:
            df = df[df["date"] > pd.Timestamp(last_price_date)]
        result.price_df = df
    except Exception as exc:
        result.errors.append(f"price_history: {exc}")

    try:
        result.options_signals = options_client.fetch_options_signals(symbol)
    except Exception as exc:
        result.errors.append(f"options_signals: {exc}")

    if is_weekly:
        try:
            result.fundamentals = yfinance_client.fetch_fundamentals(symbol)
        except Exception as exc:
            result.errors.append(f"fundamentals: {exc}")
        try:
            result.analyst_consensus = yfinance_client.fetch_analyst_consensus(symbol)
        except Exception as exc:
            result.errors.append(f"analyst_consensus: {exc}")
        if sector == _REIT_SECTOR:
            # REITs are valued on P/FFO, not P/E (Section 7.2) -- fetch the
            # extra inputs only for the sector that actually uses them.
            try:
                result.ffo_inputs = yfinance_client.fetch_ffo_inputs(symbol)
            except Exception as exc:
                result.errors.append(f"ffo_inputs: {exc}")
        try:
            result.short_interest = short_interest_client.fetch_short_interest(symbol)
        except Exception as exc:
            result.errors.append(f"short_interest: {exc}")
        try:
            result.insider_df = edgar_client.fetch_insider_transactions(symbol)
        except Exception as exc:
            result.errors.append(f"insider_transactions: {exc}")
        if not _NEWS_REFRESH_ON_WEEKLY_ONLY or is_weekly:
            try:
                result.tier1_news_df = news_client.fetch_all_tier1_news(symbol, company_name)
            except Exception as exc:
                result.errors.append(f"tier1_news: {exc}")

    return result


# Every column `PriceHistory` declares NOT NULL. A bar missing any of them is
# not a bar -- yfinance emits these for halted names, the not-yet-settled
# current session, and tickers that changed listing status mid-fetch.
_PRICE_REQUIRED_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


def _upsert_price_history(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    # Drop incomplete bars *before* the insert. Passing a NaN into a NOT NULL
    # column raises IntegrityError, and because this loop is the one write stage
    # that isn't behind `step()`, that exception used to abort the entire
    # nightly run: every later ticker's prices, fundamentals and smart-money
    # rows were rolled back with it. That is exactly how the deployed demo went
    # four days without a price update while GitHub Actions still reported
    # success (see run 30878926050). A missing bar should cost that one bar.
    present = [c for c in _PRICE_REQUIRED_COLUMNS if c in df.columns]
    clean = df.dropna(subset=present) if present else df
    if len(clean) < len(df):
        dropped = len(df) - len(clean)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else "?"
        logger.warning("%s: dropped %d incomplete price bar(s)", symbol, dropped)
    if clean.empty:
        return 0
    records = clean.to_dict("records")
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


def _fundamentals_with_ffo(
    fundamentals: dict[str, Any], ffo_inputs: dict[str, Any] | None
) -> dict[str, Any]:
    """Attach a computed P/FFO to a REIT's fundamentals, into `sector_specific_metrics`.

    Wires `fundamental.compute_p_ffo` (previously computed nowhere) into the
    stored snapshot so Phase 6's Real Estate scoring, which weights `p_ffo`,
    has real data to read out of the `sector_specific_metrics` JSON column
    (Section 7.2, Section 13). A missing/undefined P/FFO is simply omitted.
    """
    if not ffo_inputs:
        return fundamentals
    p_ffo = fundamental.compute_p_ffo(
        ffo_inputs.get("market_cap"),
        ffo_inputs.get("net_income"),
        ffo_inputs.get("depreciation_amortization"),
    )
    if p_ffo is None:
        return fundamentals
    return {**fundamentals, "sector_specific_metrics": {"p_ffo": p_ffo}}


def _in_session(fn: "Any") -> int:
    """Run `fn(session)` in a fresh committed session and return its row count.

    The wrapper the standalone (own-session) refresh steps share, so `run`'s
    step list stays a flat sequence of `_in_session(...)` calls.
    """
    with get_session() as session:
        return int(fn(session))


def _persist_tier1_news(
    results: list[TickerFetchResult],
    universe: pd.DataFrame,
    today: date,
    *,
    deadline: float | None = None,
) -> int:
    """Run the (session-free) Tier-1 news models, then persist the results in one session.

    The heavy model pass (`process_tier1_news`) deliberately holds no DB
    session while it runs, so SQLite's single writer isn't locked for the
    duration of hundreds of classifications.
    """
    sentiment_records, news_records = process_tier1_news(
        results, universe, today, deadline=deadline
    )
    if not sentiment_records and not news_records:
        return 0
    with get_session() as session:
        return persistence.upsert_sentiment_scores(
            session, sentiment_records
        ) + persistence.upsert_news_events(session, news_records)


def _records_from_df(df: pd.DataFrame | None, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Table-ready records from `df`, keeping only `columns` and mapping NaN/NaT -> None.

    The Phase 4/5 ingestion clients return DataFrames with pandas missing
    sentinels (`NaN`/`NaT`); SQLite wants real `None`, so this normalizes them
    on the way into the persistence helpers.
    """
    if df is None or df.empty:
        return []
    present = [c for c in columns if c in df.columns]
    subset = df[present].astype(object).where(pd.notna(df[present]), None)
    return list(subset.to_dict("records"))


def _active_universe(session: Session) -> pd.DataFrame:
    """`(symbol, name, sector)` for active equities -- the shape the gazetteer/13F/regime want."""
    rows = session.execute(
        select(Ticker.symbol, Ticker.name, Ticker.sector).where(
            Ticker.is_active, Ticker.asset_type == "equity"
        )
    ).all()
    return pd.DataFrame(rows, columns=["symbol", "name", "sector"])


def refresh_cross_asset_macro(session: Session, today: date) -> int:
    """Ingest VIX + commodity/currency closes into `macro_indicators` (Section 28).

    Daily: the Market Regime Index (VIX percentile + level) and the sector
    commodity/currency overlay both read these back out of `macro_indicators`.
    """
    records: list[dict[str, Any]] = []
    for series_name, ticker in _CROSS_ASSET_TICKERS.items():
        try:
            df = yfinance_client.fetch_price_history(ticker, period="5d")
        except Exception:
            logger.exception("Failed to fetch cross-asset series %s (%s)", series_name, ticker)
            continue
        if df.empty:
            continue
        latest = df.sort_values("date").iloc[-1]
        records.append(
            {
                "date": pd.Timestamp(latest["date"]).date(),
                "indicator_name": series_name,
                "value": float(latest["close"]),
            }
        )
    if not records:
        return 0
    stmt = sqlite_insert(MacroIndicator).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "indicator_name"], set_={"value": stmt.excluded["value"]}
    )
    session.execute(stmt)
    return len(records)


def refresh_static_config(session: Session, today: date) -> int:
    """Refresh the config-derived tables (thematic baskets + economic calendar)."""
    basket_records = [
        {"theme_name": theme, "symbol": symbol}
        for theme, symbol in thematic_mapping.iter_basket_membership()
    ]
    rows = persistence.replace_thematic_baskets(session, basket_records)

    events = economic_calendar.upcoming_events(today, lookahead_days=120)
    calendar_records = [
        {"event_date": event.event_date, "event_name": event.event_name} for event in events
    ]
    rows += persistence.upsert_economic_calendar(session, calendar_records)
    return rows


def refresh_institutional_ownership(session: Session, universe: pd.DataFrame, today: date) -> int:
    """Ingest the current quarter's 13F institutional-ownership trend (Section 24).

    A ~100MB quarterly bulk download, cached indefinitely once fetched -- so a
    weekly re-run only re-does real work when a new quarter's file appears.
    """
    window = edgar_13f_client.quarter_window_for(today)
    try:
        trend = edgar_13f_client.fetch_institutional_ownership_trend(window, universe)
    except Exception:
        logger.exception("Failed to fetch 13F institutional ownership for window %s", window)
        return 0
    records = _records_from_df(trend, _INSTITUTIONAL_COLUMNS)
    return persistence.upsert_institutional_ownership(session, records)


def _macro_news_tone(today: date) -> float | None:
    """Latest GDELT macro-tone reading for the Market Regime Index's Tier-3 input."""
    try:
        tone_df = gdelt_client.fetch_tone_timeline(_MACRO_TONE_QUERY, timespan="14d")
    except Exception:
        logger.exception("Failed to fetch GDELT macro tone")
        return None
    if tone_df.empty:
        return None
    latest = tone_df.sort_values("date").iloc[-1]
    return float(latest["tone"]) if pd.notna(latest["tone"]) else None


def refresh_market_regime(session: Session, today: date) -> int:
    """Compute and persist today's Market Regime Index (Sections 5, 7.3 Tier 3, 28).

    Reads its four inputs back out of already-refreshed tables (VIX + yield
    curve from `macro_indicators`, breadth from `price_history`) plus a live
    GDELT macro-tone pull, so it must run after `refresh_cross_asset_macro`.
    """
    vix_level = persistence.read_latest_macro_value(session, macro.VIX, as_of=today)
    vix_history = persistence.read_macro_series(
        session, macro.VIX, as_of=today, lookback_days=_VIX_PERCENTILE_LOOKBACK_DAYS
    )
    dgs10 = persistence.read_latest_macro_value(
        session, fred_client.TREASURY_YIELD_10Y, as_of=today
    )
    dgs2 = persistence.read_latest_macro_value(session, fred_client.TREASURY_YIELD_2Y, as_of=today)
    spread = macro.yield_curve_spread(dgs10, dgs2)

    price_history = persistence.read_active_price_history(
        session, as_of=today, lookback_days=_BREADTH_LOOKBACK_DAYS
    )
    breadth = market_regime.compute_breadth(price_history, today)

    reading = market_regime.compute_market_regime(
        today,
        vix_level=vix_level,
        vix_history=vix_history,
        breadth_pct=breadth,
        macro_tone=_macro_news_tone(today),
        yield_curve_spread_value=spread,
    )
    return persistence.upsert_market_regime(session, market_regime.regime_to_record(reading))


def _none_if_nan(value: Any) -> float | None:
    """`None` for a pandas/NumPy missing value, else the value as a plain float."""
    return None if value is None or pd.isna(value) else float(value)


def _rows_by_symbol(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """`{symbol: row-dict}` for a per-symbol frame (empty dict for an empty frame)."""
    if df.empty:
        return {}
    return {str(record["symbol"]): record for record in df.to_dict("records")}


def refresh_composite_scores(session: Session, universe: pd.DataFrame, today: date) -> int:
    """Score every symbol across the seven categories and persist the ranking (Section 7.5).

    Reads each category's inputs point-in-time (only rows dated <= `today`),
    derives the seven raw sub-scores per symbol, then normalizes/weights/rates
    them into `composite_scores`. Runs last in the nightly job so it sees the
    freshly-written prices, sentiment, smart-money, and market-regime rows.
    Missing categories simply lower a symbol's `data_confidence` rather than
    dropping it -- only a symbol with *no* usable category is left unranked.

    Writes one ranking per profile in `_COMPOSITE_PROFILES`. The expensive parts
    -- indicators, smart money, the news joins -- are computed once and shared;
    only the two categories a profile genuinely re-scores (fundamentals under
    the income tilt, momentum under the low-volatility tilt) are recomputed.
    """
    fundamentals = persistence.read_latest_fundamentals(session, as_of=today)
    fundamental_cache: dict[bool, dict[str, float]] = {}

    def _fundamental_scores(*, income_tilt: bool) -> dict[str, float]:
        """Sector-relative fundamental scores, optionally dividend-tilted (Section 23).

        Memoized on the tilt: two of the three stored profiles share the
        untilted scoring, and ranking every sector twice for an identical
        answer is pure waste in the job's last and heaviest step.
        """
        if income_tilt not in fundamental_cache:
            if fundamentals.empty:
                fundamental_cache[income_tilt] = {}
            else:
                scored = fundamental.score_fundamentals(
                    fundamentals, income_tilt=income_tilt
                ).set_index("symbol")
                fundamental_cache[income_tilt] = dict(scored["fundamental_score"].to_dict())
        return fundamental_cache[income_tilt]

    ohlcv = persistence.read_active_ohlcv(
        session, as_of=today, lookback_days=_COMPOSITE_PRICE_LOOKBACK_DAYS
    )
    # A DatetimeIndex (not the DB's date-object column) is required by the
    # indicator library's time-anchored calculations (e.g. VWAP).
    ohlcv_by_symbol = {
        str(symbol): group[["open", "high", "low", "close", "volume"]].set_axis(
            pd.DatetimeIndex(pd.to_datetime(group["date"].to_numpy()))
        )
        for symbol, group in ohlcv.groupby("symbol")
    }
    latest_close = {
        symbol: float(prices["close"].iloc[-1])
        for symbol, prices in ohlcv_by_symbol.items()
        if not prices["close"].dropna().empty
    }

    analyst_history = persistence.read_analyst_history(session, as_of=today)
    analyst_by_symbol = (
        {str(symbol): group for symbol, group in analyst_history.groupby("symbol")}
        if not analyst_history.empty
        else {}
    )
    sentiment_by_symbol = persistence.read_latest_sentiment(session, as_of=today)
    tier2_news = persistence.read_tier2_news(session, as_of=today)
    theme_members = persistence.read_theme_members(session)
    regime_score = persistence.read_latest_regime_score(session, as_of=today)

    insider = persistence.read_recent_insider_transactions(session, as_of=today)
    institutional_by_symbol = _rows_by_symbol(
        persistence.read_latest_institutional(session, as_of=today)
    )
    options_by_symbol = _rows_by_symbol(persistence.read_latest_options(session, as_of=today))
    short_by_symbol = _rows_by_symbol(persistence.read_latest_short_interest(session, as_of=today))

    raw_by_symbol: dict[str, dict[str, float | None]] = {}
    for symbol in universe["symbol"]:
        prices = ohlcv_by_symbol.get(symbol)
        analyst_frame = analyst_by_symbol.get(symbol)
        analyst_raw = (
            analyst_consensus.score_analyst_consensus(analyst_frame, latest_close.get(symbol))[
                "analyst_score"
            ]
            if analyst_frame is not None
            else None
        )
        options_row = options_by_symbol.get(symbol) or {}
        smart = smart_money.compute_smart_money_score(
            symbol,
            insider_transactions=(
                insider[insider["symbol"] == symbol] if not insider.empty else insider
            ),
            institutional_trend_row=institutional_by_symbol.get(symbol),
            options_signals=options_row,
            short_interest=short_by_symbol.get(symbol) or {},
            iv_rank=options_row.get("iv_rank"),
        )
        raw_by_symbol[symbol] = {
            "technical": scoring.score_technical(prices) if prices is not None else None,
            "momentum": scoring.score_momentum(prices) if prices is not None else None,
            # The conservative profile scores this category toward LOW volatility
            # instead of raw momentum (Section 23), which is a different reading
            # of the same prices, not a re-weighting -- so it has to be computed
            # here rather than derived from the value above.
            "momentum_low_vol": (
                scoring.score_momentum(prices, prefer_low_volatility=True)
                if prices is not None
                else None
            ),
            "analyst": analyst_raw,
            "sentiment": scoring.sentiment_to_raw(sentiment_by_symbol.get(symbol)),
            "industry_macro": scoring.tier2_thematic_tilt(symbol, tier2_news, theme_members),
            "smart_money": smart.score,
        }

    if not raw_by_symbol:
        return 0
    shared = pd.DataFrame.from_dict(raw_by_symbol, orient="index")

    written = 0
    for profile_name in _COMPOSITE_PROFILES:
        profile = get_profile(profile_name)
        category_raw = shared.drop(columns=["momentum_low_vol"]).copy()
        if profile.prefer_low_volatility:
            category_raw["momentum"] = shared["momentum_low_vol"]
        category_raw["fundamental"] = category_raw.index.map(
            _fundamental_scores(income_tilt=profile.income_tilt)
        )
        written += _store_composite(session, category_raw, profile_name, today, regime_score)
    return written


def _store_composite(
    session: Session,
    category_raw: pd.DataFrame,
    profile_name: str,
    today: date,
    regime_score: float | None,
) -> int:
    """Score `category_raw` under one profile and append the `composite_scores` rows."""
    result = scoring.build_composite(category_raw, profile=profile_name, regime_score=regime_score)
    if result.scores.empty:
        return 0

    records = [
        {
            "symbol": row.symbol,
            "date": today,
            "profile": result.profile,
            "fundamental_score": _none_if_nan(row.fundamental_score),
            "technical_score": _none_if_nan(row.technical_score),
            "analyst_score": _none_if_nan(row.analyst_score),
            "sentiment_score": _none_if_nan(row.sentiment_score),
            "momentum_score": _none_if_nan(row.momentum_score),
            "industry_macro_score": _none_if_nan(row.industry_macro_score),
            "smart_money_score": _none_if_nan(row.smart_money_score),
            # The pre-normalization inputs, so a stored row can be re-scored in
            # absolute mode later -- a percentile cannot be un-ranked.
            "fundamental_raw": _none_if_nan(row.fundamental_raw),
            "technical_raw": _none_if_nan(row.technical_raw),
            "analyst_raw": _none_if_nan(row.analyst_raw),
            "sentiment_raw": _none_if_nan(row.sentiment_raw),
            "momentum_raw": _none_if_nan(row.momentum_raw),
            "industry_macro_raw": _none_if_nan(row.industry_macro_raw),
            "smart_money_raw": _none_if_nan(row.smart_money_raw),
            "composite_score": float(row.composite_score),
            "percentile_rank": _none_if_nan(row.percentile_rank),
            "rating": row.rating,
            "data_confidence": float(row.data_confidence),
        }
        for row in result.scores.itertuples(index=False)
    ]
    return persistence.upsert_composite_scores(session, records)


def refresh_pattern_signals(session: Session, universe: pd.DataFrame, today: date) -> int:
    """Detect and persist geometric chart patterns for every active name (Section 7.1).

    `analysis/patterns.py` -- head-and-shoulders, double top/bottom, triangles,
    wedges, channels and cup-and-handle, each with a confidence score rather
    than a yes/no verdict -- had no producer at all. `pattern_signals` had a
    table, a migration, a reader and a "Detected patterns" panel in *both* front
    ends; the one missing link was anything that ever wrote a row, so the panel
    was permanently empty and the README's "4 pattern families detected" was a
    claim about a library rather than about the app.

    Runs daily rather than weekly: measured over the real 503-name universe on
    ~288 bars each, the whole sweep takes about one second, so there is no cost
    argument for staleness. Storage is deduped on the formation's own
    (symbol, date, pattern_type), so re-detecting the same double top tomorrow
    recognises it instead of logging it again.

    Candlestick patterns are deliberately NOT stored alongside these. The same
    sweep yields roughly two per bar -- 579 rows for one name over 400 days,
    dominated by "longline"/"belthold"/"shortline" at a flat confidence of 100 --
    which would bury the handful of geometric formations a reader can actually
    act on. `technical.detect_candlestick_patterns` stays available for a caller
    that wants them.
    """
    ohlcv = persistence.read_active_ohlcv(
        session, as_of=today, lookback_days=_PATTERN_LOOKBACK_DAYS
    )
    if ohlcv.empty:
        return 0
    frames = _ohlcv_frames_by_symbol(ohlcv)

    records: list[dict[str, Any]] = []
    for symbol in universe["symbol"]:
        prices = frames.get(symbol)
        if prices is None or prices.empty:
            continue
        try:
            found = patterns.detect_chart_patterns(prices, symbol=symbol)
        except ValueError:
            # A malformed frame for one ticker must not cost the other 502.
            logger.exception("Pattern detection failed for %s; skipping it", symbol)
            continue
        records.extend(
            {
                "symbol": row.symbol,
                "date": pd.Timestamp(row.date).date(),
                "pattern_type": row.pattern_type,
                "direction": row.direction,
                "confidence": float(row.confidence),
            }
            for row in found.itertuples(index=False)
        )
    return persistence.upsert_pattern_signals(session, records)


def _ohlcv_frames_by_symbol(ohlcv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """`{symbol: OHLCV frame on a DatetimeIndex}` from the long point-in-time read.

    The DatetimeIndex (not the DB's date-object column) is what the forecasting
    feature engineering and the indicator library both expect.
    """
    return {
        str(symbol): group[["open", "high", "low", "close", "volume"]].set_axis(
            pd.DatetimeIndex(pd.to_datetime(group["date"].to_numpy()))
        )
        for symbol, group in ohlcv.groupby("symbol")
    }


def _pooled_hit_rates(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, int], tuple[float, int]]:
    """Each (model, horizon)'s out-of-sample hit-rate **and its window count**, pooled.

    Runs the look-ahead-free walk-forward (`backtest.walk_forward_accuracy`) for
    every runner and horizon over the `_ACCURACY_SAMPLE_SIZE` longest-history
    names, pools the per-fold predicted/realized pairs across those names, and
    scores one honest hit-rate per (model_name, horizon). This is the stat shown
    next to every individual forecast (Section 7.6) -- computed on a bounded
    sample so it stays affordable, since the point estimate is a property of the
    model, not of any one stock.

    **The returned count is distinct evaluation windows, not pooled pairs**, and
    a rate measured over fewer than `backtest.MIN_GRADED_WINDOWS` of them is
    dropped rather than published. Pooling twenty symbols multiplies the pair
    count twentyfold without adding a single new window: the sampled names all
    share one trading calendar, so they are twenty readings of the *same*
    history. Measured on real data, that inflated the 1-year horizon's apparent
    sample from 1-3 windows to 20-60 pairs, and the ML model's "60% hit rate vs
    52% naive" there was twenty correlated readings of a single year.
    """
    sample = sorted(frames.values(), key=len, reverse=True)[:_ACCURACY_SAMPLE_SIZE]
    pooled: dict[tuple[str, int], tuple[list[float], list[float], set[pd.Timestamp]]] = {}
    for prices in sample:
        for model_fn, model_name in _FORECAST_RUNNERS.values():
            for horizon in _FORECAST_HORIZONS:
                result = backtest.walk_forward_accuracy(
                    prices, model_fn=model_fn, horizon_days=horizon, model_name=model_name
                )
                if result is None:
                    continue
                pred, real, windows = pooled.setdefault((model_name, horizon), ([], [], set()))
                pred.extend(result.predicted.tolist())
                real.extend(result.realized.tolist())
                windows.update(result.as_of)

    hit_rates: dict[tuple[str, int], tuple[float, int]] = {}
    for key, (pred, real, windows) in pooled.items():
        rate = backtest.directional_hit_rate(pred, real)
        if rate is None:
            continue
        if len(windows) < backtest.MIN_GRADED_WINDOWS:
            logger.info(
                "Not publishing the %s hit rate at h=%d: %d graded window(s), "
                "below the %d needed for the figure to mean anything",
                key[0],
                key[1],
                len(windows),
                backtest.MIN_GRADED_WINDOWS,
            )
            continue
        hit_rates[key] = (rate, len(windows))
    return hit_rates


def refresh_forecasts(session: Session, universe: pd.DataFrame, today: date) -> int:
    """Generate and persist each name's price forecasts, tagged with the model's track record.

    For every active equity with enough history, runs the baseline/statistical/ML
    models at every horizon (`forecasting.generate_forecasts`) and appends the
    point-in-time `forecasts` rows -- each carrying the fan-chart band and the
    pooled `historical_hit_rate` for its (model, horizon) so a forecast is never
    shown without its accuracy (Section 7.6). Append-only: a same-day re-run
    leaves the first forecast untouched (Section 6.8).
    """
    ohlcv = persistence.read_active_ohlcv(
        session, as_of=today, lookback_days=_FORECAST_PRICE_LOOKBACK_DAYS
    )
    if ohlcv.empty:
        return 0
    frames = _ohlcv_frames_by_symbol(ohlcv)
    hit_rates = _pooled_hit_rates(frames)

    records: list[dict[str, Any]] = []
    for symbol in universe["symbol"]:
        prices = frames.get(symbol)
        if prices is None:
            continue
        for fc in forecasting.generate_forecasts(prices, horizons=_FORECAST_HORIZONS):
            graded = hit_rates.get((fc.model_name, fc.horizon_days))
            # The naive null's rate over the same horizon, so the UI can show
            # "55% vs 53% naive" instead of a bare number that reads as skill.
            # `baseline` is itself one of the runners, so this is the identical
            # pooling over the identical sample -- not a second,
            # differently-computed statistic.
            baseline_graded = hit_rates.get(("baseline", fc.horizon_days))
            records.append(
                {
                    "symbol": symbol,
                    "generated_date": today,
                    "horizon_days": fc.horizon_days,
                    "model_name": fc.model_name,
                    "point_return": fc.point_return,
                    "point_price": fc.point_price,
                    "lower_price": fc.lower_price,
                    "upper_price": fc.upper_price,
                    "historical_hit_rate": graded[0] if graded else None,
                    "baseline_hit_rate": baseline_graded[0] if baseline_graded else None,
                    # How many distinct out-of-sample windows the rate above was
                    # measured over -- null when there is no rate to qualify.
                    "hit_rate_windows": graded[1] if graded else None,
                }
            )
    return persistence.upsert_forecasts(session, records)


def _coerce_date(value: Any) -> date | None:
    """A plain `date` (or None) from a DB value that may be a date/Timestamp/NaT."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):  # covers pd.Timestamp, a datetime subclass
        return value.date()
    return value


def _eligible_universe_fn(intervals: pd.DataFrame) -> Callable[[date], set[str]]:
    """Build the survivorship-aware `eligible(as_of)` callback from membership intervals.

    A name is eligible on `as_of` when it had been added and not yet removed
    (`added <= as_of < removed`, removed-null meaning still a member). Loaded once
    into memory so the per-rebalance lookup is a set comprehension, not a query.
    """
    rows = [
        (str(r.symbol), _coerce_date(r.added_date), _coerce_date(r.removed_date))
        for r in intervals.itertuples(index=False)
    ]
    valid = [(s, added, removed) for s, added, removed in rows if added is not None]

    def eligible(as_of: date) -> set[str]:
        return {
            s
            for s, added, removed in valid
            if added <= as_of and (removed is None or removed > as_of)
        }

    return eligible


def _equal_weight_benchmark(panel: pd.DataFrame) -> pd.Series:
    """An equal-weighted index level from the panel, as a buy-and-hold market proxy.

    Each name's adjusted close is rebased to its own first observation, then
    averaged across whatever names exist on each date. A dedicated S&P 500 price
    series isn't ingested, so this survivorship-aware average of the universe is
    the honest stand-in for the buy-and-hold benchmark (Section 7.6).

    Names whose first observation is not strictly positive are dropped rather
    than rebased. Dividing by a zero first price yields `inf` for that column,
    and one such column drags the whole cross-sectional mean to `inf` -- a
    single bad symbol silently destroying the benchmark every strategy number
    is compared against. `persistence.read_adj_close_panel` already filters
    these out; this is the second line of defence for any other caller.
    """
    positive = panel.where(panel > 0)
    first = positive.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
    usable = first[first > 0].index
    if len(usable) == 0:
        return pd.Series(index=panel.index, dtype=float)
    normalized = positive[usable].divide(first[usable], axis=1)
    return normalized.mean(axis=1, skipna=True)


def _momentum_signal(as_of: date, panel: pd.DataFrame) -> dict[str, float]:
    """Trailing `_BACKTEST_MOMENTUM_LOOKBACK`-day return per name -- the strategy's ranking.

    Reads only `panel` (already sliced to `<= as_of` by the engine), so it is
    point-in-time by construction. Names without enough history are simply
    omitted from the ranking.
    """
    if len(panel) <= _BACKTEST_MOMENTUM_LOOKBACK:
        return {}
    recent = panel.iloc[-1]
    past = panel.iloc[-1 - _BACKTEST_MOMENTUM_LOOKBACK]
    signal: dict[str, float] = {}
    for symbol in panel.columns:
        now, then = recent.get(symbol), past.get(symbol)
        if pd.notna(now) and pd.notna(then) and then > 0:
            signal[str(symbol)] = float(now / then - 1.0)
    return signal


def refresh_backtest(session: Session, today: date) -> int:
    """Run and persist the survivorship-aware, cost-aware strategy track record (Section 7.6).

    Reconstructs the point-in-time universe from `index_membership_history`, reads
    the adjusted-close panel over the trailing window (including names since
    removed -- the whole point of a survivorship-honest run), and backtests the
    momentum-ranked, monthly-rebalanced, transaction-cost-charged strategy against
    the equal-weight universe benchmark. Appends one `backtest_results` row; a
    thin/empty universe degrades to 0 written rather than raising.
    """
    intervals = persistence.read_membership_intervals(session, hist.INDEX_NAME)
    if intervals.empty:
        logger.info("refresh_backtest: no index_membership_history rows; skipping")
        return 0
    start = today - timedelta(days=_BACKTEST_LOOKBACK_DAYS)
    panel = persistence.read_adj_close_panel(
        session, start=start, end=today, symbols=list(intervals["symbol"].unique())
    )
    schedule = backtest.rebalance_dates(panel.index, _BACKTEST_CADENCE) if not panel.empty else []
    if len(schedule) < 2:
        logger.info("refresh_backtest: not enough price history for a backtest; skipping")
        return 0

    result = backtest.backtest_strategy(
        panel,
        signal_fn=_momentum_signal,
        cadence=_BACKTEST_CADENCE,
        top_fraction=_BACKTEST_TOP_FRACTION,
        transaction_cost=_BACKTEST_TXN_COST,
        benchmark=_equal_weight_benchmark(panel),
        eligible=_eligible_universe_fn(intervals),
        schedule=schedule,
    )
    if result is None:
        return 0
    if result.n_periods < backtest.MIN_TRACK_RECORD_PERIODS:
        # A CAGR raises a period's growth to the power of the periods per year,
        # so a couple of monthly periods over five weeks annualize into a
        # headline nobody earned -- and the interval that would have exposed
        # that is exactly what a run this short cannot produce.
        logger.info(
            "refresh_backtest: %d period(s) is below the %d needed for a track record; "
            "not storing a run whose headline cannot be bracketed",
            result.n_periods,
            backtest.MIN_TRACK_RECORD_PERIODS,
        )
        return 0
    if result.avg_turnover <= 0:
        # The strategy never took a position -- normally because the signal had
        # too little history to rank anything, so every period sat in cash.
        # Storing that as a 0% track record reads as "the strategy lost to the
        # market" when the strategy never ran.
        logger.info(
            "refresh_backtest: the signal never produced a ranking, so the strategy held "
            "cash throughout; not storing a run in which nothing was traded"
        )
        return 0
    if result.invested_fraction < backtest.MIN_INVESTED_FRACTION:
        # "Never traded" is not the only way for a run to describe nothing. A
        # run that was in cash for most of its periods -- typically because the
        # membership history behind `eligible()` is shallower than the price
        # panel, so the point-in-time universe comes back empty -- still reports
        # a respectable Sharpe, because cash periods are exact zeros and zeros
        # have no variance. Measured here: 38 of 39 periods in cash produced
        # Sharpe 0.555 and a 0.99% CAGR against a 29.3% benchmark.
        logger.warning(
            "refresh_backtest: the strategy held a position in only %d of %d periods "
            "(%.0f%%, below the %.0f%% needed). This usually means index membership "
            "history is shallower than the backtest window, so the point-in-time "
            "universe was empty; not storing a run that mostly describes cash",
            result.invested_periods,
            result.n_periods,
            result.invested_fraction * 100,
            backtest.MIN_INVESTED_FRACTION * 100,
        )
        return 0
    # Bracket the headline metrics with block-bootstrap confidence intervals so
    # the stored track record reports whether the result is distinguishable from
    # luck, not just a flattering point estimate (Section 7.6). A run too short
    # to bootstrap honestly stores nulls rather than a fabricated interval.
    significance = backtest.bootstrap_strategy_significance(result)
    sharpe_ci, cagr_ci = significance.sharpe, significance.cagr
    any_ci = sharpe_ci or cagr_ci

    return persistence.insert_backtest_result(
        session,
        {
            "run_date": today,
            "period_start": schedule[0].date(),
            "period_end": schedule[-1].date(),
            "cadence": _BACKTEST_CADENCE,
            "n_periods": result.n_periods,
            "sharpe": result.sharpe,
            "sharpe_ci_low": sharpe_ci.low if sharpe_ci else None,
            "sharpe_ci_high": sharpe_ci.high if sharpe_ci else None,
            "cagr": result.cagr,
            "cagr_ci_low": cagr_ci.low if cagr_ci else None,
            "cagr_ci_high": cagr_ci.high if cagr_ci else None,
            "ci_confidence_level": any_ci.confidence_level if any_ci else None,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "payoff_ratio": result.payoff_ratio,
            "benchmark_cagr": result.benchmark_cagr,
            "benchmark_sharpe": result.benchmark_sharpe,
            "avg_turnover": result.avg_turnover,
            "assumed_txn_cost": result.assumed_txn_cost,
        },
    )


def _persist_per_ticker_smart_money(
    session: Session, result: TickerFetchResult, today: date
) -> int:
    """Write one ticker's options / short-interest / insider rows (Section 24)."""
    rows = 0
    if result.options_signals is not None:
        atm_iv = result.options_signals.get("atm_implied_volatility")
        iv_rank = None
        if atm_iv is not None:
            history = persistence.read_recent_atm_iv(
                session, result.symbol, before=today, lookback_days=_IV_RANK_LOOKBACK_DAYS
            )
            iv_rank = options_client.compute_iv_rank(atm_iv, history)
        rows += persistence.upsert_options_signals(
            session,
            [
                {
                    "symbol": result.symbol,
                    "date": today,
                    "expiration": result.options_signals.get("expiration"),
                    "put_call_ratio": result.options_signals.get("put_call_ratio"),
                    "atm_implied_volatility": atm_iv,
                    "iv_rank": iv_rank,
                }
            ],
        )
    if result.short_interest is not None:
        rows += persistence.upsert_short_interest(
            session,
            [
                {
                    "symbol": result.symbol,
                    "as_of_date": today,
                    "pct_float_short": result.short_interest.get("pct_float_short"),
                    "days_to_cover": result.short_interest.get("days_to_cover"),
                }
            ],
        )
    if result.insider_df is not None and not result.insider_df.empty:
        rows += persistence.insert_insider_transactions(
            session, _records_from_df(result.insider_df, _INSIDER_COLUMNS)
        )
    return rows


def _cap_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Order newest-first and bound the batch to `_MAX_SENTIMENT_ARTICLES`.

    A second line of defence behind `news_client.MAX_ARTICLES_PER_SYMBOL`. The
    per-symbol cap bounds the *typical* run, but the universe size is not fixed
    and a new Tier-1 source could be added, so this pins the total.

    Sorting always (not only when over the limit) is what makes the separate
    classification cap downstream meaningful: `classify_articles` takes rows in
    frame order, so newest-first here means the expensive model spends its
    budget on the articles the recency-decay step weights most heavily.

    What was dropped is logged rather than silently discarded -- a cap nobody
    can see reads as "we processed everything", which is exactly the kind of
    quiet truncation Section 22 warns about.
    """
    ordered = articles.sort_values("published_at", ascending=False, na_position="last")
    if len(ordered) <= _MAX_SENTIMENT_ARTICLES:
        return ordered.reset_index(drop=True)
    kept = ordered.head(_MAX_SENTIMENT_ARTICLES).reset_index(drop=True)
    logger.warning(
        "Tier-1 news: capped %d articles to the newest %d (oldest kept: %s). "
        "Raise _MAX_SENTIMENT_ARTICLES only with a measured runtime -- this step "
        "is the whole weekly budget.",
        len(articles),
        _MAX_SENTIMENT_ARTICLES,
        kept["published_at"].min() if "published_at" in kept else "unknown",
    )
    return kept


def process_tier1_news(
    results: list[TickerFetchResult],
    universe: pd.DataFrame,
    today: date,
    *,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Entity-tag, classify, and sentiment-score Tier-1 news into persistable records.

    Runs the three local models (spaCy NER, BART-MNLI, FinBERT) once over the
    whole night's articles rather than per ticker, then derives (a) per-symbol
    decay-weighted `sentiment_scores` rows and (b) per-article `news_events`
    rows. Returns `([], [])` without loading any model when there are no
    articles, so a news-less run stays cheap.
    """
    frames = [
        r.tier1_news_df
        for r in results
        if r.tier1_news_df is not None and not r.tier1_news_df.empty
    ]
    if not frames:
        return [], []

    articles = pd.concat(frames, ignore_index=True)
    articles = articles.drop_duplicates(subset=["link"]).reset_index(drop=True)
    articles = _cap_articles(articles)

    # The three model passes run cheapest-first, and the order is load-bearing.
    #
    # Classification used to run before sentiment. When the step then exhausted
    # its wall-clock budget it was killed mid-classification, so *nothing* was
    # persisted -- not the event types it had managed, and not the sentiment it
    # had never reached. Sentiment is the half that feeds the composite score,
    # so it now runs first and the expensive classifier takes whatever time is
    # left. Each pass is timed, because "step tier1_news failed after 5400s"
    # says nothing about which of the three spent it.
    started = time.monotonic()

    gazetteer = entity_extraction.build_gazetteer(universe)
    articles["matched_symbols"] = entity_extraction.tag_articles(articles, gazetteer)
    tagged_at = time.monotonic()

    articles["sentiment"] = sentiment.score_articles(articles)
    scored_at = time.monotonic()

    # Keep the EventType enum in-frame so the decay step reads each event's
    # half-life directly; stringify only when building the stored row.
    # Newest-first from `_cap_articles`, so the expensive classifier spends its
    # budget where the recency-decay step gives the most weight. Everything past
    # the limit -- or past the deadline -- is `EventType.OTHER`.
    classifications = event_classifier.classify_articles(
        articles,
        max_classified=_MAX_CLASSIFIED_ARTICLES,
        deadline=deadline,
    )
    articles["event_type"] = classifications.apply(lambda c: c.event_type)
    logger.info(
        "Tier-1 news over %d articles: entity tagging %.0fs, sentiment %.0fs, "
        "classification %.0fs (cap %d).",
        len(articles),
        tagged_at - started,
        scored_at - tagged_at,
        time.monotonic() - scored_at,
        _MAX_CLASSIFIED_ARTICLES,
    )

    news_records: list[dict[str, Any]] = []
    for row in articles.itertuples():
        news_records.append(
            {
                "article_id": persistence.article_id_for(row.link, fallback=str(row.title)),
                "tier": 1,
                "title": row.title,
                "published_at": row.published_at if pd.notna(row.published_at) else None,
                "matched_symbols": list(row.matched_symbols),
                "matched_theme": None,
                "event_type": str(row.event_type),
                "sentiment_score": row.sentiment.polarity,
                "source": row.source,
                "source_url": row.link,
            }
        )

    sentiment_records: list[dict[str, Any]] = []
    for symbol in universe["symbol"]:
        aggregated = sentiment.aggregate_decayed_sentiment(symbol, articles, today)
        if aggregated is not None:
            sentiment_records.append(
                {
                    "symbol": symbol,
                    "date": today,
                    "source": "tier1_aggregate",
                    "sentiment_score": aggregated.score,
                    "mention_volume": aggregated.mention_volume,
                    "total_weight": aggregated.total_weight,
                }
            )
    return sentiment_records, news_records


def refresh_tier2_news(session: Session, today: date) -> int:
    """Ingest Tier-2 industry/thematic news from GDELT into `news_events` (Section 7.3).

    One GDELT query per curated thematic basket, each article classified and
    sentiment-scored and stored with its `matched_theme`. Phase 6 reads these
    (with `thematic_baskets`) to propagate a basket-level move to its members;
    the propagation math itself stays in `thematic_mapping`.
    """
    records: list[dict[str, Any]] = []
    for basket in thematic_mapping.THEMATIC_BASKETS:
        if not basket.keywords:
            continue
        query = "(" + " OR ".join(f'"{keyword}"' for keyword in basket.keywords) + ")"
        try:
            articles = gdelt_client.fetch_articles(query, timespan="1d")
        except Exception:
            logger.exception("GDELT Tier-2 fetch failed for basket %s", basket.name)
            continue
        if articles.empty:
            continue
        classifications = event_classifier.classify_articles(articles)
        sentiments = sentiment.score_articles(articles)
        for position, row in enumerate(articles.itertuples()):
            records.append(
                {
                    "article_id": persistence.article_id_for(row.url, fallback=str(row.title)),
                    "tier": 2,
                    "title": row.title,
                    "published_at": row.published_at if pd.notna(row.published_at) else None,
                    "matched_symbols": None,
                    "matched_theme": basket.name,
                    "event_type": str(classifications.iloc[position].event_type),
                    "sentiment_score": sentiments.iloc[position].polarity,
                    "source": "gdelt",
                    "source_url": row.url,
                }
            )
    return persistence.upsert_news_events(session, records)


def run(
    job_name: str = "refresh_data",
    *,
    force_weekly: bool = False,
    ignore_market_calendar: bool = False,
) -> str:
    """Run the refresh. Both overrides exist for operations, not for the schedule.

    `force_weekly` runs the weekly branch (fundamentals, analyst consensus,
    13F, forecasts, backtest, news and sentiment) on a day that is not Monday.
    The weekly branch is the one that has historically failed to complete --
    it once exhausted the six-hour job limit -- and waiting a week to find out
    whether a fix worked is a bad feedback loop. It is also how a database that
    has never had a successful weekly run gets caught up without pretending
    today is Monday.

    `ignore_market_calendar` runs on a closed day. Normally a non-trading day is
    a deliberate cheap no-op; when catching up, refusing to run because it
    happens to be Saturday is the wrong answer.

    Neither is set by the schedule. `.github/workflows/refresh_data.yml` exposes
    both as manual-dispatch inputs.
    """
    run_id = configure_logging(get_settings().log_level)
    logger.info(
        "%s starting (run_id=%s, force_weekly=%s, ignore_market_calendar=%s)",
        job_name,
        run_id,
        force_weekly,
        ignore_market_calendar,
    )
    # Before anything expensive. A missed `alembic upgrade head` otherwise
    # surfaces as "table X has no column named Y" from whichever step writes
    # that column first -- nine minutes and 503 tickers into the run. Checked
    # through this job's OWN session, so it verifies the database the run will
    # actually write to rather than whatever the module-level engine happens to
    # point at.
    with get_session() as session:
        assert_schema_current(session.connection())
    started_at = datetime.now()
    # The trading day is decided in EXCHANGE time, not the runner's. GitHub's
    # runners are UTC, and the schedule fires in the US evening -- so a naive
    # `date.today()` reads one day ahead of the session whose closing prices are
    # actually being fetched, stamping Monday's data as Tuesday and asking
    # `is_trading_day` about the wrong date. It also makes the job's behaviour
    # depend on how far GitHub's scheduler happens to slip, which is routinely
    # hours.
    today = datetime.now(_MARKET_TZ).date()
    status = "success"
    rows_updated = 0
    failed_steps: list[str] = []

    if not is_trading_day(today) and not ignore_market_calendar:
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
        return "skipped_non_trading_day"

    def step(name: str, fn: "Any") -> int:
        """Run one refresh sub-step, isolating its failure from the rest of the run.

        A single source being down (GDELT, SEC, Finnhub) must not take out the
        rest of the nightly run, so every step is isolated here and the run still
        records whatever else succeeded (Section 6.12).

        **But not every step is equally optional**, and treating them alike is
        how a run reports "partial" while having produced nothing a user can
        see. `_CRITICAL_STEPS` names the ones the app is unusable without:
        if one of those fails the run is "failed", not "partial", so the
        workflow goes red. Everything else degrades quietly as before, and the
        names are collected so the summary can say *what* degraded rather than
        leaving a reader to grep a nine-minute log.
        """
        nonlocal status
        budget = _STEP_TIMEOUT_SECONDS.get(name, _DEFAULT_STEP_TIMEOUT_SECONDS)
        started = time.monotonic()
        try:
            with _step_timeout(budget, name):
                return fn()
        except Exception:
            logger.exception(
                "%s: step %s failed after %.0fs", job_name, name, time.monotonic() - started
            )
            failed_steps.append(name)
            if name in _CRITICAL_STEPS:
                status = "failed"
            elif status != "failed":
                status = "partial"
            return 0

    try:
        with get_session() as session:
            rows_updated += sync_universe(session)
            # The searchable catalogue of everything else. Cheap (one file
            # fetch, names only) and it must not be able to break the ranked
            # universe, so it runs behind `step()` like any other optional
            # source: a Nasdaq outage costs search coverage for a day, not the
            # night's prices.
            # Its OWN session, deliberately. `step()` swallows the exception but
            # cannot undo what it did to the session: a failed flush leaves the
            # shared one needing a rollback, so the *next* statement on it fails
            # too and an isolated step takes down the whole run. That is exactly
            # what a malformed listing row did the first time this ran.
            rows_updated += step("catalogue", lambda: _in_session(sync_catalogue))
            # Record the index add/drop this sync just detected into the
            # point-in-time membership history, so it stays honest for the
            # survivorship-aware backtest between cold-start re-seeds (Section 6.9).
            rows_updated += reconcile_index_membership(session, today)
            active_tickers = session.execute(
                select(Ticker.symbol, Ticker.name).where(Ticker.is_active)
            ).all()
            active = {symbol: _last_price_date(session, symbol) for symbol, _ in active_tickers}
            universe_df = _active_universe(session)
        name_by_symbol = {symbol: name for symbol, name in active_tickers}
        sector_by_symbol = dict(zip(universe_df["symbol"], universe_df["sector"], strict=True))

        is_weekly = force_weekly or today.weekday() == _WEEKLY_REFRESH_WEEKDAY

        results: list[TickerFetchResult] = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    fetch_ticker_data,
                    symbol,
                    last_date,
                    is_weekly,
                    company_name=name_by_symbol.get(symbol),
                    sector=sector_by_symbol.get(symbol),
                    today=today,
                ): symbol
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
                # Each symbol's writes go in their own SAVEPOINT. Every other
                # stage in this run is isolated behind `step()`; without the
                # same treatment here, a single malformed row from one of ~500
                # tickers raises mid-loop and discards every symbol's writes,
                # turning one bad ticker into a total outage that still exits 0.
                # Now a bad ticker costs that ticker and downgrades to "partial".
                try:
                    with session.begin_nested():
                        symbol_rows = 0
                        if result.price_df is not None:
                            symbol_rows += _upsert_price_history(session, result.price_df)
                        if result.fundamentals is not None:
                            _upsert_fundamentals(
                                session,
                                result.symbol,
                                today,
                                _fundamentals_with_ffo(result.fundamentals, result.ffo_inputs),
                            )
                            symbol_rows += 1
                        if result.analyst_consensus is not None:
                            _upsert_analyst_consensus(
                                session, result.symbol, today, result.analyst_consensus
                            )
                            symbol_rows += 1
                        symbol_rows += _persist_per_ticker_smart_money(session, result, today)
                except Exception:
                    logger.exception("Failed persisting %s; skipping it", result.symbol)
                    status = "partial"
                else:
                    rows_updated += symbol_rows

        # Daily cross-asset ingestion feeds the (daily) Market Regime Index.
        rows_updated += step(
            "cross_asset_macro",
            lambda: _in_session(lambda s: refresh_cross_asset_macro(s, today)),
        )

        if is_weekly:
            rows_updated += step("macro_indicators", lambda: _in_session(refresh_macro_indicators))
            rows_updated += step(
                "static_config", lambda: _in_session(lambda s: refresh_static_config(s, today))
            )
            rows_updated += step(
                "institutional_ownership",
                lambda: _in_session(
                    lambda s: refresh_institutional_ownership(s, universe_df, today)
                ),
            )

        # Regime before composite: the composite reads the regime score back
        # for its risk-off rating dampener, so ordering matters.
        rows_updated += step(
            "market_regime", lambda: _in_session(lambda s: refresh_market_regime(s, today))
        )

        # Chart patterns are a daily technical read off the same price window the
        # composite uses, and the whole 503-name sweep costs about a second.
        rows_updated += step(
            "pattern_signals",
            lambda: _in_session(lambda s: refresh_pattern_signals(s, universe_df, today)),
        )

        # Composite scoring runs before forecasting/backtesting -- it reads back
        # every category's freshly-written rows (prices, sentiment, smart money,
        # regime) and turns them into the day's ranking (Section 7.5).
        rows_updated += step(
            "composite_scores",
            lambda: _in_session(lambda s: refresh_composite_scores(s, universe_df, today)),
        )

        # Phase 7 forecasting + backtesting ride the weekly cadence (the heaviest
        # steps; see the constants above). Forecasts read the fresh price history
        # written earlier this run; the backtest reads the price panel + point-in-
        # time membership, so both come after the day's prices are persisted.
        if is_weekly:
            rows_updated += step(
                "forecasts",
                lambda: _in_session(lambda s: refresh_forecasts(s, universe_df, today)),
            )
            rows_updated += step(
                "backtest", lambda: _in_session(lambda s: refresh_backtest(s, today))
            )

            # News-intelligence runs LAST, after everything a visitor actually
            # looks at has been written.
            #
            # It is by far the most expensive step (three local ML models over
            # the week's articles) and the only one that has ever exhausted the
            # job's time budget. When it sat mid-run, its stall took the whole
            # weekly branch with it -- market regime, patterns, composite
            # scores, forecasts and the backtest all sat behind it and never
            # ran, and because the job was killed rather than failing, the
            # workflow's commit step never executed either. Ordering it last
            # means the worst case for a news outage is stale sentiment, not a
            # night with no ratings.
            #
            # The trade is explicit and small: `refresh_composite_scores` reads
            # the latest *stored* sentiment point-in-time, so on the weekly run
            # the sentiment category is one run old (the rest of the week
            # already reads exactly these rows). Sentiment is a weekly signal to
            # begin with, so this costs a single day of freshness in one of
            # seven categories -- against a failure mode that has so far cost
            # every category, every week.
            # The classifier gets whatever is left of the step's own budget,
            # minus a margin for persisting what it produced. Without this the
            # deadline would be the SIGALRM that kills the step, which is the
            # failure being fixed rather than a graceful stop before it.
            news_deadline = time.monotonic() + (
                _STEP_TIMEOUT_SECONDS["tier1_news"] - _NEWS_PERSIST_MARGIN_SECONDS
            )
            rows_updated += step(
                "tier1_news",
                lambda: _persist_tier1_news(results, universe_df, today, deadline=news_deadline),
            )
            rows_updated += step(
                "tier2_news", lambda: _in_session(lambda s: refresh_tier2_news(s, today))
            )

    except Exception:
        logger.exception("%s failed", job_name)
        status = "failed"

    if failed_steps:
        # Name them. "partial" on its own sends a reader to grep a long log for
        # a traceback; this puts the answer in the last line.
        logger.warning(
            "%s finished %s -- failed step(s): %s", job_name, status, ", ".join(failed_steps)
        )
    else:
        logger.info("%s finished %s (%d rows)", job_name, status, rows_updated)

    with get_session() as session:
        session.add(
            RefreshLog(
                job_name=job_name,
                run_timestamp=started_at,
                status=status,
                rows_updated=rows_updated,
            )
        )
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh the QuantPulse database.")
    parser.add_argument(
        "--force-weekly",
        action="store_true",
        help="Run the weekly steps (fundamentals, analyst, 13F, forecasts, backtest, "
        "news, sentiment) even when today is not Monday.",
    )
    parser.add_argument(
        "--ignore-market-calendar",
        action="store_true",
        help="Run even when the exchange is closed, instead of skipping.",
    )
    arguments = parser.parse_args()

    # Exit non-zero on a hard failure so the scheduled workflow actually goes
    # red. Previously `run()` caught everything, recorded status="failed" in
    # `refresh_log`, and still exited 0 -- so GitHub Actions reported four
    # consecutive successes while the deployed demo's prices sat frozen and
    # nobody had any reason to look. "partial" stays green on purpose: that is
    # the designed degradation for one flaky upstream source, not an outage.
    if (
        run(
            force_weekly=arguments.force_weekly,
            ignore_market_calendar=arguments.ignore_market_calendar,
        )
        == "failed"
    ):
        sys.exit(1)
