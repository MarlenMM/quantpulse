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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.analysis import fundamental, macro
from quantpulse.config import get_settings
from quantpulse.ingestion import (
    economic_calendar,
    edgar_13f_client,
    edgar_client,
    fred_client,
    gdelt_client,
    news_client,
    options_client,
    short_interest_client,
    wikipedia_client,
    yfinance_client,
)
from quantpulse.news_intelligence import (
    entity_extraction,
    event_classifier,
    market_regime,
    sentiment,
    thematic_mapping,
)
from quantpulse.storage import persistence
from quantpulse.storage.db import get_session
from quantpulse.storage.models import (
    AnalystConsensus,
    FundamentalsSnapshot,
    MacroIndicator,
    PriceHistory,
    RefreshLog,
    Ticker,
)
from quantpulse.utils.log import configure_logging
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

# Section 6.3 calls for daily news; running three local ML models over the
# whole universe every night is the single heaviest workload here and needs the
# concurrency/model-cache work (Sections 6.10-6.11) to be daily-affordable on a
# free runner. Until then the news-intelligence and slow smart-money signals
# (short interest, insider filings, quarterly 13F) ride the weekly cadence --
# a documented, single-constant deviation, not a silent gap.
_NEWS_REFRESH_ON_WEEKLY_ONLY = True

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


_REIT_SECTOR = "Real Estate"


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
    results: list[TickerFetchResult], universe: pd.DataFrame, today: date
) -> int:
    """Run the (session-free) Tier-1 news models, then persist the results in one session.

    The heavy model pass (`process_tier1_news`) deliberately holds no DB
    session while it runs, so SQLite's single writer isn't locked for the
    duration of hundreds of classifications.
    """
    sentiment_records, news_records = process_tier1_news(results, universe, today)
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


def process_tier1_news(
    results: list[TickerFetchResult], universe: pd.DataFrame, today: date
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

    gazetteer = entity_extraction.build_gazetteer(universe)
    articles["matched_symbols"] = entity_extraction.tag_articles(articles, gazetteer)
    # Keep the EventType enum in-frame so the decay step reads each event's
    # half-life directly; stringify only when building the stored row.
    articles["event_type"] = event_classifier.classify_articles(articles).apply(
        lambda c: c.event_type
    )
    articles["sentiment"] = sentiment.score_articles(articles)

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


def run(job_name: str = "refresh_data") -> None:
    run_id = configure_logging(get_settings().log_level)
    logger.info("%s starting (run_id=%s)", job_name, run_id)
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

    def step(name: str, fn: "Any") -> int:
        """Run one refresh sub-step, degrading a failure to 'partial' rather than aborting.

        A single source being down (GDELT, SEC, Finnhub) must not take out the
        rest of the nightly run, so every optional step is isolated here and
        the run still records whatever else succeeded (Section 6.12).
        """
        nonlocal status
        try:
            return fn()
        except Exception:
            logger.exception("%s: step %s failed", job_name, name)
            status = "partial"
            return 0

    try:
        with get_session() as session:
            rows_updated += sync_universe(session)
            active_tickers = session.execute(
                select(Ticker.symbol, Ticker.name).where(Ticker.is_active)
            ).all()
            active = {symbol: _last_price_date(session, symbol) for symbol, _ in active_tickers}
            universe_df = _active_universe(session)
        name_by_symbol = {symbol: name for symbol, name in active_tickers}
        sector_by_symbol = dict(zip(universe_df["symbol"], universe_df["sector"], strict=True))

        is_weekly = today.weekday() == _WEEKLY_REFRESH_WEEKDAY

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
                if result.price_df is not None:
                    rows_updated += _upsert_price_history(session, result.price_df)
                if result.fundamentals is not None:
                    _upsert_fundamentals(
                        session,
                        result.symbol,
                        today,
                        _fundamentals_with_ffo(result.fundamentals, result.ffo_inputs),
                    )
                    rows_updated += 1
                if result.analyst_consensus is not None:
                    _upsert_analyst_consensus(
                        session, result.symbol, today, result.analyst_consensus
                    )
                    rows_updated += 1
                rows_updated += _persist_per_ticker_smart_money(session, result, today)

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
            rows_updated += step(
                "tier1_news", lambda: _persist_tier1_news(results, universe_df, today)
            )
            rows_updated += step(
                "tier2_news", lambda: _in_session(lambda s: refresh_tier2_news(s, today))
            )

        # Regime runs last: it reads back the VIX/breadth/yield rows the steps
        # above just wrote, so ordering matters.
        rows_updated += step(
            "market_regime", lambda: _in_session(lambda s: refresh_market_regime(s, today))
        )

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
