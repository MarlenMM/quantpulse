from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base. Tables land here starting in Phase 1 (Section 13)."""


class Ticker(Base):
    """Universe definition. `is_active` flags current-vs-removed without deleting history."""

    __tablename__ = "tickers"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    exchange: Mapped[str | None] = mapped_column(String(20))
    asset_type: Mapped[str] = mapped_column(String(10), default="equity")
    is_active: Mapped[bool] = mapped_column(default=True)


class IndexMembershipHistory(Base):
    """Point-in-time index membership, for survivorship-bias-aware backtests (Section 5, 22).

    Table only in Phase 1 — populating it correctly during the cold-start
    backfill is Phase 1's Opus-owned half of the work.
    """

    __tablename__ = "index_membership_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"))
    added_date: Mapped[date] = mapped_column(Date)
    removed_date: Mapped[date | None] = mapped_column(Date)


class PriceHistory(Base):
    """Raw OHLCV price data, keyed so a re-run can find each symbol's last stored date."""

    __tablename__ = "price_history"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


class FundamentalsSnapshot(Base):
    """Point-in-time fundamentals — append-only by (symbol, as_of_date), never overwritten."""

    __tablename__ = "fundamentals_snapshot"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    pe: Mapped[float | None] = mapped_column(Float)
    pb: Mapped[float | None] = mapped_column(Float)
    ps: Mapped[float | None] = mapped_column(Float)
    peg: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    revenue_growth: Mapped[float | None] = mapped_column(Float)
    debt_equity: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    roa: Mapped[float | None] = mapped_column(Float)
    fcf: Mapped[float | None] = mapped_column(Float)
    div_yield: Mapped[float | None] = mapped_column(Float)
    # Sector-specific substitutes (e.g. FFO for REITs, Section 7.2) — populated in Phase 3.
    sector_specific_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class AnalystConsensus(Base):
    """Wall Street analyst rating counts + mean price target, point-in-time."""

    __tablename__ = "analyst_consensus"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    strong_buy: Mapped[int] = mapped_column(default=0)
    buy: Mapped[int] = mapped_column(default=0)
    hold: Mapped[int] = mapped_column(default=0)
    sell: Mapped[int] = mapped_column(default=0)
    strong_sell: Mapped[int] = mapped_column(default=0)
    mean_price_target: Mapped[float | None] = mapped_column(Float)


class MacroIndicator(Base):
    """Raw FRED macro series (Section 5) — one row per indicator per date."""

    __tablename__ = "macro_indicators"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    indicator_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class PatternSignal(Base):
    """Detected chart/candlestick patterns (Section 13).

    Shared by candlestick detection (Phase 2, native library detectors) and
    the geometric chart-pattern algorithms (Phase 2's Opus half, e.g.
    head-and-shoulders) -- both normalize to this same shape.
    """

    __tablename__ = "pattern_signals"
    __table_args__ = (UniqueConstraint("symbol", "date", "pattern_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"))
    date: Mapped[date] = mapped_column(Date)
    pattern_type: Mapped[str] = mapped_column(String(50))
    direction: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)


class RefreshLog(Base):
    """Pipeline health monitoring for each nightly/cold-start run."""

    __tablename__ = "refresh_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(50))
    run_timestamp: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))
    rows_updated: Mapped[int] = mapped_column(default=0)


# ---------------------------------------------------------------------------
# Phase 4 — News & Event Intelligence (Section 7.3, 13)
#
# The Phase-4 analysis modules (entity_extraction / event_classifier /
# sentiment / thematic_mapping / market_regime) were built as pure functions
# over in-memory frames; these tables are their persistence layer, landing
# together with the writer that populates them (scripts/refresh_data.py) per
# the project's "schema alongside its writer" convention.
# ---------------------------------------------------------------------------


class SentimentScore(Base):
    """Tier-1 company-level decay-weighted sentiment (Section 7.3 step 4, Section 13).

    One row per (symbol, as-of date, source): the decay-weighted average
    polarity `AggregatedSentiment` produces. `total_weight` (sum of the decay
    weights actually used) is where staleness shows up -- a value near 0 means
    the score rests on old evidence -- so it's persisted alongside the score
    rather than discarded (see `sentiment.aggregate_decayed_sentiment`).
    """

    __tablename__ = "sentiment_scores"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(30), primary_key=True, default="tier1_aggregate")
    sentiment_score: Mapped[float] = mapped_column(Float)  # decay-weighted polarity, [-1, 1]
    mention_volume: Mapped[int] = mapped_column(default=0)
    total_weight: Mapped[float | None] = mapped_column(Float)


class NewsEvent(Base):
    """Every ingested article, tagged with tier/matched entities/event type (Section 13).

    The raw material behind `sentiment_scores` (Tier 1) and the sector/macro
    adjustments (Tiers 2-3). `article_id` is a stable hash of the source URL so
    re-ingesting the same article dedupes rather than duplicating.
    `matched_symbols` is the JSON list `entity_extraction.tag_articles`
    produces (no FK -- an article can name tickers outside the tracked
    universe).
    """

    __tablename__ = "news_events"

    article_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tier: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(String(500))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    matched_symbols: Mapped[list[str] | None] = mapped_column(JSON)
    matched_theme: Mapped[str | None] = mapped_column(String(50))
    event_type: Mapped[str | None] = mapped_column(String(30))
    sentiment_score: Mapped[float | None] = mapped_column(Float)  # per-article polarity, [-1, 1]
    source: Mapped[str | None] = mapped_column(String(30))
    source_url: Mapped[str | None] = mapped_column(String(1000))


class ThematicBasket(Base):
    """Config-driven ticker->theme membership (Section 7.3 step 5, Section 13).

    Persisted form of `thematic_mapping.iter_basket_membership()`. No FK on
    `symbol`: curated baskets deliberately include names outside the S&P 500
    universe (e.g. TSM, ASML), which is the whole point of a thematic basket.
    """

    __tablename__ = "thematic_baskets"

    theme_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)


class MarketRegime(Base):
    """Daily Market Regime Index output (Section 5, 7.3 Tier 3, 28; Section 13).

    The quantitative "build-your-own Fear/Greed" composite: VIX, breadth
    (% of universe above its 200-DMA), GDELT macro tone, and the 10Y-2Y
    yield-curve spread (Section 28) blended into a 0-100 `regime_score` and a
    `regime_label` (risk_on / neutral / risk_off). Any input may be null when
    unavailable; `regime_score` is renormalized over whatever was present.
    """

    __tablename__ = "market_regime"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    vix_level: Mapped[float | None] = mapped_column(Float)
    breadth_pct_above_200dma: Mapped[float | None] = mapped_column(Float)
    macro_news_tone: Mapped[float | None] = mapped_column(Float)
    yield_curve_spread: Mapped[float | None] = mapped_column(Float)  # 10Y-2Y, Section 28
    regime_score: Mapped[float | None] = mapped_column(Float)  # 0-100, higher = risk-on
    regime_label: Mapped[str | None] = mapped_column(String(20))


class EconomicCalendarEvent(Base):
    """Scheduled macro releases (FOMC/CPI/jobs report) for the "uncertainty ahead" flag.

    Section 28, Section 13. Persisted form of `ingestion.economic_calendar`'s
    static schedule.
    """

    __tablename__ = "economic_calendar"

    event_date: Mapped[date] = mapped_column(Date, primary_key=True)
    event_name: Mapped[str] = mapped_column(String(50), primary_key=True)


# ---------------------------------------------------------------------------
# Phase 5 — Smart Money Signals (Section 24, 13)
# ---------------------------------------------------------------------------


class InsiderTransaction(Base):
    """Form-4 non-derivative insider buy/sell transactions (Section 24, Section 13).

    One row per parsed transaction. Dedupe is by the natural key below rather
    than a filing-level id, since the ingestion client discards the accession
    number after parsing -- re-ingesting the same window is idempotent because
    identical (symbol, insider, date, code, shares) rows collide.
    """

    __tablename__ = "insider_transactions"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "insider_name",
            "transaction_date",
            "transaction_code",
            "shares",
            name="uq_insider_transaction",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"))
    insider_name: Mapped[str | None] = mapped_column(String(255))
    insider_title: Mapped[str | None] = mapped_column(String(255))
    filing_date: Mapped[date | None] = mapped_column(Date)
    transaction_date: Mapped[date | None] = mapped_column(Date)
    transaction_code: Mapped[str | None] = mapped_column(String(5))  # P/S (open-market), etc.
    acquired_disposed_code: Mapped[str | None] = mapped_column(String(1))  # A / D
    shares: Mapped[float | None] = mapped_column(Float)
    price_per_share: Mapped[float | None] = mapped_column(Float)
    shares_owned_after: Mapped[float | None] = mapped_column(Float)


class InstitutionalOwnership(Base):
    """13F institutional ownership trend, aggregated per symbol per quarter (Section 24).

    Section 13's schema lists a per-institution `institution_name`, but 13F
    holdings can't be mapped to a ticker without a (non-free) CUSIP table, so
    the ingestion client matches by normalized issuer name and returns one
    aggregated row per symbol per quarter instead -- a documented limitation
    (see `edgar_13f_client`). `change_from_prior_quarter` is null when there's
    no comparable prior-quarter figure (an honest unknown, not a filled zero).
    """

    __tablename__ = "institutional_ownership"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    quarter_end_date: Mapped[date] = mapped_column(Date, primary_key=True)
    total_shares_held: Mapped[float | None] = mapped_column(Float)
    total_value: Mapped[float | None] = mapped_column(Float)
    num_filers: Mapped[int | None] = mapped_column(Integer)
    change_from_prior_quarter: Mapped[float | None] = mapped_column(Float)


class OptionsSignal(Base):
    """Daily options-positioning snapshot (Section 24, Section 13).

    `put_call_ratio` and `atm_implied_volatility` are the raw daily snapshot;
    `iv_rank` (today's IV vs. its own trailing range) is null until enough
    daily rows accumulate for `options_client.compute_iv_rank` to rank against
    -- persisting this table over time is exactly what makes IV-rank possible.
    """

    __tablename__ = "options_signals"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    expiration: Mapped[str | None] = mapped_column(String(10))
    put_call_ratio: Mapped[float | None] = mapped_column(Float)
    atm_implied_volatility: Mapped[float | None] = mapped_column(Float)
    iv_rank: Mapped[float | None] = mapped_column(Float)


class ShortInterest(Base):
    """Short-interest snapshot (Section 24, Section 13).

    Deliberately never collapsed into a directional score -- both
    `pct_float_short` (bearish conviction) and `days_to_cover` (squeeze setup)
    are surfaced as-is (see `smart_money.read_short_interest`). Either may be
    null if Finnhub doesn't cover the name.
    """

    __tablename__ = "short_interest"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    pct_float_short: Mapped[float | None] = mapped_column(Float)
    days_to_cover: Mapped[float | None] = mapped_column(Float)


# ---------------------------------------------------------------------------
# Phase 6 — Composite Scoring & Ranking (Section 7.5, 13)
# ---------------------------------------------------------------------------


class CompositeScore(Base):
    """The core ranking output: seven category sub-scores blended into one rating.

    Append-only, keyed by `as_of` date and never overwritten (Section 6.8), so
    "what did the algorithm say about AAPL on June 3rd" always returns exactly
    that. The seven `*_score` columns are the 0-100 normalized (percentile,
    sector-relative for fundamentals) per-category sub-scores -- these are
    weight-*independent*, which is what lets the UI re-weight to a different
    investor profile client-side without re-running the pipeline (Section 8).
    Any sub-score is null when that category had no usable data for the symbol.

    `profile` records which investor-profile weights produced `composite_score`
    / `percentile_rank` / `rating` (the balanced default nightly; other
    profiles can be stored alongside since it's part of the primary key).
    `data_confidence` (0-100) is the fraction of category weight that had usable
    data -- an honesty signal so a thinly-covered micro-cap isn't shown with the
    same confidence as a mega-cap (Section 7.5 step 6).
    """

    __tablename__ = "composite_scores"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    profile: Mapped[str] = mapped_column(String(30), primary_key=True, default="balanced")
    fundamental_score: Mapped[float | None] = mapped_column(Float)
    technical_score: Mapped[float | None] = mapped_column(Float)
    analyst_score: Mapped[float | None] = mapped_column(Float)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    momentum_score: Mapped[float | None] = mapped_column(Float)
    industry_macro_score: Mapped[float | None] = mapped_column(Float)
    smart_money_score: Mapped[float | None] = mapped_column(Float)
    composite_score: Mapped[float] = mapped_column(Float)
    percentile_rank: Mapped[float | None] = mapped_column(Float)
    rating: Mapped[str] = mapped_column(String(15))
    data_confidence: Mapped[float] = mapped_column(Float)
