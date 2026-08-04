"""Pydantic response models for the REST API (ADR 4.1's stretch goal).

These types are the API's contract, and they exist for a reason beyond
serialization: they are what makes the TypeScript client's types honest. Every
field that can be absent is typed `| None` rather than defaulted to zero,
because the whole project's coverage discipline (a missing sub-score is not a
zero sub-score, Section 7.5) has to survive the trip through JSON or the
frontend will quietly render fabricated confidence.

Field names deliberately match the database columns and the Streamlit code
rather than being re-cased for JavaScript. One vocabulary across the Python
engine, the API and the React client is worth more than idiomatic camelCase in
one of the three.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

__all__ = [
    "HealthResponse",
    "GlossaryTerm",
    "TickerSummary",
    "ScreenerRow",
    "ScreenerResponse",
    "RatingChange",
    "PriceBar",
    "ForecastRow",
    "PatternRow",
    "AnalystConsensusModel",
    "NewsItem",
    "StockDetail",
    "RegimePoint",
    "BacktestRun",
]


class HealthResponse(BaseModel):
    """Liveness plus what the pipeline has actually produced.

    `has_data` lets a client show the same honest "nothing ingested yet" screen
    the Streamlit app shows, instead of an array of empty widgets that reads as
    a broken deployment rather than an un-run pipeline.
    """

    status: str = "ok"
    has_data: bool
    freshness: dict[str, date | None]


class GlossaryTerm(BaseModel):
    term: str
    category: str
    definition: str


class TickerSummary(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    asset_type: str | None = None


class ScreenerRow(BaseModel):
    """One scored symbol. Every sub-score is nullable — see the module docstring."""

    symbol: str
    name: str | None = None
    sector: str | None = None
    date: date
    fundamental_score: float | None = None
    technical_score: float | None = None
    analyst_score: float | None = None
    sentiment_score: float | None = None
    momentum_score: float | None = None
    industry_macro_score: float | None = None
    smart_money_score: float | None = None
    composite_score: float
    percentile_rank: float | None = None
    rating: str
    data_confidence: float | None = None


class ScreenerResponse(BaseModel):
    """The ranked table plus the context needed to read it honestly.

    `rating_mode` travels with the rows so a client cannot present a relative
    ranking as an absolute judgment (Section 22) simply because it forgot to
    ask which scheme produced it.
    """

    as_of: date | None
    profile: str
    rating_mode: str = "relative"
    count: int
    rows: list[ScreenerRow]


class RatingChange(BaseModel):
    symbol: str
    previous_rating: str
    rating: str
    previous_score: float
    composite_score: float
    score_change: float


class PriceBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float | None = None


class ForecastRow(BaseModel):
    """A stored forecast, always carrying its own track record.

    `historical_hit_rate` is part of the row rather than a separate endpoint on
    purpose: Section 7.6 requires a forecast to be shown next to its own
    accuracy, and a client that has to make a second call to get it is a client
    that will ship without it.
    """

    model_name: str
    horizon_days: int
    point_return: float
    point_price: float | None = None
    lower_price: float | None = None
    upper_price: float | None = None
    historical_hit_rate: float | None = None
    baseline_hit_rate: float | None = None
    generated_date: date


class PatternRow(BaseModel):
    date: date
    pattern_type: str
    direction: str
    confidence: float


class AnalystConsensusModel(BaseModel):
    as_of_date: date
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    mean_price_target: float | None = None


class NewsItem(BaseModel):
    title: str | None = None
    published_at: datetime | None = None
    tier: int | None = None
    matched_theme: str | None = None
    event_type: str | None = None
    sentiment_score: float | None = None
    source: str | None = None
    source_url: str | None = None


class StockDetail(BaseModel):
    """Everything the Stock Detail page needs, in one round trip.

    Bundled rather than split across six endpoints because they are always
    rendered together; six requests would buy nothing but latency and a
    partially-populated page while they land.
    """

    symbol: str
    summary: TickerSummary
    score: ScreenerRow | None = None
    prices: list[PriceBar] = Field(default_factory=list)
    forecasts: list[ForecastRow] = Field(default_factory=list)
    patterns: list[PatternRow] = Field(default_factory=list)
    analyst_consensus: AnalystConsensusModel | None = None
    news: list[NewsItem] = Field(default_factory=list)


class RegimePoint(BaseModel):
    date: date
    vix_level: float | None = None
    breadth_pct_above_200dma: float | None = None
    macro_news_tone: float | None = None
    yield_curve_spread: float | None = None
    regime_score: float | None = None
    regime_label: str | None = None


class BacktestRun(BaseModel):
    """One stored backtest, with its confidence intervals attached.

    The CI bounds are nullable because a run too short to bootstrap honestly
    stores nulls rather than a fabricated interval (Section 7.6) — the client
    must be able to tell "no interval" from "interval of zero width".
    """

    run_date: date
    period_start: date | None = None
    period_end: date | None = None
    cadence: str
    n_periods: int
    sharpe: float | None = None
    sharpe_ci_low: float | None = None
    sharpe_ci_high: float | None = None
    cagr: float | None = None
    cagr_ci_low: float | None = None
    cagr_ci_high: float | None = None
    ci_confidence_level: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    benchmark_cagr: float | None = None
    benchmark_sharpe: float | None = None
    avg_turnover: float | None = None
    assumed_txn_cost: float
