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

    `strong_buy_cutoff` is the percentile a name must reach to be a Strong Buy
    *tonight* -- normally 90, lifted toward 95 by the Market Regime Index in a
    risk-off market (Section 7.3 Tier 3). A client that re-weights the composite
    locally has to re-rate as well, and re-deriving the dampener in its own
    language would put a second implementation of a market-wide judgment call in
    the codebase. Sending it means both front ends hand out the same count of
    Strong Buys on the same night.
    """

    as_of: date | None
    profile: str
    rating_mode: str = "relative"
    strong_buy_cutoff: float = 90.0
    count: int
    rows: list[ScreenerRow]


class InvestorProfileModel(BaseModel):
    """One of Section 23's presets, with everything a client needs to offer it.

    `rescores` is the load-bearing field. Four of the six presets differ from
    balanced by category *weights* alone, so a client holding the stored
    weight-independent sub-scores can apply them locally and instantly. The
    other two genuinely re-score a category -- income ranks fundamentals against
    a dividend-leaning sector config, conservative scores momentum toward low
    volatility -- and neither can be recovered by re-weighting a finished
    sub-score, so the nightly stores their rankings separately and a client must
    *fetch* them rather than compute them. A client that treated all six alike
    would silently show balanced sub-scores under an income label.
    """

    name: str
    description: str
    weights: dict[str, float]
    rescores: bool


class AbsoluteRating(BaseModel):
    """One symbol's absolute-mode score and rating (see `/api/screener/absolute`)."""

    symbol: str
    composite_score: float
    rating: str


class AbsoluteRatingResponse(BaseModel):
    """Absolute-mode ratings for the scored universe, or an honest refusal.

    `available` is false when the stored rows predate the raw category columns:
    an absolute rating genuinely cannot be recovered from a percentile, and
    saying so is the honest option -- quietly returning relative ratings under
    an "absolute" label would be the exact mislabelling the mode exists to stop.
    """

    available: bool
    profile: str
    rating_mode: str = "absolute"
    rows: list[AbsoluteRating] = []


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
    that will ship without it. `hit_rate_windows` travels with it for the same
    reason -- a rate measured over 40 distinct out-of-sample windows and one
    measured over 3 are different claims, and a client that has to ask for the
    sample size separately will render the percentage without it.
    """

    model_name: str
    horizon_days: int
    point_return: float
    point_price: float | None = None
    lower_price: float | None = None
    upper_price: float | None = None
    historical_hit_rate: float | None = None
    baseline_hit_rate: float | None = None
    hit_rate_windows: int | None = None
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


class ShortInterestReading(BaseModel):
    """Section 24's two readings, deliberately never collapsed into one verdict.

    Heavy shorting can mean informed money is betting against the company, or it
    can set up a squeeze if sentiment turns. `smart_money.py` keeps it out of the
    blended score entirely and returns both figures intact; the client is
    expected to present them the same way, which is why `elevated` is a flag
    rather than a direction.
    """

    pct_float_short: float | None = None
    days_to_cover: float | None = None
    elevated: bool = False


class RiskProfileModel(BaseModel):
    """Section 7.7's per-name risk block.

    Every field is nullable because each estimator declines independently when
    its own data floor isn't met -- a short-history name yields a partly filled
    block rather than fabricated numbers. `ratio_min_observations` is sent so the
    client can explain *why* Sharpe and Sortino are absent instead of rendering
    two unexplained dashes.
    """

    historical_volatility: float | None = None
    implied_volatility: float | None = None
    implied_premium: float | None = None
    beta: float | None = None
    beta_r_squared: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    value_at_risk: float | None = None
    expected_shortfall: float | None = None
    var_confidence: float | None = None
    n_observations: int = 0
    ratio_min_observations: int = 0


class MonteCarloBand(BaseModel):
    """One day of the simulated fan: the percentile spread of simulated prices."""

    day: int
    lower: float
    median: float
    upper: float


class MonteCarloFan(BaseModel):
    """Section 7.6's simulated-path fan chart.

    The same random-walk-with-drift model `baseline_forecast` uses, executed by
    simulation instead of in closed form -- which is exactly why it is not a
    fourth competing model in the forecast table. It answers a different kind of
    question: not "what is the number" but "how wide does the range get".
    """

    horizon_days: int
    n_paths: int
    n_train: int
    mu: float
    sigma: float
    last_close: float
    bands: list[MonteCarloBand] = Field(default_factory=list)


class MacroOverlayComponent(BaseModel):
    driver: str
    sensitivity: float
    move: float | None = None


class MacroOverlay(BaseModel):
    """Section 28's targeted commodity/currency tilt for this stock's sector.

    Deliberately sparse: oil for Energy, gold for Materials, the dollar for the
    sectors dominated by multinationals earning abroad, and nothing at all for
    every other sector ("a small biotech doesn't care about oil prices").
    """

    sector: str
    adjustment: float
    components: list[MacroOverlayComponent] = Field(default_factory=list)


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
    short_interest: ShortInterestReading | None = None
    risk: RiskProfileModel | None = None
    monte_carlo: MonteCarloFan | None = None
    macro_overlay: MacroOverlay | None = None


class SectorStrength(BaseModel):
    """One sector's strength relative to the market over the lookback window."""

    sector: str
    relative_return: float
    n_symbols: int


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
    # Section 27's fractional-Kelly sizing. `payoff_ratio` is the stored mean
    # win / mean loss; `kelly_fraction` is computed server-side by
    # `optimization.kelly_position_fraction` rather than in the client, so the
    # two front ends cannot arrive at different position sizes from the same
    # track record -- the same one-implementation rule that keeps the
    # Strong-Buy cutoff out of TypeScript. `None` when the run has no losing
    # period (an undefined payoff ratio would feed Kelly a bet it thinks
    # cannot lose) or when there is no positive edge to size.
    payoff_ratio: float | None = None
    kelly_fraction: float | None = None
