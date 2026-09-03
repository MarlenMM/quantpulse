"""FastAPI read API over the analysis engine (ADR 4.1's stretch goal).

ADR 4.1 keeps Streamlit as the MVP and offers React + FastAPI as a level-up
"once the analysis engine (the actual hard part) is done — the engine is
UI-agnostic by design (Section 14), so this migration only touches the
presentation layer." That claim is now load-bearing rather than aspirational:
this module adds a second front end without a single line changing in
`analysis/`, `portfolio/` or `news_intelligence/`. Every route below is a thin
translation of a `storage.persistence` reader into a Pydantic model — the same
readers the Streamlit pages use, so the two UIs cannot disagree about what the
engine said.

**The API is deliberately read-only.** It serves computed research data; it
exposes no way to record a transaction or mutate a portfolio. That is not an
oversight:

* Portfolio state is per-user and ADR 4.5 splits it between a browser session
  and a local SQLite file. Neither maps onto a stateless public REST API
  without the authentication Section 18 explicitly says the single-user MVP
  does not have.
* A read-only surface cannot be turned into an unauthenticated write endpoint
  against someone's real holdings by a mistake in a later refactor.

Portfolio management therefore stays in the Streamlit app, which is where its
storage backends already live. If the React client ever needs it, the honest
prerequisite is auth, not a quick POST route.

Run it with:  `uv run uvicorn quantpulse.api.main:app --reload`
Interactive docs are then at `/docs` (FastAPI generates them from the schemas).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from quantpulse.analysis import forecasting, macro, risk, scoring, smart_money, technical
from quantpulse.analysis.investor_profiles import CATEGORIES, get_profile, profile_names
from quantpulse.api.schemas import (
    AbsoluteRating,
    AbsoluteRatingResponse,
    AnalystConsensusModel,
    BacktestRun,
    ForecastRow,
    GlossaryTerm,
    HealthResponse,
    InvestorProfileModel,
    MacroOverlay,
    MacroOverlayComponent,
    MonteCarloBand,
    MonteCarloFan,
    NewsItem,
    PatternRow,
    PriceBar,
    RatingChange,
    RegimePoint,
    RiskProfileModel,
    ScreenerResponse,
    ScreenerRow,
    SectorStrength,
    ShortInterestReading,
    StockDetail,
    TickerSummary,
)
from quantpulse.glossary import TERMS
from quantpulse.portfolio.optimization import kelly_position_fraction
from quantpulse.storage import persistence
from quantpulse.storage.db import get_session

# The Vite dev server's default origins. Only local development needs CORS at
# all -- in production the built SPA is served as static files from the same
# origin, so no cross-origin request happens. Kept to an explicit allow-list
# rather than "*" so this never becomes a wide-open API by default.
_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

# Trailing window for the market series beta regresses against (the stored
# `^GSPC` index, or the equal-weight proxy where it has not been backfilled --
# `risk.resolve_market_returns` decides), and for the commodity/currency series
# behind the Section 28 overlay's "last ~3 months" reading.
#
# Read from `risk` rather than redeclared: this page and the Streamlit Stock
# Detail page must regress against the same window or they publish different
# betas for the same stock, which is exactly what happened while there were
# three independent copies of the number.
_RISK_PANEL_DAYS = risk.MARKET_PANEL_DAYS
_MACRO_OVERLAY_DAYS = 120
# Rotation is a one-month relative-strength read, so a short panel is the right
# cost/benefit -- the same window `app/lib/data.universe_panel` uses.
_ROTATION_PANEL_DAYS = 150

app = FastAPI(
    title="QuantPulse API",
    version="0.1.0",
    summary="Read-only access to QuantPulse's computed research data.",
    description=(
        "Serves the same computed numbers the Streamlit app shows. All analysis is "
        "performed by the offline pipeline; this API never computes a score, forecast "
        "or risk metric on request. Educational/research use only — not financial "
        "advice, and not a registered investment advisor."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_DEV_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def db_session() -> Iterator[Session]:
    """Request-scoped database session (FastAPI dependency)."""
    with get_session() as session:
        yield session


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe dicts, with pandas' NaN mapped to None.

    Pydantic would happily coerce a float NaN into a JSON `NaN` literal, which
    is not valid JSON and which `JSON.parse` rejects. Converting to None here
    means an absent value stays absent all the way to the client rather than
    becoming either a crash or, worse, a zero.
    """
    if frame.empty:
        return []
    return frame.replace({float("nan"): None}).where(pd.notna(frame), None).to_dict("records")


def _none_if_nan(value: Any) -> float | None:
    """A scalar read straight off a DataFrame row, with NaN mapped to None.

    `_rows` handles whole frames; this is the single-cell equivalent for the
    computed sections below, which read one value out of a matched row rather
    than serializing the frame.
    """
    return None if value is None or pd.isna(value) else float(value)


# --------------------------------------------------------------------------- #
# Health & reference data
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health(session: Session = Depends(db_session)) -> HealthResponse:
    """Liveness plus per-dataset freshness (Section 12's data-freshness rule)."""
    freshness = persistence.read_data_freshness(session)
    return HealthResponse(
        has_data=any(value is not None for value in freshness.values()),
        freshness=freshness,
    )


@app.get("/api/glossary", response_model=list[GlossaryTerm], tags=["meta"])
def glossary() -> list[GlossaryTerm]:
    """Every glossary term (Section 10), served from the same dict Streamlit uses."""
    return [
        GlossaryTerm(term=term, category=category, definition=definition)
        for term, (category, definition) in TERMS.items()
    ]


@app.get("/api/universe", response_model=list[TickerSummary], tags=["meta"])
def universe(session: Session = Depends(db_session)) -> list[TickerSummary]:
    """The active ticker universe — what the client's symbol autocomplete searches."""
    return [TickerSummary(**row) for row in _rows(persistence.read_ticker_universe(session))]


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #


@app.get("/api/screener", response_model=ScreenerResponse, tags=["screener"])
def screener(
    profile: str = Query("balanced", description="Investor-profile weighting to read."),
    session: Session = Depends(db_session),
) -> ScreenerResponse:
    """The ranked table, newest scoring date, highest composite first.

    Returns every weight-independent sub-score so a client can re-weight the
    composite locally (Section 8's sliders) without another request — the same
    property that lets the Streamlit sidebar recompute instantly.
    """
    frame = persistence.read_screener_rows(session, profile=profile)
    rows = [ScreenerRow(**row) for row in _rows(frame)]
    as_of = rows[0].date if rows else None
    return ScreenerResponse(
        as_of=as_of,
        profile=profile,
        # The client re-rates locally when the sliders move, so it needs the
        # same Strong-Buy cutoff the stored ratings used -- including tonight's
        # risk-off lift. Derived here rather than in TypeScript so there is one
        # implementation of the Tier-3 dampener.
        strong_buy_cutoff=scoring.strong_buy_cutoff(
            persistence.read_latest_regime_score(session, as_of=as_of) if as_of else None
        ),
        count=len(rows),
        rows=rows,
    )


@app.get("/api/profiles", response_model=list[InvestorProfileModel], tags=["screener"])
def profiles() -> list[InvestorProfileModel]:
    """Section 23's six investor-profile presets, balanced first.

    Served rather than duplicated in the client for the same reason the
    glossary is: these weights are the scoring engine's own configuration, and a
    second copy in TypeScript would drift the moment one is retuned.
    """
    return [
        InvestorProfileModel(
            name=profile.name,
            description=profile.description,
            weights=dict(profile.weights),
            # Only these two need their own stored ranking; the rest are a
            # re-weighting the client can do instantly on rows it already has.
            rescores=bool(profile.income_tilt or profile.prefer_low_volatility),
        )
        for profile in (get_profile(name) for name in profile_names())
    ]


@app.get("/api/screener/absolute", response_model=AbsoluteRatingResponse, tags=["screener"])
def screener_absolute(
    profile: str = Query("balanced", description="Investor profile whose weights to apply."),
    session: Session = Depends(db_session),
) -> AbsoluteRatingResponse:
    """Re-rate the scored universe against a *fixed* bar instead of its peers.

    A relative rating always names a top decile Strong Buy however the whole
    market looks -- Section 22's warning, not a bug. Absolute mode measures each
    category against a fixed scale instead, so a broadly falling market
    genuinely produces fewer Strong Buys (and can produce none).

    Computed here rather than in the client because it needs
    `scoring.build_composite(..., rating_mode="absolute")` over the stored raw
    category values. Reimplementing that mapping in TypeScript would put a
    second copy of the scoring engine in the codebase; the Streamlit page
    delegates to the same function for exactly this reason.

    Returns `available=false` when the stored rows predate the raw columns: an
    absolute rating cannot be recovered from a percentile, and saying so is
    honest where silently serving relative ratings under an absolute label
    would not be.
    """
    rows = persistence.read_screener_rows(session, profile=profile)
    if rows.empty:
        return AbsoluteRatingResponse(available=False, profile=profile)

    raw_columns = [f"{category}_raw" for category in CATEGORIES]
    if not set(raw_columns).issubset(rows.columns):
        return AbsoluteRatingResponse(available=False, profile=profile)

    raw = rows[raw_columns].copy()
    raw.columns = list(CATEGORIES)
    raw.index = rows["symbol"]
    if raw.notna().to_numpy().sum() == 0:
        return AbsoluteRatingResponse(available=False, profile=profile)

    scored = scoring.build_composite(
        raw, profile=get_profile(profile), rating_mode="absolute"
    ).scores
    if scored.empty:
        return AbsoluteRatingResponse(available=False, profile=profile)

    return AbsoluteRatingResponse(
        available=True,
        profile=profile,
        # `build_composite` promotes the symbol index to a column and resets to
        # a positional index, so the symbol must be read from the column --
        # taking it from `iterrows`'s index yields "0", "1", "2".
        rows=[
            AbsoluteRating(
                symbol=str(record["symbol"]),
                composite_score=float(record["composite_score"]),
                rating=str(record["rating"]),
            )
            for _, record in scored.iterrows()
        ],
    )


@app.get("/api/screener/changes", response_model=list[RatingChange], tags=["screener"])
def rating_changes(
    profile: str = Query("balanced"),
    limit: int = Query(25, ge=1, le=200),
    session: Session = Depends(db_session),
) -> list[RatingChange]:
    """Symbols whose rating moved between the last two stored snapshots (Section 10)."""
    frame = persistence.read_rating_changes(session, profile=profile, limit=limit)
    return [RatingChange(**row) for row in _rows(frame)]


# --------------------------------------------------------------------------- #
# Per-symbol detail
# --------------------------------------------------------------------------- #


@app.get("/api/stocks/{symbol}", response_model=StockDetail, tags=["stocks"])
def stock_detail(
    symbol: str,
    lookback_days: int = Query(400, ge=30, le=2000),
    session: Session = Depends(db_session),
) -> StockDetail:
    """Everything the detail view renders, in one round trip.

    404s only when the symbol is not in the ticker universe at all. A known
    symbol with no scores or no forecasts yet returns a populated envelope with
    empty lists, because "we track this but haven't computed it yet" and "this
    company does not exist" are different answers and the client should be able
    to say which one it got.

    The market index is the one row that is in `tickers` and is nonetheless not
    a subject here: it exists so `price_history` has somewhere to put `^GSPC`
    for the beta regression, and a "stock page" for it would be a company
    profile of an index -- no score, no fundamentals, no analyst coverage, and
    a sector of `None`. Nothing links to it, but reachable-by-URL and
    intentional are different things.
    """
    ticker = symbol.strip().upper()
    universe_frame = persistence.read_ticker_universe(session, active_only=False)
    match = universe_frame[universe_frame["symbol"] == ticker]
    if match.empty or str(match.iloc[0]["asset_type"]) == "index":
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {ticker}")

    scores = persistence.read_screener_rows(session)
    score_rows = _rows(scores[scores["symbol"] == ticker]) if not scores.empty else []
    consensus = persistence.read_latest_analyst_consensus(session, ticker)
    bars = persistence.read_symbol_ohlcv(session, ticker, lookback_days=lookback_days)

    return StockDetail(
        symbol=ticker,
        summary=TickerSummary(**_rows(match)[0]),
        score=ScreenerRow(**score_rows[0]) if score_rows else None,
        prices=[PriceBar(**row) for row in _rows(bars)],
        forecasts=[
            ForecastRow(**row) for row in _rows(persistence.read_symbol_forecasts(session, ticker))
        ],
        patterns=[
            PatternRow(**row) for row in _rows(persistence.read_symbol_patterns(session, ticker))
        ],
        analyst_consensus=AnalystConsensusModel(**consensus) if consensus else None,
        news=[NewsItem(**row) for row in _rows(persistence.read_symbol_news(session, ticker))],
        short_interest=_short_interest(session, ticker),
        risk=_risk_profile(session, ticker, bars),
        monte_carlo=_monte_carlo(bars),
        macro_overlay=_macro_overlay(session, str(match.iloc[0]["sector"] or "")),
    )


# --------------------------------------------------------------------------- #
# Stock-detail sections that are computed rather than read straight from a table
#
# These four were reachable only from the Streamlit app, which meant the React
# client showed 7 of the 12 sections its sibling did -- including short
# interest, which Section 24 explicitly requires be surfaced (as two readings,
# never one verdict). Each helper calls the same analysis function the Streamlit
# page calls, so the two front ends cannot disagree about a number.
# --------------------------------------------------------------------------- #

# Matches `app/pages/2_Stock_Detail.py`.
_MONTE_CARLO_HORIZON = 63
_MIN_MONTE_CARLO_BARS = 60


def _short_interest(session: Session, symbol: str) -> ShortInterestReading | None:
    frame = persistence.read_latest_short_interest(session, as_of=date.today())
    if frame.empty:
        return None
    match = frame[frame["symbol"] == symbol]
    if match.empty:
        return None
    reading = smart_money.read_short_interest(_rows(match)[0])
    if reading.pct_float_short is None and reading.days_to_cover is None:
        return None
    return ShortInterestReading(
        pct_float_short=reading.pct_float_short,
        days_to_cover=reading.days_to_cover,
        elevated=reading.elevated,
    )


def _risk_profile(session: Session, symbol: str, bars: pd.DataFrame) -> RiskProfileModel | None:
    if bars.empty:
        return None
    closes = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["date"])))["close"]
    returns = risk.to_returns(closes)
    if returns.empty:
        return None

    # Same two-step the Streamlit pages take through `app/lib/data.market_series`:
    # one window (`_RISK_PANEL_DAYS` is `risk.MARKET_PANEL_DAYS`) and one
    # resolver deciding *which* market. Both halves have to be shared — the
    # window alone was, and the two front ends still published different betas,
    # because each built its own market series underneath it.
    start = date.today() - timedelta(days=_RISK_PANEL_DAYS)
    market_series = risk.resolve_market_returns(
        persistence.read_benchmark_closes(
            session, symbol=risk.MARKET_INDEX_SYMBOL, start=start, end=date.today()
        ),
        persistence.read_adj_close_panel(session, start=start, end=date.today()),
    )
    market = market_series.returns
    options = persistence.read_latest_options(session, as_of=date.today())
    implied = None
    if not options.empty:
        match = options[options["symbol"] == symbol]
        if not match.empty:
            implied = _none_if_nan(match.iloc[0].get("atm_implied_volatility"))

    profile = risk.stock_risk_profile(
        returns,
        market_returns=market if not market.empty else None,
        implied_volatility=implied,
    )
    var = profile.value_at_risk
    return RiskProfileModel(
        historical_volatility=profile.volatility.historical,
        implied_volatility=profile.volatility.implied,
        implied_premium=profile.volatility.implied_premium,
        beta=profile.beta.beta if profile.beta else None,
        beta_r_squared=profile.beta.r_squared if profile.beta else None,
        beta_benchmark=market_series.label if profile.beta else None,
        sharpe=profile.sharpe,
        sortino=profile.sortino,
        max_drawdown=profile.max_drawdown,
        value_at_risk=var.var if var else None,
        expected_shortfall=var.expected_shortfall if var else None,
        var_confidence=var.confidence if var else None,
        n_observations=profile.n_observations,
        # Sent so the client can say *why* the two ratios are absent rather than
        # rendering an unexplained dash.
        ratio_min_observations=risk.min_ratio_observations(risk.TRADING_DAYS_PER_YEAR),
    )


def _monte_carlo(bars: pd.DataFrame) -> MonteCarloFan | None:
    if bars.empty or len(bars) < _MIN_MONTE_CARLO_BARS:
        return None
    fan = forecasting.monte_carlo_fan_chart(bars.rename(columns=str.lower), _MONTE_CARLO_HORIZON)
    if fan is None:
        return None
    low, mid, high = (fan.percentiles[p] for p in (5.0, 50.0, 95.0))
    return MonteCarloFan(
        horizon_days=fan.horizon_days,
        n_paths=fan.n_paths,
        n_train=fan.n_train,
        mu=fan.mu,
        sigma=fan.sigma,
        last_close=fan.last_close,
        bands=[
            MonteCarloBand(day=int(day), lower=float(lo), median=float(md), upper=float(hi))
            for day, lo, md, hi in zip(fan.days, low, mid, high, strict=True)
        ],
    )


def _macro_overlay(session: Session, sector: str) -> MacroOverlay | None:
    sensitivities = macro.SECTOR_COMMODITY_SENSITIVITY.get(sector)
    if not sensitivities:
        return None
    components = []
    moves: dict[str, float] = {}
    for series, sensitivity in sensitivities.items():
        history = persistence.read_macro_series(
            session, series, as_of=date.today(), lookback_days=_MACRO_OVERLAY_DAYS
        )
        change = macro.pct_change(history)
        if change is not None:
            moves[series] = change
        components.append(
            MacroOverlayComponent(driver=series, sensitivity=float(sensitivity), move=change)
        )
    if not moves:
        return None
    return MacroOverlay(
        sector=sector,
        adjustment=macro.commodity_overlay_adjustment(sector, moves),
        components=components,
    )


# --------------------------------------------------------------------------- #
# Market-level data
# --------------------------------------------------------------------------- #


@app.get("/api/sectors/rotation", response_model=list[SectorStrength], tags=["market"])
def sector_rotation(
    lookback_days: int = Query(21, ge=5, le=252),
    session: Session = Depends(db_session),
) -> list[SectorStrength]:
    """Which sectors money has rotated into over the lookback window (Section 7.1).

    Measured against an **equal-weight** proxy for the market, deliberately: each
    sector's strength here is an equal-weight average of its members, so an
    equal-weight market keeps both sides of "relative to the market" on one
    weighting scheme. (Beta is the opposite case and regresses against the
    stored `^GSPC` index.) Sorted strongest first, so the top row is where money
    has been going.
    """
    panel = persistence.read_adj_close_panel(
        session,
        start=date.today() - timedelta(days=_ROTATION_PANEL_DAYS),
        end=date.today(),
    )
    if panel.empty or panel.shape[1] < 2:
        return []
    benchmark = risk.equal_weight_market_returns(panel)
    if benchmark.empty:
        return []

    universe = persistence.read_ticker_universe(session)
    sectors = {
        row.symbol: row.sector for row in universe.itertuples() if isinstance(row.sector, str)
    }
    counts: dict[str, int] = {}
    for symbol in panel.columns:
        sector = sectors.get(symbol)
        if sector:
            counts[sector] = counts.get(sector, 0) + 1

    rotation = technical.compute_sector_rotation(
        {column: panel[column].dropna() for column in panel.columns},
        sectors,
        # `compute_relative_strength` wants a price *level*, not returns.
        (1.0 + benchmark).cumprod(),
        lookback_days=lookback_days,
    )
    return [
        SectorStrength(
            sector=str(row.sector),
            relative_return=float(row.relative_strength_change_pct),
            n_symbols=counts.get(str(row.sector), 0),
        )
        for row in rotation.itertuples()
    ]


@app.get("/api/regime", response_model=list[RegimePoint], tags=["market"])
def regime(
    limit: int = Query(90, ge=1, le=730),
    session: Session = Depends(db_session),
) -> list[RegimePoint]:
    """Market Regime Index history, oldest first (Sections 5, 12)."""
    frame = persistence.read_recent_market_regime(session, limit=limit)
    return [RegimePoint(**row) for row in _rows(frame)]


@app.get("/api/news", response_model=list[NewsItem], tags=["market"])
def market_news(
    limit: int = Query(8, ge=1, le=50),
    session: Session = Depends(db_session),
) -> list[NewsItem]:
    """Recent Tier-2/3 industry and macro stories (Section 12's dashboard panel)."""
    frame = persistence.read_market_moving_news(session, limit=limit)
    return [NewsItem(**row) for row in _rows(frame)]


@app.get("/api/backtest", response_model=list[BacktestRun], tags=["market"])
def backtest(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(db_session),
) -> list[BacktestRun]:
    """Stored strategy backtest runs, newest first, with bootstrap CIs (Section 7.6).

    Each run carries its own fractional-Kelly position size, computed here from
    the run's realized win rate and payoff ratio. Deriving it server-side keeps
    one implementation of Section 27's sizing rule: the Streamlit Track Record
    page calls `kelly_position_fraction` directly, and a second copy in
    TypeScript could quietly disagree about how much to bet.
    """
    frame = persistence.read_backtest_history(session, limit=limit)
    runs = []
    for row in _rows(frame):
        run = BacktestRun(**row)
        run.kelly_fraction = kelly_position_fraction(run.win_rate, run.payoff_ratio)
        runs.append(run)
    return runs


@app.get("/api/prices/{symbol}", response_model=list[PriceBar], tags=["stocks"])
def prices(
    symbol: str,
    lookback_days: int = Query(400, ge=5, le=3650),
    session: Session = Depends(db_session),
) -> list[PriceBar]:
    """One symbol's OHLCV bars, oldest first."""
    frame = persistence.read_symbol_ohlcv(
        session, symbol.strip().upper(), lookback_days=lookback_days
    )
    return [PriceBar(**row) for row in _rows(frame)]


@app.get("/api/prices/{symbol}/range", response_model=dict, tags=["stocks"])
def price_range(
    symbol: str,
    days: int = Query(30, ge=1, le=3650),
    session: Session = Depends(db_session),
) -> dict[str, Any]:
    """First/last close over a window — the cheap call for a sparkline or a badge."""
    frame = persistence.read_symbol_ohlcv(session, symbol.strip().upper(), lookback_days=days)
    if frame.empty:
        return {"symbol": symbol.upper(), "start": None, "end": None, "change": None}
    first, last = float(frame["close"].iloc[0]), float(frame["close"].iloc[-1])
    return {
        "symbol": symbol.upper(),
        "start": first,
        "end": last,
        "change": (last / first - 1.0) if first > 0 else None,
        "from_date": frame["date"].iloc[0],
        "to_date": frame["date"].iloc[-1],
        "window_days": days,
        "as_of": date.today() - timedelta(days=0),
    }
