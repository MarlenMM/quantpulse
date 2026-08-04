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

from quantpulse.analysis import scoring
from quantpulse.api.schemas import (
    AnalystConsensusModel,
    BacktestRun,
    ForecastRow,
    GlossaryTerm,
    HealthResponse,
    NewsItem,
    PatternRow,
    PriceBar,
    RatingChange,
    RegimePoint,
    ScreenerResponse,
    ScreenerRow,
    StockDetail,
    TickerSummary,
)
from quantpulse.glossary import TERMS
from quantpulse.storage import persistence
from quantpulse.storage.db import get_session

# The Vite dev server's default origins. Only local development needs CORS at
# all -- in production the built SPA is served as static files from the same
# origin, so no cross-origin request happens. Kept to an explicit allow-list
# rather than "*" so this never becomes a wide-open API by default.
_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

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
    """
    ticker = symbol.strip().upper()
    universe_frame = persistence.read_ticker_universe(session, active_only=False)
    match = universe_frame[universe_frame["symbol"] == ticker]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {ticker}")

    scores = persistence.read_screener_rows(session)
    score_rows = _rows(scores[scores["symbol"] == ticker]) if not scores.empty else []
    consensus = persistence.read_latest_analyst_consensus(session, ticker)

    return StockDetail(
        symbol=ticker,
        summary=TickerSummary(**_rows(match)[0]),
        score=ScreenerRow(**score_rows[0]) if score_rows else None,
        prices=[
            PriceBar(**row)
            for row in _rows(
                persistence.read_symbol_ohlcv(session, ticker, lookback_days=lookback_days)
            )
        ],
        forecasts=[
            ForecastRow(**row) for row in _rows(persistence.read_symbol_forecasts(session, ticker))
        ],
        patterns=[
            PatternRow(**row) for row in _rows(persistence.read_symbol_patterns(session, ticker))
        ],
        analyst_consensus=AnalystConsensusModel(**consensus) if consensus else None,
        news=[NewsItem(**row) for row in _rows(persistence.read_symbol_news(session, ticker))],
    )


# --------------------------------------------------------------------------- #
# Market-level data
# --------------------------------------------------------------------------- #


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
    """Stored strategy backtest runs, newest first, with bootstrap CIs (Section 7.6)."""
    frame = persistence.read_backtest_history(session, limit=limit)
    return [BacktestRun(**row) for row in _rows(frame)]


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
