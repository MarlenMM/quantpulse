"""Analyse any listed symbol on request, rather than only the ones scored nightly.

The nightly job can afford to fetch and score a few hundred names. The catalogue
knows about thirteen thousand. This is what closes the gap: given a symbol the
pipeline has never touched, fetch what is needed right now, run it through the
*same* analysis functions the nightly job uses, and return the result in a few
seconds.

**Why this lives at the top level and not in `analysis/`.** `analysis/` imports
nothing outside its own package -- it is pure functions over frames, which is
what makes it testable without a network or a database. The ingestion-to-analysis
wiring lives outside it, in `scripts/refresh_data.py` for the batch path. This is
the same wiring for the interactive path, so it belongs beside that, not inside
the layer it is wiring together.

**What it deliberately does not do.**

* It does not write anything. An on-demand result is not stored, does not enter
  the ranking, and cannot end up in the committed demo database -- which is the
  only reason a 13,000-symbol catalogue is affordable at all.
* It does not claim a relative rating. "Strong Buy" in this app means top decile
  of a scored universe, and a symbol that was never in that universe has no
  decile. The rating here is `build_composite`'s **absolute** mode -- fixed bars,
  no peer group -- and the percentile is reported separately as a *placement*
  against the stored distribution, clearly labelled as such.
* It does not fabricate what it could not fetch. A category with no data is
  absent, exactly as in the nightly path, and `data_confidence` reports the
  coverage it actually got.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from quantpulse.analysis import analyst_consensus, forecasting, fundamental, patterns, scoring
from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.ingestion import yfinance_client

logger = logging.getLogger(__name__)

# Two years: enough for the 63-day forecast horizon's 3x-history floor and a
# 200-day moving average, without making the fetch slow enough to feel like a
# page load. The 252-day horizon needs three years and is deliberately not
# offered here -- see `MAX_HORIZON_DAYS`.
DEFAULT_PERIOD = "2y"

# Horizons this path can support honestly. `forecasting.min_bars_for_horizon`
# requires 3x the horizon in history, so a 252-day forecast from two years of
# data is exactly the over-reach that guard exists to prevent.
HORIZONS = (5, 20, 63)

# Fetches that are independent of each other, run together because three
# sequential round trips is the difference between a snappy lookup and a slow
# one. Bounded at 3: this is one user waiting, not a batch job.
_FETCH_WORKERS = 3


@dataclass(frozen=True)
class OnDemandAnalysis:
    """One symbol, analysed just now. Every field is honest about its own absence."""

    symbol: str
    name: str | None
    sector: str | None
    as_of: date
    computed_at: datetime
    prices: pd.DataFrame
    category_raw: dict[str, float | None]
    composite_score: float | None
    absolute_rating: str | None
    data_confidence: float | None
    patterns: list[Any]
    forecasts: list[Any]
    fundamentals: dict[str, Any] | None
    analyst: dict[str, Any] | None
    # Placement against the stored ranking, when the caller supplied it.
    percentile_vs_ranked: float | None = None
    ranked_universe_size: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def covered_categories(self) -> list[str]:
        return [name for name, value in self.category_raw.items() if value is not None]

    def patterns_frame(self) -> pd.DataFrame:
        """Detected formations as a table, newest first.

        `detect_all_chart_patterns` returns `ChartPattern` objects, not a frame.
        Keeping them as objects here and converting only for display means the
        caller that wants the geometry still has it, and the caller that wants a
        table does not have to know the dataclass's field names.
        """
        if not self.patterns:
            return pd.DataFrame()
        frame = pd.DataFrame([vars(pattern) for pattern in self.patterns])
        if "end_date" in frame.columns:
            frame = frame.sort_values("end_date", ascending=False)
        return frame.reset_index(drop=True)


def _indicator_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLCV indexed by date, which is what the category scorers require.

    Not cosmetic: the indicator library's time-anchored calculations (VWAP is
    the one that bites) call `.to_period()` on the index, so a default
    RangeIndex raises rather than degrading. `refresh_data` builds exactly this
    shape before scoring, and says so in a comment; a live lookup has to build
    it too, or it fails on the first stock it is ever asked about.
    """
    return prices[["open", "high", "low", "close", "volume"]].set_axis(
        pd.DatetimeIndex(pd.to_datetime(prices["date"].to_numpy()))
    )


def _safe(label: str, fn: Any, notes: list[str]) -> Any:
    """Run one fetch, turning a failure into a recorded absence rather than a crash.

    A live lookup touches three independent endpoints. Any of them can be down,
    rate-limited, or simply have nothing for a small company -- and none of that
    is a reason to show the visitor an error page instead of the four things
    that did come back.
    """
    try:
        return fn()
    except Exception:
        logger.warning("on-demand: %s unavailable", label, exc_info=True)
        notes.append(f"{label} could not be fetched just now.")
        return None


def analyse(
    symbol: str,
    *,
    name: str | None = None,
    sector: str | None = None,
    period: str = DEFAULT_PERIOD,
    ranked_composites: pd.Series | None = None,
    sector_peers: pd.DataFrame | None = None,
    include_fundamentals: bool = True,
) -> OnDemandAnalysis | None:
    """Fetch and analyse `symbol` now. Returns None when it has no usable price history.

    `ranked_composites` is the stored universe's composite scores. When given,
    the result carries where this symbol would have placed among them -- which is
    the only defensible way to answer "is this good?" for a stock that was never
    ranked. Without it the score still stands on its own absolute bars.
    """
    symbol = symbol.strip().upper()
    notes: list[str] = []

    prices = _safe(
        "Price history", lambda: yfinance_client.fetch_price_history(symbol, period), notes
    )
    if prices is None or prices.empty:
        return None
    prices = prices.sort_values("date").reset_index(drop=True)

    fundamentals: dict[str, Any] | None = None
    analyst: dict[str, Any] | None = None
    if include_fundamentals:
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            futures = {
                "Fundamentals": pool.submit(yfinance_client.fetch_fundamentals, symbol),
                "Analyst estimates": pool.submit(yfinance_client.fetch_analyst_consensus, symbol),
            }
            fundamentals = _safe("Fundamentals", futures["Fundamentals"].result, notes)
            analyst = _safe("Analyst estimates", futures["Analyst estimates"].result, notes)

    category_raw = _score_categories(
        symbol,
        prices,
        fundamentals,
        analyst,
        sector=sector,
        sector_peers=sector_peers,
        notes=notes,
    )

    composite, rating, confidence = _absolute_composite(symbol, category_raw)

    detected = _safe("Chart patterns", lambda: patterns.detect_all_chart_patterns(prices), notes)
    forecasts = _forecasts(prices, notes)

    percentile, universe_size = _placement(composite, ranked_composites)

    return OnDemandAnalysis(
        symbol=symbol,
        name=name,
        sector=sector,
        as_of=pd.Timestamp(prices["date"].iloc[-1]).date(),
        computed_at=datetime.now(),
        prices=prices,
        category_raw=category_raw,
        composite_score=composite,
        absolute_rating=rating,
        data_confidence=confidence,
        patterns=list(detected) if detected else [],
        forecasts=forecasts,
        fundamentals=fundamentals,
        analyst=analyst,
        percentile_vs_ranked=percentile,
        ranked_universe_size=universe_size,
        notes=notes,
    )


def _score_categories(
    symbol: str,
    prices: pd.DataFrame,
    fundamentals: dict[str, Any] | None,
    analyst: dict[str, Any] | None,
    *,
    sector: str | None,
    sector_peers: pd.DataFrame | None,
    notes: list[str],
) -> dict[str, float | None]:
    """Per-category raw values, using the same scorers the nightly job uses.

    Sentiment, industry/macro and smart money are absent by construction: each
    needs either the news pipeline's model pass or a multi-day aggregate that
    cannot be produced from one live fetch. Saying so is the point -- a category
    that quietly scored zero would drag the composite down for a company whose
    only failing is that nobody ran a model over its headlines.
    """
    raw: dict[str, float | None] = dict.fromkeys(CATEGORIES, None)

    indexed = _indicator_frame(prices)
    raw["technical"] = scoring.score_technical(indexed)
    raw["momentum"] = scoring.score_momentum(indexed)

    if fundamentals:
        raw["fundamental"] = _fundamental_score(
            symbol, fundamentals, sector=sector, peers=sector_peers, notes=notes
        )

    if analyst:
        # `score_analyst_consensus` expects a point-in-time *history* and reads
        # the newest row as today, so a live snapshot has to carry a date. One
        # row means no trend, which the scorer already reports as None rather
        # than inventing a direction from a single observation -- the honest
        # outcome for a symbol whose estimates have never been stored.
        snapshot = pd.DataFrame([{**analyst, "as_of_date": prices["date"].iloc[-1]}])
        last_close = float(prices["close"].iloc[-1])
        scored_analyst = _safe(
            "Analyst score",
            lambda: analyst_consensus.score_analyst_consensus(snapshot, last_close),
            notes,
        )
        if scored_analyst:
            value = scored_analyst.get("analyst_score")
            raw["analyst"] = None if value is None or pd.isna(value) else float(value)

    missing = [c for c in ("sentiment", "industry_macro", "smart_money") if raw[c] is None]
    if missing:
        notes.append(
            "News sentiment, industry/macro and smart-money signals are not computed for a "
            "live lookup -- they need the nightly model pass and multi-day aggregates. They "
            "are left out of the score rather than counted as zero."
        )
    return raw


# Below this many sector peers, a percentile is not a ranking, it is an
# accident of who happened to be in the frame. `score_fundamentals` ranks within
# a sector group, so eight is already generous for a number presented as 0-100.
MIN_SECTOR_PEERS = 8


def _fundamental_score(
    symbol: str,
    fundamentals: dict[str, Any],
    *,
    sector: str | None,
    peers: pd.DataFrame | None,
    notes: list[str],
) -> float | None:
    """This company's fundamental score, ranked against real sector peers or not at all.

    `score_fundamentals` percentile-ranks within a sector group. Handed a frame
    containing only this one company, it therefore returns **100 every single
    time** -- a stock is always the best of itself. That is not a small
    inaccuracy: fundamentals carry the largest weight of any category, so a
    guaranteed 100 inflates every on-demand composite and every placement drawn
    from it, and it does so invisibly, because 100 is a perfectly plausible
    score.

    So the row is scored inside the stored fundamentals for its sector, and when
    there are not enough of those to rank against, the category is left out --
    the same treatment every other missing category gets.
    """
    if peers is None or peers.empty or sector is None:
        notes.append(
            "Fundamentals were fetched but not scored: a fundamental score is a rank against "
            "sector peers, and no stored peers were available to rank this company against."
        )
        return None

    cohort = peers[peers["sector"] == sector]
    if len(cohort) < MIN_SECTOR_PEERS:
        notes.append(
            f"Fundamentals were fetched but not scored: only {len(cohort)} stored "
            f"{sector} peers to rank against, below the {MIN_SECTOR_PEERS} needed for a "
            "percentile to mean anything."
        )
        return None

    row = {**fundamentals, "symbol": symbol, "sector": sector}
    frame = pd.concat([cohort, pd.DataFrame([row])], ignore_index=True)
    scored = _safe("Fundamental score", lambda: fundamental.score_fundamentals(frame), notes)
    if scored is None or scored.empty or "fundamental_score" not in scored.columns:
        return None

    mine = scored[scored["symbol"] == symbol]
    if mine.empty:
        return None
    value = mine["fundamental_score"].iloc[0]
    return None if pd.isna(value) else float(value)


def _absolute_composite(
    symbol: str, category_raw: dict[str, float | None]
) -> tuple[float | None, str | None, float | None]:
    """Blend into an absolute-mode composite, or return three Nones.

    Absolute mode, not relative, and that is the whole judgment call. A relative
    rating is a statement about a peer group -- "top decile of the scored
    universe" -- and this symbol is not in one. Rating it relatively against a
    single-row frame would make every on-demand stock a Strong Buy, since it
    would be the top decile of itself.
    """
    present = {name: value for name, value in category_raw.items() if value is not None}
    if not present:
        return None, None, None

    frame = pd.DataFrame([present], index=[symbol])
    result = scoring.build_composite(frame, rating_mode="absolute")
    if result.scores.empty:
        return None, None, None

    row = result.scores.iloc[0]
    return (
        float(row["composite_score"]),
        str(row["rating"]),
        float(row["data_confidence"]) if pd.notna(row["data_confidence"]) else None,
    )


def _forecasts(prices: pd.DataFrame, notes: list[str]) -> list[Any]:
    """Forecasts at the horizons two years of history can actually support."""
    produced: list[Any] = []
    for horizon in HORIZONS:
        result = _safe(
            f"{horizon}-day forecast",
            lambda h=horizon: forecasting.forecast_horizon(prices, h),
            notes,
        )
        if result:
            produced.extend(result)
    if not produced:
        notes.append(
            "No forecast: this symbol does not have enough price history for even the "
            "shortest horizon, which needs several times the horizon in past data."
        )
    return produced


def _placement(
    composite: float | None, ranked_composites: pd.Series | None
) -> tuple[float | None, int | None]:
    """Where this score would sit among the stored ranking.

    Reported as a placement rather than folded into the rating, because it is a
    weaker claim than a real percentile: this symbol was scored from whatever
    could be fetched in a few seconds, while the ranked universe was scored from
    a full nightly pipeline. Comparable enough to be informative, not comparable
    enough to be a ranking.
    """
    if composite is None or ranked_composites is None or ranked_composites.empty:
        return None, None
    values = ranked_composites.dropna()
    if values.empty:
        return None, None
    return float((values < composite).mean() * 100.0), int(len(values))
