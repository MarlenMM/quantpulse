"""Market Regime Index — the "build-your-own Fear/Greed" composite (Section 5, 7.3 Tier 3, 28).

Section 5 argues for computing an in-house market-sentiment index from data the
project already has for free, rather than scraping a paywalled one. This module
is that index: it blends four well-established, independently-sourced market-
wide signals into a single 0-100 `regime_score` (higher = risk-on / healthier
tape) and a coarse `regime_label` (risk_on / neutral / risk_off):

- **VIX** (Section 5) -- the market's expected volatility. Scored as calm =
  100 - percentile(current VIX within its own trailing history), so a VIX at
  the top of its recent range reads as maximally risk-off. Falls back to a
  fixed level band when too little history is stored to rank against.
- **Breadth** (Section 5) -- the share of the universe trading above its
  200-day moving average. Already a 0-100 risk-on reading as-is.
- **Macro news tone** (Section 7.3 Tier 3) -- GDELT's average tone across broad
  economic/political themes, mapped from its roughly [-10, +10] range to 0-100.
- **Yield-curve spread** (Section 28) -- the 10Y-2Y spread; a flat/inverted
  curve pulls the regime toward risk-off.

The blend renormalizes over whichever inputs are actually available (the same
coverage discipline as `fundamental.py` / `smart_money.py`), so a missing VIX
or empty tone series degrades gracefully instead of forcing a false neutral.

Tier 3 feeds the Market Regime Index, which in turn acts as a market-wide
dampening filter on the screener (Section 7.3): in a risk-off regime the
composite scorer can moderate how many Strong Buys it hands out, the way a
human analyst discounts individual stock-picking in a broadly stressed market.
That consumption is Phase 6's job; this module only produces the index.
"""

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

# Blend weights (must sum to 1.0). VIX and breadth are the two most direct,
# highest-frequency risk gauges, so they carry the most; the yield curve is a
# slower structural signal; macro news tone is the noisiest, so it carries the
# least. A documented, tunable starting point, not fit to data (Section 22's
# don't-overfit-the-weights caution applies here too).
_REGIME_WEIGHTS: dict[str, float] = {
    "vix": 0.35,
    "breadth": 0.30,
    "yield_curve": 0.20,
    "tone": 0.15,
}

# VIX-level fallback band (used only when too little VIX history is stored to
# take a real percentile): ~12 reads as calm, ~40 as panic, linear between.
_VIX_CALM_LEVEL = 12.0
_VIX_PANIC_LEVEL = 40.0
_MIN_VIX_HISTORY_FOR_PERCENTILE = 30

# GDELT tone is roughly [-10, +10]; map that span onto 0-100.
_TONE_MIN, _TONE_MAX = -10.0, 10.0

# Yield-curve spread (percentage points) mapped onto 0-100: a ~1pt-inverted
# curve reads as maximally risk-off, a ~1.5pt-steep curve as maximally risk-on.
_CURVE_MIN_SPREAD, _CURVE_MAX_SPREAD = -1.0, 1.5

# Label cutoffs on the 0-100 composite.
_RISK_ON_AT = 60.0
_RISK_OFF_AT = 35.0

_MA_WINDOW = 200


@dataclass(frozen=True)
class MarketRegimeReading:
    """One day's Market Regime Index, shaped to the `market_regime` table (Section 13)."""

    date: date
    vix_level: float | None
    breadth_pct_above_200dma: float | None
    macro_news_tone: float | None
    yield_curve_spread: float | None
    regime_score: float | None
    regime_label: str | None


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def compute_breadth(
    price_history: pd.DataFrame, as_of: date, *, ma_window: int = _MA_WINDOW
) -> float | None:
    """Percent of symbols in `price_history` trading above their `ma_window`-day MA at `as_of`.

    `price_history` is long-form `symbol, date, adj_close` (as
    `persistence.read_active_price_history` returns). For each symbol with at
    least `ma_window` bars up to and including `as_of`, compares its latest
    close to the mean of its trailing `ma_window` closes. Returns the share (0-
    100) above, or None if no symbol has enough history -- an honest "can't
    measure breadth yet," not a misleading 0 or 50.
    """
    if price_history.empty:
        return None

    # `date` may arrive as datetime64 (a synthetic frame) or Python `date`
    # objects (a SQLAlchemy read); normalize before comparing/sorting so both
    # paths behave identically.
    frame = price_history.assign(date=pd.to_datetime(price_history["date"]))
    frame = frame[frame["date"] <= pd.Timestamp(as_of)]
    evaluable = 0
    above = 0
    for _symbol, group in frame.groupby("symbol"):
        closes = group.sort_values("date")["adj_close"].dropna()
        if len(closes) < ma_window:
            continue
        evaluable += 1
        moving_average = closes.iloc[-ma_window:].mean()
        if closes.iloc[-1] > moving_average:
            above += 1

    if evaluable == 0:
        return None
    return 100.0 * above / evaluable


def vix_calm_score(vix_level: float | None, vix_history: list[float]) -> float | None:
    """0-100 "calm" score from the VIX (100 = calmest), or None if `vix_level` is missing.

    With enough trailing history, calm = 100 - percentile-rank of the current
    level within that history (a VIX at the top of its recent range → 0 calm).
    With too little history to rank meaningfully, falls back to a fixed level
    band between `_VIX_CALM_LEVEL` and `_VIX_PANIC_LEVEL`.
    """
    if vix_level is None:
        return None

    if len(vix_history) >= _MIN_VIX_HISTORY_FOR_PERCENTILE:
        at_or_below = sum(1 for v in vix_history if v <= vix_level)
        percentile = 100.0 * at_or_below / len(vix_history)
        return _clip(100.0 - percentile)

    span = _VIX_PANIC_LEVEL - _VIX_CALM_LEVEL
    calm = 100.0 * (_VIX_PANIC_LEVEL - vix_level) / span
    return _clip(calm)


def tone_score(macro_tone: float | None) -> float | None:
    """Map GDELT macro tone (~[-10, 10]) onto 0-100, or None if absent."""
    if macro_tone is None:
        return None
    return _clip(100.0 * (macro_tone - _TONE_MIN) / (_TONE_MAX - _TONE_MIN))


def yield_curve_score(spread: float | None) -> float | None:
    """Map the 10Y-2Y spread onto 0-100 (inverted → risk-off), or None if absent."""
    if spread is None:
        return None
    span = _CURVE_MAX_SPREAD - _CURVE_MIN_SPREAD
    return _clip(100.0 * (spread - _CURVE_MIN_SPREAD) / span)


def _label_for(score: float) -> str:
    if score >= _RISK_ON_AT:
        return "risk_on"
    if score <= _RISK_OFF_AT:
        return "risk_off"
    return "neutral"


def compute_market_regime(
    as_of: date,
    *,
    vix_level: float | None,
    vix_history: list[float],
    breadth_pct: float | None,
    macro_tone: float | None,
    yield_curve_spread_value: float | None,
) -> MarketRegimeReading:
    """Blend the four market-wide signals into one day's Market Regime Index.

    Each sub-signal is scored to 0-100 (higher = more risk-on), then combined
    as a weighted average renormalized over whichever sub-signals were
    available (`_REGIME_WEIGHTS`). `regime_score`/`regime_label` are None when
    *no* sub-signal was available. The raw inputs (`vix_level`, `breadth_pct`,
    `macro_tone`, `yield_curve_spread_value`) are echoed onto the reading so the
    stored row records exactly what drove the score.
    """
    sub_scores: dict[str, float | None] = {
        "vix": vix_calm_score(vix_level, vix_history),
        "breadth": _clip(breadth_pct) if breadth_pct is not None else None,
        "yield_curve": yield_curve_score(yield_curve_spread_value),
        "tone": tone_score(macro_tone),
    }

    available_weight = sum(
        _REGIME_WEIGHTS[name] for name, score in sub_scores.items() if score is not None
    )
    if available_weight <= 0:
        regime_score: float | None = None
        regime_label: str | None = None
    else:
        weighted_sum = sum(
            _REGIME_WEIGHTS[name] * score for name, score in sub_scores.items() if score is not None
        )
        regime_score = weighted_sum / available_weight
        regime_label = _label_for(regime_score)

    return MarketRegimeReading(
        date=as_of,
        vix_level=vix_level,
        breadth_pct_above_200dma=breadth_pct,
        macro_news_tone=macro_tone,
        yield_curve_spread=yield_curve_spread_value,
        regime_score=regime_score,
        regime_label=regime_label,
    )


def regime_to_record(reading: MarketRegimeReading) -> dict[str, object]:
    """The `market_regime`-table row dict for `reading` (Section 13)."""
    return asdict(reading)
