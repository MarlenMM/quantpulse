"""Composite scoring & ranking -- the project's core methodology (Section 7.5).

This is where the seven independent category signals become one ranking. It is
deliberately the most carefully-documented module in the repo, because (per
Section 21) this is exactly the place a subtle normalization or data-leakage
mistake silently produces wrong numbers rather than an error.

The pipeline, matching Section 7.5's numbered steps:

1. **Raw sub-score per category.** For the categories with a dedicated scorer,
   that module produces the raw score (`fundamental.score_fundamentals`,
   `analyst_consensus.score_analyst_consensus`, `smart_money.
   compute_smart_money_score`). The remaining categories -- technical,
   momentum/risk-adjusted, Tier-1 sentiment, and the Tier-2 industry tilt --
   are derived here from their underlying data (`score_technical`,
   `score_momentum`, sentiment polarity, `tier2_thematic_tilt`).

2. **Normalize to a 0-100 percentile within the peer universe.** Every
   category is rank-percentiled across the scored universe, EXCEPT fundamental,
   which arrives already sector-relative (Section 7.5 step 2's "optionally
   sector-relative for fundamentals") and is used as-is. Higher always means
   better; a category with no usable data for a symbol stays missing (NaN) and
   simply drops out of that symbol's weighting rather than counting as a zero.

3. **Weighted composite**, using an `InvestorProfile`'s weights (Section 23),
   renormalized over whichever categories had data -- the same coverage
   discipline as `fundamental.py`/`smart_money.py`, so a thinly-covered stock
   isn't penalized with phantom zeros. `data_confidence` (Section 7.5 step 6)
   is the fraction of category weight that had data.

4. **Rating**, default relative (peer-ranked, Section 7.5 step 4): top 10% Strong
   Buy, next 20% Buy, middle 40% Hold, next 20% Sell, bottom 10% Strong Sell.
   An absolute-threshold mode is also supported. In a risk-off market the
   Market Regime Index (Tier 3, Section 7.3) tightens the Strong-Buy cutoff so
   fewer are handed out -- the plan's "market-wide dampening filter."

**On Tier 3 / the Market Regime Index.** A market-wide value is the *same* for
every stock, so after a relative (percentile) normalization it cancels out and
cannot differentiate names -- normalizing it per-stock would be a no-op dressed
up as a signal. So Tier 3 does not enter the per-stock `industry_macro`
sub-score (that captures only the Tier-2 tilt, which genuinely varies by stock);
instead it acts where a market-wide signal belongs, on the rating step, exactly
as Section 7.3 describes. This is the honest reading of Section 22's "don't
treat a relative ranking as an absolute judgment."

**Point-in-time (Section 7.5 step 5).** Every input this module scores must be
data that was actually available as-of the scoring date. This module is pure --
it scores whatever frames it's handed -- so the point-in-time guarantee is the
caller's job (the nightly refresh reads only rows dated <= the as-of date); the
technical/momentum scorers additionally never read a bar past the frame's end.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpulse.analysis import technical
from quantpulse.analysis.investor_profiles import CATEGORIES, InvestorProfile, get_profile

# Column name each category writes to `composite_scores` (Section 13).
CATEGORY_SCORE_COLUMNS: dict[str, str] = {category: f"{category}_score" for category in CATEGORIES}

RATINGS = ("strong_buy", "buy", "hold", "sell", "strong_sell")

# Relative-rating percentile cutoffs (Section 7.5 step 4): top 10% / next 20% /
# middle 40% / next 20% / bottom 10%, expressed as the lower percentile bound
# of each rating.
_RELATIVE_CUTOFFS = ((90.0, "strong_buy"), (70.0, "buy"), (30.0, "hold"), (10.0, "sell"))
# Absolute-mode composite-score cutoffs -- an illustrative fixed bar for the
# alternative to peer-ranking (Section 7.5 step 4), tunable like the weights.
_ABSOLUTE_CUTOFFS = ((75.0, "strong_buy"), (60.0, "buy"), (40.0, "hold"), (25.0, "sell"))

# How far a fully risk-off regime lifts the Strong-Buy percentile cutoff (from
# the top 10% toward the top 5%) -- the modest market-wide dampening filter
# (Section 7.3 Tier 3). Neutral/​risk-on regimes leave the cutoffs untouched.
_MAX_REGIME_STRONG_BUY_LIFT = 5.0
_REGIME_NEUTRAL = 50.0

# Trailing windows (trading days) for the derived technical/momentum scorers.
_MOMENTUM_WINDOW = 126  # ~6 months of risk-adjusted trailing return
_MIN_MOMENTUM_BARS = 21  # need at least ~1 month to say anything

# Technical-signal saturation bands: a move of this size reads as a maxed-out
# (+/-1) signal before averaging the component tilts.
_TREND_LONG_FULL = 0.20  # 20% above/below the 200-DMA
_TREND_MID_FULL = 0.10  # 10% above/below the 50-DMA
_ADX_FULL_STRENGTH = 25.0  # a "trending" ADX


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


# --------------------------------------------------------------------------- #
# Derived category raw scorers (the categories without a dedicated module)
# --------------------------------------------------------------------------- #


def score_technical(prices: pd.DataFrame) -> float | None:
    """A single 0-100 technical bullishness score from `prices` (Section 7.1 -> 7.5).

    Averages a handful of transparent, non-redundant trend/momentum signals --
    price vs the 50- and 200-day MAs, the MACD histogram, RSI's tilt off 50, and
    the ADX-weighted directional index -- each mapped to [-1, 1], then rescaled
    to 0-100. Only the signals whose indicators are defined (enough history) are
    averaged, so a short-history name still scores on what it has; None if none
    are defined. A documented, tunable composite, not a fitted one.
    """
    if prices.empty:
        return None
    enriched = technical.compute_indicators(prices)
    row = enriched.iloc[-1]
    close = row.get("close")
    if close is None or pd.isna(close) or close <= 0:
        return None

    signals: list[float] = []
    if pd.notna(row.get("sma_200")) and row["sma_200"] > 0:
        signals.append(_clip_unit((close / row["sma_200"] - 1.0) / _TREND_LONG_FULL))
    if pd.notna(row.get("sma_50")) and row["sma_50"] > 0:
        signals.append(_clip_unit((close / row["sma_50"] - 1.0) / _TREND_MID_FULL))
    atr = row.get("atr_14")
    if pd.notna(row.get("macd_hist")) and pd.notna(atr) and atr > 0:
        signals.append(_clip_unit(float(row["macd_hist"]) / atr))
    if pd.notna(row.get("rsi_14")):
        signals.append(_clip_unit((float(row["rsi_14"]) - 50.0) / 50.0))
    plus_di, minus_di, adx = row.get("plus_di_14"), row.get("minus_di_14"), row.get("adx_14")
    if pd.notna(plus_di) and pd.notna(minus_di) and pd.notna(adx):
        di_sum = float(plus_di) + float(minus_di)
        if di_sum > 0:
            direction = (float(plus_di) - float(minus_di)) / di_sum
            strength = min(float(adx) / _ADX_FULL_STRENGTH, 1.0)
            signals.append(_clip_unit(direction * strength))

    if not signals:
        return None
    return 50.0 + 50.0 * (sum(signals) / len(signals))


def score_momentum(prices: pd.DataFrame, *, prefer_low_volatility: bool = False) -> float | None:
    """Trailing risk-adjusted momentum from `prices` (Section 7.5's momentum category).

    Risk-adjusted trailing return -- mean daily return divided by daily
    volatility over the trailing window (a Sharpe-like measure; the ranking is
    scale-invariant so annualization is unnecessary). Returned raw (any real
    number, higher = stronger), to be percentile-normalized alongside the other
    categories. `prices` must be date-ordered with a `close` column; only bars
    up to the frame's end are read (point-in-time).

    `prefer_low_volatility` (the conservative profile, Section 23) instead
    scores toward *lower* trailing volatility -- returning negative volatility
    so a calmer name ranks higher -- rather than raw momentum.
    """
    if "close" not in prices.columns:
        raise ValueError("prices is missing required column: 'close'")
    closes = prices["close"].dropna()
    if len(closes) <= _MIN_MOMENTUM_BARS:
        return None
    returns = closes.pct_change().dropna().iloc[-_MOMENTUM_WINDOW:]
    if len(returns) < _MIN_MOMENTUM_BARS:
        return None
    volatility = float(returns.std())
    if prefer_low_volatility:
        return -volatility  # lower volatility -> higher (less negative) score
    if volatility <= 0:
        return None  # a flat series has no risk-adjusted momentum to speak of
    return float(returns.mean()) / volatility


def sentiment_to_raw(polarity: float | None) -> float | None:
    """Tier-1 sentiment polarity ([-1, 1]) as a raw score, or None if absent.

    A pass-through: the polarity's ordering is all the percentile step needs, so
    no rescaling happens here (a [-1, 1] value ranks identically to its 0-100
    remap). Kept as a named seam so the sentiment category reads like the others.
    """
    if polarity is None or pd.isna(polarity):
        return None
    return float(polarity)


def tier2_thematic_tilt(
    symbol: str, tier2_events: pd.DataFrame, theme_members: dict[str, set[str]]
) -> float | None:
    """Average Tier-2 industry-news sentiment for the baskets `symbol` belongs to.

    `tier2_events` has `matched_theme` and `sentiment_score` columns (the
    stored Tier-2 `news_events`); `theme_members` maps a theme/basket name to
    its member symbols (from `thematic_baskets`). Returns the mean sentiment of
    events tagged to a basket that contains `symbol`, in [-1, 1], or None when
    the symbol is in no basket or none of its baskets have scored Tier-2 news --
    an honest "no industry signal," not a neutral zero (Section 22).

    This is the per-stock-varying part of the Industry/Macro category; the
    market-wide Tier-3 regime deliberately does not enter here (module
    docstring) -- it acts on the rating step instead.
    """
    themes = {theme for theme, members in theme_members.items() if symbol in members}
    if not themes or tier2_events.empty or "matched_theme" not in tier2_events.columns:
        return None
    relevant = tier2_events[tier2_events["matched_theme"].isin(themes)]
    scores = pd.to_numeric(relevant.get("sentiment_score"), errors="coerce").dropna()
    if scores.empty:
        return None
    return float(scores.mean())


# --------------------------------------------------------------------------- #
# Normalization + composite + rating
# --------------------------------------------------------------------------- #


def percentile_normalize(raw: pd.Series) -> pd.Series:
    """Rank-percentile `raw` to 0-100 within the (non-missing) peer universe.

    Higher raw -> higher percentile. Missing values stay missing (they don't
    count toward the ranking and don't receive a rank), so a category with no
    data for a symbol drops cleanly out of that symbol's weighting rather than
    being ranked as if it were the worst. A single non-missing value ranks at
    100 (it can't be discriminated against anyone) -- the same benign edge
    `fundamental.py` documents.
    """
    return raw.rank(pct=True) * 100.0


def _normalized_subscores(category_raw: pd.DataFrame) -> pd.DataFrame:
    """Per-category 0-100 sub-scores: fundamental as-is (sector-relative), rest percentiled."""
    normalized = pd.DataFrame(index=category_raw.index)
    for category in CATEGORIES:
        if category not in category_raw.columns:
            normalized[category] = np.nan
            continue
        column = category_raw[category]
        if category == "fundamental":
            # Already a sector-relative 0-100 percentile (Section 7.5 step 2).
            normalized[category] = pd.to_numeric(column, errors="coerce")
        else:
            normalized[category] = percentile_normalize(pd.to_numeric(column, errors="coerce"))
    return normalized


def _rating_from_percentile(percentile_rank: float, strong_buy_cutoff: float) -> str:
    if percentile_rank >= strong_buy_cutoff:
        return "strong_buy"
    for cutoff, rating in _RELATIVE_CUTOFFS[1:]:
        if percentile_rank >= cutoff:
            return rating
    return "strong_sell"


def _rating_from_absolute(composite_score: float) -> str:
    for cutoff, rating in _ABSOLUTE_CUTOFFS:
        if composite_score >= cutoff:
            return rating
    return "strong_sell"


def _strong_buy_cutoff(regime_score: float | None) -> float:
    """The Strong-Buy percentile cutoff, lifted in a risk-off regime (Section 7.3 Tier 3)."""
    base = _RELATIVE_CUTOFFS[0][0]
    if regime_score is None:
        return base
    # 0 when calm/neutral, up to 1 at a maximally risk-off regime.
    risk_off = max(0.0, (_REGIME_NEUTRAL - regime_score) / _REGIME_NEUTRAL)
    return base + _MAX_REGIME_STRONG_BUY_LIFT * risk_off


@dataclass(frozen=True)
class CompositeResult:
    """The scored universe (`scores`) and the profile/rating context it was built with."""

    scores: pd.DataFrame  # one row per ranked symbol, `composite_scores`-shaped
    profile: str
    rating_mode: str


def build_composite(
    category_raw: pd.DataFrame,
    *,
    profile: InvestorProfile | str | None = None,
    rating_mode: str = "relative",
    regime_score: float | None = None,
) -> CompositeResult:
    """Blend per-category raw scores into the ranked `composite_scores` frame (Section 7.5).

    `category_raw` is indexed by symbol with a column per category (a subset of
    `CATEGORIES`; fundamental's column is already a sector-relative 0-100 score,
    the rest are raw). Steps 2-4 of Section 7.5 happen here: normalize, weight
    (renormalized over available categories), rate. Symbols with usable data in
    no category are dropped -- they can't be scored, so they aren't ranked
    (rather than given a fake composite).

    `rating_mode` is "relative" (peer-ranked, the default) or "absolute"
    (fixed thresholds). `regime_score` (0-100, the Market Regime Index) tightens
    the relative Strong-Buy cutoff in a risk-off market.

    Returns a `CompositeResult` whose `scores` has columns: symbol, the seven
    `<category>_score` sub-scores, composite_score, percentile_rank, rating,
    data_confidence.
    """
    resolved = profile if isinstance(profile, InvestorProfile) else get_profile(profile)
    if rating_mode not in ("relative", "absolute"):
        raise ValueError(f"rating_mode must be 'relative' or 'absolute', got {rating_mode!r}")

    normalized = _normalized_subscores(category_raw)
    weights = pd.Series(resolved.weights)

    present = normalized.notna()
    available_weight = present.mul(weights, axis=1).sum(axis=1)
    weighted_sum = normalized.fillna(0.0).mul(weights, axis=1).sum(axis=1)

    scored = available_weight > 0
    composite = pd.Series(np.nan, index=normalized.index)
    composite[scored] = weighted_sum[scored] / available_weight[scored]
    data_confidence = available_weight * 100.0  # weights sum to 1.0

    result = normalized.copy()
    result.insert(0, "symbol", result.index)
    result["composite_score"] = composite
    result["data_confidence"] = data_confidence
    result = result[scored].copy()

    if result.empty:
        result["percentile_rank"] = pd.Series(dtype=float)
        result["rating"] = pd.Series(dtype=str)
    else:
        percentile_rank = result["composite_score"].rank(pct=True) * 100.0
        result["percentile_rank"] = percentile_rank
        if rating_mode == "relative":
            cutoff = _strong_buy_cutoff(regime_score)
            result["rating"] = percentile_rank.apply(lambda pr: _rating_from_percentile(pr, cutoff))
        else:
            result["rating"] = result["composite_score"].apply(_rating_from_absolute)

    ordered_columns = (
        ["symbol"]
        + [CATEGORY_SCORE_COLUMNS[c] for c in CATEGORIES]
        + ["composite_score", "percentile_rank", "rating", "data_confidence"]
    )
    result = result.rename(columns=CATEGORY_SCORE_COLUMNS)
    result = result[ordered_columns].sort_values("composite_score", ascending=False)
    result = result.reset_index(drop=True)
    return CompositeResult(scores=result, profile=resolved.name, rating_mode=rating_mode)
