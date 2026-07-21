"""Analyst consensus aggregation + estimate-revision trend (Section 7.4).

Wall Street rating counts and price targets become both an input to the
composite score and a comparison point in the UI ("our algorithm says X,
Wall Street analysts say Y -- here's where they agree/disagree and why").

The refinement Section 7.4 insists on building in from day one rather than
bolting on later: the *trend* of analyst estimates matters more than their
static level -- a stock where targets have been quietly rising over the
trailing quarter is a meaningfully different signal from one at the same
current level but drifting down. Since `analyst_consensus` is stored
point-in-time and never overwritten (Section 6.8), that trend comes
directly from its history, not a separate mechanism.
"""

import numpy as np
import pandas as pd

_HISTORY_COLUMNS = (
    "as_of_date",
    "strong_buy",
    "buy",
    "hold",
    "sell",
    "strong_sell",
    "mean_price_target",
)

# An even 0/25/50/75/100 spacing across the five-point Wall Street scale --
# "Hold" sits at the neutral midpoint, matching how it's actually used.
_RATING_WEIGHTS = {"strong_buy": 100.0, "buy": 75.0, "hold": 50.0, "sell": 25.0, "strong_sell": 0.0}

DEFAULT_TREND_LOOKBACK_DAYS = 91  # ~1 calendar quarter, per Section 7.4's own phrasing

# How much of the trend (in rating-score points, over the lookback window)
# to fold into the level. Bounded and modest by design: the trend is a real,
# meaningfully different signal (Section 7.4), not a replacement for the
# current consensus itself.
_TREND_WEIGHT = 0.5


def compute_rating_score(
    strong_buy: float, buy: float, hold: float, sell: float, strong_sell: float
) -> float | None:
    """Weighted-average analyst rating, 0 (unanimous Strong Sell) to 100 (unanimous Strong Buy)."""
    counts = {
        "strong_buy": strong_buy or 0,
        "buy": buy or 0,
        "hold": hold or 0,
        "sell": sell or 0,
        "strong_sell": strong_sell or 0,
    }
    total = sum(counts.values())
    if total <= 0:
        return None
    return sum(counts[k] * _RATING_WEIGHTS[k] for k in counts) / total


def compute_price_target_upside(
    current_price: float | None, mean_price_target: float | None
) -> float | None:
    """% upside (or downside, if negative) of the mean analyst target over the current price."""
    if current_price is None or mean_price_target is None or current_price <= 0:
        return None
    return (mean_price_target - current_price) / current_price * 100


def _fit_line_endpoints(dates: pd.Series, values: pd.Series) -> tuple[float, float] | None:
    """Fit a line to (date, value) and return (fitted_start, fitted_end), or None if unfittable.

    Smoothing over every available point in the window (rather than just
    differencing the two endpoint snapshots) is more robust to a single noisy
    or stale data point -- while staying, per Section 7.4, a "simple slope."
    """
    valid = values.notna()
    d = pd.to_datetime(dates[valid])
    v = values[valid].to_numpy(dtype=float)
    if len(v) < 2:
        return None
    offsets = (d - d.min()).dt.days.to_numpy(dtype=float)
    span = float(offsets.max())
    if span == 0:
        return None
    slope, intercept = np.polyfit(offsets, v, 1)
    return float(intercept), float(intercept + slope * span)


def score_analyst_consensus(
    history: pd.DataFrame,
    current_price: float | None = None,
    lookback_days: int = DEFAULT_TREND_LOOKBACK_DAYS,
) -> dict[str, float | None]:
    """One symbol's Wall Street analyst score, from its full point-in-time `analyst_consensus`
    history.

    `history` must have `as_of_date` plus the rating-count and
    `mean_price_target` columns -- every point-in-time snapshot available for
    one symbol, in any order. The most recent row is treated as "today"; the
    trend is fit over the trailing `lookback_days` (~1 quarter) of snapshots
    relative to it.

    Returns a dict with:
    - `rating_score` (0-100, current snapshot; None if no analysts cover it)
    - `price_target_upside_pct` (current snapshot; None without `current_price`
      or a target)
    - `rating_score_trend` (points moved over the window; None if fewer than
      2 usable snapshots)
    - `price_target_trend_pct` (% moved over the window; same requirement)
    - `analyst_score` (0-100: `rating_score` nudged by a bounded fraction of
      its trend -- None if `rating_score` itself is None)
    """
    missing = [c for c in _HISTORY_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing required column(s): {missing}")

    result: dict[str, float | None] = {
        "rating_score": None,
        "price_target_upside_pct": None,
        "rating_score_trend": None,
        "price_target_trend_pct": None,
        "analyst_score": None,
    }
    if history.empty:
        return result

    ordered = history.assign(as_of_date=pd.to_datetime(history["as_of_date"])).sort_values(
        "as_of_date"
    )
    latest = ordered.iloc[-1]

    rating_score = compute_rating_score(
        latest["strong_buy"], latest["buy"], latest["hold"], latest["sell"], latest["strong_sell"]
    )
    result["rating_score"] = rating_score
    result["price_target_upside_pct"] = compute_price_target_upside(
        current_price, latest["mean_price_target"]
    )

    window_start = latest["as_of_date"] - pd.Timedelta(days=lookback_days)
    window = ordered[ordered["as_of_date"] >= window_start]

    rating_series = window.apply(
        lambda r: compute_rating_score(
            r["strong_buy"], r["buy"], r["hold"], r["sell"], r["strong_sell"]
        ),
        axis=1,
    )
    rating_fit = _fit_line_endpoints(window["as_of_date"], rating_series)
    if rating_fit is not None:
        result["rating_score_trend"] = rating_fit[1] - rating_fit[0]

    price_fit = _fit_line_endpoints(window["as_of_date"], window["mean_price_target"])
    if price_fit is not None and price_fit[0] != 0:
        result["price_target_trend_pct"] = (price_fit[1] - price_fit[0]) / price_fit[0] * 100

    if rating_score is not None:
        trend = result["rating_score_trend"] or 0.0
        result["analyst_score"] = float(np.clip(rating_score + _TREND_WEIGHT * trend, 0.0, 100.0))

    return result
