"""Geometric chart-pattern detection (Section 7.1, the "harder, more interesting part").

Candlestick libraries detect 1-3 bar shapes; this module detects the multi-week
geometric formations they can't -- head-and-shoulders, double top/bottom,
triangles/wedges/channels, and cup-and-handle -- by reasoning about the geometry
of price *pivots* rather than individual candles.

Everything is built on one primitive: `find_pivots`, a zig-zag filter that
reduces a noisy price series to a clean, strictly-alternating sequence of swing
highs and lows, ignoring moves smaller than a threshold. Each detector then
encodes the classic geometric rules for its pattern over that pivot sequence,
and -- per Section 7.1 -- attaches a graded confidence (how closely the shape
matches the ideal) rather than a binary yes/no.

Design choices worth stating, because they shape every result here:

- **Pivots are computed on the closing price.** Using close (not intraday
  high/low) keeps the zig-zag deterministic and the geometry internally
  consistent; the small cost is that a "high" pivot sits at its bar's close,
  not its true intraday peak. This is the standard, defensible simplification
  for shape detection.
- **Detection is point-in-time honest.** Every function reads only the price
  data it is given. Called as-of date D (with data through D), the most recent
  pivot is provisional (the current swing) and all earlier pivots are already
  confirmed -- so a backtest that calls these with data up to each historical
  date sees exactly what a viewer would have seen then, with no look-ahead. A
  pattern's stored date is its final pivot's date (when its shape completed).
- **Confidence is a graded geometry score, gated by hard structural rules.**
  Structural facts (correct pivot alternation, head strictly highest, cup opens
  upward) are pass/fail gates; the returned confidence [0, 100] then grades the
  quality of the softer constraints (symmetry, flatness, prominence). A
  textbook-perfect shape scores ~100; a shape that only barely clears the gates
  scores near 0.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

PivotKind = Literal["high", "low"]
Direction = Literal["bullish", "bearish", "neutral"]

# --- Default tolerances (all scale-invariant fractions of price) ------------
DEFAULT_ZIGZAG_THRESHOLD = 0.05  # a swing must reverse >= 5% to register a pivot

# Head-and-shoulders
_SHOULDER_TOL = 0.06  # max |left - right shoulder| / avg to qualify
_NECKLINE_TOL = 0.06  # max |trough1 - trough2| / avg to qualify
_HEAD_PROMINENCE_MIN = 0.03  # head must clear the taller shoulder by >= 3%
_HEAD_PROMINENCE_TARGET = 0.10  # ...and clearing it by 10%+ is "ideal" for scoring

# Double top / bottom
_PEAK_TOL = 0.05  # max |peak1 - peak2| / avg to qualify
_RETRACE_MIN = 0.03  # the middle pullback must be >= 3% deep
_RETRACE_TARGET = 0.12  # ...and 12%+ is "ideal" for scoring

# Trendline (triangle / wedge / channel)
_TREND_WINDOW = 6  # consecutive pivots per formation window (=> 3 highs, 3 lows)
_TREND_FLAT_TOL = 0.03  # |line's fractional move across window| below this = "flat"
_TREND_CONVERGE_TOL = 0.25  # gap must shrink/grow >25% to be converging/diverging
_TREND_FIT_TARGET = 0.04  # per-point residual (frac of price) for a full fit score

# Cup-and-handle
_CUP_RIM_TOL = 0.06  # max |left rim - right rim| / avg
_CUP_MIN_DEPTH = 0.10  # cup must be >= 10% deep (rim to bottom)
_CUP_DEPTH_TARGET = 0.30  # ...30% is "ideal" for scoring
_CUP_HANDLE_MAX_RETRACE = 0.5  # handle pullback <= half the cup depth
_CUP_BOTTOM_CENTER_BAND = (0.25, 0.75)  # parabola vertex must sit in the middle half
_CUP_MIN_BOTTOM_DWELL = 0.35  # >= this fraction of cup bars must sit in its lower third
_CUP_BOTTOM_THIRD = 1 / 3  # "lower third" of the cup's price range


@dataclass(frozen=True)
class Pivot:
    """One swing point in the zig-zag reduction of a price series."""

    index: int  # positional index into the price series
    date: object  # the index label (e.g. a pandas Timestamp) at that position
    price: float
    kind: PivotKind


@dataclass(frozen=True)
class ChartPattern:
    """A detected geometric formation, with a graded confidence and key levels."""

    pattern_type: str
    direction: Direction
    confidence: float  # 0..100
    start_index: int  # positional price index of the first pivot
    end_index: int  # positional price index of the last pivot (completion)
    start_date: object
    end_date: object
    key_levels: dict[str, float] = field(default_factory=dict)


# --- Pivot detection --------------------------------------------------------


def find_pivots(
    prices: pd.DataFrame,
    threshold: float = DEFAULT_ZIGZAG_THRESHOLD,
    price_col: str = "close",
) -> list[Pivot]:
    """Reduce a price series to alternating swing highs/lows via a zig-zag filter.

    A new pivot is confirmed only once price reverses by at least `threshold`
    (a fraction, e.g. 0.05 = 5%) away from the running extreme; smaller wiggles
    are ignored. The returned list strictly alternates high/low. The final entry
    is the current, still-provisional swing extreme (it hasn't reversed yet) --
    which is what lets a *forming* pattern be detected as-of the latest bar.
    """
    if price_col not in prices.columns:
        raise ValueError(f"prices is missing required column: {price_col!r}")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    series = prices[price_col].to_numpy(dtype=float)
    labels = prices.index
    n = series.size
    pivots: list[Pivot] = []
    if n < 2:
        return pivots

    # Phase 1 (direction unknown): track the running high and low from the
    # start until their span first exceeds the threshold; the earlier of the
    # two extremes becomes the first pivot and fixes the initial direction.
    max_i, max_p = 0, series[0]
    min_i, min_p = 0, series[0]
    direction = 0  # +1 = currently rising (tracking a high), -1 = falling, 0 = unknown
    ext_i, ext_p = 0, series[0]

    for i in range(1, n):
        price = series[i]

        if direction == 0:
            if price > max_p:
                max_i, max_p = i, price
            if price < min_p:
                min_i, min_p = i, price
            if min_p > 0 and (max_p - min_p) / min_p >= threshold:
                if min_i < max_i:  # low came first -> now rising, low is pivot 0
                    pivots.append(Pivot(min_i, labels[min_i], min_p, "low"))
                    direction = 1
                    ext_i, ext_p = max_i, max_p
                else:  # high came first -> now falling
                    pivots.append(Pivot(max_i, labels[max_i], max_p, "high"))
                    direction = -1
                    ext_i, ext_p = min_i, min_p

        elif direction > 0:  # rising: extend the high, or reverse down
            if price > ext_p:
                ext_i, ext_p = i, price
            elif ext_p > 0 and (ext_p - price) / ext_p >= threshold:
                pivots.append(Pivot(ext_i, labels[ext_i], ext_p, "high"))
                direction = -1
                ext_i, ext_p = i, price

        else:  # falling: extend the low, or reverse up
            if price < ext_p:
                ext_i, ext_p = i, price
            elif ext_p > 0 and (price - ext_p) / ext_p >= threshold:
                pivots.append(Pivot(ext_i, labels[ext_i], ext_p, "low"))
                direction = 1
                ext_i, ext_p = i, price

    if direction > 0:
        pivots.append(Pivot(ext_i, labels[ext_i], ext_p, "high"))
    elif direction < 0:
        pivots.append(Pivot(ext_i, labels[ext_i], ext_p, "low"))

    return pivots


# --- Small geometry helpers -------------------------------------------------


def _rel_diff(a: float, b: float) -> float:
    """|a - b| relative to their average (a symmetric fractional difference)."""
    denom = (abs(a) + abs(b)) / 2
    return abs(a - b) / denom if denom else 0.0


def _score(deviation: float, tolerance: float) -> float:
    """Map a deviation to a [0, 1] quality score: 0 at/above tolerance, 1 at perfect."""
    if tolerance <= 0:
        return 1.0 if deviation == 0 else 0.0
    return float(np.clip(1.0 - deviation / tolerance, 0.0, 1.0))


def _greedy_non_overlapping(patterns: list[ChartPattern]) -> list[ChartPattern]:
    """Keep the highest-confidence patterns whose price spans don't overlap."""
    kept: list[ChartPattern] = []
    for pattern in sorted(patterns, key=lambda p: p.confidence, reverse=True):
        if all(
            pattern.end_index < k.start_index or pattern.start_index > k.end_index for k in kept
        ):
            kept.append(pattern)
    return kept


# --- Head-and-shoulders (and its inverse) -----------------------------------


def _score_head_and_shoulders(window: list[Pivot], inverse: bool) -> ChartPattern | None:
    """Grade a 5-pivot window against the head-and-shoulders geometry, or reject it.

    Top (bearish): high, low, high, low, high with the middle high (head) the
    tallest and the two outer highs (shoulders) similar. Inverse (bullish) is
    the mirror: the middle low is the deepest and the outer lows are similar.
    The two inner pivots form the neckline and should be roughly level.
    """
    ls, inner1, head, inner2, rs = window
    shoulder_a, shoulder_b = ls.price, rs.price
    neck_a, neck_b = inner1.price, inner2.price
    pattern_type: str
    direction: Direction

    if not inverse:
        # Head strictly highest; shoulders sit above the neckline troughs.
        if not (head.price > shoulder_a and head.price > shoulder_b):
            return None
        if min(shoulder_a, shoulder_b) <= max(neck_a, neck_b):
            return None
        reference = max(shoulder_a, shoulder_b)
        prominence = (head.price - reference) / reference
        pattern_type, direction = "head_and_shoulders", "bearish"
    else:
        if not (head.price < shoulder_a and head.price < shoulder_b):
            return None
        if max(shoulder_a, shoulder_b) >= min(neck_a, neck_b):
            return None
        reference = min(shoulder_a, shoulder_b)
        prominence = (reference - head.price) / reference
        pattern_type, direction = "inverse_head_and_shoulders", "bullish"

    if prominence < _HEAD_PROMINENCE_MIN:
        return None

    dev_shoulder = _rel_diff(shoulder_a, shoulder_b)
    dev_neckline = _rel_diff(neck_a, neck_b)
    if dev_shoulder > _SHOULDER_TOL or dev_neckline > _NECKLINE_TOL:
        return None

    confidence = 100.0 * float(
        np.mean(
            [
                _score(dev_shoulder, _SHOULDER_TOL),
                _score(dev_neckline, _NECKLINE_TOL),
                float(np.clip(prominence / _HEAD_PROMINENCE_TARGET, 0.0, 1.0)),
            ]
        )
    )
    return ChartPattern(
        pattern_type=pattern_type,
        direction=direction,
        confidence=confidence,
        start_index=ls.index,
        end_index=rs.index,
        start_date=ls.date,
        end_date=rs.date,
        key_levels={
            "neckline": (neck_a + neck_b) / 2,
            "head": head.price,
            "left_shoulder": shoulder_a,
            "right_shoulder": shoulder_b,
        },
    )


def detect_head_and_shoulders(pivots: list[Pivot]) -> list[ChartPattern]:
    """Find every head-and-shoulders / inverse formation in the pivot sequence."""
    found: list[ChartPattern] = []
    for i in range(len(pivots) - 4):
        window = pivots[i : i + 5]
        kinds = [p.kind for p in window]
        if kinds == ["high", "low", "high", "low", "high"]:
            detection = _score_head_and_shoulders(window, inverse=False)
        elif kinds == ["low", "high", "low", "high", "low"]:
            detection = _score_head_and_shoulders(window, inverse=True)
        else:
            detection = None
        if detection is not None:
            found.append(detection)
    return _greedy_non_overlapping(found)


# --- Double top / double bottom ---------------------------------------------


def _score_double(window: list[Pivot], inverse: bool) -> ChartPattern | None:
    """Grade a 3-pivot window as a double top (H,L,H) or double bottom (L,H,L)."""
    first, middle, second = window
    peak_a, peak_b = first.price, second.price
    dev_peak = _rel_diff(peak_a, peak_b)
    if dev_peak > _PEAK_TOL:
        return None

    pattern_type: str
    direction: Direction
    edge_avg = (peak_a + peak_b) / 2
    if not inverse:  # two tops, a dip between -> bearish
        depth = (edge_avg - middle.price) / edge_avg
        pattern_type, direction = "double_top", "bearish"
        key = {"resistance": edge_avg, "neckline": middle.price}
    else:  # two bottoms, a bounce between -> bullish
        depth = (middle.price - edge_avg) / edge_avg
        pattern_type, direction = "double_bottom", "bullish"
        key = {"support": edge_avg, "neckline": middle.price}

    if depth < _RETRACE_MIN:
        return None

    confidence = 100.0 * float(
        np.mean(
            [
                _score(dev_peak, _PEAK_TOL),
                float(np.clip(depth / _RETRACE_TARGET, 0.0, 1.0)),
            ]
        )
    )
    return ChartPattern(
        pattern_type=pattern_type,
        direction=direction,
        confidence=confidence,
        start_index=first.index,
        end_index=second.index,
        start_date=first.date,
        end_date=second.date,
        key_levels=key,
    )


def detect_double_patterns(pivots: list[Pivot]) -> list[ChartPattern]:
    """Find every double-top and double-bottom formation in the pivot sequence."""
    found: list[ChartPattern] = []
    for i in range(len(pivots) - 2):
        window = pivots[i : i + 3]
        kinds = [p.kind for p in window]
        if kinds == ["high", "low", "high"]:
            detection = _score_double(window, inverse=False)
        elif kinds == ["low", "high", "low"]:
            detection = _score_double(window, inverse=True)
        else:
            detection = None
        if detection is not None:
            found.append(detection)
    return _greedy_non_overlapping(found)


# --- Trendline patterns: triangles / wedges / channels ----------------------


def _fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _fit_score(
    xs: np.ndarray, ys: np.ndarray, slope: float, intercept: float, scale: float
) -> float:
    """[0, 1] score for how tightly points hug their fitted line (1 = on the line)."""
    residuals = np.abs(ys - (slope * xs + intercept))
    per_point = [_score(float(r) / scale, _TREND_FIT_TARGET) for r in residuals]
    return float(np.mean(per_point))


def _classify_trendlines(
    slope_high_pct: float,
    slope_low_pct: float,
    gap_start: float,
    gap_end: float,
) -> tuple[str, Direction] | None:
    """Map the two fitted trendlines' slopes + convergence to a pattern, or None.

    Slopes are expressed as each line's fractional move across the window, so a
    single `_TREND_FLAT_TOL` cleanly separates flat / rising / falling. The gap
    between the lines shrinking vs staying parallel separates triangles/wedges
    (converging) from channels (parallel); diverging (broadening) is out of scope.
    """
    flat_high = abs(slope_high_pct) < _TREND_FLAT_TOL
    flat_low = abs(slope_low_pct) < _TREND_FLAT_TOL
    rising_high, falling_high = (
        slope_high_pct >= _TREND_FLAT_TOL,
        slope_high_pct <= -_TREND_FLAT_TOL,
    )
    rising_low, falling_low = slope_low_pct >= _TREND_FLAT_TOL, slope_low_pct <= -_TREND_FLAT_TOL

    if gap_start <= 0 or gap_end <= 0:
        return None  # lines cross within the window -> degenerate, not a clean formation
    gap_change = (gap_end - gap_start) / gap_start
    converging = gap_change < -_TREND_CONVERGE_TOL
    diverging = gap_change > _TREND_CONVERGE_TOL
    parallel = not converging and not diverging

    if converging:
        if flat_high and rising_low:
            return "ascending_triangle", "bullish"
        if falling_high and flat_low:
            return "descending_triangle", "bearish"
        if falling_high and rising_low:
            return "symmetrical_triangle", "neutral"
        if rising_high and rising_low:
            return "rising_wedge", "bearish"
        if falling_high and falling_low:
            return "falling_wedge", "bullish"
        return None
    if parallel:
        if rising_high and rising_low:
            return "rising_channel", "bullish"
        if falling_high and falling_low:
            return "falling_channel", "bearish"
        if flat_high and flat_low:
            return "rectangle", "neutral"
        return None
    _ = diverging  # broadening formations are intentionally not classified
    return None


def _score_trendline_window(window: list[Pivot]) -> ChartPattern | None:
    highs = [p for p in window if p.kind == "high"]
    lows = [p for p in window if p.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    xs_high = np.array([p.index for p in highs], dtype=float)
    ys_high = np.array([p.price for p in highs], dtype=float)
    xs_low = np.array([p.index for p in lows], dtype=float)
    ys_low = np.array([p.price for p in lows], dtype=float)

    slope_high, intercept_high = _fit_line(xs_high, ys_high)
    slope_low, intercept_low = _fit_line(xs_low, ys_low)

    x_start = float(min(p.index for p in window))
    x_end = float(max(p.index for p in window))
    span = x_end - x_start
    mean_price = float(np.mean([p.price for p in window]))
    if span <= 0 or mean_price <= 0:
        return None

    slope_high_pct = slope_high * span / mean_price
    slope_low_pct = slope_low * span / mean_price
    gap_start = (slope_high * x_start + intercept_high) - (slope_low * x_start + intercept_low)
    gap_end = (slope_high * x_end + intercept_high) - (slope_low * x_end + intercept_low)

    classification = _classify_trendlines(slope_high_pct, slope_low_pct, gap_start, gap_end)
    if classification is None:
        return None
    pattern_type, direction = classification

    confidence = 100.0 * float(
        np.mean(
            [
                _fit_score(xs_high, ys_high, slope_high, intercept_high, mean_price),
                _fit_score(xs_low, ys_low, slope_low, intercept_low, mean_price),
            ]
        )
    )
    first, last = window[0], window[-1]
    return ChartPattern(
        pattern_type=pattern_type,
        direction=direction,
        confidence=confidence,
        start_index=first.index,
        end_index=last.index,
        start_date=first.date,
        end_date=last.date,
        key_levels={
            "upper_slope": slope_high,
            "lower_slope": slope_low,
            "resistance": slope_high * x_end + intercept_high,
            "support": slope_low * x_end + intercept_low,
        },
    )


def detect_trendline_patterns(
    pivots: list[Pivot], window: int = _TREND_WINDOW
) -> list[ChartPattern]:
    """Slide a fixed window of consecutive pivots, fitting and classifying trendlines.

    A window of `window` alternating pivots yields the highs and lows needed to
    fit an upper and a lower line. Overlapping windows that both classify are
    resolved to the single highest-confidence formation per price span.
    """
    found: list[ChartPattern] = []
    for start in range(len(pivots) - window + 1):
        detection = _score_trendline_window(pivots[start : start + window])
        if detection is not None:
            found.append(detection)
    return _greedy_non_overlapping(found)


# --- Cup-and-handle ---------------------------------------------------------


def _roundedness_score(cup_closes: np.ndarray) -> float | None:
    """Score how well the cup segment forms a rounded U, or None if it isn't one.

    Fits a quadratic to the segment: a genuine cup opens upward (a > 0) with its
    low roughly centered in time (not a one-sided V or a descending slide). The
    returned score in [0, 1] grades the parabola fit and how centered the bottom
    is; None means it fails the shape gate outright.
    """
    n = cup_closes.size
    if n < 5:
        return None
    xs = np.arange(n, dtype=float)
    a, b, c = np.polyfit(xs, cup_closes, 2)
    if a <= 0:  # opens downward or flat -> not a cup
        return None

    vertex_frac = (-b / (2 * a)) / (n - 1)
    lo, hi = _CUP_BOTTOM_CENTER_BAND
    if not (lo <= vertex_frac <= hi):
        return None

    # Time-near-bottom: a rounded U dwells in its lower third; a sharp V just
    # touches it. This is what actually separates the two -- a V fits a parabola
    # deceptively well and has a centered vertex, but doesn't linger at the low.
    low, high = float(cup_closes.min()), float(cup_closes.max())
    price_range = high - low
    if price_range <= 0:
        return None
    threshold_price = low + price_range * _CUP_BOTTOM_THIRD
    bottom_dwell = float(np.mean(cup_closes <= threshold_price))
    if bottom_dwell < _CUP_MIN_BOTTOM_DWELL:
        return None

    fitted = a * xs**2 + b * xs + c
    scale = float(np.mean(cup_closes))
    residual = float(np.mean(np.abs(cup_closes - fitted)))
    fit_score = _score(residual / scale, _TREND_FIT_TARGET)
    center_score = _score(abs(vertex_frac - 0.5), 0.25)
    dwell_score = float(np.clip((bottom_dwell - 0.2) / (0.5 - 0.2), 0.0, 1.0))
    return float(np.mean([fit_score, center_score, dwell_score]))


def _score_cup_and_handle(
    left_rim: Pivot, bottom: Pivot, right_rim: Pivot, closes: np.ndarray, labels: pd.Index
) -> ChartPattern | None:
    """Grade a rim-bottom-rim triple plus a trailing handle as cup-and-handle.

    Conservative by design (Section 7.1 calls this out as the hardest to detect
    without false positives): it requires similar rims, a genuine rounded bottom,
    a real minimum cup depth, and a *shallow* handle pullback after the right rim
    that then recovers. The handle is read from the raw closes after the right
    rim, because a shallow handle is often smaller than the zig-zag threshold and
    so leaves no pivot of its own.
    """
    dev_rim = _rel_diff(left_rim.price, right_rim.price)
    if dev_rim > _CUP_RIM_TOL:
        return None

    rim_avg = (left_rim.price + right_rim.price) / 2
    depth = (rim_avg - bottom.price) / rim_avg
    if depth < _CUP_MIN_DEPTH:
        return None

    cup_closes = closes[left_rim.index : right_rim.index + 1]
    roundedness = _roundedness_score(cup_closes)
    if roundedness is None:
        return None

    # Handle: the shallowest, most recent pullback after the right rim that then
    # recovers. Requires at least a couple of bars of post-rim data to exist.
    after = closes[right_rim.index + 1 :]
    if after.size < 2:
        return None
    handle_low = float(np.min(after))
    handle_low_pos = right_rim.index + 1 + int(np.argmin(after))
    handle_depth = (right_rim.price - handle_low) / right_rim.price
    if handle_depth <= 0 or handle_depth > _CUP_HANDLE_MAX_RETRACE * depth:
        return None
    if float(after[-1]) <= handle_low:  # must recover off the handle low, not still falling
        return None

    handle_shallowness = _score(handle_depth, _CUP_HANDLE_MAX_RETRACE * depth)
    confidence = 100.0 * float(
        np.mean(
            [
                _score(dev_rim, _CUP_RIM_TOL),
                float(np.clip(depth / _CUP_DEPTH_TARGET, 0.0, 1.0)),
                roundedness,
                handle_shallowness,
            ]
        )
    )
    return ChartPattern(
        pattern_type="cup_and_handle",
        direction="bullish",
        confidence=confidence,
        start_index=left_rim.index,
        end_index=handle_low_pos,
        start_date=left_rim.date,
        end_date=labels[handle_low_pos],
        key_levels={
            "rim": rim_avg,
            "cup_bottom": bottom.price,
            "handle_low": handle_low,
        },
    )


def detect_cup_and_handle(pivots: list[Pivot], prices: pd.DataFrame) -> list[ChartPattern]:
    """Find cup-and-handle formations: a rounded cup (high, low, high) plus a handle."""
    if "close" not in prices.columns:
        raise ValueError("prices is missing required column: 'close'")
    closes = prices["close"].to_numpy(dtype=float)
    labels = prices.index

    found: list[ChartPattern] = []
    for i in range(len(pivots) - 2):
        window = pivots[i : i + 3]
        if [p.kind for p in window] != ["high", "low", "high"]:
            continue
        detection = _score_cup_and_handle(window[0], window[1], window[2], closes, labels)
        if detection is not None:
            found.append(detection)
    return _greedy_non_overlapping(found)


# --- Aggregate entry point --------------------------------------------------


def detect_all_chart_patterns(
    prices: pd.DataFrame, threshold: float = DEFAULT_ZIGZAG_THRESHOLD
) -> list[ChartPattern]:
    """Run every geometric detector and return the combined `ChartPattern` objects.

    Overlapping formations of the *same* family are already resolved inside each
    detector; overlaps *across* families (e.g. a double top that is also the top
    of a head-and-shoulders) are kept, since they are genuinely different
    readings of the same price action and the UI may want to surface both.
    """
    if "close" not in prices.columns:
        raise ValueError("prices is missing required column: 'close'")
    pivots = find_pivots(prices, threshold=threshold)
    return [
        *detect_head_and_shoulders(pivots),
        *detect_double_patterns(pivots),
        *detect_trendline_patterns(pivots),
        *detect_cup_and_handle(pivots, prices),
    ]


def detect_chart_patterns(
    prices: pd.DataFrame,
    symbol: str | None = None,
    threshold: float = DEFAULT_ZIGZAG_THRESHOLD,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Detect chart patterns, normalized to the `pattern_signals` row shape.

    Output matches `technical.detect_candlestick_patterns` exactly -- optional
    `symbol` column, then date/pattern_type/direction/confidence -- so the
    eventual persistence layer treats candlestick and chart patterns uniformly.
    Each pattern's `date` is its completion (final pivot / handle) date; combined
    with the (symbol, date, pattern_type) uniqueness of `pattern_signals`, that
    means re-detecting the same formation on a later run won't duplicate it.
    """
    columns = (["symbol"] if symbol else []) + ["date", "pattern_type", "direction", "confidence"]
    patterns = [
        p for p in detect_all_chart_patterns(prices, threshold) if p.confidence >= min_confidence
    ]
    if not patterns:
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "date": p.end_date,
            "pattern_type": p.pattern_type,
            "direction": p.direction,
            "confidence": p.confidence,
        }
        for p in patterns
    ]
    result = pd.DataFrame(rows)
    if symbol is not None:
        result.insert(0, "symbol", symbol)
    return result.sort_values(["date", "pattern_type"]).reset_index(drop=True)
