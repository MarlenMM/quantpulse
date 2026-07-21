import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis.patterns import (
    detect_chart_patterns,
    detect_cup_and_handle,
    detect_double_patterns,
    detect_head_and_shoulders,
    detect_trendline_patterns,
    find_pivots,
)


def _series(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": np.array(closes, dtype=float)},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def _from_pivots(pivot_prices: list[float], gap: int = 5) -> pd.DataFrame:
    """Interpolate `gap` bars between each successive pivot price into a close series."""
    closes: list[float] = []
    for a, b in zip(pivot_prices[:-1], pivot_prices[1:], strict=True):
        closes.extend(np.linspace(a, b, gap + 1)[:-1])
    closes.append(pivot_prices[-1])
    return _series(closes)


# --- find_pivots ------------------------------------------------------------


class TestFindPivots:
    def test_alternates_high_low(self) -> None:
        pivots = find_pivots(_from_pivots([100, 120, 95, 125, 100]), threshold=0.05)
        kinds = [p.kind for p in pivots]
        assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))

    def test_ignores_sub_threshold_wiggles(self) -> None:
        # every move is under 5%, so nothing should register
        pivots = find_pivots(_series([100, 102, 100, 103, 101, 102]), threshold=0.05)
        assert pivots == []

    def test_captures_a_significant_swing(self) -> None:
        pivots = find_pivots(_series([100, 100, 100, 90, 90, 100, 100]), threshold=0.05)
        kinds = [p.kind for p in pivots]
        assert "low" in kinds

    def test_flat_series_has_no_pivots(self) -> None:
        assert find_pivots(_series([100] * 10), threshold=0.05) == []

    def test_raises_on_missing_column(self) -> None:
        with pytest.raises(ValueError, match="close"):
            find_pivots(pd.DataFrame({"open": [1, 2, 3]}))

    def test_raises_on_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            find_pivots(_series([1, 2, 3]), threshold=0)


# --- Head-and-shoulders -----------------------------------------------------


class TestHeadAndShoulders:
    def test_detects_textbook_top_with_high_confidence(self) -> None:
        prices = _from_pivots([100, 120, 105, 140, 105, 120, 100])
        detections = detect_head_and_shoulders(find_pivots(prices, threshold=0.05))
        hs = [d for d in detections if d.pattern_type == "head_and_shoulders"]
        assert len(hs) == 1
        assert hs[0].direction == "bearish"
        assert hs[0].confidence > 95
        assert hs[0].key_levels["neckline"] == pytest.approx(105.0)

    def test_detects_inverse_bottom(self) -> None:
        prices = _from_pivots([100, 80, 95, 60, 95, 80, 100])
        detections = detect_head_and_shoulders(find_pivots(prices, threshold=0.05))
        inverse = [d for d in detections if d.pattern_type == "inverse_head_and_shoulders"]
        assert len(inverse) == 1
        assert inverse[0].direction == "bullish"
        assert inverse[0].confidence > 95

    def test_rejects_when_head_is_not_the_highest(self) -> None:
        # middle "head" (118) lower than the left shoulder (125) -> not an H&S
        prices = _from_pivots([100, 125, 105, 118, 105, 120, 100])
        hs = [
            d
            for d in detect_head_and_shoulders(find_pivots(prices, threshold=0.05))
            if d.pattern_type == "head_and_shoulders"
        ]
        assert hs == []

    def test_rejects_wildly_asymmetric_shoulders(self) -> None:
        # shoulders 110 vs 150 differ far beyond tolerance
        prices = _from_pivots([100, 110, 105, 165, 105, 150, 100])
        hs = [
            d
            for d in detect_head_and_shoulders(find_pivots(prices, threshold=0.05))
            if d.pattern_type == "head_and_shoulders"
        ]
        assert hs == []

    def test_does_not_fire_on_a_clean_uptrend(self) -> None:
        prices = _from_pivots([100, 108, 104, 118, 113, 128])
        assert detect_head_and_shoulders(find_pivots(prices, threshold=0.05)) == []


# --- Double top / bottom ----------------------------------------------------


class TestDoublePatterns:
    def test_detects_double_top(self) -> None:
        prices = _from_pivots([100, 130, 110, 130, 100])
        dt = [
            d
            for d in detect_double_patterns(find_pivots(prices, threshold=0.05))
            if d.pattern_type == "double_top"
        ]
        assert len(dt) == 1
        assert dt[0].direction == "bearish"
        assert dt[0].confidence > 95

    def test_detects_double_bottom(self) -> None:
        prices = _from_pivots([120, 90, 110, 90, 120])
        db = [
            d
            for d in detect_double_patterns(find_pivots(prices, threshold=0.05))
            if d.pattern_type == "double_bottom"
        ]
        assert len(db) == 1
        assert db[0].direction == "bullish"

    def test_rejects_dissimilar_peaks(self) -> None:
        # 130 vs 150 differ beyond the peak tolerance
        prices = _from_pivots([100, 130, 110, 150, 100])
        dt = [
            d
            for d in detect_double_patterns(find_pivots(prices, threshold=0.05))
            if d.pattern_type == "double_top"
        ]
        assert dt == []


# --- Trendline patterns -----------------------------------------------------


class TestTrendlinePatterns:
    def _only(self, prices: pd.DataFrame) -> list[str]:
        return [d.pattern_type for d in detect_trendline_patterns(find_pivots(prices, 0.03))]

    def test_ascending_triangle(self) -> None:
        prices = _from_pivots([100, 120, 108, 120, 114, 120, 118], gap=6)
        assert "ascending_triangle" in self._only(prices)

    def test_descending_triangle(self) -> None:
        prices = _from_pivots([100, 130, 100, 122, 100, 115, 100], gap=6)
        assert "descending_triangle" in self._only(prices)

    def test_symmetrical_triangle(self) -> None:
        prices = _from_pivots([100, 130, 106, 124, 112, 118], gap=6)
        assert "symmetrical_triangle" in self._only(prices)

    def test_rising_wedge_is_bearish(self) -> None:
        prices = _from_pivots([100, 125, 112, 130, 124, 133], gap=6)
        wedges = [
            d
            for d in detect_trendline_patterns(find_pivots(prices, 0.03))
            if d.pattern_type == "rising_wedge"
        ]
        assert wedges and wedges[0].direction == "bearish"

    def test_falling_wedge_is_bullish(self) -> None:
        prices = _from_pivots([133, 108, 121, 104, 109, 101], gap=6)
        wedges = [
            d
            for d in detect_trendline_patterns(find_pivots(prices, 0.03))
            if d.pattern_type == "falling_wedge"
        ]
        assert wedges and wedges[0].direction == "bullish"

    def test_rising_channel(self) -> None:
        prices = _from_pivots([100, 120, 108, 128, 116, 136, 124], gap=6)
        assert "rising_channel" in self._only(prices)

    def test_broadening_formation_is_not_classified(self) -> None:
        # highs rising, lows falling -> lines diverge -> intentionally unclassified
        prices = _from_pivots([110, 120, 100, 128, 92, 136, 84], gap=6)
        assert self._only(prices) == []


# --- Cup-and-handle ---------------------------------------------------------


def _rounded_cup(depth_from: float = 120, bottom: float = 90, span: int = 41) -> list[float]:
    center = (span - 1) / 2
    a = (depth_from - bottom) / center**2
    return [a * (x - center) ** 2 + bottom for x in range(span)]


class TestCupAndHandle:
    def _handle(self, rim: float, low: float, recover: float) -> list[float]:
        return list(np.linspace(rim, low, 6)) + list(np.linspace(low, recover, 5))[1:]

    def test_detects_rounded_cup_with_shallow_handle(self) -> None:
        prices = _series(_rounded_cup() + self._handle(120, 116.4, 119))
        detections = detect_cup_and_handle(find_pivots(prices, 0.05), prices)
        assert len(detections) == 1
        assert detections[0].direction == "bullish"
        assert detections[0].confidence > 70

    def test_rejects_a_sharp_v(self) -> None:
        v = list(np.linspace(120, 90, 21)) + list(np.linspace(90, 120, 21))[1:]
        prices = _series(v + self._handle(120, 116.4, 119))
        assert detect_cup_and_handle(find_pivots(prices, 0.05), prices) == []

    def test_rejects_cup_without_a_handle(self) -> None:
        prices = _series(_rounded_cup() + list(np.linspace(120, 135, 8)))
        assert detect_cup_and_handle(find_pivots(prices, 0.05), prices) == []

    def test_rejects_too_shallow_a_cup(self) -> None:
        # only ~4% deep, below the minimum cup depth
        prices = _series(_rounded_cup(depth_from=120, bottom=115) + self._handle(120, 117, 119))
        assert detect_cup_and_handle(find_pivots(prices, 0.02), prices) == []


# --- Aggregate entry point --------------------------------------------------


class TestDetectChartPatterns:
    def test_output_contract_with_symbol(self) -> None:
        prices = _from_pivots([100, 130, 110, 130, 100])
        result = detect_chart_patterns(prices, symbol="AAPL", threshold=0.05)
        assert list(result.columns) == [
            "symbol",
            "date",
            "pattern_type",
            "direction",
            "confidence",
        ]
        assert (result["symbol"] == "AAPL").all()

    def test_output_contract_without_symbol(self) -> None:
        prices = _from_pivots([100, 130, 110, 130, 100])
        result = detect_chart_patterns(prices, threshold=0.05)
        assert "symbol" not in result.columns

    def test_min_confidence_filters(self) -> None:
        prices = _from_pivots([100, 130, 110, 130, 100])
        everything = detect_chart_patterns(prices, threshold=0.05, min_confidence=0)
        strict = detect_chart_patterns(prices, threshold=0.05, min_confidence=99)
        assert len(strict) <= len(everything)
        assert (strict["confidence"] >= 99).all()

    def test_flat_series_returns_empty_with_correct_columns(self) -> None:
        result = detect_chart_patterns(_series([100] * 30), symbol="FLAT")
        assert list(result.columns) == [
            "symbol",
            "date",
            "pattern_type",
            "direction",
            "confidence",
        ]
        assert len(result) == 0


# --- Look-ahead / point-in-time honesty -------------------------------------


class TestLookAheadStability:
    def test_confirmed_pivots_do_not_change_as_future_data_arrives(self) -> None:
        # The core point-in-time guarantee: appending later bars must never
        # rewrite pivots that were already confirmed on earlier data.
        full = _from_pivots([100, 120, 105, 135, 110, 130, 100, 140], gap=5)
        prefix = full.iloc[:30]

        full_pivots = find_pivots(full, threshold=0.05)
        prefix_pivots = find_pivots(prefix, threshold=0.05)

        # All but the last prefix pivot are confirmed; they must match the full
        # series' pivots exactly in index, price, and kind.
        confirmed = prefix_pivots[:-1]
        for a, b in zip(confirmed, full_pivots, strict=False):
            assert (a.index, a.kind) == (b.index, b.kind)
            assert a.price == pytest.approx(b.price)
        assert len(confirmed) <= len(full_pivots)
