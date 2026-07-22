import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis.technical import (
    compute_indicators,
    compute_relative_strength,
    compute_sector_rotation,
    detect_anomalies,
    detect_candlestick_patterns,
    find_support_resistance_levels,
)


def _dates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def _ohlcv(n: int = 300, start_price: float = 100.0, drift: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(drift / 252, 0.01, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=_dates(n),
    )


class TestComputeIndicators:
    def test_adds_expected_columns(self) -> None:
        df = compute_indicators(_ohlcv())
        expected = {
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_hist",
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "rsi_14",
            "stoch_k",
            "stoch_d",
            "ao",
            "bb_lower",
            "bb_mid",
            "bb_upper",
            "bb_bandwidth",
            "bb_percent",
            "atr_14",
            "obv",
            "vwap",
            "cmf_20",
        }
        assert expected.issubset(df.columns)

    def test_sma_matches_manual_rolling_mean(self) -> None:
        prices = _ohlcv()
        df = compute_indicators(prices)
        expected_sma20 = prices["close"].rolling(20).mean()
        pd.testing.assert_series_equal(df["sma_20"], expected_sma20, check_names=False, atol=1e-6)

    def test_raises_on_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            compute_indicators(pd.DataFrame({"close": [1, 2, 3]}))

    def test_short_series_yields_nan_columns_instead_of_crashing(self) -> None:
        # A freshly-added ticker with a handful of bars: multi-column indicators
        # (MACD/ADX/Stochastic/Bollinger) return None here, which must degrade to
        # NaN columns, not crash the pass.
        df = compute_indicators(_ohlcv(n=3))
        for column in ("macd", "adx_14", "stoch_k", "bb_upper"):
            assert column in df.columns
            assert df[column].isna().all()


def _flat_then_engulfing(n_flat: int = 8) -> pd.DataFrame:
    # n_flat neutral candles, then a small bearish candle, then a larger
    # bullish candle whose body fully engulfs the previous one -- the
    # textbook bullish engulfing (needs >= 10 total rows, Section 7.1).
    flat = pd.DataFrame(
        {
            "open": [10.0] * n_flat,
            "high": [10.1] * n_flat,
            "low": [9.9] * n_flat,
            "close": [10.0] * n_flat,
        }
    )
    engulfing = pd.DataFrame(
        {
            "open": [10.0, 8.5],
            "high": [10.2, 11.5],
            "low": [8.8, 8.4],
            "close": [9.0, 11.0],
        }
    )
    df = pd.concat([flat, engulfing], ignore_index=True)
    df.index = _dates(len(df))
    return df


class TestDetectCandlestickPatterns:
    def test_detects_a_hand_built_bullish_engulfing(self) -> None:
        df = _flat_then_engulfing()
        result = detect_candlestick_patterns(df)
        # The engulfing candle is the last row; earlier flat "doji" candles
        # (zero-width bodies) can trivially get "engulfed" by the first real
        # candle too, so check the specific date our pattern lands on.
        on_final_day = result[result["date"] == df.index[-1]]
        engulfing = on_final_day[on_final_day["pattern_type"] == "engulfing"]
        assert not engulfing.empty
        assert engulfing.iloc[0]["direction"] == "bullish"

    def test_includes_symbol_column_only_when_given(self) -> None:
        df = _flat_then_engulfing()
        assert "symbol" not in detect_candlestick_patterns(df).columns
        with_symbol = detect_candlestick_patterns(df, symbol="AAPL")
        assert "symbol" in with_symbol.columns
        assert (with_symbol["symbol"] == "AAPL").all()

    def test_returns_empty_below_the_minimum_row_count_instead_of_crashing(self) -> None:
        df = pd.DataFrame(
            {"open": [10.0] * 5, "high": [10.2] * 5, "low": [9.8] * 5, "close": [10.0] * 5},
            index=_dates(5),
        )
        result = detect_candlestick_patterns(df, symbol="NEWCO")
        assert list(result.columns) == ["symbol", "date", "pattern_type", "direction", "confidence"]
        assert len(result) == 0

    def test_confidence_is_bounded_0_to_100(self) -> None:
        result = detect_candlestick_patterns(_ohlcv(200))
        if not result.empty:
            assert (result["confidence"] > 0).all()
            assert (result["confidence"] <= 100).all()

    def test_flat_series_yields_no_patterns_but_correct_shape(self) -> None:
        flat = pd.DataFrame(
            {"open": [10.0] * 5, "high": [10.0] * 5, "low": [10.0] * 5, "close": [10.0] * 5},
            index=_dates(5),
        )
        result = detect_candlestick_patterns(flat, symbol="FLAT")
        assert list(result.columns) == ["symbol", "date", "pattern_type", "direction", "confidence"]
        assert len(result) == 0


class TestFindSupportResistanceLevels:
    def test_finds_levels_the_price_bounces_between(self) -> None:
        # Oscillate cleanly between ~100 (support) and ~110 (resistance).
        n = 120
        t = np.arange(n)
        wave = 105 + 5 * np.sin(t / 3)
        df = pd.DataFrame({"high": wave + 0.3, "low": wave - 0.3}, index=_dates(n))

        levels = find_support_resistance_levels(df, order=3, proximity_pct=0.02, min_touches=2)

        assert not levels.empty
        assert (levels["level"].min() < 102) and (levels["level"].max() > 108)
        assert (levels["touches"] >= 2).all()

    def test_raises_on_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            find_support_resistance_levels(pd.DataFrame({"close": [1, 2, 3]}))

    def test_empty_result_has_correct_columns(self) -> None:
        # Monotonic series has no local extrema at all.
        df = pd.DataFrame({"high": range(20), "low": range(20)}, index=_dates(20))
        levels = find_support_resistance_levels(df, order=5)
        assert list(levels.columns) == ["level", "touches"]


class TestComputeRelativeStrength:
    def test_outperformance_ends_above_100(self) -> None:
        idx = _dates(50)
        symbol = pd.Series(np.linspace(100, 200, 50), index=idx)  # doubles
        benchmark = pd.Series(np.full(50, 100.0), index=idx)  # flat

        rs = compute_relative_strength(symbol, benchmark)

        assert rs.iloc[0] == pytest.approx(100.0)
        assert rs.iloc[-1] == pytest.approx(200.0)

    def test_raises_when_no_overlapping_dates(self) -> None:
        symbol = pd.Series([1, 2, 3], index=_dates(3, start="2020-01-01"))
        benchmark = pd.Series([1, 2, 3], index=_dates(3, start="2030-01-01"))
        with pytest.raises(ValueError, match="overlapping"):
            compute_relative_strength(symbol, benchmark)


class TestComputeSectorRotation:
    def test_ranks_outperforming_sector_first(self) -> None:
        idx = _dates(40)
        benchmark = pd.Series(np.full(40, 100.0), index=idx)
        winners = {
            "WIN1": pd.Series(np.linspace(100, 150, 40), index=idx),
            "WIN2": pd.Series(np.linspace(100, 140, 40), index=idx),
        }
        losers = {
            "LOSE1": pd.Series(np.linspace(100, 80, 40), index=idx),
            "LOSE2": pd.Series(np.linspace(100, 90, 40), index=idx),
        }
        prices = {**winners, **losers}
        sectors = {"WIN1": "Winning", "WIN2": "Winning", "LOSE1": "Losing", "LOSE2": "Losing"}

        rotation = compute_sector_rotation(prices, sectors, benchmark, lookback_days=30)

        assert rotation.iloc[0]["sector"] == "Winning"
        assert rotation.iloc[0]["relative_strength_change_pct"] > 0
        assert rotation.iloc[-1]["sector"] == "Losing"
        assert rotation.iloc[-1]["relative_strength_change_pct"] < 0

    def test_excludes_symbols_with_no_sector_mapping(self) -> None:
        idx = _dates(40)
        benchmark = pd.Series(np.full(40, 100.0), index=idx)
        prices = {"KNOWN": pd.Series(np.linspace(100, 120, 40), index=idx)}
        rotation = compute_sector_rotation(prices, {}, benchmark, lookback_days=30)
        assert rotation.empty


class TestDetectAnomalies:
    def test_flags_a_clear_spike(self) -> None:
        values = pd.Series([100.0] * 40 + [100.0])
        values.iloc[39] = 100000.0  # a single, huge spike at the end
        result = detect_anomalies(values, window=20, z_threshold=3.0)
        assert result["is_anomaly"].iloc[39]
        assert not result["is_anomaly"].iloc[10]

    def test_baseline_never_includes_the_current_point(self) -> None:
        # A day's own value must not leak into its own rolling mean/std.
        values = pd.Series([10.0] * 25)
        values.iloc[24] = 10000.0
        result = detect_anomalies(values, window=20, z_threshold=3.0)
        assert result["rolling_mean"].iloc[24] == pytest.approx(10.0)

    def test_early_points_without_a_baseline_are_never_anomalies(self) -> None:
        values = pd.Series([1.0, 2.0, 1000.0, 1.0, 1.0])
        result = detect_anomalies(values, window=20, z_threshold=3.0)
        assert not result["is_anomaly"].any()
