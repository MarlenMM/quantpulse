"""Property-based tests for `analysis.risk` (Section 29's hypothesis complement).

Section 21 flags this module's whole reason for existing as "a ratio with a
vanishing denominator is not a ratio" -- the degenerate-input guards on
Sharpe/Sortino/VaR. `test_risk.py` already hand-checks one constant series per
guard; these tests generalize each guard across randomly generated constants,
lengths, and confidence levels, plus the module's other stated invariants
(expected shortfall is never milder than VaR, Sortino divides by the full
sample not the loss count).
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.analysis import risk

_RETURN = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
_CONSTANT = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestSharpeDegenerateGuard:
    @given(_CONSTANT, st.integers(min_value=2, max_value=200))
    def test_any_constant_series_has_no_defined_sharpe(self, value: float, length: int) -> None:
        assert risk.sharpe_ratio(_series([value] * length), periods_per_year=12) is None

    @given(
        st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=0.3, allow_nan=False, allow_infinity=False),
        st.integers(min_value=2, max_value=200),
        st.integers(min_value=1, max_value=252),
    )
    def test_genuinely_varying_series_is_always_defined(
        self, base: float, delta: float, length: int, periods_per_year: int
    ) -> None:
        # All but one bar at `base`, one bar meaningfully offset by `delta` --
        # a real (if small) dispersion, never mistaken for floating-point noise.
        values = [base] * length
        values[0] = base + delta
        result = risk.sharpe_ratio(_series(values), periods_per_year=periods_per_year)
        assert result is not None
        assert math.isfinite(result)


class TestSortinoDegenerateGuard:
    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        )
    )
    def test_no_period_below_target_is_always_none(self, non_negative_returns: list[float]) -> None:
        # Every return is >= the (zero) target, so there is no downside at all.
        result = risk.sortino_ratio(_series(non_negative_returns), periods_per_year=12)
        assert result is None

    @given(
        st.lists(_RETURN, min_size=20, max_size=150),
        st.integers(min_value=1, max_value=252),
    )
    def test_matches_a_full_sample_downside_deviation_reference(
        self, values: list[float], periods_per_year: int
    ) -> None:
        result = risk.sortino_ratio(_series(values), periods_per_year=periods_per_year)
        arr = np.asarray(values, dtype=float)
        shortfall = np.minimum(arr, 0.0)
        # The documented, deliberately-not-the-common-bug formula: mean of the
        # squared shortfall over ALL n periods, not just the losing ones.
        downside_deviation = math.sqrt(float(np.mean(shortfall**2)))
        scale = float(np.abs(arr).max())
        if downside_deviation <= scale * 1e-12:
            assert result is None
            return
        expected = float(arr.mean()) / downside_deviation * math.sqrt(periods_per_year)
        assert result == pytest.approx(expected)


class TestValueAtRiskInvariants:
    @given(_CONSTANT, st.integers(min_value=20, max_value=300))
    def test_any_constant_series_has_no_parametric_var(self, value: float, length: int) -> None:
        assert risk.value_at_risk(_series([value] * length), method="parametric") is None

    @given(
        st.lists(_RETURN, min_size=150, max_size=300),
        st.sampled_from(["historical", "parametric"]),
    )
    def test_expected_shortfall_is_never_milder_than_var(
        self, values: list[float], method: str
    ) -> None:
        result = risk.value_at_risk(_series(values), method=method)
        if result is None:
            return
        # Both are positive-loss-magnitude; the shortfall averages the tail
        # beyond the VaR cutoff, so it can only be as bad or worse.
        assert result.expected_shortfall >= result.var - 1e-9

    @given(st.floats(min_value=0.5, max_value=0.99))
    def test_min_historical_obs_grows_as_confidence_tightens(self, confidence: float) -> None:
        looser = risk._min_historical_var_obs(confidence)
        tighter = risk._min_historical_var_obs(min(confidence + 0.005, 0.999))
        assert tighter >= looser

    @given(st.lists(_RETURN, min_size=200, max_size=200))
    def test_higher_confidence_never_needs_fewer_observations(self, values: list[float]) -> None:
        series = _series(values)
        low = risk.value_at_risk(series, confidence=0.90)
        high = risk.value_at_risk(series, confidence=0.99)
        # 200 obs clears 90%'s floor but not 99%'s (500) -- if the tighter
        # confidence ever DOES resolve, the looser one must too.
        if high is not None:
            assert low is not None
