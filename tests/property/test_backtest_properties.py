"""Property-based tests for `analysis.backtest`'s bootstrap CI mechanics (Section 29).

`test_backtest.py` already hand-checks the moving-block bootstrap's mechanics
(`_block_indices`, `_default_block_size`, interval bracketing) at a handful of
fixed `n`/`block_size`/seed combinations. The mechanics themselves are
supposed to hold for *any* sample size and block size, though -- that's the
whole premise of resampling **positions** rather than values (module
docstring: "what makes paired resampling correct by construction") -- so these
tests sweep `n`/`block_size`/`confidence_level` at random instead of one point
each.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.analysis import backtest as bt


class TestBlockIndicesProperties:
    @given(
        st.integers(min_value=2, max_value=300),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_always_the_right_length_and_in_range(self, n: int, block_size: int, seed: int) -> None:
        block_size = min(block_size, n)
        rng = np.random.default_rng(seed)
        idx = bt._block_indices(n, block_size, rng)
        assert len(idx) == n
        assert idx.min() >= 0
        assert idx.max() < n

    @given(
        st.integers(min_value=2, max_value=300),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_full_blocks_are_consecutive_and_increasing(
        self, n: int, block_size: int, seed: int
    ) -> None:
        block_size = min(block_size, n)
        rng = np.random.default_rng(seed)
        idx = bt._block_indices(n, block_size, rng)
        for start in range(0, n - block_size + 1, block_size):
            block = idx[start : start + block_size]
            assert list(block) == list(range(block[0], block[0] + block_size))


class TestDefaultBlockSizeProperties:
    @given(st.integers(min_value=2, max_value=5000))
    def test_always_at_least_one_and_capped_at_half(self, n: int) -> None:
        size = bt._default_block_size(n)
        assert 1 <= size <= max(1, n // 2)

    @given(st.integers(min_value=0, max_value=1))
    def test_below_two_observations_is_always_one(self, n: int) -> None:
        assert bt._default_block_size(n) == 1


class TestBlockBootstrapCiProperties:
    # NOTE: "the interval always brackets the point estimate" is deliberately
    # NOT asserted here as a universal property, even though it holds for
    # every well-behaved (roughly-symmetric, reasonably-sized) return series
    # `test_backtest.py`'s fixed examples use. It is a statistical tendency,
    # not a guarantee: a percentile interval is built from the RESAMPLED
    # statistic's distribution, whose center need not coincide with the
    # observed statistic on a small, heavily skewed sample at a low confidence
    # level -- hypothesis found a concrete counterexample (16 mostly-zero
    # observations with two large outliers, 50% confidence). That is
    # expected behavior of a percentile bootstrap, not a bug in
    # `block_bootstrap_ci`, so the two properties below are the ones the
    # implementation actually guarantees by construction.
    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=8,
            max_size=120,
        ),
        st.integers(min_value=0, max_value=2**31 - 1),
        st.floats(min_value=0.5, max_value=0.99),
    )
    def test_point_is_always_the_statistic_on_the_original_sample(
        self, values: list[float], seed: int, confidence_level: float
    ) -> None:
        arr = np.asarray(values, dtype=float)

        def statistic(positions: np.ndarray) -> float:
            return float(arr[positions].mean())

        ci = bt.block_bootstrap_ci(
            len(arr),
            statistic,
            confidence_level=confidence_level,
            n_resamples=200,
            random_state=seed,
        )
        if ci is None:
            return
        assert ci.low <= ci.high
        assert ci.point == pytest.approx(statistic(np.arange(len(arr))))

    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=8,
            max_size=120,
        ),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_n_defined_never_exceeds_n_resamples(self, values: list[float], seed: int) -> None:
        arr = np.asarray(values, dtype=float)

        def statistic(positions: np.ndarray) -> float:
            return float(arr[positions].mean())

        ci = bt.block_bootstrap_ci(len(arr), statistic, n_resamples=150, random_state=seed)
        if ci is None:
            return
        assert ci.n_defined <= 150
        assert ci.n_observations == len(arr)

    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=8,
            max_size=80,
        ),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_same_seed_is_fully_reproducible(self, values: list[float], seed: int) -> None:
        arr = np.asarray(values, dtype=float)

        def statistic(positions: np.ndarray) -> float:
            return float(arr[positions].mean())

        a = bt.block_bootstrap_ci(len(arr), statistic, n_resamples=100, random_state=seed)
        b = bt.block_bootstrap_ci(len(arr), statistic, n_resamples=100, random_state=seed)
        assert a is not None and b is not None
        assert (a.low, a.high, a.point) == (b.low, b.high, b.point)

    @given(st.integers(min_value=0, max_value=7))
    def test_below_min_obs_always_abstains(self, n: int) -> None:
        assert bt.block_bootstrap_ci(n, lambda pos: float(len(pos))) is None
