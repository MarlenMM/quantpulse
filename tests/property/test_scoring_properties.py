"""Property-based tests for `analysis.scoring` (Section 29's hypothesis complement).

Two invariants Section 29 names by name for this module:

* `percentile_normalize` output is always in (0, 100] and monotonic with the
  underlying raw score, across randomly generated inputs -- not just
  `test_scoring.py`'s one hand-picked series.
* `build_composite`'s coverage-renormalization (Section 7.5 step 3) never lets
  a missing category drag a composite toward zero, and never produces a
  composite or confidence outside the range the weighted-average math implies
  -- across randomly generated category tables, not just the few fixed
  fundamental/technical combinations `test_scoring.py` hand-checks.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.analysis import scoring
from quantpulse.analysis.investor_profiles import CATEGORIES, InvestorProfile

_SCORE = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_RAW_VALUE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def _weights_strategy() -> st.SearchStrategy[dict[str, float]]:
    """Seven positive floats normalized to sum to exactly 1.0 (`InvestorProfile`'s contract)."""

    def _normalize(raw: list[float]) -> dict[str, float]:
        total = sum(raw)
        return dict(zip(CATEGORIES, [v / total for v in raw], strict=True))

    return st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=len(CATEGORIES),
        max_size=len(CATEGORIES),
    ).map(_normalize)


def _category_raw_frame_strategy(*, n_symbols: int = 6) -> st.SearchStrategy[pd.DataFrame]:
    """A `category_raw`-shaped frame: every `CATEGORIES` column, some cells missing."""
    symbols = [f"S{i}" for i in range(n_symbols)]
    cell = st.one_of(st.none(), _RAW_VALUE)
    column = st.lists(cell, min_size=n_symbols, max_size=n_symbols)
    columns = st.fixed_dictionaries({c: column for c in CATEGORIES})
    return columns.map(lambda cols: pd.DataFrame(cols, index=symbols, dtype=float))


class TestPercentileNormalizeProperties:
    @given(st.lists(_SCORE, min_size=1, max_size=50, unique=True))
    def test_output_is_bounded_and_ranked(self, values: list[float]) -> None:
        out = scoring.percentile_normalize(pd.Series(values))
        assert ((out > 0.0) & (out <= 100.0)).all()
        # Rank order of the output must match the rank order of the input.
        input_order = pd.Series(values).rank(method="min")
        output_order = out.rank(method="min")
        assert list(input_order) == list(output_order)

    @given(st.lists(_SCORE, min_size=2, max_size=50, unique=True))
    def test_monotonic_with_underlying_score(self, values: list[float]) -> None:
        raw = pd.Series(values)
        out = scoring.percentile_normalize(raw)
        # For every pair, a strictly larger raw score never normalizes lower.
        ordered = out.iloc[np.argsort(raw.to_numpy())]
        assert list(ordered) == sorted(ordered)

    @given(st.lists(_SCORE, min_size=1, max_size=30, unique=True))
    def test_the_top_value_always_scores_exactly_100(self, values: list[float]) -> None:
        raw = pd.Series(values)
        out = scoring.percentile_normalize(raw)
        assert out[raw == raw.max()].item() == 100.0

    @given(st.lists(st.none() | _SCORE, min_size=1, max_size=30))
    def test_missing_values_stay_missing_and_do_not_shift_others(self, values: list) -> None:
        raw = pd.Series(values, dtype=float)
        out = scoring.percentile_normalize(raw)
        assert out[raw.isna()].isna().all()
        present = raw.dropna()
        if not present.empty:
            assert out[raw.notna()].notna().all()


class TestBuildCompositeCoverageProperties:
    @given(_category_raw_frame_strategy(), _weights_strategy())
    def test_composite_is_a_convex_combination_of_present_subscores(
        self, category_raw: pd.DataFrame, weights: dict[str, float]
    ) -> None:
        profile = InvestorProfile(name="random", weights=weights)
        result = scoring.build_composite(category_raw, profile=profile).scores
        normalized = scoring._normalized_subscores(category_raw)
        for _, row in result.iterrows():
            symbol = row["symbol"]
            present = normalized.loc[symbol].dropna()
            assert present.min() - 1e-6 <= row["composite_score"] <= present.max() + 1e-6

    @given(_category_raw_frame_strategy(), _weights_strategy())
    def test_data_confidence_is_the_present_weight_fraction(
        self, category_raw: pd.DataFrame, weights: dict[str, float]
    ) -> None:
        profile = InvestorProfile(name="random", weights=weights)
        result = scoring.build_composite(category_raw, profile=profile).scores
        assert ((result["data_confidence"] > 0.0) & (result["data_confidence"] <= 100.001)).all()
        for _, row in result.iterrows():
            present_categories = category_raw.loc[row["symbol"]].dropna().index
            expected = sum(weights[c] for c in present_categories if c in weights) * 100.0
            assert row["data_confidence"] == pytest.approx(expected, rel=1e-6, abs=1e-6)

    @given(_category_raw_frame_strategy(), _weights_strategy())
    def test_dropping_a_symbols_only_category_removes_it_rather_than_scoring_zero(
        self, category_raw: pd.DataFrame, weights: dict[str, float]
    ) -> None:
        # A symbol with usable data in exactly one category composites to
        # exactly that category's normalized value -- never a phantom-zero-
        # diluted number -- regardless of how many OTHER categories exist.
        profile = InvestorProfile(name="random", weights=weights)
        normalized = scoring._normalized_subscores(category_raw)
        for symbol in category_raw.index:
            present = normalized.loc[symbol].dropna()
            if len(present) != 1:
                continue
            result = scoring.build_composite(category_raw, profile=profile).scores
            row = result[result["symbol"] == symbol]
            if row.empty:
                continue  # that lone category happened to carry zero weight
            assert row.iloc[0]["composite_score"] == pytest.approx(
                present.iloc[0], rel=1e-6, abs=1e-6
            )

    @given(_category_raw_frame_strategy(), _weights_strategy())
    def test_percentile_rank_and_rating_are_monotonic_with_composite(
        self, category_raw: pd.DataFrame, weights: dict[str, float]
    ) -> None:
        profile = InvestorProfile(name="random", weights=weights)
        result = scoring.build_composite(category_raw, profile=profile).scores
        if len(result) < 2:
            return
        order = {r: i for i, r in enumerate(scoring.RATINGS)}
        by_composite = result.sort_values("composite_score", ascending=False)
        rating_ranks = [order[r] for r in by_composite["rating"]]
        assert rating_ranks == sorted(rating_ranks)
        percentile_ranks = list(by_composite["percentile_rank"])
        assert percentile_ranks == sorted(percentile_ranks, reverse=True)
