"""Property-based tests for `portfolio.recommendations` (Section 29's hypothesis complement).

`herfindahl_index`/`effective_position_count`/`concentration_warnings` are
small, closed-form functions whose invariants hold for *any* weight
distribution, not just the handful `test_recommendations.py` hand-picks (equal
four positions, a single position, etc.) -- so a random-`N`, random-weight
sweep is exactly where a property test earns its keep over another example.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.portfolio.recommendations import (
    KNOWN_GICS_SECTORS,
    concentration_warnings,
    effective_position_count,
    herfindahl_index,
    sector_gaps,
)

# Bounded strictly below 1.0 so `threshold=1.0` (the valid range's upper edge,
# `_validate_threshold` requires `threshold <= 1.0`) can always exceed every
# generated weight in `TestConcentrationWarningsProperties`.
_WEIGHT = st.floats(min_value=1e-6, max_value=0.999, allow_nan=False, allow_infinity=False)


def _weights_dict_strategy(
    *, min_size: int = 1, max_size: int = 12
) -> st.SearchStrategy[dict[str, float]]:
    labels = st.integers(min_value=0, max_value=10_000).map(lambda i: f"P{i}")
    return st.dictionaries(labels, _WEIGHT, min_size=min_size, max_size=max_size)


class TestHerfindahlIndexProperties:
    @given(_weights_dict_strategy())
    def test_bounded_between_zero_and_one_for_fractional_weights(
        self, weights: dict[str, float]
    ) -> None:
        # Renormalize to a 0-1 portfolio (the function's documented domain) so
        # the classic HHI bound applies regardless of how many labels are drawn.
        total = sum(weights.values())
        fractional = {k: v / total for k, v in weights.items()}
        hhi = herfindahl_index(fractional)
        assert 0.0 < hhi <= 1.0

    @given(st.integers(min_value=1, max_value=50))
    def test_equal_weighted_n_positions_is_exactly_one_over_n(self, n: int) -> None:
        weights = {f"P{i}": 1.0 / n for i in range(n)}
        assert herfindahl_index(weights) == pytest.approx(1.0 / n)

    @given(_weights_dict_strategy())
    def test_effective_position_count_is_the_exact_inverse(self, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        fractional = {k: v / total for k, v in weights.items()}
        hhi = herfindahl_index(fractional)
        count = effective_position_count(hhi)
        assert count is not None
        assert count == 1.0 / hhi
        # A diversification-count can never exceed the number of positions held.
        assert count <= len(fractional) + 1e-9

    def test_zero_weights_never_increase_hhi(self) -> None:
        with_zero = herfindahl_index({"A": 0.5, "B": 0.5, "C": 0.0})
        without_zero = herfindahl_index({"A": 0.5, "B": 0.5})
        assert with_zero == without_zero


class TestConcentrationWarningsProperties:
    @given(_weights_dict_strategy(), st.floats(min_value=0.001, max_value=0.999))
    def test_every_flagged_label_strictly_exceeds_the_threshold(
        self, weights: dict[str, float], threshold: float
    ) -> None:
        warnings = concentration_warnings(weights, kind="position", threshold=threshold)
        assert all(w.weight > threshold for w in warnings)
        flagged_labels = {w.label for w in warnings}
        for label, weight in weights.items():
            if weight <= threshold:
                assert label not in flagged_labels

    @given(_weights_dict_strategy(), st.floats(min_value=0.001, max_value=0.999))
    def test_flagged_labels_are_sorted_by_weight_descending(
        self, weights: dict[str, float], threshold: float
    ) -> None:
        warnings = concentration_warnings(weights, kind="position", threshold=threshold)
        flagged_weights = [w.weight for w in warnings]
        assert flagged_weights == sorted(flagged_weights, reverse=True)

    @given(_weights_dict_strategy())
    def test_a_threshold_above_every_weight_flags_nothing(self, weights: dict[str, float]) -> None:
        # `_WEIGHT` never reaches 1.0, so the top of the valid (0, 1] range
        # always strictly exceeds every generated weight.
        assert concentration_warnings(weights, kind="position", threshold=1.0) == []


class TestSectorGapsProperties:
    @given(st.sets(st.sampled_from(KNOWN_GICS_SECTORS)))
    def test_reported_gaps_are_exactly_the_unheld_known_sectors(
        self, held_sectors: set[str]
    ) -> None:
        sector_weights = {sector: 0.1 for sector in held_sectors}
        gaps = sector_gaps(sector_weights)
        gap_names = {g.sector for g in gaps}
        assert gap_names == set(KNOWN_GICS_SECTORS) - held_sectors

    @given(st.sets(st.sampled_from(KNOWN_GICS_SECTORS)))
    def test_gaps_preserve_known_sectors_order(self, held_sectors: set[str]) -> None:
        sector_weights = {sector: 0.1 for sector in held_sectors}
        gaps = sector_gaps(sector_weights)
        expected_order = [s for s in KNOWN_GICS_SECTORS if s not in held_sectors]
        assert [g.sector for g in gaps] == expected_order
