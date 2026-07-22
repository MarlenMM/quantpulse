import pytest

from quantpulse.analysis import macro


class TestYieldCurveSpread:
    def test_computes_10y_minus_2y(self) -> None:
        assert macro.yield_curve_spread(4.5, 4.2) == pytest.approx(0.3)

    def test_inverted_curve_is_negative(self) -> None:
        spread = macro.yield_curve_spread(4.0, 4.6)
        assert spread is not None and spread < 0
        assert macro.is_yield_curve_inverted(spread) is True

    def test_normal_curve_is_not_inverted(self) -> None:
        assert macro.is_yield_curve_inverted(0.5) is False

    def test_missing_yield_yields_none(self) -> None:
        assert macro.yield_curve_spread(None, 4.2) is None
        assert macro.yield_curve_spread(4.5, None) is None

    def test_none_spread_is_not_inverted(self) -> None:
        assert macro.is_yield_curve_inverted(None) is False


class TestCommodityOverlay:
    def test_energy_rises_with_oil(self) -> None:
        adj = macro.commodity_overlay_adjustment("Energy", {macro.OIL_WTI: 5.0})
        assert adj == 0.5  # +5% / 10% full-swing * sign +1.0

    def test_strong_dollar_is_a_headwind_for_tech(self) -> None:
        # DXY +5% with a -1.0 sensitivity -> negative adjustment.
        adj = macro.commodity_overlay_adjustment(
            "Information Technology", {macro.DOLLAR_INDEX: 5.0}
        )
        assert adj == -0.5

    def test_irrelevant_sector_gets_zero_not_noise(self) -> None:
        # Section 28: a biotech doesn't care about oil -- targeted, not universal.
        assert macro.commodity_overlay_adjustment("Health Care", {macro.OIL_WTI: 20.0}) == 0.0

    def test_unknown_or_none_sector_gets_zero(self) -> None:
        assert macro.commodity_overlay_adjustment(None, {macro.OIL_WTI: 5.0}) == 0.0
        assert macro.commodity_overlay_adjustment("Nonexistent", {macro.OIL_WTI: 5.0}) == 0.0

    def test_missing_series_return_is_skipped(self) -> None:
        # Energy cares about oil + dollar; only oil provided -> just oil's contribution.
        adj = macro.commodity_overlay_adjustment("Energy", {macro.OIL_WTI: 3.0})
        assert adj == 0.3

    def test_total_is_clipped_to_unit_range(self) -> None:
        # A huge oil spike saturates rather than exceeding +1.0.
        adj = macro.commodity_overlay_adjustment("Energy", {macro.OIL_WTI: 500.0})
        assert adj == 1.0

    def test_opposing_contributions_net_out(self) -> None:
        # Oil up (tailwind +0.5) and dollar up (headwind -0.25) net to +0.25.
        adj = macro.commodity_overlay_adjustment(
            "Energy", {macro.OIL_WTI: 5.0, macro.DOLLAR_INDEX: 5.0}
        )
        assert adj == 0.25


class TestPctChange:
    def test_computes_percent_change_first_to_last(self) -> None:
        assert macro.pct_change([100.0, 105.0, 110.0]) == 10.0

    def test_negative_move(self) -> None:
        assert macro.pct_change([100.0, 90.0]) == -10.0

    def test_too_short_series_is_none(self) -> None:
        assert macro.pct_change([]) is None
        assert macro.pct_change([100.0]) is None

    def test_non_positive_base_is_none(self) -> None:
        assert macro.pct_change([0.0, 5.0]) is None
        assert macro.pct_change([-1.0, 5.0]) is None
