import pytest

from quantpulse.portfolio import recommendations as rec
from quantpulse.portfolio.rebalancing import RebalancePlan


class TestHoldingRecommendation:
    @pytest.mark.parametrize(
        "rating,expected_action",
        [
            ("strong_buy", "add"),
            ("buy", "add"),
            ("hold", "hold"),
            ("sell", "trim"),
            ("strong_sell", "sell"),
        ],
    )
    def test_rating_to_action_mapping(self, rating: str, expected_action: str) -> None:
        result = rec.holding_recommendation("AAPL", 0.05, rating)
        assert result.action == expected_action

    def test_add_is_downgraded_to_hold_when_already_overweight(self) -> None:
        result = rec.holding_recommendation("AAPL", 0.20, "buy", concentration_threshold=0.15)
        assert result.action == "hold"
        assert "20%" in result.reason
        assert "held rather than added to" in result.reason

    def test_add_survives_below_threshold(self) -> None:
        result = rec.holding_recommendation(
            "AAPL", 0.10, "strong_buy", concentration_threshold=0.15
        )
        assert result.action == "add"

    def test_sell_is_not_dampened_by_weight(self) -> None:
        # A tiny, already-shrunk position rated Strong Sell is still "sell".
        result = rec.holding_recommendation("AAPL", 0.01, "strong_sell")
        assert result.action == "sell"

    def test_reason_cites_purchase_rating_when_it_differs(self) -> None:
        result = rec.holding_recommendation("AAPL", 0.05, "strong_sell", purchase_rating="hold")
        assert "Strong Sell" in result.reason
        assert "was Hold at purchase" in result.reason

    def test_reason_omits_purchase_context_when_unchanged(self) -> None:
        result = rec.holding_recommendation("AAPL", 0.05, "hold", purchase_rating="hold")
        assert "at purchase" not in result.reason

    def test_reason_omits_purchase_context_when_absent(self) -> None:
        result = rec.holding_recommendation("AAPL", 0.05, "hold")
        assert "at purchase" not in result.reason
        assert result.purchase_rating is None

    def test_rejects_bad_rating(self) -> None:
        with pytest.raises(ValueError, match="rating must be one of"):
            rec.holding_recommendation("AAPL", 0.05, "super_buy")

    def test_rejects_bad_purchase_rating(self) -> None:
        with pytest.raises(ValueError, match="purchase_rating must be one of"):
            rec.holding_recommendation("AAPL", 0.05, "hold", purchase_rating="super_buy")

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="weight must be >= 0"):
            rec.holding_recommendation("AAPL", -0.1, "hold")

    def test_rejects_bad_threshold(self) -> None:
        with pytest.raises(ValueError, match="concentration_threshold must be in"):
            rec.holding_recommendation("AAPL", 0.05, "hold", concentration_threshold=0.0)


class TestHerfindahlIndex:
    def test_equal_weighted_four_positions(self) -> None:
        hhi = rec.herfindahl_index({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        assert hhi == pytest.approx(0.25)
        assert rec.effective_position_count(hhi) == pytest.approx(4.0)

    def test_single_position_is_maximally_concentrated(self) -> None:
        hhi = rec.herfindahl_index({"A": 1.0})
        assert hhi == pytest.approx(1.0)
        assert rec.effective_position_count(hhi) == pytest.approx(1.0)

    def test_zero_weights_are_ignored(self) -> None:
        hhi = rec.herfindahl_index({"A": 0.5, "B": 0.5, "C": 0.0})
        assert hhi == pytest.approx(0.5)

    def test_empty_weights_have_no_effective_count(self) -> None:
        assert rec.herfindahl_index({}) == pytest.approx(0.0)
        assert rec.effective_position_count(0.0) is None


class TestConcentrationWarnings:
    def test_flags_positions_above_threshold(self) -> None:
        warnings = rec.concentration_warnings(
            {"A": 0.20, "B": 0.05, "C": 0.75}, kind="position", threshold=0.15
        )
        assert [w.label for w in warnings] == ["C", "A"]  # highest weight first
        assert warnings[0].weight == pytest.approx(0.75)
        assert "75%" in warnings[0].message

    def test_nothing_flagged_below_threshold(self) -> None:
        warnings = rec.concentration_warnings({"A": 0.10, "B": 0.10}, kind="position")
        assert warnings == []

    def test_boundary_is_exclusive(self) -> None:
        # exactly at the threshold is not "above" it
        warnings = rec.concentration_warnings({"A": 0.15}, kind="position", threshold=0.15)
        assert warnings == []
        warnings = rec.concentration_warnings({"A": 0.150001}, kind="position", threshold=0.15)
        assert len(warnings) == 1

    def test_sector_kind_uses_sector_wording(self) -> None:
        warnings = rec.concentration_warnings({"Tech": 0.62}, kind="sector", threshold=0.15)
        assert warnings[0].kind == "sector"
        assert "sector-concentration" in warnings[0].message

    def test_rejects_bad_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            rec.concentration_warnings({"A": 0.5}, kind="position", threshold=1.5)


class TestSectorGaps:
    def test_missing_sectors_are_reported_in_known_order(self) -> None:
        gaps = rec.sector_gaps(
            {"Energy": 0.5, "Materials": 0.5},
            known_sectors=("Energy", "Materials", "Health Care", "Utilities"),
        )
        assert [g.sector for g in gaps] == ["Health Care", "Utilities"]

    def test_held_sectors_are_not_reported(self) -> None:
        gaps = rec.sector_gaps({"Energy": 1.0}, known_sectors=("Energy",))
        assert gaps == []

    def test_zero_weight_sector_still_counts_as_missing(self) -> None:
        gaps = rec.sector_gaps({"Energy": 0.0}, known_sectors=("Energy",))
        assert [g.sector for g in gaps] == ["Energy"]

    def test_candidates_are_attached_and_capped(self) -> None:
        gaps = rec.sector_gaps(
            {},
            known_sectors=("Health Care",),
            candidates={"Health Care": ["A", "B", "C", "D", "E", "F", "G"]},
        )
        assert gaps[0].candidates == ["A", "B", "C", "D", "E"]
        assert "A, B, C, D, E" in gaps[0].message

    def test_no_candidates_supplied_is_an_honest_empty_list(self) -> None:
        gaps = rec.sector_gaps({}, known_sectors=("Health Care",))
        assert gaps[0].candidates == []
        assert "Top-ranked" not in gaps[0].message

    def test_default_known_sectors_has_eleven_gics_sectors(self) -> None:
        assert len(rec.KNOWN_GICS_SECTORS) == 11
        assert len(set(rec.KNOWN_GICS_SECTORS)) == 11


class TestRecommend:
    def test_assembles_all_four_pieces(self) -> None:
        holdings = {
            "AAPL": rec.HoldingContext(weight=0.30, rating="buy", sector="Information Technology"),
            "XOM": rec.HoldingContext(weight=0.10, rating="strong_sell", sector="Energy"),
            "JPM": rec.HoldingContext(weight=0.05, rating="hold", sector="Financials"),
        }
        result = rec.recommend(
            holdings,
            known_sectors=("Information Technology", "Energy", "Financials", "Health Care"),
        )

        by_symbol = {h.symbol: h for h in result.holdings}
        assert by_symbol["AAPL"].action == "hold"  # 30% >= 15% threshold, downgraded from add
        assert by_symbol["XOM"].action == "sell"
        assert by_symbol["JPM"].action == "hold"

        assert result.concentration.position_hhi == pytest.approx(0.30**2 + 0.10**2 + 0.05**2)
        position_warning_labels = {
            w.label for w in result.concentration.warnings if w.kind == "position"
        }
        assert position_warning_labels == {"AAPL"}

        assert [g.sector for g in result.sector_gaps] == ["Health Care"]

        assert result.rebalance.triggered is True
        assert any("Sell" in r for r in result.rebalance.reasons)

    def test_no_warnings_or_sells_means_pointer_not_triggered(self) -> None:
        holdings = {
            "AAPL": rec.HoldingContext(weight=0.10, rating="hold", sector="Information Technology"),
            "XOM": rec.HoldingContext(weight=0.10, rating="hold", sector="Energy"),
        }
        result = rec.recommend(holdings, known_sectors=("Information Technology", "Energy"))
        assert result.rebalance.triggered is False
        assert result.rebalance.reasons == []
        assert result.concentration.warnings == []

    def test_cash_and_sectorless_holdings_do_not_affect_sector_stats(self) -> None:
        holdings = {
            "CASH": rec.HoldingContext(weight=0.50, rating="hold", sector=None),
            "SPY": rec.HoldingContext(weight=0.50, rating="hold", sector=None),
        }
        result = rec.recommend(holdings, known_sectors=("Energy",))
        assert result.concentration.sector_hhi is None
        assert result.concentration.sector_effective_count is None
        assert [g.sector for g in result.sector_gaps] == ["Energy"]
        assert all(w.kind == "position" for w in result.concentration.warnings)

    def test_sector_weight_is_the_sum_across_holdings_in_that_sector(self) -> None:
        holdings = {
            "AAPL": rec.HoldingContext(weight=0.10, rating="hold", sector="Information Technology"),
            "MSFT": rec.HoldingContext(weight=0.10, rating="hold", sector="Information Technology"),
        }
        result = rec.recommend(holdings, sector_threshold=0.15, known_sectors=())
        sector_warning = next(w for w in result.concentration.warnings if w.kind == "sector")
        assert sector_warning.label == "Information Technology"
        assert sector_warning.weight == pytest.approx(0.20)

    def test_pointer_triggers_on_sector_concentration_alone(self) -> None:
        holdings = {
            "AAPL": rec.HoldingContext(weight=0.09, rating="hold", sector="Information Technology"),
            "MSFT": rec.HoldingContext(weight=0.09, rating="hold", sector="Information Technology"),
        }
        result = rec.recommend(holdings, sector_threshold=0.15, known_sectors=())
        assert result.rebalance.triggered is True
        assert any("sector" in r for r in result.rebalance.reasons)

    def test_supplied_plan_passes_through_untouched(self) -> None:
        plan = RebalancePlan(
            trades=[],
            total_value=1000.0,
            cash_before=0.0,
            cash_after=0.0,
            estimated_transaction_cost=0.0,
            turnover=0.0,
            whole_shares=False,
            current_weights={},
            target_weights={},
            achieved_weights={},
        )
        holdings = {"AAPL": rec.HoldingContext(weight=0.05, rating="hold")}
        result = rec.recommend(holdings, rebalance_plan=plan)
        assert result.rebalance.plan is plan

    def test_no_plan_supplied_is_none(self) -> None:
        holdings = {"AAPL": rec.HoldingContext(weight=0.05, rating="hold")}
        result = rec.recommend(holdings)
        assert result.rebalance.plan is None

    def test_rejects_bad_rating_in_holdings(self) -> None:
        holdings = {"AAPL": rec.HoldingContext(weight=0.05, rating="super_buy")}
        with pytest.raises(ValueError, match=r"holdings\['AAPL'\].rating"):
            rec.recommend(holdings)

    def test_rejects_negative_weight_in_holdings(self) -> None:
        holdings = {"AAPL": rec.HoldingContext(weight=-0.05, rating="hold")}
        with pytest.raises(ValueError, match=r"holdings\['AAPL'\].weight"):
            rec.recommend(holdings)

    def test_empty_portfolio_produces_an_empty_but_valid_result(self) -> None:
        result = rec.recommend({}, known_sectors=("Energy",))
        assert result.holdings == []
        assert result.concentration.position_hhi == pytest.approx(0.0)
        assert result.concentration.position_effective_count is None
        assert result.concentration.sector_hhi is None
        assert [g.sector for g in result.sector_gaps] == ["Energy"]
        assert result.rebalance.triggered is False
