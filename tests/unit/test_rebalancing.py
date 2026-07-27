import pytest

from quantpulse.portfolio import rebalancing as reb


def _conserves_value(plan: reb.RebalancePlan) -> bool:
    """Rebalancing reallocates value; it must never create or destroy it."""
    total = sum(plan.achieved_weights.values()) + plan.cash_after / plan.total_value
    return total == pytest.approx(1.0)


class TestValidation:
    def test_rejects_negative_current_shares(self) -> None:
        with pytest.raises(ValueError, match="current_shares must be non-negative"):
            reb.build_rebalance_plan({"A": -1.0}, {"A": 10.0}, {"A": 1.0})

    def test_rejects_negative_target_weights(self) -> None:
        with pytest.raises(ValueError, match="target_weights must be non-negative"):
            reb.build_rebalance_plan({"A": 1.0}, {"A": 10.0}, {"A": -0.5})

    def test_rejects_target_weights_over_one(self) -> None:
        with pytest.raises(ValueError, match="sum to at most 1.0"):
            reb.build_rebalance_plan({}, {"A": 10.0, "B": 10.0}, {"A": 0.6, "B": 0.6})

    def test_allows_a_sum_of_exactly_one_at_the_boundary(self) -> None:
        weights = {"A": 1.0 / 3, "B": 1.0 / 3, "C": 1.0 / 3}  # sums to exactly 1.0 in float
        plan = reb.build_rebalance_plan({}, {"A": 10.0, "B": 10.0, "C": 10.0}, weights, cash=100.0)
        assert plan is not None

    def test_allows_a_sum_a_hair_over_one_within_tolerance(self) -> None:
        weights = {"A": 0.5, "B": 0.5 + 1e-10}  # over 1.0 by less than the tolerance
        plan = reb.build_rebalance_plan({}, {"A": 10.0, "B": 10.0}, weights, cash=100.0)
        assert plan is not None

    def test_rejects_missing_price(self) -> None:
        with pytest.raises(ValueError, match="no price for holdings"):
            reb.build_rebalance_plan({"A": 1.0}, {}, {"A": 1.0})

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            reb.build_rebalance_plan({"A": 1.0}, {"A": 0.0}, {"A": 1.0})

    def test_rejects_negative_cash(self) -> None:
        with pytest.raises(ValueError, match="cash must be >= 0"):
            reb.build_rebalance_plan({}, {}, {}, cash=-5.0)

    def test_rejects_negative_transaction_cost(self) -> None:
        with pytest.raises(ValueError, match="transaction_cost must be >= 0"):
            reb.build_rebalance_plan({"A": 1.0}, {"A": 10.0}, {"A": 1.0}, transaction_cost=-0.1)

    def test_rejects_negative_min_trade_value(self) -> None:
        with pytest.raises(ValueError, match="min_trade_value must be >= 0"):
            reb.build_rebalance_plan({"A": 1.0}, {"A": 10.0}, {"A": 1.0}, min_trade_value=-1.0)

    def test_no_value_to_reallocate_yields_none(self) -> None:
        assert reb.build_rebalance_plan({}, {}, {}, cash=0.0) is None


class TestBasicRebalance:
    def test_sell_one_buy_another_hand_check(self) -> None:
        # Own 100 shares of A @ $10 (=$1000), 0 of B @ $20. Target: 0% A, 100% B.
        plan = reb.build_rebalance_plan({"A": 100.0}, {"A": 10.0, "B": 20.0}, {"A": 0.0, "B": 1.0})
        assert plan is not None
        assert plan.total_value == pytest.approx(1000.0)
        by_symbol = {t.symbol: t for t in plan.trades}
        assert by_symbol["A"].action == "sell"
        assert by_symbol["A"].shares == pytest.approx(100.0)
        assert by_symbol["B"].action == "buy"
        assert by_symbol["B"].shares == pytest.approx(50.0)  # $1000 / $20
        assert _conserves_value(plan)

    def test_new_position_not_currently_held(self) -> None:
        plan = reb.build_rebalance_plan({}, {"A": 50.0}, {"A": 1.0}, cash=500.0)
        assert plan is not None
        assert len(plan.trades) == 1
        trade = plan.trades[0]
        assert trade.symbol == "A" and trade.action == "buy"
        assert trade.shares == pytest.approx(10.0)
        assert plan.cash_after == pytest.approx(0.0)

    def test_full_exit_of_a_held_symbol_omitted_from_target(self) -> None:
        # A isn't mentioned in target_weights at all -> treated as target 0.
        plan = reb.build_rebalance_plan({"A": 20.0}, {"A": 25.0}, {})
        assert plan is not None
        assert len(plan.trades) == 1
        assert plan.trades[0].action == "sell"
        assert plan.trades[0].shares == pytest.approx(20.0)
        assert plan.cash_after == pytest.approx(500.0)

    def test_already_at_target_produces_no_trades(self) -> None:
        plan = reb.build_rebalance_plan(
            {"A": 50.0, "B": 25.0}, {"A": 10.0, "B": 20.0}, {"A": 0.5, "B": 0.5}
        )
        assert plan is not None
        assert plan.trades == []
        assert plan.turnover == pytest.approx(0.0)
        assert plan.estimated_transaction_cost == pytest.approx(0.0)

    def test_target_cash_weight_is_the_implicit_remainder(self) -> None:
        # Fully invested in A; target wants only 50% in A, 50% cash.
        plan = reb.build_rebalance_plan({"A": 100.0}, {"A": 10.0}, {"A": 0.5})
        assert plan is not None
        assert plan.cash_before == pytest.approx(0.0)
        assert plan.cash_after == pytest.approx(500.0)
        assert plan.trades[0].action == "sell"
        assert plan.trades[0].shares == pytest.approx(50.0)

    def test_partial_trim_not_a_full_exit(self) -> None:
        plan = reb.build_rebalance_plan({"A": 100.0}, {"A": 10.0}, {"A": 0.4})
        assert plan is not None
        # Own $1000 of A, target 40% of a $1000 total -> trim to $400, sell $600.
        assert plan.trades[0].action == "sell"
        assert plan.trades[0].shares == pytest.approx(60.0)
        assert plan.achieved_weights["A"] == pytest.approx(0.4)


class TestWholeShares:
    def test_rounds_to_the_nearest_whole_share(self) -> None:
        # Target implies 33.33 shares; whole-share mode rounds to 33.
        plan = reb.build_rebalance_plan({}, {"A": 30.0}, {"A": 1.0}, cash=1000.0, whole_shares=True)
        assert plan is not None
        assert plan.trades[0].shares == pytest.approx(33.0)
        assert plan.whole_shares is True

    def test_fractional_mode_keeps_the_exact_amount(self) -> None:
        plan = reb.build_rebalance_plan(
            {}, {"A": 30.0}, {"A": 1.0}, cash=1000.0, whole_shares=False
        )
        assert plan is not None
        assert plan.trades[0].shares == pytest.approx(1000.0 / 30.0)

    def test_a_full_exit_never_oversells_a_fractional_holding(self) -> None:
        # Own 11.6 shares; a naive round() of a full-exit sell would round to
        # -12, which is more than is actually held. Must clamp to exactly 11.6.
        plan = reb.build_rebalance_plan({"A": 11.6}, {"A": 10.0}, {}, whole_shares=True)
        assert plan is not None
        assert plan.trades[0].action == "sell"
        assert plan.trades[0].shares == pytest.approx(11.6)
        assert _conserves_value(plan)

    def test_rounding_residual_is_reflected_in_achieved_weight_and_cash(self) -> None:
        plan = reb.build_rebalance_plan({}, {"A": 30.0}, {"A": 1.0}, cash=1000.0, whole_shares=True)
        assert plan is not None
        # 33 shares @ $30 = $990, not the full $1000 target.
        assert plan.achieved_weights["A"] == pytest.approx(990.0 / 1000.0)
        assert plan.cash_after == pytest.approx(10.0)
        assert _conserves_value(plan)


class TestMinTradeValue:
    def test_filters_out_a_trade_below_the_dollar_floor(self) -> None:
        # $1000 total: A is $1 off target (below the $5 floor); B is $50 off
        # (a real trade that must survive the same filter).
        plan = reb.build_rebalance_plan(
            {"A": 100.0},
            {"A": 1.0, "B": 10.0},
            {"A": 0.099, "B": 0.05},
            cash=900.0,
            min_trade_value=5.0,
        )
        assert plan is not None
        symbols_traded = {t.symbol for t in plan.trades}
        assert "A" not in symbols_traded  # $1 drift too small to bother with
        assert "B" in symbols_traded

    def test_default_is_no_filtering(self) -> None:
        plan = reb.build_rebalance_plan({"A": 99.0}, {"A": 1.0}, {"A": 1.0}, cash=1.0)
        assert plan is not None
        assert len(plan.trades) == 1  # the $1 drift still produces a trade

    def test_filtered_trade_leaves_the_position_untouched(self) -> None:
        plan = reb.build_rebalance_plan(
            {"A": 100.0}, {"A": 10.0}, {"A": 0.999}, min_trade_value=50.0
        )
        assert plan is not None
        assert plan.trades == []
        assert plan.achieved_weights["A"] == plan.current_weights["A"]


class TestTurnoverAndCost:
    def test_turnover_matches_the_backtests_own_convention(self) -> None:
        # A full swap from 100% A to 100% B is turnover=2.0 in weight-fraction
        # units -- the same definition `backtest.backtest_strategy` uses.
        plan = reb.build_rebalance_plan({"A": 100.0}, {"A": 10.0, "B": 20.0}, {"B": 1.0})
        assert plan is not None
        assert plan.turnover == pytest.approx(2.0)

    def test_cost_scales_with_the_configured_rate(self) -> None:
        cheap = reb.build_rebalance_plan(
            {"A": 100.0}, {"A": 10.0, "B": 20.0}, {"B": 1.0}, transaction_cost=0.001
        )
        pricier = reb.build_rebalance_plan(
            {"A": 100.0}, {"A": 10.0, "B": 20.0}, {"B": 1.0}, transaction_cost=0.01
        )
        assert cheap is not None and pricier is not None
        assert pricier.estimated_transaction_cost == pytest.approx(
            cheap.estimated_transaction_cost * 10
        )

    def test_default_transaction_cost_matches_the_backtest_convention(self) -> None:
        from quantpulse.analysis import backtest

        assert reb.DEFAULT_TRANSACTION_COST == 0.001
        # Not literally the same default parameter, but the same documented
        # bid-ask stand-in Section 7.6 specifies -- pin both so they can't
        # silently drift apart in a future edit.
        import inspect

        assert (
            inspect.signature(backtest.backtest_strategy).parameters["transaction_cost"].default
            == reb.DEFAULT_TRANSACTION_COST
        )


class TestTradeOrdering:
    def test_sells_come_before_buys(self) -> None:
        # Full exit of A ($1000) and B ($1000), full entry into C ($2000):
        # deterministic order is sell A, sell B (tied trade value, alphabetical
        # tie-break), then buy C.
        plan = reb.build_rebalance_plan(
            {"A": 100.0, "B": 50.0}, {"A": 10.0, "B": 20.0, "C": 5.0}, {"C": 1.0}
        )
        assert plan is not None
        assert [t.symbol for t in plan.trades] == ["A", "B", "C"]
        assert [t.action for t in plan.trades] == ["sell", "sell", "buy"]


class TestConservation:
    def test_value_is_conserved_across_a_realistic_multi_asset_rebalance(self) -> None:
        plan = reb.build_rebalance_plan(
            {"A": 40.0, "B": 10.0, "C": 5.0},
            {"A": 10.0, "B": 40.0, "C": 100.0, "D": 25.0},
            {"A": 0.2, "B": 0.3, "C": 0.1, "D": 0.3},
            cash=50.0,
        )
        assert plan is not None
        assert _conserves_value(plan)

    def test_value_is_conserved_with_whole_share_rounding_and_filtering(self) -> None:
        plan = reb.build_rebalance_plan(
            {"A": 33.0, "B": 7.0},
            {"A": 9.87, "B": 51.3},
            {"A": 0.15, "B": 0.6},
            cash=123.45,
            whole_shares=True,
            min_trade_value=10.0,
        )
        assert plan is not None
        assert _conserves_value(plan)
