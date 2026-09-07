"""What the forward test decides to hold, and in what order it trades.

Pure functions over frames -- no account, no network. Every number asserted
here came from measuring the committed demo database on 2026-09-07 rather than
from taste; see `strategy.DEFAULT_POSITIONS` for the table.
"""

import pandas as pd
import pytest

from quantpulse.execution import strategy


def _scores(*rows: tuple[str, str, float, float]) -> pd.DataFrame:
    """symbol, rating, percentile_rank, last close."""
    return pd.DataFrame(
        [{"symbol": s, "rating": r, "percentile_rank": pct, "price": px} for s, r, pct, px in rows]
    )


class TestTargetPositions:
    def test_it_holds_the_top_names_by_published_rank(self) -> None:
        target = strategy.target_positions(
            _scores(
                ("AAA", "strong_buy", 99.0, 100.0),
                ("BBB", "buy", 80.0, 100.0),
                ("CCC", "buy", 71.0, 100.0),
            ),
            equity=30_000.0,
            positions=2,
        )
        assert set(target) == {"AAA", "BBB"}

    def test_only_buy_and_strong_buy_are_eligible(self) -> None:
        """The forward test follows the published rating. A Hold is not a buy,
        however well it ranks in a weak market."""
        target = strategy.target_positions(
            _scores(
                ("AAA", "hold", 99.0, 100.0),
                ("BBB", "sell", 98.0, 100.0),
                ("CCC", "strong_sell", 97.0, 100.0),
                ("DDD", "buy", 71.0, 100.0),
            ),
            equity=10_000.0,
            positions=4,
        )
        assert set(target) == {"DDD"}

    def test_it_is_equal_weight_in_dollars_floored_to_whole_shares(self) -> None:
        target = strategy.target_positions(
            _scores(("AAA", "buy", 90.0, 300.0), ("BBB", "buy", 80.0, 40.0)),
            equity=10_000.0,
            positions=2,
        )
        # $5,000 each: 16 x $300 = $4,800, 125 x $40 = $5,000.
        assert target == {"AAA": 16, "BBB": 125}

    def test_a_name_too_expensive_for_its_slice_is_left_out(self) -> None:
        """Not held at 0 shares, which would read as a position."""
        target = strategy.target_positions(
            _scores(("AAA", "buy", 90.0, 9_000.0), ("BBB", "buy", 80.0, 40.0)),
            equity=10_000.0,
            positions=2,
        )
        assert "AAA" not in target and target["BBB"] == 125

    def test_a_name_with_no_usable_price_is_left_out(self) -> None:
        """Sizing needs a price. A missing one must skip the name, not crash the
        rebalance for the other nineteen."""
        for bad in (None, float("nan"), 0.0, -5.0):
            target = strategy.target_positions(
                _scores(("AAA", "buy", 90.0, bad), ("BBB", "buy", 80.0, 40.0)),
                equity=10_000.0,
                positions=2,
            )
            assert "AAA" not in target, bad

    def test_an_empty_or_unrated_universe_targets_nothing(self) -> None:
        assert strategy.target_positions(pd.DataFrame(), equity=10_000.0) == {}
        assert (
            strategy.target_positions(_scores(("AAA", "hold", 99.0, 10.0)), equity=10_000.0) == {}
        )

    def test_no_equity_targets_nothing(self) -> None:
        """A blown-up or brand-new account must not produce orders.

        Belt and braces: the whole-share floor below would reach the same
        answer on its own, since a non-positive slice can never afford a share.
        The early exit is kept for saying so, not for the result.
        """
        assert strategy.target_positions(_scores(("AAA", "buy", 90.0, 10.0)), equity=0.0) == {}
        assert strategy.target_positions(_scores(("AAA", "buy", 90.0, 10.0)), equity=-1.0) == {}

    def test_asking_for_no_positions_does_not_divide_by_zero(self) -> None:
        """The part of that guard which is not redundant. `equity / positions`
        raises on 0, and a caller reading a misconfigured setting is exactly
        where a 0 comes from -- so the rebalance would die rather than hold
        nothing."""
        assert (
            strategy.target_positions(_scores(("AAA", "buy", 90.0, 10.0)), equity=1e6, positions=0)
            == {}
        )
        assert (
            strategy.target_positions(_scores(("AAA", "buy", 90.0, 10.0)), equity=1e6, positions=-5)
            == {}
        )

    def test_ties_break_deterministically(self) -> None:
        """Two runs on identical inputs must not churn the portfolio."""
        rows = _scores(("BBB", "buy", 90.0, 10.0), ("AAA", "buy", 90.0, 10.0))
        first = strategy.target_positions(rows, equity=1_000.0, positions=1)
        second = strategy.target_positions(rows.iloc[::-1], equity=1_000.0, positions=1)
        assert first == second == {"AAA": 100}

    def test_the_default_position_count_is_the_measured_one(self) -> None:
        """20, and the reason is arithmetic rather than preference.

        Measured over the 151 names rated buy/strong_buy on 2026-09-05 against
        a $100,000 paper account: at 100 names (the backtest's top-20% of 503)
        two names round to zero shares, the median weight is 6.5% off target
        and 12.4% of the account sits idle -- the "equal weight" is fiction. At
        20 the median drift is 1.1% and nothing rounds away.
        """
        assert strategy.DEFAULT_POSITIONS == 20


class TestOrders:
    def test_it_buys_what_is_missing_and_sells_what_is_gone(self) -> None:
        orders = strategy.orders_for(current={"OLD": 5.0}, target={"NEW": 3})
        assert strategy.Order("OLD", 5, "sell") in orders
        assert strategy.Order("NEW", 3, "buy") in orders

    def test_it_trades_only_the_difference_on_a_held_name(self) -> None:
        assert strategy.orders_for(current={"AAA": 10.0}, target={"AAA": 14}) == [
            strategy.Order("AAA", 4, "buy")
        ]
        assert strategy.orders_for(current={"AAA": 10.0}, target={"AAA": 6}) == [
            strategy.Order("AAA", 4, "sell")
        ]

    def test_an_unchanged_position_produces_no_order(self) -> None:
        """Every order costs the spread. A rebalance that re-states a position
        it already holds pays for nothing."""
        assert strategy.orders_for(current={"AAA": 10.0}, target={"AAA": 10}) == []

    def test_sells_come_before_buys(self) -> None:
        """Not cosmetic. Alpaca checks buying power on submission, so a rebalance
        that funds its buys from the sells it has not placed yet gets those buys
        rejected -- and the account ends the week holding only what it sold out
        of."""
        orders = strategy.orders_for(
            current={"OLD1": 5.0, "OLD2": 7.0}, target={"NEW1": 3, "NEW2": 4}
        )
        sides = [o.side for o in orders]
        assert sides == ["sell", "sell", "buy", "buy"], orders

    def test_orders_are_ordered_deterministically_within_a_side(self) -> None:
        orders = strategy.orders_for(current={}, target={"ZZZ": 1, "AAA": 1})
        assert [o.symbol for o in orders] == ["AAA", "ZZZ"]

    def test_a_short_position_is_closed_by_buying_it_back(self) -> None:
        """The strategy is long-only, so a short can only be an artefact of a
        hand-placed trade in the same account -- and leaving it open would put
        an unmanaged short in a record labelled as this strategy's."""
        assert strategy.orders_for(current={"AAA": -4.0}, target={}) == [
            strategy.Order("AAA", 4, "buy")
        ]

    def test_a_fractional_holding_is_not_over_sold(self) -> None:
        """Whole-share orders only, so 2.5 held sells 2 and leaves the dust."""
        assert strategy.orders_for(current={"AAA": 2.5}, target={}) == [
            strategy.Order("AAA", 2, "sell")
        ]

    def test_flat_to_flat_is_no_orders(self) -> None:
        assert strategy.orders_for(current={}, target={}) == []

    def test_every_order_is_a_positive_whole_quantity(self) -> None:
        """`alpaca.submit_order` rejects anything else, and it would reject it
        halfway through a rebalance."""
        orders = strategy.orders_for(
            current={"A": 3.0, "B": -2.0, "C": 1.5}, target={"B": 0, "D": 7}
        )
        assert orders
        for order in orders:
            assert order.qty > 0 and isinstance(order.qty, int)
            assert order.side in ("buy", "sell")


class TestTurnoverIsVisible:
    def test_the_planned_turnover_is_reported(self) -> None:
        """So a run that churned the whole book is distinguishable in the record
        from one that changed two names."""
        assert strategy.turnover(current={"A": 10.0}, target={"A": 10}) == pytest.approx(0.0)
        assert strategy.turnover(current={}, target={"A": 10}) == pytest.approx(1.0)
        assert strategy.turnover(current={"A": 10.0}, target={"B": 10}) == pytest.approx(1.0)

    def test_turnover_of_an_empty_rebalance_is_zero_not_undefined(self) -> None:
        assert strategy.turnover(current={}, target={}) == pytest.approx(0.0)
