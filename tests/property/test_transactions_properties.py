"""Property-based tests for `portfolio.transactions` (Section 29's hypothesis complement).

FIFO tax-lot bookkeeping is exactly the kind of accounting code where a
conservation law should hold no matter the input: shares and dollars can move
between "open lot" and "realized gain," but never appear or vanish. These
tests generate random buy sequences (and random partial sells/splits on top of
them) and check that conservation directly, complementing `test_transactions.py`'s
hand-picked FIFO/split scenarios rather than repeating them.
"""

from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.portfolio.transactions import (
    StockSplit,
    Transaction,
    build_lot_book,
    holding_term,
)

_SHARES = st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_PRICE = st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_RATIO = st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False)
_FRACTION = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _buys_strategy(*, min_size: int = 1, max_size: int = 8) -> st.SearchStrategy[list[Transaction]]:
    def _build(rows: list[tuple[float, float]]) -> list[Transaction]:
        base = date(2020, 1, 1)
        return [
            Transaction(
                symbol="AAA",
                action="buy",
                shares=shares,
                price=price,
                date=base + timedelta(days=7 * i),
            )
            for i, (shares, price) in enumerate(rows)
        ]

    return st.lists(st.tuples(_SHARES, _PRICE), min_size=min_size, max_size=max_size).map(_build)


class TestBuyOnlyConservation:
    @given(_buys_strategy())
    def test_open_shares_and_cost_basis_equal_the_sum_of_buys(
        self, buys: list[Transaction]
    ) -> None:
        book = build_lot_book(buys)
        lots = book.open_lots["AAA"]
        assert sum(lot.shares for lot in lots) == pytest.approx(sum(tx.shares for tx in buys))
        expected_cost = sum(tx.shares * tx.price for tx in buys)
        assert sum(lot.cost_basis for lot in lots) == pytest.approx(expected_cost)
        assert book.realized_gains == []

    @given(_buys_strategy(min_size=2))
    def test_lots_stay_in_purchase_order_fifo(self, buys: list[Transaction]) -> None:
        book = build_lot_book(buys)
        dates = [lot.purchase_date for lot in book.open_lots["AAA"]]
        assert dates == sorted(dates)


class TestBuyThenSellConservation:
    @given(_buys_strategy(min_size=1, max_size=6), _FRACTION)
    def test_shares_and_cost_basis_are_conserved_across_a_partial_sell(
        self, buys: list[Transaction], sell_fraction: float
    ) -> None:
        total_shares = sum(tx.shares for tx in buys)
        total_cost = sum(tx.shares * tx.price for tx in buys)
        # A tiny epsilon keeps the sell strictly within what was bought, since
        # sell_fraction == 1.0 exactly can round up past the held total in fp.
        sell_shares = total_shares * sell_fraction * 0.999999
        if sell_shares <= 1e-9:
            return  # build_lot_book requires strictly positive transaction shares
        sell = Transaction(
            symbol="AAA", action="sell", shares=sell_shares, price=1.0, date=date(2021, 1, 1)
        )
        book = build_lot_book([*buys, sell])

        realized_shares = sum(g.shares for g in book.realized_gains)
        remaining_shares = sum(lot.shares for lot in book.open_lots.get("AAA", []))
        assert realized_shares + remaining_shares == pytest.approx(total_shares, abs=1e-6)
        assert realized_shares == pytest.approx(sell_shares, abs=1e-6)

        realized_cost = sum(g.cost_basis for g in book.realized_gains)
        remaining_cost = sum(lot.cost_basis for lot in book.open_lots.get("AAA", []))
        assert realized_cost + remaining_cost == pytest.approx(total_cost, abs=1e-4)

    @given(_buys_strategy(min_size=1, max_size=6), _FRACTION)
    def test_realized_gain_equals_proceeds_minus_cost_basis_per_lot(
        self, buys: list[Transaction], sell_fraction: float
    ) -> None:
        total_shares = sum(tx.shares for tx in buys)
        sell_shares = total_shares * sell_fraction * 0.999999
        if sell_shares <= 1e-9:
            return
        sell = Transaction(
            symbol="AAA", action="sell", shares=sell_shares, price=7.5, date=date(2021, 1, 1)
        )
        book = build_lot_book([*buys, sell])
        for gain in book.realized_gains:
            assert gain.gain == pytest.approx(gain.proceeds - gain.cost_basis, abs=1e-6)
            assert gain.proceeds == pytest.approx(gain.shares * 7.5, abs=1e-6)

    @given(_buys_strategy(min_size=2, max_size=6), _FRACTION)
    def test_fifo_consumes_the_oldest_lots_first(
        self, buys: list[Transaction], sell_fraction: float
    ) -> None:
        total_shares = sum(tx.shares for tx in buys)
        sell_shares = total_shares * sell_fraction * 0.999999
        if sell_shares <= 1e-9:
            return
        sell = Transaction(
            symbol="AAA", action="sell", shares=sell_shares, price=1.0, date=date(2021, 1, 1)
        )
        book = build_lot_book([*buys, sell])
        realized_dates = [g.purchase_date for g in book.realized_gains]
        assert realized_dates == sorted(realized_dates)
        remaining_dates = [lot.purchase_date for lot in book.open_lots.get("AAA", [])]
        if realized_dates and remaining_dates:
            # Nothing realized can be dated after anything still open (oldest-first).
            assert max(realized_dates) <= min(remaining_dates)


class TestSplitConservation:
    @given(_buys_strategy(min_size=1, max_size=6), _RATIO)
    def test_split_scales_shares_and_preserves_total_cost_basis(
        self, buys: list[Transaction], ratio: float
    ) -> None:
        total_shares = sum(tx.shares for tx in buys)
        total_cost = sum(tx.shares * tx.price for tx in buys)
        split = StockSplit(symbol="AAA", date=date(2022, 1, 1), ratio=ratio)
        book = build_lot_book(buys, splits=[split])
        lots = book.open_lots["AAA"]
        assert sum(lot.shares for lot in lots) == pytest.approx(total_shares * ratio, rel=1e-6)
        assert sum(lot.cost_basis for lot in lots) == pytest.approx(total_cost, rel=1e-6)


class TestHoldingTermProperty:
    @given(
        st.dates(min_value=date(1990, 1, 1), max_value=date(2100, 1, 1)),
        st.integers(min_value=367, max_value=3000),
    )
    def test_well_over_a_year_later_is_always_long(
        self, purchase_date: date, days_later: int
    ) -> None:
        as_of = purchase_date + timedelta(days=days_later)
        assert holding_term(purchase_date, as_of=as_of) == "long"

    @given(
        st.dates(min_value=date(1990, 1, 1), max_value=date(2100, 1, 1)),
        st.integers(min_value=0, max_value=364),
    )
    def test_well_under_a_year_later_is_always_short(
        self, purchase_date: date, days_later: int
    ) -> None:
        as_of = purchase_date + timedelta(days=days_later)
        assert holding_term(purchase_date, as_of=as_of) == "short"
