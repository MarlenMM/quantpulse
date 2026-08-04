from datetime import date

import pytest

from quantpulse.portfolio import transactions as tx


def _buy(symbol: str, shares: float, price: float, d: date) -> tx.Transaction:
    return tx.Transaction(symbol=symbol, action="buy", shares=shares, price=price, date=d)


def _sell(symbol: str, shares: float, price: float, d: date) -> tx.Transaction:
    return tx.Transaction(symbol=symbol, action="sell", shares=shares, price=price, date=d)


class TestHoldingTerm:
    def test_more_than_one_year_is_long(self) -> None:
        assert tx.holding_term(date(2023, 1, 15), as_of=date(2024, 1, 16)) == "long"

    def test_exactly_one_year_is_still_short(self) -> None:
        assert tx.holding_term(date(2023, 1, 15), as_of=date(2024, 1, 15)) == "short"

    def test_well_under_one_year_is_short(self) -> None:
        assert tx.holding_term(date(2024, 1, 1), as_of=date(2024, 6, 1)) == "short"

    def test_leap_day_purchase_does_not_crash(self) -> None:
        # 2024 is a leap year; 2025 has no Feb 29.
        assert tx.holding_term(date(2024, 2, 29), as_of=date(2025, 3, 1)) == "long"
        assert tx.holding_term(date(2024, 2, 29), as_of=date(2025, 2, 27)) == "short"


class TestValidation:
    def test_rejects_non_positive_shares(self) -> None:
        with pytest.raises(ValueError, match="shares must be > 0"):
            tx.build_lot_book([_buy("A", 0.0, 10.0, date(2024, 1, 1))])

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(ValueError, match="price must be > 0"):
            tx.build_lot_book([_buy("A", 10.0, 0.0, date(2024, 1, 1))])

    def test_rejects_nan_and_infinite_shares(self) -> None:
        # `NaN <= 0` is False, so a NaN slipped past the old `<= 0` guard. It
        # then failed every later comparison too, so the holding vanished from
        # the portfolio with no error -- an imported CSV could silently drop a
        # position. Infinity passed the old guard outright and produced an
        # infinite market value on the Portfolio page.
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="shares must be > 0"):
                tx.build_lot_book([_buy("A", bad, 10.0, date(2024, 1, 1))])

    def test_rejects_nan_and_infinite_price(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="price must be > 0"):
                tx.build_lot_book([_buy("A", 10.0, bad, date(2024, 1, 1))])

    def test_rejects_nan_and_infinite_split_ratio(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="ratio must be > 0"):
                tx.build_lot_book(
                    [], splits=[tx.StockSplit(symbol="A", date=date(2024, 1, 1), ratio=bad)]
                )

    def test_rejects_bad_action(self) -> None:
        bad = tx.Transaction(
            symbol="A", action="short", shares=1.0, price=1.0, date=date(2024, 1, 1)
        )  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="action must be 'buy' or 'sell'"):
            tx.build_lot_book([bad])

    def test_rejects_non_positive_split_ratio(self) -> None:
        with pytest.raises(ValueError, match="ratio must be > 0"):
            tx.build_lot_book(
                [], splits=[tx.StockSplit(symbol="A", date=date(2024, 1, 1), ratio=0.0)]
            )

    def test_selling_more_than_held_raises(self) -> None:
        txs = [_buy("A", 10.0, 100.0, date(2024, 1, 1)), _sell("A", 15.0, 110.0, date(2024, 2, 1))]
        with pytest.raises(ValueError, match="only 10.000000 were held"):
            tx.build_lot_book(txs)

    def test_selling_with_no_prior_holding_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot sell"):
            tx.build_lot_book([_sell("A", 1.0, 10.0, date(2024, 1, 1))])


class TestFifoBasics:
    def test_single_buy_single_partial_sell(self) -> None:
        txs = [_buy("A", 100.0, 10.0, date(2024, 1, 1)), _sell("A", 40.0, 15.0, date(2024, 6, 1))]
        book = tx.build_lot_book(txs)

        assert len(book.open_lots["A"]) == 1
        remaining = book.open_lots["A"][0]
        assert remaining.shares == pytest.approx(60.0)
        assert remaining.cost_basis == pytest.approx(600.0)  # 60 shares @ original $10

        assert len(book.realized_gains) == 1
        gain = book.realized_gains[0]
        assert gain.shares == pytest.approx(40.0)
        assert gain.proceeds == pytest.approx(600.0)  # 40 @ $15
        assert gain.cost_basis == pytest.approx(400.0)  # 40 @ $10
        assert gain.gain == pytest.approx(200.0)
        assert gain.term == "short"

    def test_full_sell_closes_the_lot(self) -> None:
        txs = [_buy("A", 10.0, 100.0, date(2024, 1, 1)), _sell("A", 10.0, 120.0, date(2024, 6, 1))]
        book = tx.build_lot_book(txs)
        assert "A" not in book.open_lots
        assert len(book.realized_gains) == 1
        assert book.realized_gains[0].gain == pytest.approx(200.0)

    def test_sell_spans_multiple_lots_oldest_first(self) -> None:
        txs = [
            _buy("A", 10.0, 100.0, date(2022, 1, 1)),  # long-term by the later sell
            _buy("A", 10.0, 200.0, date(2024, 1, 1)),  # short-term by the later sell
            _sell("A", 15.0, 300.0, date(2024, 6, 1)),
        ]
        book = tx.build_lot_book(txs)

        assert len(book.realized_gains) == 2
        first, second = book.realized_gains  # sorted by (sale_date, symbol, purchase_date)
        assert first.purchase_date == date(2022, 1, 1)
        assert first.shares == pytest.approx(10.0)
        assert first.term == "long"
        assert second.purchase_date == date(2024, 1, 1)
        assert second.shares == pytest.approx(5.0)
        assert second.term == "short"

        # 5 shares remain of the second (2024) lot.
        assert len(book.open_lots["A"]) == 1
        assert book.open_lots["A"][0].purchase_date == date(2024, 1, 1)
        assert book.open_lots["A"][0].shares == pytest.approx(5.0)
        assert book.open_lots["A"][0].cost_basis == pytest.approx(1000.0)  # 5 @ $200

    def test_fractional_shares_throughout(self) -> None:
        txs = [
            _buy("A", 10.5, 40.0, date(2024, 1, 1)),
            _sell("A", 3.25, 50.0, date(2024, 3, 1)),
        ]
        book = tx.build_lot_book(txs)
        remaining = book.open_lots["A"][0]
        assert remaining.shares == pytest.approx(10.5 - 3.25)
        assert remaining.cost_basis == pytest.approx((10.5 - 3.25) * 40.0)
        assert book.realized_gains[0].proceeds == pytest.approx(3.25 * 50.0)

    def test_independent_symbols_do_not_leak_lots(self) -> None:
        txs = [
            _buy("A", 10.0, 10.0, date(2024, 1, 1)),
            _buy("B", 5.0, 20.0, date(2024, 1, 1)),
            _sell("A", 10.0, 12.0, date(2024, 2, 1)),
        ]
        book = tx.build_lot_book(txs)
        assert "A" not in book.open_lots
        assert book.open_lots["B"][0].shares == pytest.approx(5.0)


class TestSplits:
    def test_split_scales_shares_and_preserves_total_cost_basis(self) -> None:
        txs = [_buy("A", 100.0, 50.0, date(2024, 1, 1))]  # $5000 total
        splits = [tx.StockSplit(symbol="A", date=date(2024, 6, 1), ratio=2.0)]
        book = tx.build_lot_book(txs, splits=splits)

        lot = book.open_lots["A"][0]
        assert lot.shares == pytest.approx(200.0)
        assert lot.cost_basis == pytest.approx(5000.0)  # unchanged in dollars
        assert lot.cost_basis / lot.shares == pytest.approx(25.0)  # cost/share halved

    def test_reverse_split(self) -> None:
        txs = [_buy("A", 100.0, 10.0, date(2024, 1, 1))]  # $1000 total
        splits = [tx.StockSplit(symbol="A", date=date(2024, 6, 1), ratio=0.5)]
        book = tx.build_lot_book(txs, splits=splits)

        lot = book.open_lots["A"][0]
        assert lot.shares == pytest.approx(50.0)
        assert lot.cost_basis == pytest.approx(1000.0)

    def test_split_applies_before_a_later_sell(self) -> None:
        txs = [
            _buy("A", 100.0, 50.0, date(2024, 1, 1)),
            _sell("A", 150.0, 30.0, date(2024, 7, 1)),  # only sellable post-split (200 shares)
        ]
        splits = [tx.StockSplit(symbol="A", date=date(2024, 6, 1), ratio=2.0)]
        book = tx.build_lot_book(txs, splits=splits)

        assert book.realized_gains[0].shares == pytest.approx(150.0)
        assert book.realized_gains[0].cost_basis == pytest.approx(150.0 * 25.0)  # post-split $25/sh
        assert book.open_lots["A"][0].shares == pytest.approx(50.0)

    def test_split_does_not_affect_a_lot_opened_before_it_that_was_already_sold(self) -> None:
        # A lot fully sold BEFORE the split shouldn't be resurrected or scaled.
        txs = [
            _buy("A", 10.0, 100.0, date(2024, 1, 1)),
            _sell("A", 10.0, 110.0, date(2024, 3, 1)),
        ]
        splits = [tx.StockSplit(symbol="A", date=date(2024, 6, 1), ratio=2.0)]
        book = tx.build_lot_book(txs, splits=splits)
        assert "A" not in book.open_lots
        assert book.realized_gains[0].shares == pytest.approx(10.0)  # not scaled

    def test_same_day_split_applies_before_that_days_buy(self) -> None:
        # Day-1 lot: 100 sh @ $100 (=$10,000). Same-day split 2-for-1, then a
        # FRESH buy also that day at the already-post-split price. The fresh
        # buy must NOT be additionally scaled by that day's split.
        txs = [
            _buy("A", 100.0, 100.0, date(2024, 1, 1)),
            _buy("A", 50.0, 52.0, date(2024, 6, 1)),
        ]
        splits = [tx.StockSplit(symbol="A", date=date(2024, 6, 1), ratio=2.0)]
        book = tx.build_lot_book(txs, splits=splits)

        lots = {lot.purchase_date: lot for lot in book.open_lots["A"]}
        assert lots[date(2024, 1, 1)].shares == pytest.approx(200.0)  # split-adjusted
        assert lots[date(2024, 1, 1)].cost_basis == pytest.approx(10_000.0)
        assert lots[date(2024, 6, 1)].shares == pytest.approx(50.0)  # untouched by the split
        assert lots[date(2024, 6, 1)].cost_basis == pytest.approx(2_600.0)


class TestPositions:
    def test_priced_position_reports_market_value_and_gain(self) -> None:
        book = tx.build_lot_book([_buy("A", 10.0, 100.0, date(2024, 1, 1))])
        result = tx.positions(book, current_prices={"A": 120.0})
        pos = result["A"]
        assert pos.shares == pytest.approx(10.0)
        assert pos.cost_basis == pytest.approx(1000.0)
        assert pos.average_cost == pytest.approx(100.0)
        assert pos.current_price == pytest.approx(120.0)
        assert pos.market_value == pytest.approx(1200.0)
        assert pos.unrealized_gain == pytest.approx(200.0)
        assert pos.is_stale is False

    def test_missing_price_degrades_gracefully(self) -> None:
        book = tx.build_lot_book([_buy("A", 10.0, 100.0, date(2024, 1, 1))])
        result = tx.positions(book, current_prices={})
        pos = result["A"]
        assert pos.current_price is None
        assert pos.market_value is None
        assert pos.unrealized_gain is None
        assert pos.is_stale is True
        # The position itself is still reported, not dropped.
        assert pos.shares == pytest.approx(10.0)

    def test_nan_or_non_positive_price_is_treated_as_stale(self) -> None:
        book = tx.build_lot_book([_buy("A", 10.0, 100.0, date(2024, 1, 1))])
        assert tx.positions(book, current_prices={"A": float("nan")})["A"].is_stale is True
        assert tx.positions(book, current_prices={"A": 0.0})["A"].is_stale is True
        assert tx.positions(book, current_prices={"A": -5.0})["A"].is_stale is True

    def test_infinite_quote_is_stale_not_an_infinite_portfolio_value(self) -> None:
        # A NaN-only screen let infinity through, so a bad quote rendered as an
        # infinite market value marked FRESH -- strictly worse than the stale
        # path this guard exists to take.
        book = tx.build_lot_book([_buy("A", 10.0, 100.0, date(2024, 1, 1))])
        position = tx.positions(book, current_prices={"A": float("inf")})["A"]
        assert position.is_stale is True
        assert position.current_price is None
        assert position.market_value is None

    def test_closed_position_does_not_appear(self) -> None:
        book = tx.build_lot_book(
            [_buy("A", 10.0, 100.0, date(2024, 1, 1)), _sell("A", 10.0, 110.0, date(2024, 3, 1))]
        )
        assert tx.positions(book, current_prices={"A": 110.0}) == {}

    def test_default_no_prices_marks_everything_stale(self) -> None:
        book = tx.build_lot_book([_buy("A", 10.0, 100.0, date(2024, 1, 1))])
        result = tx.positions(book)
        assert result["A"].is_stale is True
