"""A portfolio's history: what it was worth, how it did, and what it paid you.

The Portfolio Manager knows FIFO tax lots, three optimisers, correlation
clusters, VaR and sector gaps — every one of them as of *now*. None of it has a
sense of time, and the transaction ledger needed to give it one has been stored
since Phase 10.

The methodology matters more here than the arithmetic. A portfolio that is fed
money cannot be compared against an index by putting their values side by side:
buying £10,000 of stock raises the portfolio's value by £10,000 and says nothing
about performance. So the comparison is a **time-weighted return**, which chains
each period's return with that period's cash flow removed — the standard way to
measure a portfolio against a benchmark precisely because it is blind to when
money arrived.
"""

from datetime import date

import pandas as pd
import pytest

from quantpulse.portfolio import performance
from quantpulse.portfolio.transactions import Transaction


def _prices(**series: list[float]) -> pd.DataFrame:
    """A price panel: dates down, symbols across."""
    index = pd.to_datetime([date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)])
    return pd.DataFrame(series, index=index)


def _buy(symbol: str, shares: float, price: float, day: date) -> Transaction:
    return Transaction(symbol=symbol, action="buy", shares=shares, price=price, date=day)


def _sell(symbol: str, shares: float, price: float, day: date) -> Transaction:
    return Transaction(symbol=symbol, action="sell", shares=shares, price=price, date=day)


class TestSharesHeld:
    def test_shares_step_up_on_the_day_of_a_buy(self) -> None:
        held = performance.shares_held(
            [_buy("AAA", 10, 100.0, date(2026, 1, 6))], _prices(AAA=[1, 1, 1, 1]).index
        )
        assert list(held["AAA"]) == [0.0, 10.0, 10.0, 10.0]

    def test_shares_step_down_on_the_day_of_a_sell(self) -> None:
        held = performance.shares_held(
            [
                _buy("AAA", 10, 100.0, date(2026, 1, 5)),
                _sell("AAA", 4, 120.0, date(2026, 1, 7)),
            ],
            _prices(AAA=[1, 1, 1, 1]).index,
        )
        assert list(held["AAA"]) == [10.0, 10.0, 6.0, 6.0]

    def test_a_trade_before_the_window_is_already_held_on_day_one(self) -> None:
        """The window is the price history available, not the life of the
        portfolio. A position opened in 2024 is held throughout a 2026 window,
        not absent from it."""
        held = performance.shares_held(
            [_buy("AAA", 10, 100.0, date(2024, 5, 1))], _prices(AAA=[1, 1, 1, 1]).index
        )
        assert list(held["AAA"]) == [10.0, 10.0, 10.0, 10.0]

    def test_a_trade_after_the_window_never_appears(self) -> None:
        held = performance.shares_held(
            [_buy("AAA", 10, 100.0, date(2027, 5, 1))], _prices(AAA=[1, 1, 1, 1]).index
        )
        assert list(held["AAA"]) == [0.0, 0.0, 0.0, 0.0]

    def test_an_empty_ledger_holds_nothing(self) -> None:
        assert performance.shares_held([], _prices(AAA=[1, 1, 1, 1]).index).empty


class TestValueSeries:
    def test_value_is_shares_times_price_each_day(self) -> None:
        value = performance.value_series(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))], _prices(AAA=[100, 110, 90, 105])
        )
        assert list(value) == [1000.0, 1100.0, 900.0, 1050.0]

    def test_two_holdings_add_up(self) -> None:
        value = performance.value_series(
            [
                _buy("AAA", 10, 100.0, date(2026, 1, 5)),
                _buy("BBB", 5, 20.0, date(2026, 1, 5)),
            ],
            _prices(AAA=[100, 110, 90, 105], BBB=[20, 20, 20, 20]),
        )
        assert list(value) == [1100.0, 1200.0, 1000.0, 1150.0]

    def test_a_symbol_with_no_price_column_is_skipped_not_zeroed(self) -> None:
        """A delisted or never-ingested holding must not silently value the
        portfolio as if that position had become worthless."""
        value = performance.value_series(
            [
                _buy("AAA", 10, 100.0, date(2026, 1, 5)),
                _buy("GONE", 5, 20.0, date(2026, 1, 5)),
            ],
            _prices(AAA=[100, 100, 100, 100]),
        )
        assert list(value) == [1000.0] * 4

    def test_an_empty_ledger_is_an_empty_series(self) -> None:
        assert performance.value_series([], _prices(AAA=[1, 1, 1, 1])).empty


class TestCashFlows:
    def test_a_buy_is_money_into_the_portfolio(self) -> None:
        flows = performance.cash_flows(
            [_buy("AAA", 10, 100.0, date(2026, 1, 6))], _prices(AAA=[1, 1, 1, 1]).index
        )
        assert list(flows) == [0.0, 1000.0, 0.0, 0.0]

    def test_a_sell_is_money_out(self) -> None:
        flows = performance.cash_flows(
            [_sell("AAA", 4, 125.0, date(2026, 1, 7))], _prices(AAA=[1, 1, 1, 1]).index
        )
        assert list(flows) == [0.0, 0.0, -500.0, 0.0]

    def test_flows_use_the_traded_price_not_the_closing_price(self) -> None:
        """The ledger records what was actually paid. Valuing the flow at the
        day's close would book the intraday difference as performance."""
        flows = performance.cash_flows(
            [_buy("AAA", 10, 97.0, date(2026, 1, 6))], _prices(AAA=[100, 100, 100, 100]).index
        )
        assert flows.iloc[1] == pytest.approx(970.0)


class TestTimeWeightedReturn:
    """The measure that makes a benchmark comparison honest."""

    def test_a_portfolio_with_no_flows_tracks_its_holding(self) -> None:
        twr = performance.time_weighted_return(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))], _prices(AAA=[100, 110, 90, 105])
        )
        # Cumulative growth of 1.0 invested on day one: 100 -> 105 is +5%.
        assert twr.iloc[0] == pytest.approx(1.0)
        assert twr.iloc[-1] == pytest.approx(1.05)

    def test_depositing_money_is_not_a_return(self) -> None:
        """The entire reason this is not "portfolio value versus index level".

        Buying a second position doubles the portfolio's value and has earned
        nothing. A naive value-ratio would report +100% for that day.
        """
        transactions = [
            _buy("AAA", 10, 100.0, date(2026, 1, 5)),
            _buy("BBB", 10, 100.0, date(2026, 1, 6)),
        ]
        prices = _prices(AAA=[100, 100, 100, 100], BBB=[100, 100, 100, 100])
        twr = performance.time_weighted_return(transactions, prices)
        assert list(twr) == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_withdrawing_money_is_not_a_loss(self) -> None:
        transactions = [
            _buy("AAA", 10, 100.0, date(2026, 1, 5)),
            _sell("AAA", 5, 100.0, date(2026, 1, 6)),
        ]
        twr = performance.time_weighted_return(transactions, _prices(AAA=[100, 100, 100, 100]))
        assert list(twr) == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_a_gain_alongside_a_deposit_is_measured_cleanly(self) -> None:
        """Both happening at once is where a naive measure goes furthest wrong."""
        transactions = [
            _buy("AAA", 10, 100.0, date(2026, 1, 5)),
            _buy("BBB", 10, 110.0, date(2026, 1, 6)),
        ]
        # AAA rises 10% on day two; the BBB purchase arrives the same day.
        prices = _prices(AAA=[100, 110, 110, 110], BBB=[110, 110, 110, 110])
        twr = performance.time_weighted_return(transactions, prices)
        assert twr.iloc[1] == pytest.approx(1.10)
        assert twr.iloc[-1] == pytest.approx(1.10)

    def test_a_day_starting_from_nothing_contributes_no_return(self) -> None:
        """Dividing by a zero opening value would be an infinite return; the
        first day of a portfolio's life is exactly that case."""
        twr = performance.time_weighted_return(
            [_buy("AAA", 10, 100.0, date(2026, 1, 6))], _prices(AAA=[100, 100, 120, 120])
        )
        assert twr.iloc[1] == pytest.approx(1.0)
        assert twr.iloc[-1] == pytest.approx(1.2)

    def test_an_empty_ledger_returns_an_empty_series(self) -> None:
        assert performance.time_weighted_return([], _prices(AAA=[1, 1, 1, 1])).empty

    def test_a_net_short_book_does_not_produce_a_sign_flipped_return(self) -> None:
        """A negative opening value is reachable: `shares_held` records a sell
        with no matching buy as a negative position, which a hand-entered or
        CSV-restored ledger can contain. Dividing by it would flip the sign of
        every subsequent period, drawing a loss as a gain.
        """
        transactions = [_sell("AAA", 10, 100.0, date(2026, 1, 5))]
        twr = performance.time_weighted_return(transactions, _prices(AAA=[100, 110, 120, 130]))
        assert (twr > 0).all(), twr.tolist()
        assert twr.iloc[-1] == pytest.approx(1.0)


class TestBenchmarkComparison:
    def test_the_benchmark_is_rebased_to_the_same_start(self) -> None:
        """Both curves start at 1.0 so the comparison is like-for-like."""
        transactions = [_buy("AAA", 10, 100.0, date(2026, 1, 5))]
        prices = _prices(AAA=[100, 110, 90, 105])
        benchmark = pd.Series([4000.0, 4040.0, 4000.0, 4080.0], index=prices.index)
        frame = performance.compare_to_benchmark(transactions, prices, benchmark)
        assert frame["portfolio"].iloc[0] == pytest.approx(1.0)
        assert frame["benchmark"].iloc[0] == pytest.approx(1.0)
        assert frame["benchmark"].iloc[-1] == pytest.approx(1.02)

    def test_a_benchmark_gap_is_carried_forward_not_interpolated(self) -> None:
        """A missing index close must not invent a level between two real ones.

        The levels either side of the gap deliberately differ. An earlier
        version bracketed the gap with 4000 and 4000, where carrying forward and
        interpolating give the same answer — so the test passed with either and
        proved nothing about the behaviour it is named for.
        """
        transactions = [_buy("AAA", 10, 100.0, date(2026, 1, 5))]
        prices = _prices(AAA=[100, 100, 100, 100])
        benchmark = pd.Series([4000.0, float("nan"), 4200.0, 4200.0], index=prices.index)
        frame = performance.compare_to_benchmark(transactions, prices, benchmark)
        assert frame["benchmark"].iloc[1] == pytest.approx(1.0), "carried forward, not 1.025"

    def test_an_empty_portfolio_gets_no_benchmark_column_either(self) -> None:
        """A benchmark curve beside no portfolio curve is a chart of the index
        wearing a portfolio's label."""
        prices = _prices(AAA=[100, 110, 90, 105])
        benchmark = pd.Series([4000.0, 4040.0, 4000.0, 4080.0], index=prices.index)
        frame = performance.compare_to_benchmark([], prices, benchmark)
        assert "benchmark" not in frame.columns

    def test_no_benchmark_data_leaves_the_column_absent(self) -> None:
        """ "The index was flat" and "we have no index" are different claims."""
        frame = performance.compare_to_benchmark(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))],
            _prices(AAA=[100, 110, 90, 105]),
            pd.Series(dtype=float),
        )
        assert "benchmark" not in frame.columns
        assert "portfolio" in frame.columns


class TestDividends:
    """What the portfolio actually paid you, from shares held on each ex-date."""

    def _divs(self, rows: list[tuple[str, date, float]]) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": s, "ex_date": d, "amount": a} for s, d, a in rows])

    def test_it_pays_on_the_shares_held_on_the_ex_date(self) -> None:
        paid = performance.dividend_income(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))],
            self._divs([("AAA", date(2026, 1, 7), 0.25)]),
        )
        assert len(paid) == 1
        assert paid.iloc[0]["income"] == pytest.approx(2.50)

    def test_a_dividend_before_you_owned_it_pays_nothing(self) -> None:
        """Buying after the ex-date does not entitle you to that dividend, and
        crediting it would invent income the account never received."""
        paid = performance.dividend_income(
            [_buy("AAA", 10, 100.0, date(2026, 1, 8))],
            self._divs([("AAA", date(2026, 1, 7), 0.25)]),
        )
        assert paid.empty

    def test_a_dividend_after_you_sold_pays_nothing(self) -> None:
        paid = performance.dividend_income(
            [
                _buy("AAA", 10, 100.0, date(2026, 1, 5)),
                _sell("AAA", 10, 100.0, date(2026, 1, 6)),
            ],
            self._divs([("AAA", date(2026, 1, 7), 0.25)]),
        )
        assert paid.empty

    def test_a_partial_sale_pays_on_what_is_left(self) -> None:
        paid = performance.dividend_income(
            [
                _buy("AAA", 10, 100.0, date(2026, 1, 5)),
                _sell("AAA", 6, 100.0, date(2026, 1, 6)),
            ],
            self._divs([("AAA", date(2026, 1, 7), 0.25)]),
        )
        assert paid.iloc[0]["income"] == pytest.approx(1.00)

    def test_income_is_summed_per_symbol_across_pay_dates(self) -> None:
        paid = performance.dividend_income(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))],
            self._divs([("AAA", date(2026, 1, 6), 0.25), ("AAA", date(2026, 1, 7), 0.30)]),
        )
        assert len(paid) == 2
        assert paid["income"].sum() == pytest.approx(5.50)

    def test_no_dividend_history_is_an_empty_frame_not_zero_income(self) -> None:
        paid = performance.dividend_income(
            [_buy("AAA", 10, 100.0, date(2026, 1, 5))], pd.DataFrame()
        )
        assert paid.empty
