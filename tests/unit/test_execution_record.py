"""Summarising the forward test's record without overstating how long it is."""

from datetime import date, timedelta

import pandas as pd
import pytest

from quantpulse.execution import record


def _history(*equities: float, benchmark: list[float] | None = None) -> pd.DataFrame:
    start = date(2026, 9, 7)
    return pd.DataFrame(
        [
            {
                "run_date": start + timedelta(days=i),
                "equity": equity,
                "benchmark_close": (benchmark[i] if benchmark else None),
                "rebalanced": i == 0,
                "signal_name": "composite_rating",
            }
            for i, equity in enumerate(equities)
        ]
    )


def test_an_empty_record_summarises_to_nothing_rather_than_zero() -> None:
    """Nobody has set the Alpaca secrets is the normal state, and "0.0% return"
    would be a claim about a strategy that has never held anything."""
    summary = record.summarise(pd.DataFrame())
    assert summary.n_snapshots == 0
    assert summary.total_return is None
    assert summary.first_date is None and summary.is_meaningful is False


def test_one_snapshot_is_not_yet_a_return() -> None:
    summary = record.summarise(_history(100_000.0))
    assert summary.n_snapshots == 1
    assert summary.total_return is None


def test_two_snapshots_give_a_total_return() -> None:
    summary = record.summarise(_history(100_000.0, 101_000.0))
    assert summary.total_return == pytest.approx(0.01)
    assert summary.n_snapshots == 2


def test_the_return_is_never_annualised() -> None:
    """The exact bug `MIN_TRACK_RECORD_PERIODS` exists for, in a new place.

    Two monthly periods spanning 35 days were once published as a 26.6% CAGR.
    A +1% move over two days annualises to over 500%; a total return over a
    stated window needs no periods-per-year assumption and cannot do that.
    """
    summary = record.summarise(_history(100_000.0, 101_000.0))
    assert summary.total_return < 0.02
    assert not hasattr(summary, "cagr")


def test_a_short_record_is_marked_as_not_yet_meaningful(self=None) -> None:
    summary = record.summarise(_history(*[100_000.0 + i for i in range(5)]))
    assert summary.n_snapshots == 5
    assert summary.is_meaningful is False
    assert summary.total_return is not None, "the number is still shown, just labelled"


def test_a_long_enough_record_is_marked_meaningful() -> None:
    summary = record.summarise(
        _history(*[100_000.0 + i for i in range(record.MIN_FORWARD_TEST_DAYS)])
    )
    assert summary.is_meaningful is True


def test_the_benchmark_is_measured_over_the_same_span() -> None:
    """Comparing a strategy's whole window against a benchmark's different one
    is how a comparison flatters itself."""
    summary = record.summarise(_history(100_000.0, 110_000.0, benchmark=[5000.0, 5100.0]))
    assert summary.total_return == pytest.approx(0.10)
    assert summary.benchmark_total_return == pytest.approx(0.02)


def test_a_missing_benchmark_is_none_not_zero(self=None) -> None:
    """A null benchmark must not read as "the index went nowhere"."""
    summary = record.summarise(_history(100_000.0, 110_000.0))
    assert summary.benchmark_total_return is None


def test_one_benchmark_close_is_not_a_benchmark_return() -> None:
    """A single close compared against itself is exactly 0.0%, which on a page
    reads as "the index went nowhere while the strategy moved" -- the most
    flattering possible misreading, and a fabricated one."""
    summary = record.summarise(_history(100.0, 110.0, benchmark=[5000.0, None]))
    assert summary.benchmark_total_return is None
    assert summary.total_return == pytest.approx(0.10)


def test_the_benchmark_uses_the_first_and_last_dates_that_have_one(self=None) -> None:
    summary = record.summarise(_history(100.0, 110.0, 120.0, benchmark=[None, 5000.0, 5100.0]))
    assert summary.benchmark_total_return == pytest.approx(0.02)


def test_a_zero_starting_equity_does_not_divide_by_zero() -> None:
    assert record.summarise(_history(0.0, 100.0)).total_return is None


def test_rebalances_are_counted(self=None) -> None:
    history = _history(100.0, 110.0, 120.0)
    assert record.summarise(history).rebalances == 1


def test_the_signal_is_carried_through(self=None) -> None:
    """The whole point of the exercise: this record ranks what the backtest
    cannot."""
    assert record.summarise(_history(100.0, 110.0)).signal_name == "composite_rating"
