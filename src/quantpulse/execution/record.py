"""Reading the forward test's record back, without overstating how long it is.

Computed server-side, once, so the Streamlit page and the React page cannot
arrive at different numbers from the same rows -- the same rule that keeps the
Kelly fraction and the Strong-Buy cutoff out of TypeScript.

**Nothing here annualises.** `backtest.MIN_TRACK_RECORD_PERIODS` exists because
two monthly periods spanning 35 calendar days were once published as a 26.6%
CAGR: arithmetically correct, describing nothing. A forward test starts at one
day long and grows by one a day, so it spends its entire first months in
exactly that trap. A total return over a stated window is the honest shape --
it needs no periods-per-year assumption, so it cannot turn a +1% week into a
headline.

What it does instead is say how long the record is, and mark whether that is
long enough to call a track record yet.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

__all__ = ["ForwardTestSummary", "MIN_FORWARD_TEST_DAYS", "summarise"]

#: Trading days before the record may be described as a track record rather
#: than as a start. About a month -- long enough to span a rebalance or four and
#: some ordinary market weather, short enough to be reached in the first season.
#:
#: Below it the numbers are still shown, deliberately. Hiding a measured figure
#: teaches a reader nothing, while showing it next to "8 trading days" teaches
#: them exactly how much to trust it; Section 22's rule is that an
#: honestly-labelled limitation beats a silently inflated number, not that
#: numbers should be withheld.
MIN_FORWARD_TEST_DAYS = 20


@dataclass(frozen=True)
class ForwardTestSummary:
    """Headline numbers for the forward test, with their own age attached."""

    n_snapshots: int
    first_date: date | None
    last_date: date | None
    start_equity: float | None
    latest_equity: float | None
    total_return: float | None
    benchmark_total_return: float | None
    rebalances: int
    signal_name: str | None
    is_meaningful: bool


def _total_return(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first <= 0:
        return None
    return last / first - 1.0


def summarise(history: pd.DataFrame) -> ForwardTestSummary:
    """Summarise `persistence.read_paper_trading_history`'s frame (oldest first).

    An empty record summarises to `None`s rather than to zeros: "0.0% return"
    is a claim about a strategy, and no strategy has run. That is the normal
    state of this table until someone sets the two Alpaca secrets.

    The benchmark is measured between the first and last rows **that have a
    close**, not over the strategy's own span, so a gap in the index series
    cannot silently compare the strategy's whole window against a shorter one
    for the benchmark.
    """
    if history.empty:
        return ForwardTestSummary(
            n_snapshots=0,
            first_date=None,
            last_date=None,
            start_equity=None,
            latest_equity=None,
            total_return=None,
            benchmark_total_return=None,
            rebalances=0,
            signal_name=None,
            is_meaningful=False,
        )

    equity = history["equity"].astype(float)
    start_equity = float(equity.iloc[0])
    latest_equity = float(equity.iloc[-1])

    benchmark = history["benchmark_close"].dropna().astype(float)
    benchmark_return = (
        _total_return(float(benchmark.iloc[0]), float(benchmark.iloc[-1]))
        if len(benchmark) >= 2
        else None
    )

    n = int(len(history))
    return ForwardTestSummary(
        n_snapshots=n,
        first_date=history["run_date"].iloc[0],
        last_date=history["run_date"].iloc[-1],
        start_equity=start_equity,
        latest_equity=latest_equity,
        total_return=_total_return(start_equity, latest_equity) if n >= 2 else None,
        benchmark_total_return=benchmark_return,
        rebalances=int(history["rebalanced"].astype(bool).sum()),
        signal_name=str(history["signal_name"].iloc[-1]),
        is_meaningful=n >= MIN_FORWARD_TEST_DAYS,
    )
