"""A portfolio's history: what it was worth, how it did, and what it paid you.

The Portfolio Manager knows FIFO tax lots, three optimisers, correlation
clusters, VaR, concentration and sector gaps — and every one of them as of
*now*. This module is the missing time axis, built from the transaction ledger
that has been stored since Phase 10 and needs no new input to do it.

**The benchmark comparison is a time-weighted return, and that is the point of
the module rather than a detail of it.** Putting a portfolio's value beside an
index level is the most natural-looking way to do this and it is wrong: buying
£10,000 of stock raises the portfolio's value by £10,000 and has earned
nothing, so a value-ratio reports a triumph for a deposit. A time-weighted
return chains each period's return with that period's cash flow removed, which
is exactly why it is the standard for measuring a portfolio *against* a
benchmark — it is blind to when money arrived, so what is left is the part the
holdings actually did.

**Cash is deliberately outside all of this.** `PortfolioState` carries a single
current cash balance, not a history of deposits and withdrawals, so there is no
honest way to reconstruct an uninvested balance on a past date. Everything here
therefore measures the *invested* portfolio, with buys and sells as its cash
flows, and the pages say so. Inventing a cash history would put a number on
screen that no record supports.

Pure functions over frames: ledger and prices in, series out. No database, no
network.
"""

from collections.abc import Sequence
from datetime import date

import pandas as pd

from quantpulse.portfolio.transactions import Transaction

__all__ = [
    "cash_flows",
    "compare_to_benchmark",
    "dividend_income",
    "shares_held",
    "time_weighted_return",
    "value_series",
]


def _as_timestamp(day: date) -> pd.Timestamp:
    return pd.Timestamp(day)


def shares_held(transactions: Sequence[Transaction], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares of each symbol held on each date in `index`.

    A trade dated before the window is already reflected on day one -- the
    window is however much price history exists, not the life of the portfolio,
    and a position opened in 2024 is *held* throughout a 2026 window rather than
    absent from it. A trade dated after the window never appears.
    """
    if not transactions or len(index) == 0:
        return pd.DataFrame(index=index)

    symbols = sorted({tx.symbol for tx in transactions})
    held = pd.DataFrame(0.0, index=index, columns=symbols)
    for tx in transactions:
        signed = tx.shares if tx.action == "buy" else -tx.shares
        when = _as_timestamp(tx.date)
        # Everything from the trade date onward. A trade before the window
        # starts is `>= index[0]` for every row, which is the intent.
        held.loc[index >= when, tx.symbol] += signed
    return held


def value_series(transactions: Sequence[Transaction], prices: pd.DataFrame) -> pd.Series:
    """Market value of the invested portfolio on each date in `prices`.

    A symbol with no price column is **skipped rather than valued at zero**. A
    delisted or never-ingested holding is a gap in what we know, and pricing it
    at zero would draw a portfolio that had suddenly lost that position's whole
    value -- the same distinction the Portfolio page already makes for a stale
    holding (Section 30).
    """
    held = shares_held(transactions, prices.index)
    if held.empty:
        return pd.Series(dtype=float)
    priced = [symbol for symbol in held.columns if symbol in prices.columns]
    if not priced:
        return pd.Series(0.0, index=prices.index)
    return (held[priced] * prices[priced]).sum(axis=1)


def cash_flows(transactions: Sequence[Transaction], index: pd.DatetimeIndex) -> pd.Series:
    """Net money into the invested portfolio on each date: buys positive, sells negative.

    Valued at the **traded** price the ledger recorded, not at that day's close.
    Using the close would book the difference between what was paid and where
    the stock finished as performance, which is the one thing a cash flow must
    never contribute.
    """
    flows = pd.Series(0.0, index=index)
    for tx in transactions:
        when = _as_timestamp(tx.date)
        if when not in flows.index:
            continue
        signed = tx.shares * tx.price
        flows.loc[when] += signed if tx.action == "buy" else -signed
    return flows


def time_weighted_return(transactions: Sequence[Transaction], prices: pd.DataFrame) -> pd.Series:
    """Cumulative time-weighted return, indexed like `prices` and starting at 1.0.

    Each day's return is `(value_end - flow) / value_start`, so money arriving
    that day is removed before the division and contributes nothing. Chaining
    those gives growth of a single unit invested at the start -- the number that
    can be laid beside a rebased benchmark without flattering either.

    A day whose *opening* value is zero contributes a return of exactly 1.0
    rather than an infinite one. That is not an edge case to be tolerated: it is
    the first day of every portfolio's life, and the day any fully-sold
    portfolio is re-entered.
    """
    value = value_series(transactions, prices)
    if value.empty:
        return pd.Series(dtype=float)
    flows = cash_flows(transactions, prices.index)

    factors = []
    previous = 0.0
    for when in prices.index:
        end, flow = float(value.loc[when]), float(flows.loc[when])
        factors.append(1.0 if previous <= 0 else (end - flow) / previous)
        previous = end
    return pd.Series(factors, index=prices.index).cumprod()


def compare_to_benchmark(
    transactions: Sequence[Transaction],
    prices: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Portfolio and benchmark as growth of 1.0 over the same dates.

    The benchmark is rebased to its first available level so both curves start
    together; a gap in the index series is carried forward rather than
    interpolated, because inventing a level between two real ones is inventing
    a day the market had.

    With no benchmark data the column is **absent**, not flat. "The index went
    nowhere" and "we have no index" are different claims and only one of them
    is ours to make.
    """
    portfolio = time_weighted_return(transactions, prices)
    frame = pd.DataFrame({"portfolio": portfolio})
    if benchmark.empty or portfolio.empty:
        return frame

    aligned = benchmark.reindex(prices.index).ffill()
    if aligned.dropna().empty:
        return frame
    base = float(aligned.dropna().iloc[0])
    if base <= 0:
        return frame
    frame["benchmark"] = aligned / base
    return frame


def dividend_income(transactions: Sequence[Transaction], dividends: pd.DataFrame) -> pd.DataFrame:
    """Cash dividends this portfolio earned: shares held on each ex-date × amount.

    `dividends` carries symbol, ex_date and amount (per share). Entitlement is
    decided by the shares held **on the ex-date** -- buying the day after pays
    nothing, and crediting it would invent income the account never received.

    Returns one row per (symbol, ex-date) that actually paid, so a caller can
    total it, chart it, or show which holdings produce the income. Empty when
    there is no dividend history, which is a different answer from zero income.
    """
    if dividends.empty or not transactions:
        return pd.DataFrame(columns=["symbol", "ex_date", "amount", "shares", "income"])

    rows = []
    for record in dividends.itertuples(index=False):
        ex_date = pd.Timestamp(record.ex_date).date()
        shares = sum(
            (tx.shares if tx.action == "buy" else -tx.shares)
            for tx in transactions
            if tx.symbol == record.symbol and tx.date <= ex_date
        )
        if shares <= 0:
            continue
        rows.append(
            {
                "symbol": record.symbol,
                "ex_date": ex_date,
                "amount": float(record.amount),
                "shares": float(shares),
                "income": float(shares) * float(record.amount),
            }
        )
    frame = pd.DataFrame(rows, columns=["symbol", "ex_date", "amount", "shares", "income"])
    return frame.sort_values(["ex_date", "symbol"]).reset_index(drop=True)
