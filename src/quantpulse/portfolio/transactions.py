"""Transaction log & FIFO tax-lot bookkeeping (Section 30).

Section 21 rates this row Sonnet/Medium ("well-defined accounting logic, but
many small edge cases worth care"). Section 30 is explicit about the design:
log every buy/sell as its own row and derive current holdings FROM that log,
never maintain a holdings snapshot as separately-edited state -- so the engine
here takes a plain transaction list in and produces open tax lots + realized
gains out, rather than treating "current shares held" as its own mutable
concept that could drift from the log that's supposed to be its source of
truth.

Three edge cases Section 30 names by name, each handled explicitly:

* **FIFO.** A sell consumes the OLDEST open lots first. A single sell can span
  several lots bought on different dates at different prices -- and therefore
  can realize gains at different short-/long-term holding periods in one
  transaction -- so `_consume_fifo` reports one `RealizedGain` per lot touched,
  not one blended number per sell.
* **Fractional shares.** Every share count here is a float from the start
  (Section 30: "make the `shares` field a decimal, not an integer, from day
  one") -- lots can be partially consumed, split, and partially consumed again
  without ever needing to round to a whole share.
* **Stock splits.** A split scales every open lot's `shares` by the split
  ratio while leaving its total `cost_basis` unchanged (a 2-for-1 split
  doubles your share count and halves your per-share cost, but the dollars you
  actually paid don't change). Splits and transactions for a symbol are merged
  into one date-ordered event stream and applied in that order, with a
  same-day split applied *before* that day's transactions -- a split takes
  effect before that day's trading, so a same-day buy is booked fresh at the
  post-split price it was actually paid, not retroactively rescaled.

**Deliberately pure, no schema yet** -- like every other Phase 8 row before
this one (`risk.py`, `optimization.py`, `rebalancing.py`), this operates on a
plain list of `Transaction` records the caller supplies, not the
`portfolio_transactions` table. Section 13's `portfolio_holdings` /
`portfolio_transactions` tables and the session-vs-sqlite backend switch
(Section 4.5) belong to `portfolio/holdings.py`, whichever row builds the
Portfolio Manager's actual persistence -- adding that schema now, before
anything reads or writes it, would be exactly the "don't add schema for a
later part to populate" trap this project already avoids.

**Deliberately not tax advice (Section 2).** `holding_term` labels a lot
short-/long-term using the plain "more than one year" rule as a purely
descriptive flag next to a position -- there is no wash-sale handling, no tax
bracket math, and no claim about actual tax liability anywhere in this module.

`Position.is_stale` is Section 30's fourth bullet ("delisted or acquired
holdings... fail gracefully"): `positions()` never raises for a symbol missing
from `current_prices` -- it reports the position with `current_price=None`
and `is_stale=True` so a caller can show a "delisted/inactive" flag on that
one row instead of losing the whole page.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

__all__ = [
    "Transaction",
    "StockSplit",
    "TaxLot",
    "RealizedGain",
    "Position",
    "LotBook",
    "holding_term",
    "build_lot_book",
    "positions",
]

# A lot (or leftover fraction of one) below this many shares is floating-point
# residue from repeated partial consumption/splitting, not a real remaining
# position -- dropped rather than lingering as economically meaningless dust.
_MIN_LOT_SHARES = 1e-9


@dataclass(frozen=True)
class Transaction:
    """One buy or sell, in the shape of a Section 13 `portfolio_transactions` row.

    `price` is the actual per-share price paid/received on `date`, in the
    terms in effect that day -- if a later split changes the economics of
    shares bought earlier, that's `StockSplit`'s job, not this record's.
    """

    symbol: str
    action: Literal["buy", "sell"]
    shares: float
    price: float
    date: date


@dataclass(frozen=True)
class StockSplit:
    """A split (or reverse split): every open lot's shares scale by `ratio`.

    `ratio=2.0` is a standard 2-for-1 split (shares double, cost/share halves);
    `ratio=0.5` is a 1-for-2 reverse split. Total dollar cost basis is
    invariant under a split -- only the per-share economics change.
    """

    symbol: str
    date: date
    ratio: float


@dataclass(frozen=True)
class TaxLot:
    """One still-open purchase lot: `shares` bought on `purchase_date` for `cost_basis` total.

    `cost_basis` is the lot's TOTAL remaining cost, not a per-share figure --
    dividing by `shares` gives the average cost, and it stays exact through
    partial sells and splits without ever rounding to a per-share number.
    """

    symbol: str
    purchase_date: date
    shares: float
    cost_basis: float


@dataclass(frozen=True)
class RealizedGain:
    """The realized gain/loss from selling (all or part of) one lot.

    A single sell that spans several lots produces one `RealizedGain` per lot
    touched (FIFO, oldest first) -- so a sell can realize a long-term gain on
    an old lot and a short-term gain on a recent one in the same transaction,
    and both are reported separately rather than blended into one number that
    would obscure which is which.

    `term` is Section 9's short-/long-term holding-period flag, purely
    descriptive (Section 2: informational only, never tax advice).
    """

    symbol: str
    sale_date: date
    purchase_date: date
    shares: float
    proceeds: float
    cost_basis: float
    gain: float
    term: Literal["short", "long"]


@dataclass(frozen=True)
class Position:
    """A symbol's current open holdings, derived from its still-open lots.

    `current_price` / `market_value` / `unrealized_gain` are `None` and
    `is_stale=True` when `current_prices` had nothing usable for this symbol
    -- Section 30's "delisted or acquired holdings... fail gracefully" --
    rather than the whole snapshot raising over one bad ticker.
    """

    symbol: str
    shares: float
    cost_basis: float
    average_cost: float
    current_price: float | None
    market_value: float | None
    unrealized_gain: float | None
    is_stale: bool


@dataclass(frozen=True)
class LotBook:
    """The full result of processing a transaction log: open lots + realized history.

    `open_lots[symbol]` is oldest-first (FIFO order), so a caller inspecting it
    directly (e.g. to show each open lot's own holding-period via
    `holding_term`) sees lots in the order they would actually be sold.
    `realized_gains` is sorted chronologically by sale date.
    """

    open_lots: dict[str, list[TaxLot]]
    realized_gains: list[RealizedGain]


def holding_term(purchase_date: date, *, as_of: date) -> Literal["short", "long"]:
    """Section 9's short-/long-term holding-period flag: "more than one year" (informational only).

    Uses the actual calendar rule (more than one year, not a flat 365-day
    count) so a purchase spanning a leap day isn't mislabeled by a day.
    Purely descriptive -- consult a tax professional (Section 2); this is not
    a tax computation.
    """
    try:
        one_year_later = purchase_date.replace(year=purchase_date.year + 1)
    except ValueError:
        # purchase_date was Feb 29 and the following year has no Feb 29.
        one_year_later = purchase_date.replace(year=purchase_date.year + 1, day=28)
    return "long" if as_of > one_year_later else "short"


def _apply_split(lot: TaxLot, ratio: float) -> TaxLot:
    return TaxLot(
        symbol=lot.symbol,
        purchase_date=lot.purchase_date,
        shares=lot.shares * ratio,
        cost_basis=lot.cost_basis,
    )


def _consume_fifo(lots: list[TaxLot], sell: Transaction) -> tuple[list[TaxLot], list[RealizedGain]]:
    """Consume `sell.shares` from `lots` oldest-first; returns the survivors + gains realized."""
    remaining = sell.shares
    gains: list[RealizedGain] = []
    survivors: list[TaxLot] = []

    for lot in lots:
        if remaining <= _MIN_LOT_SHARES:
            survivors.append(lot)
            continue
        consumed = min(lot.shares, remaining)
        consumed_cost = lot.cost_basis * (consumed / lot.shares)
        proceeds = consumed * sell.price
        gains.append(
            RealizedGain(
                symbol=sell.symbol,
                sale_date=sell.date,
                purchase_date=lot.purchase_date,
                shares=consumed,
                proceeds=proceeds,
                cost_basis=consumed_cost,
                gain=proceeds - consumed_cost,
                term=holding_term(lot.purchase_date, as_of=sell.date),
            )
        )
        remaining -= consumed
        leftover_shares = lot.shares - consumed
        if leftover_shares > _MIN_LOT_SHARES:
            survivors.append(
                TaxLot(
                    symbol=lot.symbol,
                    purchase_date=lot.purchase_date,
                    shares=leftover_shares,
                    cost_basis=lot.cost_basis - consumed_cost,
                )
            )

    if remaining > _MIN_LOT_SHARES:
        held = sell.shares - remaining
        raise ValueError(
            f"cannot sell {sell.shares:.6f} shares of {sell.symbol} on {sell.date}: "
            f"only {held:.6f} were held (long-only, Section 2)"
        )
    return survivors, gains


def build_lot_book(
    transactions: Sequence[Transaction], *, splits: Sequence[StockSplit] = ()
) -> LotBook:
    """Replay `transactions` (and any `splits`) in date order into open lots + realized gains.

    Transactions and splits for each symbol are merged into one chronological
    stream; a split dated the same day as a transaction is applied first (see
    the module docstring), so a same-day buy is booked at the price it was
    actually paid. Ties among same-day, same-type events keep the relative
    order they were passed in, so a caller relying on list order for genuine
    same-day ambiguity gets a deterministic result.

    Raises `ValueError` if any transaction has non-positive shares/price, any
    split has a non-positive ratio, an unrecognized `action`, or a sell
    requests more shares than were open at that point (long-only, Section 2).
    """
    by_symbol_tx: dict[str, list[Transaction]] = {}
    for tx in transactions:
        if tx.shares <= 0:
            raise ValueError(f"transaction shares must be > 0, got {tx.shares} for {tx.symbol}")
        if tx.price <= 0:
            raise ValueError(f"transaction price must be > 0, got {tx.price} for {tx.symbol}")
        if tx.action not in ("buy", "sell"):
            raise ValueError(f"transaction action must be 'buy' or 'sell', got {tx.action!r}")
        by_symbol_tx.setdefault(tx.symbol, []).append(tx)

    by_symbol_splits: dict[str, list[StockSplit]] = {}
    for sp in splits:
        if sp.ratio <= 0:
            raise ValueError(f"split ratio must be > 0, got {sp.ratio} for {sp.symbol}")
        by_symbol_splits.setdefault(sp.symbol, []).append(sp)

    open_lots: dict[str, list[TaxLot]] = {}
    realized: list[RealizedGain] = []

    for symbol in sorted(set(by_symbol_tx) | set(by_symbol_splits)):
        # Sort key: (date, split-before-transaction, original list position) --
        # never the event object itself, so ties never fall back to comparing
        # two dataclasses that have no ordering defined.
        events: list[tuple[date, int, int, Transaction | StockSplit]] = [
            (sp.date, 0, i, sp) for i, sp in enumerate(by_symbol_splits.get(symbol, []))
        ] + [(tx.date, 1, i, tx) for i, tx in enumerate(by_symbol_tx.get(symbol, []))]
        events.sort(key=lambda e: (e[0], e[1], e[2]))

        lots: list[TaxLot] = []
        for _, _, _, event in events:
            if isinstance(event, StockSplit):
                lots = [_apply_split(lot, event.ratio) for lot in lots]
            elif event.action == "buy":
                lots.append(
                    TaxLot(
                        symbol=symbol,
                        purchase_date=event.date,
                        shares=event.shares,
                        cost_basis=event.shares * event.price,
                    )
                )
            else:
                lots, gains = _consume_fifo(lots, event)
                realized.extend(gains)

        remaining_lots = [lot for lot in lots if lot.shares > _MIN_LOT_SHARES]
        if remaining_lots:
            open_lots[symbol] = remaining_lots

    realized.sort(key=lambda g: (g.sale_date, g.symbol, g.purchase_date))
    return LotBook(open_lots=open_lots, realized_gains=realized)


def positions(
    lot_book: LotBook, *, current_prices: Mapping[str, float] | None = None
) -> dict[str, Position]:
    """The current holdings snapshot Section 13's `portfolio_holdings` derives from `lot_book`.

    A symbol missing (or non-positive/NaN) in `current_prices` gets
    `current_price=None`, `market_value=None`, `unrealized_gain=None`, and
    `is_stale=True` -- Section 30's "fail gracefully... rather than erroring
    out the whole Portfolio page over one bad ticker" -- instead of raising.
    Only symbols with open lots are returned; a fully-closed position doesn't
    appear (there is nothing current to report).
    """
    prices = current_prices or {}
    result: dict[str, Position] = {}
    for symbol, lots in lot_book.open_lots.items():
        shares = sum(lot.shares for lot in lots)
        cost_basis = sum(lot.cost_basis for lot in lots)
        raw_price = prices.get(symbol)
        price = (
            float(raw_price)
            if raw_price is not None and not math.isnan(raw_price) and raw_price > 0
            else None
        )
        market_value = shares * price if price is not None else None
        result[symbol] = Position(
            symbol=symbol,
            shares=shares,
            cost_basis=cost_basis,
            average_cost=cost_basis / shares,
            current_price=price,
            market_value=market_value,
            unrealized_gain=(market_value - cost_basis) if market_value is not None else None,
            is_stale=price is None,
        )
    return result
