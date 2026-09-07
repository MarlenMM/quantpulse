"""What the forward test holds, and what it has to trade to get there (Section 32).

Pure functions over frames: scores in, share counts out. No account, no
network, no storage -- so the decision can be tested, and inspected in a run's
record, without anything having been placed.

**The signal is the published composite rating**, which is the entire point of
the exercise. `analysis/backtest.py` cannot rank by it: five of the seven
categories have weeks of stored history rather than years, so a 2023 rebalance
ranked by them would be using 2026 data to pick 2023 stocks. The Track Record
page says so, and says the rating "becomes testable once the stored composite
history spans the window" -- 22 days of it, on 2026-09-07, against a backtest
window of 1,183. Forward testing is the way round that: it accumulates the
history instead of needing it to already exist, and a decision recorded before
its outcome is known cannot be fitted to that outcome.

The portfolio is long-only and equal-weight over the top `positions` names
rated Buy or Strong Buy, rebalanced weekly, in whole shares.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import floor, isfinite

import pandas as pd

__all__ = [
    "DEFAULT_POSITIONS",
    "ELIGIBLE_RATINGS",
    "Order",
    "orders_for",
    "target_positions",
    "turnover",
]

#: Only the two ratings that actually say "buy this". The forward test follows
#: the published rating rather than the ranking underneath it, so a Hold at the
#: 99th percentile in a weak market is still a Hold.
ELIGIBLE_RATINGS = frozenset({"strong_buy", "buy"})

#: How many names to hold, equal-weight.
#:
#: 20, and the reason is arithmetic. Whole-share sizing means the position count
#: and the account size together decide whether "equal weight" is real. Measured
#: over the 151 names rated buy/strong_buy on 2026-09-05, against Alpaca's
#: default $100,000 paper account:
#:
#: ====  ========  =============  ===========
#:    N   $/name   median drift   cash idle
#: ====  ========  =============  ===========
#:   10   $10,000           0.7%        2.8%
#:   20    $5,000           1.1%        4.0%
#:   50    $2,000           3.7%        6.0%
#:  100    $1,000           6.5%       12.4%   (2 names round to zero shares)
#: ====  ========  =============  ===========
#:
#: So this deliberately does **not** match the backtest's top-20%-of-503: at 100
#: names the equal weighting is fiction, an eighth of the account never gets
#: invested, and the most expensive names silently drop out. 20 is where the
#: weighting is still real (about 1% off target) while holding enough names to
#: be a portfolio rather than a bet.
DEFAULT_POSITIONS = 20


@dataclass(frozen=True)
class Order:
    """One whole-share market order. `qty` is positive; `side` carries direction."""

    symbol: str
    qty: int
    side: str  # "buy" | "sell"


def _usable_price(value: object) -> float | None:
    try:
        price = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return price if isfinite(price) and price > 0 else None


def target_positions(
    scores: pd.DataFrame,
    *,
    equity: float,
    positions: int = DEFAULT_POSITIONS,
) -> dict[str, int]:
    """`{symbol: share count}` for the portfolio the published ratings imply.

    `scores` carries symbol, rating, percentile_rank and price (the last stored
    close). Names are ranked by percentile, then by symbol so a tie cannot churn
    the book between two runs on identical inputs.

    A name whose slice does not cover one share is **left out rather than held
    at zero**, because a zero in this mapping would read downstream as a
    position that happens to be empty. A name with no usable price is left out
    too: sizing needs one, and one missing price must not cost the other
    nineteen their rebalance.
    """
    if scores.empty or equity <= 0 or positions <= 0:
        return {}
    eligible = scores[scores["rating"].isin(ELIGIBLE_RATINGS)]
    if eligible.empty:
        return {}
    ranked = eligible.sort_values(["percentile_rank", "symbol"], ascending=[False, True]).head(
        positions
    )

    slice_dollars = equity / positions
    target: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        price = _usable_price(row.price)
        if price is None:
            continue
        shares = floor(slice_dollars / price)
        if shares > 0:
            target[str(row.symbol)] = shares
    return target


def orders_for(current: Mapping[str, float], target: Mapping[str, int]) -> list[Order]:
    """The whole-share orders that move `current` to `target`, **sells first**.

    Sells first is not cosmetic. Alpaca checks buying power when an order is
    submitted, not when the batch settles, so a rebalance that funds its buys
    out of sells it has not placed yet has those buys rejected -- and the
    account ends the week holding only the things it sold out of.

    A short position is closed by buying it back. The strategy is long-only, so
    a short can only have arrived from a hand-placed trade in the same account,
    and leaving it open would put an unmanaged short inside a record labelled as
    this strategy's. A fractional holding sells only its whole part: the dust is
    left rather than the order rejected.
    """
    orders: list[Order] = []
    for symbol in sorted(set(current) | set(target)):
        held = float(current.get(symbol, 0.0))
        wanted = int(target.get(symbol, 0))
        # Truncate toward zero: never sell more than is held, never buy back
        # more than is short.
        delta = wanted - int(held)
        if delta > 0:
            orders.append(Order(symbol, delta, "buy"))
        elif delta < 0:
            orders.append(Order(symbol, -delta, "sell"))
    orders.sort(key=lambda order: (order.side != "sell", order.symbol))
    return orders


def turnover(current: Mapping[str, float], target: Mapping[str, int]) -> float:
    """Share-weighted fraction of the book this rebalance changes, 0-1.

    Recorded with each run so a week that churned the whole portfolio is
    distinguishable afterwards from one that swapped two names -- turnover is
    what a forward test pays for in spread, and the backtest's own cost
    assumption is stated for exactly this reason (Section 7.6).

    Shares rather than dollars, because it is computed from the same two
    mappings the orders come from and needs no price to be honest about what it
    is measuring. Zero for a rebalance with nothing on either side.
    """
    traded = 0.0
    total = 0.0
    for symbol in set(current) | set(target):
        held = float(current.get(symbol, 0.0))
        wanted = float(target.get(symbol, 0))
        traded += abs(wanted - held)
        total += max(abs(held), abs(wanted))
    return traded / total if total else 0.0
