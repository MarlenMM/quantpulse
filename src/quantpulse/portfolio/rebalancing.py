"""Rebalancing trade-list generation -- target allocation to concrete orders (Section 27).

Section 21 rates this row Sonnet/Medium ("arithmetic once a target allocation
exists") -- the hard part (getting to a defensible target allocation at all) is
`optimization.py`'s job; this module's only job is turning that target, plus
what the portfolio currently holds, into Section 27's own worked example:
**"sell 12 shares of X, buy 8 shares of Y"**, not an abstract weights chart.

Two things below aren't quite "just arithmetic," and are worth flagging even at
Medium effort:

* **Whole-share rounding cannot be allowed to oversell a position.** Rounding a
  sell to the nearest whole share can round its magnitude *up* -- e.g. a
  fractional holding of 11.6 shares fully exiting would round to a 12-share
  sell, which is impossible: you cannot sell shares you don't own. Every sell is
  clamped to the shares actually held *after* rounding, not before, so a
  requested full exit always empties the position exactly rather than either
  overselling or leaving a rounding-induced fractional remainder.
* **The turnover and cost figures describe what would actually trade, not the
  idealized target.** They're computed from the *achieved* post-rounding,
  post-filter allocation, using the exact same "transaction_cost per unit of
  turnover" convention `backtest.backtest_strategy` already uses (turnover in
  weight-fraction units; cost = rate * turnover * total value) -- one
  friction-cost convention across the whole project, not two.

`target_weights` is exactly the shape `optimization.OptimizedPortfolio.weights`
produces, so the natural call is
`build_rebalance_plan(current_shares, prices, optimized.weights, cash=...)`.
Cash is handled the same way `risk.portfolio_risk` handles it: not a key inside
the weights dict, but tracked alongside it (`cash_before`/`cash_after`) --
`target_weights` may sum to less than 1, and the remainder is the implicit
target cash weight.

Explicitly out of scope: FIFO tax-lot bookkeeping and the transaction ledger
that would supply real current share counts (Section 30, a separate row); this
takes a plain current-holdings snapshot as given. Nothing here executes a
trade or touches a brokerage (Section 2) -- it is a list to review, not an
order to submit. Pure function: holdings/prices/weights in, a plan out; no
storage or network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_TRANSACTION_COST",
    "Trade",
    "RebalancePlan",
    "build_rebalance_plan",
]

# The same bid-ask stand-in Section 7.6 specifies and `backtest.
# backtest_strategy`'s own `transaction_cost` default already uses ("0.1% per
# trade") -- one friction assumption, reused rather than re-invented here.
DEFAULT_TRANSACTION_COST = 0.001

# A trade below this many shares is floating-point residue, not a decision --
# dropped unconditionally. `min_trade_value` (a caller-supplied dollar floor)
# is a separate, opt-in filter on top of this one.
_MIN_SHARE_EPSILON = 1e-9

# Tolerance for the "target_weights sums to at most 1.0" check, so an
# already-normalized dict (e.g. straight from `optimization.py`) that lands at
# 1.0 plus float noise isn't rejected.
_WEIGHT_SUM_TOL = 1e-9


@dataclass(frozen=True)
class Trade:
    """One concrete order -- Section 27's own example: "sell 12 shares of X".

    `shares` is always a positive magnitude; direction lives in `action`, which
    reads naturally for display and matches the plan's own wording better than
    a signed share count would. `current_weight`/`target_weight`/
    `achieved_weight` are that symbol's share of `RebalancePlan.total_value`,
    so a caller can render "AAPL: 22% -> target 15%, landing at 15.1%" without
    recomputing anything.
    """

    symbol: str
    action: Literal["buy", "sell"]
    shares: float
    price: float
    trade_value: float  # shares * price, always positive
    current_weight: float
    target_weight: float
    achieved_weight: float


@dataclass(frozen=True)
class RebalancePlan:
    """The trade list plus enough context to judge it, not just execute it.

    `trades` is ordered sells-first-then-buys (largest first within each side)
    -- the natural reading order of Section 27's own example. `current_weights`
    / `target_weights` / `achieved_weights` cover every symbol in the
    current-or-target universe; cash is tracked separately via `cash_before` /
    `cash_after` (mirroring `risk.portfolio_risk`'s `cash_weight` split), and
    `sum(achieved_weights.values()) + cash_after / total_value == 1.0`
    exactly -- rebalancing reallocates value, it never creates or destroys it.

    `turnover` and `estimated_transaction_cost` describe the *achieved* trade
    list (after rounding and the `min_trade_value` filter), not the idealized
    target, using the same rate-times-turnover convention
    `backtest.backtest_strategy` already applies to its hypothetical strategy.
    """

    trades: list[Trade]
    total_value: float
    cash_before: float
    cash_after: float
    estimated_transaction_cost: float
    turnover: float
    whole_shares: bool
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    achieved_weights: dict[str, float]


def build_rebalance_plan(
    current_shares: Mapping[str, float],
    prices: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    cash: float = 0.0,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    whole_shares: bool = False,
    min_trade_value: float = 0.0,
) -> RebalancePlan | None:
    """The concrete buy/sell orders that move `current_shares` toward `target_weights`.

    `current_shares` and `target_weights` need not cover the same symbols: a
    symbol only in `target_weights` is a new position to open; a symbol only in
    `current_shares` (or held but weighted 0/omitted) is a full exit.
    `target_weights` must be non-negative and sum to at most 1.0 (long-only, no
    leverage -- Section 2); any shortfall is the implicit target cash weight.

    `whole_shares=True` rounds each trade to the nearest whole share for
    brokers that don't support fractional orders -- but a sell is always capped
    at the shares actually held, even if that leaves a fractional trade, since
    rounding a sell's magnitude *up* would require selling shares that don't
    exist (see the module docstring). `min_trade_value` optionally drops any
    trade below a dollar floor, so a rebalance doesn't propose trading a few
    cents of a position whose transaction cost would exceed its size; it
    defaults to 0.0 (no filtering) since Section 27 names no specific floor.

    Returns `None` when the portfolio (holdings plus cash) has no positive
    value to reallocate -- there is no meaningful plan to propose, and an
    empty-but-present plan would misleadingly suggest one was computed.
    """
    if cash < 0:
        raise ValueError(f"cash must be >= 0, got {cash}")
    if transaction_cost < 0:
        raise ValueError(f"transaction_cost must be >= 0, got {transaction_cost}")
    if min_trade_value < 0:
        raise ValueError(f"min_trade_value must be >= 0, got {min_trade_value}")

    negative_current = sorted(s for s, v in current_shares.items() if v < 0)
    if negative_current:
        raise ValueError(
            f"current_shares must be non-negative (long-only, Section 2); got {negative_current}"
        )
    negative_target = sorted(s for s, v in target_weights.items() if v < 0)
    if negative_target:
        raise ValueError(
            f"target_weights must be non-negative (long-only, Section 2); got {negative_target}"
        )
    weight_total = sum(target_weights.values())
    if weight_total > 1.0 + _WEIGHT_SUM_TOL:
        raise ValueError(
            f"target_weights must sum to at most 1.0 (Section 2, no leverage); "
            f"got {weight_total:.6f}"
        )

    universe = sorted(set(current_shares) | set(target_weights))
    missing_prices = [s for s in universe if s not in prices]
    if missing_prices:
        raise ValueError(f"no price for holdings: {missing_prices}")
    non_positive_prices = sorted(s for s in universe if prices[s] <= 0)
    if non_positive_prices:
        raise ValueError(f"prices must be positive; got non-positive for: {non_positive_prices}")

    current_value = {s: current_shares.get(s, 0.0) * prices[s] for s in universe}
    total_value = sum(current_value.values()) + cash
    if total_value <= 0:
        return None

    current_weights = {s: current_value[s] / total_value for s in universe}
    resolved_target_weights = {s: target_weights.get(s, 0.0) for s in universe}

    kept: list[tuple[str, float, float]] = []  # (symbol, signed shares, signed trade value)
    achieved_value = dict(current_value)

    for symbol in universe:
        price = prices[symbol]
        held = current_shares.get(symbol, 0.0)
        target_value = resolved_target_weights[symbol] * total_value
        raw_shares = (target_value - current_value[symbol]) / price
        shares_signed = float(round(raw_shares)) if whole_shares else raw_shares

        if shares_signed < 0.0 and -shares_signed > held:
            # Rounding (or float noise) asked to sell more than is held --
            # clamp to a full exit rather than overselling.
            shares_signed = -held

        trade_value_signed = shares_signed * price
        if abs(shares_signed) < _MIN_SHARE_EPSILON or abs(trade_value_signed) < min_trade_value:
            continue

        achieved_value[symbol] = current_value[symbol] + trade_value_signed
        kept.append((symbol, shares_signed, trade_value_signed))

    achieved_weights = {s: achieved_value[s] / total_value for s in universe}
    turnover = sum(abs(achieved_weights[s] - current_weights[s]) for s in universe)
    cash_after = cash - sum(value for _, _, value in kept)

    kept.sort(key=lambda row: (row[2], row[0]))  # sells (negative) first, then symbol
    trades = [
        Trade(
            symbol=symbol,
            action="buy" if trade_value_signed > 0 else "sell",
            shares=abs(shares_signed),
            price=prices[symbol],
            trade_value=abs(trade_value_signed),
            current_weight=current_weights[symbol],
            target_weight=resolved_target_weights[symbol],
            achieved_weight=achieved_weights[symbol],
        )
        for symbol, shares_signed, trade_value_signed in kept
    ]

    return RebalancePlan(
        trades=trades,
        total_value=total_value,
        cash_before=cash,
        cash_after=cash_after,
        estimated_transaction_cost=transaction_cost * turnover * total_value,
        turnover=turnover,
        whole_shares=whole_shares,
        current_weights=current_weights,
        target_weights=resolved_target_weights,
        achieved_weights=achieved_weights,
    )
