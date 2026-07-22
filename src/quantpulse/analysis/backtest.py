"""Backtesting engine -- walk-forward, look-ahead- & survivorship-bias-free, cost-aware (7.6).

Section 21 marks this **Opus / Extra**: the single easiest place in the whole
project to fool yourself with a bug that looks like a great result. A backtest
that quietly peeks at the future, or quietly runs only against the names that
survived to today, or quietly ignores what trading actually costs, produces a
beautiful equity curve that no real investor could ever have earned. Everything
here is built so those three specific failure modes are *structurally* hard
rather than merely discouraged:

* **No look-ahead.** Both evaluators only ever hand a model/signal a slice of
  history ending at the as-of date. The realized forward return used to score a
  prediction is read from later data, but it is never passed back into the thing
  being scored -- it's the answer key, computed after the fact. The training
  slice ends strictly before the outcome it's graded against, by construction.
* **No survivorship bias.** `backtest_strategy` takes an `eligible(as_of)`
  callback returning the point-in-time index membership for that date (Section 5's
  `index_membership_history`), so a company that was in the S&P 500 in 2019 and
  later went bankrupt is *in* the 2019 rebalance and realizes its loss -- it
  doesn't silently vanish from history (Section 22).
* **Realistic cost & cadence.** Rebalances happen weekly/monthly, not daily
  (Section 7.6: a strategy that "rebalances" every day racks up turnover no real
  investor would), and every unit of turnover pays `transaction_cost` (a
  conservative bid-ask stand-in even in a commission-free world). Skipping this
  is the single easiest way to flatter a backtest.

Scope (Phase 7 is split across five Section-21 rows): this module is the
"Backtesting engine" row. The **bootstrap confidence intervals** around the
headline Sharpe/CAGR are a separate later row -- so the engine deliberately
returns the full per-period return series and the paired predicted/realized
arrays, which is exactly the raw material that part needs to resample. **Monte
Carlo** simulation and the broader **risk analytics** (beta/VaR/Sortino) are
their own rows too. Everything here is pure: series/frames in, metrics out; no
storage or network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quantpulse.analysis import forecasting
from quantpulse.analysis.forecasting import Forecast

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "sharpe_ratio",
    "cagr",
    "max_drawdown",
    "directional_hit_rate",
    "rmse",
    "AccuracyResult",
    "walk_forward_accuracy",
    "StrategyResult",
    "backtest_strategy",
    "rebalance_dates",
]

TRADING_DAYS_PER_YEAR = 252.0
# How often a rebalance cadence recurs per year -- the annualization factor for
# metrics computed on per-period returns.
_PERIODS_PER_YEAR = {"weekly": 52.0, "monthly": 12.0}
# Per-model minimum history before a walk-forward fold is even attempted, so the
# first evaluation isn't fit on a handful of bars.
_MIN_ACCURACY_TRAIN = 60


def _closes(prices: pd.DataFrame) -> pd.Series:
    """The clean, date-sorted, strictly-positive close series `prices` implies.

    Mirrors `forecasting`'s own close extraction so the return the walk-forward
    grades against is measured on exactly the series the forecast model fit on.
    """
    if "close" not in prices.columns:
        raise ValueError("prices is missing required column: 'close'")
    close = pd.to_numeric(prices["close"], errors="coerce").sort_index().dropna()
    return close[close > 0]


# --------------------------------------------------------------------------- #
# Performance metrics (pure, on a Series of periodic simple returns)
# --------------------------------------------------------------------------- #


def sharpe_ratio(
    returns: pd.Series, *, periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Annualized Sharpe ratio of a periodic simple-return series.

    `risk_free_rate` is an *annual* rate, converted to the return series' period
    before subtracting. Returns `None` when there are fewer than two returns or
    the excess-return volatility is zero (a Sharpe would be undefined or
    infinite -- an honest "not enough to say," not a fabricated number).
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    excess = clean - risk_free_rate / periods_per_year
    std = float(excess.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return None
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def cagr(returns: pd.Series, *, periods_per_year: float) -> float | None:
    """Compound annual growth rate implied by a periodic simple-return series.

    Returns `None` if the series is empty or compounds to a non-positive equity
    (a wipeout leaves the fractional-power growth rate undefined).
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return None
    total_growth = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if total_growth <= 0 or years <= 0:
        return None
    return float(total_growth ** (1.0 / years) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve implied by `returns` (<= 0).

    Compounds the returns into an equity curve, tracks its running peak, and
    returns the most negative (equity / peak - 1). `0.0` for an empty series or
    one that only ever rises.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def directional_hit_rate(
    predicted: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray
) -> float | None:
    """Fraction of predictions whose *sign* matched the realized move.

    A hit is `sign(predicted) == sign(realized)`. Pairs where the realized move
    was exactly flat (no direction to call) or either value is NaN are dropped
    from the denominator; a zero prediction against a non-flat move counts as a
    miss (declining to call isn't a correct call). `None` if no gradable pairs
    remain. This is the metric a forecast must beat the naive baseline on to be
    worth anything (Section 7.6).
    """
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    if pred.size == 0 or act.size == 0:
        return None
    gradable = ~np.isnan(pred) & ~np.isnan(act) & (act != 0.0)
    if not gradable.any():
        return None
    hits = np.sign(pred[gradable]) == np.sign(act[gradable])
    return float(hits.mean())


def rmse(
    predicted: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray
) -> float | None:
    """Root-mean-square error between predicted and realized values; `None` if no pairs."""
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    if pred.size == 0 or act.size == 0:
        return None
    valid = ~np.isnan(pred) & ~np.isnan(act)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((pred[valid] - act[valid]) ** 2)))


# --------------------------------------------------------------------------- #
# Walk-forward forecast accuracy (produces each model's own hit-rate, Section 7.6)
# --------------------------------------------------------------------------- #

# A model is any callable turning a price frame + horizon into a Forecast (or
# None when it can't fit) -- e.g. `forecasting.ml_forecast`.
ModelFn = Callable[[pd.DataFrame, int], "Forecast | None"]


@dataclass(frozen=True)
class AccuracyResult:
    """Out-of-sample accuracy of one model at one horizon, vs the naive baseline.

    `hit_rate`/`rmse` are the model's; `baseline_hit_rate`/`baseline_rmse` the
    random-walk-drift null it must beat. `predicted`/`realized` are the paired
    per-fold forward returns (the model's), kept so the bootstrap sub-part can
    resample a confidence interval around the hit-rate without re-running the
    walk-forward.
    """

    model_name: str
    horizon_days: int
    n: int
    hit_rate: float | None
    rmse: float | None
    baseline_hit_rate: float | None
    baseline_rmse: float | None
    predicted: np.ndarray
    realized: np.ndarray


def walk_forward_accuracy(
    prices: pd.DataFrame,
    *,
    model_fn: ModelFn,
    horizon_days: int,
    model_name: str,
    baseline_fn: ModelFn | None = None,
    step: int | None = None,
    min_train: int = _MIN_ACCURACY_TRAIN,
) -> AccuracyResult | None:
    """Walk `model_fn` forward over `prices`, grading each forecast against the future.

    At each evaluation index `i` (from `min_train` up to `len - horizon - 1`,
    spaced by `step`), the model sees only `prices.iloc[: i + 1]` -- data through
    date `t_i` and no further -- and predicts the `horizon_days`-forward return.
    That prediction is graded against the *realized* return
    `close[i + h] / close[i] - 1`, which is read from later bars but never shown
    to the model. This train-slice-ends-before-the-outcome structure is the whole
    point: it makes look-ahead bias impossible rather than merely avoided
    (Section 22).

    `step` defaults to `horizon_days`, so evaluation windows don't overlap and the
    folds stay roughly independent (overlapping folds inflate the effective
    sample -- an honest `n` matters for the later significance test). Returns
    `None` if the series is too short to produce even one fold.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    close = _closes(prices)
    if baseline_fn is None:
        baseline_fn = forecasting.baseline_forecast
    stride = step if step is not None else horizon_days
    if stride < 1:
        raise ValueError(f"step must be >= 1, got {stride}")

    last_eval = len(close) - horizon_days - 1  # need close[i + h] to exist
    if last_eval < min_train:
        return None

    ordered = prices.loc[close.index]  # align caller's frame to the clean close index
    model_pred: list[float] = []
    model_real: list[float] = []
    base_pred: list[float] = []
    base_real: list[float] = []

    for i in range(min_train, last_eval + 1, stride):
        train = ordered.iloc[: i + 1]
        realized = float(close.iloc[i + horizon_days] / close.iloc[i] - 1.0)

        model_fc = model_fn(train, horizon_days)
        if model_fc is not None:
            model_pred.append(model_fc.point_return)
            model_real.append(realized)
        base_fc = baseline_fn(train, horizon_days)
        if base_fc is not None:
            base_pred.append(base_fc.point_return)
            base_real.append(realized)

    predicted = np.asarray(model_pred, dtype=float)
    realized_arr = np.asarray(model_real, dtype=float)
    if predicted.size == 0:
        return None

    return AccuracyResult(
        model_name=model_name,
        horizon_days=horizon_days,
        n=int(predicted.size),
        hit_rate=directional_hit_rate(predicted, realized_arr),
        rmse=rmse(predicted, realized_arr),
        baseline_hit_rate=directional_hit_rate(base_pred, base_real),
        baseline_rmse=rmse(base_pred, base_real),
        predicted=predicted,
        realized=realized_arr,
    )


# --------------------------------------------------------------------------- #
# Strategy backtest ("followed the algorithm's ratings", Section 7.6)
# --------------------------------------------------------------------------- #

# A signal is any callable turning an as-of date + the price panel *through that
# date* into a {symbol: score} ranking (higher = more attractive). It sees only
# `panel.loc[:as_of]`, so it cannot peek at the future.
SignalFn = Callable[[date, pd.DataFrame], "dict[str, float]"]
# The point-in-time eligible universe for a date (survivorship-bias-free).
EligibleFn = Callable[[date], "set[str]"]


@dataclass(frozen=True)
class StrategyResult:
    """The track record of a rebalanced, cost-aware, follow-the-ratings strategy.

    `period_returns` is the net-of-cost simple return of each holding period (the
    raw material the bootstrap sub-part resamples); `equity_curve` compounds it.
    The scalar metrics are annualized to `periods_per_year`. `benchmark_*` are the
    same metrics for buy-and-hold over the identical dates, so the comparison is
    apples-to-apples. `avg_turnover` and `assumed_txn_cost` make the cost
    assumption explicit and auditable (Section 7.6).
    """

    period_returns: pd.Series
    equity_curve: pd.Series
    sharpe: float | None
    cagr: float | None
    max_drawdown: float
    win_rate: float | None
    benchmark_return: pd.Series
    benchmark_cagr: float | None
    benchmark_sharpe: float | None
    assumed_txn_cost: float
    avg_turnover: float
    n_periods: int
    periods_per_year: float


def rebalance_dates(index: pd.Index, cadence: str = "monthly") -> list[pd.Timestamp]:
    """Trading days on which to rebalance: the last available bar of each week/month.

    Uses the *actual* dates present in `index` (already market-calendar-filtered
    upstream), so a rebalance always lands on a real trading day -- never a
    weekend or holiday that happens to be a month-end.
    """
    if cadence not in _PERIODS_PER_YEAR:
        raise ValueError(f"cadence must be one of {sorted(_PERIODS_PER_YEAR)}, got {cadence!r}")
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) == 0:
        return []
    frame = pd.Series(idx, index=idx)
    freq = "W" if cadence == "weekly" else "ME"
    grouped = frame.groupby(pd.Grouper(freq=freq)).last().dropna()
    return [pd.Timestamp(v) for v in grouped]


def _as_date(value: pd.Timestamp | date) -> date:
    return value.date() if isinstance(value, pd.Timestamp) else value


def _period_return(
    panel: pd.DataFrame, weights: dict[str, float], start: pd.Timestamp, end: pd.Timestamp
) -> float:
    """Weighted simple return of `weights` held from `start` to `end` over `panel`.

    A holding with no price at `end` (e.g. delisted mid-period) realizes its last
    observed price in `(start, end]` instead of silently disappearing; if it has
    no later price at all, it contributes a flat 0 for the period (held but
    untradeable) rather than an invented gain.
    """
    total = 0.0
    for symbol, weight in weights.items():
        if symbol not in panel.columns:
            continue
        series = panel[symbol]
        start_price = series.get(start)
        if start_price is None or pd.isna(start_price) or start_price <= 0:
            continue
        window = series.loc[start:end].dropna()
        end_price = float(window.iloc[-1]) if not window.empty else float(start_price)
        total += weight * (end_price / float(start_price) - 1.0)
    return total


def _benchmark_period_return(benchmark: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Buy-and-hold return of a benchmark price series between two rebalance dates."""
    series = pd.to_numeric(benchmark, errors="coerce").sort_index()
    start_window = series.loc[:start].dropna()
    end_window = series.loc[:end].dropna()
    if start_window.empty or end_window.empty:
        return 0.0
    start_price = float(start_window.iloc[-1])
    end_price = float(end_window.iloc[-1])
    if start_price <= 0:
        return 0.0
    return end_price / start_price - 1.0


def backtest_strategy(
    price_panel: pd.DataFrame,
    *,
    signal_fn: SignalFn,
    cadence: str = "monthly",
    top_fraction: float = 0.2,
    transaction_cost: float = 0.001,
    benchmark: pd.Series | None = None,
    eligible: EligibleFn | None = None,
    schedule: Sequence[pd.Timestamp] | None = None,
) -> StrategyResult | None:
    """Simulate following the algorithm's top-ranked names, rebalanced and cost-aware.

    `price_panel` is wide (DatetimeIndex x symbol columns, adjusted close). At
    each rebalance date the strategy asks `signal_fn(as_of, panel.loc[:as_of])`
    for a ranking -- point-in-time, so it never sees a future price -- keeps the
    top `top_fraction` of the *eligible* universe equal-weighted, pays
    `transaction_cost` on every unit of turnover, and holds until the next
    rebalance. `eligible(as_of)` supplies the survivorship-bias-free membership
    for that date; without it, every symbol priced on that date is eligible.

    Returns `None` if fewer than two rebalance periods can be formed (nothing to
    annualize). The benchmark, if given, is a price series carried buy-and-hold
    over the identical period boundaries for an apples-to-apples comparison.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    if transaction_cost < 0:
        raise ValueError(f"transaction_cost must be >= 0, got {transaction_cost}")

    panel = price_panel.sort_index()
    raw_dates = schedule if schedule is not None else rebalance_dates(panel.index, cadence)
    dates = [pd.Timestamp(d) for d in raw_dates]
    if len(dates) < 2:
        return None

    current: dict[str, float] = {}
    period_returns: list[float] = []
    period_index: list[pd.Timestamp] = []
    benchmark_returns: list[float] = []
    turnovers: list[float] = []

    for start, end in zip(dates[:-1], dates[1:], strict=True):
        as_of = _as_date(start)
        scores = signal_fn(as_of, panel.loc[:start])
        allowed = eligible(as_of) if eligible is not None else None
        priced = {s for s in panel.columns if pd.notna(panel[s].get(start))}
        ranked = {
            s: v
            for s, v in scores.items()
            if v is not None
            and not pd.isna(v)
            and s in priced
            and (allowed is None or s in allowed)
        }

        if ranked:
            k = max(1, round(len(ranked) * top_fraction))
            top = sorted(ranked, key=lambda s: ranked[s], reverse=True)[:k]
            target = {s: 1.0 / len(top) for s in top}
        else:
            target = {}  # no signal this period -> sit in cash

        traded = sum(
            abs(target.get(s, 0.0) - current.get(s, 0.0)) for s in set(target) | set(current)
        )
        gross = _period_return(panel, target, start, end)
        period_returns.append(gross - transaction_cost * traded)
        period_index.append(end)
        turnovers.append(traded)
        current = target

        if benchmark is not None:
            benchmark_returns.append(_benchmark_period_return(benchmark, start, end))

    returns = pd.Series(period_returns, index=pd.DatetimeIndex(period_index))
    ppy = _PERIODS_PER_YEAR[cadence]
    bench = (
        pd.Series(benchmark_returns, index=returns.index)
        if benchmark is not None
        else pd.Series(dtype=float)
    )

    return StrategyResult(
        period_returns=returns,
        equity_curve=(1.0 + returns).cumprod(),
        sharpe=sharpe_ratio(returns, periods_per_year=ppy),
        cagr=cagr(returns, periods_per_year=ppy),
        max_drawdown=max_drawdown(returns),
        win_rate=float((returns > 0).mean()) if not returns.empty else None,
        benchmark_return=bench,
        benchmark_cagr=cagr(bench, periods_per_year=ppy) if not bench.empty else None,
        benchmark_sharpe=sharpe_ratio(bench, periods_per_year=ppy) if not bench.empty else None,
        assumed_txn_cost=transaction_cost,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        n_periods=int(len(returns)),
        periods_per_year=ppy,
    )
