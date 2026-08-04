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

Scope (Phase 7 is split across five Section-21 rows): this module owns two of
them -- the "Backtesting engine" row (the walk-forward evaluator and the
cost/survivorship-aware strategy simulator above) and the **"Statistical
significance testing on backtest metrics (bootstrap CI)"** row (the final
section below). The engine deliberately returns the full per-period return
series and the paired predicted/realized arrays precisely so the bootstrap can
resample them without re-running anything. **Monte Carlo** simulation lives in
`forecasting.py`; portfolio **risk analytics** (beta/VaR/Sortino) is Phase 8.
Everything here is pure: series/frames in, metrics out; no storage or network.
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
    "payoff_ratio",
    "rmse",
    "AccuracyResult",
    "MIN_GRADED_WINDOWS",
    "walk_forward_accuracy",
    "StrategyResult",
    "backtest_strategy",
    "rebalance_dates",
    "DEFAULT_CI_CONFIDENCE",
    "DEFAULT_N_RESAMPLES",
    "BootstrapCI",
    "StrategySignificance",
    "block_bootstrap_ci",
    "bootstrap_sharpe_ci",
    "bootstrap_cagr_ci",
    "bootstrap_hit_rate_ci",
    "bootstrap_strategy_significance",
]

TRADING_DAYS_PER_YEAR = 252.0
# How often a rebalance cadence recurs per year -- the annualization factor for
# metrics computed on per-period returns.
_PERIODS_PER_YEAR = {"weekly": 52.0, "monthly": 12.0}
# Per-model minimum history before a walk-forward fold is even attempted, so the
# first evaluation isn't fit on a handful of bars.
_MIN_ACCURACY_TRAIN = 60

# Distinct evaluation windows a pooled hit rate needs before it is worth
# publishing at all.
#
# A hit rate is a proportion, so its standard error is at most 0.5/sqrt(n): at
# 20 windows that is 11 percentage points, meaning a 90% interval spans +/-18pp
# and "50%" is indistinguishable from "68%". Thirty brings it to about +/-15pp,
# which is the point at which the interval is narrower than the gap between "no
# skill" and a rate a reader would find interesting.
#
# The number that matters is *windows*, not pooled pairs. Measured on real
# history over the nightly's read window: the 5-day horizon grades 163 windows
# per name and the 20-day horizon 40, but the 63-day horizon only 12 and the
# 1-year horizon none at all. Pooling twenty symbols multiplies the pair count
# twentyfold without adding a single new window -- the same three years of one
# market, read twenty times -- so a "60% hit rate" at the 1-year horizon was
# twenty correlated readings of a single year.
MIN_GRADED_WINDOWS = 30

# A return series whose standard deviation is this small *relative to its own
# magnitude* is constant to within floating-point noise, so its Sharpe is
# undefined rather than astronomically large (see `sharpe_ratio`). Orders of
# magnitude above float64's ~1e-16 relative noise and far below any genuine
# variation in a real return series.
_DEGENERATE_STD_REL_TOL = 1e-12

# Bootstrap defaults. 90% matches Section 7.6's own worked example ("Sharpe 0.8,
# 90% CI [0.3, 1.3]"); 2000 resamples is comfortably enough for stable 5th/95th
# percentiles and still runs in milliseconds on a few hundred observations.
DEFAULT_CI_CONFIDENCE = 0.90
DEFAULT_N_RESAMPLES = 2000
# Below this many observations a confidence interval is theatre, not evidence --
# the bootstrap can only resample the information actually present.
_MIN_BOOTSTRAP_OBS = 8
# If fewer than this fraction of resamples yield a defined statistic (e.g. a
# zero-variance resample makes Sharpe undefined), the CI isn't trustworthy.
_MIN_DEFINED_RESAMPLE_FRACTION = 0.5


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
    the excess-return volatility is (effectively) zero -- a Sharpe would be
    undefined, and an honest "not enough to say" beats a fabricated number.

    "Effectively zero" is judged *relative to the size of the returns*, not
    against exact 0.0: a constant series rarely differences to a std of exactly
    zero in floating point (it lands around 1e-18), and dividing by that noise
    yields a Sharpe of ~1e16 that is not merely wrong but wrong with enormous
    apparent confidence -- and would then be handed a meaninglessly tight
    bootstrap interval around it.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    excess = clean - risk_free_rate / periods_per_year
    std = float(excess.std(ddof=1))
    if not np.isfinite(std):
        return None
    scale = float(excess.abs().max())
    if std <= scale * _DEGENERATE_STD_REL_TOL:
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


def payoff_ratio(returns: pd.Series) -> float | None:
    """Mean winning period divided by mean losing period, both as magnitudes.

    The "b" in the Kelly criterion: how much a win pays relative to what a loss
    costs. Paired with a hit rate it is everything
    `optimization.kelly_position_fraction` needs, and that function's docstring
    points here deliberately -- a position size is only as honest as the track
    record behind it, and this is the module that measures one out-of-sample.

    `None` unless there is at least one win *and* one loss: with no losses the
    ratio is infinite rather than large, and an infinite payoff ratio would feed
    Kelly a bet it thinks cannot lose. Flat periods (exactly zero) count as
    neither, so they neither flatter nor penalize the ratio.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    wins = clean[clean > 0]
    losses = clean[clean < 0]
    if wins.empty or losses.empty:
        return None
    mean_loss = float(losses.abs().mean())
    if mean_loss <= 0:
        return None
    return float(wins.mean()) / mean_loss


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

    `as_of` is the evaluation date behind each pair, and it is not decoration.
    A hit rate pooled across many symbols has as many *pairs* as symbols x
    folds, but only as many independent *observations* as there were distinct
    evaluation windows -- twenty stocks graded over the same three one-year
    windows is three pieces of evidence wearing a sample size of sixty. Carrying
    the dates is what lets a caller count the former rather than the latter.
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
    as_of: tuple[pd.Timestamp, ...] = ()


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
    model_dates: list[pd.Timestamp] = []
    base_pred: list[float] = []
    base_real: list[float] = []

    for i in range(min_train, last_eval + 1, stride):
        train = ordered.iloc[: i + 1]
        realized = float(close.iloc[i + horizon_days] / close.iloc[i] - 1.0)

        model_fc = model_fn(train, horizon_days)
        if model_fc is not None:
            model_pred.append(model_fc.point_return)
            model_real.append(realized)
            model_dates.append(pd.Timestamp(close.index[i]))
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
        as_of=tuple(model_dates),
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
    # Mean winning period / mean losing period, both as positive magnitudes.
    # Together with `win_rate` this is exactly what a Kelly position size needs
    # (`optimization.kelly_position_fraction`), and its docstring points here
    # because this is the only place in the project that measures either from a
    # real out-of-sample track record. `None` when the run has no wins or no
    # losses -- a payoff ratio needs both sides to be defined.
    payoff_ratio: float | None
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
        payoff_ratio=payoff_ratio(returns),
        benchmark_return=bench,
        benchmark_cagr=cagr(bench, periods_per_year=ppy) if not bench.empty else None,
        benchmark_sharpe=sharpe_ratio(bench, periods_per_year=ppy) if not bench.empty else None,
        assumed_txn_cost=transaction_cost,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        n_periods=int(len(returns)),
        periods_per_year=ppy,
    )


# --------------------------------------------------------------------------- #
# Bootstrap significance testing on the headline metrics (Section 7.6)
#
# Section 21 flags exactly one failure mode for this row: "easy to implement the
# bootstrap mechanically wrong (e.g. resampling that breaks time-ordering) in a
# way that looks fine but isn't." That is the textbook i.i.d. bootstrap applied
# to a time series. Drawing individual observations with replacement assumes
# they are independent; financial returns are not (volatility clusters, momentum
# and mean-reversion persist), and destroying that dependence makes each
# resample *more* internally random than the real series, which shrinks the
# spread of the bootstrap distribution and yields a confidence interval that is
# too narrow. The failure is silent and flattering: it makes a mediocre Sharpe
# look reliably positive.
#
# So the default here is the **moving-block bootstrap** (Kunsch 1989): resample
# contiguous blocks of consecutive observations rather than single points, so
# the within-block serial structure survives into every resample. `block_size=1`
# reduces it to the i.i.d. bootstrap, available explicitly for a caller who has
# genuinely independent observations -- never as a silent default.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapCI:
    """A metric with its bootstrap confidence interval and the sample behind it.

    Exactly what Section 7.6 asks the track-record page to be able to say --
    "Sharpe 0.8, 90% CI [0.3, 1.3]" -- instead of a bare number implying false
    precision. `point` is the metric on the observed sample (never a bootstrap
    average, which would be biased); `low`/`high` are the percentile bounds.

    `n_observations`, `block_size`, and `n_defined` are carried so a reader can
    judge how much the interval is worth: a 90% CI from 11 monthly periods with
    block size 2 is a very different claim from one built on 200, and hiding
    that would repeat exactly the false-precision mistake the CI exists to fix.
    """

    point: float
    low: float
    high: float
    confidence_level: float
    n_observations: int
    n_resamples: int
    n_defined: int
    block_size: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval sits on one side of zero.

        The honest reading of "is this result statistically meaningful?" -- a
        Sharpe whose CI straddles zero has not been distinguished from luck,
        however good its point estimate looks (Section 7.6, Section 22).
        """
        return self.low > 0.0 or self.high < 0.0


def _default_block_size(n: int) -> int:
    """Block length for the moving-block bootstrap: the standard n**(1/3) rule.

    Long enough to carry the short-range dependence typical of return series,
    short enough to still produce many distinct resamples. Capped at `n // 2` so
    at least two independent blocks always fit -- a block as long as the sample
    would make every resample the original series and collapse the interval to a
    point, an interval that looks impressively tight because it is measuring
    nothing.
    """
    if n < 2:
        return 1
    return max(1, min(round(n ** (1.0 / 3.0)), n // 2))


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Positions for one moving-block bootstrap resample of length `n`.

    Draws `ceil(n / block_size)` block start positions uniformly from the
    `n - block_size + 1` possible starts, concatenates the blocks, and truncates
    back to `n`. Consecutive observations inside a block stay adjacent and in
    their original order, which is the property that keeps the resampled series'
    serial correlation (and therefore the width of the interval) honest.
    """
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()
    return idx[:n]


def block_bootstrap_ci(
    n_observations: int,
    statistic_fn: Callable[[np.ndarray], float | None],
    *,
    confidence_level: float = DEFAULT_CI_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    random_state: int | None = 0,
) -> BootstrapCI | None:
    """Percentile bootstrap CI for any statistic of an ordered sample.

    The general engine every wrapper below delegates to. `statistic_fn` receives
    an array of *positions* into the original sample (not the values), so a
    caller can compute a statistic over several parallel arrays -- e.g. a
    directional hit-rate over paired predicted/realized returns -- while keeping
    those arrays aligned. Resampling positions rather than values is what makes
    paired resampling correct by construction: break the pairing and a hit-rate
    CI silently measures nothing.

    Blocks default to `_default_block_size` (see the section header for why
    blocks rather than single points). A resample where the statistic is
    undefined (`None`/NaN -- e.g. a zero-variance draw makes Sharpe undefined)
    is dropped, and `None` is returned if too few remain to trust the quantiles.

    `random_state` defaults to a fixed seed so a stored track record is
    reproducible; pass `None` for a fresh draw.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if n_observations < _MIN_BOOTSTRAP_OBS:
        return None

    resolved_block = _default_block_size(n_observations) if block_size is None else block_size
    if resolved_block < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    resolved_block = min(resolved_block, n_observations)

    observed = statistic_fn(np.arange(n_observations))
    if observed is None or not np.isfinite(observed):
        return None  # undefined on the real sample -> nothing to bracket

    rng = np.random.default_rng(random_state)
    draws: list[float] = []
    for _ in range(n_resamples):
        value = statistic_fn(_block_indices(n_observations, resolved_block, rng))
        if value is not None and np.isfinite(value):
            draws.append(float(value))

    if len(draws) < max(2, int(n_resamples * _MIN_DEFINED_RESAMPLE_FRACTION)):
        return None

    tail = (1.0 - confidence_level) / 2.0
    values = np.asarray(draws, dtype=float)
    return BootstrapCI(
        point=float(observed),
        low=float(np.quantile(values, tail)),
        high=float(np.quantile(values, 1.0 - tail)),
        confidence_level=confidence_level,
        n_observations=n_observations,
        n_resamples=n_resamples,
        n_defined=len(draws),
        block_size=resolved_block,
    )


def _clean_returns(returns: pd.Series) -> np.ndarray:
    return pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CI_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    random_state: int | None = 0,
) -> BootstrapCI | None:
    """Bootstrap CI around the annualized Sharpe ratio of a periodic return series."""
    values = _clean_returns(returns)

    def statistic(idx: np.ndarray) -> float | None:
        return sharpe_ratio(
            pd.Series(values[idx]),
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

    return block_bootstrap_ci(
        len(values),
        statistic,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        block_size=block_size,
        random_state=random_state,
    )


def bootstrap_cagr_ci(
    returns: pd.Series,
    *,
    periods_per_year: float,
    confidence_level: float = DEFAULT_CI_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    random_state: int | None = 0,
) -> BootstrapCI | None:
    """Bootstrap CI around the CAGR implied by a periodic return series."""
    values = _clean_returns(returns)

    def statistic(idx: np.ndarray) -> float | None:
        return cagr(pd.Series(values[idx]), periods_per_year=periods_per_year)

    return block_bootstrap_ci(
        len(values),
        statistic,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        block_size=block_size,
        random_state=random_state,
    )


def bootstrap_hit_rate_ci(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    *,
    confidence_level: float = DEFAULT_CI_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    random_state: int | None = 0,
) -> BootstrapCI | None:
    """Bootstrap CI around a forecast model's out-of-sample directional hit-rate.

    Consumes `AccuracyResult.predicted` / `.realized` straight from
    `walk_forward_accuracy`. The two arrays are resampled **jointly** (the same
    block positions index both), so every resampled prediction keeps its own
    realized outcome -- resampling them independently would shuffle answers onto
    the wrong questions and drive the hit-rate toward 50% no matter how good the
    model is.

    Blocks matter here whenever `walk_forward_accuracy` ran with a `step`
    smaller than the horizon: those folds share overlapping windows and are
    therefore serially dependent (the default `step=horizon_days` produces
    non-overlapping folds, where blocks are merely harmless).
    """
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    if pred.size != act.size:
        raise ValueError(f"predicted and actual must be the same length: {pred.size} vs {act.size}")

    def statistic(idx: np.ndarray) -> float | None:
        return directional_hit_rate(pred[idx], act[idx])

    return block_bootstrap_ci(
        pred.size,
        statistic,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        block_size=block_size,
        random_state=random_state,
    )


@dataclass(frozen=True)
class StrategySignificance:
    """Bootstrap CIs around a strategy backtest's headline metrics (Section 7.6).

    `sharpe`/`cagr` are `None` when the run was too short to bootstrap honestly
    -- an absent interval, not a fabricated one.

    **`max_drawdown` is deliberately not bootstrapped.** It is a path-dependent
    extremum: its value depends on the specific order in which returns arrived,
    and a resampled series is a different path that never contained the actual
    drawdown episode. A "confidence interval" around it would be a well-formed
    number describing a quantity nobody experienced. Sharpe and CAGR are
    order-independent functions of the return distribution (a mean/std ratio and
    a product), which is exactly why resampling them is meaningful.
    """

    sharpe: BootstrapCI | None
    cagr: BootstrapCI | None


def bootstrap_strategy_significance(
    result: StrategyResult,
    *,
    confidence_level: float = DEFAULT_CI_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int | None = None,
    random_state: int | None = 0,
) -> StrategySignificance:
    """Bootstrap the Sharpe and CAGR of a completed strategy backtest.

    Reads `result.period_returns` -- the net-of-cost per-period series the
    engine already returns -- and annualizes each resample at the same
    `periods_per_year` the run used, so the interval is on the same scale as the
    headline number it brackets.
    """
    return StrategySignificance(
        sharpe=bootstrap_sharpe_ci(
            result.period_returns,
            periods_per_year=result.periods_per_year,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            block_size=block_size,
            random_state=random_state,
        ),
        cagr=bootstrap_cagr_ci(
            result.period_returns,
            periods_per_year=result.periods_per_year,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            block_size=block_size,
            random_state=random_state,
        ),
    )
