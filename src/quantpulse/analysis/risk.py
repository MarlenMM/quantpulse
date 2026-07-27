"""Risk & portfolio analytics -- volatility, beta, Sharpe/Sortino, VaR, correlation (7.7).

Section 21 marks this row **Opus / High** for one reason: "errors here mislead
the user about real risk." Every number in this module is one a reader will act
on -- a beta that says "this moves with the market," a VaR that says "a bad day
costs you 2%," a correlation matrix that says "you're diversified." Each of
those has a well-known way to be quietly wrong, and each of those failure modes
is closed here deliberately rather than incidentally:

* **A statistic computed from too little data is not a statistic.** Every
  estimator has an explicit minimum-observation floor and returns `None` below
  it instead of a confident-looking number. Historical VaR's floor is derived
  from the confidence level itself (`_min_historical_var_obs`), because a 95%
  historical VaR from 30 days is a single order statistic wearing a percentage
  sign.
* **A ratio with a vanishing denominator is not a ratio.** Zero-variance and
  zero-downside cases return `None`, never an astronomical or infinite value --
  the same relative-tolerance guard `backtest.sharpe_ratio` documents (a
  "constant" float series differences to ~1e-18, not 0).
* **Two series must be aligned before they are compared.** `beta` intersects
  the asset and market returns on their shared dates and counts what actually
  overlapped; a beta silently computed across misaligned indices is the classic
  way to get a plausible number from unrelated data.
* **Sortino divides by the full sample, not by the loss count.** See
  `sortino_ratio` -- the single most common implementation error in this module's
  subject matter, and one that flatters a rarely-losing series enormously.
* **Sign conventions are stated, not assumed.** `value_at_risk` reports a
  *positive loss magnitude* (0.02 = "down 2% or worse"), while `max_drawdown`
  (reused from `backtest`) reports a *negative* return. That asymmetry is
  conventional in both cases, so it is documented at each site rather than
  silently harmonized into something a finance-literate reader would misread.

**Reuse, not duplication.** `sharpe_ratio` and `max_drawdown` already exist in
`backtest.py` (Phase 7) and are imported and re-exported here rather than
reimplemented, so the Sharpe on the track-record page and the Sharpe on the
portfolio page are the same function and can never drift apart. `sortino_ratio`
is new -- the backtest never needed it.

Scope (Section 21's "Portfolio risk analytics (beta, VaR, correlation)" row,
Section 7.7): the per-stock and portfolio-level *risk statistics* consumed by
the Portfolio Manager (Section 9). Position bookkeeping (holdings, cost basis,
FIFO tax lots, P/L, sector allocation, HHI concentration), optimization
(Section 27) and rebalancing trade lists are separate Phase-8 rows and are
deliberately *not* here. Nothing in this module persists: Section 13 has no risk
table, and these are cheap functions over data already stored -- prices in,
metrics out, no storage or network, like `forecasting.py` and `backtest.py`.

`scoring.score_momentum` (Phase 6) is likewise left alone even though Section
7.7 names the momentum/risk-adjusted category as a consumer: it computes a
deliberately scale-invariant, un-annualized risk-adjusted trailing return
because all the composite needs from it is a *ranking*, and rewriting a scorer
whose output is already stored point-in-time in `composite_scores` would
retroactively redefine a published number for no gain (Section 6.8).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantpulse.analysis.backtest import TRADING_DAYS_PER_YEAR, max_drawdown, sharpe_ratio

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_VAR_CONFIDENCE",
    "sharpe_ratio",
    "max_drawdown",
    "to_returns",
    "returns_panel",
    "equal_weight_market_returns",
    "historical_volatility",
    "VolatilityProfile",
    "volatility_profile",
    "sortino_ratio",
    "BetaResult",
    "beta",
    "portfolio_beta",
    "ValueAtRisk",
    "value_at_risk",
    "correlation_matrix",
    "average_pairwise_correlation",
    "most_correlated_pairs",
    "portfolio_returns",
    "RiskProfile",
    "stock_risk_profile",
    "PortfolioRisk",
    "portfolio_risk",
]

# Section 7.7's "Value-at-Risk"; 95% is the conventional retail-facing level
# (99% is a regulatory-capital convention and needs ~5x the history to estimate
# empirically -- see `_min_historical_var_obs`).
DEFAULT_VAR_CONFIDENCE = 0.95

# Minimum observations before an estimator will speak at all. A month of bars is
# the floor for a volatility/Sortino estimate; a beta regressed on fewer than a
# quarter of overlapping bars is dominated by estimation error, and stating it
# to two decimal places would be the false precision Section 22 warns about.
_MIN_RETURN_OBS = 20
_MIN_BETA_OBS = 60
_MIN_CORRELATION_OBS = 30

# Historical VaR reads an empirical tail quantile, so it needs enough
# observations for that tail to contain more than one or two points. Requiring
# `_MIN_TAIL_OBS` beyond the cutoff means a 95% VaR needs 100 observations and a
# 99% VaR needs 500 -- a real constraint, and the honest one: below it the
# quantile is a single lucky/unlucky day, not a distribution.
_MIN_TAIL_OBS = 5

# A dispersion measure this small *relative to the size of the data* is zero to
# within floating-point noise, so the ratio it would divide is undefined rather
# than enormous. Mirrors `backtest.py`'s `_DEGENERATE_STD_REL_TOL` (kept as its
# own constant here rather than importing a private, the same decoupling
# `backtest._closes` uses).
_DEGENERATE_STD_REL_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Return series (the input every metric below actually consumes)
# --------------------------------------------------------------------------- #


def to_returns(prices: pd.Series) -> pd.Series:
    """Simple period-over-period returns from a price series.

    Sorts by date, drops non-numeric/missing values, and discards non-positive
    prices (bad data: a return measured off a zero price is infinite). The
    result is what every metric in this module consumes -- taking returns rather
    than prices everywhere removes any ambiguity about which one a function
    wants, and matches `backtest.py`'s metric signatures.
    """
    clean = pd.to_numeric(prices, errors="coerce").sort_index().dropna()
    clean = clean[clean > 0]
    return clean.pct_change().dropna()


def returns_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol simple returns from a wide price panel (index=date, columns=symbol).

    Each column is differenced against its own previous *observed* bar, and a
    return spanning a missing bar stays missing rather than being reported as a
    one-period move: a name that stopped trading for a week did not have a
    one-day 12% return when it resumed, and silently pretending otherwise would
    inflate its volatility and distort every correlation it appears in.
    """
    frame = price_panel.sort_index().apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0)
    return frame.pct_change().dropna(how="all")


def equal_weight_market_returns(price_panel: pd.DataFrame, *, min_names: int = 2) -> pd.Series:
    """A daily-rebalanced equal-weight market-proxy return series from a price panel.

    Section 7.7 asks for "beta vs S&P 500", but no S&P 500 price series is
    ingested anywhere in the pipeline (Section 5 lists the index only as a
    *constituent list*). Two honest options follow: ingest an index series and
    pass its returns to `beta` directly, or -- until then -- regress against the
    universe itself, which is what this builds. Dates with fewer than
    `min_names` priced constituents are dropped rather than letting one or two
    survivors stand in for "the market".

    This is the return-space sibling of the backtest's buy-and-hold
    `_equal_weight_benchmark` level series; they differ in rebalancing (constant
    weights here, drifting there), which is deliberate -- a beta regression wants
    a constant-weight market return, a benchmark track record wants the
    buy-and-hold path an investor would actually have held.
    """
    returns = returns_panel(price_panel)
    if returns.empty:
        return pd.Series(dtype=float)
    usable = returns[returns.notna().sum(axis=1) >= min_names]
    return usable.mean(axis=1, skipna=True).dropna()


# --------------------------------------------------------------------------- #
# Volatility (Section 7.7: historical & implied where available)
# --------------------------------------------------------------------------- #


def historical_volatility(
    returns: pd.Series,
    *,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    lookback: int | None = None,
) -> float | None:
    """Annualized standard deviation of `returns`; `None` below `_MIN_RETURN_OBS` bars.

    `lookback`, if given, uses only the most recent N returns -- the trailing
    window a "current volatility" reading should be measured over, rather than
    letting a calm decade dilute a turbulent quarter. Sample standard deviation
    (ddof=1) scaled by sqrt(periods_per_year), the standard convention.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if lookback is not None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        clean = clean.iloc[-lookback:]
    if len(clean) < _MIN_RETURN_OBS:
        return None
    std = float(clean.std(ddof=1))
    if not np.isfinite(std):
        return None
    return std * math.sqrt(periods_per_year)


@dataclass(frozen=True)
class VolatilityProfile:
    """Realized vs option-implied volatility for one name (Section 7.7).

    Both are annualized fractions (0.25 = 25%/yr). `implied` comes from
    `options_signals.atm_implied_volatility` where an options chain was
    available and is `None` otherwise -- Section 7.7's "where available" is a
    real caveat, not a formality, since many names have thin or absent chains.

    `implied_premium` (implied - historical) is the readable form of the
    comparison: positive means the options market is pricing *more* movement
    than the stock has recently delivered -- typically an event (earnings, a
    pending deal) the price history hasn't seen yet. It is a spread between two
    volatilities, not a forecast, and deliberately not collapsed into a
    directional signal (the same reading discipline Section 24 imposes on short
    interest).
    """

    historical: float | None
    implied: float | None
    implied_premium: float | None
    n_observations: int


def volatility_profile(
    returns: pd.Series,
    *,
    implied_volatility: float | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    lookback: int | None = None,
) -> VolatilityProfile:
    """Pair a name's realized volatility with its implied volatility, where available."""
    historical = historical_volatility(
        returns, periods_per_year=periods_per_year, lookback=lookback
    )
    implied = (
        float(implied_volatility)
        if implied_volatility is not None
        and not pd.isna(implied_volatility)
        and implied_volatility > 0
        else None
    )
    premium = implied - historical if (implied is not None and historical is not None) else None
    return VolatilityProfile(
        historical=historical,
        implied=implied,
        implied_premium=premium,
        n_observations=int(pd.to_numeric(returns, errors="coerce").dropna().size),
    )


# --------------------------------------------------------------------------- #
# Risk-adjusted return: Sortino (Sharpe is imported from `backtest`)
# --------------------------------------------------------------------------- #


def sortino_ratio(
    returns: pd.Series,
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
    target_return: float | None = None,
) -> float | None:
    """Annualized Sortino ratio: excess return per unit of *downside* deviation.

    The Sharpe ratio penalizes upside and downside volatility identically, which
    is the wrong shape for an investor who only minds one of them; Sortino
    replaces the denominator with the deviation below a target. `target_return`
    is a per-period threshold and defaults to the per-period `risk_free_rate`
    (an *annual* rate, converted here), making this the exact analogue of
    `backtest.sharpe_ratio`'s excess-return convention.

    **The denominator divides by the full sample size, not by the number of
    losing periods.** This is the single most common way this ratio is
    implemented wrongly: averaging the squared shortfalls over only the
    down-periods answers "how bad were the bad days" rather than "how much
    downside did this series carry," and it rewards a series for having few
    losses *twice* -- once by dropping them from the numerator's drag, again by
    shrinking the denominator's count. On a series that loses one period in ten,
    that mistake inflates the ratio by roughly sqrt(10). The definition used
    here is the standard target semideviation: sqrt(mean over ALL periods of
    min(0, r - target)^2).

    Returns `None` when there are fewer than two returns, or when no period fell
    below the target at all -- a series with zero downside has an infinite
    Sortino, and "no downside observed in this window" is a statement a caller
    should render as text, not as a number that will be sorted, averaged, or
    plotted against its peers.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    threshold = target_return if target_return is not None else risk_free_rate / periods_per_year
    excess = clean - threshold
    shortfall = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(shortfall**2)))
    if not np.isfinite(downside_deviation):
        return None
    scale = float(excess.abs().max())
    if downside_deviation <= scale * _DEGENERATE_STD_REL_TOL:
        return None
    return float(excess.mean() / downside_deviation * math.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Beta (Section 7.7: vs the S&P 500; Section 9: portfolio beta)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BetaResult:
    """A name's beta against a market series, with the evidence behind it.

    `beta` alone is a famously over-trusted number, so the context that decides
    how much it is worth travels with it: `r_squared` is the fraction of the
    asset's variance the market actually explains (a beta of 1.4 with an R^2 of
    0.05 means "this mostly does its own thing", which is the opposite of what
    "beta 1.4" suggests to most readers), and `n_observations` is how many
    *overlapping* dates the estimate came from.
    """

    beta: float
    r_squared: float | None
    n_observations: int


def beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    *,
    min_obs: int = _MIN_BETA_OBS,
) -> BetaResult | None:
    """Beta of `asset_returns` against `market_returns`: cov(asset, market) / var(market).

    The two series are **intersected on their shared dates** before anything is
    computed, and only pairs where both are present count. Aligning by position
    instead (or letting pandas broadcast two differently-indexed series) is how
    a beta gets computed against the wrong days entirely and still returns a
    perfectly plausible 1.1.

    Returns `None` when fewer than `min_obs` dates overlap, or when the market
    series has no usable variance over them (a flat market makes beta undefined
    -- there is nothing to be relative *to*).
    """
    paired = pd.concat(
        {
            "asset": pd.to_numeric(asset_returns, errors="coerce"),
            "market": pd.to_numeric(market_returns, errors="coerce"),
        },
        axis=1,
        join="inner",
    ).dropna()
    if len(paired) < min_obs:
        return None

    market = paired["market"]
    market_std = float(market.std(ddof=1))
    scale = float(market.abs().max())
    if not np.isfinite(market_std) or market_std <= scale * _DEGENERATE_STD_REL_TOL:
        return None

    covariance = float(paired["asset"].cov(market))
    if not np.isfinite(covariance):
        return None
    correlation = paired["asset"].corr(market)
    return BetaResult(
        beta=covariance / market_std**2,
        r_squared=float(correlation**2) if pd.notna(correlation) else None,
        n_observations=int(len(paired)),
    )


def portfolio_beta(betas: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    """Weighted average of holdings' individual betas (Section 9's portfolio beta).

    Holdings whose beta is unknown (`None`/NaN, or absent from `betas`) drop out
    and the remaining weights are renormalized, the same coverage discipline
    `scoring.py`/`fundamental.py` use -- an unmeasurable holding shouldn't be
    silently treated as beta 0, which would drag the portfolio figure toward
    "market-neutral" purely because of a data gap. A **cash** position is the
    genuine exception and should be passed explicitly as `0.0`: cash really does
    have zero beta, and including it is what makes a 50%-cash portfolio's beta
    correctly read as half the market's.

    Returns `None` when no holding has a usable beta. Note that this equals the
    beta of the portfolio's own return series against the same market over the
    same window (beta is linear in the weights) -- `portfolio_risk` computes it
    that way when it has the return panel, and this function exists for the case
    where a caller has per-symbol betas but no aligned panel.
    """
    total = 0.0
    covered = 0.0
    for symbol, weight in weights.items():
        value = betas.get(symbol)
        if value is None or pd.isna(value) or weight is None or pd.isna(weight):
            continue
        total += float(weight) * float(value)
        covered += float(weight)
    if covered <= 0:
        return None
    return total / covered


# --------------------------------------------------------------------------- #
# Value-at-Risk (Section 7.7: historical or parametric)
# --------------------------------------------------------------------------- #


def _min_historical_var_obs(confidence: float) -> int:
    """Observations needed for the empirical tail to hold `_MIN_TAIL_OBS` points."""
    return int(math.ceil(_MIN_TAIL_OBS / (1.0 - confidence)))


@dataclass(frozen=True)
class ValueAtRisk:
    """A one-period Value-at-Risk estimate, reported as a **positive loss fraction**.

    `var=0.023` at `confidence=0.95` reads: "on the worst 5% of periods this
    series lost 2.3% or more." The positive-magnitude convention is the standard
    way VaR is quoted and deliberately differs from `max_drawdown`'s negative
    return -- both conventions are conventional for their own metric, and
    quietly flipping one to match the other would mislead a reader who knows the
    field. (A *negative* `var` is possible and meaningful: it says even the tail
    quantile of this window was a gain, which is a statement about an unusually
    strong or unusually short sample, not a risk-free asset.)

    `expected_shortfall` is the average loss *given* that the VaR threshold was
    breached -- the answer to the question VaR structurally cannot answer ("how
    bad is bad?"). It is always at least as large as `var`, and the gap between
    them is where fat tails live.

    `method` records which estimator produced the numbers, because they carry
    different assumptions and should never be compared as if interchangeable.
    `historical` reads the empirical distribution: no shape assumption, but only
    as informative as the window is long. `parametric` fits a normal to the mean
    and standard deviation: it uses every observation, but it is describing a
    shape real return series don't have, and a few outliers inflating the fitted
    sigma can push its moderate quantiles *above* what actually happened while
    its extreme quantiles still sit *below* the real tail. Which way it errs
    depends on the sample and the confidence level -- that unpredictability is
    the reason `method` is stored next to the number.
    """

    var: float
    expected_shortfall: float
    confidence: float
    method: str
    n_observations: int


def value_at_risk(
    returns: pd.Series,
    *,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    method: str = "historical",
) -> ValueAtRisk | None:
    """One-period VaR + expected shortfall of `returns` (Section 7.7).

    `method="historical"` takes the empirical `1 - confidence` quantile and
    averages the observations at or below it; `method="parametric"` assumes
    normality and evaluates the same quantile analytically. Returns `None` when
    the sample is too small for the chosen estimator to mean anything --
    `_min_historical_var_obs(confidence)` observations for the historical
    method (100 at 95%), `_MIN_RETURN_OBS` for the parametric one, which
    estimates only a mean and a standard deviation -- or when the series has no
    usable variance.

    Deliberately **not** offered: a `horizon_days` argument that scales a
    one-period VaR by sqrt(t). That scaling is only valid for independent,
    identically distributed returns, which real return series are not
    (volatility clusters), and it reliably understates multi-day risk in exactly
    the stressed periods a reader cares about. A caller who wants a 10-day VaR
    should pass 10-day returns.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if method not in ("historical", "parametric"):
        raise ValueError(f"method must be 'historical' or 'parametric', got {method!r}")

    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    tail_probability = 1.0 - confidence

    if method == "historical":
        if values.size < _min_historical_var_obs(confidence):
            return None
        quantile = float(np.quantile(values, tail_probability))
        tail = values[values <= quantile]
        shortfall = float(tail.mean()) if tail.size else quantile
    else:
        if values.size < _MIN_RETURN_OBS:
            return None
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        scale = float(np.abs(values).max())
        if not np.isfinite(std) or std <= scale * _DEGENERATE_STD_REL_TOL:
            return None
        z = float(norm.ppf(tail_probability))
        quantile = mean + std * z
        # Closed-form normal expected shortfall: the conditional mean of the
        # tail below the quantile.
        shortfall = mean - std * float(norm.pdf(z)) / tail_probability

    return ValueAtRisk(
        var=-quantile,
        expected_shortfall=-shortfall,
        confidence=confidence,
        method=method,
        n_observations=int(values.size),
    )


# --------------------------------------------------------------------------- #
# Correlation across holdings (Section 7.7, Section 9)
# --------------------------------------------------------------------------- #


def correlation_matrix(
    returns: pd.DataFrame, *, min_periods: int = _MIN_CORRELATION_OBS
) -> pd.DataFrame:
    """Pairwise return-correlation matrix across holdings (Section 9).

    Pairs with fewer than `min_periods` overlapping observations come back as
    NaN rather than as a correlation computed from a handful of days -- an
    unhelpful-looking blank is the honest rendering of "these two barely
    overlap", where a confident 0.87 from six shared days is not.

    Because pairs are evaluated on their own overlap, the result is not
    guaranteed positive semi-definite when holdings have different histories.
    That is fine for the diagnostic display Section 9 asks for; the optimization
    row (Section 27) needs a PSD covariance estimate and should build one from a
    common date window rather than reusing this matrix.

    `clustering.compute_return_correlation_matrix` (Section 7.1) is the same
    idea applied to the whole screened universe for clustering; this one is
    scoped to a portfolio's holdings and keeps the minimum-overlap guard.
    """
    return returns.corr(min_periods=min_periods)


def _upper_triangle_pairs(matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Every distinct (a, b, correlation) pair with a defined value, once each."""
    labels = list(matrix.columns)
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value):
                pairs.append((str(left), str(right), float(value)))
    return pairs


def average_pairwise_correlation(matrix: pd.DataFrame) -> float | None:
    """Mean off-diagonal correlation -- one number for "how correlated is this book?".

    Section 9's question ("are you diversified, or do you just own five things
    that all move together?") in scalar form. `None` when no pair has a defined
    correlation.
    """
    pairs = _upper_triangle_pairs(matrix)
    if not pairs:
        return None
    return float(np.mean([value for _, _, value in pairs]))


def most_correlated_pairs(matrix: pd.DataFrame, *, top_n: int = 5) -> list[tuple[str, str, float]]:
    """The `top_n` most correlated holding pairs, highest first.

    The actionable half of the correlation matrix: a heatmap shows that a
    concentration exists, this names it ("these two moved together at 0.94"),
    which is what a diversification warning needs to say. Ties break on symbol
    order, so the output is deterministic.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    pairs = _upper_triangle_pairs(matrix)
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return pairs[:top_n]


# --------------------------------------------------------------------------- #
# Portfolio aggregation (Section 9's portfolio-level analytics)
# --------------------------------------------------------------------------- #


def portfolio_returns(
    returns: pd.DataFrame, weights: Mapping[str, float], *, cash_weight: float = 0.0
) -> pd.Series:
    """Return series of a constant-weight portfolio over `returns`.

    **This is a hypothetical, and saying so is the whole point.** It applies the
    portfolio's *current* weights across past returns -- "what would this mix
    have done" -- not the path the holdings actually took, which depends on when
    each position was bought and sold and can only be reconstructed from the
    transaction log (Section 30, a separate row). Every metric derived from this
    series inherits that framing.

    Weights are renormalized to sum to 1 *including* `cash_weight`, so Section
    9's cash pseudo-position correctly dilutes the portfolio's volatility and
    VaR instead of being ignored. Dates where any held name lacks a return are
    dropped, so every reported period is a complete cross-section rather than a
    partial one silently rescaled.

    Raises `ValueError` on a symbol missing from `returns` rather than treating
    it as zero-return: that failure mode would quietly convert a data gap into
    an imaginary cash sleeve and understate risk.
    """
    if cash_weight < 0:
        raise ValueError(f"cash_weight must be >= 0, got {cash_weight}")
    negative = sorted(s for s, w in weights.items() if w < 0)
    if negative:
        raise ValueError(f"weights must be non-negative (long-only, Section 2); got {negative}")
    missing = sorted(s for s in weights if s not in returns.columns)
    if missing:
        raise ValueError(f"no return series for holdings: {missing}")

    total = float(sum(weights.values())) + float(cash_weight)
    if total <= 0:
        raise ValueError("weights (plus cash_weight) must sum to a positive number")
    held = list(weights)
    if not held:
        return pd.Series(dtype=float)

    aligned = returns[held].dropna()
    normalized = pd.Series({s: float(w) / total for s, w in weights.items()})
    return aligned.mul(normalized, axis=1).sum(axis=1)


@dataclass(frozen=True)
class RiskProfile:
    """The Section 7.7 risk block for a single name.

    Any field may be `None` -- each estimator independently declines when its own
    data floor isn't met, so a short-history name yields a partly-filled profile
    rather than an all-or-nothing failure or a set of fabricated numbers.
    """

    volatility: VolatilityProfile
    beta: BetaResult | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    value_at_risk: ValueAtRisk | None
    n_observations: int


def stock_risk_profile(
    returns: pd.Series,
    *,
    market_returns: pd.Series | None = None,
    implied_volatility: float | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
    var_method: str = "historical",
    volatility_lookback: int | None = None,
) -> RiskProfile:
    """Every Section 7.7 statistic for one name, computed from its return series.

    `market_returns` (see `equal_weight_market_returns` for the proxy to use
    until an index series is ingested) enables beta; `implied_volatility` is the
    stored `options_signals.atm_implied_volatility` where a chain existed.

    No estimator silently substitutes for another: if the historical VaR's data
    floor isn't met, `value_at_risk` is `None` rather than quietly falling back
    to the parametric estimate, whose normality assumption would understate the
    tail without the reader ever being told the method changed.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    return RiskProfile(
        volatility=volatility_profile(
            clean,
            implied_volatility=implied_volatility,
            periods_per_year=periods_per_year,
            lookback=volatility_lookback,
        ),
        beta=beta(clean, market_returns) if market_returns is not None else None,
        sharpe=sharpe_ratio(
            clean, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        sortino=sortino_ratio(
            clean, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        max_drawdown=max_drawdown(clean),
        value_at_risk=value_at_risk(clean, confidence=var_confidence, method=var_method),
        n_observations=int(clean.size),
    )


@dataclass(frozen=True)
class PortfolioRisk:
    """Section 9's portfolio-level risk block, plus the context to read it.

    `returns` is the constant-weight series everything else is derived from (see
    `portfolio_returns` for what it does and doesn't claim). `cash_weight` and
    `n_holdings` are carried because a low volatility means something very
    different at 60% cash than at 0%, and `average_correlation` /
    `most_correlated` answer Section 9's diversification question directly.
    """

    returns: pd.Series
    volatility: float | None
    beta: BetaResult | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    value_at_risk: ValueAtRisk | None
    correlations: pd.DataFrame
    average_correlation: float | None
    most_correlated: list[tuple[str, str, float]]
    n_holdings: int
    cash_weight: float
    n_observations: int


def portfolio_risk(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    cash_weight: float = 0.0,
    market_returns: pd.Series | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
    var_method: str = "historical",
    correlation_min_periods: int = _MIN_CORRELATION_OBS,
    top_correlated: int = 5,
) -> PortfolioRisk:
    """The portfolio-level analytics Section 9 asks for, from a return panel + weights.

    Beta is measured by regressing the portfolio's own return series on
    `market_returns`. That is equivalent to Section 9's "weighted average of
    holdings' individual betas" (beta is linear in the weights, so the two agree
    exactly when both are estimated over the same dates), but it needs no
    per-name coverage handling and reports a real R^2 and observation count for
    the portfolio as a whole. `portfolio_beta` remains available for a caller
    holding per-symbol betas without an aligned panel.

    The correlation matrix covers the held names only -- the diversification
    question is about what you own, not about the universe.
    """
    series = portfolio_returns(returns, weights, cash_weight=cash_weight)
    held = [s for s in weights if s in returns.columns]
    correlations = correlation_matrix(returns[held], min_periods=correlation_min_periods)

    return PortfolioRisk(
        returns=series,
        volatility=historical_volatility(series, periods_per_year=periods_per_year),
        beta=beta(series, market_returns) if market_returns is not None else None,
        sharpe=sharpe_ratio(
            series, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        sortino=sortino_ratio(
            series, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        max_drawdown=max_drawdown(series),
        value_at_risk=value_at_risk(series, confidence=var_confidence, method=var_method),
        correlations=correlations,
        average_correlation=average_pairwise_correlation(correlations),
        most_correlated=most_correlated_pairs(correlations, top_n=top_correlated),
        n_holdings=len(held),
        cash_weight=float(cash_weight),
        n_observations=int(series.size),
    )
