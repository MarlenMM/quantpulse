"""Portfolio optimization -- mean-variance, HRP, Black-Litterman (Section 27).

Section 21 marks this row **Opus / High** and names the failure mode exactly:
"getting the 'views' input, covariance estimation, and constraints wrong
produces a target allocation that looks precise but is quietly unstable or
unreasonable." An optimizer always returns weights that sum to 1 and always
looks authoritative, so there is no crash to warn you -- the number of decimal
places is the same whether the inputs were sound or nonsense. Every choice
below exists to close one of those three doors, and three of them were settled
by running the real library rather than by reading its docstrings:

1. **Covariance is shrunk, never raw sample covariance.** Mean-variance
   optimization is an "error maximizer": it concentrates weight precisely where
   covariance is *underestimated* and expected return *overestimated*, so the
   noisiest inputs get the biggest bets. Sample covariance across N names on T
   days is badly conditioned whenever T is not >> N. `annualized_covariance`
   uses Ledoit-Wolf shrinkage (`risk_models.CovarianceShrinkage`) as the only
   default, computed over a **common date window** so no pair's covariance is
   measured on a different sample than another's -- which is also why
   `risk.correlation_matrix`'s pairwise-overlap matrix is deliberately *not*
   reused here (its own docstring says so).

2. **The Black-Litterman prior is reverse-optimized from equal weights, not
   `pi="equal"`.** PyPortfolioOpt's `pi="equal"` sets the prior return vector to
   `np.ones(n) / n` -- that is `1/n` used as an *expected return*, so a 6-name
   universe gets a 16.7% prior and a 500-name universe gets 0.2%, purely as a
   function of universe size. Verified against the installed library, not
   assumed. `equal_weight_prior` instead does the textbook reverse optimization
   (`delta * S @ w_eq`), which is what "equilibrium prior" actually means. A
   cap-weighted equilibrium would be better still, but market cap is not stored
   anywhere in the schema (Section 13) -- an equal-weight equilibrium is the
   honest stand-in, and it is the same proxy the backtest benchmark and
   `risk.equal_weight_market_returns` already use. Round-trip property, asserted
   in the tests: optimizing the equal-weight-implied prior returns equal weights.

3. **Views are tilts around that prior, so a neutral universe changes nothing.**
   A composite score of 90 does not mean "this will return 12%" -- it means
   "top decile among peers". `views_from_composite_scores` centers the scores on
   their own cross-sectional mean and maps them to at most `max_view_tilt` of
   annual excess return around the prior. The property that makes this honest is
   testable and tested: a universe where every stock scores identically produces
   views identical to the prior, hence a posterior identical to the prior, hence
   the equal-weight allocation -- no tilt invented out of a signal that wasn't
   there.

4. **Per-name view confidence is opt-in, and here is why it is not the default.**
   Feeding Section 7.5's `data_confidence` into the view uncertainty is an
   appealing idea, and it is available via `view_confidences`. But Black-Litterman
   propagates every view through the covariance matrix, so when views on all N
   names carry *different* certainties, a confident bearish view on one name
   drags correlated names down with it. Measured on real library output: with
   mixed confidences, the posterior tilt stopped being monotonic in the scores
   and several names' tilts flipped sign against their own view -- the
   top-scored name received a *smaller* weight than the third-ranked one. That
   is mathematically correct Black-Litterman and exactly the "looks precise but
   is quietly unreasonable" outcome Section 21 warns about, so uniform
   confidence is the default and the realized tilt is exposed on the result for
   inspection rather than hidden inside it.

Scope: this row is the optimization *math* -- target weights, nothing else. The
concrete **rebalancing trade list** ("sell 12 shares of X") is Section 21's own
separate Sonnet row and is not here; neither is the portfolio bookkeeping that
would supply current share counts (Section 30). `kelly_position_fraction` is
included because Section 27 raises it in this section and it has no Section-21
row of its own -- leaving it unbuilt is how Section 28's overlay went missing
across all four Phase-4 sub-parts. Section 2's "no goal-based planning" boundary
is respected throughout: this optimizes the mathematics of an allocation, and
never reasons about anyone's personal goals or timeline.

Pure functions: prices and scores in, weights out. No storage, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as _sch

# --------------------------------------------------------------------------- #
# Upstream compatibility shim -- must run before pypfopt.HRPOpt is used.
#
# PyPortfolioOpt 1.6.0's `HRPOpt.optimize` validates its `linkage_method`
# argument against `scipy.cluster.hierarchy._LINKAGE_METHODS`, a *private*
# scipy attribute removed in scipy 1.18 (this project pins scipy>=1.18). The
# check is the only thing that breaks: every other line of that method uses
# public scipy API (`linkage`, `squareform`), and the HRP math itself is
# untouched. Restoring the constant is a narrower, more honest fix than pinning
# scipy backwards for the whole project or reimplementing recursive bisection
# ourselves. Remove once PyPortfolioOpt drops the private-attribute check.
# --------------------------------------------------------------------------- #
if not hasattr(_sch, "_LINKAGE_METHODS"):  # pragma: no cover - environment-dependent
    _sch._LINKAGE_METHODS = {
        "single": 0,
        "complete": 1,
        "average": 2,
        "weighted": 3,
        "centroid": 4,
        "median": 5,
        "ward": 6,
    }

from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models  # noqa: E402
from pypfopt.black_litterman import (  # noqa: E402
    BlackLittermanModel,
    market_implied_prior_returns,
)
from pypfopt.exceptions import OptimizationError  # noqa: E402

__all__ = [
    "DEFAULT_RISK_AVERSION",
    "DEFAULT_TAU",
    "DEFAULT_MAX_WEIGHT",
    "DEFAULT_MAX_VIEW_TILT",
    "DEFAULT_KELLY_FRACTION",
    "TRADING_DAYS_PER_YEAR",
    "OptimizedPortfolio",
    "usable_common_window",
    "annualized_covariance",
    "equal_weight_prior",
    "views_from_composite_scores",
    "mean_variance_optimize",
    "hierarchical_risk_parity",
    "black_litterman_optimize",
    "kelly_position_fraction",
]

TRADING_DAYS_PER_YEAR = 252

# Black-Litterman's risk-aversion coefficient (delta). 2.0-2.5 is the
# conventional range in the original literature; it scales the equilibrium
# prior's magnitude, not the relative ordering, so the allocation is far less
# sensitive to it than to the covariance estimate.
DEFAULT_RISK_AVERSION = 2.0
# Tau scales the uncertainty of the prior itself. 0.05 is PyPortfolioOpt's own
# default and the usual choice in practice.
DEFAULT_TAU = 0.05

# The largest weight any single name may take. Section 9 flags a position above
# ~15% of the portfolio as a concentration risk, so an optimizer that happily
# proposed 40% in one name would contradict the app's own warning one page over.
DEFAULT_MAX_WEIGHT = 0.15

# The most annual excess return a maximally-attractive composite score is
# allowed to claim over the equilibrium prior. Deliberately modest: this is a
# rank signal being translated into return space, and a large number here would
# let the ranking overwhelm the covariance structure entirely.
DEFAULT_MAX_VIEW_TILT = 0.05

# Section 27: "always fractional (e.g. quarter-Kelly), since full Kelly is
# well-known to be too aggressive for real use".
DEFAULT_KELLY_FRACTION = 0.25

# Minimums below which an optimization is not worth running: covariance across
# N names needs meaningfully more than N observations to be estimable at all,
# and a one-asset "optimization" has only one answer.
_MIN_OBSERVATIONS = 60
_MIN_ASSETS = 2


@dataclass(frozen=True)
class OptimizedPortfolio:
    """A target allocation plus the assumptions that produced it.

    `weights` sums to 1 and is long-only (Section 2). `expected_return` /
    `volatility` / `sharpe` are PyPortfolioOpt's ex-ante estimates *under this
    method's own inputs* -- they are a description of the optimizer's beliefs,
    not a forecast, and specifically not a backtested track record (that is
    `backtest.py`'s job, and the honest place to look for whether any of this
    worked).

    `method` and `n_observations` travel with the weights because an allocation
    is only as trustworthy as the window it was estimated on, and because HRP
    and mean-variance answers are not comparable like-for-like. `view_tilt`
    (Black-Litterman only) is the realized posterior return tilt versus the
    prior, per name -- exposed rather than hidden so a caller can see when the
    covariance coupling moved a name against its own view.
    """

    weights: dict[str, float]
    method: str
    expected_return: float | None
    volatility: float | None
    sharpe: float | None
    n_assets: int
    n_observations: int
    max_weight: float | None
    view_tilt: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Inputs: a clean common-window panel and a shrunk covariance
# --------------------------------------------------------------------------- #


def _clean_panel(prices: pd.DataFrame, *, min_observations: int) -> pd.DataFrame:
    """The common-date, strictly-positive sub-panel the estimators can share.

    Drops columns that are entirely missing, then drops any date where a
    surviving name has no price, so every covariance entry is measured over the
    *same* sample. Estimating pair (A, B) on one window and (A, C) on another is
    what produces a covariance matrix that isn't positive semi-definite, which
    an optimizer will happily consume and turn into a confident, meaningless
    allocation.
    """
    frame = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0).dropna(axis=1, how="all")
    frame = frame.dropna(axis=0, how="any")
    return frame


def usable_common_window(
    prices: pd.DataFrame, *, min_observations: int = _MIN_OBSERVATIONS
) -> tuple[pd.DataFrame, list[str]]:
    """The largest sub-panel meeting `min_observations`, plus the names dropped to get it.

    `_clean_panel` requires a *common* date window so every covariance entry is
    measured on the same sample -- necessary, but it means one recently-listed
    holding truncates the window for everything else. Measured on real data: a
    panel of eight names over 289 trading days collapsed to 22 usable rows
    because a single symbol had only 22 bars, and all three optimizers then
    (correctly) abstained. A user seeing "no allocation could be computed" for
    an eight-holding portfolio has no way to guess that one name caused it.

    So the short-history names are identified and returned rather than left
    implicit: drop whichever holding starts latest, re-measure, repeat until the
    common window is long enough or fewer than `_MIN_ASSETS` names remain. The
    caller can then optimize over what's left *and say which holdings were
    excluded and why*, which is a far more useful answer than silence.

    Returns `(panel, excluded_symbols)`; `excluded_symbols` is sorted and empty
    when nothing had to be dropped.
    """
    working = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    working = working.where(working > 0).dropna(axis=1, how="all")
    excluded: list[str] = []

    while working.shape[1] >= _MIN_ASSETS:
        common = working.dropna(axis=0, how="any")
        if len(common) >= min_observations:
            return common, sorted(excluded)
        starts = [
            (start, column)
            for column in working.columns
            if (start := working[column].first_valid_index()) is not None
        ]
        if not starts:
            break
        # The name whose history starts latest is the one costing the most rows.
        latest = max(starts)[1]
        excluded.append(str(latest))
        working = working.drop(columns=[latest])

    return pd.DataFrame(), sorted(excluded)


def annualized_covariance(
    prices: pd.DataFrame, *, frequency: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Annualized Ledoit-Wolf-shrunk covariance of `prices`' returns.

    Shrinkage pulls the noisy sample covariance toward a structured target,
    which is what keeps the matrix invertible and the resulting weights stable.
    Raw sample covariance is deliberately not offered as an option: on any
    universe this project screens, it is the wrong default and its failure is
    silent (see the module docstring).
    """
    return risk_models.CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()


def _normalized_weights(raw: dict[str, float], *, cutoff: float = 1e-4) -> dict[str, float]:
    """Drop negligible positions and renormalize so the weights sum to exactly 1.

    PyPortfolioOpt's `clean_weights` rounds to five decimals, which leaves the
    total a hair off 1.0; a caller multiplying by a portfolio value would
    silently lose or invent a few cents per rebalance.
    """
    kept = {symbol: float(weight) for symbol, weight in raw.items() if abs(weight) >= cutoff}
    total = sum(kept.values())
    if total <= 0:
        return {}
    return {symbol: weight / total for symbol, weight in kept.items()}


def _validate_bounds(n_assets: int, max_weight: float | None) -> None:
    """Reject a max-weight cap that makes a fully-invested long-only portfolio impossible."""
    if max_weight is None:
        return
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    if max_weight * n_assets < 1.0:
        raise ValueError(
            f"max_weight={max_weight} cannot fill a portfolio of {n_assets} assets "
            f"(needs >= {1.0 / n_assets:.4f}); the solver would report an infeasible problem"
        )


# --------------------------------------------------------------------------- #
# Mean-variance (the classic efficient frontier)
# --------------------------------------------------------------------------- #


def mean_variance_optimize(
    prices: pd.DataFrame,
    *,
    objective: str = "max_sharpe",
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    risk_free_rate: float = 0.0,
    frequency: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = _MIN_OBSERVATIONS,
) -> OptimizedPortfolio | None:
    """Classic Markowitz optimization on the efficient frontier (Section 27).

    `objective` is `"max_sharpe"` (the tangency portfolio) or `"min_volatility"`
    (the frontier's left-most point, which needs no expected-return estimate at
    all and is correspondingly far more stable).

    **This is the least robust of the three methods here, and the reason is the
    expected-return input, not the optimizer.** `max_sharpe` requires estimating
    each asset's expected return from its own history, and historical mean
    return is a notoriously noisy estimator -- a name that happened to run up
    over the window looks permanently attractive. Section 27 recommends HRP
    (needs no expected returns) or Black-Litterman (views anchored to an
    equilibrium prior) precisely to avoid this step. `min_volatility` sidesteps
    it too, which is why it is worth preferring when a covariance-only answer
    will do.

    Returns `None` rather than raising when the window is too short, too few
    names survive, or the solver cannot find a feasible optimum -- an absent
    allocation is a usable signal, a fabricated one is not.
    """
    if objective not in ("max_sharpe", "min_volatility"):
        raise ValueError(f"objective must be 'max_sharpe' or 'min_volatility', got {objective!r}")

    panel = _clean_panel(prices, min_observations=min_observations)
    if len(panel) < min_observations or panel.shape[1] < _MIN_ASSETS:
        return None
    _validate_bounds(panel.shape[1], max_weight)

    covariance = annualized_covariance(panel, frequency=frequency)
    mu = expected_returns.mean_historical_return(panel, frequency=frequency)
    bounds = (0.0, max_weight if max_weight is not None else 1.0)

    try:
        frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)
        if objective == "max_sharpe":
            frontier.max_sharpe(risk_free_rate=risk_free_rate)
        else:
            frontier.min_volatility()
        weights = _normalized_weights(dict(frontier.clean_weights()))
        performance = frontier.portfolio_performance(risk_free_rate=risk_free_rate)
    except (OptimizationError, ValueError):
        # `max_sharpe` raises ValueError when no asset's expected return clears
        # the risk-free rate -- a real, meaningful "no tangency portfolio
        # exists here" answer, not an internal error to paper over.
        return None
    if not weights:
        return None

    return OptimizedPortfolio(
        weights=weights,
        method=f"mean_variance:{objective}",
        expected_return=float(performance[0]),
        volatility=float(performance[1]),
        sharpe=float(performance[2]),
        n_assets=len(weights),
        n_observations=len(panel),
        max_weight=max_weight,
    )


# --------------------------------------------------------------------------- #
# Hierarchical Risk Parity
# --------------------------------------------------------------------------- #


def hierarchical_risk_parity(
    prices: pd.DataFrame,
    *,
    linkage_method: str = "single",
    frequency: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = _MIN_OBSERVATIONS,
    risk_free_rate: float = 0.0,
) -> OptimizedPortfolio | None:
    """Hierarchical Risk Parity: allocation from the correlation *structure* alone.

    Section 27 calls HRP "a more robust alternative that doesn't require the
    fragile step of estimating expected returns", and that is the whole appeal:
    it clusters assets by correlation, then splits risk down the resulting tree,
    so it never inverts the covariance matrix and never needs a return forecast.
    On a universe where the return estimates are mostly noise -- which is the
    normal case -- it tends to produce far more stable weights than mean-variance
    across rebalances.

    Note that HRP takes no weight cap: the allocation falls out of the cluster
    tree rather than a constrained solve, so `max_weight` is reported as `None`.
    Concentration is a property to *check* on the output here (Section 9's
    threshold), not a constraint that can be imposed on the way in.

    This is the same correlation-clustering idea `analysis/clustering.py`
    (Section 7.1) applies for diversification diagnostics, used here for
    allocation instead of grouping.
    """
    panel = _clean_panel(prices, min_observations=min_observations)
    if len(panel) < min_observations or panel.shape[1] < _MIN_ASSETS:
        return None

    returns = panel.pct_change().dropna(how="any")
    if returns.empty:
        return None

    try:
        model = HRPOpt(returns)
        model.optimize(linkage_method=linkage_method)
        weights = _normalized_weights(dict(model.clean_weights()))
        performance = model.portfolio_performance(risk_free_rate=risk_free_rate)
    except (OptimizationError, ValueError, AttributeError):
        return None
    if not weights:
        return None

    return OptimizedPortfolio(
        weights=weights,
        method="hierarchical_risk_parity",
        expected_return=float(performance[0]),
        volatility=float(performance[1]),
        sharpe=float(performance[2]),
        n_assets=len(weights),
        n_observations=len(panel),
        max_weight=None,
    )


# --------------------------------------------------------------------------- #
# Black-Litterman -- where the composite scores become the views (Section 27)
# --------------------------------------------------------------------------- #


def equal_weight_prior(
    covariance: pd.DataFrame, *, risk_aversion: float = DEFAULT_RISK_AVERSION
) -> pd.Series:
    """Equilibrium expected returns implied by an equal-weight portfolio.

    The textbook Black-Litterman prior is reverse-optimized from the market
    portfolio: `pi = delta * Sigma @ w_market`, i.e. "what returns would make
    today's market weights optimal?". Market-cap weights aren't available (no
    `market_cap` column exists in Section 13's schema), so this uses equal
    weights -- the same market proxy the backtest benchmark and
    `risk.equal_weight_market_returns` already stand on, which at least keeps
    the whole project honest about using one definition of "the market".

    The consequence worth knowing: an equal-weight equilibrium implies higher
    prior returns for higher-covariance names, which is a real and defensible
    equilibrium, just not the cap-weighted one the original model assumes.
    """
    symbols = list(covariance.columns)
    equal = pd.Series(1.0 / len(symbols), index=symbols)
    return market_implied_prior_returns(equal, risk_aversion, covariance)


def views_from_composite_scores(
    scores: pd.Series,
    prior: pd.Series,
    *,
    max_tilt: float = DEFAULT_MAX_VIEW_TILT,
    confidences: pd.Series | None = None,
) -> pd.Series:
    """Turn Section 7.5 composite scores into Black-Litterman absolute views.

    This is the architectural connection Section 27 is most pleased about: the
    same ranking engine that drives the Screener also drives portfolio
    construction, rather than the two being unrelated features.

    The mapping is deliberately a *tilt around the prior*, never a standalone
    return forecast. Scores are centered on their own cross-sectional mean and
    scaled so the most extreme name claims at most `max_tilt` of annual excess
    return; the view for asset i is `prior_i + tilt_i`. Two properties follow,
    both of which the tests assert:

    * A universe scoring identically everywhere yields views equal to the prior,
      so the posterior is the prior and the allocation is the equilibrium one.
      A ranking with no dispersion contains no information, and this is what
      "contains no information" has to look like downstream.
    * The mapping is rank-preserving and sign-symmetric about the mean, so a
      score above the universe average always produces a positive tilt.

    `confidences` shrinks each view *toward the prior* individually -- a
    thinly-covered stock makes a weaker claim rather than an equally loud one.
    It must be a **0-1 fraction**; Section 7.5's `data_confidence` is on a
    0-100 scale, so pass `data_confidence / 100.0`. Values outside [0, 1] raise
    rather than being clipped, because clipping would silently turn every
    `data_confidence` into 1.0 and quietly disable the whole mechanism. Read
    `black_litterman_optimize`'s docstring before using it: non-uniform
    confidences interact with the covariance coupling in a way that can move a
    name against its own view.
    """
    if max_tilt <= 0:
        raise ValueError(f"max_tilt must be > 0, got {max_tilt}")

    aligned = pd.to_numeric(scores, errors="coerce").reindex(prior.index).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    centered = aligned - aligned.mean()
    spread = float(centered.abs().max())
    tilt = centered * 0.0 if spread <= 0 else centered / spread * max_tilt

    if confidences is not None:
        weight = pd.to_numeric(confidences, errors="coerce").reindex(tilt.index).fillna(0.0)
        # Reject an out-of-range scale loudly instead of clipping it away. This
        # parameter's documented source is Section 7.5's `data_confidence`,
        # which `scoring.build_composite` emits on a **0-100** scale -- and a
        # silent `.clip(0, 1)` would map every one of those values to exactly
        # 1.0, so a thinly-covered stock and a fully-covered one would make
        # equally loud claims while the caller believed they were being
        # down-weighted. That is the failure this check exists to prevent: pass
        # `data_confidence / 100.0`.
        out_of_range = weight[(weight < 0.0) | (weight > 1.0)]
        if not out_of_range.empty:
            worst = float(out_of_range.abs().max())
            raise ValueError(
                "confidences must be fractions in [0, 1], but "
                f"{len(out_of_range)} value(s) fall outside it (max magnitude "
                f"{worst:g}). Section 7.5's `data_confidence` is on a 0-100 "
                "scale -- pass `data_confidence / 100.0`."
            )
        tilt = tilt * weight

    return prior.reindex(tilt.index) + tilt


def black_litterman_optimize(
    prices: pd.DataFrame,
    composite_scores: pd.Series,
    *,
    confidences: pd.Series | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    max_view_tilt: float = DEFAULT_MAX_VIEW_TILT,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    tau: float = DEFAULT_TAU,
    risk_free_rate: float = 0.0,
    frequency: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = _MIN_OBSERVATIONS,
) -> OptimizedPortfolio | None:
    """Black-Litterman allocation with the composite score as the view (Section 27).

    Blends an equal-weight equilibrium prior (`equal_weight_prior`) with views
    derived from Section 7.5's composite scores (`views_from_composite_scores`),
    then runs the posterior through a long-only, weight-capped max-Sharpe solve.
    Because the views are anchored to an equilibrium rather than estimated from
    each name's own past returns, this avoids `mean_variance_optimize`'s most
    fragile input while still expressing the ranking.

    **On `confidences`.** Passing per-name `data_confidence` is supported and is
    the theoretically appealing thing to do, but it is not the default, because
    Black-Litterman propagates each view through the covariance matrix: when
    views on all names carry different certainties, a confident view on one name
    pulls its correlated neighbours along with it. Measured against the real
    library on correlated synthetic data, mixed confidences made the posterior
    tilt non-monotonic in the scores and flipped several names' tilts against
    their own view -- the top-ranked name ended up with a smaller weight than
    the third-ranked one. Nothing is broken when that happens; it is what the
    model actually says. But it is surprising enough that it should be an
    explicit choice, and `OptimizedPortfolio.view_tilt` reports the realized
    per-name tilt so the effect is visible instead of buried.

    Returns `None` when the panel is too short, too few names overlap between
    prices and scores, or the solver finds no feasible optimum.
    """
    panel = _clean_panel(prices, min_observations=min_observations)
    if len(panel) < min_observations or panel.shape[1] < _MIN_ASSETS:
        return None

    covariance = annualized_covariance(panel, frequency=frequency)
    prior = equal_weight_prior(covariance, risk_aversion=risk_aversion)
    views = views_from_composite_scores(
        composite_scores, prior, max_tilt=max_view_tilt, confidences=confidences
    )
    if len(views) < _MIN_ASSETS:
        return None

    # Restrict the covariance to the names that actually carry a view, so the
    # prior, the views and the solve all describe the same universe.
    symbols = list(views.index)
    covariance = covariance.loc[symbols, symbols]
    prior = prior.loc[symbols]
    _validate_bounds(len(symbols), max_weight)

    # Diagonal view-uncertainty matrix proportional to each asset's own prior
    # variance -- PyPortfolioOpt's default shape, constructed explicitly so the
    # confidence handling above stays the single place uncertainty is expressed.
    omega = np.diag(tau * np.diag(covariance.to_numpy()))
    bounds = (0.0, max_weight if max_weight is not None else 1.0)

    try:
        model = BlackLittermanModel(
            covariance, pi=prior, absolute_views=views, omega=omega, tau=tau
        )
        posterior_returns = model.bl_returns()
        posterior_cov = model.bl_cov()
        frontier = EfficientFrontier(posterior_returns, posterior_cov, weight_bounds=bounds)
        frontier.max_sharpe(risk_free_rate=risk_free_rate)
        weights = _normalized_weights(dict(frontier.clean_weights()))
        performance = frontier.portfolio_performance(risk_free_rate=risk_free_rate)
    except (OptimizationError, ValueError):
        return None
    if not weights:
        return None

    tilt = posterior_returns - prior
    return OptimizedPortfolio(
        weights=weights,
        method="black_litterman",
        expected_return=float(performance[0]),
        volatility=float(performance[1]),
        sharpe=float(performance[2]),
        n_assets=len(weights),
        n_observations=len(panel),
        max_weight=max_weight,
        view_tilt={str(symbol): float(value) for symbol, value in tilt.items()},
    )


# --------------------------------------------------------------------------- #
# Fractional Kelly position sizing (Section 27; no Section-21 row of its own)
# --------------------------------------------------------------------------- #


def kelly_position_fraction(
    hit_rate: float | None,
    payoff_ratio: float | None,
    *,
    fraction: float = DEFAULT_KELLY_FRACTION,
    max_position: float = DEFAULT_MAX_WEIGHT,
) -> float | None:
    """Fractional-Kelly position size from a strategy's own track record (Section 27).

    The Kelly criterion for a bet winning `p` of the time at odds `b` is
    `f* = (b*p - (1-p)) / b` -- the growth-optimal fraction of capital. Feed it
    the backtest's realized `hit_rate` and payoff ratio (mean win / mean loss,
    both from `backtest.py`, which is the only place in this project that
    measures either honestly) and it answers "how much" rather than merely
    "add or trim".

    Three guards, all of which matter more than the formula:

    * **Always fractional.** `fraction` defaults to a quarter (Section 27:
      "always fractional... since full Kelly is well-known to be too aggressive
      for real use"). Full Kelly is growth-optimal only if your probability
      estimate is exactly right; it is brutally punishing when it is optimistic,
      and a hit rate measured on a few dozen backtest periods is certainly not
      exactly right.
    * **Capped.** The result is clipped to `max_position`, so a flattering
      short-sample edge cannot propose a concentration Section 9 would warn
      about on the very next page.
    * **Never negative.** A negative Kelly means "bet the other way", which in a
      long-only tool (Section 2) means "don't hold this" -- returned as 0.0.

    `None` when either input is missing or nonsensical (a non-positive payoff
    ratio, a hit rate outside [0, 1]) -- the same "no number beats a fabricated
    number" discipline as the rest of the analysis layer. This is one input to a
    decision, never an instruction.
    """
    if hit_rate is None or payoff_ratio is None:
        return None
    if pd.isna(hit_rate) or pd.isna(payoff_ratio):
        return None
    if not 0.0 <= hit_rate <= 1.0 or payoff_ratio <= 0:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if max_position <= 0:
        raise ValueError(f"max_position must be > 0, got {max_position}")

    edge = (payoff_ratio * hit_rate - (1.0 - hit_rate)) / payoff_ratio
    if edge <= 0:
        return 0.0
    return float(min(edge * fraction, max_position))
