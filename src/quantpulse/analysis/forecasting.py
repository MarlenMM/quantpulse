"""Price forecasting models -- baseline, statistical, and ML (Section 7.6).

Section 21 flags forecasting as **Opus / High**: the subtlety here isn't in the
libraries (statsmodels and scikit-learn do the fitting), it's in the framing.
Three decisions, made once here, are what keep every downstream number honest:

1. **Forecast forward RETURNS, not absolute prices.** Predicting a price level
   is an ill-posed, non-stationary problem; predicting the forward return over a
   fixed horizon is stationary, standard, and far more tractable (Section 7.6).
   Internally everything works in *log-return* space (additive across time,
   symmetric); the public `Forecast` reports both the simple return and the
   implied price so the UI can render either.

2. **No look-ahead / no label leakage.** The ML target for a date `t` is the
   return realized from `t` to `t + horizon` (`log_close.shift(-h) - log_close`).
   The most recent `h` rows therefore have no known target -- they are
   *predict-only*, never trained on. Silently training on them (with a truncated
   or forward-filled target) is the classic leakage bug that makes a backtest
   look brilliant and mean nothing (Section 22). We drop them explicitly, and
   the one row we predict is the as-of date itself. Every engineered feature is
   trailing-only, so a feature for date `t` never peeks past `t`.

3. **Different horizons get different feature SETS, not just a different number
   of days** (Section 7.6). Short horizons lean on technical/momentum signals;
   long horizons pull in fundamentals/valuation and macro, which mean-revert
   over quarters. Because gradient-boosted trees are invariant to any monotonic
   per-feature rescaling, "weighting" a feature by multiplying its column would
   be a no-op dressed up as a signal -- so the horizon emphasis is expressed as
   genuine feature *inclusion* (`horizon_feature_weights` -> `select_features`),
   which actually changes the fitted model.

Scope note (Phase 7 is split across five Section-21 rows, in order): row 1
(Opus) built the baseline/statistical/ML models above; row 2 (Opus) built the
walk-forward backtest and cost/survivorship-aware strategy engine
(`analysis/backtest.py`), which also fills each forecast's own
`historical_hit_rate` and owns persistence to the `forecasts` table (Section
13); row 3 (Opus) hardened the cold-start backfill's survivorship-coverage
reporting and index-reconstitution handling (`scripts/seed_initial_data.py`,
`scripts/refresh_data.py`) -- unrelated files, not this module.

This row (4, Sonnet -- "well-defined, standard technique") adds the **Monte
Carlo** GBM fan-chart simulation below. It deliberately does *not* produce a
`Forecast` or feed the backtest/`forecasts`-table pipeline: a GBM path
calibrated to historical mean/volatility is the exact same random-walk-with-
drift model `baseline_forecast` already evaluates in closed form, so treating
it as a fourth competing model would just grade baseline against itself under
simulation noise. Its actual job is different in *kind* -- a full day-by-day
percentile path (the "fan chart" Section 7.6 calls "the most honest way to
visually communicate 'possible price forecast'"), not a single point+band at
one horizon -- so it gets its own result type. Row 5 (Opus, not yet built)
adds the bootstrap confidence intervals around the backtest's headline
Sharpe/CAGR. Every function here is pure: prices in, results out; no storage
or network.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Forecast",
    "DEFAULT_HORIZONS",
    "DEFAULT_CONFIDENCE",
    "baseline_forecast",
    "statistical_forecast",
    "ml_forecast",
    "forecast_horizon",
    "generate_forecasts",
    "horizon_feature_weights",
    "build_features",
    "select_features",
    "MonteCarloFanChart",
    "simulate_gbm_paths",
    "monte_carlo_fan_chart",
]

# Horizons in *trading* days: ~1 week, 1 month, 1 quarter, 1 year (Section 7.6
# forecasts "5-day, 20-day" and the longer 3-month/1-year emphasis shift).
DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 63, 252)
DEFAULT_CONFIDENCE = 0.90

# Multi-lookback momentum/volatility windows (trading days). These are the
# lookback-gated technical features -- a 1-year momentum feature has no business
# driving a 5-day forecast, so `select_features` filters them by horizon.
_MOMENTUM_LOOKBACKS: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
_VOL_LOOKBACKS: tuple[int, ...] = (10, 21, 63)

# Minimum history each model needs before it will produce anything rather than a
# spuriously-confident number off a handful of bars.
_MIN_BASELINE_BARS = 20
_MIN_ARIMA_BARS = 60
# Drift specification for the ARIMA fit. statsmodels silently defaults to "n"
# (no trend) when d > 0, which makes the h-step forecast flat at every horizon;
# "t" is a time trend on the undifferenced log price, i.e. drift on the returns.
# See `statistical_forecast`'s docstring for the measurements behind this.
_ARIMA_TREND = "t"
_MIN_ML_TRAIN_ROWS = 120
_MIN_ML_VAL_ROWS = 20
# Minimum core-fold rows left *after* purging the `horizon_days` rows whose
# labels would otherwise reach into the holdout window (see `ml_forecast`).
# Below this there isn't enough clean history to fit a band-estimating model,
# so the empirical band is declined rather than fit on a handful of rows.
_MIN_ML_CORE_ROWS = 60
# A feature needs at least this many non-NaN rows in the fold being fit to be
# usable: a column that's all-NaN within a fold (a long-lookback feature still
# warming up over a short training window) can't be binned at all, and one with
# only a handful of values can't be learned. Same coverage discipline the
# scoring modules use -- an under-covered feature drops out rather than crashing
# or contributing noise.
_MIN_FEATURE_NONNULL = 20

# Fixed, modest gradient-boosting hyperparameters -- deliberately NOT tuned to a
# backtest. Section 22 warns that searching hyperparameters for the best
# backtested Sharpe is fitting historical noise; a small, regularized, fixed
# model is the honest default. Early stopping is OFF on purpose (see
# `ml_forecast`): sklearn's internal validation split shuffles, which would leak
# time order.
_ML_MAX_ITER = 200
_ML_LEARNING_RATE = 0.05
_ML_MAX_DEPTH = 3
_ML_MIN_LEAF = 20
_ML_L2 = 1.0
_ML_VAL_FRACTION = 0.2

# Horizon -> feature-bucket weights (anchors; linearly interpolated between).
# Short horizons ride technical/momentum; long horizons lean fundamental +
# macro (Section 7.6). A bucket is *included* only when its weight clears
# `_BUCKET_INCLUDE_MIN`, so a 5-day model genuinely never sees the valuation
# features -- a different feature set, not merely a rescaled one.
_HORIZON_BUCKET_ANCHORS: dict[int, dict[str, float]] = {
    5: {"technical": 0.80, "fundamental": 0.10, "macro": 0.10},
    20: {"technical": 0.55, "fundamental": 0.30, "macro": 0.15},
    63: {"technical": 0.35, "fundamental": 0.45, "macro": 0.20},
    252: {"technical": 0.20, "fundamental": 0.60, "macro": 0.20},
}
_BUCKETS = ("technical", "fundamental", "macro")
_BUCKET_INCLUDE_MIN = 0.15

# Lookback-gated technical features survive only if their window is within
# `_LOOKBACK_HORIZON_MULT` x horizon (with a floor so the shortest few are
# always kept). Realizes "different number of days -> different lookbacks."
_LOOKBACK_HORIZON_MULT = 4
_LOOKBACK_FLOOR = 21

# "Thousands" of Monte Carlo paths (Section 7.6); vectorized NumPy makes even
# this many trivially fast (milliseconds) at any horizon in DEFAULT_HORIZONS.
_DEFAULT_MC_PATHS = 10_000


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Forecast:
    """One model's forecast for one horizon, carried in both return and price terms.

    `point_return` is the forecast forward *simple* return over `horizon_days`
    (e.g. 0.03 == +3%); `[lower_return, upper_return]` is the `confidence_level`
    band around it. The `*_price` fields are the same three numbers translated
    through `last_close` for the fan chart. `n_train` is the sample size behind
    the estimate -- surfaced so the UI can be honest about a thin one.

    `historical_hit_rate` (Section 13's `forecasts` column) is intentionally
    absent: it's the walk-forward backtest's output, filled in a later Phase-7
    sub-part, not something a single point-in-time fit can know about itself.
    """

    model_name: str
    horizon_days: int
    as_of: pd.Timestamp
    last_close: float
    point_return: float
    lower_return: float
    upper_return: float
    point_price: float
    lower_price: float
    upper_price: float
    confidence_level: float
    n_train: int


def _forecast_from_log(
    *,
    model_name: str,
    horizon_days: int,
    as_of: pd.Timestamp,
    last_close: float,
    mean_log: float,
    lower_log: float,
    upper_log: float,
    confidence_level: float,
    n_train: int,
) -> Forecast:
    """Build a `Forecast` from a horizon log-return point + band.

    Converts the log-return triple to simple returns (`expm1`) and to prices
    (`last_close * exp(log_return)`), keeping the two representations exactly
    consistent (`point_price == last_close * (1 + point_return)`).
    """
    return Forecast(
        model_name=model_name,
        horizon_days=horizon_days,
        as_of=as_of,
        last_close=last_close,
        point_return=float(np.expm1(mean_log)),
        lower_return=float(np.expm1(lower_log)),
        upper_return=float(np.expm1(upper_log)),
        point_price=float(last_close * np.exp(mean_log)),
        lower_price=float(last_close * np.exp(lower_log)),
        upper_price=float(last_close * np.exp(upper_log)),
        confidence_level=confidence_level,
        n_train=n_train,
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _validate_horizon(horizon_days: int) -> None:
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")


def _close_series(prices: pd.DataFrame) -> pd.Series:
    """The clean, date-sorted, strictly-positive close series `prices` implies.

    Sorts by index (point-in-time reads assume ascending dates), coerces to
    numeric, drops NaN, and filters non-positive closes (a log price needs
    `close > 0`; a 0/negative close is bad data, not a real quote).
    """
    if "close" not in prices.columns:
        raise ValueError("prices is missing required column: 'close'")
    close = pd.to_numeric(prices["close"], errors="coerce").sort_index().dropna()
    close = close[close > 0]
    return close


def _z_for(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence_level / 2.0))


def _calibrate_random_walk(
    close: pd.Series, lookback: int | None, *, drift: bool = True
) -> tuple[float, float, int] | None:
    """The historical daily log-return drift/volatility a random walk calibrates to.

    Returns `(mu, sigma, n_obs)`. Shared by `baseline_forecast`'s closed-form
    band and the Monte Carlo simulation below -- both evaluate the identical
    random-walk-with-drift model, one analytically, one by simulating paths
    (Section 7.6's GBM "calibrated to historical volatility" is precisely this
    estimator). `mu` is the mean historical daily log return (zeroed when
    `drift=False`, the strictest driftless null); `sigma` is its standard
    deviation; `n_obs` is the number of log-returns behind the estimate,
    surfaced by callers as `Forecast.n_train` / `MonteCarloFanChart.n_train` so
    the UI can be honest about a calibration resting on a short history. `None`
    if `lookback` leaves no returns to estimate from.
    """
    log_returns = np.log(close).diff().dropna()
    if lookback is not None:
        log_returns = log_returns.iloc[-lookback:]
    if log_returns.empty:
        return None

    mu = float(log_returns.mean()) if drift else 0.0
    sigma = float(log_returns.std(ddof=1))
    if not np.isfinite(sigma):
        sigma = 0.0
    return mu, sigma, int(len(log_returns))


# --------------------------------------------------------------------------- #
# 1. Baseline -- random walk / drift (the null hypothesis)
# --------------------------------------------------------------------------- #


def baseline_forecast(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    drift: bool = True,
    confidence_level: float = DEFAULT_CONFIDENCE,
    lookback: int | None = None,
) -> Forecast | None:
    """Naive random-walk / drift forecast -- the null every other model must beat.

    This isn't filler (Section 7.6): a statistical or ML model that can't beat a
    coin-flip drift extrapolation on out-of-sample directional hit-rate is worth
    nothing, and showing that comparison honestly is the credibility signal. The
    forward log-return over the horizon is modelled as a random walk with (by
    default) drift `mu`: horizon mean `mu * h`, standard deviation `sigma *
    sqrt(h)` where `mu`, `sigma` are the per-day log-return mean/std. With
    `drift=False` it's the pure driftless walk (expected return 0) -- the
    strictest null.

    `lookback` optionally restricts the `mu`/`sigma` estimate to the most recent
    N returns; `None` uses all available history. Returns `None` below
    `_MIN_BASELINE_BARS` bars (too little to estimate volatility meaningfully).
    """
    _validate_horizon(horizon_days)
    close = _close_series(prices)
    if len(close) < _MIN_BASELINE_BARS:
        return None
    calibration = _calibrate_random_walk(close, lookback, drift=drift)
    if calibration is None:
        return None
    mu, sigma, n_obs = calibration

    h = horizon_days
    mean_log = mu * h
    sd_log = sigma * np.sqrt(h)
    z = _z_for(confidence_level)

    return _forecast_from_log(
        model_name="baseline",
        horizon_days=h,
        as_of=close.index[-1],
        last_close=float(close.iloc[-1]),
        mean_log=mean_log,
        lower_log=mean_log - z * sd_log,
        upper_log=mean_log + z * sd_log,
        confidence_level=confidence_level,
        n_train=n_obs,
    )


# --------------------------------------------------------------------------- #
# 2. Statistical -- ARIMA / SARIMA (trend + seasonality)
# --------------------------------------------------------------------------- #


def statistical_forecast(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    trend: str = _ARIMA_TREND,
) -> Forecast | None:
    """ARIMA/SARIMA forecast on the log-price series (Section 7.6).

    Fits on *log price* with `d=1` differencing (so the model works on returns,
    which are stationary -- fitting a raw non-stationary price level is a
    textbook mistake). The `h`-step-ahead predicted log price and its
    `confidence_level` prediction interval come straight from statsmodels; we
    subtract the last observed log price to express the result as a horizon
    log-return, matching the other models.

    `order` is a robust ARIMA(1,1,1) default; pass a `seasonal_order`
    (e.g. `(1, 0, 1, 5)` for a weekly cycle) to make it SARIMA. Returns `None`
    below `_MIN_ARIMA_BARS` bars, or if the fit fails to converge -- a graceful
    degradation to "no statistical forecast," never a crash (many real series
    won't fit a given order).

    **`trend` must include a drift term, and that is not a detail.** statsmodels
    defaults to `trend="n"` whenever `d > 0`, and with no drift the h-step
    forecast of a differenced series decays to a constant within a few steps.
    This model ran that way against real history and the result was a forecast
    that was *identical at 5, 20, 63 and 252 days* for every symbol, 83% of them
    under 0.1% -- a model structurally incapable of expressing a view, presented
    to users as one of three competing forecasts with its own track record. With
    `trend="t"` (a time trend on the undifferenced log price, i.e. drift on the
    returns) the forecast scales with the horizon as it should: measured on AMD,
    +0.21% / +0.85% / +2.69% / +10.78%, implying a per-bar drift of 0.000428
    against the series' actual 0.000429. Note `"c"` and `"ct"` are rejected
    outright by statsmodels when `d > 0`, so `"t"` is the drift specification
    here, not a stylistic pick.

    **What this model is worth, measured rather than assumed.** Walked forward
    over 6 names and their last 1,200 bars, drift-ARIMA is statistically
    indistinguishable from `baseline_forecast`: mean |prediction| 0.00249 vs
    0.00224 at h=5, 0.00939 vs 0.00918 at h=20, 0.03290 vs 0.03192 at h=63,
    with hit rates and RMSE matching to the third decimal. That is not a defect
    -- it is what ARIMA(1,1,1) *is* once it has a drift term. The ARMA(1,1) part
    captures a little short-range autocorrelation, and beyond a few steps the
    forecast converges to drift x h, which is precisely the random-walk-with-
    drift null. Neither beat that null on the sample (hit rates 0.44-0.48).

    Two consequences worth stating plainly, because the UI shows three model
    names and a reader will assume three independent opinions:

    * `arima` and `baseline` are *near-duplicates by construction*, not two
      independent votes. Agreement between them is not corroboration.
    * The model is kept anyway -- Section 7.6 asks for a statistical model, and
      the `vs naive` column now makes its (lack of) edge visible instead of
      implied. Showing honestly that a classical time-series model does not beat
      a random walk is a more useful result than hiding the comparison.

    **Deliberately NOT done: searching `order` for something that beats the
    baseline.** That is Section 22's overfitting trap exactly -- trying a dozen
    specifications and keeping whichever scored best on history is fitting
    noise, not discovering signal. The order stays a fixed, documented default.
    """
    _validate_horizon(horizon_days)
    close = _close_series(prices)
    if len(close) < _MIN_ARIMA_BARS:
        return None

    from statsmodels.tsa.arima.model import ARIMA

    log_price = np.log(close.to_numpy(dtype=float))
    try:
        with warnings.catch_warnings():
            # ARIMA is chatty about convergence / lack of a frequency on a plain
            # array; neither affects an h-step-ahead point forecast.
            warnings.simplefilter("ignore")
            fitted = ARIMA(
                log_price,
                order=order,
                seasonal_order=seasonal_order or (0, 0, 0, 0),
                trend=trend,
            ).fit()
            forecast = fitted.get_forecast(steps=horizon_days)
            mean_path = np.asarray(forecast.predicted_mean, dtype=float)
            conf = np.asarray(forecast.conf_int(alpha=1.0 - confidence_level), dtype=float)
    except Exception:  # noqa: BLE001
        # statsmodels raises a wide, undocumented variety of errors (ValueError,
        # LinAlgError, and more) on a series a given order can't fit; any of them
        # means "no statistical forecast," not a crash.
        return None

    last_log_price = float(log_price[-1])
    mean_log = float(mean_path[-1]) - last_log_price
    lower_log = float(conf[-1, 0]) - last_log_price
    upper_log = float(conf[-1, 1]) - last_log_price

    return _forecast_from_log(
        model_name="sarima" if seasonal_order else "arima",
        horizon_days=horizon_days,
        as_of=close.index[-1],
        last_close=float(close.iloc[-1]),
        mean_log=mean_log,
        lower_log=lower_log,
        upper_log=upper_log,
        confidence_level=confidence_level,
        n_train=int(len(log_price)),
    )


# --------------------------------------------------------------------------- #
# 3. ML -- gradient-boosted trees on engineered, horizon-selected features
# --------------------------------------------------------------------------- #


def build_features(
    prices: pd.DataFrame,
    *,
    exog: pd.DataFrame | None = None,
    exog_buckets: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[str, int | None]]]:
    """Trailing, look-ahead-free feature matrix + per-feature (bucket, lookback) tags.

    All price features are computed from `close` alone (so a caller with only a
    close series still gets a usable model) and are strictly trailing: a value at
    date `t` uses only closes up to `t`. Momentum is the trailing log-return over
    each lookback; volatility is the rolling std of daily returns; RSI(14) and
    the MACD histogram are the usual oscillators, plus distance from the 50/200
    moving averages.

    `exog` (optional) is any already-aligned, point-in-time exogenous frame --
    fundamentals, macro, sentiment -- joined on the price index (values carried
    forward is the caller's responsibility, since only they know the as-of dates
    of each source). Each exog column is bucketed via `exog_buckets` (default
    bucket "fundamental"), so `select_features` can bring it in only at the
    horizons where it belongs.

    Returns `(features, meta)` where `meta[col] = (bucket, lookback_or_None)`;
    a `None` lookback means the feature is not lookback-gated (kept whenever its
    bucket is included).
    """
    import pandas_ta_classic as ta

    close = _close_series(prices)
    features = pd.DataFrame(index=close.index)
    meta: dict[str, tuple[str, int | None]] = {}

    log_close = np.log(close)
    daily_return = log_close.diff()

    for lb in _MOMENTUM_LOOKBACKS:
        name = f"mom_{lb}"
        features[name] = log_close - log_close.shift(lb)
        meta[name] = ("technical", lb)

    for lb in _VOL_LOOKBACKS:
        name = f"vol_{lb}"
        features[name] = daily_return.rolling(lb).std()
        meta[name] = ("technical", lb)

    rsi = ta.rsi(close, length=14)
    features["rsi_14"] = (rsi - 50.0) / 50.0 if rsi is not None else np.nan
    meta["rsi_14"] = ("technical", None)

    macd = ta.macd(close)
    features["macd_hist_norm"] = (macd["MACDh_12_26_9"] / close) if macd is not None else np.nan
    meta["macd_hist_norm"] = ("technical", None)

    sma_50 = ta.sma(close, length=50)
    features["dist_sma_50"] = (close / sma_50 - 1.0) if sma_50 is not None else np.nan
    meta["dist_sma_50"] = ("technical", 50)

    sma_200 = ta.sma(close, length=200)
    features["dist_sma_200"] = (close / sma_200 - 1.0) if sma_200 is not None else np.nan
    meta["dist_sma_200"] = ("technical", 200)

    if exog is not None and not exog.empty:
        aligned = exog.reindex(close.index)
        for col in exog.columns:
            bucket = (exog_buckets or {}).get(col, "fundamental")
            if bucket not in _BUCKETS:
                raise ValueError(f"exog column {col!r} has unknown bucket {bucket!r}")
            features[col] = pd.to_numeric(aligned[col], errors="coerce")
            meta[col] = (bucket, None)

    return features, meta


def horizon_feature_weights(horizon_days: int) -> dict[str, float]:
    """Feature-bucket weights for a horizon, interpolated between the anchors.

    Short horizons weight `technical` heavily; long horizons shift toward
    `fundamental` and `macro` (Section 7.6's horizon-dependent emphasis). Below
    the shortest anchor or above the longest, the nearest anchor is used
    (no extrapolation). Weights are renormalized to sum to 1.
    """
    _validate_horizon(horizon_days)
    anchors = sorted(_HORIZON_BUCKET_ANCHORS)
    if horizon_days <= anchors[0]:
        weights = dict(_HORIZON_BUCKET_ANCHORS[anchors[0]])
    elif horizon_days >= anchors[-1]:
        weights = dict(_HORIZON_BUCKET_ANCHORS[anchors[-1]])
    else:
        hi = next(a for a in anchors if a >= horizon_days)
        lo = max(a for a in anchors if a <= horizon_days)
        if lo == hi:
            weights = dict(_HORIZON_BUCKET_ANCHORS[lo])
        else:
            frac = (horizon_days - lo) / (hi - lo)
            low_w, high_w = _HORIZON_BUCKET_ANCHORS[lo], _HORIZON_BUCKET_ANCHORS[hi]
            weights = {b: low_w[b] + frac * (high_w[b] - low_w[b]) for b in _BUCKETS}

    total = sum(weights.values())
    return {b: weights[b] / total for b in _BUCKETS}


def _lookback_relevant(lookback: int, horizon_days: int) -> bool:
    """Is a lookback-gated feature relevant to this horizon?

    Keep lookbacks up to `_LOOKBACK_HORIZON_MULT` x horizon, with a floor so the
    shortest windows are always available. A 252-day momentum feature is dropped
    from a 5-day forecast (it would swamp the short-term signal), but kept for a
    quarterly/annual one.
    """
    return lookback <= max(_LOOKBACK_FLOOR, _LOOKBACK_HORIZON_MULT * horizon_days)


def select_features(meta: dict[str, tuple[str, int | None]], horizon_days: int) -> list[str]:
    """The subset of features whose bucket is included at, and lookback suits, this horizon.

    A bucket is in only when `horizon_feature_weights` gives it at least
    `_BUCKET_INCLUDE_MIN`; a lookback-gated feature additionally survives only if
    `_lookback_relevant`. This is where "different horizon -> different feature
    set" (Section 7.6) actually takes effect -- and because trees are invariant
    to monotonic rescaling, inclusion is the only way graded emphasis can change
    the model at all.
    """
    _validate_horizon(horizon_days)
    weights = horizon_feature_weights(horizon_days)
    included_buckets = {b for b in _BUCKETS if weights[b] >= _BUCKET_INCLUDE_MIN}

    selected: list[str] = []
    for name, (bucket, lookback) in meta.items():
        if bucket not in included_buckets:
            continue
        if lookback is not None and not _lookback_relevant(lookback, horizon_days):
            continue
        selected.append(name)
    return selected


def _calibrated_residual_quantiles(
    n_residuals: int, *, horizon_days: int, confidence_level: float
) -> tuple[float, float]:
    """Quantile levels for the residual band, corrected for overlap and small samples.

    Taking the plain empirical `(1±C)/2` quantiles of the holdout residuals
    produces a band that is **narrower than it claims to be**. Measured on real
    history (four names, 236 non-overlapping folds, h=20), the nominal 90% band
    contained the realized return only 78.8% of the time -- reality landed
    outside it roughly twice as often as advertised.

    Two compounding reasons, both finite-sample rather than modelling errors:

    * **The residuals overlap.** Consecutive horizon returns share `h - 1` days,
      so `n` residuals carry only about `n / h` independent observations. The
      raw count flatters the quantile estimate enormously: 200 residuals at
      h=20 are worth roughly 10.
    * **Extreme order statistics of a small sample are biased inward.** The
      empirical 5th percentile of `k` draws is, in expectation, less extreme
      than the true 5th percentile -- precisely the tail the band is made of.

    The fix is the standard split-conformal finite-sample correction, applied at
    the *effective* sample size rather than the raw count: use the
    `(n_eff + 1) / n_eff` inflated tail level. When even that cannot be
    supported -- fewer independent residuals than a two-sided band at this
    confidence needs -- the levels collapse to the observed min/max residual,
    the widest honest statement the sample can make, rather than interpolating
    a tighter one that the data does not support.

    Returns `(lower_level, upper_level)` for `np.quantile`, symmetric about 0.5.
    """
    tail = (1.0 - confidence_level) / 2.0
    n_eff = max(1, int(n_residuals // max(1, horizon_days)))
    inflated = (1.0 - tail) * (n_eff + 1) / n_eff
    upper = min(1.0, inflated)
    return 1.0 - upper, upper


def ml_forecast(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    exog: pd.DataFrame | None = None,
    exog_buckets: dict[str, str] | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE,
    random_state: int = 0,
) -> Forecast | None:
    """Gradient-boosted-tree forecast of the horizon return (Section 7.6).

    Predicts the forward horizon *log-return* from horizon-selected trailing
    features with a `HistGradientBoostingRegressor` -- the plan's named choice:
    at a solo project's data volume (hundreds of tickers, years of history),
    regularized tree ensembles on tabular features match or beat deep sequence
    models, train in seconds, handle NaNs natively, and stay interpretable.

    Leakage discipline (the whole point of Opus/High here):

    * The target `y_t = log_close(t + h) - log_close(t)` is built with
      `shift(-h)`, so the final `h` rows have no known target and are excluded
      from training. The single row we predict is the as-of date itself.
    * The prediction interval is estimated out-of-sample on a **chronological**
      tail (the last `_ML_VAL_FRACTION` of training rows) -- never a random
      split. sklearn's own early-stopping validation split *shuffles*, which
      would leak time order, so we disable it (`early_stopping=False`) and hold
      out the tail ourselves. Model hyperparameters are fixed and modest, not
      tuned to any backtest (Section 22).
    * That chronological split is additionally **purged**: because row `t`'s
      label is built from `close(t + h)`, the last `h` rows before the split
      would otherwise be trained on outcomes drawn from inside the holdout
      window. Those rows are dropped, so the residuals really are out-of-sample
      and the band isn't quietly too narrow. See the inline note at the split.
    * The band is non-parametric: empirical quantiles of the holdout residuals
      (`actual - predicted`) added to the point forecast, so it makes no
      normality assumption and can be asymmetric.

    Caveat, documented rather than hidden: consecutive horizon returns overlap by
    `h - 1` days, so both the training labels and the holdout residuals are
    autocorrelated. That doesn't bias the point forecast, but it means the
    effective sample is smaller than `n_train` and the band is an approximation;
    honest confidence intervals on aggregate track-record metrics are the
    backtest's bootstrap sub-part, which must block-resample to respect it.

    Returns `None` when there isn't enough clean training history
    (`_MIN_ML_TRAIN_ROWS`) or the as-of row's features are unusable.
    """
    _validate_horizon(horizon_days)
    close = _close_series(prices)
    features, meta = build_features(prices, exog=exog, exog_buckets=exog_buckets)
    selected = select_features(meta, horizon_days)
    if not selected:
        return None

    log_close = np.log(close)
    target = log_close.shift(-horizon_days) - log_close  # forward horizon log-return
    target = target.reindex(features.index)

    x_all = features[selected]
    as_of = features.index[-1]
    x_predict = x_all.loc[[as_of]]
    if x_predict.isna().all(axis=1).iloc[0]:
        return None  # the row we'd forecast has no usable feature at all

    # Training rows: a known (non-NaN) target and at least one usable feature.
    # HistGBR handles residual NaNs in features natively, so warm-up rows with
    # only some features missing are still fair game.
    trainable = target.notna() & ~x_all.isna().all(axis=1)
    x_train_full = x_all[trainable]
    y_train_full = target[trainable]
    if len(x_train_full) < _MIN_ML_TRAIN_ROWS:
        return None

    # Chronological holdout for the residual band (rows are already date-sorted).
    #
    # **Purging.** A chronological split alone is NOT enough here, because the
    # label of row `i` is built from `close[i + h]`. Every core row within `h`
    # bars of the validation boundary therefore has a target computed from
    # prices that lie *inside* the holdout window -- the core model is trained
    # on the very outcomes it is about to be graded on. Untreated, that shrinks
    # the residuals and hands out a prediction interval that is too narrow, i.e.
    # a forecast that looks more certain than it is (Section 22's "presenting
    # every number with the same confidence"). At h=252 with the default 20%
    # holdout it is total: every bar of the validation window is reachable from
    # some core label, so the "out-of-sample" band is not out of sample at all.
    #
    # The standard remedy (a purge/embargo) is to drop the last `h` core rows,
    # which is exactly the set whose labels reach the boundary. If too little
    # clean core history survives, we decline the empirical band entirely rather
    # than compute a leaky one -- the caller already has an honest fallback
    # (the random-walk volatility band below).
    n_val = int(len(x_train_full) * _ML_VAL_FRACTION)
    n_core = len(x_train_full) - n_val - horizon_days
    use_holdout = n_val >= _MIN_ML_VAL_ROWS and n_core >= _MIN_ML_CORE_ROWS

    # Prune features without enough coverage to be binned/learned. Coverage is
    # judged on the *core* fold when we hold one out -- the earliest, shortest
    # window, where a long-lookback feature may still be entirely NaN -- so the
    # holdout and final models share exactly one feature set.
    coverage_ref = x_train_full.iloc[:n_core] if use_holdout else x_train_full
    usable_cols = [
        c for c in x_train_full.columns if coverage_ref[c].notna().sum() >= _MIN_FEATURE_NONNULL
    ]
    if not usable_cols or x_predict[usable_cols].isna().all(axis=1).iloc[0]:
        return None
    x_train_full = x_train_full[usable_cols]
    x_predict = x_predict[usable_cols]

    from sklearn.ensemble import HistGradientBoostingRegressor

    def _new_model() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=_ML_MAX_ITER,
            learning_rate=_ML_LEARNING_RATE,
            max_depth=_ML_MAX_DEPTH,
            min_samples_leaf=_ML_MIN_LEAF,
            l2_regularization=_ML_L2,
            early_stopping=False,  # see docstring: sklearn's split shuffles
            random_state=random_state,
        )

    residuals: np.ndarray | None = None
    if use_holdout:
        # `n_core` already excludes the purged `horizon_days` rows between the
        # two folds, so no core label reads a price from the holdout window.
        x_core, y_core = x_train_full.iloc[:n_core], y_train_full.iloc[:n_core]
        x_val, y_val = x_train_full.iloc[-n_val:], y_train_full.iloc[-n_val:]
        holdout_model = _new_model().fit(x_core, y_core)
        residuals = y_val.to_numpy(dtype=float) - holdout_model.predict(x_val)

    # Final point model is refit on ALL training rows; the holdout only informed
    # the error distribution, not the point estimate (standard practice).
    point_model = _new_model().fit(x_train_full, y_train_full)
    mean_log = float(point_model.predict(x_predict)[0])

    if residuals is not None and residuals.size:
        lower_q, upper_q = _calibrated_residual_quantiles(
            residuals.size, horizon_days=horizon_days, confidence_level=confidence_level
        )
        lower_log = mean_log + float(np.quantile(residuals, lower_q))
        upper_log = mean_log + float(np.quantile(residuals, upper_q))
    else:
        # Too little holdout for an empirical band: fall back to a random-walk
        # volatility band around the ML point, so the forecast is never handed
        # out with a fake-tight interval.
        sigma = float(np.log(close).diff().dropna().std(ddof=1))
        z = _z_for(confidence_level)
        half = (z * sigma * np.sqrt(horizon_days)) if np.isfinite(sigma) else 0.0
        lower_log, upper_log = mean_log - half, mean_log + half

    return _forecast_from_log(
        model_name="gbr",
        horizon_days=horizon_days,
        as_of=as_of,
        last_close=float(close.iloc[-1]),
        mean_log=mean_log,
        lower_log=lower_log,
        upper_log=upper_log,
        confidence_level=confidence_level,
        n_train=int(len(x_train_full)),
    )


# --------------------------------------------------------------------------- #
# 4. Monte Carlo -- GBM-simulated fan chart (Section 7.6)
# --------------------------------------------------------------------------- #


def simulate_gbm_paths(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    n_paths: int = _DEFAULT_MC_PATHS,
    lookback: int | None = None,
    random_state: int | None = 0,
) -> np.ndarray | None:
    """Simulate `n_paths` GBM price paths, calibrated to historical volatility (Section 7.6).

    Each path is a random walk in log-price: day-to-day log returns are drawn
    iid from Normal(mu, sigma) -- the same historical mean/std
    `baseline_forecast` calibrates to (`_calibrate_random_walk`), so this is the
    identical random-walk-with-drift model, executed by simulation instead of
    closed form (Section 7.6's "GBM calibrated to historical volatility").

    Returns an `(n_paths, horizon_days + 1)` array: column 0 is the
    deterministic last close (day 0, identical across every path), column `t`
    is the simulated price `t` trading days ahead. `None` when there isn't
    enough history to calibrate (`_MIN_BASELINE_BARS` bars), matching the other
    models' abstention discipline.

    `random_state` defaults to a fixed seed so repeated calls against the same
    data are reproducible (matching `ml_forecast`'s convention); pass `None` for
    a fresh draw each call.
    """
    _validate_horizon(horizon_days)
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    close = _close_series(prices)
    if len(close) < _MIN_BASELINE_BARS:
        return None
    calibration = _calibrate_random_walk(close, lookback)
    if calibration is None:
        return None
    mu, sigma, _ = calibration

    rng = np.random.default_rng(random_state)
    daily_log_returns = rng.normal(mu, sigma, size=(n_paths, horizon_days))
    log_paths = np.cumsum(daily_log_returns, axis=1)
    last_close = float(close.iloc[-1])
    future_prices = last_close * np.exp(log_paths)
    day_zero = np.full((n_paths, 1), last_close)
    return np.concatenate([day_zero, future_prices], axis=1)


@dataclass(frozen=True)
class MonteCarloFanChart:
    """Day-by-day percentile price paths from a GBM Monte Carlo simulation (Section 7.6).

    The full "fan chart" Section 7.6 calls "the most honest way to visually
    communicate 'possible price forecast'": unlike a `Forecast` (one point +
    band at a single horizon), this reports a percentile price at *every*
    trading day from 1 to `horizon_days`, so the visualized uncertainty widens
    the way a real random walk's actually does (proportional to sqrt(time))
    instead of jumping straight to a fixed final-day range.

    `days` (1..horizon_days) are the trading-day offsets the `percentiles`
    arrays are aligned to; `percentiles` maps each requested percentile (e.g.
    `50.0`) to its price path. `mu`/`sigma` are the calibrated daily log-return
    drift/volatility; `n_train` is the log-return sample size behind that
    calibration -- the same honesty fields `Forecast` carries.
    """

    as_of: pd.Timestamp
    last_close: float
    horizon_days: int
    n_paths: int
    mu: float
    sigma: float
    n_train: int
    days: np.ndarray
    percentiles: dict[float, np.ndarray]


def monte_carlo_fan_chart(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    n_paths: int = _DEFAULT_MC_PATHS,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    lookback: int | None = None,
    random_state: int | None = 0,
) -> MonteCarloFanChart | None:
    """Simulate a GBM fan chart: percentile price paths for every day out to the horizon.

    Simulates `n_paths` independent price paths (`simulate_gbm_paths`) and
    reports the requested `percentiles` (default 5th/50th/95th, matching
    Section 7.6's spec exactly) of the simulated price *at each day*
    independently -- the standard fan-chart construction (a cross-sectional
    percentile across the ensemble at each time step, not any single simulated
    trajectory -- so "the 50th-percentile path" is not one actual draw, it's the
    running median across all of them).

    `None` when there isn't enough history to calibrate (`_MIN_BASELINE_BARS`
    bars).
    """
    _validate_horizon(horizon_days)
    if not percentiles:
        raise ValueError("percentiles must be non-empty")
    for p in percentiles:
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"each percentile must be in [0, 100], got {p}")

    close = _close_series(prices)
    if len(close) < _MIN_BASELINE_BARS:
        return None
    calibration = _calibrate_random_walk(close, lookback)
    if calibration is None:
        return None
    mu, sigma, n_obs = calibration

    paths = simulate_gbm_paths(
        prices, horizon_days, n_paths=n_paths, lookback=lookback, random_state=random_state
    )
    if paths is None:
        return None

    future = paths[:, 1:]  # exclude the deterministic day-0 column
    day_percentiles = {p: np.percentile(future, p, axis=0) for p in percentiles}

    return MonteCarloFanChart(
        as_of=close.index[-1],
        last_close=float(close.iloc[-1]),
        horizon_days=horizon_days,
        n_paths=paths.shape[0],
        mu=mu,
        sigma=sigma,
        n_train=n_obs,
        days=np.arange(1, horizon_days + 1),
        percentiles=day_percentiles,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

_MODEL_RUNNERS = ("baseline", "arima", "ml")


def forecast_horizon(
    prices: pd.DataFrame,
    horizon_days: int,
    *,
    exog: pd.DataFrame | None = None,
    exog_buckets: dict[str, str] | None = None,
    models: tuple[str, ...] = _MODEL_RUNNERS,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> list[Forecast]:
    """Run the requested models for a single horizon, dropping any that abstain.

    A model returning `None` (too little history, a non-converging ARIMA fit)
    simply doesn't appear in the result -- the same coverage discipline the
    scoring modules use, so a thinly-covered name still gets whatever forecasts
    it can support rather than nothing.
    """
    unknown = set(models) - set(_MODEL_RUNNERS)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}; valid: {_MODEL_RUNNERS}")

    out: list[Forecast] = []
    if "baseline" in models:
        result = baseline_forecast(prices, horizon_days, confidence_level=confidence_level)
        if result is not None:
            out.append(result)
    if "arima" in models:
        result = statistical_forecast(prices, horizon_days, confidence_level=confidence_level)
        if result is not None:
            out.append(result)
    if "ml" in models:
        result = ml_forecast(
            prices,
            horizon_days,
            exog=exog,
            exog_buckets=exog_buckets,
            confidence_level=confidence_level,
        )
        if result is not None:
            out.append(result)
    return out


def generate_forecasts(
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    exog: pd.DataFrame | None = None,
    exog_buckets: dict[str, str] | None = None,
    models: tuple[str, ...] = _MODEL_RUNNERS,
    confidence_level: float = DEFAULT_CONFIDENCE,
) -> list[Forecast]:
    """Every requested (model, horizon) forecast for one ticker's price history.

    The flat list is exactly `forecasts`-table shaped (Section 13): one row per
    model per horizon, each carrying its own point/band. Persisting it -- and
    filling each row's `historical_hit_rate` -- happens in the backtest
    sub-part, not here.
    """
    out: list[Forecast] = []
    for horizon in horizons:
        out.extend(
            forecast_horizon(
                prices,
                horizon,
                exog=exog,
                exog_buckets=exog_buckets,
                models=models,
                confidence_level=confidence_level,
            )
        )
    return out
