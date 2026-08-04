import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import forecasting as fc
from quantpulse.analysis.forecasting import Forecast


def _prices(closes: list[float] | np.ndarray, start: str = "2021-01-01") -> pd.DataFrame:
    """OHLCV frame on a business-day index from a close path."""
    idx = pd.date_range(start, periods=len(closes), freq="B")
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1_000_000.0},
        index=idx,
    )


def _random_walk(n: int, mu: float = 0.0005, sigma: float = 0.015, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))


# --------------------------------------------------------------------------- #
# Forecast result invariants
# --------------------------------------------------------------------------- #


class TestForecastConsistency:
    def test_price_and_return_representations_agree(self) -> None:
        r = fc.baseline_forecast(_prices(_random_walk(300)), 20)
        assert r is not None
        assert r.point_price == pytest.approx(r.last_close * (1 + r.point_return))
        assert r.lower_price == pytest.approx(r.last_close * (1 + r.lower_return))
        assert r.upper_price == pytest.approx(r.last_close * (1 + r.upper_return))

    def test_band_is_ordered(self) -> None:
        for model in (fc.baseline_forecast, fc.statistical_forecast, fc.ml_forecast):
            r = model(_prices(_random_walk(400, seed=3)), 20)
            assert r is not None
            assert r.lower_return <= r.point_return <= r.upper_return
            assert r.lower_price <= r.point_price <= r.upper_price


# --------------------------------------------------------------------------- #
# Baseline (random walk / drift)
# --------------------------------------------------------------------------- #


class TestBaseline:
    def test_drift_extrapolates_the_mean_return(self) -> None:
        # A steady uptrend has positive mean log-return -> positive drift point.
        up = fc.baseline_forecast(_prices(np.linspace(100, 200, 300)), 20)
        assert up is not None and up.point_return > 0
        down = fc.baseline_forecast(_prices(np.linspace(200, 100, 300)), 20)
        assert down is not None and down.point_return < 0

    def test_no_drift_is_the_zero_return_null(self) -> None:
        r = fc.baseline_forecast(_prices(np.linspace(100, 200, 300)), 20, drift=False)
        assert r is not None
        assert r.point_return == pytest.approx(0.0)  # exp(0) - 1
        assert r.point_price == pytest.approx(r.last_close)

    def test_band_widens_with_horizon_like_sqrt_h(self) -> None:
        prices = _prices(_random_walk(400, seed=1))
        wide = fc.baseline_forecast(prices, 80, drift=False)
        narrow = fc.baseline_forecast(prices, 20, drift=False)
        assert wide is not None and narrow is not None
        # driftless band half-width is z*sigma*sqrt(h); 4x horizon -> ~2x width.
        narrow_half = np.log1p(narrow.upper_return)
        wide_half = np.log1p(wide.upper_return)
        assert wide_half == pytest.approx(2.0 * narrow_half, rel=1e-6)

    def test_lookback_restricts_the_estimate(self) -> None:
        # Flat for a long time, then a sharp recent uptrend: a short lookback
        # sees only the ramp and drifts up more than the full-history estimate.
        path = np.concatenate([np.full(300, 100.0), np.linspace(100, 140, 60)])
        full = fc.baseline_forecast(_prices(path), 20)
        recent = fc.baseline_forecast(_prices(path), 20, lookback=40)
        assert full is not None and recent is not None
        assert recent.point_return > full.point_return

    def test_too_short_is_none(self) -> None:
        assert fc.baseline_forecast(_prices(_random_walk(10)), 20) is None

    def test_n_train_reported(self) -> None:
        r = fc.baseline_forecast(_prices(_random_walk(300)), 20)
        assert r is not None and r.n_train == 299  # 300 closes -> 299 returns


# --------------------------------------------------------------------------- #
# Statistical (ARIMA / SARIMA)
# --------------------------------------------------------------------------- #


class TestStatistical:
    def test_produces_forecast_on_adequate_history(self) -> None:
        r = fc.statistical_forecast(_prices(_random_walk(400, seed=7)), 20)
        assert isinstance(r, Forecast)
        assert r.model_name == "arima"

    def test_seasonal_order_names_it_sarima(self) -> None:
        r = fc.statistical_forecast(
            _prices(_random_walk(400, seed=2)), 10, seasonal_order=(1, 0, 1, 5)
        )
        assert r is not None and r.model_name == "sarima"

    def test_too_short_is_none(self) -> None:
        assert fc.statistical_forecast(_prices(_random_walk(40)), 20) is None

    def test_degenerate_series_degrades_gracefully_not_crash(self) -> None:
        # A perfectly constant price is unfittable for most orders; must return
        # None (or a Forecast), never raise.
        out = fc.statistical_forecast(_prices([100.0] * 200), 20)
        assert out is None or isinstance(out, Forecast)


# --------------------------------------------------------------------------- #
# Feature engineering + horizon-dependent selection
# --------------------------------------------------------------------------- #


class TestFeatures:
    def test_features_are_trailing_only_no_look_ahead(self) -> None:
        # A feature row for date t must not change when future rows are appended
        # -- the core point-in-time guarantee the ML target relies on.
        full = _prices(_random_walk(400, seed=5))
        truncated = full.iloc[:300]
        feat_full, _ = fc.build_features(full)
        feat_trunc, _ = fc.build_features(truncated)
        common = feat_trunc.index
        pd.testing.assert_frame_equal(
            feat_full.loc[common], feat_trunc.loc[common], check_exact=False
        )

    def test_exog_columns_are_bucketed(self) -> None:
        prices = _prices(_random_walk(300))
        exog = pd.DataFrame({"pe": 20.0, "regime": 55.0}, index=prices.index)
        _, meta = fc.build_features(
            prices, exog=exog, exog_buckets={"pe": "fundamental", "regime": "macro"}
        )
        assert meta["pe"] == ("fundamental", None)
        assert meta["regime"] == ("macro", None)

    def test_exog_defaults_to_fundamental_bucket(self) -> None:
        prices = _prices(_random_walk(300))
        exog = pd.DataFrame({"book_value": 5.0}, index=prices.index)
        _, meta = fc.build_features(prices, exog=exog)
        assert meta["book_value"] == ("fundamental", None)

    def test_unknown_exog_bucket_raises(self) -> None:
        prices = _prices(_random_walk(300))
        exog = pd.DataFrame({"x": 1.0}, index=prices.index)
        with pytest.raises(ValueError, match="unknown bucket"):
            fc.build_features(prices, exog=exog, exog_buckets={"x": "nonsense"})


class TestHorizonWeights:
    def test_weights_sum_to_one(self) -> None:
        for h in (1, 5, 20, 63, 252, 500):
            assert sum(fc.horizon_feature_weights(h).values()) == pytest.approx(1.0)

    def test_short_horizon_is_technical_heavy_long_is_fundamental_heavy(self) -> None:
        short = fc.horizon_feature_weights(5)
        long = fc.horizon_feature_weights(252)
        assert short["technical"] > short["fundamental"]
        assert long["fundamental"] > long["technical"]

    def test_interpolates_between_anchors(self) -> None:
        # Halfway (in index terms) between the 5 and 20 anchors, technical weight
        # lands between the two anchor values.
        w = fc.horizon_feature_weights(12)
        assert 0.55 < w["technical"] < 0.80

    def test_clamps_outside_anchor_range(self) -> None:
        assert fc.horizon_feature_weights(1) == fc.horizon_feature_weights(5)
        assert fc.horizon_feature_weights(9999) == fc.horizon_feature_weights(252)


class TestSelectFeatures:
    def _meta(self) -> dict[str, tuple[str, int | None]]:
        prices = _prices(_random_walk(300))
        exog = pd.DataFrame({"pe": 20.0, "regime": 55.0}, index=prices.index)
        _, meta = fc.build_features(
            prices, exog=exog, exog_buckets={"pe": "fundamental", "regime": "macro"}
        )
        return meta

    def test_short_horizon_drops_long_lookbacks_and_fundamentals(self) -> None:
        selected = set(fc.select_features(self._meta(), 5))
        assert "mom_5" in selected and "mom_10" in selected and "mom_21" in selected
        assert "mom_252" not in selected  # 252 >> 4*5, dropped
        assert "mom_63" not in selected
        assert "pe" not in selected  # fundamental bucket below include-min at h=5
        assert "regime" not in selected  # macro below include-min at h=5

    def test_long_horizon_includes_long_lookbacks_and_fundamentals(self) -> None:
        selected = set(fc.select_features(self._meta(), 252))
        assert "mom_252" in selected
        assert "pe" in selected  # fundamentals enter at long horizons
        assert "regime" in selected  # macro too

    def test_feature_set_actually_differs_by_horizon(self) -> None:
        meta = self._meta()
        assert set(fc.select_features(meta, 5)) != set(fc.select_features(meta, 252))


# --------------------------------------------------------------------------- #
# ML (gradient-boosted trees)
# --------------------------------------------------------------------------- #


class TestML:
    def test_follows_a_persistent_uptrend(self) -> None:
        # A deterministic ramp has consistently positive forward returns; the
        # tree should forecast a positive move.
        r = fc.ml_forecast(_prices(np.linspace(100, 300, 500)), 20)
        assert r is not None and r.model_name == "gbr"
        assert r.point_return > 0

    def test_follows_a_persistent_downtrend(self) -> None:
        r = fc.ml_forecast(_prices(np.linspace(300, 100, 500)), 20)
        assert r is not None and r.point_return < 0

    def test_too_little_history_is_none(self) -> None:
        assert fc.ml_forecast(_prices(_random_walk(120)), 20) is None

    def test_predict_row_is_never_trained_on(self) -> None:
        # With N closes and horizon h, the last h rows have no known target and
        # are excluded from training; n_train must not exceed N - h.
        n, h = 400, 20
        r = fc.ml_forecast(_prices(_random_walk(n, seed=9)), h)
        assert r is not None
        assert r.n_train <= n - h
        assert r.as_of == _prices(_random_walk(n, seed=9)).index[-1]

    def test_exog_forecast_runs_at_long_horizon(self) -> None:
        # A one-year horizon now needs ~three years of history (see
        # `_MIN_HISTORY_HORIZON_MULT`), so this is sized to clear that gate --
        # the point of the test is the exogenous-feature path at a long horizon,
        # not the gate.
        n = fc.min_bars_for_horizon(252, floor=fc._MIN_ML_TRAIN_ROWS) + 50
        prices = _prices(_random_walk(n, seed=4))
        # Exogenous fundamental signal available every day.
        exog = pd.DataFrame({"value_score": np.linspace(0, 1, n)}, index=prices.index)
        r = fc.ml_forecast(prices, 252, exog=exog, exog_buckets={"value_score": "fundamental"})
        assert isinstance(r, Forecast)

    def test_a_horizon_longer_than_the_history_supports_is_declined(self) -> None:
        # The demo database really held 26 daily bars and was days from
        # publishing a one-year forecast off 25 returns (+157% for Apple, with a
        # nominal 90% band that never touched zero). Every model must abstain.
        short = _prices(_random_walk(26, seed=5))
        assert fc.baseline_forecast(short, 252) is None
        assert fc.statistical_forecast(short, 252) is None
        assert fc.ml_forecast(short, 252) is None
        assert fc.monte_carlo_fan_chart(short, 252) is None
        assert fc.simulate_gbm_paths(short, 252) is None
        # ...while the short horizon its history *does* support still works.
        assert fc.baseline_forecast(short, 5) is not None
        assert fc.generate_forecasts(short) == fc.forecast_horizon(short, 5)

    def test_deterministic(self) -> None:
        prices = _prices(_random_walk(400, seed=11))
        a = fc.ml_forecast(prices, 20)
        b = fc.ml_forecast(prices, 20)
        assert a is not None and b is not None
        assert a.point_return == b.point_return


# --------------------------------------------------------------------------- #
# Residual-band holdout purging
#
# Row `t`'s label is `close(t + h)`, so a plain chronological split still lets
# the last `h` core rows be trained on outcomes drawn from inside the holdout
# window. That makes the residuals -- and therefore the published prediction
# interval -- narrower than the model has earned. These tests pin the purge.
# --------------------------------------------------------------------------- #


class TestMLHoldoutPurge:
    @staticmethod
    def _spy_on_fits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Record the training-row count of every model fit, in order."""
        import sklearn.ensemble as skl

        real = skl.HistGradientBoostingRegressor
        seen: list[int] = []

        class Spy(real):  # type: ignore[misc,valid-type]
            def fit(self, x, y, **kwargs):  # type: ignore[no-untyped-def]
                seen.append(len(x))
                return super().fit(x, y, **kwargs)

        monkeypatch.setattr(skl, "HistGradientBoostingRegressor", Spy)
        return seen

    def test_core_fold_drops_exactly_the_horizon_rows_before_the_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._spy_on_fits(monkeypatch)
        n, h = 500, 63
        result = fc.ml_forecast(_prices(_random_walk(n, seed=3)), h)
        assert result is not None
        # Two fits: the holdout (band) model, then the final point model on all
        # training rows. The second is the honest count of trainable rows.
        assert len(seen) == 2
        n_core, n_train = seen
        n_val = int(n_train * fc._ML_VAL_FRACTION)
        assert n_core == n_train - n_val - h, (
            "core fold must exclude the holdout AND the h rows whose labels reach into it"
        )

    def test_long_horizon_declines_the_band_rather_than_leaking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Just enough history to clear the horizon gate at h=63, which still
        # leaves no clean core after purging -- so the empirical band must be
        # declined outright (only the point model is fit) and the honest
        # random-walk fallback (symmetric in log space) is used instead.
        seen = self._spy_on_fits(monkeypatch)
        bars = fc.min_bars_for_horizon(63, floor=fc._MIN_ML_TRAIN_ROWS)
        result = fc.ml_forecast(_prices(_random_walk(bars, seed=4)), 63)
        assert result is not None
        assert len(seen) == 1, "a leaky holdout model must not be fit at all"
        assert result.lower_price < result.point_price < result.upper_price
        assert result.point_price**2 == pytest.approx(
            result.lower_price * result.upper_price, rel=1e-9
        )

    def test_residuals_come_only_from_rows_the_core_model_never_saw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The property the purge exists to guarantee, stated directly: the last
        # bar any core label reads (core_end - 1 + h) must fall strictly before
        # the first holdout row. Without the purge this is violated for every
        # horizon >= 1.
        seen = self._spy_on_fits(monkeypatch)
        n, h = 500, 63
        assert fc.ml_forecast(_prices(_random_walk(n, seed=7)), h) is not None
        n_core, n_train = seen
        first_holdout_row = n_train - int(n_train * fc._ML_VAL_FRACTION)
        last_bar_a_core_label_reads = (n_core - 1) + h
        assert last_bar_a_core_label_reads < first_holdout_row


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


class TestOrchestration:
    def test_forecast_horizon_runs_all_models(self) -> None:
        results = fc.forecast_horizon(_prices(_random_walk(400, seed=6)), 20)
        assert {r.model_name for r in results} == {"baseline", "arima", "gbr"}

    def test_models_that_abstain_are_dropped(self) -> None:
        # ~130 closes: baseline is fine, ML lacks the 120 training rows it needs
        # after dropping the horizon tail -> only the models that can run appear.
        results = fc.forecast_horizon(_prices(_random_walk(130, seed=8)), 20)
        names = {r.model_name for r in results}
        assert "baseline" in names
        assert "gbr" not in names

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            fc.forecast_horizon(_prices(_random_walk(300)), 20, models=("baseline", "lstm"))

    def test_generate_forecasts_is_model_by_horizon_cross_product(self) -> None:
        prices = _prices(_random_walk(600, seed=12))
        results = fc.generate_forecasts(prices, horizons=(5, 20, 63))
        keys = {(r.model_name, r.horizon_days) for r in results}
        assert keys == {(m, h) for m in ("baseline", "arima", "gbr") for h in (5, 20, 63)}

    def test_can_select_a_subset_of_models(self) -> None:
        results = fc.generate_forecasts(
            _prices(_random_walk(400)), horizons=(20,), models=("baseline",)
        )
        assert [r.model_name for r in results] == ["baseline"]


# --------------------------------------------------------------------------- #
# Monte Carlo -- GBM simulated fan chart
# --------------------------------------------------------------------------- #


class TestSimulateGbmPaths:
    def test_shape_includes_day_zero(self) -> None:
        paths = fc.simulate_gbm_paths(_prices(_random_walk(300)), 20, n_paths=500)
        assert paths is not None
        assert paths.shape == (500, 21)

    def test_day_zero_is_deterministic_last_close(self) -> None:
        prices = _prices(_random_walk(300))
        paths = fc.simulate_gbm_paths(prices, 10, n_paths=200)
        assert paths is not None
        assert np.allclose(paths[:, 0], prices["close"].iloc[-1])

    def test_all_prices_are_positive(self) -> None:
        # GBM lives in log space, so a simulated price can never be <= 0.
        paths = fc.simulate_gbm_paths(_prices(_random_walk(300, sigma=0.05)), 60, n_paths=1000)
        assert paths is not None
        assert np.all(paths > 0)

    def test_deterministic_with_fixed_seed(self) -> None:
        prices = _prices(_random_walk(300))
        a = fc.simulate_gbm_paths(prices, 20, n_paths=100, random_state=42)
        b = fc.simulate_gbm_paths(prices, 20, n_paths=100, random_state=42)
        assert a is not None and b is not None
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        prices = _prices(_random_walk(300))
        a = fc.simulate_gbm_paths(prices, 20, n_paths=100, random_state=1)
        b = fc.simulate_gbm_paths(prices, 20, n_paths=100, random_state=2)
        assert a is not None and b is not None
        assert not np.array_equal(a, b)

    def test_calibration_matches_baseline_forecast_analytically(self) -> None:
        # Same random-walk model, simulated vs closed-form: with many paths the
        # simulated terminal log-return distribution's mean/std must converge to
        # baseline_forecast's analytic mu*h / sigma*sqrt(h) -- a real proof the
        # simulation implements GBM correctly, not just a plausible shape check.
        prices = _prices(_random_walk(500, mu=0.0006, sigma=0.02, seed=9))
        horizon = 40
        paths = fc.simulate_gbm_paths(prices, horizon, n_paths=50_000, random_state=7)
        assert paths is not None
        last_close = prices["close"].iloc[-1]
        terminal_log_returns = np.log(paths[:, -1] / last_close)

        baseline = fc.baseline_forecast(prices, horizon)
        assert baseline is not None
        analytic_mean = np.log1p(baseline.point_return)
        from scipy.stats import norm

        z = norm.ppf(0.5 + fc.DEFAULT_CONFIDENCE / 2.0)
        analytic_sd = (np.log1p(baseline.upper_return) - analytic_mean) / z

        assert terminal_log_returns.mean() == pytest.approx(analytic_mean, abs=0.01)
        assert terminal_log_returns.std(ddof=1) == pytest.approx(analytic_sd, rel=0.05)

    def test_too_short_is_none(self) -> None:
        assert fc.simulate_gbm_paths(_prices(_random_walk(10)), 20) is None

    def test_invalid_n_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            fc.simulate_gbm_paths(_prices(_random_walk(300)), 20, n_paths=0)


class TestMonteCarloFanChart:
    def test_default_percentiles_match_the_plan_spec(self) -> None:
        chart = fc.monte_carlo_fan_chart(_prices(_random_walk(300)), 20, n_paths=2000)
        assert chart is not None
        assert set(chart.percentiles) == {5.0, 50.0, 95.0}

    def test_days_and_percentile_arrays_span_the_full_horizon(self) -> None:
        horizon = 30
        chart = fc.monte_carlo_fan_chart(_prices(_random_walk(300)), horizon, n_paths=1000)
        assert chart is not None
        assert list(chart.days) == list(range(1, horizon + 1))
        for path in chart.percentiles.values():
            assert len(path) == horizon

    def test_percentiles_are_ordered_at_every_day(self) -> None:
        chart = fc.monte_carlo_fan_chart(
            _prices(_random_walk(300)), 40, n_paths=3000, percentiles=(5.0, 50.0, 95.0)
        )
        assert chart is not None
        p5, p50, p95 = chart.percentiles[5.0], chart.percentiles[50.0], chart.percentiles[95.0]
        assert np.all(p5 <= p50) and np.all(p50 <= p95)

    def test_fan_widens_over_time(self) -> None:
        # The defining shape of a "fan": the 5th-95th spread on the last day
        # must exceed the spread on the first day.
        prices = _prices(_random_walk(300, sigma=0.02))
        chart = fc.monte_carlo_fan_chart(prices, 60, n_paths=5000, random_state=3)
        assert chart is not None
        spread = chart.percentiles[95.0] - chart.percentiles[5.0]
        assert spread[-1] > spread[0]

    def test_median_path_matches_baseline_point_forecast(self) -> None:
        prices = _prices(_random_walk(400, mu=0.0004, sigma=0.015, seed=4))
        horizon = 20
        chart = fc.monte_carlo_fan_chart(prices, horizon, n_paths=20_000, random_state=11)
        assert chart is not None
        baseline = fc.baseline_forecast(prices, horizon)
        assert baseline is not None
        assert chart.percentiles[50.0][-1] == pytest.approx(baseline.point_price, rel=0.03)

    def test_calibration_fields_are_reported(self) -> None:
        chart = fc.monte_carlo_fan_chart(_prices(_random_walk(300)), 20, n_paths=500)
        assert chart is not None
        assert chart.n_train == 299  # 300 closes -> 299 log returns
        assert chart.n_paths == 500

    def test_too_short_is_none(self) -> None:
        assert fc.monte_carlo_fan_chart(_prices(_random_walk(10)), 20) is None

    def test_empty_percentiles_raises(self) -> None:
        with pytest.raises(ValueError, match="percentiles"):
            fc.monte_carlo_fan_chart(_prices(_random_walk(300)), 20, percentiles=())

    def test_out_of_range_percentile_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            fc.monte_carlo_fan_chart(_prices(_random_walk(300)), 20, percentiles=(5.0, 150.0))

    def test_deterministic_with_fixed_seed(self) -> None:
        prices = _prices(_random_walk(300))
        a = fc.monte_carlo_fan_chart(prices, 20, n_paths=500, random_state=5)
        b = fc.monte_carlo_fan_chart(prices, 20, n_paths=500, random_state=5)
        assert a is not None and b is not None
        for p in a.percentiles:
            assert np.array_equal(a.percentiles[p], b.percentiles[p])


# --------------------------------------------------------------------------- #
# Input validation shared across models
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_missing_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close"):
            fc.baseline_forecast(pd.DataFrame({"open": [1.0, 2.0]}), 20)

    @pytest.mark.parametrize(
        "model",
        ["baseline_forecast", "statistical_forecast", "ml_forecast", "simulate_gbm_paths"],
    )
    def test_non_positive_horizon_raises(self, model: str) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            getattr(fc, model)(_prices(_random_walk(300)), 0)

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            fc.baseline_forecast(_prices(_random_walk(300)), 20, confidence_level=1.5)


class TestBandCalibration:
    """The residual band must be as wide as it claims to be.

    Measured on real history before this correction, the nominal 90% band
    contained the realized return only 78.8% of the time across 236
    non-overlapping folds -- reality landed outside it about twice as often as
    advertised. Two finite-sample causes: overlapping horizon returns mean `n`
    residuals carry only ~`n/h` independent observations, and extreme order
    statistics of a small sample are biased inward.
    """

    def test_correction_widens_the_band_relative_to_the_plain_quantiles(self) -> None:
        lower, upper = fc._calibrated_residual_quantiles(200, horizon_days=5, confidence_level=0.90)
        assert lower < 0.05 and upper > 0.95, "must be wider than the naive 5th/95th"
        assert lower == pytest.approx(1.0 - upper), "band stays symmetric in level"

    def test_more_overlap_means_a_wider_band_for_the_same_residual_count(self) -> None:
        # Same 200 residuals, longer horizon -> fewer independent observations
        # -> the band must widen, not stay put.
        _, short_h = fc._calibrated_residual_quantiles(200, horizon_days=5, confidence_level=0.90)
        _, long_h = fc._calibrated_residual_quantiles(200, horizon_days=20, confidence_level=0.90)
        assert long_h > short_h

    def test_collapses_to_the_observed_extremes_when_too_few_independent_residuals(self) -> None:
        # 200 residuals at h=63 is ~3 independent observations; a two-sided 90%
        # band needs ~19. The honest answer is the widest the sample supports --
        # the observed min/max -- not an interpolated, tighter one.
        lower, upper = fc._calibrated_residual_quantiles(
            200, horizon_days=63, confidence_level=0.90
        )
        assert (lower, upper) == (0.0, 1.0)

    def test_never_exceeds_the_valid_quantile_range(self) -> None:
        for n_res in (10, 60, 200, 5_000):
            for h in (1, 5, 20, 63, 252):
                for conf in (0.5, 0.8, 0.9, 0.99):
                    lower, upper = fc._calibrated_residual_quantiles(
                        n_res, horizon_days=h, confidence_level=conf
                    )
                    assert 0.0 <= lower <= 0.5 <= upper <= 1.0

    def test_a_real_forecast_band_is_wider_than_the_uncorrected_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End to end: the published band must be strictly wider than the plain
        # empirical quantiles it replaced, on the same series and residuals.
        prices = _prices(_random_walk(900, seed=21))
        corrected = fc.ml_forecast(prices, 20)

        monkeypatch.setattr(
            fc,
            "_calibrated_residual_quantiles",
            lambda n, *, horizon_days, confidence_level: (
                (1.0 - confidence_level) / 2.0,
                1.0 - (1.0 - confidence_level) / 2.0,
            ),
        )
        uncorrected = fc.ml_forecast(prices, 20)

        assert corrected is not None and uncorrected is not None
        assert corrected.point_return == pytest.approx(uncorrected.point_return), (
            "the correction must move only the band, never the point forecast"
        )
        corrected_width = corrected.upper_return - corrected.lower_return
        uncorrected_width = uncorrected.upper_return - uncorrected.lower_return
        assert corrected_width > uncorrected_width


class TestArimaDrift:
    """ARIMA must be able to express a horizon-dependent view.

    statsmodels silently defaults to `trend="n"` when `d > 0`. With no drift
    term the h-step forecast of a differenced series decays to a constant within
    a few steps, so the model returned the SAME number at 5, 20, 63 and 252 days
    for every symbol -- 83% of them under 0.1% on real history. It was presented
    as one of three competing models, with its own track record, while being
    structurally incapable of forecasting anything.
    """

    def test_forecast_scales_with_the_horizon(self) -> None:
        # A series with real drift must produce a visibly larger move at a
        # longer horizon. This is the exact property `trend="n"` destroyed.
        prices = _prices(_random_walk(900, mu=0.0006, sigma=0.012, seed=31))
        short = fc.statistical_forecast(prices, 5)
        long = fc.statistical_forecast(prices, 252)
        assert short is not None and long is not None
        assert abs(long.point_return) > abs(short.point_return) * 5

    def test_no_drift_collapses_to_a_flat_forecast(self) -> None:
        # Pins the diagnosis itself, so a future edit that drops the trend term
        # fails here with an explanation rather than silently regressing.
        prices = _prices(_random_walk(900, mu=0.0006, sigma=0.012, seed=31))
        flat = [
            fc.statistical_forecast(prices, h, trend="n").point_return  # type: ignore[union-attr]
            for h in (20, 63, 252)
        ]
        assert max(flat) - min(flat) < 1e-4, "no-drift ARIMA is flat across horizons"
        assert all(abs(v) < 0.01 for v in flat)

    def test_recovers_the_drift_of_a_deterministic_ramp(self) -> None:
        # An exponential ramp has a known constant log-drift; a drifting ARIMA
        # should forecast close to `drift * h` in log space.
        n, per_bar = 600, 0.0005
        path = 100.0 * np.exp(np.arange(n) * per_bar)
        result = fc.statistical_forecast(_prices(path), 63)
        assert result is not None
        assert np.log1p(result.point_return) == pytest.approx(per_bar * 63, rel=0.25)
