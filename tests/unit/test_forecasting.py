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
        prices = _prices(_random_walk(500, seed=4))
        # Exogenous fundamental signal available every day.
        exog = pd.DataFrame({"value_score": np.linspace(0, 1, 500)}, index=prices.index)
        r = fc.ml_forecast(prices, 252, exog=exog, exog_buckets={"value_score": "fundamental"})
        assert isinstance(r, Forecast)

    def test_deterministic(self) -> None:
        prices = _prices(_random_walk(400, seed=11))
        a = fc.ml_forecast(prices, 20)
        b = fc.ml_forecast(prices, 20)
        assert a is not None and b is not None
        assert a.point_return == b.point_return


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
# Input validation shared across models
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_missing_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close"):
            fc.baseline_forecast(pd.DataFrame({"open": [1.0, 2.0]}), 20)

    @pytest.mark.parametrize("model", ["baseline_forecast", "statistical_forecast", "ml_forecast"])
    def test_non_positive_horizon_raises(self, model: str) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            getattr(fc, model)(_prices(_random_walk(300)), 0)

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            fc.baseline_forecast(_prices(_random_walk(300)), 20, confidence_level=1.5)
