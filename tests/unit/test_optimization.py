import numpy as np
import pandas as pd
import pytest

from quantpulse.portfolio import optimization as opt


def _correlated_prices(
    n_assets: int = 6, n_days: int = 500, seed: int = 0, start: str = "2022-01-01"
) -> pd.DataFrame:
    """A panel with one common factor, so names have genuinely different betas."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="B")
    common = rng.normal(0.0004, 0.01, n_days)
    return pd.DataFrame(
        {
            f"S{i}": 100 * np.cumprod(1 + (common * (0.5 + i * 0.2) + rng.normal(0, 0.008, n_days)))
            for i in range(n_assets)
        },
        index=idx,
    )


def _weights_sum_to_one(portfolio: opt.OptimizedPortfolio) -> bool:
    return sum(portfolio.weights.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Covariance & panel hygiene
# --------------------------------------------------------------------------- #


class TestCovariance:
    def test_shrunk_covariance_is_annualized_and_symmetric(self) -> None:
        prices = _correlated_prices()
        cov = opt.annualized_covariance(prices)
        assert cov.shape == (6, 6)
        assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
        # Annualized variance of a ~1.5%/day name is far above its daily value.
        daily = prices.pct_change().dropna().var(ddof=1)
        assert (np.diag(cov.to_numpy()) > daily.to_numpy() * 100).all()

    def test_shrinkage_produces_a_positive_definite_matrix(self) -> None:
        # More assets than observations -- exactly where sample covariance is
        # singular and an optimizer's weights become meaningless.
        prices = _correlated_prices(n_assets=12, n_days=80)
        cov = opt.annualized_covariance(prices)
        assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)

    def test_common_window_drops_dates_with_any_missing_name(self) -> None:
        prices = _correlated_prices(n_assets=3, n_days=200)
        prices.iloc[5, 1] = np.nan
        panel = opt._clean_panel(prices, min_observations=60)
        assert len(panel) == 199
        assert panel.notna().all().all()


# --------------------------------------------------------------------------- #
# Mean-variance
# --------------------------------------------------------------------------- #


class TestMeanVariance:
    def test_max_sharpe_respects_long_only_and_the_weight_cap(self) -> None:
        result = opt.mean_variance_optimize(_correlated_prices(), max_weight=0.3)
        assert result is not None
        assert _weights_sum_to_one(result)
        assert all(w >= 0 for w in result.weights.values())
        assert max(result.weights.values()) <= 0.3 + 1e-6
        assert result.method == "mean_variance:max_sharpe"

    def test_min_volatility_beats_max_sharpe_on_volatility(self) -> None:
        prices = _correlated_prices()
        min_vol = opt.mean_variance_optimize(prices, objective="min_volatility", max_weight=1.0)
        max_sharpe = opt.mean_variance_optimize(prices, objective="max_sharpe", max_weight=1.0)
        assert min_vol is not None and max_sharpe is not None
        assert min_vol.volatility is not None and max_sharpe.volatility is not None
        assert min_vol.volatility <= max_sharpe.volatility

    def test_no_asset_beating_the_risk_free_rate_yields_none(self) -> None:
        # A universe that only ever falls has no tangency portfolio; the honest
        # answer is "no allocation", not a fabricated one.
        idx = pd.date_range("2022-01-01", periods=300, freq="B")
        prices = pd.DataFrame({f"S{i}": np.linspace(100, 60 - i, 300) for i in range(3)}, index=idx)
        # max_weight=1.0 so the feasibility guard doesn't fire first -- this
        # test is about the "no tangency portfolio exists" path.
        assert opt.mean_variance_optimize(prices, max_weight=1.0, risk_free_rate=0.02) is None

    def test_too_little_history_yields_none(self) -> None:
        assert opt.mean_variance_optimize(_correlated_prices(n_days=40)) is None

    def test_single_asset_yields_none(self) -> None:
        assert opt.mean_variance_optimize(_correlated_prices(n_assets=1)) is None

    def test_infeasible_weight_cap_is_rejected_loudly(self) -> None:
        # 6 assets capped at 10% each can only ever be 60% invested.
        with pytest.raises(ValueError, match="cannot fill a portfolio"):
            opt.mean_variance_optimize(_correlated_prices(), max_weight=0.1)

    def test_rejects_unknown_objective(self) -> None:
        with pytest.raises(ValueError, match="objective must be"):
            opt.mean_variance_optimize(_correlated_prices(), objective="max_return")


# --------------------------------------------------------------------------- #
# HRP -- including the scipy-private-attribute shim actually working
# --------------------------------------------------------------------------- #


class TestHierarchicalRiskParity:
    def test_produces_a_valid_long_only_allocation(self) -> None:
        result = opt.hierarchical_risk_parity(_correlated_prices())
        assert result is not None
        assert _weights_sum_to_one(result)
        assert all(w >= 0 for w in result.weights.values())
        assert result.n_assets == 6
        assert result.max_weight is None  # HRP takes no cap, by construction

    def test_allocates_more_to_the_lower_volatility_names(self) -> None:
        # Loadings on the common factor rise with the index, so S0 is the
        # calmest name and S5 the most volatile; risk parity should tilt toward
        # the calm end.
        prices = _correlated_prices()
        result = opt.hierarchical_risk_parity(prices)
        assert result is not None
        assert result.weights["S0"] > result.weights["S5"]

    def test_survives_the_scipy_linkage_methods_removal(self) -> None:
        # Regression guard for the upstream break: pypfopt 1.6.0 validates
        # against scipy's private _LINKAGE_METHODS, removed in scipy 1.18.
        for method in ("single", "average", "complete"):
            result = opt.hierarchical_risk_parity(_correlated_prices(), linkage_method=method)
            assert result is not None, f"HRP failed for linkage_method={method}"

    def test_too_little_history_yields_none(self) -> None:
        assert opt.hierarchical_risk_parity(_correlated_prices(n_days=40)) is None


# --------------------------------------------------------------------------- #
# Black-Litterman -- the prior, the views, and their invariants
# --------------------------------------------------------------------------- #


class TestEqualWeightPrior:
    def test_round_trip_optimizing_the_prior_returns_equal_weights(self) -> None:
        # The defining property of a reverse-optimized equilibrium prior: the
        # weights it was derived from must be what optimizing it gives back.
        # This is what `pi="equal"` (literally 1/n as a return) fails.
        prices = _correlated_prices()
        cov = opt.annualized_covariance(prices)
        prior = opt.equal_weight_prior(cov)

        from pypfopt import EfficientFrontier

        frontier = EfficientFrontier(prior, cov, weight_bounds=(0, 1))
        frontier.max_sharpe()
        weights = frontier.clean_weights()
        assert all(w == pytest.approx(1 / 6, abs=0.005) for w in weights.values())

    def test_prior_scales_with_covariance_not_universe_size(self) -> None:
        # `pi="equal"` would make the prior 1/n, so a 3-name and a 6-name
        # universe would get wildly different priors for the same assets.
        prices = _correlated_prices()
        wide = opt.equal_weight_prior(opt.annualized_covariance(prices))
        narrow = opt.equal_weight_prior(opt.annualized_covariance(prices[["S0", "S1", "S2"]]))
        assert wide["S0"] == pytest.approx(narrow["S0"], rel=0.5)
        assert all(v > 0 for v in wide)

    def test_higher_covariance_names_carry_a_higher_prior_return(self) -> None:
        prior = opt.equal_weight_prior(opt.annualized_covariance(_correlated_prices()))
        assert prior["S5"] > prior["S0"]


class TestViewsFromScores:
    def _prior(self) -> pd.Series:
        return opt.equal_weight_prior(opt.annualized_covariance(_correlated_prices()))

    def test_identical_scores_reproduce_the_prior_exactly(self) -> None:
        # A ranking with no dispersion carries no information, so it must not
        # move the allocation off the equilibrium.
        prior = self._prior()
        scores = pd.Series({s: 50.0 for s in prior.index})
        views = opt.views_from_composite_scores(scores, prior)
        assert np.allclose(views.to_numpy(), prior.to_numpy())

    def test_tilt_is_rank_preserving_and_centered(self) -> None:
        prior = self._prior()
        scores = pd.Series({f"S{i}": 10.0 + i * 16 for i in range(6)})
        views = opt.views_from_composite_scores(scores, prior, max_tilt=0.05)
        tilt = views - prior
        assert np.all(np.diff(tilt.to_numpy()) > 0)  # monotonic in the score
        assert tilt.abs().max() == pytest.approx(0.05)  # respects the cap
        assert tilt.sum() == pytest.approx(0.0, abs=1e-12)  # symmetric about the mean

    def test_above_average_scores_always_tilt_positive(self) -> None:
        prior = self._prior()
        scores = pd.Series({"S0": 10.0, "S1": 20.0, "S2": 30.0, "S3": 95.0, "S4": 96.0, "S5": 97.0})
        tilt = opt.views_from_composite_scores(scores, prior) - prior
        mean_score = scores.mean()
        for symbol, score in scores.items():
            assert (tilt[symbol] > 0) == (score > mean_score)

    def test_confidence_shrinks_a_view_toward_the_prior(self) -> None:
        prior = self._prior()
        scores = pd.Series({f"S{i}": 10.0 + i * 16 for i in range(6)})
        full = opt.views_from_composite_scores(scores, prior) - prior
        damped = (
            opt.views_from_composite_scores(
                scores, prior, confidences=pd.Series({s: 0.25 for s in prior.index})
            )
            - prior
        )
        assert damped.abs().sum() == pytest.approx(full.abs().sum() * 0.25)

    def test_rejects_a_non_positive_tilt_cap(self) -> None:
        with pytest.raises(ValueError, match="max_tilt"):
            opt.views_from_composite_scores(pd.Series({"S0": 50.0}), self._prior(), max_tilt=0.0)


class TestBlackLitterman:
    def _scores(self, ascending: bool = True) -> pd.Series:
        values = [10.0 + i * 16 for i in range(6)]
        if not ascending:
            values = values[::-1]
        return pd.Series({f"S{i}": v for i, v in enumerate(values)})

    def test_produces_a_valid_capped_long_only_allocation(self) -> None:
        result = opt.black_litterman_optimize(_correlated_prices(), self._scores(), max_weight=0.35)
        assert result is not None
        assert _weights_sum_to_one(result)
        assert all(w >= 0 for w in result.weights.values())
        assert max(result.weights.values()) <= 0.35 + 1e-6
        assert result.method == "black_litterman"

    def test_allocates_toward_the_higher_scoring_names(self) -> None:
        prices = _correlated_prices()
        result = opt.black_litterman_optimize(prices, self._scores(), max_weight=0.35)
        assert result is not None
        top = result.weights.get("S5", 0.0) + result.weights.get("S4", 0.0)
        bottom = result.weights.get("S0", 0.0) + result.weights.get("S1", 0.0)
        assert top > bottom

    def test_reversing_the_scores_reverses_the_tilt(self) -> None:
        prices = _correlated_prices()
        up = opt.black_litterman_optimize(prices, self._scores(True), max_weight=0.35)
        down = opt.black_litterman_optimize(prices, self._scores(False), max_weight=0.35)
        assert up is not None and down is not None
        assert up.view_tilt["S5"] > 0 > down.view_tilt["S5"]
        assert down.view_tilt["S0"] > 0 > up.view_tilt["S0"]

    def test_flat_scores_give_the_equilibrium_allocation(self) -> None:
        # No dispersion in the ranking -> no tilt -> the equal-weight
        # equilibrium the prior was reverse-optimized from.
        prices = _correlated_prices()
        flat = pd.Series({f"S{i}": 50.0 for i in range(6)})
        result = opt.black_litterman_optimize(prices, flat, max_weight=1.0)
        assert result is not None
        assert all(w == pytest.approx(1 / 6, abs=0.01) for w in result.weights.values())
        assert all(abs(t) < 1e-9 for t in result.view_tilt.values())

    def test_uniform_confidence_keeps_the_tilt_monotonic_in_the_scores(self) -> None:
        prices = _correlated_prices()
        result = opt.black_litterman_optimize(
            prices,
            self._scores(),
            confidences=pd.Series({f"S{i}": 1.0 for i in range(6)}),
            max_weight=0.35,
        )
        assert result is not None
        tilts = [result.view_tilt[f"S{i}"] for i in range(6)]
        assert np.all(np.diff(tilts) > 0)

    def test_view_tilt_is_reported_so_covariance_coupling_is_visible(self) -> None:
        # Documented caveat: with mixed per-name confidences the posterior tilt
        # need not follow the score order, because Black-Litterman propagates
        # each view through the covariance matrix. The point of this test is
        # that the realized tilt is *exposed*, not that it stays monotonic.
        mixed = pd.Series({f"S{i}": c for i, c in enumerate([0.9, 0.9, 0.5, 0.5, 0.1, 0.1])})
        result = opt.black_litterman_optimize(
            _correlated_prices(), self._scores(), confidences=mixed, max_weight=0.35
        )
        assert result is not None
        assert set(result.view_tilt) == {f"S{i}" for i in range(6)}
        assert all(np.isfinite(v) for v in result.view_tilt.values())

    def test_scores_are_matched_by_symbol_not_position(self) -> None:
        prices = _correlated_prices()
        shuffled = pd.Series({"S3": 90.0, "S0": 10.0, "S5": 95.0, "S1": 20.0})
        result = opt.black_litterman_optimize(prices, shuffled, max_weight=0.5)
        assert result is not None
        assert set(result.weights) <= {"S0", "S1", "S3", "S5"}  # unscored names drop out
        assert result.view_tilt["S5"] > result.view_tilt["S0"]

    def test_too_few_scored_names_yields_none(self) -> None:
        prices = _correlated_prices()
        assert opt.black_litterman_optimize(prices, pd.Series({"S0": 80.0})) is None

    def test_too_little_history_yields_none(self) -> None:
        prices = _correlated_prices(n_days=40)
        assert opt.black_litterman_optimize(prices, self._scores()) is None


# --------------------------------------------------------------------------- #
# Fractional Kelly
# --------------------------------------------------------------------------- #


class TestKelly:
    def test_hand_check_against_the_formula(self) -> None:
        # p=0.6, b=2 -> f* = (2*0.6 - 0.4)/2 = 0.4; quarter-Kelly = 0.10
        assert opt.kelly_position_fraction(0.6, 2.0, max_position=1.0) == pytest.approx(0.10)

    def test_defaults_to_a_quarter_of_full_kelly(self) -> None:
        full = opt.kelly_position_fraction(0.6, 2.0, fraction=1.0, max_position=1.0)
        quarter = opt.kelly_position_fraction(0.6, 2.0, max_position=1.0)
        assert full is not None and quarter is not None
        assert quarter == pytest.approx(full * opt.DEFAULT_KELLY_FRACTION)

    def test_no_edge_yields_zero_not_a_negative_position(self) -> None:
        # p=0.3, b=1 -> f* = -0.4; long-only means "don't hold it", not "short it".
        assert opt.kelly_position_fraction(0.3, 1.0) == 0.0

    def test_capped_at_the_concentration_limit(self) -> None:
        # A flattering edge must not propose a position Section 9 would flag.
        assert opt.kelly_position_fraction(0.95, 5.0, max_position=0.15) == pytest.approx(0.15)

    def test_missing_or_nonsensical_inputs_yield_none(self) -> None:
        assert opt.kelly_position_fraction(None, 2.0) is None
        assert opt.kelly_position_fraction(0.6, None) is None
        assert opt.kelly_position_fraction(np.nan, 2.0) is None
        assert opt.kelly_position_fraction(1.5, 2.0) is None  # hit rate out of range
        assert opt.kelly_position_fraction(0.6, 0.0) is None  # non-positive payoff

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            opt.kelly_position_fraction(0.6, 2.0, fraction=0.0)
        with pytest.raises(ValueError, match="max_position"):
            opt.kelly_position_fraction(0.6, 2.0, max_position=0.0)


# --------------------------------------------------------------------------- #
# Cross-method consistency
# --------------------------------------------------------------------------- #


class TestMethodComparison:
    def test_all_three_methods_agree_on_the_basic_contract(self) -> None:
        prices = _correlated_prices()
        scores = pd.Series({f"S{i}": 10.0 + i * 16 for i in range(6)})
        results = [
            opt.mean_variance_optimize(prices, max_weight=0.35),
            opt.hierarchical_risk_parity(prices),
            opt.black_litterman_optimize(prices, scores, max_weight=0.35),
        ]
        for result in results:
            assert result is not None
            assert _weights_sum_to_one(result)
            assert all(w >= 0 for w in result.weights.values())
            assert result.n_observations == 500
