import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantpulse.analysis import backtest as bt
from quantpulse.analysis import risk


def _series(values: list[float] | np.ndarray, start: str = "2021-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _panel(series_by_symbol: dict[str, list[float]], start: str = "2021-01-01") -> pd.DataFrame:
    n = len(next(iter(series_by_symbol.values())))
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(series_by_symbol, index=idx, dtype=float)


def _random_walk(n: int, *, mu: float = 0.0004, sigma: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))


# --------------------------------------------------------------------------- #
# Return construction
# --------------------------------------------------------------------------- #


class TestReturns:
    def test_to_returns_hand_check(self) -> None:
        returns = risk.to_returns(_series([100.0, 110.0, 99.0]))
        assert list(returns) == pytest.approx([0.1, -0.1])

    def test_to_returns_sorts_and_drops_bad_prices(self) -> None:
        prices = _series([100.0, 0.0, 200.0])
        prices.iloc[1] = np.nan
        assert list(risk.to_returns(prices)) == pytest.approx([1.0])

    def test_panel_never_invents_a_return_across_a_gap(self) -> None:
        panel = _panel({"A": [100.0, np.nan, 112.0, 113.0], "B": [10.0, 11.0, 12.0, 13.0]})
        returns = risk.returns_panel(panel)
        # The 100 -> 112 move spans a missing bar, so it stays missing rather
        # than being reported as a one-day +12%.
        assert returns["A"].isna().tolist() == [True, True, False]
        assert returns["A"].iloc[-1] == pytest.approx(113.0 / 112.0 - 1.0)
        assert returns["B"].notna().all()  # the dense column is unaffected

    def test_equal_weight_market_is_the_cross_sectional_mean(self) -> None:
        panel = _panel({"A": [100.0, 110.0], "B": [50.0, 50.0]})
        market = risk.equal_weight_market_returns(panel)
        assert list(market) == pytest.approx([0.05])  # (0.10 + 0.00) / 2

    def test_equal_weight_market_drops_thinly_covered_dates(self) -> None:
        panel = _panel({"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, np.nan, 55.0, 60.5]})
        # B's gap swallows both of its middle returns, so only the last bar has
        # a return for both names.
        assert len(risk.equal_weight_market_returns(panel, min_names=2)) == 1
        assert len(risk.equal_weight_market_returns(panel, min_names=1)) == 3


# --------------------------------------------------------------------------- #
# Volatility (Section 7.7: historical & implied)
# --------------------------------------------------------------------------- #


class TestVolatility:
    def test_annualization_hand_check(self) -> None:
        n = 40
        returns = _series([0.01, -0.01] * (n // 2))
        # mean 0, so sample variance = n * 0.01**2 / (n - 1)
        expected = 0.01 * np.sqrt(n / (n - 1)) * np.sqrt(252.0)
        assert risk.historical_volatility(returns) == pytest.approx(expected)

    def test_too_short_is_none(self) -> None:
        assert risk.historical_volatility(_series([0.01, -0.01] * 9)) is None

    def test_lookback_uses_only_the_recent_window(self) -> None:
        calm = [0.001, -0.001] * 30
        stormy = [0.05, -0.05] * 30
        returns = _series(calm + stormy)
        recent = risk.historical_volatility(returns, lookback=60)
        full = risk.historical_volatility(returns)
        assert recent is not None and full is not None
        assert recent > full

    def test_implied_premium_is_implied_minus_historical(self) -> None:
        returns = _series([0.01, -0.01] * 30)
        profile = risk.volatility_profile(returns, implied_volatility=0.40)
        assert profile.historical is not None
        assert profile.implied == pytest.approx(0.40)
        assert profile.implied_premium == pytest.approx(0.40 - profile.historical)

    def test_absent_or_unusable_implied_stays_none(self) -> None:
        returns = _series([0.01, -0.01] * 30)
        assert risk.volatility_profile(returns).implied is None
        assert risk.volatility_profile(returns, implied_volatility=np.nan).implied is None
        assert risk.volatility_profile(returns, implied_volatility=0.0).implied is None

    def test_short_history_still_reports_implied(self) -> None:
        profile = risk.volatility_profile(_series([0.01, -0.01]), implied_volatility=0.3)
        assert profile.historical is None
        assert profile.implied == pytest.approx(0.3)
        assert profile.implied_premium is None


# --------------------------------------------------------------------------- #
# Sortino -- the denominator is the whole point
# --------------------------------------------------------------------------- #


class TestSortino:
    def test_downside_deviation_divides_by_the_full_sample(self) -> None:
        # One -10% period and nine +2% periods. Target 0, annualization off.
        returns = _series([-0.10] + [0.02] * 9)
        mean = (9 * 0.02 - 0.10) / 10
        correct_dd = np.sqrt(0.10**2 / 10)  # squared shortfall averaged over ALL 10
        naive_dd = np.sqrt(0.10**2 / 1)  # ...averaged over only the 1 losing period

        value = risk.sortino_ratio(returns, periods_per_year=1.0)
        assert value == pytest.approx(mean / correct_dd)
        # The classic bug inflates the ratio by sqrt(n / n_losses) -- here 3.16x.
        assert value != pytest.approx(mean / naive_dd)
        assert value / (mean / naive_dd) == pytest.approx(np.sqrt(10.0))

    def test_no_downside_is_none_not_infinity(self) -> None:
        assert risk.sortino_ratio(_series([0.01] * 20), periods_per_year=1.0) is None

    def test_upside_volatility_is_not_penalized(self) -> None:
        # Enlarging an already-positive period leaves the downside deviation
        # untouched, so Sortino scales exactly with the mean -- while Sharpe,
        # which charges for volatility in both directions, does not.
        base = _series([0.02, -0.01] * 4)
        boosted = base.copy()
        boosted.iloc[0] = 0.10
        mean_ratio = float(boosted.mean() / base.mean())

        sortino_base = risk.sortino_ratio(base, periods_per_year=1.0)
        sortino_boosted = risk.sortino_ratio(boosted, periods_per_year=1.0)
        assert sortino_base is not None and sortino_boosted is not None
        assert sortino_boosted / sortino_base == pytest.approx(mean_ratio)

        sharpe_base = bt.sharpe_ratio(base, periods_per_year=1.0)
        sharpe_boosted = bt.sharpe_ratio(boosted, periods_per_year=1.0)
        assert sharpe_base is not None and sharpe_boosted is not None
        assert sharpe_boosted / sharpe_base < mean_ratio

    def test_target_return_shifts_the_threshold(self) -> None:
        returns = _series([0.03] * 5 + [0.01] * 5)
        assert risk.sortino_ratio(returns, periods_per_year=1.0) is None  # nothing below 0
        below_target = risk.sortino_ratio(returns, periods_per_year=1.0, target_return=0.02)
        assert below_target is not None

    def test_annual_risk_free_is_converted_to_the_period(self) -> None:
        returns = _series([0.01] * 12)
        # 12% annual risk-free = 1% monthly, so every period sits exactly at the
        # threshold: no downside, hence no defined ratio.
        assert risk.sortino_ratio(returns, periods_per_year=12.0, risk_free_rate=0.12) is None

    def test_too_short_is_none(self) -> None:
        assert risk.sortino_ratio(_series([-0.01]), periods_per_year=1.0) is None


# --------------------------------------------------------------------------- #
# Beta
# --------------------------------------------------------------------------- #


class TestBeta:
    def _market(self) -> pd.Series:
        return risk.to_returns(_series(_random_walk(200)))

    def test_exact_multiple_of_the_market(self) -> None:
        market = self._market()
        result = risk.beta(market * 2.0, market)
        assert result is not None
        assert result.beta == pytest.approx(2.0)
        assert result.r_squared == pytest.approx(1.0)
        assert result.n_observations == len(market)

    def test_market_against_itself_is_one(self) -> None:
        market = self._market()
        result = risk.beta(market, market)
        assert result is not None and result.beta == pytest.approx(1.0)

    def test_r_squared_exposes_a_beta_worth_little(self) -> None:
        rng = np.random.default_rng(7)
        market = _series(rng.normal(0.0, 0.01, 300))
        noisy = _series(market.to_numpy() + rng.normal(0.0, 0.08, 300))
        result = risk.beta(noisy, market)
        assert result is not None
        assert result.beta == pytest.approx(1.0, abs=0.3)  # still ~1...
        assert result.r_squared is not None and result.r_squared < 0.1  # ...but explains nothing

    def test_uses_only_overlapping_dates(self) -> None:
        market = _series(np.tile([0.01, -0.02, 0.015, -0.005], 50), start="2021-01-01")
        # Asset overlaps the market's last 100 bars at exactly 3x, and its
        # earlier (non-overlapping) bars are deliberately unrelated noise.
        overlap = market.iloc[-100:] * 3.0
        earlier = _series(np.full(100, -0.5), start="2019-01-01")
        asset = pd.concat([earlier, overlap])

        result = risk.beta(asset, market)
        assert result is not None
        assert result.n_observations == 100
        assert result.beta == pytest.approx(3.0)

    def test_too_little_overlap_is_none(self) -> None:
        market = _series(np.tile([0.01, -0.01], 30))
        assert risk.beta(market.iloc[:59], market) is None
        assert risk.beta(market.iloc[:60], market) is not None

    def test_flat_market_is_none(self) -> None:
        flat = _series([0.0] * 100)
        moving = _series(np.tile([0.01, -0.01], 50))
        assert risk.beta(moving, flat) is None

    def test_portfolio_beta_is_the_weighted_average(self) -> None:
        value = risk.portfolio_beta({"A": 2.0, "B": 0.5}, {"A": 0.25, "B": 0.75})
        assert value == pytest.approx(0.25 * 2.0 + 0.75 * 0.5)

    def test_portfolio_beta_counts_cash_as_zero_when_told_to(self) -> None:
        value = risk.portfolio_beta({"A": 1.2, "CASH": 0.0}, {"A": 0.5, "CASH": 0.5})
        assert value == pytest.approx(0.6)

    def test_portfolio_beta_renormalizes_over_known_betas(self) -> None:
        # B's beta is unknown -- it drops out rather than being read as zero,
        # which would drag the portfolio figure toward market-neutral.
        value = risk.portfolio_beta({"A": 1.5, "B": None}, {"A": 0.5, "B": 0.5})
        assert value == pytest.approx(1.5)

    def test_portfolio_beta_with_no_known_betas_is_none(self) -> None:
        assert risk.portfolio_beta({"A": None}, {"A": 1.0}) is None
        assert risk.portfolio_beta({}, {"A": 1.0}) is None


# --------------------------------------------------------------------------- #
# Value-at-Risk
# --------------------------------------------------------------------------- #


class TestValueAtRisk:
    def test_historical_hand_check_and_sign_convention(self) -> None:
        # 10 losses of -10%, 90 gains of +1%: the 5% quantile lands squarely on
        # a loss, so VaR is reported as a positive 0.10 loss fraction.
        returns = _series([-0.10] * 10 + [0.01] * 90)
        result = risk.value_at_risk(returns)
        assert result is not None
        assert result.var == pytest.approx(0.10)
        assert result.expected_shortfall == pytest.approx(0.10)
        assert result.method == "historical"
        assert result.n_observations == 100

    def test_expected_shortfall_exceeds_var_on_a_graded_tail(self) -> None:
        tail = [-0.10 - 0.01 * i for i in range(10)]  # -0.10 .. -0.19
        returns = _series(tail + [0.01] * 90)
        result = risk.value_at_risk(returns)
        assert result is not None
        assert result.var == pytest.approx(0.1405)
        assert result.expected_shortfall == pytest.approx(0.17)  # mean of the worst five
        assert result.expected_shortfall > result.var

    def test_a_window_with_no_losing_tail_reports_negative_var(self) -> None:
        # Honest rather than clamped: even the 5th-percentile period gained.
        result = risk.value_at_risk(_series([0.01] * 50 + [0.02] * 50))
        assert result is not None and result.var < 0

    def test_parametric_matches_the_normal_quantile(self) -> None:
        returns = _series(np.random.default_rng(3).normal(0.001, 0.02, 250))
        values = returns.to_numpy()
        expected = -(values.mean() + values.std(ddof=1) * norm.ppf(0.05))
        result = risk.value_at_risk(returns, method="parametric")
        assert result is not None
        assert result.var == pytest.approx(expected)
        assert result.method == "parametric"

    def test_historical_needs_a_populated_tail(self) -> None:
        returns = _series(np.random.default_rng(4).normal(0.0, 0.02, 99))
        assert risk.value_at_risk(returns) is None  # 95% needs 100 observations
        assert risk.value_at_risk(returns, method="parametric") is not None
        longer = _series(np.random.default_rng(4).normal(0.0, 0.02, 100))
        assert risk.value_at_risk(longer) is not None

    def test_higher_confidence_demands_far_more_history(self) -> None:
        returns = _series(np.random.default_rng(5).normal(0.0, 0.02, 200))
        assert risk.value_at_risk(returns, confidence=0.95) is not None
        assert risk.value_at_risk(returns, confidence=0.99) is None  # needs 500

    def test_flat_series_has_no_parametric_var(self) -> None:
        assert risk.value_at_risk(_series([0.01] * 50), method="parametric") is None

    def test_rejects_bad_arguments(self) -> None:
        returns = _series([0.01] * 200)
        with pytest.raises(ValueError):
            risk.value_at_risk(returns, confidence=1.0)
        with pytest.raises(ValueError):
            risk.value_at_risk(returns, method="montecarlo")


# --------------------------------------------------------------------------- #
# Correlation across holdings
# --------------------------------------------------------------------------- #


class TestCorrelation:
    def _returns(self) -> pd.DataFrame:
        base = np.tile([0.01, -0.02, 0.015, -0.005], 15)  # 60 observations
        return _panel({"A": list(base), "B": list(base * 2.0), "C": list(-base)})

    def test_perfectly_correlated_pairs(self) -> None:
        matrix = risk.correlation_matrix(self._returns())
        assert matrix.loc["A", "B"] == pytest.approx(1.0)
        assert matrix.loc["A", "C"] == pytest.approx(-1.0)

    def test_short_overlap_yields_nan_not_a_confident_number(self) -> None:
        base = np.tile([0.01, -0.02], 5)  # only 10 observations
        matrix = risk.correlation_matrix(_panel({"A": list(base), "B": list(base)}))
        assert pd.isna(matrix.loc["A", "B"])

    def test_average_pairwise_correlation_hand_check(self) -> None:
        matrix = risk.correlation_matrix(self._returns())
        # pairs: A-B = 1, A-C = -1, B-C = -1
        assert risk.average_pairwise_correlation(matrix) == pytest.approx(-1 / 3)

    def test_average_pairwise_correlation_without_pairs_is_none(self) -> None:
        assert risk.average_pairwise_correlation(pd.DataFrame()) is None

    def test_most_correlated_pairs_are_ranked_and_capped(self) -> None:
        pairs = risk.most_correlated_pairs(risk.correlation_matrix(self._returns()), top_n=2)
        assert [(a, b) for a, b, _ in pairs] == [("A", "B"), ("A", "C")]
        assert pairs[0][2] == pytest.approx(1.0)

    def test_most_correlated_pairs_rejects_bad_top_n(self) -> None:
        with pytest.raises(ValueError):
            risk.most_correlated_pairs(risk.correlation_matrix(self._returns()), top_n=0)


# --------------------------------------------------------------------------- #
# Portfolio aggregation
# --------------------------------------------------------------------------- #


class TestPortfolioReturns:
    def test_equal_weights_average_the_holdings(self) -> None:
        returns = _panel({"A": [0.10, -0.02], "B": [0.00, 0.02]})
        series = risk.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
        assert list(series) == pytest.approx([0.05, 0.0])

    def test_weights_are_renormalized(self) -> None:
        returns = _panel({"A": [0.10, -0.02], "B": [0.00, 0.02]})
        raw = risk.portfolio_returns(returns, {"A": 20.0, "B": 20.0})
        normalized = risk.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
        assert list(raw) == pytest.approx(list(normalized))

    def test_cash_dilutes_the_portfolio(self) -> None:
        returns = _panel({"A": [0.10, -0.02]})
        series = risk.portfolio_returns(returns, {"A": 0.5}, cash_weight=0.5)
        assert list(series) == pytest.approx([0.05, -0.01])

    def test_incomplete_cross_sections_are_dropped(self) -> None:
        returns = _panel({"A": [0.10, 0.01, 0.02], "B": [0.00, np.nan, 0.04]})
        series = risk.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
        assert len(series) == 2
        assert list(series) == pytest.approx([0.05, 0.03])

    def test_missing_price_history_raises_instead_of_becoming_cash(self) -> None:
        returns = _panel({"A": [0.10, -0.02]})
        with pytest.raises(ValueError, match="no return series"):
            risk.portfolio_returns(returns, {"A": 0.5, "GONE": 0.5})

    def test_short_positions_rejected(self) -> None:
        returns = _panel({"A": [0.10], "B": [0.01]})
        with pytest.raises(ValueError, match="long-only"):
            risk.portfolio_returns(returns, {"A": 1.5, "B": -0.5})

    def test_zero_total_weight_rejected(self) -> None:
        returns = _panel({"A": [0.10]})
        with pytest.raises(ValueError):
            risk.portfolio_returns(returns, {"A": 0.0})


class TestPortfolioRisk:
    def _panel_returns(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        return _panel({name: list(rng.normal(0.0005, 0.015, 400)) for name in ("A", "B", "C")})

    def test_beta_equals_the_weighted_average_of_holding_betas(self) -> None:
        returns = self._panel_returns()
        market = risk.equal_weight_market_returns((1.0 + returns).cumprod())
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        summary = risk.portfolio_risk(returns, weights, market_returns=market)
        per_name = {
            symbol: (result.beta if (result := risk.beta(returns[symbol], market)) else None)
            for symbol in weights
        }
        assert summary.beta is not None
        # Section 9 defines portfolio beta as the weighted average of holdings'
        # betas; regressing the portfolio series is the same number (beta is
        # linear in the weights) over the same dates.
        assert summary.beta.beta == pytest.approx(risk.portfolio_beta(per_name, weights))

    def test_diversification_lowers_volatility_below_the_weighted_average(self) -> None:
        returns = self._panel_returns()  # three independent series
        weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        summary = risk.portfolio_risk(returns, weights)
        individual = [risk.historical_volatility(returns[s]) for s in weights]
        assert summary.volatility is not None
        assert all(v is not None for v in individual)
        assert summary.volatility < np.mean([v for v in individual if v is not None])

    def test_cash_reduces_reported_risk_proportionally(self) -> None:
        returns = self._panel_returns()
        weights = {"A": 0.5, "B": 0.5}
        invested = risk.portfolio_risk(returns, weights)
        half_cash = risk.portfolio_risk(returns, {"A": 0.25, "B": 0.25}, cash_weight=0.5)
        assert invested.volatility is not None and half_cash.volatility is not None
        assert half_cash.volatility == pytest.approx(invested.volatility / 2.0)
        assert half_cash.cash_weight == pytest.approx(0.5)

    def test_correlations_cover_only_the_held_names(self) -> None:
        returns = self._panel_returns()
        summary = risk.portfolio_risk(returns, {"A": 0.5, "B": 0.5})
        assert list(summary.correlations.columns) == ["A", "B"]
        assert summary.n_holdings == 2
        assert summary.average_correlation is not None

    def test_full_profile_is_populated(self) -> None:
        returns = self._panel_returns()
        market = risk.equal_weight_market_returns((1.0 + returns).cumprod())
        summary = risk.portfolio_risk(returns, {"A": 0.6, "B": 0.4}, market_returns=market)
        assert summary.sharpe is not None
        assert summary.sortino is not None
        assert summary.value_at_risk is not None
        assert summary.max_drawdown <= 0.0
        assert summary.n_observations == len(returns)


class TestStockRiskProfile:
    def _returns(self) -> pd.Series:
        prices = _random_walk(400, seed=2)
        return risk.to_returns(_series(prices))

    def test_populates_every_section_7_7_metric(self) -> None:
        returns = self._returns()
        market = _series(
            np.random.default_rng(9).normal(0.0004, 0.015, len(returns)),
            start=str(returns.index[0].date()),
        )
        profile = risk.stock_risk_profile(returns, market_returns=market, implied_volatility=0.35)
        assert profile.volatility.historical is not None
        assert profile.volatility.implied == pytest.approx(0.35)
        assert profile.beta is not None
        assert profile.sharpe is not None
        assert profile.sortino is not None
        assert profile.max_drawdown <= 0.0
        assert profile.value_at_risk is not None
        assert profile.n_observations == len(returns)

    def test_thin_history_degrades_per_metric_rather_than_failing(self) -> None:
        profile = risk.stock_risk_profile(_series([0.01, -0.02, 0.03]))
        assert profile.volatility.historical is None  # under the 20-bar floor
        assert profile.beta is None  # no market series supplied
        assert profile.value_at_risk is None  # nowhere near a populated 5% tail
        assert profile.sharpe is None  # under the one-year ratio floor
        assert profile.sortino is None
        # ...but what *can* honestly be computed still is. Max drawdown is a
        # description of the window that actually happened, not an estimate of
        # a population parameter, so a short window makes it narrow rather than
        # unreliable.
        assert profile.max_drawdown == pytest.approx(-0.02)
        assert profile.n_observations == 3

    def test_no_silent_fallback_between_var_methods(self) -> None:
        returns = _series(np.random.default_rng(6).normal(0.0, 0.02, 60))
        assert risk.stock_risk_profile(returns).value_at_risk is None
        parametric = risk.stock_risk_profile(returns, var_method="parametric").value_at_risk
        assert parametric is not None and parametric.method == "parametric"


class TestSharedMetrics:
    def test_sharpe_and_drawdown_are_the_backtests_own_functions(self) -> None:
        # Reused, not reimplemented: the portfolio page and the track-record
        # page can never drift apart on what "Sharpe" means.
        assert risk.sharpe_ratio is bt.sharpe_ratio
        assert risk.max_drawdown is bt.max_drawdown


class TestRatioSampleFloor:
    """Sharpe and Sortino abstain below a year of data.

    Both divide a mean by a dispersion, which makes them far noisier than either
    input; the standard error of the annualized ratio is roughly
    `sqrt(periods_per_year / n)`. On the 25 daily returns the demo database
    actually holds, 49% of 503 real S&P 500 names produced a Sortino above 3 and
    8.5% above 10 -- arithmetically correct, and meaningless. Value-at-Risk and
    the optimizers on the very same pages already refuse to answer on a sample
    this short; these two were the inconsistency.
    """

    def test_floor_is_one_year_at_any_frequency(self) -> None:
        assert risk.min_ratio_observations(252.0) == 252  # daily
        assert risk.min_ratio_observations(52.0) == 52  # weekly
        assert risk.min_ratio_observations(12.0) == 12  # monthly
        assert risk.min_ratio_observations(1.0) == 2  # never below two points

    def test_stock_profile_withholds_both_ratios_on_a_short_sample(self) -> None:
        rng = np.random.default_rng(0)
        short = _series(list(rng.normal(0.012, 0.03, 25)))  # the demo's shape
        profile = risk.stock_risk_profile(short)

        assert profile.sortino is None
        assert profile.sharpe is None
        # Descriptive statistics of the window itself still stand.
        assert profile.n_observations == 25
        assert profile.max_drawdown <= 0.0

    def test_stock_profile_reports_both_once_there_is_a_year(self) -> None:
        rng = np.random.default_rng(1)
        year = _series(list(rng.normal(0.0004, 0.01, 252)))
        profile = risk.stock_risk_profile(year)

        assert profile.sharpe is not None
        assert profile.sortino is not None
        # And it lands in a believable range rather than the demo's 19.55.
        assert abs(profile.sortino) < 5.0

    def test_portfolio_risk_applies_the_same_floor(self) -> None:
        rng = np.random.default_rng(2)
        index = pd.date_range("2026-01-01", periods=25, freq="B")
        panel = pd.DataFrame(
            {"AAA": rng.normal(0.01, 0.02, 25), "BBB": rng.normal(0.01, 0.02, 25)}, index=index
        )
        summary = risk.portfolio_risk(panel, {"AAA": 0.5, "BBB": 0.5})

        assert summary.sharpe is None
        assert summary.sortino is None

    def test_the_backtest_primitive_is_deliberately_not_gated(self) -> None:
        # The track record pairs its Sharpe with a bootstrap confidence
        # interval, which discloses a short sample directly. Gating the shared
        # primitive would silently blank that page instead.
        assert bt.sharpe_ratio(_series([0.01, -0.005, 0.02] * 3), periods_per_year=12.0) is not None
