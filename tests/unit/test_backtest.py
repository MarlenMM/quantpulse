import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import backtest as bt
from quantpulse.analysis.forecasting import Forecast, baseline_forecast
from quantpulse.portfolio.optimization import kelly_position_fraction


def _prices(closes: list[float] | np.ndarray, start: str = "2021-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B")
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1_000_000.0},
        index=idx,
    )


def _panel(series_by_symbol: dict[str, list[float]], start: str = "2021-01-01") -> pd.DataFrame:
    n = len(next(iter(series_by_symbol.values())))
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(series_by_symbol, index=idx, dtype=float)


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #


class TestMetrics:
    def test_sharpe_zero_mean_is_zero(self) -> None:
        s = bt.sharpe_ratio(pd.Series([0.1, -0.1, 0.1, -0.1]), periods_per_year=12)
        assert s == pytest.approx(0.0)

    def test_sharpe_constant_returns_is_none(self) -> None:
        assert bt.sharpe_ratio(pd.Series([0.01, 0.01, 0.01]), periods_per_year=12) is None

    def test_sharpe_too_short_is_none(self) -> None:
        assert bt.sharpe_ratio(pd.Series([0.01]), periods_per_year=12) is None

    def test_cagr_doubling_in_one_year(self) -> None:
        assert bt.cagr(pd.Series([1.0]), periods_per_year=1) == pytest.approx(1.0)

    def test_cagr_flat_is_zero(self) -> None:
        assert bt.cagr(pd.Series([0.0] * 12), periods_per_year=12) == pytest.approx(0.0)

    def test_cagr_wipeout_is_none(self) -> None:
        assert bt.cagr(pd.Series([-1.0, 0.5]), periods_per_year=12) is None

    def test_max_drawdown_hand_check(self) -> None:
        # equity 1.1 -> 0.55 -> 0.605; worst = 0.55/1.1 - 1 = -0.5
        assert bt.max_drawdown(pd.Series([0.1, -0.5, 0.1])) == pytest.approx(-0.5)

    def test_max_drawdown_monotonic_up_is_zero(self) -> None:
        assert bt.max_drawdown(pd.Series([0.1, 0.1, 0.1])) == pytest.approx(0.0)

    def test_directional_hit_rate_hand_check(self) -> None:
        # flat actual (0) is dropped; (1,1) hit, (-1,1) miss, (1,-1) miss -> 1/3
        rate = bt.directional_hit_rate([1.0, -1.0, 1.0, 0.5], [1.0, 1.0, -1.0, 0.0])
        assert rate == pytest.approx(1 / 3)

    def test_directional_hit_rate_no_gradable_pairs_is_none(self) -> None:
        assert bt.directional_hit_rate([0.0, 0.0], [0.0, 0.0]) is None
        assert bt.directional_hit_rate([], []) is None

    def test_rmse_hand_check(self) -> None:
        assert bt.rmse([1.0, 2.0], [0.0, 0.0]) == pytest.approx(np.sqrt(2.5))


# --------------------------------------------------------------------------- #
# Walk-forward forecast accuracy -- look-ahead is the whole point
# --------------------------------------------------------------------------- #


def _fixed_forecast(point_return: float) -> Forecast:
    return Forecast(
        model_name="spy",
        horizon_days=20,
        as_of=pd.Timestamp("2021-01-01"),
        last_close=100.0,
        point_return=point_return,
        lower_return=point_return,
        upper_return=point_return,
        point_price=100.0,
        lower_price=100.0,
        upper_price=100.0,
        confidence_level=0.9,
        n_train=1,
    )


class TestWalkForwardAccuracy:
    def test_training_slice_never_reaches_the_graded_future(self) -> None:
        # The core anti-look-ahead guarantee: no training slice handed to the
        # model may extend into the last `horizon` bars, because those are the
        # outcomes being graded. Capture every slice length and assert it.
        prices = _prices(list(np.linspace(100, 200, 300)))
        horizon = 20
        seen_lengths: list[int] = []

        def spy_model(train: pd.DataFrame, h: int) -> Forecast:
            seen_lengths.append(len(train))
            return _fixed_forecast(0.01)

        bt.walk_forward_accuracy(prices, model_fn=spy_model, horizon_days=horizon, model_name="spy")
        assert seen_lengths  # folds ran
        assert max(seen_lengths) <= len(prices) - horizon

    def test_perfect_model_scores_full_hit_rate(self) -> None:
        prices = _prices(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 300))))
        closes = prices["close"]

        def oracle(train: pd.DataFrame, h: int) -> Forecast:
            # "Cheats" using the full series it can see via closure -- proves the
            # metric rewards correct direction; the engine still only *hands* it
            # the training slice.
            i = len(train) - 1
            realized = float(closes.iloc[i + h] / closes.iloc[i] - 1.0)
            return _fixed_forecast(realized)

        result = bt.walk_forward_accuracy(prices, model_fn=oracle, horizon_days=20, model_name="o")
        assert result is not None and result.hit_rate == pytest.approx(1.0)

    def test_inverse_model_scores_zero_hit_rate(self) -> None:
        prices = _prices(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 300))))
        closes = prices["close"]

        def anti(train: pd.DataFrame, h: int) -> Forecast:
            i = len(train) - 1
            realized = float(closes.iloc[i + h] / closes.iloc[i] - 1.0)
            return _fixed_forecast(-realized)  # always wrong sign

        result = bt.walk_forward_accuracy(prices, model_fn=anti, horizon_days=20, model_name="a")
        assert result is not None and result.hit_rate == pytest.approx(0.0)

    def test_every_graded_pair_carries_its_evaluation_date(self) -> None:
        # Pooling a hit rate across symbols multiplies the pair count without
        # adding windows -- twenty stocks over the same three years is three
        # pieces of evidence. Counting the distinct windows is only possible if
        # the engine says which one each pair came from.
        prices = _prices(100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.02, 300))))
        result = bt.walk_forward_accuracy(
            prices, model_fn=lambda p, h: _fixed_forecast(0.01), horizon_days=20, model_name="c"
        )
        assert result is not None
        assert len(result.as_of) == result.n == len(result.predicted)
        # Dates are the as-of bar of each fold: strictly increasing, all real
        # index values, and none inside the graded future.
        assert list(result.as_of) == sorted(result.as_of)
        assert len(set(result.as_of)) == len(result.as_of)
        assert set(result.as_of).issubset(set(prices.index))
        assert max(result.as_of) <= prices.index[-1 - 20]

    def test_a_model_that_abstains_contributes_no_window(self) -> None:
        # A model declining a fold (too little history for the horizon) must not
        # leave a date behind it, or the window count over-reports the evidence.
        prices = _prices(100 * np.exp(np.cumsum(np.random.default_rng(6).normal(0, 0.02, 300))))
        calls = {"n": 0}

        def sometimes(train: pd.DataFrame, h: int) -> Forecast | None:
            calls["n"] += 1
            return _fixed_forecast(0.01) if calls["n"] % 2 == 0 else None

        result = bt.walk_forward_accuracy(
            prices, model_fn=sometimes, horizon_days=20, model_name="half"
        )
        assert result is not None
        assert len(result.as_of) == result.n
        assert result.n * 2 <= calls["n"] + 1  # roughly half the folds produced nothing

    def test_reports_baseline_and_sample_size(self) -> None:
        prices = _prices(
            100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0005, 0.02, 300)))
        )
        result = bt.walk_forward_accuracy(
            prices, model_fn=lambda p, h: _fixed_forecast(0.01), horizon_days=20, model_name="c"
        )
        assert result is not None
        assert result.n == result.predicted.size
        assert result.baseline_hit_rate is not None  # baseline ran alongside

    def test_step_controls_fold_count(self) -> None:
        prices = _prices(list(np.linspace(100, 160, 300)))
        calls = {"n": 0}

        def counter(train: pd.DataFrame, h: int) -> Forecast:
            calls["n"] += 1
            return _fixed_forecast(0.01)

        bt.walk_forward_accuracy(prices, model_fn=counter, horizon_days=20, model_name="s", step=40)
        wide = calls["n"]
        calls["n"] = 0
        bt.walk_forward_accuracy(prices, model_fn=counter, horizon_days=20, model_name="s", step=10)
        assert calls["n"] > wide  # a smaller step evaluates more folds

    def test_too_short_is_none(self) -> None:
        assert (
            bt.walk_forward_accuracy(
                _prices(list(np.linspace(100, 110, 70))),
                model_fn=lambda p, h: _fixed_forecast(0.0),
                horizon_days=20,
                model_name="x",
            )
            is None
        )

    def test_real_baseline_model_runs(self) -> None:
        prices = _prices(
            100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0004, 0.02, 300)))
        )
        result = bt.walk_forward_accuracy(
            prices, model_fn=baseline_forecast, horizon_days=20, model_name="baseline"
        )
        assert result is not None and 0.0 <= (result.hit_rate or 0.0) <= 1.0


# --------------------------------------------------------------------------- #
# rebalance_dates
# --------------------------------------------------------------------------- #


class TestRebalanceDates:
    def test_monthly_lands_on_real_trading_days_one_per_month(self) -> None:
        idx = pd.date_range("2022-01-01", "2022-06-30", freq="B")
        dates = bt.rebalance_dates(idx, "monthly")
        assert len(dates) == 6  # Jan..Jun
        assert all(d in set(idx) for d in dates)  # never a weekend/holiday off-index

    def test_weekly_has_more_periods_than_monthly(self) -> None:
        idx = pd.date_range("2022-01-01", "2022-06-30", freq="B")
        assert len(bt.rebalance_dates(idx, "weekly")) > len(bt.rebalance_dates(idx, "monthly"))

    def test_unknown_cadence_raises(self) -> None:
        with pytest.raises(ValueError, match="cadence"):
            bt.rebalance_dates(pd.date_range("2022-01-01", periods=10, freq="B"), "daily")


# --------------------------------------------------------------------------- #
# Strategy backtest -- look-ahead, survivorship, and cost guarantees
# --------------------------------------------------------------------------- #


def _rank_by_last_price(as_of, panel: pd.DataFrame) -> dict[str, float]:
    """A point-in-time signal: rank by the most recent price in the visible slice."""
    last = panel.iloc[-1]
    return {s: float(last[s]) for s in panel.columns if pd.notna(last[s])}


class TestStrategyBacktest:
    def test_signal_never_sees_the_future(self) -> None:
        # The signal must only ever be handed prices dated <= the rebalance date.
        panel = _panel(
            {"A": list(np.linspace(100, 200, 400)), "B": list(np.linspace(200, 100, 400))}
        )
        observed: list[tuple] = []

        def spy_signal(as_of, visible: pd.DataFrame) -> dict[str, float]:
            observed.append((as_of, visible.index.max().date()))
            return _rank_by_last_price(as_of, visible)

        bt.backtest_strategy(panel, signal_fn=spy_signal, cadence="monthly")
        assert observed
        assert all(visible_max <= as_of for as_of, visible_max in observed)

    def test_higher_cost_never_helps(self) -> None:
        rng = np.random.default_rng(5)
        panel = _panel(
            {
                f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 500))))
                for i in range(6)
            }
        )
        free = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, transaction_cost=0.0)
        pricey = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, transaction_cost=0.05)
        assert free is not None and pricey is not None
        assert (free.cagr or 0) >= (pricey.cagr or 0)
        assert free.assumed_txn_cost == 0.0 and pricey.assumed_txn_cost == 0.05

    def test_ineligible_names_are_never_held(self) -> None:
        # B always has the stronger signal, but is never eligible; the strategy
        # must ride A (flat) instead of B (rising) -> restricting eligibility
        # changes (lowers) the return, proving B was excluded from holdings.
        panel = _panel({"A": [100.0] * 300, "B": list(np.linspace(100, 300, 300))})
        both = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, top_fraction=0.5)
        only_a = bt.backtest_strategy(
            panel, signal_fn=_rank_by_last_price, top_fraction=0.5, eligible=lambda d: {"A"}
        )
        assert both is not None and only_a is not None
        assert (only_a.cagr or 0) < (both.cagr or 0)

    def test_delisted_holding_does_not_crash(self) -> None:
        # C stops trading (NaN) after the first third; a survivorship-honest run
        # must handle the held-but-delisted name without raising.
        closes = np.linspace(100, 130, 300)
        with_gap = closes.copy()
        with_gap[100:] = np.nan
        panel = _panel({"C": list(with_gap), "D": list(np.linspace(100, 90, 300))})
        result = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, top_fraction=0.5)
        assert result is not None  # produced a track record, no crash

    def test_deterministic_winner_selection(self) -> None:
        # WIN rises and is always the higher-priced name, so a price-rank signal
        # always picks it; with a single rising holding, win_rate is perfect.
        panel = _panel(
            {"WIN": list(np.linspace(100, 200, 300)), "LOSE": list(np.linspace(90, 45, 300))}
        )
        result = bt.backtest_strategy(
            panel, signal_fn=_rank_by_last_price, top_fraction=0.5, transaction_cost=0.0
        )
        assert result is not None
        assert result.cagr is not None and result.cagr > 0
        assert result.win_rate == pytest.approx(1.0)

    def test_benchmark_metrics_are_computed(self) -> None:
        rng = np.random.default_rng(7)
        panel = _panel(
            {
                f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 400))))
                for i in range(5)
            }
        )
        bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400))), index=panel.index)
        result = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, benchmark=bench)
        assert result is not None
        assert result.benchmark_cagr is not None and result.benchmark_sharpe is not None
        assert not result.benchmark_return.empty

    def test_period_returns_exposed_for_bootstrap(self) -> None:
        panel = _panel({f"S{i}": list(np.linspace(100, 120 + i, 300)) for i in range(4)})
        result = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price)
        assert result is not None
        assert isinstance(result.period_returns, pd.Series)
        assert len(result.period_returns) == result.n_periods

    def test_too_few_periods_is_none(self) -> None:
        panel = _panel({"A": [100.0, 101.0, 102.0]})  # < 2 monthly rebalances
        assert bt.backtest_strategy(panel, signal_fn=_rank_by_last_price) is None

    def test_invalid_params_raise(self) -> None:
        panel = _panel({"A": list(np.linspace(100, 120, 300))})
        with pytest.raises(ValueError, match="top_fraction"):
            bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, top_fraction=1.5)
        with pytest.raises(ValueError, match="transaction_cost"):
            bt.backtest_strategy(panel, signal_fn=_rank_by_last_price, transaction_cost=-0.1)


# --------------------------------------------------------------------------- #
# Bootstrap significance testing -- preserving time-ordering is the whole point
# --------------------------------------------------------------------------- #


def _ar1_returns(n: int, phi: float, *, mean: float = 0.004, sigma: float = 0.02, seed: int = 0):
    """A serially-correlated (AR(1)) return series -- the case an i.i.d. bootstrap gets wrong."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return pd.Series(out + mean)


class TestBlockBootstrapMechanics:
    def test_point_estimate_is_the_observed_statistic_not_a_resample_average(self) -> None:
        returns = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 60))
        ci = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=0)
        assert ci is not None
        assert ci.point == pytest.approx(bt.sharpe_ratio(returns, periods_per_year=12))

    def test_interval_brackets_the_point_estimate(self) -> None:
        returns = pd.Series(np.random.default_rng(1).normal(0.01, 0.02, 80))
        ci = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=0)
        assert ci is not None
        assert ci.low <= ci.point <= ci.high

    def test_block_indices_preserve_within_block_order(self) -> None:
        rng = np.random.default_rng(0)
        idx = bt._block_indices(40, block_size=5, rng=rng)
        assert len(idx) == 40
        # Every full block is a run of consecutive positions, in order.
        for start in range(0, 40 - 5, 5):
            block = idx[start : start + 5]
            assert list(block) == list(range(block[0], block[0] + 5))

    def test_block_indices_stay_in_range(self) -> None:
        rng = np.random.default_rng(0)
        idx = bt._block_indices(37, block_size=6, rng=rng)
        assert idx.min() >= 0 and idx.max() < 37

    def test_default_block_size_grows_with_n_and_is_capped(self) -> None:
        assert bt._default_block_size(2) == 1
        assert bt._default_block_size(10) == 2
        assert bt._default_block_size(60) == 4
        # Never long enough to make every resample the original series.
        for n in (8, 20, 100, 500):
            assert bt._default_block_size(n) <= n // 2

    def test_wider_confidence_level_gives_a_wider_interval(self) -> None:
        returns = pd.Series(np.random.default_rng(2).normal(0.01, 0.02, 100))
        narrow = bt.bootstrap_sharpe_ci(
            returns, periods_per_year=12, confidence_level=0.50, random_state=0
        )
        wide = bt.bootstrap_sharpe_ci(
            returns, periods_per_year=12, confidence_level=0.99, random_state=0
        )
        assert narrow is not None and wide is not None
        assert (wide.high - wide.low) > (narrow.high - narrow.low)

    def test_deterministic_with_fixed_seed(self) -> None:
        returns = pd.Series(np.random.default_rng(3).normal(0.01, 0.02, 60))
        a = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=7)
        b = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=7)
        assert a is not None and b is not None
        assert (a.low, a.high) == (b.low, b.high)


class TestBlockBootstrapVersusIID:
    def test_blocks_widen_the_interval_on_autocorrelated_returns(self) -> None:
        # The core claim of this whole row (Section 21): resampling single
        # observations destroys serial dependence and understates uncertainty.
        # With strong positive autocorrelation the honest block interval must be
        # materially wider than the naive i.i.d. one.
        returns = _ar1_returns(240, phi=0.6, seed=0)
        iid = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, block_size=1, random_state=1)
        block = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=1)
        assert iid is not None and block is not None
        assert (block.high - block.low) > 1.3 * (iid.high - iid.low)

    def test_iid_bootstrap_can_falsely_claim_significance(self) -> None:
        # The concrete, silent, flattering failure: on this autocorrelated
        # series the i.i.d. interval excludes zero (reads as "a real edge")
        # while the block interval straddles it (honestly "not distinguished
        # from luck"). Same data, same point estimate -- only the resampling
        # scheme differs.
        returns = _ar1_returns(240, phi=0.6, seed=0)
        iid = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, block_size=1, random_state=1)
        block = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, random_state=1)
        assert iid is not None and block is not None
        assert iid.point == pytest.approx(block.point)  # identical headline number
        assert iid.excludes_zero
        assert not block.excludes_zero

    def test_block_size_one_is_the_iid_bootstrap(self) -> None:
        returns = pd.Series(np.random.default_rng(4).normal(0.01, 0.02, 60))
        ci = bt.bootstrap_sharpe_ci(returns, periods_per_year=12, block_size=1, random_state=0)
        assert ci is not None and ci.block_size == 1


class TestExcludesZero:
    def test_true_when_wholly_positive(self) -> None:
        strong = pd.Series(np.random.default_rng(5).normal(0.03, 0.01, 60))
        ci = bt.bootstrap_sharpe_ci(strong, periods_per_year=12, random_state=0)
        assert ci is not None and ci.excludes_zero

    def test_false_when_straddling_zero(self) -> None:
        noise = pd.Series(np.random.default_rng(6).normal(0.0, 0.05, 60))
        ci = bt.bootstrap_sharpe_ci(noise, periods_per_year=12, random_state=0)
        assert ci is not None and not ci.excludes_zero


class TestBootstrapAbstention:
    def test_too_few_observations_gives_no_interval(self) -> None:
        # A CI from a handful of points is theatre; abstain rather than invent.
        assert bt.bootstrap_sharpe_ci(pd.Series([0.01, -0.02, 0.03]), periods_per_year=12) is None

    def test_constant_series_has_no_defined_sharpe_or_interval(self) -> None:
        # Constant returns differ by floating-point noise, not real variation:
        # the Sharpe must be undefined rather than ~1e16 with a razor-thin CI.
        constant = pd.Series([0.01] * 30)
        assert bt.sharpe_ratio(constant, periods_per_year=12) is None
        assert bt.bootstrap_sharpe_ci(constant, periods_per_year=12) is None

    def test_genuine_tiny_variation_still_computes(self) -> None:
        # The degenerate-std guard is relative, so real (if small) dispersion is
        # still scored rather than swallowed.
        varied = pd.Series([0.01, 0.0100001] * 15)
        assert bt.sharpe_ratio(varied, periods_per_year=12) is not None

    def test_invalid_parameters_raise(self) -> None:
        returns = pd.Series(np.random.default_rng(7).normal(0.01, 0.02, 60))
        with pytest.raises(ValueError, match="confidence_level"):
            bt.bootstrap_sharpe_ci(returns, periods_per_year=12, confidence_level=1.5)
        with pytest.raises(ValueError, match="n_resamples"):
            bt.bootstrap_sharpe_ci(returns, periods_per_year=12, n_resamples=0)
        with pytest.raises(ValueError, match="block_size"):
            bt.bootstrap_sharpe_ci(returns, periods_per_year=12, block_size=0)


class TestBootstrapHitRate:
    def test_pairing_is_preserved_across_resamples(self) -> None:
        # A perfect model stays perfect in every resample only if each
        # prediction keeps its own realized outcome. Independent resampling of
        # the two arrays would drag this toward 0.5.
        predicted = np.array([1.0, -1.0] * 30)
        realized = predicted.copy()
        ci = bt.bootstrap_hit_rate_ci(predicted, realized, random_state=0)
        assert ci is not None
        assert ci.low == pytest.approx(1.0) and ci.high == pytest.approx(1.0)

    def test_coin_flip_model_interval_brackets_one_half(self) -> None:
        rng = np.random.default_rng(8)
        ci = bt.bootstrap_hit_rate_ci(rng.normal(0, 1, 200), rng.normal(0, 1, 200), random_state=0)
        assert ci is not None
        assert ci.low <= 0.5 <= ci.high

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            bt.bootstrap_hit_rate_ci([1.0, 2.0], [1.0])

    def test_consumes_walk_forward_accuracy_output(self) -> None:
        prices = _prices(
            100 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0005, 0.02, 600)))
        )
        accuracy = bt.walk_forward_accuracy(
            prices, model_fn=baseline_forecast, horizon_days=5, model_name="baseline", step=5
        )
        assert accuracy is not None
        ci = bt.bootstrap_hit_rate_ci(accuracy.predicted, accuracy.realized, random_state=0)
        assert ci is not None
        assert 0.0 <= ci.low <= ci.high <= 1.0
        assert ci.point == pytest.approx(accuracy.hit_rate)


class TestStrategySignificance:
    def _result(self) -> bt.StrategyResult:
        rng = np.random.default_rng(10)
        panel = _panel(
            {
                f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 900))))
                for i in range(6)
            }
        )
        result = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price)
        assert result is not None
        return result

    def test_brackets_the_runs_headline_metrics(self) -> None:
        result = self._result()
        significance = bt.bootstrap_strategy_significance(result)
        assert significance.sharpe is not None and significance.cagr is not None
        assert significance.sharpe.point == pytest.approx(result.sharpe)
        assert significance.cagr.point == pytest.approx(result.cagr)

    def test_uses_the_runs_own_annualization(self) -> None:
        result = self._result()
        significance = bt.bootstrap_strategy_significance(result)
        assert significance.sharpe is not None
        assert significance.sharpe.n_observations == result.n_periods

    def test_short_run_yields_no_interval_rather_than_a_fake_one(self) -> None:
        panel = _panel({f"S{i}": list(np.linspace(100, 130 + i, 120)) for i in range(3)})
        result = bt.backtest_strategy(panel, signal_fn=_rank_by_last_price)
        assert result is not None and result.n_periods < 8
        significance = bt.bootstrap_strategy_significance(result)
        assert significance.sharpe is None and significance.cagr is None


class TestPayoffRatio:
    """The "b" in Kelly: what a win pays relative to what a loss costs.

    Exists because `optimization.kelly_position_fraction` needs it and its own
    docstring points at this module for it -- but nothing measured it, which is
    why that Section 27 function had no caller at all.
    """

    def test_ratio_of_mean_win_to_mean_loss(self) -> None:
        # wins mean +0.06, losses mean -0.02 -> ratio 3.0
        returns = pd.Series([0.04, 0.08, -0.01, -0.03])
        assert bt.payoff_ratio(returns) == pytest.approx(3.0)

    def test_none_without_any_loss(self) -> None:
        # No losses makes the ratio infinite, not large -- and an infinite payoff
        # ratio would feed Kelly a bet it believes cannot lose.
        assert bt.payoff_ratio(pd.Series([0.01, 0.02, 0.03])) is None

    def test_none_without_any_win(self) -> None:
        assert bt.payoff_ratio(pd.Series([-0.01, -0.02])) is None

    def test_flat_periods_count_as_neither(self) -> None:
        with_flat = pd.Series([0.04, 0.08, 0.0, 0.0, -0.01, -0.03])
        without = pd.Series([0.04, 0.08, -0.01, -0.03])
        assert bt.payoff_ratio(with_flat) == pytest.approx(bt.payoff_ratio(without))

    def test_empty_and_all_nan_are_none(self) -> None:
        assert bt.payoff_ratio(pd.Series(dtype=float)) is None
        assert bt.payoff_ratio(pd.Series([float("nan"), float("nan")])) is None

    def test_strategy_result_carries_it(self) -> None:
        # It must reach the dataclass the refresh job persists from, not just
        # exist as a loose helper.
        index = pd.date_range("2021-01-01", periods=260, freq="B")
        panel = pd.DataFrame(
            {
                "AAA": np.linspace(100, 180, len(index)),
                "BBB": np.linspace(100, 120, len(index)),
                "CCC": np.linspace(100, 90, len(index)),
            },
            index=index,
        )
        result = bt.backtest_strategy(panel, signal_fn=lambda as_of, hist: {"AAA": 1.0, "BBB": 0.5})
        assert result is not None
        assert result.payoff_ratio is None or result.payoff_ratio > 0


class TestMostlyCashRuns:
    """A run that sat in cash is not a track record, even if it traded once.

    Found on real data: three years of prices with an index-membership history
    only a few days deep made `eligible()` return an empty universe for 38 of 39
    monthly periods. The strategy held cash throughout and took one position at
    the very end -- and the run stored Sharpe 0.555 with a 0.99% CAGR against a
    29.3% benchmark, which reads as "we underperformed" rather than "nothing was
    tested". `avg_turnover` was 1/39, so the existing "never traded" guard
    passed it.
    """

    @staticmethod
    def _panel(n_days: int = 400) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
        rng = np.random.default_rng(11)
        data = {
            sym: 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, n_days)))
            for sym in ("AAA", "BBB", "CCC", "DDD", "EEE")
        }
        return pd.DataFrame(data, index=idx)

    def test_cash_periods_are_counted_separately_from_all_periods(self) -> None:
        panel = self._panel()
        schedule = bt.rebalance_dates(panel.index, "monthly")
        # Eligible only in the final period -- exactly the shallow-membership shape.
        last = schedule[-2].date()

        result = bt.backtest_strategy(
            panel,
            signal_fn=lambda as_of, hist: dict.fromkeys(panel.columns, 1.0),
            eligible=lambda as_of: set(panel.columns) if as_of >= last else set(),
            schedule=schedule,
        )
        assert result is not None
        assert result.n_periods > 10
        assert result.invested_periods == 1
        assert result.invested_fraction == pytest.approx(1 / result.n_periods)
        # The trap itself: a long run of exact-zero cash periods has no variance,
        # so Sharpe comes out looking perfectly respectable.
        assert result.sharpe is not None
        assert result.avg_turnover > 0  # the old "never traded" guard passes

    def test_a_fully_invested_run_reports_every_period_invested(self) -> None:
        panel = self._panel()
        result = bt.backtest_strategy(
            panel, signal_fn=lambda as_of, hist: dict.fromkeys(panel.columns, 1.0)
        )
        assert result is not None
        assert result.invested_periods == result.n_periods
        assert result.invested_fraction == 1.0

    def test_invested_fraction_is_zero_when_nothing_was_ever_held(self) -> None:
        panel = self._panel()
        result = bt.backtest_strategy(panel, signal_fn=lambda as_of, hist: {})
        assert result is not None
        assert result.invested_periods == 0
        assert result.invested_fraction == 0.0


class TestExcessReturnKellyInputs:
    """Win rate and payoff against the benchmark — what a position size needs.

    Kelly maximizes the growth rate of whatever is being bet, and an active
    strategy bets the *tilt away from* the benchmark: holding the benchmark was
    free. Sized on absolute returns, the Track Record page recommended betting
    6.2% of capital on a run whose CAGR trailed its own buy-and-hold benchmark
    (20.8% vs 28.6%) — it was measuring the market's return and reporting it as
    the strategy's edge.
    """

    @staticmethod
    def _panel(strategy_leg: list[float], other_leg: list[float]) -> pd.DataFrame:
        """Two names, so top_fraction=0.5 picks exactly the first one."""
        dates = pd.date_range("2024-01-31", periods=len(strategy_leg), freq="ME")
        return pd.DataFrame({"WIN": strategy_leg, "LOSE": other_leg}, index=dates)

    def test_excess_metrics_are_measured_against_the_benchmark(self) -> None:
        # A strategy that rises 1%/period against a benchmark rising 3%/period:
        # every period is a *win* in absolute terms and a *loss* against the
        # benchmark. The two pairs of numbers must disagree, or the excess
        # measure is not measuring anything the absolute one does not.
        periods = 12
        panel = self._panel(
            [100.0 * 1.01**i for i in range(periods)],
            [100.0 * 1.01**i for i in range(periods)],
        )
        benchmark = pd.Series(
            [100.0 * 1.03**i for i in range(periods)], index=panel.index, dtype=float
        )
        result = bt.backtest_strategy(
            panel,
            signal_fn=lambda as_of, p: {"WIN": 1.0, "LOSE": 0.0},
            cadence="monthly",
            top_fraction=0.5,
            transaction_cost=0.0,
            benchmark=benchmark,
        )
        assert result is not None
        assert result.win_rate == pytest.approx(1.0), "every period was up in absolute terms"
        assert result.excess_win_rate == pytest.approx(0.0), (
            "no period beat the benchmark, so the excess win rate must be zero -- "
            "if it tracks the absolute one, Kelly is being fed the market's return"
        )

    def test_a_strategy_that_trails_its_benchmark_gets_no_position(self) -> None:
        """The regression, end to end: absolute Kelly says bet, excess Kelly says don't.

        The strategy alternates +3% and -1%, so it has genuine losing periods and
        a well-defined absolute payoff ratio (the all-up fixture above has none,
        and Kelly correctly declines to size a bet that never loses). Its ~1%
        per period still trails a steady 2% benchmark.
        """
        periods = 13
        leg = [100.0]
        for i in range(periods - 1):
            leg.append(leg[-1] * (1.03 if i % 2 == 0 else 0.99))
        panel = self._panel(leg, list(leg))
        benchmark = pd.Series(
            [100.0 * 1.02**i for i in range(periods)], index=panel.index, dtype=float
        )
        result = bt.backtest_strategy(
            panel,
            signal_fn=lambda as_of, p: {"WIN": 1.0, "LOSE": 0.0},
            cadence="monthly",
            top_fraction=0.5,
            transaction_cost=0.0,
            benchmark=benchmark,
        )
        assert result is not None
        absolute = kelly_position_fraction(result.win_rate, result.payoff_ratio)
        excess = kelly_position_fraction(result.excess_win_rate, result.excess_payoff_ratio)
        assert absolute is not None and absolute > 0, (
            "the old basis has to actually recommend a bet here, or this test proves nothing"
        )
        assert excess is None or excess <= 0, (
            f"a strategy that lost to its benchmark in every single period was sized at "
            f"{excess} -- Kelly is still being fed absolute returns"
        )

    def test_excess_metrics_are_none_without_a_benchmark(self) -> None:
        """No benchmark, no excess: `None`, never a silent comparison against zero."""
        panel = self._panel([100.0 * 1.01**i for i in range(12)], [100.0] * 12)
        result = bt.backtest_strategy(
            panel,
            signal_fn=lambda as_of, p: {"WIN": 1.0, "LOSE": 0.0},
            cadence="monthly",
            top_fraction=0.5,
            transaction_cost=0.0,
        )
        assert result is not None
        assert result.excess_win_rate is None
        assert result.excess_payoff_ratio is None
        assert result.win_rate is not None, "the absolute measures still work without one"
