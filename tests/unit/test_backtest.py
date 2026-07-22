import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import backtest as bt
from quantpulse.analysis.forecasting import Forecast, baseline_forecast


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
