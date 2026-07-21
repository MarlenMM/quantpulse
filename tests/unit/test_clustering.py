import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis.clustering import cluster_by_correlation, compute_return_correlation_matrix


def _two_group_prices(n: int = 250, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    # Each group shares a common daily-return factor; per-symbol noise is
    # small relative to it, so within-group correlation is strong and
    # across-group correlation is weak (independent factors).
    common_a = rng.normal(0, 0.01, n)
    common_b = rng.normal(0, 0.01, n)

    def _prices(factor: np.ndarray, noise_scale: float = 0.002) -> pd.Series:
        returns = factor + rng.normal(0, noise_scale, n)
        return pd.Series(100 * np.cumprod(1 + returns), index=idx)

    return {
        "A1": _prices(common_a),
        "A2": _prices(common_a),
        "B1": _prices(common_b),
        "B2": _prices(common_b),
    }


class TestComputeReturnCorrelationMatrix:
    def test_diagonal_is_one(self) -> None:
        corr = compute_return_correlation_matrix(_two_group_prices())
        assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    def test_symbols_within_a_group_correlate_more_than_across(self) -> None:
        corr = compute_return_correlation_matrix(_two_group_prices())
        assert corr.loc["A1", "A2"] > corr.loc["A1", "B1"]
        assert corr.loc["B1", "B2"] > corr.loc["A1", "B1"]


class TestClusterByCorrelation:
    @pytest.mark.parametrize("method", ["hierarchical", "kmeans"])
    def test_recovers_the_two_known_groups(self, method: str) -> None:
        corr = compute_return_correlation_matrix(_two_group_prices())
        labels = cluster_by_correlation(corr, n_clusters=2, method=method)

        assert labels["A1"] == labels["A2"]
        assert labels["B1"] == labels["B2"]
        assert labels["A1"] != labels["B1"]

    def test_raises_on_unknown_method(self) -> None:
        corr = compute_return_correlation_matrix(_two_group_prices())
        with pytest.raises(ValueError, match="unknown method"):
            cluster_by_correlation(corr, n_clusters=2, method="dbscan")  # type: ignore[arg-type]

    def test_raises_a_clear_error_on_nan_correlation(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = {
            "NORMAL": pd.Series(100 * np.cumprod(1 + np.full(50, 0.001)), index=idx),
            "CONSTANT": pd.Series([100.0] * 50, index=idx),  # zero variance -> NaN correlation
        }
        corr = compute_return_correlation_matrix(prices)
        with pytest.raises(ValueError, match="CONSTANT"):
            cluster_by_correlation(corr, n_clusters=2)
