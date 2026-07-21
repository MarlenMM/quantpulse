"""Correlation-based stock clustering (Section 7.1).

Groups stocks that actually move together, regardless of official GICS
sector -- e.g. two "different sector" stocks that are secretly both AI-capex
plays. This is a standard technique on an existing library (Section 21:
Sonnet/Medium), unlike the geometric chart-pattern work in `patterns.py`.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


def compute_return_correlation_matrix(prices_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise correlation of daily returns across symbols, aligned on common dates."""
    wide = pd.DataFrame(prices_by_symbol)
    returns = wide.pct_change().dropna(how="all")
    return returns.corr()


def cluster_by_correlation(
    correlation_matrix: pd.DataFrame,
    n_clusters: int,
    method: Literal["hierarchical", "kmeans"] = "hierarchical",
) -> dict[str, int]:
    """Assign each symbol in `correlation_matrix` to one of `n_clusters` groups.

    `hierarchical` (default) clusters directly on the correlation-derived
    distance (1 - correlation) via average-linkage agglomeration -- the
    natural fit for a precomputed distance/similarity matrix. `kmeans`
    clusters each symbol's row of correlations as a feature vector instead.
    """
    nan_symbols = correlation_matrix.index[correlation_matrix.isna().any(axis=1)].tolist()
    if nan_symbols:
        raise ValueError(
            f"correlation matrix has NaN entries for {nan_symbols} "
            "(likely a symbol with zero price variance in the window); drop it before clustering"
        )

    symbols = correlation_matrix.index.tolist()

    if method == "hierarchical":
        distance = (1 - correlation_matrix).to_numpy(copy=True)
        np.fill_diagonal(distance, 0.0)
        distance = np.clip(distance, 0.0, None)  # guard tiny negative floats from fp error
        condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    elif method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(
            correlation_matrix.to_numpy(copy=True)
        )
    else:
        raise ValueError(f"unknown method: {method!r}")

    return {symbol: int(label) for symbol, label in zip(symbols, labels, strict=True)}
