"""Fundamental scoring: sector-relative percentile ranking with per-sector config (Section 7.2).

Comparing a bank's P/E to a software company's P/E is meaningless, so every
metric is percentile-ranked *within its own sector* rather than against the
whole universe. The harder point Section 7.2 insists on building in from day
one: some sectors don't just need a different peer group for the same
ratios, they need different ratios entirely. Banks run high leverage by
design (a debt/equity ratio that would alarm an industrial company is normal
for a bank); REITs are conventionally valued on FFO, not GAAP earnings,
because real-estate depreciation makes P/E misleading for them. Each sector
gets its own small weight config rather than forcing one universal formula.
"""

from dataclasses import dataclass

import pandas as pd

# Valuation multiples where the ratio is only meaningful for a positive
# denominator -- a negative P/E means "no earnings," not "cheap," and must
# not be scored as if it were. Growth/return metrics (roe, roa,
# revenue_growth) are deliberately excluded: a negative ROE is a real,
# meaningful (bad) signal, not an undefined ratio, and should rank at the bottom.
_UNDEFINED_IF_NON_POSITIVE = frozenset({"pe", "pb", "ps", "peg", "p_ffo"})

# Metrics where a LOWER raw value is the better outcome (cheap valuation,
# less leverage). Everything else configured is "higher is better."
_LOWER_IS_BETTER = frozenset({"pe", "pb", "ps", "peg", "debt_equity", "p_ffo"})

DEFAULT_SECTOR = "_default"


@dataclass(frozen=True)
class SectorFundamentalConfig:
    """Which metrics matter for a sector, and how much each counts (Section 7.2)."""

    sector: str
    weights: dict[str, float]
    notes: str = ""

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"{self.sector} weights must sum to ~1.0, got {total}")


# NOTE on free cash flow: `fundamentals_snapshot.fcf` is ingested and stored,
# but deliberately carries no weight in any config here. Raw dollar FCF is an
# absolute magnitude, so percentile-ranking it cross-sectionally would just
# rank companies by size (a mega-cap always out-"scores" a small-cap on raw
# FCF), which is a bias, not a quality signal. A meaningful FCF factor is FCF
# *yield* (FCF / market cap) or FCF margin (FCF / revenue) -- neither input is
# in the snapshot yet -- so FCF is intentionally left out of scoring rather
# than folded in as a size proxy (Section 22: don't ship a misleading signal).
# Adding market cap to the snapshot would unlock a proper FCF-yield metric.
_DEFAULT_WEIGHTS = {
    "pe": 0.20,
    "pb": 0.10,
    "ps": 0.10,
    "peg": 0.10,
    "revenue_growth": 0.15,
    "roe": 0.15,
    "roa": 0.10,
    "debt_equity": 0.05,
    "div_yield": 0.05,
}

# Financials: leverage is normal by design (banks/insurers run high
# debt/equity as part of their business model, not as a risk red flag), so
# debt_equity and price/sales (revenue isn't a comparable concept across
# banks) are dropped entirely; ROE/ROA -- the metrics that actually
# distinguish a well-run bank -- carry much more weight instead.
_FINANCIALS_WEIGHTS = {
    "pe": 0.20,
    "pb": 0.20,
    "peg": 0.05,
    "roe": 0.30,
    "roa": 0.15,
    "div_yield": 0.10,
}

# Real Estate (REITs): GAAP earnings are depressed by real-estate
# depreciation, making P/E and ROE misleading, so P/E is replaced by P/FFO
# (Price / (Net Income + D&A), the standard simplified FFO proxy -- Section
# 7.2) and ROE is dropped. Dividend yield is weighted heavily since REITs
# are required to distribute most of their taxable income.
_REAL_ESTATE_WEIGHTS = {
    "p_ffo": 0.30,
    "pb": 0.10,
    "div_yield": 0.30,
    "revenue_growth": 0.10,
    "debt_equity": 0.10,
    "roa": 0.10,
}

# Utilities: regulated, capital-intensive businesses that carry high debt by
# design (similar reasoning to Financials, but less extreme) and are
# classic income plays -- dividend yield is upweighted, growth/PEG downweighted
# since regulated-return utilities aren't priced on growth the way most
# sectors are.
_UTILITIES_WEIGHTS = {
    "pe": 0.20,
    "pb": 0.10,
    "roe": 0.15,
    "roa": 0.10,
    "div_yield": 0.25,
    "revenue_growth": 0.10,
    "debt_equity": 0.10,
}

SECTOR_CONFIGS: dict[str, SectorFundamentalConfig] = {
    DEFAULT_SECTOR: SectorFundamentalConfig(
        DEFAULT_SECTOR, _DEFAULT_WEIGHTS, "Generic cross-sector weighting."
    ),
    "Financials": SectorFundamentalConfig(
        "Financials",
        _FINANCIALS_WEIGHTS,
        "Leverage is normal for banks/insurers -- debt_equity and ps dropped; "
        "ROE/ROA upweighted as the metrics that actually differentiate them.",
    ),
    "Real Estate": SectorFundamentalConfig(
        "Real Estate",
        _REAL_ESTATE_WEIGHTS,
        "P/E replaced by P/FFO (real-estate depreciation makes GAAP earnings "
        "misleading); dividend yield upweighted (REITs must distribute most income).",
    ),
    "Utilities": SectorFundamentalConfig(
        "Utilities",
        _UTILITIES_WEIGHTS,
        "Regulated capital-intensive businesses: high leverage tolerated, "
        "dividend yield upweighted, growth/PEG downweighted.",
    ),
}


def get_sector_config(sector: str | None) -> SectorFundamentalConfig:
    """The `SectorFundamentalConfig` for `sector`, falling back to the generic default."""
    if sector is None:
        return SECTOR_CONFIGS[DEFAULT_SECTOR]
    return SECTOR_CONFIGS.get(sector, SECTOR_CONFIGS[DEFAULT_SECTOR])


def compute_p_ffo(
    market_cap: float | None,
    net_income: float | None,
    depreciation_amortization: float | None,
) -> float | None:
    """Price/FFO, the standard REIT valuation multiple in place of P/E (Section 7.2).

    FFO = Net Income + Depreciation & Amortization (the simplified NAREIT
    approximation). Returns None if any input is missing or FFO is
    non-positive -- same "undefined, not just a low score" treatment as a
    negative P/E.
    """
    if market_cap is None or net_income is None or depreciation_amortization is None:
        return None
    ffo = net_income + depreciation_amortization
    if ffo <= 0:
        return None
    return market_cap / ffo


def _clean_metric_series(values: pd.Series, metric: str) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce")
    if metric in _UNDEFINED_IF_NON_POSITIVE:
        cleaned = cleaned.where(cleaned > 0)
    return cleaned


def _score_sector_group(group: pd.DataFrame, config: SectorFundamentalConfig) -> pd.DataFrame:
    weights_present = {m: w for m, w in config.weights.items() if m in group.columns}
    percentiles = pd.DataFrame(index=group.index)

    for metric, _weight in weights_present.items():
        cleaned = _clean_metric_series(group[metric], metric)
        ascending = metric not in _LOWER_IS_BETTER
        # min_count-style behavior: a sector with only 1 usable value can't be
        # ranked against anyone, so rank() correctly gives it 100% alone --
        # that's fine, a lone data point just can't be discriminated further.
        percentiles[metric] = cleaned.rank(pct=True, ascending=ascending) * 100

    weight_series = pd.Series(weights_present)
    available_weight = percentiles.notna().mul(weight_series, axis=1).sum(axis=1)
    weighted_sum = percentiles.fillna(0).mul(weight_series, axis=1).sum(axis=1)

    total_configured_weight = sum(weights_present.values()) if weights_present else 0.0
    score = weighted_sum / available_weight.replace(0, pd.NA)
    coverage = (
        available_weight / total_configured_weight
        if total_configured_weight
        else available_weight * 0
    )

    return pd.DataFrame(
        {
            "symbol": group["symbol"],
            "sector": group["sector"],
            "fundamental_score": score,
            "coverage": coverage,
        },
        index=group.index,
    )


def score_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Sector-relative fundamental scores (Section 7.2), one row per symbol.

    `fundamentals` must have `symbol` and `sector` columns plus zero or more
    of the configured metric columns (pe, pb, ps, peg, revenue_growth,
    debt_equity, roe, roa, div_yield, p_ffo, ...) -- missing columns or
    missing/undefined values are fine. Ranking happens within each sector
    group (rows with an unconfigured sector fall back to `DEFAULT_SECTOR`'s
    weights), so a stock is only ever compared to true peers.

    `fundamental_score` (0-100) is renormalized over whichever configured
    metrics actually had usable data for that row -- a stock isn't punished
    for one missing data point. `coverage` (0-1), the fraction of the
    sector's *configured weight* that had usable data, is returned alongside
    it so a later phase can render the two separately (Section 7.5's
    data-completeness idea: a thinly-covered score should not be displayed
    with the same visual confidence as a fully-covered one).
    """
    if "symbol" not in fundamentals.columns or "sector" not in fundamentals.columns:
        raise ValueError("fundamentals must have 'symbol' and 'sector' columns")

    parts: list[pd.DataFrame] = []
    for sector, group in fundamentals.groupby("sector", dropna=False):
        config = get_sector_config(sector if isinstance(sector, str) else None)
        parts.append(_score_sector_group(group, config))

    if not parts:
        return pd.DataFrame(columns=["symbol", "sector", "fundamental_score", "coverage"])

    return pd.concat(parts).sort_index().reset_index(drop=True)
