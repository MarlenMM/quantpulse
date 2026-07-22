"""Extended macro & cross-asset signals (Section 28).

Two independent, well-established macro overlays that extend the Tier-3 work in
Section 7.3 and feed both the Market Regime Index (`news_intelligence.
market_regime`) and Phase 6's per-stock scoring:

1. **Yield-curve inversion as a named signal.** The 10Y-2Y Treasury spread is
   one of the best-known recession-risk indicators in macro finance; Section 28
   asks for it to be computed explicitly and labeled, not folded anonymously
   into "macro indicators." A negative spread (short rates above long rates) is
   the classic inversion. This feeds the Market Regime Index.

2. **Commodity/currency overlays for the sectors that actually care.** Oil for
   Energy, gold/metals for Materials, and the US Dollar Index for the sectors
   dominated by large multinationals with significant overseas revenue. Applied
   as a *targeted* overlay only to the sectors each one is genuinely relevant
   to (Section 28's explicit warning: "not as a universal input -- a small
   biotech doesn't care about oil prices"). Every other sector gets a flat 0.0
   adjustment, so the signal never adds noise where it doesn't belong.

Both are pure functions; the ingestion of the underlying series (`^VIX`,
`CL=F`, `GC=F`, `DX-Y.NYB`, and the FRED `DGS10`/`DGS2` yields) into
`macro_indicators`, and the persistence of the resulting regime, live in the
nightly refresh, not here.
"""

# Canonical macro_indicators series names for the cross-asset tickers. Kept
# here (next to the sector-sensitivity config that consumes them) so the
# ingestion layer and this overlay agree on one spelling.
OIL_WTI = "oil_wti"  # CL=F
GOLD = "gold"  # GC=F (a free stand-in for the broader industrial-metals complex)
DOLLAR_INDEX = "dollar_index"  # DX-Y.NYB
VIX = "vix"  # ^VIX

# A commodity move of this magnitude (percent) saturates its overlay to +/-1.0.
# A round, documented scale, not an empirically fit threshold.
_COMMODITY_FULL_SWING_PCT = 10.0

# Which sectors are actually exposed to which cross-asset series, and with what
# sign. Keyed by the GICS sector names the `tickers.sector` column carries
# (Wikipedia's GICS "Sector"), matching `fundamental.py`'s config. A positive
# sign means a rise in the series is a tailwind for the sector.
#
# - Energy rises with oil; Materials rise with the metals complex (gold here as
#   the free proxy). Both direct, well-established relationships.
# - A stronger dollar (DXY up) is a *headwind* for sectors dominated by
#   multinationals earning abroad -- their foreign revenue translates back into
#   fewer dollars -- so the sign is negative. Scoped to the sectors with the
#   highest foreign-revenue share (Info Tech, Materials, Industrials, Consumer
#   Staples, Energy); left off the domestically-focused ones (Utilities, Real
#   Estate, Financials, Health Care, Consumer Discretionary, Communication
#   Services) rather than applied universally.
SECTOR_COMMODITY_SENSITIVITY: dict[str, dict[str, float]] = {
    "Energy": {OIL_WTI: 1.0, DOLLAR_INDEX: -0.5},
    "Materials": {GOLD: 1.0, DOLLAR_INDEX: -0.5},
    "Information Technology": {DOLLAR_INDEX: -1.0},
    "Industrials": {DOLLAR_INDEX: -0.5},
    "Consumer Staples": {DOLLAR_INDEX: -0.5},
}


def yield_curve_spread(dgs10: float | None, dgs2: float | None) -> float | None:
    """The 10Y-2Y Treasury spread (Section 28), or None if either yield is missing.

    Negative = an inverted curve (the classic recession-risk signal). Both
    inputs are the latest stored `DGS10` / `DGS2` FRED values, in percent, so
    the spread is in percentage points.
    """
    if dgs10 is None or dgs2 is None:
        return None
    return dgs10 - dgs2


def is_yield_curve_inverted(spread: float | None) -> bool:
    """Whether `spread` (from `yield_curve_spread`) represents an inverted curve."""
    return spread is not None and spread < 0.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def commodity_overlay_adjustment(sector: str | None, commodity_returns: dict[str, float]) -> float:
    """Directional adjustment in [-1, 1] for `sector` given recent commodity moves.

    `commodity_returns` maps a series name (`OIL_WTI`/`GOLD`/`DOLLAR_INDEX`) to
    its recent percent change. For each series the sector is configured to care
    about, contributes `sign * clip(return / full_swing, -1, 1)`; the
    contributions are summed and the total re-clipped to [-1, 1]. A sector with
    no configured sensitivity (or unknown/None sector) always returns 0.0 --
    the targeted-overlay guarantee (Section 28), so this never nudges a stock
    the overlay isn't relevant to.
    """
    sensitivities = SECTOR_COMMODITY_SENSITIVITY.get(sector or "")
    if not sensitivities:
        return 0.0

    total = 0.0
    for series_name, sign in sensitivities.items():
        ret = commodity_returns.get(series_name)
        if ret is None:
            continue
        total += sign * _clip(ret / _COMMODITY_FULL_SWING_PCT, -1.0, 1.0)

    return _clip(total, -1.0, 1.0)


def pct_change(series: list[float]) -> float | None:
    """Percent change from the first to the last point of `series`, or None if undefined.

    A small convenience for turning a stored `macro_indicators` window (oldest
    first, e.g. from `persistence.read_macro_series`) into the recent-move
    input `commodity_overlay_adjustment` expects. Returns None for an empty
    series or a non-positive first value (percent change is meaningless there).
    """
    if len(series) < 2:
        return None
    first, last = series[0], series[-1]
    if first <= 0:
        return None
    return (last - first) / first * 100.0
