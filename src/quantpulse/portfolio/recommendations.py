"""Portfolio recommendation logic (Section 9 "Recommendations & rebalancing").

Section 21's row is titled "Rebalancing recommendation logic," but its four
underlying deliverables are Section 9's whole "Recommendations & rebalancing"
subsection, not just the rebalance pointer at the end of it -- no other
Section-21 row covers per-holding Add/Trim/Hold/Sell, concentration warnings,
or sector-gap analysis, and this is the last Phase 8 row, so leaving them out
would mean Section 9's own explicit bullets never get built. "Rule-based once
the analytics exist" fits exactly: every number this module needs already
exists after parts 1-4 (Phase 6's composite ratings, `risk.py`'s statistics,
`optimization.py`'s target weights, `rebalancing.py`'s trade list,
`transactions.py`'s positions/holding-period) -- this row's only job is
combining them into deterministic guidance, not computing anything new.

Deliberately kept in its own file rather than folded into `rebalancing.py`
(pure arithmetic once a target exists -- a different responsibility) or a
future `analytics.py` (Section 9's *quantitative* portfolio rollups --
beta/correlation/VaR/HHI as numbers, not judgments about them).

Four pieces, each a thin, testable rule over already-computed inputs:

1. **Per-holding action** (`holding_recommendation`): a straight remap of
   Phase 6's 5-tier composite rating onto Section 9's Add/Trim/Hold/Sell
   vocabulary -- except an "Add" is downgraded to "Hold" when the position is
   already at or above the concentration threshold, so this module never
   emits an "Add" for a symbol its own concentration check would also flag as
   overweight (a coherence guarantee, not a new statistic).
2. **Concentration** (`herfindahl_index` + `concentration_warnings`): the
   Herfindahl-Hirschman Index Section 9 names by name, on both position and
   sector weights, on the standard 0-1 fractional scale (not the antitrust
   convention's 0-10000 points scale) -- plus the simpler "any single
   position/sector above a threshold" flag Section 9's own worked example
   ("e.g. 15%") asks for as a distinct check from HHI.
3. **Sector gaps** (`sector_gaps`): which sectors have zero representation,
   optionally enriched with the screener's top-ranked names in that sector if
   the caller supplies them -- and an honest empty list if not, rather than
   fabricating candidates this module has no way to know.
4. **Rebalance pointer** (`recommend`'s `rebalance` field): whether the
   qualitative signals above are significant enough to point a user at
   Section 27's optimizer + `rebalancing.build_rebalance_plan` output. This
   module never calls either itself -- it only decides *whether* to point at
   one, and passes through a plan the caller already computed, if any.

Every English string produced here is a short, deterministic template built
from structured fields, not generated prose -- free-text narration is
Section 11's LLM layer, grounded in numbers exactly like these; this module
supplies the numbers, not the writing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from quantpulse.analysis.scoring import RATINGS
from quantpulse.portfolio.rebalancing import RebalancePlan

__all__ = [
    "DEFAULT_POSITION_CONCENTRATION_THRESHOLD",
    "DEFAULT_SECTOR_CONCENTRATION_THRESHOLD",
    "KNOWN_GICS_SECTORS",
    "HoldingContext",
    "HoldingRecommendation",
    "ConcentrationWarning",
    "ConcentrationSummary",
    "SectorGap",
    "RebalancePointer",
    "PortfolioRecommendations",
    "holding_recommendation",
    "herfindahl_index",
    "effective_position_count",
    "concentration_warnings",
    "sector_gaps",
    "recommend",
]

# Section 9's own worked example ("flagging any single position above a
# configurable threshold (e.g. 15% of the portfolio)"). The sector-level
# threshold reuses the same number since Section 9 gives no distinct figure
# for sectors -- both are tunable independently via `recommend`'s parameters.
DEFAULT_POSITION_CONCENTRATION_THRESHOLD = 0.15
DEFAULT_SECTOR_CONCENTRATION_THRESHOLD = 0.15

# Section 9's example: "here are 5 top-ranked Healthcare names... to consider."
_MAX_GAP_CANDIDATES = 5

# The 11 standard GICS Level-1 sectors, spelled exactly as `macro.
# SECTOR_COMMODITY_SENSITIVITY`'s own comment documents them (Wikipedia's GICS
# "Sector" column, `tickers.sector`) -- reused naming, not reinvented. A
# convenience default: a caller with the live ticker universe on hand should
# prefer passing the sectors actually present in it (`known_sectors=`), since
# this hardcoded list could in principle drift from the ingestion source.
KNOWN_GICS_SECTORS: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)

_ACTION_BY_RATING: dict[str, Literal["add", "trim", "sell", "hold"]] = {
    "strong_buy": "add",
    "buy": "add",
    "hold": "hold",
    "sell": "trim",
    "strong_sell": "sell",
}
_RATING_LABELS: dict[str, str] = {
    "strong_buy": "Strong Buy",
    "buy": "Buy",
    "hold": "Hold",
    "sell": "Sell",
    "strong_sell": "Strong Sell",
}


def _validate_rating(rating: str, *, field: str) -> None:
    if rating not in RATINGS:
        raise ValueError(f"{field} must be one of {RATINGS}, got {rating!r}")


def _validate_threshold(threshold: float, *, field: str) -> None:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"{field} must be in (0, 1], got {threshold}")


# --------------------------------------------------------------------------- #
# Per-holding Add / Trim / Hold / Sell
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldingContext:
    """What `recommend` needs to know about one currently-held position.

    `weight` is the position's fraction of total portfolio value (0-1).
    `sector` is `None` for asset types with no company sector (cash, most
    ETFs, Section 9) -- such holdings simply don't contribute to sector
    concentration or count toward closing a sector gap. `purchase_rating`, if
    known (e.g. joined from `composite_scores` history at the purchase date),
    lets the reason cite how the rating has moved since purchase.
    """

    weight: float
    rating: str
    sector: str | None = None
    purchase_rating: str | None = None


@dataclass(frozen=True)
class HoldingRecommendation:
    """Section 9's per-position guidance: an action plus a plain-English reason."""

    symbol: str
    action: Literal["add", "trim", "sell", "hold"]
    rating: str
    purchase_rating: str | None
    weight: float
    reason: str


def holding_recommendation(
    symbol: str,
    weight: float,
    rating: str,
    *,
    purchase_rating: str | None = None,
    concentration_threshold: float = DEFAULT_POSITION_CONCENTRATION_THRESHOLD,
) -> HoldingRecommendation:
    """Section 9's Add/Trim/Hold/Sell + reason for one currently-held position.

    A straight remap of the composite rating (strong_buy/buy -> add, hold ->
    hold, sell -> trim, strong_sell -> sell) -- except "add" is downgraded to
    "hold" when `weight` is already at or above `concentration_threshold`, so
    this function never recommends adding to a position its own concentration
    check would flag as overweight (see `concentration_warnings`, which uses
    the same default threshold).
    """
    _validate_rating(rating, field="rating")
    if purchase_rating is not None:
        _validate_rating(purchase_rating, field="purchase_rating")
    if weight < 0:
        raise ValueError(f"weight must be >= 0 (long-only, Section 2), got {weight}")
    _validate_threshold(concentration_threshold, field="concentration_threshold")

    action = _ACTION_BY_RATING[rating]
    overweight_capped = action == "add" and weight >= concentration_threshold
    if overweight_capped:
        action = "hold"

    reason = f"Rated {_RATING_LABELS[rating]}"
    if purchase_rating is not None and purchase_rating != rating:
        reason += f" (was {_RATING_LABELS[purchase_rating]} at purchase)"
    reason += "."
    if overweight_capped:
        reason += (
            f" Already {weight:.0%} of the portfolio, at or above the "
            f"{concentration_threshold:.0%} concentration guideline -- held rather than added to."
        )

    return HoldingRecommendation(
        symbol=symbol,
        action=action,
        rating=rating,
        purchase_rating=purchase_rating,
        weight=weight,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Concentration: Herfindahl-Hirschman Index + threshold warnings
# --------------------------------------------------------------------------- #


def herfindahl_index(weights: Mapping[str, float]) -> float:
    """The Herfindahl-Hirschman Index of `weights` (Section 9): sum of squared weight fractions.

    On the standard 0-1 fractional-weight scale, NOT the antitrust
    convention's 0-10000-points-squared scale: ranges from 1/N (N equal-weighted
    positions) up to 1.0 (a single position). `effective_position_count` below
    is the more directly interpretable companion figure this number implies.
    Non-positive weights are ignored (a symbol at 0% doesn't contribute).
    """
    return float(sum(w * w for w in weights.values() if w > 0))


def effective_position_count(hhi: float) -> float | None:
    """1/HHI: "this concentration is as diversified as N equal-weighted positions."

    A standard, well-known transform of HHI (not a new methodology) -- an HHI
    of 0.25 reads the same as 4 equal-weighted positions, however many
    positions are actually held. `None` when `hhi <= 0` (nothing held, nothing
    to describe).
    """
    return 1.0 / hhi if hhi > 0 else None


@dataclass(frozen=True)
class ConcentrationWarning:
    """One position or sector whose weight exceeds its concentration threshold."""

    kind: Literal["position", "sector"]
    label: str  # a symbol (kind="position") or a sector name (kind="sector")
    weight: float
    threshold: float
    message: str


def concentration_warnings(
    weights: Mapping[str, float],
    *,
    kind: Literal["position", "sector"],
    threshold: float = DEFAULT_POSITION_CONCENTRATION_THRESHOLD,
) -> list[ConcentrationWarning]:
    """Every label in `weights` strictly above `threshold`, highest weight first.

    Section 9's "flagging any single position above a configurable threshold
    (e.g. 15%)" -- a simple per-label check, distinct from (and a complement
    to) the portfolio-wide summary `herfindahl_index` gives.
    """
    _validate_threshold(threshold, field="threshold")
    noun = "position" if kind == "position" else "sector"
    flagged = sorted(
        ((label, w) for label, w in weights.items() if w > threshold),
        key=lambda item: -item[1],
    )
    return [
        ConcentrationWarning(
            kind=kind,
            label=label,
            weight=weight,
            threshold=threshold,
            message=(
                f"{label} is {weight:.0%} of your portfolio, above the "
                f"{threshold:.0%} {noun}-concentration threshold."
            ),
        )
        for label, weight in flagged
    ]


@dataclass(frozen=True)
class ConcentrationSummary:
    """Section 9's concentration-risk block: HHI + threshold flags, position and sector."""

    position_hhi: float
    position_effective_count: float | None
    sector_hhi: float | None
    sector_effective_count: float | None
    warnings: list[ConcentrationWarning]


# --------------------------------------------------------------------------- #
# Sector gap analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SectorGap:
    """One sector with zero representation in the portfolio.

    `candidates` are top-ranked names in that sector, if the caller supplied
    any via `recommend`'s `sector_candidates` -- empty, never fabricated, when
    it didn't (this module has no screener access of its own).
    """

    sector: str
    candidates: list[str]
    message: str


def sector_gaps(
    sector_weights: Mapping[str, float],
    *,
    known_sectors: Sequence[str] = KNOWN_GICS_SECTORS,
    candidates: Mapping[str, Sequence[str]] | None = None,
) -> list[SectorGap]:
    """Sectors in `known_sectors` with zero weight in `sector_weights`, in `known_sectors` order.

    Section 9's "gap analysis of sectors... entirely missing from the
    portfolio," optionally enriched with `candidates[sector]` (e.g. the
    screener's top-ranked names in that sector), capped at
    `_MAX_GAP_CANDIDATES` to match Section 9's own "here are 5..." example.
    """
    held = {sector for sector, weight in sector_weights.items() if weight > 0}
    result: list[SectorGap] = []
    for sector in known_sectors:
        if sector in held:
            continue
        names = list((candidates or {}).get(sector, ()))[:_MAX_GAP_CANDIDATES]
        message = f"No {sector} holdings."
        if names:
            message += f" Top-ranked {sector} names to consider: {', '.join(names)}."
        result.append(SectorGap(sector=sector, candidates=names, message=message))
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RebalancePointer:
    """Whether the qualitative signals above are worth pointing a user at Section 27.

    `reasons` is the deterministic list of what triggered it (empty when
    `triggered` is False). `plan` is never computed here -- it's whatever
    `RebalancePlan` the caller already built via `optimization.py` +
    `rebalancing.build_rebalance_plan` and chose to pass through, or `None` if
    they didn't run one.
    """

    triggered: bool
    reasons: list[str]
    plan: RebalancePlan | None


@dataclass(frozen=True)
class PortfolioRecommendations:
    """The full Section 9 "Recommendations & rebalancing" result."""

    holdings: list[HoldingRecommendation]
    concentration: ConcentrationSummary
    sector_gaps: list[SectorGap]
    rebalance: RebalancePointer


def recommend(
    holdings: Mapping[str, HoldingContext],
    *,
    position_threshold: float = DEFAULT_POSITION_CONCENTRATION_THRESHOLD,
    sector_threshold: float = DEFAULT_SECTOR_CONCENTRATION_THRESHOLD,
    known_sectors: Sequence[str] = KNOWN_GICS_SECTORS,
    sector_candidates: Mapping[str, Sequence[str]] | None = None,
    rebalance_plan: RebalancePlan | None = None,
) -> PortfolioRecommendations:
    """Assemble Section 9's full recommendation set from already-computed inputs.

    `holdings` is keyed by symbol so duplicates are structurally impossible.
    Sector weights are derived by summing `weight` over holdings sharing a
    `sector` (holdings with `sector=None` -- cash, most ETFs -- contribute to
    neither sector concentration nor sector-gap coverage).

    The rebalance pointer triggers when any position/sector concentration
    warning fires or any holding is recommended "trim"/"sell" -- a simple,
    deterministic OR over the other three pieces' own output, not a new
    judgment call.
    """
    for symbol, ctx in holdings.items():
        _validate_rating(ctx.rating, field=f"holdings[{symbol!r}].rating")
        if ctx.purchase_rating is not None:
            _validate_rating(ctx.purchase_rating, field=f"holdings[{symbol!r}].purchase_rating")
        if ctx.weight < 0:
            raise ValueError(
                f"holdings[{symbol!r}].weight must be >= 0 (long-only, Section 2), got {ctx.weight}"
            )

    position_weights = {symbol: ctx.weight for symbol, ctx in holdings.items()}
    sector_weights: dict[str, float] = {}
    for ctx in holdings.values():
        if ctx.sector:
            sector_weights[ctx.sector] = sector_weights.get(ctx.sector, 0.0) + ctx.weight

    position_hhi = herfindahl_index(position_weights)
    sector_hhi = herfindahl_index(sector_weights) if sector_weights else None

    warnings = concentration_warnings(
        position_weights, kind="position", threshold=position_threshold
    )
    if sector_weights:
        warnings += concentration_warnings(
            sector_weights, kind="sector", threshold=sector_threshold
        )

    holding_recs = [
        holding_recommendation(
            symbol,
            ctx.weight,
            ctx.rating,
            purchase_rating=ctx.purchase_rating,
            concentration_threshold=position_threshold,
        )
        for symbol, ctx in holdings.items()
    ]

    gaps = sector_gaps(sector_weights, known_sectors=known_sectors, candidates=sector_candidates)

    reasons: list[str] = []
    if any(w.kind == "position" for w in warnings):
        reasons.append("one or more positions exceed the concentration threshold")
    if any(w.kind == "sector" for w in warnings):
        reasons.append("one or more sectors exceed the concentration threshold")
    action_flags = {rec.action for rec in holding_recs}
    if "sell" in action_flags or "trim" in action_flags:
        reasons.append("one or more holdings are rated Sell or Strong Sell")

    return PortfolioRecommendations(
        holdings=holding_recs,
        concentration=ConcentrationSummary(
            position_hhi=position_hhi,
            position_effective_count=effective_position_count(position_hhi),
            sector_hhi=sector_hhi,
            sector_effective_count=(
                effective_position_count(sector_hhi) if sector_hhi is not None else None
            ),
            warnings=warnings,
        ),
        sector_gaps=gaps,
        rebalance=RebalancePointer(triggered=bool(reasons), reasons=reasons, plan=rebalance_plan),
    )
