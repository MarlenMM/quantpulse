"""Thematic basket mapping + Tier-2/3 propagation (Section 7.3 step 5).

A per-ticker headline sentiment score misses that news moves whole *baskets*:
an "AI export controls" story moves every AI-exposed name, not just the ones
it happens to spell out (Section 7.3). This module owns two things:

1. **The baskets.** A hand-curated set of thematic baskets (`ai_theme`,
   `semiconductors`, `ev_theme`, ...) each with member tickers and the
   keyword triggers that select it, plus sector baskets built dynamically
   from the `tickers` universe. The curated set is an illustrative seed, not
   exhaustive -- it's meant to be extended, and each basket is a plain,
   auditable config entry.

2. **The propagation rule.** How a Tier-2 (industry) or Tier-3 (macro) event
   turns into a per-member score adjustment. Section 21 rates this Opus/High
   precisely because the rule is a judgment call that's easy to get subtly
   wrong -- over- or under-propagating an event's impact. The rule here is
   deliberately conservative and carries three invariants worth stating
   outright:

   - **Attenuation by tier.** A diffuse basket-level headline must not move an
     individual name as hard as a Tier-1 headline that names it directly
     would. `_TIER_PROPAGATION_WEIGHT` scales Tier-2 to 0.5 and Tier-3 to
     0.25 of a direct mention's strength.
   - **Never amplify.** An adjustment's magnitude is `|sentiment| x
     tier_weight x confidence`, and both factors are <= 1, so propagation can
     only ever *attenuate* the source sentiment, never exceed it. This is the
     structural guard against "over-propagating."
   - **No double counting.** A name already matched directly (Tier-1) in the
     same article is excluded from that article's basket propagation, so its
     direct sentiment and a basket echo of the same story don't stack.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

# Tier-1 (a headline naming the stock) is the direct per-stock sentiment and
# is NOT this module's job -- by convention its weight is 1.0. This module
# only attenuates the diffuse tiers relative to that.
_TIER_PROPAGATION_WEIGHT: dict[int, float] = {2: 0.5, 3: 0.25}


@dataclass(frozen=True)
class ThematicBasket:
    """A named group of tickers moved together by a theme, with its trigger keywords."""

    name: str
    members: frozenset[str]
    keywords: tuple[str, ...]


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _basket(name: str, members: Iterable[str], keywords: Iterable[str]) -> ThematicBasket:
    return ThematicBasket(
        name=name,
        members=frozenset(_normalize_symbol(m) for m in members),
        keywords=tuple(keywords),
    )


# Curated seed baskets (Section 7.3's "hand-curated set of thematic baskets").
# Illustrative, not exhaustive -- extend freely; each is a flat, auditable row.
THEMATIC_BASKETS: tuple[ThematicBasket, ...] = (
    _basket(
        "ai_theme",
        ["NVDA", "AMD", "MSFT", "GOOGL", "META", "AVGO", "TSM", "PLTR", "SMCI"],
        [
            "artificial intelligence",
            "generative ai",
            "large language model",
            "ai chip",
            "ai regulation",
            "ai export",
        ],
    ),
    _basket(
        "semiconductors",
        ["NVDA", "AMD", "INTC", "TSM", "AVGO", "MU", "QCOM", "AMAT", "LRCX", "ASML"],
        [
            "semiconductor",
            "chipmaker",
            "chip export",
            "export controls",
            "foundry",
            "wafer",
            "chip act",
        ],
    ),
    _basket(
        "ev_theme",
        ["TSLA", "RIVN", "LCID", "GM", "F", "NIO"],
        ["electric vehicle", "ev subsidy", "ev tax credit", "ev demand", "charging network"],
    ),
    _basket(
        "clean_energy",
        ["ENPH", "FSLR", "SEDG", "NEE", "RUN"],
        ["solar", "clean energy", "renewable energy", "wind power"],
    ),
    _basket(
        "oil_gas",
        ["XOM", "CVX", "COP", "SLB", "OXY", "MPC"],
        ["oil price", "crude oil", "opec", "natural gas", "oil supply"],
    ),
    _basket(
        "banks",
        ["JPM", "BAC", "WFC", "C", "GS", "MS"],
        ["bank capital", "capital requirements", "regional bank", "basel", "bank stress"],
    ),
)

_BASKET_BY_NAME: dict[str, ThematicBasket] = {b.name: b for b in THEMATIC_BASKETS}

# One compiled, word-boundaried, case-insensitive pattern per basket. Word
# boundaries keep a keyword like "solar" from firing inside "solarium".
_BASKET_PATTERNS: dict[str, re.Pattern[str]] = {
    b.name: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in b.keywords) + r")\b", re.IGNORECASE
    )
    for b in THEMATIC_BASKETS
    if b.keywords
}


def match_themes(text: str) -> list[str]:
    """Names of every thematic basket whose keyword triggers appear in `text`.

    An article can match more than one basket (an "AI chip export controls"
    story is both `ai_theme` and `semiconductors`) -- all are returned, so the
    caller can union their members. This is what derives a Tier-2 article's
    `matched_theme` (Section 13's `news_events`).
    """
    if not text:
        return []
    return sorted(name for name, pattern in _BASKET_PATTERNS.items() if pattern.search(text))


def basket_members(theme_name: str) -> frozenset[str]:
    """Members of the named thematic basket, or an empty set if unknown."""
    basket = _BASKET_BY_NAME.get(theme_name)
    return basket.members if basket is not None else frozenset()


def build_sector_baskets(universe: pd.DataFrame) -> dict[str, frozenset[str]]:
    """GICS sector -> member tickers, from a `(symbol, sector)` universe.

    Lets a Tier-2 event tagged to a whole sector ("bank capital requirements"
    -> Financials) propagate to every name in it, alongside the curated
    thematic baskets. Rows with no sector are skipped.
    """
    sector_to_symbols: dict[str, set[str]] = {}
    for row in universe.itertuples(index=False):
        sector = getattr(row, "sector", None)
        if not sector or pd.isna(sector):
            continue
        sector_to_symbols.setdefault(str(sector), set()).add(_normalize_symbol(row.symbol))
    return {sector: frozenset(symbols) for sector, symbols in sector_to_symbols.items()}


def propagate(
    members: Iterable[str],
    *,
    sentiment: float,
    confidence: float,
    tier: int,
    directly_named: Iterable[str] = (),
) -> dict[str, float]:
    """Per-member score adjustment for one Tier-2/3 event.

    `adjustment = sentiment * tier_weight * confidence` for every member,
    except those in `directly_named` (already carrying the event via their
    Tier-1 direct sentiment -- excluded to avoid double counting). `confidence`
    is clipped to [0, 1]; `tier` must be 2 or 3 (Tier-1 is the direct
    per-stock signal, not a propagation). Returns `{}` for an empty basket.
    """
    if tier not in _TIER_PROPAGATION_WEIGHT:
        raise ValueError(f"tier must be one of {sorted(_TIER_PROPAGATION_WEIGHT)}, got {tier}")

    weight = _TIER_PROPAGATION_WEIGHT[tier] * min(max(confidence, 0.0), 1.0)
    adjustment = sentiment * weight

    excluded = {_normalize_symbol(s) for s in directly_named}
    return {
        symbol: adjustment
        for symbol in sorted(_normalize_symbol(m) for m in members)
        if symbol not in excluded
    }


def propagate_from_text(
    text: str,
    *,
    sentiment: float,
    confidence: float,
    tier: int = 2,
    sector_baskets: dict[str, frozenset[str]] | None = None,
    directly_named: Iterable[str] = (),
) -> dict[str, float]:
    """Detect themes/sectors in `text`, then propagate the event to their union.

    Convenience wrapper chaining `match_themes` (+ optional sector-name
    matching against `sector_baskets`) into `propagate`. When a member falls
    in several matched baskets it still appears once, with a single adjustment
    (the propagation weight doesn't compound across overlapping baskets --
    another guard against over-propagation).
    """
    members: set[str] = set()
    for theme_name in match_themes(text):
        members |= basket_members(theme_name)

    if sector_baskets:
        for sector_name, sector_members in sector_baskets.items():
            if re.search(rf"\b{re.escape(sector_name)}\b", text, re.IGNORECASE):
                members |= sector_members

    return propagate(
        members,
        sentiment=sentiment,
        confidence=confidence,
        tier=tier,
        directly_named=directly_named,
    )


def iter_basket_membership() -> Iterator[tuple[str, str]]:
    """Yield `(theme_name, symbol)` rows for every curated basket.

    Table-ready for `thematic_baskets` (Section 13); a later integration can
    persist these. Kept as a generator here rather than writing the DB, in
    keeping with schema landing alongside its writer.
    """
    for basket in THEMATIC_BASKETS:
        for symbol in sorted(basket.members):
            yield basket.name, symbol
