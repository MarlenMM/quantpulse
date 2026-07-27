"""Ticker/company fuzzy search for the symbol inputs (Section 31).

Section 31's motivation is exact: "a small quality-of-life detail that matters
the moment someone doesn't remember that Alphabet's ticker is `GOOGL`." So this
matches against **company names as well as symbols**, and tolerates typos.

Ranking is deliberately staged rather than handed wholesale to a similarity
score, because the obvious intent should always win:

1. an exact symbol match ("AAPL" → AAPL) — never outranked by anything
2. a symbol that starts with the query ("AAP" → AAPL)
3. a company name that starts with the query ("Alpha" → Alphabet)
4. a company name containing the query ("Motors" → General Motors)
5. only then a fuzzy match, for actual typos ("Alphabt" → Alphabet)

A pure `difflib` ranking would let a close-but-wrong name beat an exact ticker,
which is the one thing a ticker box must never do.

Uses `difflib` from the standard library — a fuzzy-matching dependency for a
search box over a few hundred rows would not earn its place in the lockfile
(Section 29).
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

__all__ = ["MAX_SUGGESTIONS", "search_symbols", "format_choice"]

MAX_SUGGESTIONS = 8

# Below this similarity a "fuzzy match" is noise, not a typo. Tuned by eye:
# 0.6 catches "Alphabt"/"Micrsoft" while rejecting unrelated names.
_FUZZY_THRESHOLD = 0.6


def _similarity(query: str, candidate: str) -> float:
    return SequenceMatcher(None, query, candidate).ratio()


def search_symbols(
    universe: pd.DataFrame, query: str, *, limit: int = MAX_SUGGESTIONS
) -> list[str]:
    """Symbols from `universe` best matching `query`, most relevant first.

    `universe` needs `symbol` and (optionally) `name` columns -- the shape
    `persistence.read_ticker_universe` returns. An empty query returns the
    first `limit` symbols so a freshly-opened picker isn't blank.

    Matching is case-insensitive and ignores surrounding whitespace. Results
    are deduplicated while preserving rank order, so a name that matches on
    several tiers appears once, at its best rank.
    """
    if universe.empty or "symbol" not in universe.columns:
        return []

    needle = query.strip().lower()
    symbols = universe["symbol"].astype(str).tolist()
    names = (
        universe["name"].fillna("").astype(str).tolist()
        if "name" in universe.columns
        else [""] * len(symbols)
    )

    if not needle:
        return symbols[:limit]

    exact: list[str] = []
    symbol_prefix: list[str] = []
    name_prefix: list[str] = []
    name_contains: list[str] = []
    scored_fuzzy: list[tuple[float, str]] = []

    for symbol, name in zip(symbols, names, strict=True):
        symbol_lower = symbol.lower()
        name_lower = name.lower()
        if symbol_lower == needle:
            exact.append(symbol)
        elif symbol_lower.startswith(needle):
            symbol_prefix.append(symbol)
        elif name_lower.startswith(needle):
            name_prefix.append(symbol)
        elif needle in name_lower or needle in symbol_lower:
            name_contains.append(symbol)
        else:
            score = max(_similarity(needle, symbol_lower), _similarity(needle, name_lower))
            if score >= _FUZZY_THRESHOLD:
                scored_fuzzy.append((score, symbol))

    scored_fuzzy.sort(key=lambda pair: (-pair[0], pair[1]))
    ranked = [
        *exact,
        *symbol_prefix,
        *name_prefix,
        *name_contains,
        *[symbol for _, symbol in scored_fuzzy],
    ]

    seen: dict[str, None] = {}
    for symbol in ranked:
        seen.setdefault(symbol, None)
    return list(seen)[:limit]


def format_choice(universe: pd.DataFrame, symbol: str) -> str:
    """`"AAPL — Apple Inc."` for a picker label, or just the symbol if unknown.

    Showing the company name next to the ticker is half the point of the
    feature: it confirms the user picked the company they meant, not a
    similarly-spelled ticker.
    """
    if universe.empty or "name" not in universe.columns:
        return symbol
    match = universe.loc[universe["symbol"] == symbol, "name"]
    if match.empty or not str(match.iloc[0]).strip() or pd.isna(match.iloc[0]):
        return symbol
    return f"{symbol} — {match.iloc[0]}"
