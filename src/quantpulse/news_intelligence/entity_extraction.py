"""Entity extraction — ticker/company matching (Section 7.3 step 1).

Two stages, cheapest and most precise first:

1. **Gazetteer match**: build once per ticker universe (`build_gazetteer`),
   then reuse across every article. Matches three ways -- a `$CASHTAG`
   (any length, since a deliberate cashtag is already a strong signal), a
   bare uppercase ticker of 2+ characters (single-letter tickers like "T" or
   "F" are real, but matching them bare against ordinary text is far too
   noisy -- they only count via cashtag), and a suffix-stripped company name
   ("Target Corporation" -> "Target"). All three match *case-sensitively*
   against the original text: financial headlines capitalize company names
   and tickers as proper nouns, so this one constraint is most of what keeps
   an ambiguous name like "Target" (Target Corp) or "ALL" (Allstate) from
   matching ordinary lowercase usage, without needing a hand-maintained
   blocklist.

2. **spaCy NER fallback**: `en_core_web_sm` runs locally, free, no API call
   (Section 7.3) -- used only to catch *surface-form* variants the literal
   gazetteer alias list missed (e.g. different punctuation/word order in how
   a name is written), by fuzzy-matching each detected ORG entity back to
   the gazetteer's alias list. This is a real, named limitation, not a
   silent gap (Section 22): it cannot connect a distinct brand/common name
   like "Google" to its legal entity "Alphabet Inc." (GOOGL) -- that needs a
   hand-curated alias table, out of scope here.
"""

import re
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from spacy.language import Language

# Single-letter tickers only ever match via `$cashtag` -- bare, they're
# indistinguishable from ordinary capitalized words/initials.
_MIN_BARE_TICKER_LENGTH = 2
# Suffix-stripped company-name aliases shorter than this are dropped: too
# short to be a meaningfully specific match (and more likely to be noisy).
_MIN_ALIAS_LENGTH = 3

_SPACY_MODEL = "en_core_web_sm"
_SPACY_FUZZY_CUTOFF = 0.9

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6}(?:[.-][A-Za-z]{1,4})?)\b")
# Allows one hyphen/dot share-class suffix (BRK-B, BF.B) -- yfinance/Finnhub
# convention uses '-'; Wikipedia's source table uses '.' (Section 5 note).
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{1,6}(?:[.-][A-Z]{1,4})?)\b")

_TRAILING_SUFFIX_RE = re.compile(
    r"[,]?\s+(?:"
    r"incorporated|inc\.?|corporation|corp\.?|co\.?|companies|company|"
    r"holdings?|group|ltd\.?|limited|plc|l\.?l\.?c\.?|l\.?p\.?"
    r")\s*$",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^the\s+", re.IGNORECASE)
_TRAILING_AMPERSAND_RE = re.compile(r"&\s*$")


def _strip_corporate_suffix(name: str) -> str:
    """ "Target Corporation" -> "Target"; "The Coca-Cola Company" -> "Coca-Cola".

    Strips at most *one* trailing suffix, not a loop to a fixed point: looping
    is what turns "Booking Holdings Inc." into just "Booking" (stripping
    "Inc." then, wrongly, "Holdings" too) when "Holdings" here is part of the
    real, distinguishing name, not generic boilerplate. A single pass only
    ever removes the one suffix token actually at the end of the raw name.
    """
    stripped = _TRAILING_SUFFIX_RE.sub("", name.strip()).strip()
    # "JPMorgan Chase & Co." strips to "JPMorgan Chase &" -- clean up the
    # ampersand a suffix strip can leave dangling.
    stripped = _TRAILING_AMPERSAND_RE.sub("", stripped).strip()
    return _LEADING_ARTICLE_RE.sub("", stripped).strip()


_NON_ALNUM_RUN_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_fuzzy_match(text: str) -> str:
    """Punctuation/case-insensitive key for the spaCy fallback's fuzzy match.

    "Coca Cola Co" and "Coca-Cola" are the same company written two ways --
    a hyphen-vs-space and a trailing "Co" are exactly the surface noise the
    spaCy fallback exists to see through (module docstring), and raw
    `difflib` similarity on the un-normalized strings undershoots any
    reasonable cutoff for cases like this.
    """
    return _NON_ALNUM_RUN_RE.sub(" ", _strip_corporate_suffix(text).lower()).strip()


@dataclass(frozen=True)
class Gazetteer:
    """Precomputed alias -> symbol lookup for one ticker universe.

    Build once per universe refresh (Section 6.3's weekly cadence) via
    `build_gazetteer` and reuse across every article -- rebuilding this per
    article would dominate runtime for no benefit.
    """

    valid_symbols: frozenset[str]
    bare_ticker_symbols: frozenset[str]
    name_alias_to_symbol: dict[str, str]
    name_pattern: re.Pattern[str] | None
    alias_by_symbol: dict[str, list[str]]
    normalized_alias_to_symbol: dict[str, str]


def build_gazetteer(universe: pd.DataFrame) -> Gazetteer:
    """Build a `Gazetteer` from a `(symbol, name)` ticker universe.

    `universe` matches the `tickers` table / `wikipedia_client` shape: any
    DataFrame with `symbol` and `name` columns works (extra columns ignored).
    """
    valid_symbols: set[str] = set()
    bare_ticker_symbols: set[str] = set()
    name_alias_to_symbol: dict[str, str] = {}
    alias_by_symbol: dict[str, list[str]] = {}

    for row in universe.itertuples(index=False):
        symbol = str(row.symbol).strip().upper()
        if not symbol:
            continue
        valid_symbols.add(symbol)
        if len(symbol) >= _MIN_BARE_TICKER_LENGTH:
            bare_ticker_symbols.add(symbol)

        raw_name = str(getattr(row, "name", "") or "").strip()
        if not raw_name:
            continue
        aliases = {raw_name, _strip_corporate_suffix(raw_name)} - {""}
        alias_by_symbol[symbol] = sorted(aliases)
        for alias in aliases:
            if len(alias) >= _MIN_ALIAS_LENGTH:
                # First universe row wins a duplicate alias; a genuine
                # collision between two distinct companies' stripped names
                # is a known, accepted gazetteer limitation (Section 22).
                name_alias_to_symbol.setdefault(alias, symbol)

    # Longest-first so e.g. "JPMorgan Chase & Co." isn't shadowed at the
    # same text position by a shorter alias that happens to be its prefix.
    ordered_aliases = sorted(name_alias_to_symbol, key=len, reverse=True)
    name_pattern = (
        re.compile(r"\b(?:" + "|".join(re.escape(a) for a in ordered_aliases) + r")\b")
        if ordered_aliases
        else None
    )

    normalized_alias_to_symbol: dict[str, str] = {}
    for alias, symbol in name_alias_to_symbol.items():
        normalized_alias_to_symbol.setdefault(_normalize_for_fuzzy_match(alias), symbol)

    return Gazetteer(
        valid_symbols=frozenset(valid_symbols),
        bare_ticker_symbols=frozenset(bare_ticker_symbols),
        name_alias_to_symbol=name_alias_to_symbol,
        name_pattern=name_pattern,
        alias_by_symbol=alias_by_symbol,
        normalized_alias_to_symbol=normalized_alias_to_symbol,
    )


@lru_cache(maxsize=1)
def _load_nlp() -> "Language":
    import spacy

    # Only NER is needed; dropping the parser/tagger/lemmatizer roughly
    # halves per-document latency for a nightly batch of hundreds of articles.
    return spacy.load(_SPACY_MODEL, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])


def _spacy_fallback_matches(text: str, gazetteer: Gazetteer) -> set[str]:
    if not gazetteer.normalized_alias_to_symbol:
        return set()
    nlp = _load_nlp()
    normalized_aliases = list(gazetteer.normalized_alias_to_symbol.keys())
    matches: set[str] = set()
    for ent in nlp(text).ents:
        if ent.label_ != "ORG":
            continue
        candidate = _normalize_for_fuzzy_match(ent.text)
        if not candidate:
            continue
        best = get_close_matches(candidate, normalized_aliases, n=1, cutoff=_SPACY_FUZZY_CUTOFF)
        if best:
            matches.add(gazetteer.normalized_alias_to_symbol[best[0]])
    return matches


def extract_entities(
    text: str, gazetteer: Gazetteer, *, use_spacy_fallback: bool = True
) -> list[str]:
    """Ticker symbols mentioned in `text` (a headline/summary/post title).

    Case-sensitive gazetteer match first (cashtags, bare tickers, company
    names), then an optional spaCy NER fallback for surface-form name
    variants the gazetteer's literal alias list missed. Returns a sorted,
    deduplicated list -- empty if nothing matched, never raises on
    unrecognized/foreign text.
    """
    if not text:
        return []

    matched: set[str] = set()

    for cashtag_match in _CASHTAG_RE.finditer(text):
        symbol = cashtag_match.group(1).upper()
        if symbol in gazetteer.valid_symbols:
            matched.add(symbol)

    for bare_match in _BARE_TICKER_RE.finditer(text):
        symbol = bare_match.group(1)
        if symbol in gazetteer.bare_ticker_symbols:
            matched.add(symbol)

    if gazetteer.name_pattern is not None:
        for name_match in gazetteer.name_pattern.finditer(text):
            matched.add(gazetteer.name_alias_to_symbol[name_match.group(0)])

    if use_spacy_fallback:
        matched |= _spacy_fallback_matches(text, gazetteer)

    return sorted(matched)


def tag_articles(
    articles: pd.DataFrame,
    gazetteer: Gazetteer,
    *,
    text_columns: tuple[str, ...] = ("title", "summary"),
    use_spacy_fallback: bool = True,
) -> pd.Series:
    """Matched ticker symbols per row of `articles` (Tier 1/2/3 output, Section 7.3).

    Concatenates whichever of `text_columns` exist on `articles` (GDELT
    articles have no `summary`, for instance) and runs `extract_entities` on
    the result. Returns a Series aligned to `articles.index`, each entry a
    (possibly empty) sorted list of symbols -- the raw material for
    `news_events.matched_symbols` (Section 13), once a later Phase-4 part
    persists it.
    """
    present_columns = [c for c in text_columns if c in articles.columns]

    def _row_text(row: pd.Series) -> str:
        parts = [str(row[c]) for c in present_columns if pd.notna(row[c])]
        return " ".join(parts)

    combined_text = (
        articles.apply(_row_text, axis=1)
        if present_columns
        else pd.Series([""] * len(articles), index=articles.index)
    )
    return combined_text.apply(
        lambda text: extract_entities(text, gazetteer, use_spacy_fallback=use_spacy_fallback)
    )
