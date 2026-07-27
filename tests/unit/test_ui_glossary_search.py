"""Tests for the glossary and the ticker autocomplete (`app/lib`)."""

import pandas as pd
import pytest

from lib import glossary, search


class TestGlossaryContent:
    def test_covers_the_terms_section_10_names_explicitly(self) -> None:
        # Section 10: "explaining P/E, RSI, Sharpe ratio, beta, etc."
        for term in ("P/E", "RSI", "Sharpe ratio", "Beta"):
            assert glossary.define(term)

    def test_every_term_has_a_known_category(self) -> None:
        for term, (category, _) in glossary.TERMS.items():
            assert category in glossary.CATEGORIES, f"{term} has unlisted category {category}"

    def test_every_category_has_at_least_one_term(self) -> None:
        used = {category for category, _ in glossary.TERMS.values()}
        assert set(glossary.CATEGORIES) == used

    def test_definitions_are_substantial_prose_not_stubs(self) -> None:
        for term, (_, definition) in glossary.TERMS.items():
            assert len(definition) > 40, f"{term} definition is too thin"
            assert definition.strip()[-1] in ".?", f"{term} definition is unpunctuated"

    def test_definitions_avoid_defining_a_term_with_itself(self) -> None:
        # "Sharpe ratio: the ratio of Sharpe" helps nobody.
        for term, (_, definition) in glossary.TERMS.items():
            first_sentence = definition.split(".")[0].lower()
            assert first_sentence != term.lower()

    def test_relative_rating_caveat_travels_with_the_definition(self) -> None:
        # Section 22's pitfall: a relative ranking is not an absolute judgment.
        # The tooltip is exactly where a reader asks, so the caveat lives there.
        definition = glossary.define("Rating")
        assert definition is not None
        assert "relative" in definition.lower()

    def test_var_definition_states_what_it_cannot_tell_you(self) -> None:
        definition = glossary.define("Value at Risk")
        assert definition is not None
        assert "nothing about how bad" in definition.lower()


class TestTip:
    def test_formats_a_known_term_with_its_name(self) -> None:
        text = glossary.tip("Beta")
        assert text.startswith("**Beta** —")
        assert "market" in text

    def test_extra_context_is_appended(self) -> None:
        text = glossary.tip("Data coverage", "Raise this to exclude thin names.")
        assert "Data coverage" in text
        assert "Raise this to exclude thin names." in text

    def test_unknown_term_degrades_instead_of_raising(self) -> None:
        # A missing tooltip is cosmetic; taking a page down over one is not.
        assert glossary.tip("Not A Real Term") == "Not A Real Term"
        assert glossary.tip("Nope", "fallback text") == "fallback text"


class TestGlossarySearch:
    def test_empty_query_returns_everything(self) -> None:
        assert len(glossary.search_terms("")) == len(glossary.TERMS)

    def test_matches_on_term_name(self) -> None:
        assert "Sharpe ratio" in glossary.search_terms("sharpe")

    def test_matches_on_definition_text(self) -> None:
        # Someone searching a concept they can't name yet.
        assert "Survivorship bias" in glossary.search_terms("bankrupt")

    def test_is_case_insensitive(self) -> None:
        assert glossary.search_terms("SHARPE") == glossary.search_terms("sharpe")

    def test_no_match_is_empty(self) -> None:
        assert glossary.search_terms("zzzznotaterm") == []


class TestSymbolSearch:
    def _universe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "MSFT", "GM", "AMZN", "APA"],
                "name": [
                    "Apple Inc.",
                    "Alphabet Inc.",
                    "Microsoft Corp.",
                    "General Motors",
                    "Amazon.com Inc.",
                    "APA Corporation",
                ],
            }
        )

    def test_exact_symbol_always_ranks_first(self) -> None:
        # The one thing a ticker box must never do is let a close-but-wrong
        # company name outrank the ticker the user actually typed.
        assert search.search_symbols(self._universe(), "GM")[0] == "GM"
        assert search.search_symbols(self._universe(), "gm")[0] == "GM"

    def test_symbol_prefix_beats_name_match(self) -> None:
        results = search.search_symbols(self._universe(), "AP")
        assert results[0] == "APA"  # symbol prefix
        assert "AAPL" in results  # name contains "ap" too, but ranks lower

    def test_finds_a_company_by_name_not_just_ticker(self) -> None:
        # Section 31's motivating example verbatim.
        assert search.search_symbols(self._universe(), "Alphabet")[0] == "GOOGL"

    def test_name_substring_match(self) -> None:
        assert "GM" in search.search_symbols(self._universe(), "Motors")

    def test_tolerates_a_typo(self) -> None:
        assert "GOOGL" in search.search_symbols(self._universe(), "Alphabt")
        assert "MSFT" in search.search_symbols(self._universe(), "Micrsoft")

    def test_unrelated_query_returns_nothing(self) -> None:
        assert search.search_symbols(self._universe(), "zzzzqqqq") == []

    def test_empty_query_returns_a_starting_list(self) -> None:
        results = search.search_symbols(self._universe(), "  ")
        assert len(results) == len(self._universe())

    def test_results_are_deduplicated(self) -> None:
        results = search.search_symbols(self._universe(), "a")
        assert len(results) == len(set(results))

    def test_limit_is_respected(self) -> None:
        assert len(search.search_symbols(self._universe(), "a", limit=2)) == 2

    def test_empty_universe_is_safe(self) -> None:
        assert search.search_symbols(pd.DataFrame(), "AAPL") == []
        assert search.search_symbols(pd.DataFrame({"other": [1]}), "AAPL") == []

    def test_missing_name_column_still_matches_symbols(self) -> None:
        universe = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
        assert search.search_symbols(universe, "AAPL") == ["AAPL"]

    def test_null_names_do_not_crash(self) -> None:
        universe = pd.DataFrame({"symbol": ["AAPL", "XYZ"], "name": ["Apple Inc.", None]})
        assert "AAPL" in search.search_symbols(universe, "apple")


class TestFormatChoice:
    def _universe(self) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["AAPL", "XYZ"], "name": ["Apple Inc.", None]})

    def test_pairs_ticker_with_company_name(self) -> None:
        assert search.format_choice(self._universe(), "AAPL") == "AAPL — Apple Inc."

    def test_falls_back_to_the_bare_symbol(self) -> None:
        assert search.format_choice(self._universe(), "XYZ") == "XYZ"
        assert search.format_choice(self._universe(), "NOPE") == "NOPE"
        assert search.format_choice(pd.DataFrame(), "AAPL") == "AAPL"


class TestTooltipCoverage:
    """The tooltips the pages ask for must actually resolve to definitions."""

    @pytest.mark.parametrize(
        "term",
        [
            "Rating",
            "Composite score",
            "Data coverage",
            "Percentile rank",
            "Market Regime Index",
            "VIX",
            "Market breadth",
            "Yield curve spread",
            "Volatility",
            "Sharpe ratio",
            "Sortino ratio",
            "Max drawdown",
            "Beta",
            "Value at Risk",
            "Herfindahl index",
            "Unrealized P/L",
            "Cost basis",
            "Holding period",
            "Rebalancing",
            "CAGR",
            "Hit rate",
            "Benchmark",
            "Turnover",
            "Sentiment score",
            "Monte Carlo simulation",
            "Support and resistance",
            "Tier 1 / 2 / 3 news",
        ],
    )
    def test_page_referenced_term_is_defined(self, term: str) -> None:
        assert glossary.define(term), f"pages call tip({term!r}) but it has no definition"
