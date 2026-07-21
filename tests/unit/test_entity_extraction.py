import pandas as pd
import pytest

from quantpulse.news_intelligence.entity_extraction import (
    _strip_corporate_suffix,
    build_gazetteer,
    extract_entities,
    tag_articles,
)


@pytest.fixture(scope="module")
def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "TGT", "ALL", "T", "BRK-B", "KO", "BKNG"],
            "name": [
                "Apple Inc.",
                "Microsoft Corporation",
                "Target Corporation",
                "The Allstate Corporation",
                "AT&T Inc.",
                "Berkshire Hathaway Inc.",
                "The Coca-Cola Company",
                "Booking Holdings Inc.",
            ],
        }
    )


@pytest.fixture(scope="module")
def gazetteer(universe: pd.DataFrame):
    return build_gazetteer(universe)


# --- _strip_corporate_suffix -------------------------------------------------


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Target Corporation", "Target"),
        ("The Coca-Cola Company", "Coca-Cola"),
        ("Apple Inc.", "Apple"),
        ("JPMorgan Chase & Co.", "JPMorgan Chase"),
        ("Procter & Gamble Company", "Procter & Gamble"),
        ("3M Company", "3M"),
        # Regression: looping suffix-strip used to also strip "Holdings",
        # turning this into the far more ambiguous bare word "Booking".
        ("Booking Holdings Inc.", "Booking Holdings"),
        ("Berkshire Hathaway Inc.", "Berkshire Hathaway"),
        ("No Suffix Here", "No Suffix Here"),
    ],
)
def test_strip_corporate_suffix(raw_name: str, expected: str) -> None:
    assert _strip_corporate_suffix(raw_name) == expected


# --- build_gazetteer ----------------------------------------------------------


def test_build_gazetteer_excludes_single_letter_tickers_from_bare_set(
    gazetteer,
) -> None:
    assert "T" in gazetteer.valid_symbols
    assert "T" not in gazetteer.bare_ticker_symbols
    assert "AAPL" in gazetteer.bare_ticker_symbols


def test_build_gazetteer_includes_both_raw_and_stripped_name_aliases(gazetteer) -> None:
    assert gazetteer.name_alias_to_symbol["Target Corporation"] == "TGT"
    assert gazetteer.name_alias_to_symbol["Target"] == "TGT"


def test_build_gazetteer_first_row_wins_on_alias_collision() -> None:
    universe = pd.DataFrame({"symbol": ["FOO", "BAR"], "name": ["Acme Corporation", "Acme Corp"]})
    g = build_gazetteer(universe)
    # Both stripped to the same alias "Acme" -- first row (FOO) wins.
    assert g.name_alias_to_symbol["Acme"] == "FOO"


def test_build_gazetteer_returns_none_pattern_for_empty_universe() -> None:
    g = build_gazetteer(pd.DataFrame({"symbol": [], "name": []}))
    assert g.name_pattern is None
    assert g.valid_symbols == frozenset()


def test_build_gazetteer_drops_short_aliases() -> None:
    # "3M" (2 chars) is below _MIN_ALIAS_LENGTH -- excluded from name matching,
    # but still present as a valid ticker for cashtag/bare-ticker matching.
    universe = pd.DataFrame({"symbol": ["MMM"], "name": ["3M Company"]})
    g = build_gazetteer(universe)
    assert "3M" not in g.name_alias_to_symbol


# --- extract_entities: gazetteer stage ---------------------------------------


def test_extracts_multiple_company_names_from_text(gazetteer) -> None:
    text = "Apple and Microsoft shares rose today after strong earnings"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["AAPL", "MSFT"]


def test_extracts_suffix_stripped_company_name(gazetteer) -> None:
    text = "Target Corporation reported strong holiday sales"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["TGT"]


def test_lowercase_common_word_does_not_false_positive_match_company_name(gazetteer) -> None:
    text = "The target price for the stock was raised by analysts"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == []


def test_mixed_case_word_does_not_false_positive_match_ticker(gazetteer) -> None:
    text = "All of the companies reported gains"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == []


def test_bare_single_letter_ticker_does_not_match(gazetteer) -> None:
    text = "T was mentioned here as a bare letter"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == []


def test_cashtag_matches_even_single_letter_ticker(gazetteer) -> None:
    text = "$AAPL and $T both rallied on cashtag mentions"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["AAPL", "T"]


def test_hyphenated_share_class_ticker_matches_bare(gazetteer) -> None:
    text = "BRK-B shares were flat in light trading"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["BRK-B"]


def test_full_company_name_with_ampersand_matches(gazetteer) -> None:
    text = "AT&T Inc. announced a new dividend policy"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["T"]


def test_booking_holdings_full_name_matches_without_becoming_ambiguous(gazetteer) -> None:
    text = "Booking Holdings shares fell on weak travel demand"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["BKNG"]


def test_empty_text_returns_empty_list(gazetteer) -> None:
    assert extract_entities("", gazetteer) == []


def test_result_is_deduplicated_and_sorted(gazetteer) -> None:
    text = "Apple, Apple, and Apple again, also Microsoft"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == ["AAPL", "MSFT"]


# --- extract_entities: spaCy fallback stage (real model, deterministic) -----


def test_spacy_fallback_catches_punctuation_surface_variant(gazetteer) -> None:
    text = "Coca Cola Co reported strong beverage sales this quarter"
    assert extract_entities(text, gazetteer, use_spacy_fallback=False) == []
    assert extract_entities(text, gazetteer, use_spacy_fallback=True) == ["KO"]


def test_spacy_fallback_does_not_reintroduce_common_word_false_positive(gazetteer) -> None:
    # "target" here is lowercase and non-ORG in context -- spaCy shouldn't
    # tag it as an organization, so the fallback must not add TGT either.
    text = "The target price for the stock was raised by analysts"
    assert extract_entities(text, gazetteer, use_spacy_fallback=True) == []


def test_spacy_fallback_is_a_strict_superset_of_gazetteer_only(gazetteer) -> None:
    text = "Apple and Microsoft shares rose today after strong earnings"
    without = set(extract_entities(text, gazetteer, use_spacy_fallback=False))
    with_fallback = set(extract_entities(text, gazetteer, use_spacy_fallback=True))
    assert without <= with_fallback


# --- tag_articles -------------------------------------------------------------


def test_tag_articles_combines_title_and_summary(gazetteer) -> None:
    df = pd.DataFrame(
        {
            "title": ["Apple shares rise", "Microsoft announces layoffs"],
            "summary": ["Strong earnings beat.", None],
        }
    )
    result = tag_articles(df, gazetteer, use_spacy_fallback=False)
    assert result.tolist() == [["AAPL"], ["MSFT"]]
    assert list(result.index) == list(df.index)


def test_tag_articles_handles_missing_summary_column(gazetteer) -> None:
    # GDELT-shaped articles: title/domain, no summary.
    df = pd.DataFrame({"title": ["Apple unveils new iPhone"], "domain": ["example.com"]})
    result = tag_articles(df, gazetteer, use_spacy_fallback=False)
    assert result.tolist() == [["AAPL"]]


def test_tag_articles_handles_empty_dataframe(gazetteer) -> None:
    df = pd.DataFrame({"title": [], "summary": []})
    result = tag_articles(df, gazetteer, use_spacy_fallback=False)
    assert result.tolist() == []


def test_tag_articles_handles_no_matching_text_columns(gazetteer) -> None:
    df = pd.DataFrame({"domain": ["example.com"], "language": ["English"]})
    result = tag_articles(df, gazetteer, use_spacy_fallback=False)
    assert result.tolist() == [[]]
