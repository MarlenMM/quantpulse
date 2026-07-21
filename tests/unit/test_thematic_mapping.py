import pandas as pd
import pytest

from quantpulse.news_intelligence import thematic_mapping as tm

# --- theme detection ----------------------------------------------------------


def test_match_themes_detects_multiple_overlapping_baskets() -> None:
    themes = tm.match_themes("New AI chip export controls hit semiconductor makers")
    assert "ai_theme" in themes
    assert "semiconductors" in themes


def test_match_themes_returns_empty_for_unthemed_text() -> None:
    assert tm.match_themes("The company reported quarterly results in line with estimates") == []


def test_match_themes_empty_text() -> None:
    assert tm.match_themes("") == []


def test_match_themes_is_word_boundaried() -> None:
    # "solar" triggers clean_energy, but not as a substring of "solarium".
    assert "clean_energy" in tm.match_themes("A new solar farm was announced")
    assert "clean_energy" not in tm.match_themes("They installed a glass solarium extension")


def test_match_themes_is_sorted_and_deterministic() -> None:
    text = "AI chip and semiconductor and solar and oil price news"
    assert tm.match_themes(text) == sorted(tm.match_themes(text))


# --- basket config integrity --------------------------------------------------


def test_every_basket_has_members_and_keywords() -> None:
    for basket in tm.THEMATIC_BASKETS:
        assert basket.members, f"{basket.name} has no members"
        assert basket.keywords, f"{basket.name} has no keywords"


def test_basket_members_normalizes_and_unknown_is_empty() -> None:
    assert "NVDA" in tm.basket_members("ai_theme")
    assert tm.basket_members("no_such_theme") == frozenset()


def test_iter_basket_membership_yields_table_ready_rows() -> None:
    rows = list(tm.iter_basket_membership())
    assert ("ai_theme", "NVDA") in rows
    # (theme_name, symbol) shape, every symbol uppercase.
    assert all(len(r) == 2 and r[1].isupper() for r in rows)


# --- sector baskets -----------------------------------------------------------


def test_build_sector_baskets_groups_by_sector() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["JPM", "BAC", "AAPL"],
            "sector": ["Financials", "Financials", "Information Technology"],
        }
    )
    baskets = tm.build_sector_baskets(universe)
    assert baskets["Financials"] == frozenset({"JPM", "BAC"})
    assert baskets["Information Technology"] == frozenset({"AAPL"})


def test_build_sector_baskets_skips_rows_without_sector() -> None:
    universe = pd.DataFrame({"symbol": ["AAPL", "XYZ"], "sector": ["Information Technology", None]})
    baskets = tm.build_sector_baskets(universe)
    assert "XYZ" not in {s for members in baskets.values() for s in members}


# --- propagate: the core judgment --------------------------------------------


def test_propagate_applies_tier2_attenuation() -> None:
    adj = tm.propagate(["XOM"], sentiment=1.0, confidence=1.0, tier=2)
    assert adj["XOM"] == pytest.approx(0.5)  # tier-2 weight


def test_tier3_is_more_attenuated_than_tier2() -> None:
    a2 = tm.propagate(["XOM"], sentiment=1.0, confidence=1.0, tier=2)["XOM"]
    a3 = tm.propagate(["XOM"], sentiment=1.0, confidence=1.0, tier=3)["XOM"]
    assert a3 < a2


def test_propagate_never_amplifies_source_sentiment() -> None:
    # |adjustment| <= |sentiment| for all members, all tiers, any confidence.
    for tier in (2, 3):
        for confidence in (0.1, 0.5, 1.0):
            adj = tm.propagate(
                tm.basket_members("ai_theme"),
                sentiment=-0.8,
                confidence=confidence,
                tier=tier,
            )
            assert all(abs(v) <= 0.8 for v in adj.values())


def test_propagate_preserves_sentiment_sign() -> None:
    negative = tm.propagate(["XOM", "CVX"], sentiment=-0.6, confidence=1.0, tier=2)
    assert all(v < 0 for v in negative.values())
    positive = tm.propagate(["XOM", "CVX"], sentiment=0.6, confidence=1.0, tier=2)
    assert all(v > 0 for v in positive.values())


def test_propagate_excludes_directly_named_to_avoid_double_counting() -> None:
    adj = tm.propagate(
        tm.basket_members("ai_theme"),
        sentiment=-0.8,
        confidence=1.0,
        tier=2,
        directly_named=["nvda"],  # case-insensitive
    )
    assert "NVDA" not in adj
    assert "AMD" in adj


def test_propagate_clips_confidence_to_unit_interval() -> None:
    over = tm.propagate(["XOM"], sentiment=1.0, confidence=5.0, tier=2)["XOM"]
    assert over == pytest.approx(0.5)  # confidence clipped to 1.0
    under = tm.propagate(["XOM"], sentiment=1.0, confidence=-2.0, tier=2)["XOM"]
    assert under == pytest.approx(0.0)  # confidence clipped to 0.0


def test_propagate_empty_basket_returns_empty() -> None:
    assert tm.propagate([], sentiment=1.0, confidence=1.0, tier=2) == {}


def test_propagate_rejects_tier1() -> None:
    with pytest.raises(ValueError):
        tm.propagate(["AAPL"], sentiment=1.0, confidence=1.0, tier=1)


# --- propagate_from_text ------------------------------------------------------


def test_propagate_from_text_chains_detection_and_propagation() -> None:
    out = tm.propagate_from_text(
        "New bank capital requirements announced under Basel rules",
        sentiment=-0.5,
        confidence=0.8,
        tier=2,
    )
    assert set(out) == set(tm.basket_members("banks"))
    assert all(v < 0 for v in out.values())


def test_propagate_from_text_unions_overlapping_baskets_without_compounding() -> None:
    # NVDA is in both ai_theme and semiconductors; a text matching both must
    # still adjust it exactly once, at the single tier weight (no compounding).
    out = tm.propagate_from_text(
        "AI chip and semiconductor export controls tightened",
        sentiment=1.0,
        confidence=1.0,
        tier=2,
    )
    assert out["NVDA"] == pytest.approx(0.5)


def test_propagate_from_text_matches_sector_names() -> None:
    sector_baskets = {"Financials": frozenset({"JPM", "BAC"})}
    out = tm.propagate_from_text(
        "Broad weakness across the Financials sector today",
        sentiment=-0.4,
        confidence=1.0,
        tier=2,
        sector_baskets=sector_baskets,
    )
    assert set(out) == {"JPM", "BAC"}


def test_propagate_from_text_no_match_returns_empty() -> None:
    out = tm.propagate_from_text(
        "A quiet day with no notable thematic news",
        sentiment=0.5,
        confidence=1.0,
        tier=2,
    )
    assert out == {}
