"""What the nightly alert is allowed to say, and what it must leave out.

Every threshold asserted here was chosen from a measurement against the
committed demo database on 2026-09-06, not from taste:

* "any rating change, universe-wide" is 96-195 per stored snapshot pair.
  At ~45 characters a line that is 4,300-8,800 characters against Discord's
  2,000-character per-message limit -- an alert nobody can read and a webhook
  call that fails.
* changes that *land on* strong_buy / strong_sell are 17-32 per pair.
* new pattern formations are 17-80 a day across 503 names, of which 45% clear
  a confidence of 70 (the distribution's median is 67.9).

So the rules are: everything on a name you hold or watch, plus only the
top-conviction crossings from the other 480.
"""

from datetime import date

import pandas as pd
import pytest

from quantpulse.alerting import rules


def _changes(*rows: tuple[str, str, str, float]) -> pd.DataFrame:
    """`read_rating_changes`' frame shape: symbol, from, to, and the score move."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "previous_rating": before,
                "rating": after,
                "previous_score": 50.0,
                "composite_score": 50.0 + move,
                "score_change": move,
            }
            for symbol, before, after, move in rows
        ]
    )


def _patterns(*rows: tuple[str, str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": date(2026, 9, 4),
                "pattern_type": kind,
                "direction": direction,
                "confidence": confidence,
            }
            for symbol, kind, direction, confidence in rows
        ]
    )


# --------------------------------------------------------------------------- #
# Rating changes
# --------------------------------------------------------------------------- #


def test_every_change_on_a_tracked_name_alerts() -> None:
    """A name you own moving hold -> buy is the whole point; no threshold applies."""
    alerts = rules.rating_change_alerts(_changes(("NVDA", "hold", "buy", 8.4)), tracked={"NVDA"})
    assert [a.symbol for a in alerts] == ["NVDA"]
    assert alerts[0].tracked is True
    assert "Hold" in alerts[0].line and "Buy" in alerts[0].line


def test_an_untracked_middling_change_is_silent() -> None:
    """The measured ~100 of these a night are the reason the alert is bearable."""
    assert rules.rating_change_alerts(_changes(("AIZ", "hold", "buy", 8.4)), tracked=set()) == []


def test_an_untracked_change_landing_on_a_top_conviction_rating_alerts() -> None:
    alerts = rules.rating_change_alerts(
        _changes(("AAA", "buy", "strong_buy", 5.2), ("ZZZ", "sell", "strong_sell", -4.8)),
        tracked=set(),
    )
    assert {a.symbol for a in alerts} == {"AAA", "ZZZ"}
    assert all(a.tracked is False for a in alerts)


def test_leaving_a_top_conviction_rating_is_silent_for_an_untracked_name() -> None:
    """Landing on the app's strongest verdict is the event. Leaving it is not.

    Measured, `strong_* -> something` is roughly half of all strong-touching
    changes; including it would double the section for a de-escalation the
    Dashboard already shows.
    """
    assert (
        rules.rating_change_alerts(
            _changes(("AAA", "strong_buy", "buy", -5.2), ("ZZZ", "strong_sell", "sell", 4.8)),
            tracked=set(),
        )
        == []
    )


def test_a_tracked_name_crossing_into_strong_buy_alerts_once() -> None:
    """It qualifies under both rules; it must not be listed twice."""
    alerts = rules.rating_change_alerts(
        _changes(("NVDA", "buy", "strong_buy", 5.2)), tracked={"NVDA"}
    )
    assert len(alerts) == 1 and alerts[0].tracked is True


def test_the_line_names_both_ratings_and_the_score_move() -> None:
    line = rules.rating_change_alerts(
        _changes(("NVDA", "hold", "strong_buy", 12.25)), tracked={"NVDA"}
    )[0].line
    assert "NVDA" in line
    assert "Hold" in line and "Strong Buy" in line
    assert "+12.2" in line or "+12.3" in line


def test_a_direction_marker_distinguishes_an_upgrade_from_a_downgrade() -> None:
    up = rules.rating_change_alerts(_changes(("A", "hold", "buy", 3.0)), tracked={"A"})[0].line
    down = rules.rating_change_alerts(_changes(("B", "buy", "hold", -3.0)), tracked={"B"})[0].line
    assert up[0] != down[0]


def test_an_empty_change_frame_is_handled() -> None:
    assert rules.rating_change_alerts(pd.DataFrame(), tracked={"NVDA"}) == []


def test_an_unknown_rating_is_rejected_rather_than_rendered_raw() -> None:
    """A rating with no display label must not reach a webhook as its raw key.

    The message is asserted, not just the exception type. An earlier version of
    this test only asked for `KeyError` and passed even with the guard replaced
    by `RATING_LABELS.get(rating, rating)` -- the `KeyError` it was catching came
    from the *rank* lookup two lines later, so the test proved nothing about the
    thing it was named for.
    """
    with pytest.raises(ValueError, match="must not reach a reader as a raw key"):
        rules.rating_change_alerts(_changes(("A", "hold", "moon", 3.0)), tracked={"A"})


def test_the_rejection_names_the_offending_field() -> None:
    with pytest.raises(ValueError, match="previous_rating is 'moon'"):
        rules.rating_change_alerts(_changes(("A", "moon", "buy", 3.0)), tracked={"A"})


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #


def test_a_new_pattern_on_a_tracked_name_alerts() -> None:
    alerts = rules.pattern_alerts(
        _patterns(("MSFT", "cup_and_handle", "bullish", 82.0)), tracked={"MSFT"}
    )
    assert [a.symbol for a in alerts] == ["MSFT"]
    assert "cup and handle" in alerts[0].line
    assert "82" in alerts[0].line


def test_two_formations_of_the_same_type_are_told_apart_by_their_dates() -> None:
    """Found by rendering the committed demo database rather than a fixture.

    A run really does insert several double bottoms for one symbol at once --
    different formations, different completion dates, different confidences.
    Without the date the message read "UNP new double bottom" three times with
    nothing to distinguish them, which looks like a duplication bug.
    """
    rows = pd.DataFrame(
        [
            {
                "symbol": "UNP",
                "date": day,
                "pattern_type": "double_bottom",
                "direction": "bullish",
                "confidence": confidence,
            }
            for day, confidence in (
                (date(2026, 8, 14), 97.0),
                (date(2026, 9, 1), 85.0),
            )
        ]
    )
    lines = [a.line for a in rules.pattern_alerts(rows, tracked={"UNP"})]
    assert len(set(lines)) == 2
    assert "2026-08-14" in " ".join(lines) and "2026-09-01" in " ".join(lines)


def test_a_new_pattern_on_an_untracked_name_is_silent() -> None:
    """17-80 formations a day across the universe; only your own names are news."""
    assert (
        rules.pattern_alerts(_patterns(("AIZ", "double_top", "bearish", 95.0)), tracked=set()) == []
    )


def test_a_low_confidence_pattern_is_below_the_floor() -> None:
    assert (
        rules.pattern_alerts(
            _patterns(("MSFT", "double_top", "bearish", 69.9)),
            tracked={"MSFT"},
            min_confidence=70.0,
        )
        == []
    )


def test_a_pattern_exactly_on_the_floor_alerts() -> None:
    assert (
        len(
            rules.pattern_alerts(
                _patterns(("MSFT", "double_top", "bearish", 70.0)),
                tracked={"MSFT"},
                min_confidence=70.0,
            )
        )
        == 1
    )


def test_an_empty_pattern_frame_is_handled() -> None:
    assert rules.pattern_alerts(pd.DataFrame(), tracked={"MSFT"}) == []


# --------------------------------------------------------------------------- #
# The digest
# --------------------------------------------------------------------------- #


def _digest(**kwargs):
    defaults = {
        "changes": pd.DataFrame(),
        "patterns": pd.DataFrame(),
        "tracked": set(),
        "current_date": date(2026, 9, 5),
        "previous_date": date(2026, 9, 4),
    }
    return rules.build_digest(**{**defaults, **kwargs})


def test_a_quiet_night_sends_nothing_at_all() -> None:
    """No message is the correct output for no news. An alert that arrives every
    night regardless stops being read on about the fourth night."""
    assert _digest() is None


def test_the_header_names_both_dates_being_compared() -> None:
    """The comparison is "the two most recent stored snapshots", which is not
    the same as "since yesterday" -- the demo's last pair was ten days apart.
    Saying "since yesterday" over a ten-day gap is the exact class of overclaim
    this project's rules exist to prevent."""
    text = _digest(
        changes=_changes(("NVDA", "hold", "buy", 8.4)),
        tracked={"NVDA"},
        current_date=date(2026, 9, 5),
        previous_date=date(2026, 8, 26),
    )
    assert "2026-09-05" in text and "2026-08-26" in text


def test_tracked_names_get_their_own_section_above_the_universe_ones() -> None:
    text = _digest(
        changes=_changes(("NVDA", "hold", "buy", 2.0), ("AAA", "buy", "strong_buy", 9.0)),
        tracked={"NVDA"},
    )
    assert text.index("NVDA") < text.index("AAA")


def test_there_is_no_empty_your_names_heading_when_nothing_is_tracked() -> None:
    """The deployment that actually sends this has an empty portfolio (the demo
    runs PORTFOLIO_BACKEND=session), so this is the normal case there."""
    text = _digest(changes=_changes(("AAA", "buy", "strong_buy", 9.0)))
    assert rules.TRACKED_HEADING not in text


def test_the_universe_section_is_capped_and_says_how_many_it_dropped() -> None:
    """Measured: 17-32 crossings a night. The cap keeps one message under
    Discord's 2,000-character limit, and the count keeps the omission honest."""
    many = _changes(*[(f"S{i:03d}", "buy", "strong_buy", float(40 - i)) for i in range(32)])
    text = _digest(changes=many, max_universe_lines=15)
    assert text.count("Strong Buy") == 15
    assert "17 more" in text


def test_the_cap_keeps_the_biggest_movers() -> None:
    many = _changes(*[(f"S{i:03d}", "buy", "strong_buy", float(40 - i)) for i in range(32)])
    text = _digest(changes=many, max_universe_lines=3)
    assert "S000" in text and "S002" in text and "S031" not in text


def test_a_realistic_night_fits_in_one_discord_message() -> None:
    """The regression that motivated every cap above.

    503 names, ~100 rating changes of which ~20 land on a top-conviction
    rating, ~30 new formations, and a 20-name portfolio -- the measured shape of
    a real weeknight. Before the scoping rules this rendered ~100 lines and
    could not be sent.
    """
    universe = [
        (f"U{i:03d}", "hold", "buy" if i % 2 else "sell", float(i % 7) - 3.0) for i in range(80)
    ]
    crossings = [
        (f"C{i:03d}", "buy", "strong_buy" if i % 2 else "strong_sell", float(20 - i))
        for i in range(20)
    ]
    held = [(f"H{i:02d}", "hold", "buy", 4.0) for i in range(3)]
    text = _digest(
        changes=_changes(*(universe + crossings + held)),
        patterns=_patterns(*[(f"H{i:02d}", "double_bottom", "bullish", 80.0) for i in range(2)]),
        tracked={f"H{i:02d}" for i in range(20)},
    )
    assert len(text) <= 2000, f"digest is {len(text)} characters; Discord's limit is 2000"
