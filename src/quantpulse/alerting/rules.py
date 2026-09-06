"""What the nightly refresh is allowed to interrupt you about (Section 10).

The refresh already computes everything an alert would carry -- `read_rating_
changes` has produced the Dashboard's "these five names moved from Hold to Buy"
since Phase 6, and `pattern_signals` has carried completed formations since the
Phase-2 detectors got a writer. The only question this module answers is which
of them is worth a message, and the answer came from counting rather than from
taste. Measured against the committed demo database on 2026-09-06:

===============================================  ==========================
Candidate trigger                                Messages per stored pair
===============================================  ==========================
any rating change, universe-wide                 96-195
a change *landing on* strong_buy/strong_sell     17-32
a change *leaving* strong_buy/strong_sell        17-32
any new chart formation, universe-wide           17-80 a day
the same, at confidence >= 70                    6-38 a day
===============================================  ==========================

Discord's per-message limit is 2,000 characters and these lines run about 45,
so "any rating change" is not merely noisy -- at ~100 lines it is roughly twice
what the transport will accept, and would fail rather than annoy. So:

1. **A name you hold or watch: every rating change, no threshold.** You already
   decided that name is worth your attention; the app has no business filtering
   it further.
2. **A name you hold or watch: a new formation at confidence >= 70.** The
   distribution's median is 67.9, so the floor is "better than the typical
   formation" and keeps 45% of them. Across a 20-name portfolio that is under
   two a day; across all 503 it would be 6-38, which is why it is scoped.
3. **Anywhere in the universe: a rating that lands ON strong_buy or
   strong_sell.** The app's two highest-conviction verdicts, and the only
   universe-wide trigger cheap enough to send. Landing on one is the event;
   *leaving* one is a de-escalation the Dashboard already shows, and including
   it would double the section for half the information.

Everything here is a pure function over frames. The reading, the sending and
the "is it configured" question all live elsewhere, so this stays testable
without a database or a network.
"""

from collections.abc import Iterable, Set
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from quantpulse.analysis.scoring import RATING_LABELS, RATINGS

__all__ = [
    "Alert",
    "DEFAULT_PATTERN_MIN_CONFIDENCE",
    "MAX_UNIVERSE_LINES",
    "TOP_CONVICTION_RATINGS",
    "TRACKED_HEADING",
    "UNIVERSE_HEADING",
    "build_digest",
    "pattern_alerts",
    "rating_change_alerts",
]

#: The ratings whose arrival is worth a message about a name you do not own.
TOP_CONVICTION_RATINGS = frozenset({"strong_buy", "strong_sell"})

#: Confidence floor for a formation to be worth a line. See the module docstring
#: for the distribution this came from.
DEFAULT_PATTERN_MIN_CONFIDENCE = 70.0

#: How many universe-wide crossings one message will list before summarising the
#: rest. Measured at 17-32 a night, so the normal message shows most of them and
#: says how many it dropped; the cap exists so an unusual night still sends.
MAX_UNIVERSE_LINES = 15

#: How long after completing a formation may still be called new.
#:
#: Applied **in addition to** the caller's "inserted by this run" filter, never
#: instead of it -- see `pattern_alerts` for why the date alone is the wrong
#: question. This exists for the opposite failure: the first run after the
#: pattern writer lands (or after a reseed) inserts the entire lookback window
#: at once, and the committed demo database really does hold rows completing in
#: 2025 that were first stored in 2026. Those are new to the database and are
#: not news, and announcing one as "new" a year late is the kind of overclaim
#: Section 22 is about.
#:
#: 30 days is generous against the lag being allowed for: a zigzag pivot needs
#: the swing after it before the shape can be confirmed, which runs to days or
#: a couple of weeks, not months.
MAX_FORMATION_AGE_DAYS = 30

TRACKED_HEADING = "Your names"
UNIVERSE_HEADING = "Top conviction, whole universe"

# Most bullish first, which is the order `RATINGS` is already declared in --
# so a *smaller* index is an upgrade.
_RATING_RANK = {rating: index for index, rating in enumerate(RATINGS)}

_UPGRADE_MARK = "▲"
_DOWNGRADE_MARK = "▼"
_PATTERN_MARK = "◆"


@dataclass(frozen=True)
class Alert:
    """One rendered line, plus what the digest needs to order and group it."""

    kind: str  # "rating_change" | "new_pattern"
    symbol: str
    tracked: bool
    magnitude: float  # |score move| or pattern confidence -- ranks within a section
    line: str


def _label(rating: str, *, field: str) -> str:
    """A rating in words, refusing to render one the engine does not define.

    Raising rather than falling back to the raw key is the point. A webhook
    message is the one surface in this project that nothing renders in a test
    the way a page gets rendered, so `strong_buy` arriving verbatim in someone's
    Discord would be a defect only a reader could catch. `ValueError` naming the
    field matches `recommendations._validate_rating`, which guards the same
    vocabulary on the way into holding advice.
    """
    try:
        return RATING_LABELS[rating]
    except KeyError:
        raise ValueError(
            f"{field} is {rating!r}, which is not one of {RATINGS}; "
            "a rating with no display label must not reach a reader as a raw key"
        ) from None


def rating_change_alerts(changes: pd.DataFrame, *, tracked: Set[str]) -> list[Alert]:
    """Alerts for rules 1 and 3 -- see the module docstring.

    `changes` is `persistence.read_rating_changes`' frame: symbol,
    previous_rating, rating, previous_score, composite_score, score_change. A
    tracked name that also crosses into a top-conviction rating produces one
    alert, not two.
    """
    if changes.empty:
        return []
    alerts: list[Alert] = []
    for row in changes.itertuples(index=False):
        symbol = str(row.symbol)
        is_tracked = symbol in tracked
        if not is_tracked and row.rating not in TOP_CONVICTION_RATINGS:
            continue
        before = _label(row.previous_rating, field="previous_rating")
        after = _label(row.rating, field="rating")
        improving = _RATING_RANK[row.rating] < _RATING_RANK[row.previous_rating]
        mark = _UPGRADE_MARK if improving else _DOWNGRADE_MARK
        move = float(row.score_change)
        alerts.append(
            Alert(
                kind="rating_change",
                symbol=symbol,
                tracked=is_tracked,
                magnitude=abs(move),
                line=(
                    f"{mark} `{symbol}` {before} → {after} "
                    f"(score {float(row.composite_score):.1f}, {move:+.1f})"
                ),
            )
        )
    return alerts


def pattern_alerts(
    patterns: pd.DataFrame,
    *,
    tracked: Set[str],
    as_of: date | None = None,
    min_confidence: float = DEFAULT_PATTERN_MIN_CONFIDENCE,
    max_age_days: int = MAX_FORMATION_AGE_DAYS,
) -> list[Alert]:
    """Alerts for rule 2 -- a new formation on a name you hold or watch.

    `patterns` is the `pattern_signals` row shape (symbol, date, pattern_type,
    direction, confidence) restricted to rows this run actually inserted. That
    restriction is the caller's job and it matters: a formation's stored date is
    the day its shape *completed*, which can be a week before the run that first
    detects it, so "date is recent" is not the same question as "this is new".

    An age limit applies on top of that, and only on top of it. The two guard
    opposite failures: without the caller's filter, a formation the database has
    held for weeks is announced every night; without the age limit, the first
    run after the writer lands (or after a reseed) inserts the whole lookback
    window at once and announces a formation that completed a year ago. The
    committed demo database holds exactly those rows. `as_of=None` skips the age
    check, for a caller that has already bounded the window itself.
    """
    if patterns.empty:
        return []
    oldest = None if as_of is None else as_of - timedelta(days=max_age_days)
    alerts: list[Alert] = []
    for row in patterns.itertuples(index=False):
        symbol = str(row.symbol)
        confidence = float(row.confidence)
        if symbol not in tracked or confidence < min_confidence:
            continue
        completed = pd.Timestamp(row.date).date()
        if oldest is not None and completed < oldest:
            continue
        shape = str(row.pattern_type).replace("_", " ")
        # The completion date, not the detection date, and it earns its place:
        # a run routinely inserts several formations of the same type on one
        # symbol, and without it the message repeats "UNP new double bottom"
        # three times with nothing to tell them apart. (Real rows from the
        # committed demo database; the tests had not produced that shape.)
        alerts.append(
            Alert(
                kind="new_pattern",
                symbol=symbol,
                tracked=True,
                magnitude=confidence,
                line=(
                    f"{_PATTERN_MARK} `{symbol}` new {shape} "
                    f"({row.direction}, confidence {confidence:.0f}, completed {completed})"
                ),
            )
        )
    return alerts


def _section(heading: str, alerts: Iterable[Alert], *, limit: int | None = None) -> list[str]:
    """One headed block, biggest mover first, honest about anything it dropped."""
    ordered = sorted(alerts, key=lambda a: (-a.magnitude, a.symbol))
    if not ordered:
        return []
    shown = ordered if limit is None else ordered[:limit]
    lines = [f"__{heading}__", *(a.line for a in shown)]
    dropped = len(ordered) - len(shown)
    if dropped:
        lines.append(f"…and {dropped} more.")
    return lines


def build_digest(
    *,
    changes: pd.DataFrame,
    patterns: pd.DataFrame,
    tracked: Set[str],
    current_date: date,
    previous_date: date,
    min_confidence: float = DEFAULT_PATTERN_MIN_CONFIDENCE,
    max_universe_lines: int = MAX_UNIVERSE_LINES,
) -> str | None:
    """The whole message, or `None` when there is nothing worth sending.

    `None` is load-bearing. An alert that arrives every night whether or not
    anything happened stops being read by about the fourth night, at which point
    it is worse than no alert -- so a quiet night sends no message at all rather
    than a message saying it was quiet.

    The header names **both** dates it compared. The comparison is "the two most
    recent stored snapshots", which is not the same thing as "since yesterday":
    the demo's most recent pair was ten days apart, and a message that said
    "since yesterday" over that gap is exactly the overclaim Section 22 is
    about.
    """
    rating = rating_change_alerts(changes, tracked=tracked)
    formations = pattern_alerts(
        patterns, tracked=tracked, as_of=current_date, min_confidence=min_confidence
    )
    if not rating and not formations:
        return None

    body = _section(TRACKED_HEADING, [a for a in rating + formations if a.tracked])
    body += _section(
        UNIVERSE_HEADING, [a for a in rating if not a.tracked], limit=max_universe_lines
    )
    header = (
        f"**QuantPulse — {current_date:%Y-%m-%d}** "
        f"(vs the previous stored scores, {previous_date:%Y-%m-%d})"
    )
    return "\n".join([header, *body])
