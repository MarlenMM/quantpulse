"""Sentences for the freshness strip's quarterly sources (finding 36).

The strip labels every source by its age, which is right for prices and wrong
for 13F. "Institutional ownership: 169 days ago" was correct -- it was the newest
quarter SEC had published -- and it read as six months of neglect beside prices
measured in days. A quarterly source is labelled by its period instead: "Q1 2026
filings".

The second half of the sentence, "the newest SEC publishes", is a claim, and it
has to be earned by asking. SEC usually publishes a 13F window 2-9 days after it
closes (the `Last-Modified` of the eight windows before 2026), but on 2026-10-02
the June-August window was still missing, 32 days after its close. A calendar
rule would have printed that claim falsely or marked a current source behind.
So the weekly 13F step records when it last asked and what the newest published
quarter was (`source_checks`), and the sentence is composed from that record.

Composed here, once, and printed verbatim by both front ends -- the same rule as
`coverage_note` and `edge_note`, for the same reason: two copies of a sentence
drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

__all__ = [
    "CHECK_STALE_AFTER_DAYS",
    "THIRTEEN_F_CHECK",
    "FreshnessNote",
    "describe_thirteen_f",
    "quarter_name",
]

#: The `source_checks` key the weekly 13F step writes.
THIRTEEN_F_CHECK = "sec_13f"

#: How old the last check may be before "the newest SEC publishes" stops being
#: a claim the app can make. The check runs weekly; this allows one missed run,
#: the same 16 days the strip allows any weekly source.
CHECK_STALE_AFTER_DAYS = 16


@dataclass(frozen=True)
class FreshnessNote:
    """What the strip prints for a source instead of its age, and whether to mark it."""

    label: str
    behind: bool


def quarter_name(quarter_end: date) -> str:
    """`2026-03-31` -> `"Q1 2026"`."""
    return f"Q{(quarter_end.month - 1) // 3 + 1} {quarter_end.year}"


def _day_month(day: date) -> str:
    return f"{day.day} {day:%b}"


def describe_thirteen_f(
    stored_quarter: date | None,
    *,
    checked_on: date | None,
    checked_period: date | None,
    today: date,
) -> FreshnessNote | None:
    """The 13F row: the stored quarter by name, and what the last SEC check found.

    `None` when nothing is stored, so the strip keeps saying "never run" -- an
    empty table must not get a reassuring sentence.
    """
    if stored_quarter is None:
        return None
    period = f"{quarter_name(stored_quarter)} filings"
    if checked_on is None:
        return FreshnessNote(f"{period} — not yet checked against SEC", behind=False)
    if (today - checked_on).days > CHECK_STALE_AFTER_DAYS:
        return FreshnessNote(
            f"{period} — SEC not checked since {_day_month(checked_on)}", behind=True
        )
    if checked_period is not None and checked_period > stored_quarter:
        return FreshnessNote(
            f"{period} — SEC's newest is {quarter_name(checked_period)}", behind=True
        )
    return FreshnessNote(
        f"{period} — the newest SEC publishes (checked {_day_month(checked_on)})", behind=False
    )
