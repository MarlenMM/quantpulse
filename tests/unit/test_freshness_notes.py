"""Finding 36: a quarterly source labelled by its period, not its age.

The freshness strip printed "Institutional ownership: 169 days ago" beside prices
measured in days. That age was correct -- it is the newest quarter SEC had
published -- and it read as six months of neglect. The row now names the period
("Q1 2026 filings") and says whether that is the newest SEC publishes.

That claim is only true if something asked SEC. SEC usually publishes a 13F
window 2-9 days after it closes, but the June-August 2026 window was still
missing 32 days after its close, so no calendar rule can decide it. The weekly
13F step records when it last asked and what the newest published quarter was;
the sentence is composed from that record, server-side, and both front ends
print it verbatim.
"""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.analysis import freshness
from quantpulse.storage import persistence
from quantpulse.storage.models import Base, InstitutionalOwnership, Ticker

TODAY = date(2026, 10, 2)
Q1 = date(2026, 3, 31)


class TestQuarterName:
    @pytest.mark.parametrize(
        ("quarter_end", "name"),
        [
            (date(2026, 3, 31), "Q1 2026"),
            (date(2026, 6, 30), "Q2 2026"),
            (date(2026, 9, 30), "Q3 2026"),
            (date(2025, 12, 31), "Q4 2025"),
        ],
    )
    def test_names_the_calendar_quarter(self, quarter_end: date, name: str) -> None:
        assert freshness.quarter_name(quarter_end) == name


class TestDescribeThirteenF:
    def test_a_recent_check_that_found_nothing_newer_says_so(self) -> None:
        note = freshness.describe_thirteen_f(
            Q1, checked_on=date(2026, 9, 29), checked_period=Q1, today=TODAY
        )
        assert note == freshness.FreshnessNote(
            label="Q1 2026 filings — the newest SEC publishes (checked 29 Sep)", behind=False
        )

    def test_a_check_older_than_two_weekly_runs_is_behind(self) -> None:
        """The claim needs evidence that is still current: a weekly check, with
        one missed run allowed -- the same 16 days the strip allows any weekly
        source."""
        last_good = date(2026, 9, 16)
        assert (TODAY - last_good).days == freshness.CHECK_STALE_AFTER_DAYS
        assert not freshness.describe_thirteen_f(
            Q1, checked_on=last_good, checked_period=Q1, today=TODAY
        ).behind

        note = freshness.describe_thirteen_f(
            Q1, checked_on=date(2026, 9, 15), checked_period=Q1, today=TODAY
        )
        assert note == freshness.FreshnessNote(
            label="Q1 2026 filings — SEC not checked since 15 Sep", behind=True
        )

    def test_never_checked_makes_no_claim(self) -> None:
        note = freshness.describe_thirteen_f(Q1, checked_on=None, checked_period=None, today=TODAY)
        assert note == freshness.FreshnessNote(
            label="Q1 2026 filings — not yet checked against SEC", behind=False
        )

    def test_a_newer_published_quarter_than_stored_is_behind(self) -> None:
        note = freshness.describe_thirteen_f(
            Q1, checked_on=date(2026, 9, 29), checked_period=date(2026, 6, 30), today=TODAY
        )
        assert note == freshness.FreshnessNote(
            label="Q1 2026 filings — SEC's newest is Q2 2026", behind=True
        )

    def test_nothing_stored_leaves_the_strip_to_say_never_run(self) -> None:
        assert (
            freshness.describe_thirteen_f(
                None, checked_on=date(2026, 9, 29), checked_period=Q1, today=TODAY
            )
            is None
        )


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine = create_engine(f"sqlite:///{tmp_path / 'notes.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        s.add(Ticker(symbol="AAPL", name="Apple Inc.", asset_type="equity", is_active=True))
        s.commit()
        yield s


class TestStoredCheck:
    def test_a_check_is_recorded_and_replaced(self, session: Session) -> None:
        persistence.record_source_check(
            session, freshness.THIRTEEN_F_CHECK, checked_on=date(2026, 9, 21), newest_period=Q1
        )
        persistence.record_source_check(
            session, freshness.THIRTEEN_F_CHECK, checked_on=date(2026, 9, 28), newest_period=Q1
        )
        session.commit()
        assert persistence.read_source_check(session, freshness.THIRTEEN_F_CHECK) == (
            date(2026, 9, 28),
            Q1,
        )

    def test_an_unrecorded_source_reads_none(self, session: Session) -> None:
        assert persistence.read_source_check(session, freshness.THIRTEEN_F_CHECK) is None

    def test_the_note_is_composed_from_the_stored_quarter_and_the_check(
        self, session: Session
    ) -> None:
        session.add(InstitutionalOwnership(symbol="AAPL", quarter_end_date=Q1, num_filers=1))
        persistence.record_source_check(
            session, freshness.THIRTEEN_F_CHECK, checked_on=date(2026, 9, 29), newest_period=Q1
        )
        session.commit()

        notes = persistence.read_freshness_notes(session, today=TODAY)
        assert notes == {
            "institutional_ownership": freshness.FreshnessNote(
                label="Q1 2026 filings — the newest SEC publishes (checked 29 Sep)", behind=False
            )
        }
        # Keyed like the strip it annotates, so a front end can look it up by name.
        assert set(notes) <= set(persistence.read_data_freshness(session))

    def test_no_stored_quarter_means_no_note(self, session: Session) -> None:
        assert persistence.read_freshness_notes(session, today=TODAY) == {}
