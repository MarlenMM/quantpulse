"""Saying *which* categories are behind a score, not just how much weight was.

`build_composite` renormalizes over the categories that had data, and both front
ends reported that as a percentage: "good coverage (80%)". The percentage says
how much was missing and never what.

That difference hid the worst data outage this project has had. Between
2026-08-10 and 2026-09-14 the tier-1 news step died on its budget every week,
sentiment aged past the thirty days `read_latest_sentiment` looks back over, and
every one of 503 names was scored with no news sentiment at all. No step failed.
No step wrote zero rows. Every run reported success, every row was valid, and
the only visible trace anywhere was a number falling from 90 to 80.
"""

from __future__ import annotations

import pandas as pd

from quantpulse.analysis import scoring
from quantpulse.analysis.investor_profiles import CATEGORIES


def _row(**present: float) -> dict[str, float | None]:
    """A stored-shape row: every `<category>_score` column, mostly absent."""
    row: dict[str, float | None] = {
        scoring.CATEGORY_SCORE_COLUMNS[category]: None for category in CATEGORIES
    }
    for category, value in present.items():
        row[scoring.CATEGORY_SCORE_COLUMNS[category]] = value
    return row


class TestDescribeCompositeCoverage:
    def test_a_fully_covered_name_says_so_in_one_clause(self) -> None:
        row = _row(**{category: 50.0 for category in CATEGORIES})
        assert scoring.describe_composite_coverage(row) == (
            "All seven categories are behind this score."
        )

    def test_one_missing_category_is_named_in_the_singular(self) -> None:
        row = _row(**{category: 50.0 for category in CATEGORIES if category != "sentiment"})
        sentence = scoring.describe_composite_coverage(row)
        assert sentence.startswith("Six of the seven categories are behind this score")
        assert "news sentiment is missing" in sentence
        # The point of the sentence: not treated as a neutral 50.
        assert "renormalized over the rest" in sentence

    def test_the_outage_shape_names_both_categories(self) -> None:
        """The state of every row in the database between August and September."""
        row = _row(fundamental=50.0, technical=50.0, analyst=50.0, momentum=50.0, smart_money=50.0)
        sentence = scoring.describe_composite_coverage(row)
        assert sentence.startswith("Five of the seven categories are behind this score")
        assert "news sentiment and industry/macro are missing" in sentence

    def test_a_row_with_nothing_says_there_is_no_score(self) -> None:
        assert scoring.describe_composite_coverage(_row()) == (
            "No category produced a reading, so there is no score to read."
        )

    def test_a_nan_counts_as_missing(self) -> None:
        """Built the way production builds it, not with a hand-written nan.

        A SQL NULL arrives as `None` from an all-null column and as `float("nan")`
        the moment one real value promotes the column's dtype — the bug that once
        printed "Macro tone: nan" on the dashboard. So the fixture goes through a
        DataFrame column rather than asserting against a literal.
        """
        frame = pd.DataFrame(
            [
                {scoring.CATEGORY_SCORE_COLUMNS["sentiment"]: 61.0},
                {scoring.CATEGORY_SCORE_COLUMNS["sentiment"]: None},
            ]
        )
        row = {**_row(technical=50.0), **frame.iloc[1].to_dict()}
        assert pd.isna(row[scoring.CATEGORY_SCORE_COLUMNS["sentiment"]])
        assert "news sentiment" in scoring.describe_composite_coverage(row)

    def test_it_does_not_guess_why(self) -> None:
        """A stock with no analyst coverage and a step that failed last night look
        identical from here; the sentence points at the surface that knows."""
        sentence = scoring.describe_composite_coverage(_row(technical=50.0))
        assert "freshness" in sentence
        for guess in ("failed", "error", "stale", "API key"):
            assert guess not in sentence


class TestMissingCategories:
    def test_it_lists_them_in_the_canonical_order(self) -> None:
        row = _row(technical=50.0, momentum=50.0)
        assert scoring.missing_categories(row) == [
            category for category in CATEGORIES if category not in ("technical", "momentum")
        ]

    def test_a_full_row_lists_nothing(self) -> None:
        assert scoring.missing_categories(_row(**{c: 1.0 for c in CATEGORIES})) == []


class TestZeroCoverageCategories:
    def _universe(self, rows: list[dict[str, float | None]]) -> pd.DataFrame:
        """A frame big enough for "every row" to mean something."""
        padding = scoring.MIN_UNIVERSE_FOR_COVERAGE_CLAIM - len(rows)
        return pd.DataFrame(rows + [dict(rows[-1]) for _ in range(max(padding, 0))])

    def test_a_category_absent_from_every_row_is_reported(self) -> None:
        frame = self._universe([_row(technical=50.0), _row(technical=60.0)])
        assert "sentiment" in scoring.zero_coverage_categories(frame)

    def test_one_name_missing_a_category_is_not_an_outage(self) -> None:
        """The median name in the committed database carries 0.90 of the weight.
        Reporting that as a problem would make the signal meaningless."""
        frame = self._universe(
            [
                _row(**{category: 50.0 for category in CATEGORIES if category != "sentiment"}),
                *[_row(**{category: 50.0 for category in CATEGORIES}) for _ in range(20)],
            ]
        )
        assert scoring.zero_coverage_categories(frame) == []

    def test_a_universe_too_small_to_judge_reports_nothing(self) -> None:
        """On one name, "absent from every row" is just "absent from this row".

        This is not hypothetical tidiness: the refresh's end-to-end test scores a
        single mocked symbol with two price bars, and the first version of this
        function reported six categories as an outage and turned that run
        "partial".
        """
        assert scoring.zero_coverage_categories(pd.DataFrame([_row(technical=50.0)])) == []

    def test_the_floor_is_what_is_holding_it_back(self) -> None:
        """Mutation-proofing the test above: the same one-row frame *does* report
        once the floor is lowered, so the assertion is about the floor rather
        than about the row happening to have data."""
        frame = pd.DataFrame([_row(technical=50.0)])
        assert "sentiment" in scoring.zero_coverage_categories(frame, min_symbols=1)

    def test_an_empty_frame_reports_nothing(self) -> None:
        """A night that scored nothing is a different failure, and `_CRITICAL_STEPS`
        already covers it. Reporting all seven here would bury that."""
        assert scoring.zero_coverage_categories(pd.DataFrame()) == []
