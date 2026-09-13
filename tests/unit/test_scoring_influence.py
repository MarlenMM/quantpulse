"""What each category actually does to the ranking, as against what it is weighted.

The stated weights are honoured arithmetically — that was fixed in an earlier
pass, and `explain_composite` asserts the decomposition sums exactly. This is a
different question: **a weight is an input, and influence on the final ordering
is an output**, and on real data they disagree badly.

Measured over all 503 balanced scores on 2026-09-11:

    category          stated   rank correlation with the composite
    technical           0.20   0.701
    momentum            0.15   0.646
    fundamental         0.25   0.483
    smart_money         0.10   0.193
    analyst             0.10   0.133

Technical carries a fifth of the weight and moves the ranking more than
fundamental, which carries a quarter. Analyst and smart money carry a tenth each
and barely move it at all.

**The obvious measure would have hidden all of this.** A variance-share panel —
each category's share of the spread in contributions — recovers the stated
weight almost exactly (25%/20%/10%/10%/15%/9%/10% against stated
.25/.20/.10/.10/.15/.10/.10), because every sub-score is a percentile and so
every category has near-identical dispersion by construction. It would have
drawn seven bars matching seven weights and reported that everything was fine.
"""

from datetime import date  # noqa: F401  (kept for parity with sibling test modules)

import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import scoring
from quantpulse.analysis.investor_profiles import CATEGORIES, get_profile

_BALANCED = get_profile("balanced").weights


def _universe(n: int = 200, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic universe where two categories drive the composite and two are noise.

    Built so the distinction under test is unambiguous: `technical` and
    `momentum` are strongly correlated with each other, `analyst` and
    `smart_money` are independent noise, and every category is a percentile in
    0-100 so all seven have the same dispersion.
    """
    rng = np.random.default_rng(seed)
    trend = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "technical": trend + rng.normal(scale=0.3, size=n),
            "momentum": trend + rng.normal(scale=0.3, size=n),
            "fundamental": rng.normal(size=n),
            "analyst": rng.normal(size=n),
            "sentiment": rng.normal(size=n),
            "industry_macro": rng.normal(size=n),
            "smart_money": rng.normal(size=n),
        }
    )
    # Percentile-rank every column, as `build_composite` does.
    frame = frame.rank(pct=True) * 100.0
    weights = pd.Series(_BALANCED)
    composite = (frame * weights).sum(axis=1) / weights.sum()
    return frame, composite


class TestEffectiveWeights:
    def test_it_reports_one_row_per_category(self) -> None:
        frame, composite = _universe()
        rows = scoring.effective_weights(frame, composite, profile="balanced")
        assert [r.category for r in rows] == list(CATEGORIES)

    def test_the_stated_weight_is_the_profile_s(self) -> None:
        frame, composite = _universe()
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["fundamental"].stated_weight == pytest.approx(0.25)
        assert rows["momentum"].stated_weight == pytest.approx(0.15)

    def test_correlated_categories_out_punch_their_weight(self) -> None:
        """The finding this panel exists to show.

        `technical` and `momentum` reinforce each other, so between them they
        steer the ordering more than their combined 0.35 of weight suggests —
        while `fundamental`, carrying more weight than either, moves it less.
        """
        frame, composite = _universe()
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["technical"].influence > rows["fundamental"].influence
        assert rows["technical"].stated_weight < rows["fundamental"].stated_weight

    def test_uncorrelated_categories_barely_move_the_ranking(self) -> None:
        frame, composite = _universe()
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["analyst"].influence < rows["technical"].influence / 2

    def test_influence_is_not_merely_the_weight_restated(self) -> None:
        """The load-bearing assertion.

        A variance-share measure reproduces the stated weights almost exactly,
        because every sub-score is a percentile and so every category has the
        same dispersion by construction. If the influence column can be
        recovered from the weight column, the panel is decoration.
        """
        frame, composite = _universe()
        rows = scoring.effective_weights(frame, composite, profile="balanced")
        by_weight = [r.category for r in sorted(rows, key=lambda r: -r.stated_weight)]
        by_influence = [r.category for r in sorted(rows, key=lambda r: -(r.influence or -1.0))]
        assert by_weight != by_influence, (by_weight, by_influence)

    def test_ranks_are_reported_so_a_reader_can_see_the_disagreement(self) -> None:
        frame, composite = _universe()
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["fundamental"].weight_rank == 1
        assert rows["technical"].influence_rank == 1
        assert rows["fundamental"].influence_rank > 1

    def test_a_category_with_no_data_has_no_influence_rather_than_none_of_it(self) -> None:
        """`sentiment` and `industry_macro` were both 0/503 on 2026-09-11 after
        the Tier-1/Tier-2 news step failed. An influence of 0.0 would read as
        "this category does nothing"; the truth is "we cannot say"."""
        frame, composite = _universe()
        frame["sentiment"] = np.nan
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["sentiment"].coverage == 0.0
        assert rows["sentiment"].influence is None
        assert rows["sentiment"].influence_rank is None
        # And it keeps its stated weight, because the weight is still stated.
        assert rows["sentiment"].stated_weight == pytest.approx(0.10)

    def test_coverage_is_reported_as_a_fraction(self) -> None:
        frame, composite = _universe(n=100)
        frame.loc[frame.index[:25], "smart_money"] = np.nan
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["smart_money"].coverage == pytest.approx(0.75)

    def test_too_few_names_yields_no_influence_rather_than_a_spurious_one(self) -> None:
        """A rank correlation over a handful of names has a confidence interval
        spanning most of [-1, 1]; printing its point estimate beside a weight
        invites exactly the false precision this project keeps removing."""
        frame, composite = _universe(n=5)
        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert all(r.influence is None for r in rows.values())

    def test_a_constant_category_has_no_influence_and_warns_nobody(self) -> None:
        """A correlation against a column with no variance is undefined.

        The absence of a warning is asserted, not decoration: scipy returns NaN
        *and* emits a `ConstantInputWarning` here, so catching the case only by
        testing NaN afterwards leaves the warning to surface in a page render or
        a CI log as an unexplained scipy complaint.
        """
        import warnings

        frame, composite = _universe()
        frame["analyst"] = 50.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rows = {
                r.category: r
                for r in scoring.effective_weights(frame, composite, profile="balanced")
            }
        assert rows["analyst"].influence is None

    def test_influence_is_a_rank_correlation_not_a_linear_one(self) -> None:
        """The composite's product is an *ordering* — the Screener is a ranked
        table and the ratings are percentile cutoffs — so how a category moves
        positions is the question, and a monotone relationship should read as
        full influence however curved it is.

        On percentile-ranked inputs the two measures agree closely, which is why
        the first version of this test could not tell them apart.
        """
        n = 120
        base = np.linspace(0.0, 100.0, n)
        frame = pd.DataFrame({c: base for c in CATEGORIES})
        # Monotone but sharply curved: same ordering, very different spacing.
        frame["analyst"] = base**3 / (100.0**2)
        composite = pd.Series(base, index=frame.index)

        rows = {
            r.category: r for r in scoring.effective_weights(frame, composite, profile="balanced")
        }
        assert rows["analyst"].influence == pytest.approx(1.0), (
            "a monotone category must read as full influence; Pearson would report ~0.92"
        )

    def test_an_empty_universe_is_handled(self) -> None:
        empty = pd.DataFrame(columns=list(CATEGORIES))
        rows = scoring.effective_weights(empty, pd.Series(dtype=float), profile="balanced")
        assert all(r.influence is None and r.coverage == 0.0 for r in rows)


class TestCategoryCorrelations:
    def test_it_finds_the_pair_that_is_one_signal_twice(self) -> None:
        """Measured at +0.685 between technical and momentum on real data —
        the second half of the finding, and the one that explains the first."""
        frame, _ = _universe()
        pairs = scoring.category_correlations(frame)
        assert pairs, "no pairs returned at all"
        top = pairs[0]
        assert {top.a, top.b} == {"technical", "momentum"}
        assert top.correlation > 0.5

    def test_pairs_are_ordered_by_absolute_strength(self) -> None:
        """Built so three orderings all disagree.

        Generation order walks `CATEGORIES` pairwise, so it is fixed; the three
        planted pairs are laid out such that sorting by generation, by signed
        correlation, and by absolute correlation each give a different answer.
        Without that, two mutations survived: removing the sort entirely (the
        strongest pair happened to be generated first) and sorting by signed
        value (every planted correlation happened to be positive).
        """
        rng = np.random.default_rng(11)
        n = 300
        frame = pd.DataFrame({c: rng.normal(size=n) for c in CATEGORIES})

        # Weakest, and generated FIRST.
        mild = rng.normal(size=n)
        frame["fundamental"] = mild + rng.normal(scale=0.9, size=n)
        frame["technical"] = mild + rng.normal(scale=0.9, size=n)
        # Strongest, NEGATIVE, generated in the middle.
        opposed = rng.normal(size=n)
        frame["analyst"] = opposed + rng.normal(scale=0.15, size=n)
        frame["sentiment"] = -opposed + rng.normal(scale=0.15, size=n)
        # Middling, positive, generated LAST.
        shared = rng.normal(size=n)
        frame["industry_macro"] = shared + rng.normal(scale=0.7, size=n)
        frame["smart_money"] = shared + rng.normal(scale=0.7, size=n)

        pairs = scoring.category_correlations(frame)
        shown = [(p.a, p.b, round(p.correlation, 3)) for p in pairs]
        assert len(pairs) >= 3, shown

        strengths = [abs(p.correlation) for p in pairs]
        assert strengths == sorted(strengths, reverse=True), shown
        # The strongest is the negative one, which signed ordering would put last.
        assert {pairs[0].a, pairs[0].b} == {"analyst", "sentiment"}, shown
        assert pairs[0].correlation < 0, shown

    def test_a_strong_negative_pair_is_reported(self) -> None:
        """Two categories that move *against* each other are double-counting as
        surely as two that move together — the sign says which, not whether."""
        rng = np.random.default_rng(5)
        n = 200
        frame = pd.DataFrame({c: rng.normal(size=n) for c in CATEGORIES})
        opposed = rng.normal(size=n)
        frame["fundamental"] = opposed
        frame["momentum"] = -opposed + rng.normal(scale=0.1, size=n)

        pairs = scoring.category_correlations(frame)
        top = pairs[0]
        assert {top.a, top.b} == {"fundamental", "momentum"}
        assert top.correlation < -0.9

    def test_a_weak_pair_is_left_out(self) -> None:
        frame, _ = _universe()
        pairs = scoring.category_correlations(frame, min_abs_correlation=0.99)
        assert pairs == []

    def test_a_category_with_no_data_forms_no_pair(self) -> None:
        frame, _ = _universe()
        frame["sentiment"] = np.nan
        pairs = scoring.category_correlations(frame)
        assert all("sentiment" not in (p.a, p.b) for p in pairs)

    def test_each_pair_appears_once_and_never_against_itself(self) -> None:
        frame, _ = _universe()
        pairs = scoring.category_correlations(frame, min_abs_correlation=0.0)
        seen = {frozenset({p.a, p.b}) for p in pairs}
        assert len(seen) == len(pairs)
        assert all(p.a != p.b for p in pairs)

    def test_the_overlap_each_pair_was_measured_over_is_reported(self) -> None:
        """Two categories can each be well covered and still share few names."""
        frame, _ = _universe(n=100)
        frame.loc[frame.index[:60], "smart_money"] = np.nan
        pairs = scoring.category_correlations(frame, min_abs_correlation=0.0)
        smart = next(p for p in pairs if "smart_money" in (p.a, p.b))
        assert smart.overlap == 40
