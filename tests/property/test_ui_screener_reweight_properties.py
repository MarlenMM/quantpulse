"""Property-based tests for the Screener page's client-side re-scoring (Section 29).

`app/pages/1_Screener.py`'s sliders recompute the composite score *and* the
rating from already-stored, already-normalized sub-scores. It used to do the
weighting itself, in a hand-rolled `reweight` helper that mirrored
`scoring.build_composite`'s coverage rule -- and mirrored it only for the score,
leaving the Rating column and the Rating-mix chart showing the stored
balanced-profile verdict however hard the sliders were pushed.

Both halves now delegate to `scoring.build_composite`, so there is one
implementation of the weighting rather than two that can drift. These tests pin
that delegation directly: for random sub-scores and random weights the page's
output must equal the engine's, and the coverage-renormalization invariants the
old tests asserted must therefore still hold through it.

The page script can't be `import`ed normally (`pages/1_Screener.py` starts with
a digit and isn't a package), and importing it naively would execute the whole
Streamlit page -- including a live read from the real on-disk database, via the
unconditional `main()` call at the bottom. `_load_page` below loads just the
pure helpers by parsing the module and dropping that one `main()` statement.
"""

import ast
import pathlib
import types

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.analysis import scoring
from quantpulse.analysis.investor_profiles import CATEGORIES

_PAGE_PATH = pathlib.Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Screener.py"


def _load_page() -> types.ModuleType:
    source = _PAGE_PATH.read_text()
    tree = ast.parse(source, filename=str(_PAGE_PATH))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
        )
    ]
    module = types.ModuleType("screener_page_under_test")
    module.__file__ = str(_PAGE_PATH)
    exec(compile(tree, str(_PAGE_PATH), "exec"), module.__dict__)
    return module


page = _load_page()

_SUBSCORE = st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
_WEIGHT = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False)


def _rows_and_weights_strategy(
    *, n_symbols: int = 6
) -> st.SearchStrategy[tuple[pd.DataFrame, dict[str, float]]]:
    subscore_column = st.lists(_SUBSCORE, min_size=n_symbols, max_size=n_symbols)
    columns = st.fixed_dictionaries({f"{c}_score": subscore_column for c in CATEGORIES})
    # Every category carries a weight: the page renormalizes the full slider set
    # into an `InvestorProfile`, which requires all seven.
    weights = st.fixed_dictionaries(dict.fromkeys(CATEGORIES, _WEIGHT))

    def _build(cols: dict, w: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
        frame = pd.DataFrame(cols, dtype=float)
        frame.insert(0, "symbol", [f"S{i}" for i in range(n_symbols)])
        frame["composite_score"] = frame[[f"{c}_score" for c in CATEGORIES]].mean(axis=1)
        # `read_screener_rows` always supplies a stored rating; the page keeps
        # it when nothing can be re-scored (a universe with no data at all).
        frame["rating"] = "hold"
        return frame, w

    return st.tuples(columns, weights).map(lambda pair: _build(*pair))


def _engine_result(rows: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """What `scoring.build_composite` says, called directly rather than via the page."""
    sub = rows[[f"{c}_score" for c in CATEGORIES]].copy()
    sub.columns = list(CATEGORIES)
    sub.index = rows["symbol"]
    return scoring.build_composite(
        sub, profile=page._custom_profile(weights, "balanced"), rating_mode="relative"
    ).scores.set_index("symbol")


class TestRelativeRescoringDelegatesToTheEngine:
    @given(_rows_and_weights_strategy())
    def test_score_and_rating_match_build_composite_exactly(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        rows, weights = data
        page_result = page.rescore_relative(rows, weights, "balanced").set_index("symbol")
        engine = _engine_result(rows, weights)
        for symbol in engine.index:
            assert page_result.loc[symbol, "custom_score"] == pytest.approx(
                engine.loc[symbol, "composite_score"], rel=1e-9, abs=1e-9
            )
            assert page_result.loc[symbol, "rating"] == engine.loc[symbol, "rating"]

    @given(_rows_and_weights_strategy())
    def test_coverage_renormalization_survives_the_delegation(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        # The phantom-zero failure mode the old hand-rolled helper existed to
        # avoid, asserted through the page as it now stands: a scored symbol's
        # composite is a convex combination of the sub-scores it actually has,
        # never dragged toward zero by the ones it doesn't. (`build_composite`
        # percentiles first, so the bound is over the percentiled values.)
        rows, weights = data
        page_result = page.rescore_relative(rows, weights, "balanced").set_index("symbol")
        engine = _engine_result(rows, weights)
        for symbol in engine.index:
            present = engine.loc[symbol, [f"{c}_score" for c in CATEGORIES]].dropna()
            assert not present.empty
            score = page_result.loc[symbol, "custom_score"]
            assert present.min() - 1e-6 <= score <= present.max() + 1e-6

    @given(_rows_and_weights_strategy())
    def test_ratings_are_monotone_in_the_recomputed_score(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        # A relative rating is a ranking, so a higher recomputed score can never
        # earn a worse rating than a lower one. This is the coherence the page
        # lacked entirely while the Rating column showed a stored verdict.
        rows, weights = data
        result = page.rescore_relative(rows, weights, "balanced").dropna(subset=["custom_score"])
        order = {rating: i for i, rating in enumerate(scoring.RATINGS)}  # best -> worst
        ranked = result.sort_values("custom_score", ascending=False)
        ranks = [order[r] for r in ranked["rating"]]
        assert ranks == sorted(ranks)


class TestRelativeRescoringEdgeCases:
    @staticmethod
    def _rows(**overrides: list[float | None]) -> pd.DataFrame:
        rows = pd.DataFrame(
            {"symbol": ["A", "B"], "composite_score": [50.0, 60.0], "rating": ["hold", "hold"]}
        )
        for category in CATEGORIES:
            rows[f"{category}_score"] = overrides.get(category, [None, None])
        return rows

    def test_a_lone_present_category_scores_exactly_its_own_percentile(self) -> None:
        rows = self._rows(fundamental=[20.0, 80.0])
        result = page.rescore_relative(rows, dict.fromkeys(CATEGORIES, 1.0), "balanced")
        # Only one category has data, so the composite is exactly that
        # category's percentile: two names -> 50 and 100.
        assert sorted(result["custom_score"]) == pytest.approx([50.0, 100.0])

    def test_a_symbol_with_no_data_at_all_is_dropped_not_fabricated(self) -> None:
        rows = self._rows(fundamental=[10.0, None])
        result = page.rescore_relative(rows, dict.fromkeys(CATEGORIES, 1.0), "balanced").set_index(
            "symbol"
        )
        assert pd.notna(result.loc["A", "custom_score"])
        assert pd.isna(result.loc["B", "custom_score"])
        assert pd.isna(result.loc["B", "rating"])

    def test_a_risk_off_regime_tightens_the_strong_buy_cutoff(self) -> None:
        # The Tier-3 dampener has to reach a slider-driven rating too, or a
        # re-weighted table hands out Strong Buys the stored ranking withheld.
        rows = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(20)],
                "composite_score": [float(i) for i in range(20)],
                "rating": ["hold"] * 20,
            }
        )
        for category in CATEGORIES:
            rows[f"{category}_score"] = [float(i) for i in range(20)]
        weights = dict.fromkeys(CATEGORIES, 1.0)
        calm = page.rescore_relative(rows, weights, "balanced", regime_score=50.0)
        risk_off = page.rescore_relative(rows, weights, "balanced", regime_score=0.0)
        assert (risk_off["rating"] == "strong_buy").sum() < (calm["rating"] == "strong_buy").sum()
