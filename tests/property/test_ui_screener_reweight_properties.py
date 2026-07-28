"""Property-based tests for the Screener page's client-side `reweight` (Section 29).

`app/pages/1_Screener.py`'s `reweight` recomputes the composite score from
already-stored, already-normalized sub-scores under caller-supplied slider
weights -- deliberately mirroring `scoring.build_composite`'s coverage-
renormalization step (its own docstring says so) so the sliders can't quietly
disagree with the stored ranking. It had zero tests before this file: the page
script can't be `import`ed normally (`pages/1_Screener.py` starts with a digit
and isn't a package), and importing it naively would execute the whole
Streamlit page -- including a live read from the real on-disk database, via
the unconditional `main()` call at the bottom of the file. `_load_reweight`
below loads just the pure helper (and its `SCORE_COLUMNS` dependency) by
parsing the module and dropping that one `main()` statement, so this file can
exercise the real production function without any of that.
"""

import ast
import pathlib
import types
from collections.abc import Callable

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantpulse.analysis.investor_profiles import CATEGORIES

_PAGE_PATH = pathlib.Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Screener.py"


def _load_reweight() -> Callable[[pd.DataFrame, dict[str, float]], pd.Series]:
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
    return module.reweight


reweight = _load_reweight()

_SUBSCORE = st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
_WEIGHT = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False)


def _rows_and_weights_strategy(
    *, n_symbols: int = 6
) -> st.SearchStrategy[tuple[pd.DataFrame, dict[str, float]]]:
    subscore_column = st.lists(_SUBSCORE, min_size=n_symbols, max_size=n_symbols)
    columns = st.fixed_dictionaries({f"{c}_score": subscore_column for c in CATEGORIES})
    weights = st.dictionaries(
        st.sampled_from(CATEGORIES), _WEIGHT, min_size=1, max_size=len(CATEGORIES)
    )

    def _build(cols: dict, w: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
        rows = pd.DataFrame(cols, index=[f"S{i}" for i in range(n_symbols)], dtype=float)
        return rows, w

    return st.tuples(columns, weights).map(lambda pair: _build(*pair))


class TestReweightCoverageRenormalization:
    @given(_rows_and_weights_strategy())
    def test_bounded_by_the_present_subscores_it_averages(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        rows, weights = data
        out = reweight(rows, weights)
        for symbol in rows.index:
            present = rows.loc[symbol, [f"{c}_score" for c in weights]].dropna()
            if present.empty:
                assert pd.isna(out[symbol])
            else:
                assert present.min() - 1e-6 <= out[symbol] <= present.max() + 1e-6

    @given(_rows_and_weights_strategy())
    def test_matches_the_hand_rolled_coverage_renormalization_formula(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        rows, weights = data
        out = reweight(rows, weights)
        for symbol in rows.index:
            available = 0.0
            weighted = 0.0
            for category, weight in weights.items():
                value = rows.loc[symbol, f"{category}_score"]
                if pd.notna(value):
                    available += weight
                    weighted += weight * value
            if available <= 0:
                assert pd.isna(out[symbol])
            else:
                assert out[symbol] == pytest.approx(weighted / available, rel=1e-6, abs=1e-6)

    @given(_rows_and_weights_strategy())
    def test_a_lone_remaining_category_scores_exactly_its_own_subscore(
        self, data: tuple[pd.DataFrame, dict[str, float]]
    ) -> None:
        # The phantom-zero failure mode this function exists to avoid: a
        # symbol with data in exactly one weighted category must composite to
        # exactly that category's stored sub-score, never something dragged
        # toward zero by the OTHER weighted categories' missing data.
        rows, weights = data
        out = reweight(rows, weights)
        for symbol in rows.index:
            present = rows.loc[symbol, [f"{c}_score" for c in weights]].dropna()
            if len(present) == 1:
                assert out[symbol] == pytest.approx(present.iloc[0], rel=1e-6, abs=1e-6)

    def test_all_categories_present_is_the_plain_weighted_average(self) -> None:
        rows = pd.DataFrame({f"{c}_score": [80.0] for c in CATEGORIES}, index=["A"])
        rows["fundamental_score"] = 20.0
        weights = {"fundamental": 1.0, "technical": 1.0}
        rows["technical_score"] = 80.0
        out = reweight(rows, weights)
        assert out["A"] == pytest.approx(50.0)

    def test_zero_available_weight_is_nan_not_a_fabricated_score(self) -> None:
        rows = pd.DataFrame({"fundamental_score": [50.0]}, index=["A"])
        out = reweight(rows, {"fundamental": 0.0})
        assert pd.isna(out["A"])
