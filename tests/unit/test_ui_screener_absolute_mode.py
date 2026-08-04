"""The Screener's absolute-rating path (Sections 7.5 step 4, 22).

`rating_mode="absolute"` was correct code that no interface could reach, because
`composite_scores` stored only the percentiled sub-scores and a rank cannot be
turned back into an absolute reading. Storing the raw category values closed
that, and these tests cover the half that actually makes the mode usable: the
page re-scoring stored rows against the fixed bar, and declining honestly when
the rows predate the raw columns.

The page script can't be imported normally (`pages/1_Screener.py` starts with a
digit and calls `main()` at module scope), so it is loaded with the same
ast-strip-and-exec trick `test_ui_screener_reweight_properties.py` uses.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pandas as pd
import pytest

from quantpulse.analysis.investor_profiles import CATEGORIES, get_profile

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
    module = types.ModuleType("screener_page_absolute_mode")
    module.__file__ = str(_PAGE_PATH)
    exec(compile(tree, str(_PAGE_PATH), "exec"), module.__dict__)
    return module


_PAGE = _load_page()
_WEIGHTS = dict(get_profile("balanced").weights)


def _rows(technical_values: list[float], *, with_raw: bool = True) -> pd.DataFrame:
    symbols = [f"S{i}" for i in range(len(technical_values))]
    frame = pd.DataFrame({"symbol": symbols})
    for category in CATEGORIES:
        frame[f"{category}_score"] = 50.0
        if with_raw:
            frame[f"{category}_raw"] = None
    frame["technical_score"] = technical_values
    if with_raw:
        frame["technical_raw"] = technical_values
    frame["composite_score"] = technical_values
    frame["rating"] = "hold"
    frame["data_confidence"] = 20.0
    return frame


class TestAbsoluteRescore:
    def test_a_uniformly_strong_universe_earns_no_sells(self) -> None:
        # `technical` is a fixed 0-100 reading; 50 is neutral. Every name here
        # is genuinely good, so an absolute judgment should say so -- unlike the
        # relative ranking, which must always name a bottom decile.
        out = _PAGE.rescore_absolute(_rows([70.0 + i for i in range(20)]), _WEIGHTS, "balanced")
        assert out is not None
        assert "sell" not in set(out["rating"])
        assert "strong_sell" not in set(out["rating"])

    def test_a_uniformly_weak_universe_earns_no_strong_buys(self) -> None:
        out = _PAGE.rescore_absolute(
            _rows([5.0 + i * 0.5 for i in range(20)]), _WEIGHTS, "balanced"
        )
        assert out is not None
        assert "strong_buy" not in set(out["rating"])

    def test_it_keeps_every_row_and_adds_a_custom_score(self) -> None:
        rows = _rows([60.0, 70.0, 80.0])
        out = _PAGE.rescore_absolute(rows, _WEIGHTS, "balanced")
        assert out is not None
        assert set(out["symbol"]) == set(rows["symbol"])
        assert "custom_score" in out.columns
        assert out["custom_score"].notna().all()

    def test_declines_when_the_rows_predate_the_raw_columns(self) -> None:
        """Honesty over a plausible-looking answer.

        Rows written before the raw columns existed cannot support an absolute
        rating at all. Returning None makes the page say so; silently showing
        the relative ratings under an "absolute" label would be exactly the
        mislabelling this mode was fixed to stop.
        """
        assert (
            _PAGE.rescore_absolute(_rows([60.0, 70.0], with_raw=False), _WEIGHTS, "balanced")
            is None
        )

    def test_declines_when_the_raw_columns_exist_but_are_all_empty(self) -> None:
        rows = _rows([60.0, 70.0])
        for category in CATEGORIES:
            rows[f"{category}_raw"] = None
        assert _PAGE.rescore_absolute(rows, _WEIGHTS, "balanced") is None

    def test_zero_weights_fall_back_to_the_profile_rather_than_dividing_by_zero(self) -> None:
        out = _PAGE.rescore_absolute(
            _rows([60.0, 70.0, 80.0]), dict.fromkeys(CATEGORIES, 0.0), "balanced"
        )
        assert out is not None
        assert out["custom_score"].notna().all()


class TestRawValuesSurviveTheRoundTrip:
    def test_build_composite_emits_the_raw_inputs_unchanged(self) -> None:
        from quantpulse.analysis import scoring

        raw = pd.DataFrame(
            {"technical": [60.0, 40.0, 80.0], "momentum": [0.05, -0.02, 0.10]},
            index=["A", "B", "C"],
        )
        scored = scoring.build_composite(raw).scores.set_index("symbol")
        # The percentiled column is a rank; the raw column is the input itself.
        assert scored.loc["C", "technical_score"] == pytest.approx(100.0)
        assert scored.loc["C", "technical_raw"] == pytest.approx(80.0)
        assert scored.loc["B", "momentum_raw"] == pytest.approx(-0.02)

    def test_absent_categories_yield_null_raw_values_not_zeros(self) -> None:
        from quantpulse.analysis import scoring

        raw = pd.DataFrame({"technical": [60.0, 40.0]}, index=["A", "B"])
        scored = scoring.build_composite(raw).scores.set_index("symbol")
        assert scored["fundamental_raw"].isna().all()
