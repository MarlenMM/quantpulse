"""Tests for the Stock Detail page's LLM-context builders (Sections 10, 11).

`rating_narrative()` and `forecast_narrative()` decide exactly what the
optional summary paragraph and the chat box are allowed to see, so they are
worth pinning even though they live in a Streamlit page: a silent change here
(a NaN leaking through as a number, the wrong model's forecasts being
attached) would ground the model in something the page isn't showing.

The page can't be `import`ed normally -- `pages/2_Stock_Detail.py` starts with
a digit and isn't a package, and importing it would execute the whole page
including a live database read via its module-level `main()` call. The same
targeted AST strip used in `tests/property/test_ui_screener_reweight_properties.py`
loads just the pure helpers.
"""

from __future__ import annotations

import ast
import pathlib
import types
from datetime import date

import pandas as pd
import pytest

from quantpulse.analysis.investor_profiles import CATEGORIES

_PAGE_PATH = pathlib.Path(__file__).resolve().parents[2] / "app" / "pages" / "2_Stock_Detail.py"


def _load_page() -> types.ModuleType:
    tree = ast.parse(_PAGE_PATH.read_text(), filename=str(_PAGE_PATH))
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
    module = types.ModuleType("stock_detail_page_under_test")
    module.__file__ = str(_PAGE_PATH)
    exec(compile(tree, str(_PAGE_PATH), "exec"), module.__dict__)
    return module


page = _load_page()


def _row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "rating": "strong_buy",
        "composite_score": 87.3,
        "percentile_rank": 96.2,
        "data_confidence": 84.0,
        "date": date(2026, 7, 27),
    }
    values.update({f"{category}_score": 70.0 for category in CATEGORIES})
    values.update(overrides)
    return pd.Series(values)


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "gbr",
                "horizon_days": 20,
                "point_return": -0.025,
                "point_price": 120.3,
                "lower_price": None,
                "upper_price": None,
                "historical_hit_rate": None,
            },
            {
                "model_name": "gbr",
                "horizon_days": 5,
                "point_return": 0.0123,
                "point_price": 124.9,
                "lower_price": 120.1,
                "upper_price": 129.8,
                "historical_hit_rate": 0.58,
            },
            {
                "model_name": "arima",
                "horizon_days": 5,
                "point_return": 0.99,
                "point_price": 999.0,
                "lower_price": 1.0,
                "upper_price": 2.0,
                "historical_hit_rate": 0.91,
            },
        ]
    )


class TestRatingNarrative:
    def test_carries_the_rows_numbers_through(self) -> None:
        context = page.rating_narrative("NVDA", _row())
        assert context.symbol == "NVDA"
        assert context.rating == "strong_buy"
        assert context.composite_score == pytest.approx(87.3)
        assert context.percentile_rank == pytest.approx(96.2)
        assert context.data_confidence == pytest.approx(84.0)
        assert context.as_of == date(2026, 7, 27)

    def test_every_category_is_represented(self) -> None:
        context = page.rating_narrative("NVDA", _row())
        assert set(context.sub_scores) == set(CATEGORIES)

    def test_missing_subscore_becomes_none_not_nan_or_zero(self) -> None:
        # A NaN reaching the builder would render as "nan" in the context block;
        # a zero would tell the model the category scored badly rather than that
        # it had no data at all (Section 7.5's coverage honesty).
        context = page.rating_narrative("NVDA", _row(sentiment_score=float("nan")))
        assert context.sub_scores["sentiment"] is None
        assert context.sub_scores["technical"] == pytest.approx(70.0)

    def test_missing_optional_metrics_become_none(self) -> None:
        context = page.rating_narrative(
            "NVDA", _row(percentile_rank=float("nan"), data_confidence=float("nan"))
        )
        assert context.percentile_rank is None
        assert context.data_confidence is None


class TestForecastNarrative:
    def test_only_the_selected_models_rows_are_attached(self) -> None:
        # Grounding the chat in a model the user didn't select would have it
        # quoting forecasts the table above is not showing.
        context = page.forecast_narrative("NVDA", _forecast_frame(), "gbr", 123.45)
        assert context.model_name == "gbr"
        assert len(context.horizons) == 2
        assert all(h.point_return != pytest.approx(0.99) for h in context.horizons)

    def test_horizons_are_ordered_shortest_first(self) -> None:
        context = page.forecast_narrative("NVDA", _forecast_frame(), "gbr", 123.45)
        assert [h.horizon_days for h in context.horizons] == [5, 20]

    def test_ungraded_hit_rate_stays_none(self) -> None:
        context = page.forecast_narrative("NVDA", _forecast_frame(), "gbr", 123.45)
        by_horizon = {h.horizon_days: h for h in context.horizons}
        assert by_horizon[5].historical_hit_rate == pytest.approx(0.58)
        assert by_horizon[20].historical_hit_rate is None

    def test_missing_band_prices_stay_none(self) -> None:
        context = page.forecast_narrative("NVDA", _forecast_frame(), "gbr", 123.45)
        by_horizon = {h.horizon_days: h for h in context.horizons}
        assert by_horizon[20].lower_price is None
        assert by_horizon[20].upper_price is None
        assert by_horizon[5].lower_price == pytest.approx(120.1)

    def test_last_close_may_be_absent(self) -> None:
        context = page.forecast_narrative("NVDA", _forecast_frame(), "gbr", None)
        assert context.last_close is None
