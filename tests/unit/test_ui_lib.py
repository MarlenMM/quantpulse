"""Tests for the Streamlit app's shared helpers (`app/lib`).

Importable because `pyproject.toml` puts `app` on pytest's `pythonpath`. Only
`format.py` and `charts.py` are covered here: they are pure (strings in/out,
data in/`go.Figure` out) and hold all the display logic worth pinning.
`lib/data.py` is deliberately not tested directly -- it is a set of one-line
`@st.cache_data` wrappers whose only real content is the `storage.persistence`
readers, and those are tested against a real database in `test_persistence.py`.
"""

from datetime import date

import pandas as pd
import pytest

from lib import charts
from lib.format import (
    RATING_DISPLAY,
    RATING_ORDER,
    action_label,
    confidence_label,
    format_money,
    format_pct_already_scaled,
    format_percent,
    format_price,
    format_ratio,
    format_score,
    format_signed_percent,
    freshness_label,
    humanize,
    rating_color,
    rating_label,
)


class TestRatingDisplay:
    @pytest.mark.parametrize("rating", RATING_ORDER)
    def test_every_rating_pairs_an_icon_with_words(self, rating: str) -> None:
        # Section 12: never encode Buy/Sell with color alone -- roughly 1 in 12
        # men cannot distinguish the red/green that would otherwise carry it.
        label = rating_label(rating)
        icon, text, _ = RATING_DISPLAY[rating]
        assert icon and icon in label
        assert text in label

    def test_every_rating_has_a_distinct_icon_direction(self) -> None:
        icons = {RATING_DISPLAY[r][0] for r in RATING_ORDER}
        assert len(icons) == len(RATING_ORDER)  # legible without color

    def test_unknown_and_missing_ratings_degrade(self) -> None:
        assert rating_label(None) == "—"
        assert rating_label("") == "—"
        assert "Weird Thing" in rating_label("weird_thing")

    def test_color_is_decoration_with_a_neutral_fallback(self) -> None:
        assert rating_color("buy").startswith("#")
        assert rating_color(None) == "#6e7781"
        assert rating_color("nonsense") == "#6e7781"

    def test_action_labels_pair_icon_and_word(self) -> None:
        for action in ("add", "hold", "trim", "sell"):
            label = action_label(action)
            assert any(ch in label for ch in "▲■▼")
            assert label.split(" ", 1)[1]
        assert action_label(None) == "—"


class TestNumberFormatting:
    def test_none_is_always_an_em_dash_not_a_zero(self) -> None:
        # A missing number must never render as 0 -- that is a different claim.
        assert format_price(None) == "—"
        assert format_money(None) == "—"
        assert format_percent(None) == "—"
        assert format_signed_percent(None) == "—"
        assert format_score(None) == "—"
        assert format_ratio(None) == "—"
        assert format_pct_already_scaled(None) == "—"

    def test_price_and_money(self) -> None:
        assert format_price(1234.567) == "$1,234.57"
        assert format_money(1234.567) == "$1,235"

    def test_percent_and_signed_percent(self) -> None:
        assert format_percent(0.1534) == "15.3%"
        assert format_signed_percent(0.1534) == "+15.3%"
        assert format_signed_percent(-0.04) == "-4.0%"

    def test_pct_already_scaled_does_not_multiply_by_100(self) -> None:
        # The bug this guards against: compute_breadth's "share, 0-100" (e.g.
        # 62.0) fed through format_percent's *100 convention would render as
        # the nonsensical "6200.0%".
        assert format_pct_already_scaled(62.0) == "62.0%"
        assert format_pct_already_scaled(62.0421, digits=2) == "62.04%"

    def test_score_and_ratio_digits(self) -> None:
        assert format_score(87.34) == "87.3"
        assert format_score(87.34, digits=0) == "87"
        assert format_ratio(0.7812) == "0.78"

    def test_humanize(self) -> None:
        assert humanize("strong_buy") == "Strong Buy"
        assert humanize(None) == "—"


class TestFreshness:
    def test_never_run_is_distinct_from_stale(self) -> None:
        # Section 12: an empty pipeline and a day-old one must not look alike.
        assert freshness_label(None) == "never run"
        assert freshness_label(date(2026, 7, 20), today=date(2026, 7, 27)) == "7 days ago"

    def test_today_and_yesterday_read_naturally(self) -> None:
        today = date(2026, 7, 27)
        assert freshness_label(today, today=today) == "today"
        assert freshness_label(date(2026, 7, 26), today=today) == "yesterday"

    def test_future_date_is_shown_literally(self) -> None:
        assert freshness_label(date(2026, 8, 1), today=date(2026, 7, 27)) == "2026-08-01"


class TestConfidence:
    def test_bands_are_worded_not_bare_numbers(self) -> None:
        assert "good" in confidence_label(90)
        assert "partial" in confidence_label(60)
        assert "thin" in confidence_label(20)
        assert confidence_label(None) == "coverage unknown"

    def test_band_boundaries(self) -> None:
        assert "good" in confidence_label(80)
        assert "partial" in confidence_label(79.9)
        assert "partial" in confidence_label(50)
        assert "thin" in confidence_label(49.9)


class TestCharts:
    def _ohlcv(self, n: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=n, freq="B")
        close = pd.Series(range(100, 100 + n), dtype=float)
        return pd.DataFrame(
            {
                "date": idx,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000,
            }
        )

    def test_empty_figure_carries_an_explanation(self) -> None:
        fig = charts.empty_figure("nothing here")
        assert fig.layout.annotations[0].text == "nothing here"

    def test_no_title_does_not_leave_an_undefined_title(self) -> None:
        # Plotly renders the literal string "undefined" if handed title=None.
        fig = charts.empty_figure()
        assert fig.layout.title.text is None

    def test_title_is_applied_when_given(self) -> None:
        fig = charts.allocation_pie({"Tech": 0.6, "Energy": 0.4}, title="By sector")
        assert fig.layout.title.text == "By sector"

    def test_price_chart_has_candles_and_overlays(self) -> None:
        bars = self._ohlcv()
        fig = charts.price_chart(bars, overlays={"SMA 10": bars["close"].rolling(10).mean()})
        kinds = [trace.type for trace in fig.data]
        assert "candlestick" in kinds
        assert "scatter" in kinds

    def test_price_chart_without_data_explains_itself(self) -> None:
        fig = charts.price_chart(pd.DataFrame())
        assert "No price history" in fig.layout.annotations[0].text

    def test_radar_drops_missing_categories_rather_than_zeroing_them(self) -> None:
        # A missing sentiment score is not a *bad* sentiment score; plotting it
        # at the origin would say the opposite of what the data says.
        fig = charts.subscore_radar(
            {"technical": 80.0, "fundamental": 60.0, "sentiment": None, "momentum": 70.0}
        )
        assert "Sentiment" not in list(fig.data[0].theta)
        assert "Technical" in list(fig.data[0].theta)

    def test_radar_closes_the_polygon(self) -> None:
        fig = charts.subscore_radar({"a": 10.0, "b": 20.0, "c": 30.0})
        theta = list(fig.data[0].theta)
        assert theta[0] == theta[-1]

    def test_radar_needs_three_categories(self) -> None:
        fig = charts.subscore_radar({"a": 10.0, "b": 20.0})
        assert "Not enough scored categories" in fig.layout.annotations[0].text

    def test_forecast_fan_draws_a_band(self) -> None:
        forecasts = pd.DataFrame(
            {
                "model_name": ["gbr"] * 3,
                "horizon_days": [5, 20, 63],
                "point_price": [105.0, 110.0, 120.0],
                "lower_price": [100.0, 101.0, 102.0],
                "upper_price": [110.0, 120.0, 140.0],
                "generated_date": [date(2026, 7, 27)] * 3,
            }
        )
        fig = charts.forecast_fan_chart(self._ohlcv(), forecasts, model_name="gbr")
        names = [t.name for t in fig.data]
        assert "Forecast range" in names
        assert "Forecast" in names

    def test_forecast_fan_without_bounds_omits_the_band(self) -> None:
        forecasts = pd.DataFrame(
            {
                "model_name": ["gbr"],
                "horizon_days": [5],
                "point_price": [105.0],
                "lower_price": [None],
                "upper_price": [None],
                "generated_date": [date(2026, 7, 27)],
            }
        )
        fig = charts.forecast_fan_chart(pd.DataFrame(), forecasts)
        assert "Forecast range" not in [t.name for t in fig.data]

    def test_forecast_fan_unknown_model_says_so(self) -> None:
        forecasts = pd.DataFrame(
            {
                "model_name": ["gbr"],
                "horizon_days": [5],
                "point_price": [105.0],
                "lower_price": [100.0],
                "upper_price": [110.0],
                "generated_date": [date(2026, 7, 27)],
            }
        )
        fig = charts.forecast_fan_chart(pd.DataFrame(), forecasts, model_name="arima")
        assert "arima" in fig.layout.annotations[0].text

    def test_allocation_pie_drops_non_positive_slices(self) -> None:
        fig = charts.allocation_pie({"Tech": 0.5, "Energy": 0.5, "Empty": 0.0})
        assert "Empty" not in list(fig.data[0].labels)

    def test_regime_gauge_none_explains_itself(self) -> None:
        fig = charts.regime_gauge(None)
        assert "Market Regime Index" in fig.layout.annotations[0].text

    def test_regime_gauge_shows_value_and_label(self) -> None:
        fig = charts.regime_gauge(72.5, "risk_on")
        assert fig.data[0].value == pytest.approx(72.5)
        assert fig.data[0].title.text == "Risk On"

    def test_correlation_heatmap_bounds_the_scale(self) -> None:
        matrix = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["A", "B"], columns=["A", "B"])
        fig = charts.correlation_heatmap(matrix)
        assert fig.data[0].zmin == -1 and fig.data[0].zmax == 1

    def test_correlation_heatmap_empty(self) -> None:
        fig = charts.correlation_heatmap(pd.DataFrame())
        assert "overlapping price history" in fig.layout.annotations[0].text

    def test_rating_distribution_is_labelled_not_color_only(self) -> None:
        fig = charts.rating_distribution({"buy": 3, "hold": 5})
        labels = list(fig.data[0].x)
        assert any("Buy" in label for label in labels)
        assert any("Strong Sell" in label for label in labels)  # zero-count still listed

    def test_rating_distribution_empty(self) -> None:
        fig = charts.rating_distribution({})
        assert "No ratings" in fig.layout.annotations[0].text

    def test_equity_curve_with_benchmark(self) -> None:
        idx = pd.date_range("2026-01-01", periods=5, freq="ME")
        fig = charts.equity_curve_chart(
            pd.Series([1.0, 1.1, 1.2, 1.15, 1.3], index=idx),
            pd.Series([1.0, 1.05, 1.08, 1.06, 1.1], index=idx),
        )
        assert [t.name for t in fig.data] == ["Strategy", "Benchmark"]

    def test_equity_curve_empty(self) -> None:
        fig = charts.equity_curve_chart(pd.Series(dtype=float))
        assert "No backtest" in fig.layout.annotations[0].text

    def test_figures_are_theme_transparent(self) -> None:
        # Transparent backgrounds let one set of figures work in both the light
        # and dark themes (Section 31) instead of looking wrong in one.
        fig = charts.allocation_pie({"Tech": 1.0})
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
