import pandas as pd
import pytest

from quantpulse.analysis.analyst_consensus import (
    compute_price_target_upside,
    compute_rating_score,
    score_analyst_consensus,
)


class TestComputeRatingScore:
    def test_unanimous_strong_buy_is_100(self) -> None:
        assert compute_rating_score(10, 0, 0, 0, 0) == 100.0

    def test_unanimous_strong_sell_is_0(self) -> None:
        assert compute_rating_score(0, 0, 0, 0, 10) == 0.0

    def test_unanimous_hold_is_neutral_midpoint(self) -> None:
        assert compute_rating_score(0, 0, 10, 0, 0) == 50.0

    def test_no_analyst_coverage_is_none(self) -> None:
        assert compute_rating_score(0, 0, 0, 0, 0) is None

    def test_mixed_ratings_weighted_correctly(self) -> None:
        # 6 SB, 23 B, 14 H, 2 S, 2 SS -> weighted average
        score = compute_rating_score(6, 23, 14, 2, 2)
        expected = (6 * 100 + 23 * 75 + 14 * 50 + 2 * 25 + 2 * 0) / 47
        assert score == pytest.approx(expected)


class TestComputePriceTargetUpside:
    def test_positive_upside(self) -> None:
        assert compute_price_target_upside(100, 110) == pytest.approx(10.0)

    def test_negative_downside(self) -> None:
        assert compute_price_target_upside(100, 90) == pytest.approx(-10.0)

    def test_none_without_current_price(self) -> None:
        assert compute_price_target_upside(None, 110) is None

    def test_none_without_target(self) -> None:
        assert compute_price_target_upside(100, None) is None

    def test_none_for_non_positive_price(self) -> None:
        assert compute_price_target_upside(0, 110) is None


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestScoreAnalystConsensus:
    def test_same_final_level_but_improving_trend_scores_higher_than_worsening(self) -> None:
        dates = pd.date_range("2026-04-01", periods=4, freq="30D")
        # Both end at an identical final-day rating distribution (6 SB, 23 B,
        # 14 H, 2 S, 2 SS) and identical final price target -- only the path
        # getting there differs. This isolates the trend's effect cleanly.
        final_row = {
            "strong_buy": 6,
            "buy": 23,
            "hold": 14,
            "sell": 2,
            "strong_sell": 2,
            "mean_price_target": 110,
        }

        improving = _history(
            [
                {
                    "as_of_date": dates[0],
                    "strong_buy": 1,
                    "buy": 10,
                    "hold": 25,
                    "sell": 8,
                    "strong_sell": 3,
                    "mean_price_target": 90,
                },
                {
                    "as_of_date": dates[1],
                    "strong_buy": 2,
                    "buy": 15,
                    "hold": 20,
                    "sell": 5,
                    "strong_sell": 3,
                    "mean_price_target": 98,
                },
                {
                    "as_of_date": dates[2],
                    "strong_buy": 4,
                    "buy": 20,
                    "hold": 16,
                    "sell": 3,
                    "strong_sell": 2,
                    "mean_price_target": 104,
                },
                {"as_of_date": dates[3], **final_row},
            ]
        )
        worsening = _history(
            [
                {
                    "as_of_date": dates[0],
                    "strong_buy": 12,
                    "buy": 24,
                    "hold": 6,
                    "sell": 1,
                    "strong_sell": 1,
                    "mean_price_target": 130,
                },
                {
                    "as_of_date": dates[1],
                    "strong_buy": 9,
                    "buy": 24,
                    "hold": 9,
                    "sell": 1,
                    "strong_sell": 1,
                    "mean_price_target": 122,
                },
                {
                    "as_of_date": dates[2],
                    "strong_buy": 7,
                    "buy": 24,
                    "hold": 11,
                    "sell": 2,
                    "strong_sell": 2,
                    "mean_price_target": 116,
                },
                {"as_of_date": dates[3], **final_row},
            ]
        )

        result_improving = score_analyst_consensus(improving)
        result_worsening = score_analyst_consensus(worsening)

        assert result_improving["rating_score"] == pytest.approx(result_worsening["rating_score"])
        assert result_improving["rating_score_trend"] > 0
        assert result_worsening["rating_score_trend"] < 0
        assert result_improving["analyst_score"] > result_worsening["analyst_score"]

    def test_price_target_trend_direction(self) -> None:
        dates = pd.date_range("2026-01-01", periods=3, freq="30D")
        rising = _history(
            [
                {
                    "as_of_date": d,
                    "strong_buy": 5,
                    "buy": 5,
                    "hold": 5,
                    "sell": 0,
                    "strong_sell": 0,
                    "mean_price_target": p,
                }
                for d, p in zip(dates, [100, 110, 120], strict=True)
            ]
        )
        result = score_analyst_consensus(rising)
        assert result["price_target_trend_pct"] > 0

    def test_includes_upside_when_current_price_given(self) -> None:
        history = _history(
            [
                {
                    "as_of_date": pd.Timestamp("2026-01-01"),
                    "strong_buy": 5,
                    "buy": 5,
                    "hold": 0,
                    "sell": 0,
                    "strong_sell": 0,
                    "mean_price_target": 120,
                }
            ]
        )
        result = score_analyst_consensus(history, current_price=100)
        assert result["price_target_upside_pct"] == pytest.approx(20.0)

    def test_single_snapshot_has_no_trend_but_still_scores(self) -> None:
        history = _history(
            [
                {
                    "as_of_date": pd.Timestamp("2026-01-01"),
                    "strong_buy": 3,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strong_sell": 0,
                    "mean_price_target": 100,
                }
            ]
        )
        result = score_analyst_consensus(history)
        assert result["rating_score"] == pytest.approx(100.0)
        assert result["rating_score_trend"] is None
        assert result["price_target_trend_pct"] is None
        assert result["analyst_score"] == pytest.approx(100.0)

    def test_empty_history_returns_all_none(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "as_of_date",
                "strong_buy",
                "buy",
                "hold",
                "sell",
                "strong_sell",
                "mean_price_target",
            ]
        )
        result = score_analyst_consensus(empty)
        assert all(v is None for v in result.values())

    def test_no_coverage_snapshot_yields_none_score(self) -> None:
        history = _history(
            [
                {
                    "as_of_date": pd.Timestamp("2026-01-01"),
                    "strong_buy": 0,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strong_sell": 0,
                    "mean_price_target": None,
                }
            ]
        )
        result = score_analyst_consensus(history)
        assert result["rating_score"] is None
        assert result["analyst_score"] is None

    def test_trend_ignores_snapshots_outside_the_lookback_window(self) -> None:
        # A stale, very-old improving snapshot outside the window must not
        # leak into a trend computed only over the recent (flat) history.
        old = pd.Timestamp("2020-01-01")
        recent = pd.date_range("2026-05-01", periods=3, freq="15D")
        history = _history(
            [
                {
                    "as_of_date": old,
                    "strong_buy": 0,
                    "buy": 0,
                    "hold": 0,
                    "sell": 10,
                    "strong_sell": 0,
                    "mean_price_target": 50,
                },
                *[
                    {
                        "as_of_date": d,
                        "strong_buy": 5,
                        "buy": 5,
                        "hold": 0,
                        "sell": 0,
                        "strong_sell": 0,
                        "mean_price_target": 100,
                    }
                    for d in recent
                ],
            ]
        )
        result = score_analyst_consensus(history, lookback_days=91)
        assert result["rating_score_trend"] == pytest.approx(0.0, abs=1e-9)

    def test_raises_on_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            score_analyst_consensus(pd.DataFrame({"as_of_date": [pd.Timestamp("2026-01-01")]}))

    def test_analyst_score_bounded_0_to_100_even_with_extreme_trend(self) -> None:
        dates = pd.date_range("2026-01-01", periods=2, freq="1D")
        history = _history(
            [
                {
                    "as_of_date": dates[0],
                    "strong_buy": 0,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strong_sell": 10,
                    "mean_price_target": 10,
                },
                {
                    "as_of_date": dates[1],
                    "strong_buy": 10,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strong_sell": 0,
                    "mean_price_target": 200,
                },
            ]
        )
        result = score_analyst_consensus(history)
        assert 0.0 <= result["analyst_score"] <= 100.0
