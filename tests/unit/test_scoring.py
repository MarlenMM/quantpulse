import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import scoring


def _ohlcv(closes: list[float], start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="D")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1_000_000.0},
        index=idx,
    )


class TestScoreTechnical:
    def test_uptrend_scores_bullish(self) -> None:
        prices = _ohlcv(list(np.linspace(100, 200, 260)))
        score = scoring.score_technical(prices)
        assert score is not None and score > 60  # above MAs, positive MACD, RSI>50

    def test_downtrend_scores_bearish(self) -> None:
        prices = _ohlcv(list(np.linspace(200, 100, 260)))
        score = scoring.score_technical(prices)
        assert score is not None and score < 40

    def test_short_history_scores_on_available_signals(self) -> None:
        # ~30 bars: no SMA200, but RSI/MACD/short-MA signals still resolve.
        score = scoring.score_technical(_ohlcv(list(np.linspace(100, 130, 30))))
        assert score is not None

    def test_empty_is_none(self) -> None:
        assert scoring.score_technical(_ohlcv([])) is None


class TestScoreMomentum:
    def test_positive_trend_is_positive(self) -> None:
        prices = _ohlcv(list(np.linspace(100, 150, 200)))
        assert (scoring.score_momentum(prices) or 0) > 0

    def test_negative_trend_is_negative(self) -> None:
        prices = _ohlcv(list(np.linspace(150, 100, 200)))
        assert (scoring.score_momentum(prices) or 0) < 0

    def test_too_short_is_none(self) -> None:
        assert scoring.score_momentum(_ohlcv(list(np.linspace(100, 110, 10)))) is None

    def test_abstains_below_a_quarter_of_history(self) -> None:
        """A Sharpe-like ratio from ~1 month of bars is noise wearing a number.

        The floor used to be 21 bars, and the deployed demo cleared it with 23:
        every one of 503 names published a "6-month risk-adjusted momentum"
        computed from 22 daily returns, which correlated 0.96 with the technical
        score -- one short-term price signal counted twice, at full weight
        toward `data_confidence`.
        """
        one_month = _ohlcv(list(np.linspace(100, 110, 25)))
        assert scoring.score_momentum(one_month) is None
        # Just under a quarter still abstains; comfortably over it reports.
        assert scoring.score_momentum(_ohlcv(list(np.linspace(100, 110, 60)))) is None
        assert scoring.score_momentum(_ohlcv(list(np.linspace(100, 130, 90)))) is not None

    def test_abstention_drops_the_category_rather_than_scoring_zero(self) -> None:
        # The honest consequence: a thin name loses momentum's weight from its
        # data_confidence instead of being ranked on a number nobody can trust.
        raw = pd.DataFrame({"technical": [60.0, 40.0], "momentum": [None, None]}, index=["A", "B"])
        scores = scoring.build_composite(raw, profile="balanced").scores.set_index("symbol")
        assert pd.isna(scores.loc["A", "momentum_score"])
        assert scores.loc["A", "data_confidence"] == pytest.approx(20.0)  # technical only

    def test_flat_series_is_none(self) -> None:
        assert scoring.score_momentum(_ohlcv([100.0] * 200)) is None

    def test_low_volatility_preference_favors_calmer_names(self) -> None:
        rng = np.random.default_rng(0)
        base = np.linspace(100, 120, 200)
        calm = _ohlcv(list(base + rng.normal(0, 0.2, 200)))
        wild = _ohlcv(list(base + rng.normal(0, 5.0, 200)))
        calm_score = scoring.score_momentum(calm, prefer_low_volatility=True)
        wild_score = scoring.score_momentum(wild, prefer_low_volatility=True)
        assert calm_score is not None and wild_score is not None
        assert calm_score > wild_score  # calmer name ranks higher under the low-vol tilt

    def test_missing_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close"):
            scoring.score_momentum(pd.DataFrame({"open": [1.0, 2.0]}))


class TestSentimentAndTier2:
    def test_sentiment_passthrough(self) -> None:
        assert scoring.sentiment_to_raw(0.4) == 0.4
        assert scoring.sentiment_to_raw(None) is None

    def test_tier2_tilt_averages_events_for_the_symbols_baskets(self) -> None:
        events = pd.DataFrame(
            [
                {"matched_theme": "ai_theme", "sentiment_score": 0.6},
                {"matched_theme": "ai_theme", "sentiment_score": 0.2},
                {"matched_theme": "banks", "sentiment_score": -0.9},  # not NVDA's basket
            ]
        )
        members = {"ai_theme": {"NVDA", "AMD"}, "banks": {"JPM"}}
        assert scoring.tier2_thematic_tilt("NVDA", events, members) == pytest.approx(0.4)

    def test_tier2_tilt_none_when_symbol_in_no_basket(self) -> None:
        events = pd.DataFrame([{"matched_theme": "ai_theme", "sentiment_score": 0.6}])
        assert scoring.tier2_thematic_tilt("XYZ", events, {"ai_theme": {"NVDA"}}) is None

    def test_tier2_tilt_none_when_no_events(self) -> None:
        empty = pd.DataFrame(columns=["matched_theme", "sentiment_score"])
        assert scoring.tier2_thematic_tilt("NVDA", empty, {"ai_theme": {"NVDA"}}) is None


class TestPercentileNormalize:
    def test_ranks_to_0_100_higher_is_higher(self) -> None:
        out = scoring.percentile_normalize(pd.Series([10.0, 20.0, 30.0, 40.0]))
        assert list(out) == [25.0, 50.0, 75.0, 100.0]

    def test_missing_values_stay_missing(self) -> None:
        out = scoring.percentile_normalize(pd.Series([10.0, np.nan, 30.0]))
        assert pd.isna(out.iloc[1])
        assert out.iloc[0] == pytest.approx(50.0) and out.iloc[2] == pytest.approx(100.0)


class TestRatingHelpers:
    @pytest.mark.parametrize(
        "pr,expected",
        [
            (100.0, "strong_buy"),
            (90.0, "strong_buy"),
            (89.9, "buy"),
            (70.0, "buy"),
            (69.9, "hold"),
            (30.0, "hold"),
            (29.9, "sell"),
            (10.0, "sell"),
            (9.9, "strong_sell"),
            (0.0, "strong_sell"),
        ],
    )
    def test_relative_rating_cutoffs(self, pr: float, expected: str) -> None:
        assert scoring._rating_from_percentile(pr, strong_buy_cutoff=90.0) == expected

    def test_regime_lifts_strong_buy_cutoff_only_when_risk_off(self) -> None:
        assert scoring._strong_buy_cutoff(None) == 90.0
        assert scoring._strong_buy_cutoff(50.0) == 90.0  # neutral: unchanged
        assert scoring._strong_buy_cutoff(0.0) == 95.0  # fully risk-off: +5
        assert scoring._strong_buy_cutoff(25.0) == pytest.approx(92.5)


class TestBuildComposite:
    def test_weighted_composite_and_confidence_hand_check(self) -> None:
        # Every category is percentiled, fundamental included: [100,0] -> [100,50]
        # and [10,20] -> [50,100]. Only two of seven categories present.
        raw = pd.DataFrame(
            {"fundamental": [100.0, 0.0], "technical": [10.0, 20.0]}, index=["A", "B"]
        )
        result = scoring.build_composite(raw, profile="balanced")
        by_symbol = result.scores.set_index("symbol")
        # A: (0.25*100 + 0.20*50) / 0.45 ; B: (0.25*50 + 0.20*100)/0.45
        assert by_symbol.loc["A", "composite_score"] == pytest.approx(35 / 0.45)
        assert by_symbol.loc["B", "composite_score"] == pytest.approx(32.5 / 0.45)
        # Only fundamental (0.25) + technical (0.20) had data -> 45% confidence.
        assert by_symbol.loc["A", "data_confidence"] == pytest.approx(45.0)

    def test_fundamental_is_percentiled_like_every_other_category(self) -> None:
        """It is NOT already a percentile, despite arriving on a 0-100 scale.

        `fundamental.score_fundamentals` averages several within-sector metric
        percentiles, and an average of percentiles concentrates toward the
        middle rather than staying uniform. Measured on 500 synthetic names it
        had std 11.4 against 28.9 for a genuinely percentiled category -- so
        passing it through untouched made the balanced profile's nominal 25%
        fundamental weight drive only ~12% of the composite's variation.
        """
        raw = pd.DataFrame({"fundamental": [30.0, 60.0, 90.0]}, index=["A", "B", "C"])
        scores = scoring.build_composite(raw).scores.set_index("symbol")
        # [30,60,90] -> [33.3, 66.7, 100], not left at 30/60/90.
        assert scores.loc["C", "fundamental_score"] == pytest.approx(100.0)
        assert scores.loc["B", "fundamental_score"] == pytest.approx(200 / 3)
        assert scores.loc["A", "fundamental_score"] == pytest.approx(100 / 3)

    def test_percentiling_fundamental_preserves_its_sector_relative_ordering(self) -> None:
        # The transform is monotonic, which is the whole reason it is safe:
        # `score_fundamentals` decides the peer group, and this cannot disturb
        # the ordering it produced. Equal raw scores stay tied.
        raw = pd.DataFrame(
            {"fundamental": [80.0, 40.0, 80.0, 55.0]}, index=["BANK", "SOFT", "PHARMA", "UTIL"]
        )
        scores = scoring.build_composite(raw).scores.set_index("symbol")
        assert scores.loc["BANK", "fundamental_score"] == scores.loc["PHARMA", "fundamental_score"]
        assert (
            scores.loc["BANK", "fundamental_score"]
            > scores.loc["UTIL", "fundamental_score"]
            > scores.loc["SOFT", "fundamental_score"]
        )

    def test_missing_category_does_not_penalize_with_a_zero(self) -> None:
        # B is missing technical; its composite is renormalized over fundamental
        # alone (== its fundamental sub-score), not dragged down by a phantom 0.
        raw = pd.DataFrame(
            {"fundamental": [50.0, 50.0], "technical": [10.0, np.nan]}, index=["A", "B"]
        )
        scores = scoring.build_composite(raw).scores.set_index("symbol")
        # Tied raw fundamentals -> both take the tie's average rank (1.5/2 = 75).
        assert scores.loc["B", "composite_score"] == pytest.approx(75.0)
        assert scores.loc["B", "data_confidence"] == pytest.approx(25.0)  # fundamental weight only

    def test_symbol_with_no_data_is_dropped(self) -> None:
        raw = pd.DataFrame(
            {"fundamental": [50.0, np.nan], "technical": [10.0, np.nan]}, index=["A", "GHOST"]
        )
        symbols = set(scoring.build_composite(raw).scores["symbol"])
        assert symbols == {"A"}  # GHOST had no usable category

    def test_ratings_are_monotonic_with_composite(self) -> None:
        raw = pd.DataFrame(
            {"technical": [float(i) for i in range(30)]}, index=[f"S{i}" for i in range(30)]
        )
        scores = scoring.build_composite(raw).scores
        order = {r: i for i, r in enumerate(scoring.RATINGS)}  # strong_buy=0 ... strong_sell=4
        rank_of_rating = [order[r] for r in scores["rating"]]  # scores sorted desc by composite
        assert rank_of_rating == sorted(rank_of_rating)  # never improves as composite falls
        assert scores.iloc[0]["rating"] == "strong_buy"
        assert scores.iloc[-1]["rating"] == "strong_sell"

    def test_absolute_mode_uses_fixed_thresholds(self) -> None:
        # Absolute mode uses the raw fixed-bar readings, NOT peer percentiles,
        # so a 90 is a 90 whoever else is in the universe.
        raw = pd.DataFrame({"fundamental": [90.0, 50.0, 10.0]}, index=["A", "B", "C"])
        scores = scoring.build_composite(raw, rating_mode="absolute").scores.set_index("symbol")
        assert scores.loc["A", "rating"] == "strong_buy"  # >=75
        assert scores.loc["B", "rating"] == "hold"  # 40..60
        assert scores.loc["C", "rating"] == "strong_sell"  # <25
        # And the sub-score itself is the raw reading, not a rank.
        assert scores.loc["A", "fundamental_score"] == pytest.approx(90.0)

    def test_absolute_mode_judges_against_a_fixed_bar_not_the_peer_group(self) -> None:
        """The whole point of the mode, and what it previously failed to do.

        Two universes with IDENTICAL internal ranking but very different
        underlying quality must not receive the same absolute ratings. While
        absolute mode ran on percentile-normalized inputs it did exactly that --
        the composite was near-uniform on 0-100 by construction, so fixed
        cutoffs handed out the same count of Strong Buys in a boom and a crash
        (Section 22's "treating a relative ranking as an absolute judgment").
        """
        index = [f"S{i}" for i in range(40)]
        # `technical` is a fixed 0-100 reading: 50 is neutral, 90 is strong.
        healthy = pd.DataFrame({"technical": [60.0 + i * 0.5 for i in range(40)]}, index=index)
        crashing = pd.DataFrame({"technical": [5.0 + i * 0.5 for i in range(40)]}, index=index)

        healthy_ratings = list(
            scoring.build_composite(healthy, rating_mode="absolute").scores["rating"]
        )
        crashing_ratings = list(
            scoring.build_composite(crashing, rating_mode="absolute").scores["rating"]
        )

        assert healthy_ratings != crashing_ratings
        # A crashing market earns no Strong Buys at all, however its names rank
        # against each other.
        assert "strong_buy" in healthy_ratings
        assert "strong_buy" not in crashing_ratings

    def test_relative_mode_still_ranks_against_peers_in_any_market(self) -> None:
        # The counterpart, and why both modes exist: relative mode is SUPPOSED
        # to produce the same distribution regardless of market level.
        index = [f"S{i}" for i in range(40)]
        healthy = pd.DataFrame({"technical": [60.0 + i * 0.5 for i in range(40)]}, index=index)
        crashing = pd.DataFrame({"technical": [5.0 + i * 0.5 for i in range(40)]}, index=index)
        assert list(scoring.build_composite(healthy).scores["rating"]) == list(
            scoring.build_composite(crashing).scores["rating"]
        )

    def test_absolute_mode_maps_polarity_and_momentum_onto_the_same_bar(self) -> None:
        # Categories whose raw form is not 0-100 still have to land on the same
        # fixed scale, or the weighted average mixes incompatible units.
        raw = pd.DataFrame(
            {
                "sentiment": [1.0, 0.0, -1.0],
                "industry_macro": [1.0, 0.0, -1.0],
                "momentum": [10.0, 0.0, -10.0],  # far beyond the band
            },
            index=["A", "B", "C"],
        )
        absolute = scoring._absolute_subscores(raw)
        assert absolute.loc["B", "sentiment"] == pytest.approx(50.0)
        assert absolute.loc["A", "sentiment"] == pytest.approx(100.0)
        assert absolute.loc["C", "industry_macro"] == pytest.approx(0.0)
        assert absolute.loc["B", "momentum"] == pytest.approx(50.0)
        assert absolute.loc["A", "momentum"] > 99.0
        assert absolute.loc["C", "momentum"] < 1.0

    def test_risk_off_regime_hands_out_fewer_strong_buys(self) -> None:
        raw = pd.DataFrame(
            {"technical": [float(i) for i in range(40)]}, index=[f"S{i}" for i in range(40)]
        )
        neutral = scoring.build_composite(raw, regime_score=50.0).scores
        risk_off = scoring.build_composite(raw, regime_score=0.0).scores
        n_neutral = (neutral["rating"] == "strong_buy").sum()
        n_risk_off = (risk_off["rating"] == "strong_buy").sum()
        assert n_risk_off < n_neutral  # the market-wide dampening filter (Section 7.3)

    def test_profile_changes_the_ranking(self) -> None:
        # A is fundamentally strong but weak momentum; B the reverse. Value vs
        # growth should rank them oppositely.
        raw = pd.DataFrame({"fundamental": [90.0, 10.0], "momentum": [0.0, 1.0]}, index=["A", "B"])
        value_top = scoring.build_composite(raw, profile="value").scores.iloc[0]["symbol"]
        growth_top = scoring.build_composite(raw, profile="growth").scores.iloc[0]["symbol"]
        assert value_top == "A" and growth_top == "B"

    def test_output_has_composite_scores_shape(self) -> None:
        raw = pd.DataFrame({"fundamental": [50.0, 60.0]}, index=["A", "B"])
        cols = set(scoring.build_composite(raw).scores.columns)
        expected = (
            {
                "symbol",
                "composite_score",
                "percentile_rank",
                "rating",
                "data_confidence",
            }
            | set(scoring.CATEGORY_SCORE_COLUMNS.values())
            # The pre-normalization inputs travel with the row so a stored
            # score can be re-rated in absolute mode later; a percentile alone
            # cannot be un-ranked back into an absolute reading.
            | set(scoring.CATEGORY_RAW_COLUMNS.values())
        )
        assert cols == expected

    def test_invalid_rating_mode_raises(self) -> None:
        raw = pd.DataFrame({"fundamental": [50.0]}, index=["A"])
        with pytest.raises(ValueError, match="rating_mode"):
            scoring.build_composite(raw, rating_mode="nonsense")
