from datetime import date

import pandas as pd

from quantpulse.news_intelligence import market_regime


def _price_history(spec: dict[str, list[float]], *, end: str = "2026-07-22") -> pd.DataFrame:
    """Long-form (symbol, date, adj_close) frame; each symbol's closes end at `end`."""
    frames = []
    for symbol, closes in spec.items():
        dates = pd.date_range(end=end, periods=len(closes), freq="D")
        frames.append(pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes}))
    return pd.concat(frames, ignore_index=True)


class TestComputeBreadth:
    def test_percent_above_200dma(self) -> None:
        as_of = date(2026, 7, 22)
        # ABOVE: flat at 100 then jumps to 200 today -> last close well above its MA.
        # BELOW: flat at 100 then drops to 50 today -> last close below its MA.
        above = [100.0] * 200 + [200.0]
        below = [100.0] * 200 + [50.0]
        frame = _price_history({"ABOVE": above, "BELOW": below})
        assert market_regime.compute_breadth(frame, as_of, ma_window=200) == 50.0

    def test_symbols_without_enough_history_are_ignored(self) -> None:
        as_of = date(2026, 7, 22)
        frame = _price_history({"SHORT": [100.0, 101.0, 102.0]})
        assert market_regime.compute_breadth(frame, as_of, ma_window=200) is None

    def test_empty_frame_is_none(self) -> None:
        empty = pd.DataFrame(columns=["symbol", "date", "adj_close"])
        assert market_regime.compute_breadth(empty, date(2026, 7, 22)) is None

    def test_ignores_bars_after_as_of(self) -> None:
        # A future crash must not flip a point-in-time-above symbol below.
        as_of = date(2026, 7, 20)
        # 200 rising bars ending on the 20th (last=200 clearly above its ~100 MA),
        # then two future bars crashing to 1.0 dated the 21st/22nd.
        closes = [float(v) for v in range(1, 201)] + [1.0, 1.0]
        frame = _price_history({"X": closes}, end="2026-07-22")
        # If the future bars leaked in, the last close would read 1.0 (below) -> 0.0;
        # excluded correctly, the symbol is above its MA -> 100.0.
        assert market_regime.compute_breadth(frame, as_of, ma_window=200) == 100.0


class TestVixCalmScore:
    def test_percentile_used_with_enough_history(self) -> None:
        history = [float(v) for v in range(10, 50)]  # 40 points
        # A VIX at the very top of its range reads as near-zero calm.
        assert market_regime.vix_calm_score(49.0, history) == 0.0
        # At the very bottom, near-max calm.
        low = market_regime.vix_calm_score(10.0, history)
        assert low is not None and low >= 97.0

    def test_fallback_band_when_history_sparse(self) -> None:
        # 12 -> calm 100, 40 -> calm 0, 26 -> ~50.
        assert market_regime.vix_calm_score(12.0, []) == 100.0
        assert market_regime.vix_calm_score(40.0, []) == 0.0
        assert market_regime.vix_calm_score(26.0, []) == 50.0

    def test_extreme_levels_clip(self) -> None:
        assert market_regime.vix_calm_score(5.0, []) == 100.0
        assert market_regime.vix_calm_score(80.0, []) == 0.0

    def test_missing_vix_is_none(self) -> None:
        assert market_regime.vix_calm_score(None, [1.0, 2.0]) is None


class TestSubScores:
    def test_tone_score_maps_range(self) -> None:
        assert market_regime.tone_score(-10.0) == 0.0
        assert market_regime.tone_score(10.0) == 100.0
        assert market_regime.tone_score(0.0) == 50.0
        assert market_regime.tone_score(None) is None

    def test_yield_curve_score_inverted_is_risk_off(self) -> None:
        inverted = market_regime.yield_curve_score(-1.0)
        steep = market_regime.yield_curve_score(1.5)
        assert inverted == 0.0
        assert steep == 100.0
        assert market_regime.yield_curve_score(None) is None


class TestComputeMarketRegime:
    def test_blends_available_signals_and_labels(self) -> None:
        reading = market_regime.compute_market_regime(
            date(2026, 7, 22),
            vix_level=12.0,  # fallback calm = 100
            vix_history=[],
            breadth_pct=90.0,
            macro_tone=10.0,  # 100
            yield_curve_spread_value=1.5,  # 100
        )
        assert reading.regime_score is not None and reading.regime_score >= 95.0
        assert reading.regime_label == "risk_on"

    def test_renormalizes_over_available_signals(self) -> None:
        # Only breadth present -> score equals breadth, not diluted by absent signals.
        reading = market_regime.compute_market_regime(
            date(2026, 7, 22),
            vix_level=None,
            vix_history=[],
            breadth_pct=42.0,
            macro_tone=None,
            yield_curve_spread_value=None,
        )
        assert reading.regime_score == 42.0

    def test_all_missing_is_none_not_a_false_neutral(self) -> None:
        reading = market_regime.compute_market_regime(
            date(2026, 7, 22),
            vix_level=None,
            vix_history=[],
            breadth_pct=None,
            macro_tone=None,
            yield_curve_spread_value=None,
        )
        assert reading.regime_score is None
        assert reading.regime_label is None

    def test_risk_off_label(self) -> None:
        reading = market_regime.compute_market_regime(
            date(2026, 7, 22),
            vix_level=40.0,  # calm 0
            vix_history=[],
            breadth_pct=10.0,
            macro_tone=-8.0,
            yield_curve_spread_value=-0.8,
        )
        assert reading.regime_score is not None and reading.regime_score <= 35.0
        assert reading.regime_label == "risk_off"

    def test_inputs_are_echoed_onto_the_reading(self) -> None:
        reading = market_regime.compute_market_regime(
            date(2026, 7, 22),
            vix_level=18.0,
            vix_history=[],
            breadth_pct=55.0,
            macro_tone=1.0,
            yield_curve_spread_value=0.3,
        )
        record = market_regime.regime_to_record(reading)
        assert record["vix_level"] == 18.0
        assert record["breadth_pct_above_200dma"] == 55.0
        assert record["macro_news_tone"] == 1.0
        assert record["yield_curve_spread"] == 0.3
        assert record["date"] == date(2026, 7, 22)


class TestCoverageIsDescribedHonestly:
    """The index renormalizes over what it has, so the caption must say what that was.

    Both front ends called it "built from four inputs" while `yield_curve_spread`
    had been `None` since the project began -- with no `FRED_API_KEY` the 10Y-2Y
    series never arrives. The published score is a weighted average of three
    signals presented as an average of four, which misdescribes the number
    standing next to it rather than merely omitting a row.
    """

    @staticmethod
    def _row(**overrides: float | None) -> dict[str, float | None]:
        row: dict[str, float | None] = {
            "vix_level": 14.5,
            "breadth_pct_above_200dma": 66.4,
            "macro_news_tone": -0.5,
            "yield_curve_spread": 0.42,
        }
        row.update(overrides)
        return row

    def test_all_four_present_says_so_plainly(self) -> None:
        note = market_regime.describe_regime_coverage(self._row())
        assert note == "All four inputs are live."

    def test_a_missing_input_is_counted_and_named(self) -> None:
        note = market_regime.describe_regime_coverage(self._row(yield_curve_spread=None))
        assert note.startswith("Three of its four inputs are live")
        assert "the yield-curve spread is missing" in note
        # The renormalization is the reason this matters: a reader who thinks a
        # missing input scored 50 reads the headline number wrong.
        assert "renormalized" in note

    def test_two_missing_inputs_read_as_a_list(self) -> None:
        note = market_regime.describe_regime_coverage(
            self._row(yield_curve_spread=None, macro_news_tone=None)
        )
        assert note.startswith("Two of its four inputs are live")
        assert "macro news tone and the yield-curve spread are missing" in note

    def test_nan_counts_as_missing(self) -> None:
        """A stored NaN and a stored NULL are the same absence to a reader."""
        note = market_regime.describe_regime_coverage(self._row(yield_curve_spread=float("nan")))
        assert "the yield-curve spread is missing" in note

    def test_nothing_available_does_not_claim_a_score(self) -> None:
        note = market_regime.describe_regime_coverage(
            {key: None for key, _ in market_regime.REGIME_INPUT_LABELS}
        )
        assert "no score to read" in note

    def test_the_labels_cover_exactly_the_blended_signals(self) -> None:
        """A signal described here and blended there under another name would drift.

        The weights key off short names (`vix`, `breadth`, `yield_curve`,
        `tone`); this table keys off the stored columns. They have to stay the
        same set or the caption counts a different four than the score does.
        """
        assert len(market_regime.REGIME_INPUT_LABELS) == len(market_regime._REGIME_WEIGHTS)

    def test_it_matches_what_the_blend_actually_used(self) -> None:
        """The count in the sentence is the count the score was built from.

        Asserted against `compute_market_regime` rather than against the
        sentence's own arithmetic, so the two cannot disagree.
        """
        reading = market_regime.compute_market_regime(
            date(2026, 9, 6),
            vix_level=14.5,
            vix_history=[12.0, 15.0, 18.0, 20.0],
            breadth_pct=66.4,
            macro_tone=-0.5,
            yield_curve_spread_value=None,
        )
        row = market_regime.regime_to_record(reading)
        assert reading.regime_score is not None, "the fixture must still produce a score"
        assert market_regime.describe_regime_coverage(row).startswith("Three of its four")
