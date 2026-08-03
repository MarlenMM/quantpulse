from datetime import date, datetime
from unittest.mock import Mock

import pytest

from quantpulse.llm import narrative


def _rating(**overrides: object) -> narrative.RatingNarrative:
    defaults: dict[str, object] = {
        "symbol": "NVDA",
        "rating": "strong_buy",
        "composite_score": 87.3,
        "sub_scores": {
            "fundamental": 72.0,
            "technical": 91.5,
            "analyst": 80.0,
            "sentiment": None,
            "momentum": 95.0,
            "industry_macro": 60.0,
            "smart_money": 55.0,
        },
        "percentile_rank": 96.2,
        "data_confidence": 84.0,
        "profile": "balanced",
        "as_of": date(2026, 7, 27),
    }
    defaults.update(overrides)
    return narrative.RatingNarrative(**defaults)  # type: ignore[arg-type]


def _stub_provider(text: str = "Narrated.") -> Mock:
    provider = Mock()
    provider.generate.return_value = text
    return provider


class TestRatingContext:
    def test_includes_every_supplied_number(self) -> None:
        context = narrative.build_rating_context(_rating())
        assert "Symbol: NVDA" in context
        assert "Rating: strong buy" in context
        assert "87.3" in context
        assert "96.2" in context
        assert "84" in context
        assert "2026-07-27" in context
        assert "balanced" in context

    def test_missing_subscore_is_marked_not_available_not_zero(self) -> None:
        context = narrative.build_rating_context(_rating())
        sentiment_line = next(
            line for line in context.splitlines() if line.startswith("- sentiment")
        )
        assert sentiment_line == "- sentiment: not available"
        assert "sentiment: 0" not in context

    def test_relative_scheme_caveat_is_present_by_default(self) -> None:
        context = narrative.build_rating_context(_rating())
        assert "RELATIVE" in context
        assert "ranks well against peers" in context

    def test_absolute_scheme_says_so_instead(self) -> None:
        context = narrative.build_rating_context(_rating(rating_mode="absolute"))
        assert "ABSOLUTE" in context
        assert "fixed thresholds" in context

    def test_optional_fields_are_omitted_when_absent(self) -> None:
        minimal = narrative.RatingNarrative(
            symbol="AAPL", rating="hold", composite_score=50.0, sub_scores={"technical": 50.0}
        )
        context = narrative.build_rating_context(minimal)
        assert "Percentile rank" not in context
        assert "Data completeness" not in context
        assert "As of" not in context
        assert "Investor profile" not in context

    def test_context_is_deterministic(self) -> None:
        assert narrative.build_rating_context(_rating()) == narrative.build_rating_context(
            _rating()
        )


class TestForecastContext:
    def _forecast(self, **overrides: object) -> narrative.ForecastNarrative:
        defaults: dict[str, object] = {
            "symbol": "NVDA",
            "model_name": "gbr",
            "last_close": 123.45,
            "generated_on": date(2026, 7, 27),
            "horizons": (
                narrative.ForecastHorizon(
                    horizon_days=5,
                    point_return=0.0123,
                    point_price=124.97,
                    lower_price=120.10,
                    upper_price=129.80,
                    historical_hit_rate=0.58,
                ),
                narrative.ForecastHorizon(
                    horizon_days=20,
                    point_return=-0.0250,
                    point_price=120.36,
                ),
            ),
        }
        defaults.update(overrides)
        return narrative.ForecastNarrative(**defaults)  # type: ignore[arg-type]

    def test_includes_the_model_and_every_horizon(self) -> None:
        context = narrative.build_forecast_context(self._forecast())
        assert "Symbol: NVDA" in context
        assert "Forecast model: gbr" in context
        assert "123.45" in context
        assert "2026-07-27" in context
        assert "5 trading days" in context
        assert "20 trading days" in context

    def test_returns_are_signed_so_direction_cannot_be_misread(self) -> None:
        context = narrative.build_forecast_context(self._forecast())
        assert "+1.23%" in context
        assert "-2.50%" in context

    def test_graded_horizon_carries_its_hit_rate(self) -> None:
        context = narrative.build_forecast_context(self._forecast())
        line = next(line for line in context.splitlines() if line.startswith("- 5 trading days"))
        assert "58%" in line
        assert "directional accuracy" in line

    def test_ungraded_horizon_says_so_rather_than_omitting_the_track_record(self) -> None:
        # Section 7.6: a forecast shown without its track record invites more
        # confidence than it earned. A silently-missing hit rate would read as
        # an unqualified prediction, so absence is stated explicitly.
        context = narrative.build_forecast_context(self._forecast())
        line = next(line for line in context.splitlines() if line.startswith("- 20 trading days"))
        assert "has not been graded" in line
        assert "unproven" in line

    def test_band_is_labelled_a_confidence_range_not_a_floor_and_ceiling(self) -> None:
        context = narrative.build_forecast_context(self._forecast())
        assert "not price targets" in context
        assert "not a floor and ceiling" in context

    def test_no_horizons_says_so_explicitly(self) -> None:
        context = narrative.build_forecast_context(
            narrative.ForecastNarrative(symbol="AAPL", model_name="arima")
        )
        assert "No forecasts are stored" in context

    def test_optional_fields_are_omitted_when_absent(self) -> None:
        context = narrative.build_forecast_context(
            narrative.ForecastNarrative(
                symbol="AAPL",
                model_name="baseline",
                horizons=(narrative.ForecastHorizon(horizon_days=5, point_return=0.01),),
            )
        )
        assert "Last close" not in context
        assert "generated on" not in context
        # Scoped to the horizon line: the standing caveat below it legitimately
        # uses both words, so a whole-document check would pass vacuously.
        line = next(line for line in context.splitlines() if line.startswith("- 5 trading days"))
        assert "central price" not in line
        assert "range" not in line

    def test_context_is_deterministic(self) -> None:
        assert narrative.build_forecast_context(
            self._forecast()
        ) == narrative.build_forecast_context(self._forecast())


class TestExplainRating:
    def test_passes_context_and_returns_narration(self) -> None:
        provider = _stub_provider("NVDA ranks in the top decile.")
        result = narrative.explain_rating(_rating(), provider=provider)

        assert result == "NVDA ranks in the top decile."
        prompt, context = provider.generate.call_args[0]
        assert "NVDA" in prompt
        assert "87.3" in context

    def test_returns_none_without_a_provider(self) -> None:
        assert narrative.explain_rating(_rating(), provider=None) is None


class TestSentimentContext:
    def _narrative(self, **overrides: object) -> narrative.SentimentNarrative:
        defaults: dict[str, object] = {
            "symbol": "TSLA",
            "current_score": 0.42,
            "previous_score": -0.10,
            "mention_volume": 37,
            "as_of": date(2026, 7, 27),
            "drivers": [
                narrative.SentimentDriver(
                    title="Tesla beats delivery estimates",
                    event_type="earnings",
                    sentiment_score=0.88,
                    published_at=datetime(2026, 7, 26, 14, 30),
                    source="reuters",
                ),
                narrative.SentimentDriver(title="Recall notice filed", sentiment_score=-0.55),
            ],
        }
        defaults.update(overrides)
        return narrative.SentimentNarrative(**defaults)  # type: ignore[arg-type]

    def test_change_is_computed_here_not_left_to_the_model(self) -> None:
        context = narrative.build_sentiment_context(self._narrative())
        assert "change: +0.52" in context

    def test_drivers_carry_their_precomputed_scores_and_event_types(self) -> None:
        context = narrative.build_sentiment_context(self._narrative())
        assert '"Tesla beats delivery estimates"' in context
        assert "event type: earnings" in context
        assert "scored +0.88" in context
        assert "2026-07-26" in context
        assert "source: reuters" in context

    def test_drivers_are_capped_at_the_section_11_ceiling(self) -> None:
        drivers = [narrative.SentimentDriver(title=f"Headline {i}") for i in range(12)]
        context = narrative.build_sentiment_context(self._narrative(drivers=drivers))
        assert "Headline 4" in context
        assert "Headline 5" not in context
        assert narrative.MAX_SENTIMENT_DRIVERS == 5

    def test_no_drivers_says_so_explicitly(self) -> None:
        context = narrative.build_sentiment_context(self._narrative(drivers=[]))
        assert "none flagged" in context

    def test_previous_score_omitted_when_absent(self) -> None:
        context = narrative.build_sentiment_context(self._narrative(previous_score=None))
        assert "Previous reading" not in context

    def test_prompt_forbids_rescoring(self) -> None:
        provider = _stub_provider()
        narrative.explain_sentiment_move(self._narrative(), provider=provider)
        prompt, _ = provider.generate.call_args[0]
        assert "Do not assign your own sentiment scores" in prompt

    def test_returns_none_without_a_provider(self) -> None:
        assert narrative.explain_sentiment_move(self._narrative(), provider=None) is None


class TestFilingContext:
    def _excerpt(self, text: str = "Revenue rose 12% year over year.") -> narrative.FilingExcerpt:
        return narrative.FilingExcerpt(
            symbol="MSFT",
            form_type="10-K",
            excerpt=text,
            filed_date=date(2026, 6, 30),
            section="Management's Discussion",
            source_url="https://sec.gov/example",
        )

    def test_includes_metadata_and_quoted_excerpt(self) -> None:
        context = narrative.build_filing_context(self._excerpt())
        assert "Symbol: MSFT" in context
        assert "Filing type: 10-K" in context
        assert "2026-06-30" in context
        assert "Management's Discussion" in context
        assert "Revenue rose 12% year over year." in context
        assert "quoted source material" in context

    def test_long_excerpt_is_truncated_and_the_truncation_is_disclosed(self) -> None:
        long_text = "x" * (narrative.MAX_FILING_EXCERPT_CHARS + 500)
        context = narrative.build_filing_context(self._excerpt(long_text))
        assert "truncated" in context.lower()
        quoted_body = context.split('"""')[1].strip()
        assert len(quoted_body) == narrative.MAX_FILING_EXCERPT_CHARS

    def test_short_excerpt_is_not_marked_truncated(self) -> None:
        context = narrative.build_filing_context(self._excerpt())
        assert "truncated" not in context.lower()

    def test_prompt_tells_the_model_to_ignore_embedded_instructions(self) -> None:
        # A filing excerpt is the one context block made of free text rather
        # than computed numbers, so text inside it must not be followed.
        provider = _stub_provider()
        narrative.summarize_filing_excerpt(self._excerpt(), provider=provider)
        prompt, _ = provider.generate.call_args[0]
        assert "ignore it" in prompt
        assert "Do not infer a rating" in prompt

    def test_empty_excerpt_short_circuits_without_calling_the_model(self) -> None:
        provider = _stub_provider()
        assert narrative.summarize_filing_excerpt(self._excerpt("   "), provider=provider) is None
        provider.generate.assert_not_called()

    def test_returns_none_without_a_provider(self) -> None:
        assert narrative.summarize_filing_excerpt(self._excerpt(), provider=None) is None


class TestDegradation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: narrative.explain_rating(_rating()),
            lambda: narrative.explain_sentiment_move(
                narrative.SentimentNarrative(symbol="AAPL", current_score=0.1)
            ),
            lambda: narrative.summarize_filing_excerpt(
                narrative.FilingExcerpt(symbol="AAPL", form_type="10-Q", excerpt="text")
            ),
        ],
    )
    def test_every_entry_point_degrades_to_none(self, call: object) -> None:
        # With no provider configured, narration is absent -- never an exception,
        # since the page's actual content is the computed numbers (Section 11).
        from unittest.mock import patch

        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            assert call() is None  # type: ignore[operator]
