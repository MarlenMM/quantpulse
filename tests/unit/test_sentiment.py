"""Tests for FinBERT sentiment scoring + recency decay.

The real `ProsusAI/finbert` model is never loaded here -- `_load_sentiment_model`
is patched with a deterministic fake pipeline, exactly as
`test_event_classifier.py` mocks BART. The fake's shapes (single-string call
wrapped one list deeper than a batch call's per-item shape) mirror what was
verified live against the real model, not guessed.
"""

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from quantpulse.news_intelligence import sentiment as sm
from quantpulse.news_intelligence.event_classifier import EventType


def _labels(positive: float, negative: float, neutral: float) -> list[dict[str, Any]]:
    return [
        {"label": "positive", "score": positive},
        {"label": "negative", "score": negative},
        {"label": "neutral", "score": neutral},
    ]


class _FakePipeline:
    """Stands in for a loaded FinBERT pipeline; returns a preset label distribution per text.

    Matches the verified real-model shape: a single (non-list) input returns
    `[[{label dicts}]]` (one extra wrapping list); a list input returns
    `[[{label dicts}], [{label dicts}], ...]` -- one inner list per item,
    including for a one-item list.
    """

    def __init__(self, by_keyword: dict[str, list[dict[str, Any]]], default: list[dict[str, Any]]):
        self._by_keyword = by_keyword
        self._default = default
        self.calls: list[Any] = []

    def _result_for(self, text: str) -> list[dict[str, Any]]:
        for keyword, labels in self._by_keyword.items():
            if keyword.lower() in text.lower():
                return labels
        return self._default

    def __call__(self, text: Any, **kwargs: Any) -> Any:
        self.calls.append((text, kwargs))
        if isinstance(text, list):
            return [self._result_for(t) for t in text]
        return [self._result_for(text)]


# --- SentimentScore / polarity math ------------------------------------------


def test_neutral_score_constant() -> None:
    assert sm._NEUTRAL_SCORE == sm.SentimentScore(0.0, 0.0, 0.0, 1.0)


def test_result_from_raw_computes_polarity_as_positive_minus_negative() -> None:
    result = sm._result_from_raw(_labels(positive=0.7, negative=0.1, neutral=0.2))
    assert result.polarity == pytest.approx(0.6)
    assert result.positive == pytest.approx(0.7)
    assert result.negative == pytest.approx(0.1)
    assert result.neutral == pytest.approx(0.2)


# --- score_sentiment (single) ------------------------------------------------


def test_score_sentiment_empty_text_short_circuits_without_loading_model() -> None:
    with patch.object(sm, "_load_sentiment_model") as mock_load:
        result = sm.score_sentiment("   ")
    assert result == sm._NEUTRAL_SCORE
    mock_load.assert_not_called()


def test_score_sentiment_unwraps_the_single_string_double_list() -> None:
    # Regression: a single (non-batched) string call returns [[{...}]], one
    # list deeper than a batch call's per-item shape -- this must be unwrapped,
    # not passed straight to _result_from_raw.
    fake = _FakePipeline(by_keyword={}, default=_labels(positive=0.8, negative=0.1, neutral=0.1))
    with patch.object(sm, "_load_sentiment_model", return_value=fake):
        result = sm.score_sentiment("Apple beats on earnings")
    assert result.polarity == pytest.approx(0.7)


def test_score_sentiment_passes_truncation() -> None:
    fake = _FakePipeline(by_keyword={}, default=_labels(0.5, 0.3, 0.2))
    with patch.object(sm, "_load_sentiment_model", return_value=fake):
        sm.score_sentiment("some text")
    _, kwargs = fake.calls[0]
    assert kwargs["truncation"] is True


# --- score_articles (batch) --------------------------------------------------


def test_score_articles_batches_nonempty_and_defaults_empty_rows() -> None:
    fake = _FakePipeline(
        by_keyword={
            "surge": _labels(positive=0.9, negative=0.05, neutral=0.05),
            "bankruptcy": _labels(positive=0.02, negative=0.93, neutral=0.05),
        },
        default=_labels(0.33, 0.33, 0.34),
    )
    df = pd.DataFrame(
        {
            "title": ["Shares surge on strong demand", "", "Files for bankruptcy"],
            "summary": ["Record quarter", None, "Investors alarmed"],
        }
    )
    with patch.object(sm, "_load_sentiment_model", return_value=fake):
        results = sm.score_articles(df)

    assert results.iloc[0].polarity > 0.5
    assert results.iloc[1] == sm._NEUTRAL_SCORE
    assert results.iloc[2].polarity < -0.5
    assert list(results.index) == list(df.index)
    # Only the two non-empty rows were sent to the model, in one batched call.
    assert len(fake.calls) == 1
    assert len(fake.calls[0][0]) == 2


def test_score_articles_handles_single_nonempty_row() -> None:
    fake = _FakePipeline(by_keyword={}, default=_labels(0.1, 0.8, 0.1))
    df = pd.DataFrame({"title": ["Company reports steep losses"]})
    with patch.object(sm, "_load_sentiment_model", return_value=fake):
        results = sm.score_articles(df)
    assert results.iloc[0].polarity < -0.5


def test_score_articles_handles_missing_text_columns_without_model() -> None:
    df = pd.DataFrame({"domain": ["example.com"]})
    with patch.object(sm, "_load_sentiment_model") as mock_load:
        results = sm.score_articles(df)
    assert results.iloc[0] == sm._NEUTRAL_SCORE
    mock_load.assert_not_called()


def test_score_articles_empty_dataframe() -> None:
    df = pd.DataFrame({"title": [], "summary": []})
    with patch.object(sm, "_load_sentiment_model") as mock_load:
        results = sm.score_articles(df)
    assert results.tolist() == []
    mock_load.assert_not_called()


# --- decay_weight -------------------------------------------------------------


def test_decay_weight_zero_age_is_full_weight() -> None:
    assert sm.decay_weight(0.0, 5.0) == 1.0


def test_decay_weight_one_half_life_is_half() -> None:
    assert sm.decay_weight(5.0, 5.0) == pytest.approx(0.5)


def test_decay_weight_two_half_lives_is_quarter() -> None:
    assert sm.decay_weight(10.0, 5.0) == pytest.approx(0.25)


def test_decay_weight_negative_age_clamps_rather_than_amplifies() -> None:
    assert sm.decay_weight(-10.0, 5.0) == 1.0


def test_decay_weight_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError):
        sm.decay_weight(1.0, 0.0)
    with pytest.raises(ValueError):
        sm.decay_weight(1.0, -3.0)


def test_decay_weight_for_event_wires_to_event_classifier_half_life() -> None:
    assert sm.decay_weight_for_event(5.0, EventType.EARNINGS) == pytest.approx(0.5)
    assert sm.decay_weight_for_event(21.0, EventType.MACRO_MONETARY) == pytest.approx(0.5)


# --- _age_in_days --------------------------------------------------------------


def test_age_in_days_none_and_missing_values_return_none() -> None:
    as_of = datetime(2026, 7, 22)
    assert sm._age_in_days(None, as_of) is None
    assert sm._age_in_days(pd.NaT, as_of) is None
    assert sm._age_in_days(float("nan"), as_of) is None


@pytest.mark.parametrize(
    "published_at",
    [
        date(2026, 7, 20),
        datetime(2026, 7, 20),
        pd.Timestamp("2026-07-20"),
    ],
)
def test_age_in_days_accepts_date_datetime_and_timestamp(published_at: Any) -> None:
    as_of = datetime(2026, 7, 22)
    assert sm._age_in_days(published_at, as_of) == pytest.approx(2.0)


# --- aggregate_decayed_sentiment ----------------------------------------------


def _articles_fixture(as_of: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "matched_symbols": [["AAPL"], ["AAPL", "MSFT"], ["MSFT"]],
            "sentiment": [
                sm.SentimentScore(0.9, 0.9, 0.0, 0.1),  # fresh earnings beat for AAPL
                sm.SentimentScore(-0.95, 0.0, 0.95, 0.05),  # 30d-stale Fed headline, AAPL+MSFT
                sm.SentimentScore(0.2, 0.4, 0.2, 0.4),  # very stale (200d) earnings for MSFT
            ],
            "event_type": [EventType.EARNINGS, EventType.MACRO_MONETARY, EventType.EARNINGS],
            "published_at": [
                as_of - timedelta(days=1),
                as_of - timedelta(days=30),
                as_of - timedelta(days=200),
            ],
        }
    )


def test_aggregate_fresh_earnings_beats_stale_more_emphatic_macro_headline() -> None:
    as_of = datetime(2026, 7, 22)
    articles = _articles_fixture(as_of)
    agg = sm.aggregate_decayed_sentiment("AAPL", articles, as_of)
    assert agg is not None
    assert agg.symbol == "AAPL"
    assert agg.mention_volume == 2
    # Hand-computed: w0=0.5**(1/5)=0.8706, w1=0.5**(30/21)=0.3711
    # score = (0.9*w0 + -0.95*w1) / (w0+w1)
    w0 = sm.decay_weight(1, 5)
    w1 = sm.decay_weight(30, 21)
    expected = (0.9 * w0 + -0.95 * w1) / (w0 + w1)
    assert agg.score == pytest.approx(expected)
    assert agg.total_weight == pytest.approx(w0 + w1)


def test_aggregate_no_match_returns_none() -> None:
    as_of = datetime(2026, 7, 22)
    articles = _articles_fixture(as_of)
    assert sm.aggregate_decayed_sentiment("GOOGL", articles, as_of) is None


def test_aggregate_excludes_undated_articles_rather_than_guessing() -> None:
    as_of = datetime(2026, 7, 22)
    articles = pd.DataFrame(
        {
            "matched_symbols": [["TSLA"], ["TSLA"]],
            "sentiment": [
                sm.SentimentScore(0.5, 0.5, 0.0, 0.5),
                sm.SentimentScore(-0.5, 0.0, 0.5, 0.5),
            ],
            "event_type": [EventType.PRODUCT_TECHNOLOGY, EventType.PRODUCT_TECHNOLOGY],
            "published_at": [None, as_of - timedelta(days=1)],
        }
    )
    agg = sm.aggregate_decayed_sentiment("TSLA", articles, as_of)
    assert agg is not None
    assert agg.mention_volume == 2  # both rows counted as "mentions"...
    assert agg.score == pytest.approx(-0.5)  # ...but only the dated one contributes to the score


def test_aggregate_all_undated_returns_none() -> None:
    as_of = datetime(2026, 7, 22)
    articles = pd.DataFrame(
        {
            "matched_symbols": [["TSLA"]],
            "sentiment": [sm.SentimentScore(0.5, 0.0, 0.0, 0.5)],
            "event_type": [EventType.PRODUCT_TECHNOLOGY],
            "published_at": [None],
        }
    )
    assert sm.aggregate_decayed_sentiment("TSLA", articles, as_of) is None


def test_aggregate_accepts_plain_float_sentiment_column() -> None:
    as_of = datetime(2026, 7, 22)
    articles = pd.DataFrame(
        {
            "matched_symbols": [["NVDA"]],
            "sentiment": [0.6],  # a plain float, not a SentimentScore
            "event_type": [EventType.EARNINGS],
            "published_at": [as_of],
        }
    )
    agg = sm.aggregate_decayed_sentiment("NVDA", articles, as_of)
    assert agg is not None
    assert agg.score == pytest.approx(0.6)


def test_aggregate_single_article_score_equals_raw_polarity_regardless_of_staleness() -> None:
    # The normalization-cancels-decay property documented on the function:
    # with exactly one contributing article, `score` is that article's raw
    # polarity no matter how old it is -- staleness shows up in total_weight.
    as_of = datetime(2026, 7, 22)
    for age_days in (0, 10, 100):
        articles = pd.DataFrame(
            {
                "matched_symbols": [["XOM"]],
                "sentiment": [sm.SentimentScore(0.42, 0.5, 0.08, 0.42)],
                "event_type": [EventType.GEOPOLITICAL],
                "published_at": [as_of - timedelta(days=age_days)],
            }
        )
        agg = sm.aggregate_decayed_sentiment("XOM", articles, as_of)
        assert agg is not None
        assert agg.score == pytest.approx(0.42)
    # But total_weight does shrink with age.
    fresh = sm.aggregate_decayed_sentiment(
        "XOM",
        pd.DataFrame(
            {
                "matched_symbols": [["XOM"]],
                "sentiment": [sm.SentimentScore(0.42, 0.5, 0.08, 0.42)],
                "event_type": [EventType.GEOPOLITICAL],
                "published_at": [as_of],
            }
        ),
        as_of,
    )
    stale = sm.aggregate_decayed_sentiment(
        "XOM",
        pd.DataFrame(
            {
                "matched_symbols": [["XOM"]],
                "sentiment": [sm.SentimentScore(0.42, 0.5, 0.08, 0.42)],
                "event_type": [EventType.GEOPOLITICAL],
                "published_at": [as_of - timedelta(days=100)],
            }
        ),
        as_of,
    )
    assert fresh is not None and stale is not None
    assert stale.total_weight < fresh.total_weight
