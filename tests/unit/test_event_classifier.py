"""Tests for event-type classification.

The real `facebook/bart-large-mnli` model is never loaded here (it's ~1.6GB
and network-bound) -- `_load_classifier` is patched with a deterministic fake
pipeline, exactly as the ingestion clients mock their network layer. What's
tested is *our* logic on top of the model: label->type mapping, the
confidence-threshold OTHER derivation, half-life assignment, empty-text
short-circuiting, and batch alignment.
"""

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantpulse.news_intelligence import event_classifier as ec
from quantpulse.news_intelligence.event_classifier import EventType


def _raw(scores_by_type: dict[EventType, float]) -> dict[str, Any]:
    """A transformers zero-shot result dict: labels sorted by descending score."""
    ordered = sorted(scores_by_type.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "sequence": "some text",
        "labels": [ec._CANDIDATE_PHRASES[t] for t, _ in ordered],
        "scores": [s for _, s in ordered],
    }


class _FakePipeline:
    """Stands in for a loaded zero-shot pipeline; returns a preset result per text."""

    def __init__(
        self, by_keyword: dict[str, dict[EventType, float]], default: dict[EventType, float]
    ):
        self._by_keyword = by_keyword
        self._default = default
        self.calls: list[Any] = []

    def _result_for(self, text: str) -> dict[str, Any]:
        for keyword, scores in self._by_keyword.items():
            if keyword.lower() in text.lower():
                return _raw(scores)
        return _raw(self._default)

    def __call__(self, text: Any, **kwargs: Any) -> Any:
        self.calls.append((text, kwargs))
        if isinstance(text, list):
            return [self._result_for(t) for t in text]
        return self._result_for(text)


# --- config integrity --------------------------------------------------------


def test_every_event_type_has_a_half_life() -> None:
    assert set(ec.EVENT_HALF_LIFE_DAYS) == set(EventType)


def test_other_is_never_a_candidate_label() -> None:
    assert EventType.OTHER not in ec._CANDIDATE_PHRASES
    assert len(ec._CANDIDATE_LABELS) == len(EventType) - 1


def test_phrase_and_type_maps_are_a_bijection() -> None:
    assert len(ec._PHRASE_TO_TYPE) == len(ec._CANDIDATE_PHRASES)
    for event_type, phrase in ec._CANDIDATE_PHRASES.items():
        assert ec._PHRASE_TO_TYPE[phrase] is event_type


def test_half_life_days_accessor() -> None:
    assert ec.half_life_days(EventType.MACRO_MONETARY) == 21.0
    assert ec.half_life_days(EventType.MERGERS_ACQUISITIONS) == 3.0


# --- classify() single --------------------------------------------------------


def test_empty_text_short_circuits_without_loading_model() -> None:
    with patch.object(ec, "_load_classifier") as mock_load:
        result = ec.classify("   ")
    assert result.event_type is EventType.OTHER
    assert result.confidence == 0.0
    mock_load.assert_not_called()


def test_high_confidence_maps_top_phrase_to_type() -> None:
    fake = _FakePipeline(
        by_keyword={"earnings": {EventType.EARNINGS: 0.9, EventType.LABOR: 0.1}},
        default={EventType.OTHER: 1.0},  # unused
    )
    with patch.object(ec, "_load_classifier", return_value=fake):
        result = ec.classify("Apple beats on earnings")
    assert result.event_type is EventType.EARNINGS
    assert result.confidence == pytest.approx(0.9)
    assert result.half_life_days == 5.0
    assert result.scores[EventType.EARNINGS] == pytest.approx(0.9)


def test_low_confidence_top_label_is_derived_as_other() -> None:
    # Top score below CONFIDENCE_THRESHOLD -> OTHER, but the real distribution
    # and the raw top score are still retained for audit.
    fake = _FakePipeline(
        by_keyword={},
        default={EventType.GEOPOLITICAL: 0.30, EventType.MACRO_MONETARY: 0.25},
    )
    with patch.object(ec, "_load_classifier", return_value=fake):
        result = ec.classify("An ambiguous headline")
    assert result.event_type is EventType.OTHER
    assert result.confidence == pytest.approx(0.30)  # raw top, not zeroed
    assert result.half_life_days == 3.0
    assert result.scores[EventType.GEOPOLITICAL] == pytest.approx(0.30)


def test_classify_passes_expected_pipeline_arguments() -> None:
    fake = _FakePipeline(by_keyword={}, default={EventType.EARNINGS: 0.8})
    with patch.object(ec, "_load_classifier", return_value=fake):
        ec.classify("something")
    _, kwargs = fake.calls[0]
    assert kwargs["candidate_labels"] == ec._CANDIDATE_LABELS
    assert kwargs["hypothesis_template"] == ec._HYPOTHESIS_TEMPLATE
    assert kwargs["multi_label"] is False


def test_full_label_distribution_is_retained_for_audit() -> None:
    # scores must carry every returned label, not just the winner -- it's the
    # audit trail behind the chosen event_type (Section 7.3's auditability aim).
    full = {t: 1.0 / len(ec._CANDIDATE_PHRASES) for t in ec._CANDIDATE_PHRASES}
    full[EventType.EARNINGS] = 0.6
    fake = _FakePipeline(by_keyword={}, default=full)
    with patch.object(ec, "_load_classifier", return_value=fake):
        result = ec.classify("Apple reports results")
    assert result.event_type is EventType.EARNINGS
    assert set(result.scores) == set(ec._CANDIDATE_PHRASES)
    assert EventType.OTHER not in result.scores  # OTHER is derived, never scored


def test_score_just_above_threshold_stays_classified() -> None:
    # A top score exactly at the threshold is classified (>=), not downgraded.
    fake = _FakePipeline(
        by_keyword={},
        default={EventType.GEOPOLITICAL: ec.CONFIDENCE_THRESHOLD, EventType.LABOR: 0.1},
    )
    with patch.object(ec, "_load_classifier", return_value=fake):
        result = ec.classify("A borderline headline")
    assert result.event_type is EventType.GEOPOLITICAL


# --- classify_articles() batch -----------------------------------------------


def test_classify_articles_batches_nonempty_and_defaults_empty_rows() -> None:
    fake = _FakePipeline(
        by_keyword={
            "earnings": {EventType.EARNINGS: 0.9, EventType.LABOR: 0.1},
            "merger": {EventType.MERGERS_ACQUISITIONS: 0.8, EventType.LABOR: 0.2},
        },
        # Default: a below-threshold top score, so an unmatched text derives OTHER.
        default={EventType.GEOPOLITICAL: 0.2, EventType.MACRO_MONETARY: 0.18},
    )
    df = pd.DataFrame(
        {
            "title": ["Apple earnings beat", "", "Big merger announced"],
            "summary": ["strong quarter", None, "two firms combine"],
        }
    )
    with patch.object(ec, "_load_classifier", return_value=fake):
        results = ec.classify_articles(df)

    types = [r.event_type for r in results]
    assert types == [EventType.EARNINGS, EventType.OTHER, EventType.MERGERS_ACQUISITIONS]
    assert list(results.index) == list(df.index)
    # Only the two non-empty rows were sent to the model, in one batched call.
    assert len(fake.calls) == 1
    batched_text = fake.calls[0][0]
    assert isinstance(batched_text, list)
    assert len(batched_text) == 2


def test_classify_articles_handles_single_nonempty_row_returned_as_dict() -> None:
    # Defensive coverage, not a scenario the real model produces (verified
    # live: a one-item list input always comes back as a list of dicts) --
    # this exercises the fallback branch guarding a hypothetical future shape.
    single_raw = _raw({EventType.LABOR: 0.7, EventType.EARNINGS: 0.3})
    fake = Mock(return_value=single_raw)
    df = pd.DataFrame({"title": ["Union announces strike"]})
    with patch.object(ec, "_load_classifier", return_value=fake):
        results = ec.classify_articles(df)
    assert results.iloc[0].event_type is EventType.LABOR


def test_classify_articles_handles_missing_text_columns_without_model() -> None:
    df = pd.DataFrame({"domain": ["example.com"], "language": ["English"]})
    with patch.object(ec, "_load_classifier") as mock_load:
        results = ec.classify_articles(df)
    assert results.iloc[0].event_type is EventType.OTHER
    mock_load.assert_not_called()


def test_classify_articles_empty_dataframe() -> None:
    df = pd.DataFrame({"title": [], "summary": []})
    with patch.object(ec, "_load_classifier") as mock_load:
        results = ec.classify_articles(df)
    assert results.tolist() == []
    mock_load.assert_not_called()
