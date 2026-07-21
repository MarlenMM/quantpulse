"""Opt-in end-to-end validation of the event classifier against the REAL model.

The rest of the suite mocks the transformers pipeline (the unit tests in
tests/unit/test_event_classifier.py) so CI stays offline and fast. This module
instead exercises `facebook/bart-large-mnli` for real, to catch the class of
regression a mock can't -- a label-phrasing or threshold change that quietly
degrades actual classification quality.

It auto-skips unless the model is already in the local HuggingFace cache, so
it never triggers a ~1.6GB download: it runs for a developer who has the model
(and in CI if the HF cache is warmed per Section 6.10), and silently skips
everywhere else. Zero-shot classification with a fixed model is deterministic,
and these assertions use only high-margin, unambiguous headlines, so it is not
flaky.
"""

import pytest

from quantpulse.news_intelligence import event_classifier as ec
from quantpulse.news_intelligence.event_classifier import EventType


def _model_is_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    cached = try_to_load_from_cache(ec._MODEL_NAME, "config.json")
    return isinstance(cached, str)


pytestmark = pytest.mark.skipif(
    not _model_is_cached(),
    reason=f"{ec._MODEL_NAME} not in local HF cache; skipping live-model test (no 1.6GB download)",
)


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        ("Apple beats Q3 earnings estimates on strong iPhone sales", EventType.EARNINGS),
        (
            "Federal Reserve holds interest rates steady amid inflation worries",
            EventType.MACRO_MONETARY,
        ),
        ("Tesla unveils next-generation self-driving chip", EventType.PRODUCT_TECHNOLOGY),
        ("Autoworkers union launches strike at three major plants", EventType.LABOR),
    ],
)
def test_real_model_classifies_unambiguous_headlines(headline: str, expected: EventType) -> None:
    result = ec.classify(headline)
    assert result.event_type is expected
    assert result.confidence >= ec.CONFIDENCE_THRESHOLD
    # The full distribution is real and covers exactly the candidate labels.
    assert set(result.scores) == set(ec._CANDIDATE_PHRASES)


def test_real_model_empty_text_short_circuits() -> None:
    result = ec.classify("   ")
    assert result.event_type is EventType.OTHER
    assert result.confidence == 0.0
