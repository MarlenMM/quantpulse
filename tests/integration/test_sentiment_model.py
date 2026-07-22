"""Opt-in end-to-end validation of sentiment scoring against the REAL model.

Mirrors `test_event_classifier_model.py`'s approach exactly: the unit tests
(`tests/unit/test_sentiment.py`) mock the transformers pipeline so CI stays
offline and fast; this module exercises `ProsusAI/finbert` for real, to catch
what a mock structurally can't -- e.g. a pipeline-arg or output-shape change
that quietly breaks scoring (the single-string double-list-wrap this module's
own code had to be fixed for once, per its inline comment, was found exactly
this way).

Auto-skips unless the model is already in the local HuggingFace cache, so it
never triggers a download: it runs for a developer who has the model (and in
CI if the HF cache is warmed per Section 6.10), and silently skips elsewhere.
Assertions use only high-margin, unambiguous headlines, so this is
deterministic and not flaky.
"""

import pytest

from quantpulse.news_intelligence import sentiment as sm


def _model_is_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    cached = try_to_load_from_cache(sm._MODEL_NAME, "config.json")
    return isinstance(cached, str)


pytestmark = pytest.mark.skipif(
    not _model_is_cached(),
    reason=f"{sm._MODEL_NAME} not in local HF cache; skipping live-model test (no download)",
)


@pytest.mark.parametrize(
    ("headline", "expect_positive"),
    [
        ("Apple shares surge on record iPhone sales", True),
        ("Company files for bankruptcy amid mounting losses", False),
    ],
)
def test_real_model_scores_unambiguous_headlines(headline: str, expect_positive: bool) -> None:
    result = sm.score_sentiment(headline)
    assert (result.polarity > 0.3) is expect_positive
    assert (result.polarity < -0.3) is (not expect_positive)
    # The distribution is real: all three probabilities present and sum to ~1.
    total = result.positive + result.negative + result.neutral
    assert total == pytest.approx(1.0, abs=1e-3)


def test_real_model_score_articles_matches_single_score_sentiment() -> None:
    import pandas as pd

    headline = "Apple shares surge on record iPhone sales"
    single = sm.score_sentiment(headline)
    batch = sm.score_articles(pd.DataFrame({"title": [headline]})).iloc[0]
    assert single.polarity == pytest.approx(batch.polarity, abs=1e-6)


def test_real_model_empty_text_short_circuits() -> None:
    result = sm.score_sentiment("   ")
    assert result == sm._NEUTRAL_SCORE
