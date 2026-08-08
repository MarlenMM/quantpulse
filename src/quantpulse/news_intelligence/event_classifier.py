"""Event-type classification (Section 7.3 step 2).

Rather than only scoring polarity, classify *what kind* of event an article
represents, so the system can treat a short-lived "M&A rumor" differently
from a slow-decaying "Fed rate decision" (Section 7.3). Uses a free, local,
zero-shot classifier (`facebook/bart-large-mnli` via HuggingFace
`transformers`) -- no API key, no per-call cost, deterministic and auditable
(Section 7.3's closing note), same design stance as the spaCy NER in
`entity_extraction.py`.

Two judgment calls live here (the reason Section 21 rates this Opus/High):

1. **The taxonomy and how it's presented to the model.** The eight event
   types from Section 7.3 are given to the NLI model as natural-language
   phrases ("mergers and acquisitions"), not raw slugs -- zero-shot accuracy
   depends heavily on label phrasing. Crucially, `other` is *not* offered to
   the model as a candidate label: "other" is not a semantic category an NLI
   model can entail, so including it corrupts the whole softmax. Instead a
   low top-score (below `CONFIDENCE_THRESHOLD`) *derives* `other` -- an
   article the model can't confidently place is unclassified, not
   force-fit into the nearest label.

2. **The per-event-type half-life** (`EVENT_HALF_LIFE_DAYS`) -- how many days
   an event of each type should keep mattering. This is the input the
   recency-decay step (Section 7.3 step 4, the third Phase-4 part) applies its
   exponential decay against. The numbers are anchored to the plan's own
   explicit statements (earnings ~ days; a Fed decision ~ weeks; an M&A rumor
   decays fast) and kept as a clearly-labeled, tunable table rather than
   magic numbers buried in the decay code, because getting them wrong is
   exactly the "over- or under-propagating an event's impact" failure
   Section 21 warns about.
"""

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)

_MODEL_NAME = "facebook/bart-large-mnli"

# How many articles go through the model per call. Small enough that a deadline
# is checked often (a chunk is the granularity at which work can be abandoned),
# large enough that batching still pays. Inference dominates either way.
DEFAULT_CHUNK_SIZE = 32

# A finance-framed hypothesis template markedly outperforms the library
# default ("This example is {}.") on headline text -- the model is scoring
# entailment of this sentence, so naming the domain sharpens it.
_HYPOTHESIS_TEMPLATE = "This financial news article is about {}."

# Below this top-label softmax score, the article is treated as `OTHER`
# rather than force-fit into the best-but-weak label. 1/8 candidate labels =
# 0.125 is pure chance; this threshold demands a clear plurality above it.
# Empirically calibrated against real BART output on labeled headlines: clear,
# correct classifications land ~0.45-0.95 (the weakest legitimate ones --
# geopolitical/regulatory -- around 0.45-0.49), so 0.35 sits safely below the
# real-classification band while still filtering genuine ambiguity. See the
# opt-in integration test tests/integration/test_event_classifier_model.py.
CONFIDENCE_THRESHOLD = 0.35


class EventType(StrEnum):
    """The event taxonomy from Section 7.3. Values are the stable storage slugs."""

    EARNINGS = "earnings"
    MERGERS_ACQUISITIONS = "M&A"
    REGULATORY_LEGAL = "regulatory/legal"
    MACRO_MONETARY = "macro/monetary-policy"
    GEOPOLITICAL = "geopolitical"
    PRODUCT_TECHNOLOGY = "product/technology"
    MANAGEMENT_CHANGE = "management-change"
    LABOR = "labor"
    # Derived, never sent to the model as a candidate label -- see module docstring.
    OTHER = "other"


# The natural-language phrase each classifiable type is presented to the NLI
# model as. OTHER is deliberately absent: it's derived from low confidence.
_CANDIDATE_PHRASES: dict[EventType, str] = {
    EventType.EARNINGS: "corporate earnings or financial results",
    EventType.MERGERS_ACQUISITIONS: "a merger or acquisition",
    EventType.REGULATORY_LEGAL: "regulation, a lawsuit, or legal action",
    EventType.MACRO_MONETARY: "the economy, interest rates, or monetary policy",
    EventType.GEOPOLITICAL: "geopolitics, war, or international relations",
    EventType.PRODUCT_TECHNOLOGY: "a new product or technology",
    EventType.MANAGEMENT_CHANGE: "a change in company management or leadership",
    EventType.LABOR: "labor, workers, layoffs, or a strike",
}
_PHRASE_TO_TYPE: dict[str, EventType] = {v: k for k, v in _CANDIDATE_PHRASES.items()}
_CANDIDATE_LABELS: list[str] = list(_CANDIDATE_PHRASES.values())

# How long, in days, an event of each type typically keeps moving sentiment --
# the half-life the recency-decay step (Section 7.3 step 4) decays against.
# Anchored to the plan's explicit statements; see module docstring.
EVENT_HALF_LIFE_DAYS: dict[EventType, float] = {
    EventType.EARNINGS: 5.0,  # "an earnings surprise might matter for days"
    EventType.MERGERS_ACQUISITIONS: 3.0,  # "M&A rumor ... short-lived ... decay-fast"
    EventType.REGULATORY_LEGAL: 14.0,  # legal/regulatory processes unfold over weeks
    EventType.MACRO_MONETARY: 21.0,  # "a Fed decision might matter for weeks", slower-decaying
    EventType.GEOPOLITICAL: 10.0,  # persistent but news-cycle driven
    EventType.PRODUCT_TECHNOLOGY: 7.0,  # a launch's impact fades within ~a week absent follow-up
    EventType.MANAGEMENT_CHANGE: 14.0,  # reshapes the narrative for a couple of weeks
    EventType.LABOR: 7.0,  # acute (strike/layoff) then fades unless prolonged
    EventType.OTHER: 3.0,  # unknown -> conservative short memory, don't over-weight noise
}


def half_life_days(event_type: EventType) -> float:
    """The typical persistence, in days, of an event of `event_type` (Section 7.3 step 4)."""
    return EVENT_HALF_LIFE_DAYS[event_type]


@dataclass(frozen=True)
class EventClassification:
    """One article's classified event type, with the full label distribution kept for audit."""

    event_type: EventType
    confidence: float
    scores: dict[EventType, float]
    half_life_days: float


@lru_cache(maxsize=1)
def _load_classifier() -> "Pipeline":
    # Imported lazily (and cached) so importing this module -- and collecting
    # the test suite -- never pulls in torch/transformers or the ~1.6GB model
    # weights. The weights download on first real call; Section 6.10 caches
    # them between CI runs.
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=_MODEL_NAME)


def _empty_result() -> EventClassification:
    return EventClassification(
        event_type=EventType.OTHER,
        confidence=0.0,
        scores={},
        half_life_days=EVENT_HALF_LIFE_DAYS[EventType.OTHER],
    )


def _result_from_raw(raw: dict[str, Any]) -> EventClassification:
    """Turn one transformers zero-shot result dict into an `EventClassification`.

    The pipeline returns `labels` sorted by descending `score`; we map those
    phrases back to `EventType`s, then apply the confidence-threshold rule
    that derives OTHER (module docstring).
    """
    scores = {
        _PHRASE_TO_TYPE[label]: float(score)
        for label, score in zip(raw["labels"], raw["scores"], strict=True)
        if label in _PHRASE_TO_TYPE
    }
    if not scores:
        return _empty_result()

    top_type = max(scores, key=lambda t: scores[t])
    top_score = scores[top_type]
    event_type = top_type if top_score >= CONFIDENCE_THRESHOLD else EventType.OTHER
    return EventClassification(
        event_type=event_type,
        confidence=top_score,
        scores=scores,
        half_life_days=EVENT_HALF_LIFE_DAYS[event_type],
    )


def classify(text: str) -> EventClassification:
    """Classify a single article's `text` (headline, optionally + summary).

    Empty/whitespace text short-circuits to OTHER without invoking the model.
    """
    if not text or not text.strip():
        return _empty_result()
    raw = _load_classifier()(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    return _result_from_raw(dict(raw))


def classify_articles(
    articles: pd.DataFrame,
    *,
    text_columns: tuple[str, ...] = ("title", "summary"),
    max_classified: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    deadline: float | None = None,
) -> pd.Series:
    """Classify each row of `articles`, returning a Series of `EventClassification`.

    Batches all non-empty texts through the pipeline in one call (far faster
    than row-by-row over hundreds of nightly articles) while keeping empty
    rows out of the model. Aligned to `articles.index`; mirrors
    `entity_extraction.tag_articles`' text-column handling so the same
    Tier 1/2/3 frames flow through both.

    `max_classified` bounds how many rows reach the model, and exists because
    **this call is 91% of the entire news pipeline's cost**: measured on real
    headlines, zero-shot classification runs ~347 ms per article against ~19 ms
    for spaCy NER and ~14 ms for FinBERT sentiment. That asymmetry is
    structural, not incidental -- scoring eight candidate labels means eight
    entailment passes through a 400M-parameter model per article.

    Rows beyond the limit get `_empty_result()`, i.e. `EventType.OTHER` with
    confidence 0.0 -- the same value an article the model *can* see but cannot
    confidently place already receives, and which carries its own documented
    half-life. So the degradation is into an existing, honest state rather than
    a new one, and the sentiment score (which is what actually feeds the
    composite) is unaffected because it is computed separately and cheaply.

    Rows are taken in frame order, so a caller wanting "the newest N classified"
    should sort before calling. `None` (the default) classifies everything.

    `deadline` (a `time.monotonic()` value) stops the run cleanly once it
    passes, leaving the remaining rows as `_empty_result()`. It exists because
    `max_classified` alone is a bet on how fast the machine is, and that bet has
    been lost in production: a cap tuned to about 38 minutes on a development
    machine exhausted a 90-minute budget on a GitHub runner, so the step was
    killed and the entire night's news -- classification *and* the sentiment
    that feeds the composite score -- landed nowhere. A deadline degrades the
    same way an over-cap already does, into an event type carrying its own
    documented half-life, instead of into nothing at all. Rows go through in
    chunks so the clock is checked often enough to matter.
    """
    present_columns = [c for c in text_columns if c in articles.columns]

    def _row_text(row: "pd.Series[Any]") -> str:
        parts = [str(row[c]) for c in present_columns if pd.notna(row[c])]
        return " ".join(parts).strip()

    texts = (
        articles.apply(_row_text, axis=1)
        if present_columns
        else pd.Series([""] * len(articles), index=articles.index)
    )

    nonempty_positions = [i for i, t in enumerate(texts) if t]
    if max_classified is not None:
        nonempty_positions = nonempty_positions[:max_classified]
    results: list[EventClassification] = [_empty_result()] * len(texts)
    if nonempty_positions:
        # Loaded before the loop so a cold weight download is never mistaken
        # for slow inference by the first chunk's deadline check.
        classifier = _load_classifier()
        classified = 0
        for start in range(0, len(nonempty_positions), chunk_size):
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    "Event classification stopped at its deadline after %d of %d "
                    "articles; the rest are recorded as '%s'.",
                    classified,
                    len(nonempty_positions),
                    EventType.OTHER,
                )
                break
            chunk = nonempty_positions[start : start + chunk_size]
            batch = [texts.iloc[i] for i in chunk]
            raw_batch = classifier(
                batch,
                candidate_labels=_CANDIDATE_LABELS,
                hypothesis_template=_HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            # Defensive, not currently observed: verified live against the real
            # model that a list input (including a one-item list) always returns
            # a list of dicts, never a bare dict -- kept as cheap insurance
            # against a future transformers version changing that, not because
            # it's been seen to happen.
            raw_list = raw_batch if isinstance(raw_batch, list) else [raw_batch]
            for position, raw in zip(chunk, raw_list, strict=True):
                results[position] = _result_from_raw(dict(raw))
            classified += len(chunk)

    return pd.Series(results, index=articles.index)
