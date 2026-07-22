"""FinBERT sentiment scoring + recency-weighted decay (Section 7.3 steps 3-4).

Two pieces, matching the plan's own split:

1. **Sentiment scoring** (`score_sentiment` / `score_articles`): `ProsusAI/
   finbert` (free, local, no per-call cost) scores each article's polarity.
   FinBERT returns a 3-way distribution (positive/negative/neutral); this
   module reduces it to a single signed `polarity` in `[-1, 1]` as
   `P(positive) - P(negative)`, the standard, simple way to fold a 3-class
   sentiment distribution into one comparable number -- 0 means either
   genuinely neutral or a real positive/negative split, both of which are
   legitimately "no net signal," and the full distribution stays on
   `SentimentScore` for anyone who needs to tell those apart.

2. **Recency decay** (`decay_weight` / `aggregate_decayed_sentiment`): "a
   headline from three weeks ago shouldn't carry the same weight as one from
   this morning" (Section 7.3 step 4). Exponential decay,
   `weight = 0.5 ** (age_days / half_life_days)`, against the per-event-type
   half-life the Opus part's `event_classifier.EVENT_HALF_LIFE_DAYS` already
   defines -- an earnings surprise (5-day half-life) has faded to background
   noise well before a Fed decision (21-day half-life) has. Aggregating
   several articles for one symbol is then a decay-weighted average, not a
   flat mean: yesterday's earnings beat should outvote a three-week-old,
   already-stale headline, even if the stale one was more emphatic.
"""

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd

from quantpulse.news_intelligence.event_classifier import EventType, half_life_days

if TYPE_CHECKING:
    from transformers import Pipeline

_MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class SentimentScore:
    """One article's FinBERT sentiment. `polarity` is the single comparable number."""

    polarity: float  # positive - negative, in [-1, 1]
    positive: float
    negative: float
    neutral: float


_NEUTRAL_SCORE = SentimentScore(polarity=0.0, positive=0.0, negative=0.0, neutral=1.0)


@lru_cache(maxsize=1)
def _load_sentiment_model() -> "Pipeline":
    # Imported lazily (and cached) so importing this module -- and collecting
    # the test suite -- never pulls in torch/transformers or the model
    # weights. Weights download on first real call; Section 6.10 caches them
    # between CI runs. `top_k=None` requests the full 3-label distribution
    # rather than just the winning label, which `polarity` needs.
    from transformers import pipeline

    return pipeline("text-classification", model=_MODEL_NAME, top_k=None)


def _result_from_raw(raw: list[dict[str, Any]]) -> SentimentScore:
    by_label = {str(item["label"]).lower(): float(item["score"]) for item in raw}
    positive = by_label.get("positive", 0.0)
    negative = by_label.get("negative", 0.0)
    neutral = by_label.get("neutral", 0.0)
    return SentimentScore(
        polarity=positive - negative, positive=positive, negative=negative, neutral=neutral
    )


def score_sentiment(text: str) -> SentimentScore:
    """FinBERT polarity for a single article's `text` (headline, optionally + summary).

    Empty/whitespace text short-circuits to a neutral score without invoking
    the model.
    """
    if not text or not text.strip():
        return _NEUTRAL_SCORE
    # With `top_k=None`, a single (non-batched) string input still comes back
    # wrapped one list deeper than the label list itself -- [[{label dicts}]],
    # not [{label dicts}] -- verified against the live model (see the opt-in
    # integration test), not assumed.
    raw = _load_sentiment_model()(text, truncation=True)
    return _result_from_raw(raw[0])


def score_articles(
    articles: pd.DataFrame, *, text_columns: tuple[str, ...] = ("title", "summary")
) -> pd.Series:
    """FinBERT-score each row of `articles`, returning a Series of `SentimentScore`.

    Batches all non-empty texts through the pipeline in one call, mirroring
    `event_classifier.classify_articles`'s text-column handling and
    empty-row/single-item-batch treatment exactly, so the same Tier 1/2/3
    frames flow through entity extraction, event classification, and
    sentiment scoring identically.
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
    results: list[SentimentScore] = [_NEUTRAL_SCORE] * len(texts)
    if nonempty_positions:
        batch = [texts.iloc[i] for i in nonempty_positions]
        raw_batch = _load_sentiment_model()(batch, truncation=True)
        # Defensive, not currently observed: verified live that a list input
        # (including a one-item list) always returns a list of label-lists,
        # never a single flattened label-list -- kept as cheap insurance
        # against a future transformers version changing that.
        raw_list = raw_batch if isinstance(raw_batch[0], list) else [raw_batch]
        for position, raw in zip(nonempty_positions, raw_list, strict=True):
            results[position] = _result_from_raw(raw)

    return pd.Series(results, index=articles.index)


def decay_weight(age_days: float, half_life_days_value: float) -> float:
    """Exponential recency weight: `0.5 ** (age_days / half_life_days_value)`.

    A zero-age article gets weight 1.0; one `half_life_days_value` old gets
    0.5; two half-lives old gets 0.25, and so on. Negative age (a clock-skew
    or "future-dated" article) is clamped to zero rather than allowed to
    produce a weight > 1.0 -- decay only ever discounts, never amplifies.
    """
    if half_life_days_value <= 0:
        raise ValueError(f"half_life_days_value must be positive, got {half_life_days_value}")
    return 0.5 ** (max(age_days, 0.0) / half_life_days_value)


def decay_weight_for_event(age_days: float, event_type: EventType) -> float:
    """`decay_weight` using `event_type`'s own half-life (Section 7.3 step 4)."""
    return decay_weight(age_days, half_life_days(event_type))


def _age_in_days(published_at: date | datetime | Any, as_of: date | datetime) -> float | None:
    """Fractional days between `published_at` and `as_of`, or None if `published_at` is unusable.

    `pd.isna` alone correctly covers every unusable form here -- `None`,
    `float('nan')`, and `pd.NaT` all read as missing; a real date/datetime/
    `Timestamp` reads as not-missing -- so no separate type-by-type handling
    is needed.
    """
    if pd.isna(published_at):
        return None
    published = pd.Timestamp(published_at)
    reference = pd.Timestamp(as_of)
    return (reference - published).total_seconds() / 86400.0


@dataclass(frozen=True)
class AggregatedSentiment:
    """A symbol's decay-weighted sentiment across its articles.

    Feeds the `sentiment_scores` table (Section 13).
    """

    symbol: str
    score: float  # decay-weighted average polarity, in [-1, 1]
    mention_volume: int  # count of contributing articles
    total_weight: float  # sum of decay weights actually used -- near 0 means "all stale"


def aggregate_decayed_sentiment(
    symbol: str,
    articles: pd.DataFrame,
    as_of: date | datetime,
    *,
    matched_symbols_column: str = "matched_symbols",
    sentiment_column: str = "sentiment",
    event_type_column: str = "event_type",
    published_at_column: str = "published_at",
) -> AggregatedSentiment | None:
    """`symbol`'s decay-weighted average sentiment across its matched articles in `articles`.

    Filters `articles` to rows where `symbol` appears in `matched_symbols_column`
    (a list per row, as produced by `entity_extraction.tag_articles`), then
    computes a decay-weighted average of `sentiment_column`'s polarity using
    each row's `event_type_column` half-life and its age relative to `as_of`.
    Returns `None` if no article matches `symbol`, or if every matching
    article's `published_at_column` is unusable (can't decay an undated
    article, so it's excluded rather than guessed at) -- an empty result is
    a real "no signal," not a false zero.

    Note `score` is a *normalized average*, not a magnitude: with a single
    contributing article its decay weight cancels out of the division, so
    `score` equals that article's raw polarity regardless of how stale it
    is -- decay only changes how much that article outweighs *other*
    articles, not how far its own vote shrinks toward zero in isolation.
    `total_weight` is where staleness actually shows up (a lone weight near
    0 means "this score rests on old evidence"); a caller wanting sentiment
    itself to fade toward neutral as evidence goes stale should look at
    `total_weight`, not assume `score` does that.

    `sentiment_column` may hold either `SentimentScore` objects (from
    `score_articles`) or plain floats (an already-extracted polarity).
    """
    matches = articles[matched_symbols_column].apply(lambda syms: symbol in syms)
    relevant = articles[matches]

    if relevant.empty:
        return None

    weighted_sum = 0.0
    weight_sum = 0.0
    for row in relevant.itertuples():
        published_at = getattr(row, published_at_column)
        age_days = _age_in_days(published_at, as_of)
        if age_days is None:
            continue
        event_type = getattr(row, event_type_column)
        weight = decay_weight_for_event(age_days, event_type)
        sentiment = getattr(row, sentiment_column)
        polarity = sentiment.polarity if isinstance(sentiment, SentimentScore) else float(sentiment)
        weighted_sum += polarity * weight
        weight_sum += weight

    if weight_sum <= 0.0:
        return None

    return AggregatedSentiment(
        symbol=symbol,
        score=weighted_sum / weight_sum,
        mention_volume=len(relevant),
        total_weight=weight_sum,
    )
