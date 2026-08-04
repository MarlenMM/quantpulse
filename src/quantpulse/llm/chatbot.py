"""Grounded chatbot over already-computed results (Sections 10, 11 use 3).

Section 10 describes this precisely: "a small free-tier-LLM-powered chat box
that can answer questions like 'why is NVDA rated Buy?' by being fed the
already-computed structured data as context (**never asked to compute anything
itself**)." That parenthesis is the entire design brief, and it is what makes a
chatbot safe to put on a finance tool at all.

Three things this module does that a naive chat wrapper doesn't:

1. **Retrieval is the caller's job, and it is explicit.** `answer()` takes
   `context_blocks` -- the already-built blocks from `narrative.py`'s builders,
   or any other rendering of computed rows. There is no tool-calling, no
   database access, and no "let the model figure out what it needs": if a fact
   wasn't put in front of the model, it does not exist for that turn.
2. **Not knowing is an allowed, instructed answer.** `_CHAT_INSTRUCTION` tells
   the model to say the app doesn't have something rather than fill the gap.
   The most dangerous chatbot failure here isn't a refusal, it's a confident
   invented P/E ratio, and a model that has never been given permission to say
   "I don't have that" will invent one.
3. **It declines to give personalized advice, by instruction and by design.**
   Section 2 puts goal-based financial planning out of scope and Section 19
   requires the not-a-financial-advisor disclaimer; a chat box is exactly where
   a user will try "should I sell my Apple shares?" anyway. The instruction
   redirects those to what the data shows, and `ADVICE_DISCLAIMER` is exported
   so the UI can render the standing disclaimer alongside the transcript rather
   than relying on the model to reproduce it every turn.

History is capped (`MAX_HISTORY_TURNS`) because free-tier quota is per-token as
well as per-request, and an unbounded transcript quietly turns a cheap feature
into an expensive one (Section 11's budget math assumes short calls).

Like everything in `llm/`, this degrades to `None` when no provider is
configured -- the app is fully usable with the chat box simply absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from quantpulse.llm.providers import LLMProvider, generate

__all__ = [
    "MAX_HISTORY_TURNS",
    "ADVICE_DISCLAIMER",
    "ChatTurn",
    "Conversation",
    "build_chat_context",
    "answer",
]

# Kept short deliberately: each turn is re-sent as context on every request, so
# an uncapped history multiplies token cost per message. Six turns is enough
# for "why is it rated Buy?" -> "what about its debt?" style follow-ups.
MAX_HISTORY_TURNS = 6

# A single question is capped for the same reason the history is: it is re-sent
# with every subsequent turn, so one pasted wall of text inflates the cost of
# the whole remaining conversation. The deployed demo shares one free-tier key
# across every visitor, so an unbounded question box is also the cheapest way
# for one user to exhaust the day's quota for everyone. Generous enough that no
# real question is affected (the same order as `narrative`'s excerpt cap).
MAX_QUESTION_CHARS = 2000

# Section 19's disclaimer, exported so the UI can render it as standing text
# next to the chat box. Displaying it once, always, is more reliable than
# hoping the model appends it to each answer -- and doesn't burn output tokens.
ADVICE_DISCLAIMER = (
    "QuantPulse is an educational research tool, not financial advice, and not a "
    "registered investment advisor. Figures shown are computed from historical data "
    "and may be incomplete or stale."
)

_CHAT_INSTRUCTION = (
    "You are answering questions in QuantPulse's chat box about analysis the application "
    "has already computed.\n"
    "- Answer only from the data blocks provided in this conversation. If the answer isn't "
    "there, say that the app doesn't currently show that, and if useful name which page or "
    "metric would have it. Never guess a figure.\n"
    "- Do not calculate new metrics, rank stocks, or forecast prices yourself -- the "
    "application does that; you report what it produced.\n"
    "- If asked what the user personally should do (buy/sell/hold, position sizing, "
    "portfolio changes, tax treatment), do not advise. Explain what the computed data "
    "shows, note that this is an educational tool rather than financial advice, and "
    "suggest a licensed professional for a personal decision.\n"
    "- Keep answers to a few sentences unless genuinely more is needed."
)


@dataclass(frozen=True)
class ChatTurn:
    """One prior exchange in the conversation."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Conversation:
    """A capped chat transcript.

    `recent()` is what actually gets sent -- the last `MAX_HISTORY_TURNS`
    turns. Older turns stay in `turns` so a UI can still render the full
    scrollback; only what is *sent to the model* is trimmed.
    """

    turns: tuple[ChatTurn, ...] = field(default_factory=tuple)

    def with_turn(self, role: Literal["user", "assistant"], content: str) -> Conversation:
        """A new `Conversation` with one turn appended (immutable, like the dataclasses above)."""
        return Conversation(turns=(*self.turns, ChatTurn(role=role, content=content)))

    def recent(self, limit: int = MAX_HISTORY_TURNS) -> tuple[ChatTurn, ...]:
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        return self.turns[-limit:] if limit else ()


def build_chat_context(context_blocks: Sequence[str], history: Conversation | None = None) -> str:
    """Assemble the data blocks (and recent transcript) the model may answer from.

    Blocks are numbered and delimited so the model can attribute a figure to a
    specific one, and so an empty set of blocks is visibly empty rather than
    silently absent -- a question asked with no data attached should produce
    "the app doesn't show that," which only happens if the model can tell the
    difference between "no data supplied" and "data supplied but unhelpful."
    """
    sections: list[str] = []
    blocks = [block.strip() for block in context_blocks if block and block.strip()]
    if blocks:
        for index, block in enumerate(blocks, start=1):
            sections.append(f"[Data block {index}]\n{block}")
    else:
        sections.append("[No data blocks were supplied for this question.]")

    if history is not None:
        recent = history.recent()
        if recent:
            transcript = "\n".join(
                f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
                for turn in recent
            )
            sections.append(f"[Recent conversation]\n{transcript}")
    return "\n\n".join(sections)


def answer(
    question: str,
    context_blocks: Sequence[str] = (),
    *,
    history: Conversation | None = None,
    provider: LLMProvider | None = None,
) -> str | None:
    """Answer `question` strictly from `context_blocks`; `None` when no LLM is configured.

    `context_blocks` are pre-rendered strings the caller retrieved -- typically
    `narrative.build_rating_context(...)` or a similar rendering of stored rows
    for whatever the question is about. Nothing here queries the database: what
    the model can answer from is entirely determined by what the caller passed,
    which is what makes the answer auditable.

    Returns `None` for an empty question, so a stray Enter keypress in the UI
    doesn't spend a free-tier request. A question longer than
    `MAX_QUESTION_CHARS` is truncated rather than rejected -- the useful part of
    an over-long paste is almost always at the front, and silently answering a
    trimmed question beats refusing outright.
    """
    trimmed = question.strip()[:MAX_QUESTION_CHARS]
    if not trimmed:
        return None
    prompt = f"{_CHAT_INSTRUCTION}\n\nUser question: {trimmed}"
    return generate(prompt, build_chat_context(context_blocks, history), provider=provider)
