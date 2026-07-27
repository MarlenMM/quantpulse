"""Free-tier LLM provider abstraction -- Gemini / Groq / Ollama (Sections 4.3, 11).

Section 11's framing is the design constraint, not a slogan: **the LLM is a
narrator, not an analyst.** Every number this project shows was computed by the
analysis engine; the model's only job is restating those numbers in English.
Two structural choices here enforce that rather than trusting each call site to
remember it:

1. **`generate(prompt, context)` takes the grounding context as its own
   parameter**, exactly as Section 11 specifies. The context block -- the
   already-computed scores/forecasts/indicator values -- is not something a
   caller may forget to include; it is the second half of the interface. The
   base class assembles system instruction + context + question in a fixed
   order for every provider, so grounding cannot vary by backend.
2. **A single `GROUNDING_SYSTEM_INSTRUCTION`**, shared by all three adapters
   and wired into whichever system-prompt slot each API exposes (Gemini's
   `system_instruction`, Groq's `system` message, Ollama's `system` field).
   Swapping providers therefore cannot silently change the rules the model is
   working under -- which is the whole premise of ADR 4.3's "you can freely
   change providers without touching the analysis engine."

**Degradation is a feature, not an error path.** Section 11 promises the app
works with the LLM "entirely turned off," so `get_provider()` returns `None`
when narration is disabled or unconfigured, and the module-level `generate()`
returns `None` rather than raising. A missing API key is a *supported
configuration*, not a failure: the UI shows the numbers without a paragraph of
prose around them. Only a genuine call failure (network, quota, malformed
response) raises `LLMError`.

Reuse: HTTP retry/backoff/`Retry-After` handling comes from
`ingestion.http.post_json`, and repeated-failure protection from
`ingestion.circuit_breaker` -- a free-tier LLM endpoint rate-limits and falls
over like any other free-tier API, so it gets the same treatment rather than a
second, subtly different implementation (Sections 6.5, 6.12). Each provider
also carries a conservative token-bucket limiter matched to its documented
free-tier RPM.

All three adapters speak plain REST via `requests`, deliberately not the three
vendor SDKs: the request bodies below are a few lines each, and three heavy,
fast-moving dependencies for that is a poor trade in a project whose lockfile
discipline (Section 29) is a selling point.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from quantpulse.config import Settings, get_settings
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.http import post_json
from quantpulse.ingestion.rate_limit import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

__all__ = [
    "GROUNDING_SYSTEM_INSTRUCTION",
    "LLMError",
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "OllamaProvider",
    "build_provider",
    "get_provider",
    "generate",
]

# The rules every call runs under, on every backend. Deliberately explicit
# about the one failure mode that would make this whole layer a liability:
# a fluent paragraph containing a number the engine never computed is worse
# than no paragraph at all, because it looks exactly as trustworthy.
GROUNDING_SYSTEM_INSTRUCTION = (
    "You are the narration layer of QuantPulse, a quantitative stock-research tool. "
    "All analysis has already been computed by the application; your only job is to "
    "explain the supplied numbers in plain English.\n"
    "Rules:\n"
    "- Use ONLY the figures given to you in the context. Never introduce a number, "
    "price, ratio, date, or fact that is not present there.\n"
    "- Never recompute, re-score, re-rank, or second-guess the supplied values, and "
    "never predict prices of your own.\n"
    "- If the context does not contain what is needed to answer, say so plainly "
    "instead of guessing or filling the gap.\n"
    "- Be concise and neutral. No hype, no urgency, no persuasion.\n"
    "- This is educational research output, not financial advice, and you are not a "
    "financial advisor. Do not tell the reader what they personally should buy, sell, "
    "or hold, and do not tailor anything to their circumstances.\n"
    "- Explain what the numbers say and why the system reached its conclusion."
)

# Conservative free-tier pacing (Section 4.3's own quota notes). Gemini's free
# flash tier is the tightest on requests-per-minute; Groq is roomier; Ollama is
# local and unmetered but still gets a limiter so a runaway loop can't peg the
# machine. These bound bursts -- the daily quotas are far beyond what narrating
# a few dozen page views a day consumes (Section 11's budget math).
_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "gemini": (10, 60.0),
    "groq": (25, 60.0),
    "ollama": (60, 60.0),
}
_rate_limiters: dict[str, TokenBucketRateLimiter] = {
    name: TokenBucketRateLimiter(capacity=capacity, per_seconds=per_seconds)
    for name, (capacity, per_seconds) in _RATE_LIMITS.items()
}


class LLMError(RuntimeError):
    """A configured provider was called and genuinely failed.

    Distinct from "no provider configured," which is not an error at all --
    that path returns `None` (see `get_provider`), because Section 11 makes
    running without an LLM a supported mode rather than a broken one.
    """


class LLMProvider(ABC):
    """Common `generate(prompt, context) -> str` interface (Section 11).

    Subclasses implement only `_complete(system_instruction, user_message)`;
    prompt assembly, rate limiting, and the circuit breaker live here so all
    three backends behave identically in every respect except which HTTP
    endpoint they talk to.
    """

    name: str

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @abstractmethod
    def _complete(self, system_instruction: str, user_message: str) -> str:
        """Send one grounded request and return the model's text."""

    def is_available(self) -> bool:
        """Whether this provider has everything it needs to be called.

        Overridden by the keyed providers; Ollama needs no credential (its
        reachability can only be discovered by actually calling it, so an
        unreachable local server surfaces as an `LLMError`, not as silent
        unavailability -- if you explicitly configured Ollama, its being down
        is a real failure worth seeing).
        """
        return True

    def generate(self, prompt: str, context: str | None = None) -> str:
        """Narrate `context` in response to `prompt` (Section 11's interface).

        `context` is the already-computed structured data -- scores, forecasts,
        indicator values -- rendered as text by `narrative.py`'s builders. It is
        presented to the model in a clearly delimited block, ahead of the
        request, so the instruction "use only these figures" has an unambiguous
        referent.

        Raises `LLMError` if the call fails or the response is empty/malformed.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        user_message = prompt.strip()
        if context and context.strip():
            user_message = (
                "Here is the computed data you must base your answer on:\n"
                "--- BEGIN DATA ---\n"
                f"{context.strip()}\n"
                "--- END DATA ---\n\n"
                f"{prompt.strip()}"
            )

        _rate_limiters[self.name].wait()
        try:
            with get_breaker(f"llm_{self.name}").guard():
                text = self._complete(GROUNDING_SYSTEM_INSTRUCTION, user_message)
        except LLMError:
            raise
        except Exception as exc:  # network error, circuit open, bad payload
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        cleaned = text.strip()
        if not cleaned:
            raise LLMError(f"{self.name} returned an empty response")
        return cleaned


class GeminiProvider(LLMProvider):
    """Google Gemini free tier -- ADR 4.3's recommended default."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def is_available(self) -> bool:
        return bool(self._settings.gemini_api_key)

    def _complete(self, system_instruction: str, user_message: str) -> str:
        api_key = self._settings.gemini_api_key
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": self._settings.llm_temperature,
                "maxOutputTokens": self._settings.llm_max_output_tokens,
            },
        }
        response = post_json(
            f"{self.base_url}/{self._settings.gemini_model}:generateContent",
            json_body=payload,
            # The key travels in a header, not the documented `?key=` query
            # parameter: query strings end up in proxy logs and error messages,
            # and Section 18's "never logged" is easier to actually guarantee
            # when the secret was never in the URL to begin with.
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            timeout=self._settings.llm_timeout_seconds,
        )
        return _extract_gemini_text(response)


class GroqProvider(LLMProvider):
    """Groq free tier (open Llama/Qwen models), OpenAI-compatible chat completions."""

    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def is_available(self) -> bool:
        return bool(self._settings.groq_api_key)

    def _complete(self, system_instruction: str, user_message: str) -> str:
        api_key = self._settings.groq_api_key
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        payload = {
            "model": self._settings.groq_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_output_tokens,
        }
        response = post_json(
            f"{self.base_url}/chat/completions",
            json_body=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=self._settings.llm_timeout_seconds,
        )
        return _extract_openai_text(response)


class OllamaProvider(LLMProvider):
    """Local Ollama -- unlimited, offline, no key, no external dependency (ADR 4.3)."""

    name = "ollama"

    def _complete(self, system_instruction: str, user_message: str) -> str:
        payload = {
            "model": self._settings.ollama_model,
            "system": system_instruction,
            "prompt": user_message,
            "stream": False,
            "options": {
                "temperature": self._settings.llm_temperature,
                "num_predict": self._settings.llm_max_output_tokens,
            },
        }
        response = post_json(
            f"{self._settings.ollama_host.rstrip('/')}/api/generate",
            json_body=payload,
            timeout=self._settings.llm_timeout_seconds,
        )
        if not isinstance(response, dict) or not isinstance(response.get("response"), str):
            raise LLMError(f"unexpected Ollama response shape: {type(response).__name__}")
        return response["response"]


def _extract_gemini_text(response: Any) -> str:
    """Pull the text out of a Gemini `generateContent` response.

    Raises `LLMError` rather than `KeyError`/`IndexError` on an unexpected
    shape, so a caller only ever has one exception type to handle -- and a
    response blocked by a safety filter (no `candidates`) reads as a clear
    failure instead of an obscure index error.
    """
    if not isinstance(response, dict):
        raise LLMError(f"unexpected Gemini response shape: {type(response).__name__}")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMError("Gemini returned no candidates (possibly filtered or quota-limited)")
    parts = candidates[0].get("content", {}).get("parts")
    if not isinstance(parts, list) or not parts:
        raise LLMError("Gemini candidate contained no content parts")
    return "".join(str(part.get("text", "")) for part in parts)


def _extract_openai_text(response: Any) -> str:
    """Pull the text out of an OpenAI-compatible chat-completions response (Groq)."""
    if not isinstance(response, dict):
        raise LLMError(f"unexpected Groq response shape: {type(response).__name__}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("Groq returned no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise LLMError("Groq choice contained no message content")
    return content


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}


def build_provider(name: str, settings: Settings | None = None) -> LLMProvider:
    """Construct the named provider regardless of whether it's configured/enabled.

    The unconditional constructor behind `get_provider`'s policy checks --
    useful for tests and for a caller that wants to surface *why* a provider
    isn't usable (`is_available()`) rather than just getting `None`.
    """
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(f"unknown LLM provider {name!r}; expected one of {sorted(_PROVIDERS)}")
    return provider_cls(settings)


def get_provider(settings: Settings | None = None) -> LLMProvider | None:
    """The configured provider, or `None` when narration is off or unconfigured.

    `None` is the ordinary "no LLM" path Section 11 requires the whole app to
    tolerate -- disabled via `LLM_ENABLED=false`, or simply no API key set.
    Callers render their numbers without narration; nothing raises.
    """
    resolved = settings or get_settings()
    if not resolved.llm_enabled:
        return None
    provider = build_provider(resolved.llm_provider, resolved)
    if not provider.is_available():
        logger.debug("LLM provider %s is not configured; narration disabled", provider.name)
        return None
    return provider


def generate(
    prompt: str, context: str | None = None, *, provider: LLMProvider | None = None
) -> str | None:
    """Grounded one-shot narration, or `None` if unavailable/failed.

    The forgiving entry point every narration site in this project uses: it
    swallows `LLMError` into `None` (logged, not raised) because a missing
    paragraph of prose must never take down a page whose actual content --
    the computed numbers -- is already sitting there ready to render.
    """
    resolved = provider or get_provider()
    if resolved is None:
        return None
    try:
        return resolved.generate(prompt, context)
    except LLMError:
        logger.warning("LLM narration failed; continuing without it", exc_info=True)
        return None
