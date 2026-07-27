from typing import Any
from unittest.mock import Mock, patch

import pytest

from quantpulse.llm import providers


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    for limiter in providers._rate_limiters.values():
        monkeypatch.setattr(limiter, "wait", lambda: None)


def _settings(**overrides: Any) -> Mock:
    settings = Mock()
    settings.llm_enabled = True
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "test-gemini-key"
    settings.groq_api_key = "test-groq-key"
    settings.gemini_model = "gemini-2.5-flash"
    settings.groq_model = "llama-3.3-70b-versatile"
    settings.ollama_model = "llama3.1:8b"
    settings.ollama_host = "http://localhost:11434"
    settings.llm_temperature = 0.2
    settings.llm_max_output_tokens = 400
    settings.llm_timeout_seconds = 30.0
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _gemini_response(text: str = "Narrated.") -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _groq_response(text: str = "Narrated.") -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


class TestGrounding:
    def test_context_is_delimited_and_precedes_the_prompt(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(provider, "_complete", return_value="ok") as complete:
            provider.generate("Why is it rated Buy?", "Composite score: 81.0")

        _, user_message = complete.call_args[0]
        assert "--- BEGIN DATA ---" in user_message
        assert "Composite score: 81.0" in user_message
        assert user_message.index("Composite score") < user_message.index("Why is it rated Buy?")

    def test_every_provider_sends_the_same_grounding_instruction(self) -> None:
        settings = _settings()
        for provider in (
            providers.GeminiProvider(settings),
            providers.GroqProvider(settings),
            providers.OllamaProvider(settings),
        ):
            with patch.object(provider, "_complete", return_value="ok") as complete:
                provider.generate("Explain.", "data")
            system_instruction, _ = complete.call_args[0]
            assert system_instruction == providers.GROUNDING_SYSTEM_INSTRUCTION

    def test_grounding_instruction_forbids_inventing_numbers(self) -> None:
        text = providers.GROUNDING_SYSTEM_INSTRUCTION.lower()
        assert "only the figures given" in text
        assert "never introduce a number" in text
        assert "not financial advice" in text

    def test_no_context_still_works(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(provider, "_complete", return_value="ok") as complete:
            provider.generate("Hello")
        _, user_message = complete.call_args[0]
        assert user_message == "Hello"
        assert "BEGIN DATA" not in user_message

    def test_blank_context_is_treated_as_absent(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(provider, "_complete", return_value="ok") as complete:
            provider.generate("Hello", "   ")
        _, user_message = complete.call_args[0]
        assert "BEGIN DATA" not in user_message

    def test_empty_prompt_is_rejected(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with pytest.raises(ValueError, match="prompt must not be empty"):
            provider.generate("   ")


class TestGeminiProvider:
    def test_sends_expected_payload_and_parses_text(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(
            providers, "post_json", return_value=_gemini_response("Because momentum is strong.")
        ) as post:
            result = provider.generate("Why?", "score: 80")

        assert result == "Because momentum is strong."
        url, kwargs = post.call_args[0][0], post.call_args[1]
        assert url.endswith("/gemini-2.5-flash:generateContent")
        body = kwargs["json_body"]
        assert body["system_instruction"]["parts"][0]["text"] == (
            providers.GROUNDING_SYSTEM_INSTRUCTION
        )
        assert body["generationConfig"]["maxOutputTokens"] == 400

    def test_api_key_travels_in_a_header_not_the_url(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", return_value=_gemini_response()) as post:
            provider.generate("Why?")

        url, kwargs = post.call_args[0][0], post.call_args[1]
        assert "test-gemini-key" not in url
        assert kwargs.get("params") is None
        assert kwargs["headers"]["x-goog-api-key"] == "test-gemini-key"

    def test_unavailable_without_a_key(self) -> None:
        assert providers.GeminiProvider(_settings(gemini_api_key=None)).is_available() is False
        assert providers.GeminiProvider(_settings()).is_available() is True

    def test_missing_candidates_raises_llm_error(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", return_value={"candidates": []}):
            with pytest.raises(providers.LLMError, match="no candidates"):
                provider.generate("Why?")

    def test_multipart_response_is_concatenated(self) -> None:
        response = {"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", return_value=response):
            assert provider.generate("Why?") == "AB"


class TestGroqProvider:
    def test_sends_openai_shaped_messages_and_parses_content(self) -> None:
        provider = providers.GroqProvider(_settings())
        with patch.object(providers, "post_json", return_value=_groq_response("Hello.")) as post:
            result = provider.generate("Why?", "score: 80")

        assert result == "Hello."
        kwargs = post.call_args[1]
        messages = kwargs["json_body"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == providers.GROUNDING_SYSTEM_INSTRUCTION
        assert messages[1]["role"] == "user"
        assert kwargs["headers"]["Authorization"] == "Bearer test-groq-key"

    def test_unavailable_without_a_key(self) -> None:
        assert providers.GroqProvider(_settings(groq_api_key=None)).is_available() is False

    def test_missing_choices_raises_llm_error(self) -> None:
        provider = providers.GroqProvider(_settings())
        with patch.object(providers, "post_json", return_value={"choices": []}):
            with pytest.raises(providers.LLMError, match="no choices"):
                provider.generate("Why?")


class TestOllamaProvider:
    def test_sends_prompt_and_parses_response(self) -> None:
        provider = providers.OllamaProvider(_settings())
        with patch.object(providers, "post_json", return_value={"response": "Local."}) as post:
            result = provider.generate("Why?", "score: 80")

        assert result == "Local."
        url, kwargs = post.call_args[0][0], post.call_args[1]
        assert url == "http://localhost:11434/api/generate"
        assert kwargs["json_body"]["system"] == providers.GROUNDING_SYSTEM_INSTRUCTION
        assert kwargs["json_body"]["stream"] is False

    def test_trailing_slash_in_host_is_handled(self) -> None:
        provider = providers.OllamaProvider(_settings(ollama_host="http://localhost:11434/"))
        with patch.object(providers, "post_json", return_value={"response": "ok"}) as post:
            provider.generate("Why?")
        assert post.call_args[0][0] == "http://localhost:11434/api/generate"

    def test_needs_no_api_key(self) -> None:
        assert providers.OllamaProvider(_settings()).is_available() is True

    def test_bad_shape_raises_llm_error(self) -> None:
        provider = providers.OllamaProvider(_settings())
        with patch.object(providers, "post_json", return_value={"nope": 1}):
            with pytest.raises(providers.LLMError, match="unexpected Ollama response"):
                provider.generate("Why?")


class TestProviderResolution:
    def test_build_provider_by_name(self) -> None:
        settings = _settings()
        assert isinstance(providers.build_provider("gemini", settings), providers.GeminiProvider)
        assert isinstance(providers.build_provider("groq", settings), providers.GroqProvider)
        assert isinstance(providers.build_provider("ollama", settings), providers.OllamaProvider)

    def test_unknown_provider_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown LLM provider"):
            providers.build_provider("gpt-9", _settings())

    def test_disabled_yields_no_provider(self) -> None:
        assert providers.get_provider(_settings(llm_enabled=False)) is None

    def test_unconfigured_key_yields_no_provider(self) -> None:
        assert providers.get_provider(_settings(gemini_api_key=None)) is None

    def test_configured_yields_the_selected_provider(self) -> None:
        provider = providers.get_provider(_settings(llm_provider="groq"))
        assert isinstance(provider, providers.GroqProvider)


class TestModuleLevelGenerate:
    def test_returns_none_when_no_provider_configured(self) -> None:
        with patch.object(providers, "get_provider", return_value=None):
            assert providers.generate("Why?", "data") is None

    def test_swallows_llm_error_into_none(self) -> None:
        # A missing paragraph of prose must never take down a page whose
        # computed numbers are already ready to render.
        failing = Mock()
        failing.generate.side_effect = providers.LLMError("quota exhausted")
        assert providers.generate("Why?", "data", provider=failing) is None

    def test_passes_through_a_successful_narration(self) -> None:
        ok = Mock()
        ok.generate.return_value = "Because momentum is strong."
        assert providers.generate("Why?", "data", provider=ok) == "Because momentum is strong."

    def test_does_not_swallow_programming_errors(self) -> None:
        broken = Mock()
        broken.generate.side_effect = ValueError("prompt must not be empty")
        with pytest.raises(ValueError):
            providers.generate("Why?", "data", provider=broken)


class TestFailureHandling:
    def test_network_failure_becomes_llm_error(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", side_effect=OSError("connection refused")):
            with pytest.raises(providers.LLMError, match="gemini request failed"):
                provider.generate("Why?")

    def test_empty_text_response_is_an_error_not_an_empty_string(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", return_value=_gemini_response("   ")):
            with pytest.raises(providers.LLMError, match="empty response"):
                provider.generate("Why?")

    def test_response_text_is_stripped(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", return_value=_gemini_response("  spaced  ")):
            assert provider.generate("Why?") == "spaced"

    def test_repeated_failures_open_the_circuit_breaker(self) -> None:
        provider = providers.GeminiProvider(_settings())
        with patch.object(providers, "post_json", side_effect=OSError("down")):
            for _ in range(5):
                with pytest.raises(providers.LLMError):
                    provider.generate("Why?")
        # The 6th call short-circuits without another HTTP attempt.
        with patch.object(providers, "post_json") as post:
            with pytest.raises(providers.LLMError):
                provider.generate("Why?")
        post.assert_not_called()
