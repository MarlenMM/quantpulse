from unittest.mock import Mock, patch

from quantpulse.llm import chatbot


def _stub_provider(text: str = "Answered.") -> Mock:
    provider = Mock()
    provider.generate.return_value = text
    return provider


class TestConversation:
    def test_with_turn_is_immutable(self) -> None:
        first = chatbot.Conversation()
        second = first.with_turn("user", "Why is NVDA rated Buy?")
        assert first.turns == ()
        assert len(second.turns) == 1
        assert second.turns[0].role == "user"

    def test_recent_caps_what_gets_sent_but_keeps_full_scrollback(self) -> None:
        conversation = chatbot.Conversation()
        for i in range(10):
            conversation = conversation.with_turn("user", f"question {i}")

        assert len(conversation.turns) == 10  # UI can still render everything
        recent = conversation.recent()
        assert len(recent) == chatbot.MAX_HISTORY_TURNS
        assert recent[-1].content == "question 9"

    def test_recent_zero_limit_sends_nothing(self) -> None:
        conversation = chatbot.Conversation().with_turn("user", "hi")
        assert conversation.recent(0) == ()

    def test_negative_limit_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="limit must be >= 0"):
            chatbot.Conversation().recent(-1)


class TestBuildChatContext:
    def test_blocks_are_numbered_and_delimited(self) -> None:
        context = chatbot.build_chat_context(["Composite score: 81", "Beta: 1.2"])
        assert "[Data block 1]" in context
        assert "[Data block 2]" in context
        assert "Composite score: 81" in context

    def test_no_blocks_is_visibly_empty_rather_than_silently_absent(self) -> None:
        context = chatbot.build_chat_context([])
        assert "[No data blocks were supplied" in context

    def test_blank_blocks_are_dropped(self) -> None:
        context = chatbot.build_chat_context(["", "   ", "real data"])
        assert "[Data block 1]" in context
        assert "[Data block 2]" not in context

    def test_history_is_appended_when_present(self) -> None:
        history = (
            chatbot.Conversation()
            .with_turn("user", "Why Buy?")
            .with_turn("assistant", "Momentum is strong.")
        )
        context = chatbot.build_chat_context(["score: 81"], history)
        assert "[Recent conversation]" in context
        assert "User: Why Buy?" in context
        assert "Assistant: Momentum is strong." in context

    def test_empty_history_adds_no_transcript_section(self) -> None:
        context = chatbot.build_chat_context(["score: 81"], chatbot.Conversation())
        assert "[Recent conversation]" not in context

    def test_only_recent_turns_reach_the_model(self) -> None:
        conversation = chatbot.Conversation()
        for i in range(10):
            conversation = conversation.with_turn("user", f"question {i}")
        context = chatbot.build_chat_context(["data"], conversation)
        assert "question 9" in context
        assert "question 0" not in context


class TestAnswer:
    def test_grounds_the_answer_in_the_supplied_blocks(self) -> None:
        provider = _stub_provider("It ranks in the top decile.")
        result = chatbot.answer(
            "Why is NVDA rated Buy?", ["Composite score: 87"], provider=provider
        )

        assert result == "It ranks in the top decile."
        prompt, context = provider.generate.call_args[0]
        assert "Why is NVDA rated Buy?" in prompt
        assert "Composite score: 87" in context

    def test_instruction_permits_saying_the_app_lacks_the_data(self) -> None:
        provider = _stub_provider()
        chatbot.answer("What is its P/E?", ["Composite score: 87"], provider=provider)
        prompt, _ = provider.generate.call_args[0]
        assert "the app doesn't currently show that" in prompt
        assert "Never guess a figure" in prompt

    def test_instruction_forbids_computing_and_personalized_advice(self) -> None:
        provider = _stub_provider()
        chatbot.answer("Should I sell?", ["Composite score: 87"], provider=provider)
        prompt, _ = provider.generate.call_args[0]
        assert "Do not calculate new metrics" in prompt
        assert "do not advise" in prompt
        assert "licensed professional" in prompt

    def test_empty_question_spends_no_request(self) -> None:
        provider = _stub_provider()
        assert chatbot.answer("   ", ["data"], provider=provider) is None
        provider.generate.assert_not_called()

    def test_an_over_long_question_is_truncated_not_sent_whole(self) -> None:
        # The question is re-sent with every later turn, and the deployed demo
        # shares one free-tier key across all visitors -- so an unbounded box is
        # the cheapest way for one paste to exhaust everyone's daily quota.
        provider = _stub_provider()
        chatbot.answer("Q" * 50_000, ["data"], provider=provider)
        prompt, _ = provider.generate.call_args[0]
        sent = prompt.split("User question: ", 1)[1]
        assert sent == "Q" * chatbot.MAX_QUESTION_CHARS

    def test_a_normal_length_question_is_untouched(self) -> None:
        provider = _stub_provider()
        question = "Why is NVDA rated Strong Buy?"
        chatbot.answer(question, ["data"], provider=provider)
        prompt, _ = provider.generate.call_args[0]
        assert question in prompt

    def test_works_with_no_context_blocks_at_all(self) -> None:
        provider = _stub_provider()
        chatbot.answer("What is a P/E ratio?", provider=provider)
        _, context = provider.generate.call_args[0]
        assert "[No data blocks were supplied" in context

    def test_degrades_to_none_without_a_provider(self) -> None:
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            assert chatbot.answer("Why is NVDA rated Buy?", ["score: 87"]) is None


class TestDisclaimer:
    def test_disclaimer_is_exported_for_standing_ui_display(self) -> None:
        text = chatbot.ADVICE_DISCLAIMER.lower()
        assert "not financial advice" in text
        assert "not a registered investment advisor" in text
