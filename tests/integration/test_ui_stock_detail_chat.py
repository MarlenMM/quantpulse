"""End-to-end test of the Stock Detail chat box (Sections 10, 11).

Runs the real Streamlit page through `streamlit.testing.v1.AppTest` -- an
actual script run, real widgets, real session state -- against a temporary
SQLite database and a stubbed LLM provider. That covers the wiring the pure
unit tests in `test_ui_stock_detail.py` cannot: that the box renders at all,
that submitting a question reaches `chatbot.answer` with the page's own
context blocks attached, that the transcript survives a rerun, and that the
whole section disappears when no provider is configured.

Streamlit's own crash-prone browser path is deliberately avoided -- `AppTest`
runs the script in-process, so this is fast and needs no server or browser.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from datetime import time as dtime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage.models import Base, CompositeScore, Forecast, PriceHistory, Ticker

PAGE = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "2_Stock_Detail.py")
# Anchored to today, not to a literal date. Every window the Stock Detail page
# reads is measured from `date.today()` -- news over 21 days, the cross-asset
# macro series over 60 -- so fixture rows pinned to a fixed calendar date age
# out of those windows and the sections under test silently stop rendering.
# That is exactly what happened: these tests passed for a month and then began
# failing on a commit that touched none of them. Seeding relative to today
# keeps the fixtures inside the windows the code actually queries.
AS_OF = date.today()


@pytest.fixture
def seeded_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'chat.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(
            Ticker(
                symbol="NVDA",
                name="NVIDIA Corporation",
                sector="Information Technology",
                asset_type="equity",
                is_active=True,
            )
        )
        for offset in range(60):
            day = AS_OF - timedelta(days=offset)
            session.add(
                PriceHistory(
                    symbol="NVDA",
                    date=day,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0 + offset * 0.1,
                    adj_close=100.0 + offset * 0.1,
                    volume=1_000_000,
                )
            )
        session.add(
            CompositeScore(
                symbol="NVDA",
                date=AS_OF,
                profile="balanced",
                composite_score=87.3,
                percentile_rank=96.2,
                rating="strong_buy",
                data_confidence=84.0,
                **{f"{category}_score": 70.0 for category in CATEGORIES},
            )
        )
        session.add(
            Forecast(
                symbol="NVDA",
                generated_date=AS_OF,
                horizon_days=5,
                model_name="gbr",
                point_return=0.0123,
                point_price=124.9,
                lower_price=120.1,
                upper_price=129.8,
                historical_hit_rate=0.58,
            )
        )
        session.commit()
    return engine


@pytest.fixture
def app(seeded_engine: Engine) -> Iterator[AppTest]:
    """The page, wired to the temp database, with Streamlit's data cache bypassed."""
    factory = sessionmaker(bind=seeded_engine)

    from contextlib import contextmanager

    @contextmanager
    def fake_get_session() -> Iterator[object]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    # `lib.data`'s readers are `@st.cache_data`-wrapped; clearing keeps one
    # test's rows from being served to the next.
    from lib import data as lib_data

    lib_data.screener_rows.clear()
    lib_data.forecasts.clear()
    with patch("lib.data.get_session", fake_get_session):
        yield AppTest.from_file(PAGE, default_timeout=30)


def _stub_provider(reply: str = "It ranks in the top decile.") -> Mock:
    provider = Mock()
    provider.name = "stub"
    provider.generate.return_value = reply
    return provider


class TestChatRendersOnlyWithAProvider:
    def test_absent_entirely_without_an_llm(self, app: AppTest) -> None:
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            app.run()
        assert not app.exception
        headers = [element.value for element in app.subheader]
        # Guard against a vacuous pass: the page must have rendered its actual
        # content and *chosen* not to draw the chat box, rather than bailing out
        # early (an empty database would also produce "no chat input").
        assert "Forecast" in headers
        assert "Ask about this stock" not in headers
        assert not app.chat_input

    def test_present_when_a_provider_is_configured(self, app: AppTest) -> None:
        with patch("quantpulse.llm.providers.get_provider", return_value=_stub_provider()):
            app.run()
        assert not app.exception
        assert "Ask about this stock" in [element.value for element in app.subheader]
        assert len(app.chat_input) == 1

    def test_standing_disclaimer_is_rendered_without_asking_the_model_for_it(
        self, app: AppTest
    ) -> None:
        # Section 19's disclaimer is UI-rendered text, not something the model
        # is trusted to reproduce each turn (and it costs no output tokens).
        from quantpulse.llm.chatbot import ADVICE_DISCLAIMER

        with patch("quantpulse.llm.providers.get_provider", return_value=_stub_provider()):
            app.run()
        captions = [element.value for element in app.caption]
        assert ADVICE_DISCLAIMER in captions


class TestAskingAQuestion:
    def test_answer_is_grounded_in_the_pages_own_context_blocks(self, app: AppTest) -> None:
        provider = _stub_provider("Momentum and technicals lead.")
        with patch("quantpulse.llm.providers.get_provider", return_value=provider):
            app.run()
            app.chat_input[0].set_value("Why is NVDA rated Strong Buy?").run()

        assert not app.exception
        _prompt, context = provider.generate.call_args[0]
        # The rating block and the forecast block the page is displaying.
        assert "Rating scheme: RELATIVE" in context
        assert "Forecast model: gbr" in context
        assert "[Data block 1]" in context and "[Data block 2]" in context

    def test_question_and_answer_both_land_in_the_transcript(self, app: AppTest) -> None:
        provider = _stub_provider("Momentum and technicals lead.")
        with patch("quantpulse.llm.providers.get_provider", return_value=provider):
            app.run()
            app.chat_input[0].set_value("Why is NVDA rated Strong Buy?").run()

        rendered = [element.value for element in app.markdown]
        assert "Why is NVDA rated Strong Buy?" in rendered
        assert "Momentum and technicals lead." in rendered

    def test_transcript_survives_a_rerun_and_is_replayed_as_history(self, app: AppTest) -> None:
        provider = _stub_provider("First answer.")
        with patch("quantpulse.llm.providers.get_provider", return_value=provider):
            app.run()
            app.chat_input[0].set_value("First question?").run()
            provider.generate.return_value = "Second answer."
            app.chat_input[0].set_value("Second question?").run()

        # The prior exchange is sent back as conversation history, which is what
        # makes a follow-up like "what about its debt?" resolvable at all.
        _prompt, context = provider.generate.call_args[0]
        assert "[Recent conversation]" in context
        assert "First question?" in context
        assert "First answer." in context

        rendered = [element.value for element in app.markdown]
        assert "First answer." in rendered and "Second answer." in rendered

    def test_a_failed_generation_is_reported_rather_than_rendering_an_empty_bubble(
        self, app: AppTest
    ) -> None:
        provider = _stub_provider()
        provider.generate.return_value = None
        with patch("quantpulse.llm.providers.get_provider", return_value=provider):
            app.run()
            app.chat_input[0].set_value("Why is NVDA rated Strong Buy?").run()

        assert not app.exception
        assert any("did not return an answer" in element.value for element in app.markdown)


class TestNarrativeUsesTwoAndFourAreWired:
    """Section 11's other two LLM uses had no page.

    `explain_sentiment_move` and `summarize_filing_excerpt` were both fully
    built, prompt-designed and unit-tested, and neither was imported by
    anything under `app/`. Like the chat box, both must be entirely absent
    without a provider and present with one.
    """

    @pytest.fixture
    def with_news(self, seeded_engine: Engine) -> Engine:
        from datetime import datetime

        from quantpulse.storage.models import NewsEvent, SentimentScore

        factory = sessionmaker(bind=seeded_engine)
        with factory() as session:
            for index, (title, polarity) in enumerate(
                [("NVDA beats expectations", 0.8), ("Supply chain concerns", -0.4)]
            ):
                session.add(
                    NewsEvent(
                        article_id=f"a{index}",
                        tier=1,
                        title=title,
                        published_at=datetime.combine(AS_OF - timedelta(days=1), dtime.min),
                        matched_symbols=["NVDA"],
                        event_type="earnings",
                        sentiment_score=polarity,
                        source="rss",
                        source_url=f"https://example.test/{index}",
                    )
                )
            for offset, score in ((1, 0.55), (8, 0.10)):
                session.add(
                    SentimentScore(
                        symbol="NVDA",
                        date=AS_OF - timedelta(days=offset),
                        source="tier1_aggregate",
                        sentiment_score=score,
                        mention_volume=4,
                        total_weight=2.5,
                    )
                )
            session.commit()
        from lib import data as lib_data

        lib_data.symbol_news.clear()
        lib_data.sentiment_history.clear()
        return seeded_engine

    def test_sentiment_summary_is_absent_without_a_provider(
        self, app: AppTest, with_news: Engine
    ) -> None:
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            app.run()
        assert not app.exception
        assert "What's driving this" in [element.value for element in app.subheader]
        assert not [i.value for i in app.info if "coverage" in str(i.value)]

    def test_sentiment_summary_renders_and_is_grounded_in_the_headlines(
        self, app: AppTest, with_news: Engine
    ) -> None:
        provider = _stub_provider("Earnings beat drove the move.")
        with patch("quantpulse.llm.providers.get_provider", return_value=provider):
            app.run()
        assert not app.exception
        assert "Earnings beat drove the move." in [str(i.value) for i in app.info]
        # The context the model saw must contain the actual headlines and their
        # already-assigned polarities -- not a re-scoring request.
        contexts = " ".join(str(call.args[1]) for call in provider.generate.call_args_list)
        assert "NVDA beats expectations" in contexts
        assert "+0.80" in contexts
        assert "+0.55" in contexts  # the current stored reading
        assert "+0.10" in contexts  # ...and the previous one, so "move" means something

    def test_filing_summary_section_is_absent_without_a_provider(self, app: AppTest) -> None:
        with patch("quantpulse.llm.providers.get_provider", return_value=None):
            app.run()
        assert "Latest SEC filing" not in [element.value for element in app.subheader]

    def test_filing_summary_does_not_fetch_until_asked(self, app: AppTest) -> None:
        # A 10-K is several megabytes; rendering the page must not pull one.
        with (
            patch("quantpulse.llm.providers.get_provider", return_value=_stub_provider()),
            patch("quantpulse.ingestion.edgar_client.fetch_filing_excerpt") as fetch,
        ):
            app.run()
        assert "Latest SEC filing" in [element.value for element in app.subheader]
        fetch.assert_not_called()

    def test_filing_summary_renders_when_the_button_is_pressed(self, app: AppTest) -> None:
        from lib import data as lib_data

        lib_data.filing_excerpt.clear()
        provider = _stub_provider("- Revenue rose.\n- Costs rose faster.")
        excerpt = {
            "symbol": "NVDA",
            "form_type": "10-Q",
            "filed_date": AS_OF,
            "section": "Management's Discussion and Analysis",
            "excerpt": "Revenue grew because we sold more widgets.",
            "source_url": "https://example.test/filing.htm",
        }
        with (
            patch("quantpulse.llm.providers.get_provider", return_value=provider),
            patch("quantpulse.ingestion.edgar_client.fetch_filing_excerpt", return_value=excerpt),
        ):
            app.run()
            next(b for b in app.button if "Summarize" in b.label).click().run()
        assert not app.exception
        assert any("- Revenue rose." in str(i.value) for i in app.info)
        contexts = " ".join(str(call.args[1]) for call in provider.generate.call_args_list)
        assert "Revenue grew because we sold more widgets." in contexts
        # The excerpt is the only free-text payload in the LLM layer, so it must
        # be framed as quoted source material rather than pasted in bare.
        assert "quoted source material" in contexts
