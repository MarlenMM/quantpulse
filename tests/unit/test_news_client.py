from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import news_client

_SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Sample Feed</title>
<item>
<title>Apple shares rise on strong earnings</title>
<link>https://example.com/article1</link>
<description>Apple shares rose today after beating estimates.</description>
<pubDate>Wed, 22 Jul 2026 12:00:00 GMT</pubDate>
</item>
<item>
<title>Apple announces new product</title>
<link>https://example.com/article2</link>
<description>A new product was unveiled.</description>
<pubDate>Tue, 21 Jul 2026 09:30:00 GMT</pubDate>
</item>
</channel>
</rss>
"""

_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>
"""


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(news_client._yahoo_rate_limiter, "wait", lambda: None)
    monkeypatch.setattr(news_client._google_rate_limiter, "wait", lambda: None)
    monkeypatch.setattr(news_client._seeking_alpha_rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def test_fetch_yahoo_finance_news_parses_entries(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.news_client.get_text", return_value=_SAMPLE_FEED
        ) as mock_get_text,
    ):
        df = news_client.fetch_yahoo_finance_news("AAPL")

    assert list(df.columns) == news_client._COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["title"] == "Apple shares rise on strong earnings"
    assert df.iloc[0]["link"] == "https://example.com/article1"
    assert df.iloc[0]["source"] == "yahoo"
    assert df.iloc[0]["symbol"] == "AAPL"
    assert df.iloc[0]["tier"] == 1
    assert df.iloc[0]["published_at"] is not None
    kwargs = mock_get_text.call_args.kwargs
    assert kwargs["params"]["s"] == "AAPL"


def test_fetch_google_news_uses_company_name_in_query(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.news_client.get_text", return_value=_EMPTY_FEED
        ) as mock_get_text,
    ):
        news_client.fetch_google_news("AAPL", company_name="Apple Inc")

    kwargs = mock_get_text.call_args.kwargs
    assert kwargs["params"]["q"] == "Apple Inc stock"


def test_fetch_google_news_falls_back_to_symbol_without_company_name(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.news_client.get_text", return_value=_EMPTY_FEED
        ) as mock_get_text,
    ):
        news_client.fetch_google_news("AAPL")

    kwargs = mock_get_text.call_args.kwargs
    assert kwargs["params"]["q"] == "AAPL stock"


def test_fetch_seeking_alpha_news_lowercases_symbol_in_url(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.news_client.get_text", return_value=_EMPTY_FEED
        ) as mock_get_text,
    ):
        news_client.fetch_seeking_alpha_news("AAPL")

    called_url = mock_get_text.call_args.args[0]
    assert "aapl.xml" in called_url


def test_fetch_all_tier1_news_concatenates_all_three_sources(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.news_client.get_text", return_value=_SAMPLE_FEED),
    ):
        df = news_client.fetch_all_tier1_news("AAPL")

    assert set(df["source"]) == {"yahoo", "google_news", "seeking_alpha"}
    assert len(df) == 6  # 2 entries x 3 sources


def test_fetch_all_tier1_news_skips_a_failing_source(tmp_path: Path) -> None:
    def _get_text_side_effect(url: str, **kwargs: object) -> str:
        if "seekingalpha" in url:
            raise ConnectionError("feed down")
        return _SAMPLE_FEED

    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.news_client.get_text", side_effect=_get_text_side_effect),
    ):
        df = news_client.fetch_all_tier1_news("AAPL")

    assert set(df["source"]) == {"yahoo", "google_news"}
    assert len(df) == 4


def test_fetch_all_tier1_news_returns_empty_frame_if_all_sources_fail(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.news_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.news_client.get_text", side_effect=ConnectionError("down")),
    ):
        df = news_client.fetch_all_tier1_news("AAPL")

    assert df.empty
    assert list(df.columns) == news_client._COLUMNS


def _pub_date(offset_days: int) -> str:
    stamp = datetime(2026, 7, 22, 12, 0) - timedelta(days=offset_days)
    return stamp.strftime("%a, %d %b %Y %H:%M:%S")


def _many_item_feed(n: int) -> str:
    """A feed of `n` items, oldest last, so recency ordering is checkable."""
    items = "".join(
        f"<item><title>Headline {i}</title>"
        f"<link>https://example.com/a{i}</link>"
        f"<description>Body {i}</description>"
        f"<pubDate>{_pub_date(i)} GMT</pubDate>"
        f"</item>"
        for i in range(n)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>Many</title>{items}</channel></rss>"
    )


class TestArticleCap:
    """The per-symbol cap that keeps the weekly model batch finishable.

    Uncapped, the three Tier-1 feeds return ~150 headlines per ticker, which
    across a 500-name universe is ~75,000 articles pushed through a zero-shot
    NLI model and FinBERT in one batch. That ran 5h38m with no output on a CI
    runner and was killed at the 6-hour job limit, so the whole night -- prices
    and fundamentals included -- was discarded without ever being committed.
    """

    def test_caps_to_the_newest_articles(self, tmp_path: Path) -> None:
        with (
            patch(
                "quantpulse.ingestion.news_client.get_settings",
                return_value=_fake_settings(tmp_path),
            ),
            patch("quantpulse.ingestion.news_client.get_text", return_value=_many_item_feed(60)),
        ):
            df = news_client.fetch_all_tier1_news("AAPL", max_articles=10)

        assert len(df) == 10
        # Newest kept, oldest dropped -- the recency-decay step downstream would
        # have weighted the dropped ones to near nothing anyway. All three
        # sources are stubbed with the same feed here (de-duplication by link is
        # the caller's job), so each headline appears three times and the newest
        # ten rows can only come from the four most recent headlines.
        assert df["published_at"].is_monotonic_decreasing
        assert set(df["title"]) <= {f"Headline {i}" for i in range(4)}
        assert "Headline 59" not in set(df["title"])

    def test_default_cap_is_applied_without_being_asked(self, tmp_path: Path) -> None:
        # The nightly calls this with no `max_articles`, so the default is what
        # actually protects the run.
        per_source = news_client.MAX_ARTICLES_PER_SYMBOL + 25
        with (
            patch(
                "quantpulse.ingestion.news_client.get_settings",
                return_value=_fake_settings(tmp_path),
            ),
            patch(
                "quantpulse.ingestion.news_client.get_text",
                return_value=_many_item_feed(per_source),
            ),
        ):
            df = news_client.fetch_all_tier1_news("AAPL")

        assert len(df) == news_client.MAX_ARTICLES_PER_SYMBOL

    def test_none_disables_the_cap(self, tmp_path: Path) -> None:
        with (
            patch(
                "quantpulse.ingestion.news_client.get_settings",
                return_value=_fake_settings(tmp_path),
            ),
            patch("quantpulse.ingestion.news_client.get_text", return_value=_many_item_feed(50)),
        ):
            df = news_client.fetch_all_tier1_news("AAPL", max_articles=None)

        assert len(df) == 150  # 50 items x 3 sources, untouched

    def test_under_the_cap_is_returned_unchanged(self, tmp_path: Path) -> None:
        with (
            patch(
                "quantpulse.ingestion.news_client.get_settings",
                return_value=_fake_settings(tmp_path),
            ),
            patch("quantpulse.ingestion.news_client.get_text", return_value=_SAMPLE_FEED),
        ):
            df = news_client.fetch_all_tier1_news("AAPL", max_articles=100)

        assert len(df) == 6  # 2 items x 3 sources
