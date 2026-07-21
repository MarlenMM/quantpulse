from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from quantpulse.ingestion import reddit_client


def _fake_settings(
    tmp_path: Path,
    *,
    client_id: str | None = "test-id",
    client_secret: str | None = "test-secret",
) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    settings.reddit_client_id = client_id
    settings.reddit_client_secret = client_secret
    settings.reddit_user_agent = "quantpulse-test/0.1"
    return settings


def _fake_submission(post_id: str, title: str) -> Mock:
    submission = Mock()
    submission.id = post_id
    submission.title = title
    submission.created_utc = datetime(2026, 7, 20).timestamp()
    submission.score = 42
    submission.num_comments = 7
    submission.permalink = f"/r/stocks/comments/{post_id}/"
    return submission


def test_fetch_mentions_raises_without_credentials(tmp_path: Path) -> None:
    with patch(
        "quantpulse.ingestion.reddit_client.get_settings",
        return_value=_fake_settings(tmp_path, client_id=None),
    ):
        with pytest.raises(ValueError):
            reddit_client.fetch_mentions("AAPL")


def test_fetch_mentions_normalizes_submissions_across_subreddits(tmp_path: Path) -> None:
    fake_reddit = MagicMock()

    def _subreddit(name: str) -> Mock:
        sub = Mock()
        sub.search.return_value = [_fake_submission(f"{name}-1", f"AAPL discussion in {name}")]
        return sub

    fake_reddit.subreddit.side_effect = _subreddit

    with (
        patch(
            "quantpulse.ingestion.reddit_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.reddit_client.praw.Reddit", return_value=fake_reddit),
    ):
        df = reddit_client.fetch_mentions("AAPL", subreddits=("stocks", "investing"))

    assert list(df.columns) == reddit_client._COLUMNS
    assert len(df) == 2
    assert set(df["subreddit"]) == {"stocks", "investing"}
    assert (df["symbol"] == "AAPL").all()
    assert (df["tier"] == 1).all()
    assert df.iloc[0]["permalink"].startswith("https://reddit.com/r/")
    # No raw post body/selftext should ever be pulled through (Section 19).
    assert "selftext" not in df.columns
    assert "body" not in df.columns


def test_fetch_mentions_builds_read_only_client_from_settings(tmp_path: Path) -> None:
    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value.search.return_value = []

    with (
        patch(
            "quantpulse.ingestion.reddit_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.reddit_client.praw.Reddit", return_value=fake_reddit
        ) as mock_reddit_cls,
    ):
        reddit_client.fetch_mentions("AAPL", subreddits=("stocks",))

    _, kwargs = mock_reddit_cls.call_args
    assert kwargs["client_id"] == "test-id"
    assert kwargs["client_secret"] == "test-secret"
    assert kwargs["read_only"] is True
