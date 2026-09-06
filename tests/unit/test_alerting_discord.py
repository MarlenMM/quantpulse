"""Sending the digest, and the two ways sending it could go wrong quietly.

A Discord webhook URL is a credential -- anyone holding it can post into that
channel -- so Section 18's "never hardcoded, never logged" applies to it just
as it does to an API key. `requests` puts the full URL into the message of the
`HTTPError` it raises, and that message is what lands in a public CI log, so
the scrubbing here is load-bearing rather than tidy.
"""

import logging
from unittest.mock import patch

import pytest
import requests

from quantpulse.alerting import discord

_URL = "https://discord.com/api/webhooks/123456789/tOkEn-Th4t-Must-Not-Leak"
_SECRET = "tOkEn-Th4t-Must-Not-Leak"


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_a_short_message_is_one_post() -> None:
    assert discord.split_message("one line") == ["one line"]


def test_an_empty_message_is_no_posts() -> None:
    assert discord.split_message("") == []
    assert discord.split_message("   \n  ") == []


def test_a_long_message_splits_on_line_boundaries() -> None:
    text = "\n".join(f"line {i}" for i in range(400))
    chunks = discord.split_message(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= discord.MESSAGE_LIMIT for chunk in chunks)
    # Nothing lost and nothing invented: the lines come back in order.
    assert "\n".join(chunks).split("\n") == text.split("\n")


def test_a_single_line_longer_than_the_limit_is_split_rather_than_dropped() -> None:
    """Not reachable from `build_digest` today, and that is exactly why it is
    tested: a future line that grows past the limit must not vanish."""
    chunks = discord.split_message("x" * (discord.MESSAGE_LIMIT * 2 + 5))
    assert all(len(chunk) <= discord.MESSAGE_LIMIT for chunk in chunks)
    assert "".join(chunks) == "x" * (discord.MESSAGE_LIMIT * 2 + 5)


def test_the_limit_is_discords_documented_one() -> None:
    assert discord.MESSAGE_LIMIT == 2000


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #


def test_send_posts_the_text_as_discord_content() -> None:
    with patch("quantpulse.alerting.discord.http.post_for_status", return_value=204) as post:
        assert discord.send(_URL, "hello") == 1
    post.assert_called_once()
    assert post.call_args.args[0] == _URL
    assert post.call_args.kwargs["json_body"] == {"content": "hello"}


def test_send_posts_each_chunk_of_a_long_message() -> None:
    text = "\n".join(f"line {i}" for i in range(400))
    with patch("quantpulse.alerting.discord.http.post_for_status", return_value=204) as post:
        sent = discord.send(_URL, text)
    assert sent == post.call_count > 1


def test_send_does_nothing_for_an_empty_message() -> None:
    with patch("quantpulse.alerting.discord.http.post_for_status") as post:
        assert discord.send(_URL, "") == 0
    post.assert_not_called()


# --------------------------------------------------------------------------- #
# Not leaking the webhook URL
# --------------------------------------------------------------------------- #


def test_a_failed_post_raises_without_the_webhook_token_in_the_message() -> None:
    """`requests` writes the full URL into its own error message.

    Verbatim, from a real 404: "404 Client Error: Not Found for url:
    https://discord.com/api/webhooks/123/<token>". Re-raising that as-is puts a
    working credential into a CI log that anyone can read.
    """
    boom = requests.HTTPError(f"404 Client Error: Not Found for url: {_URL}")
    with patch("quantpulse.alerting.discord.http.post_for_status", side_effect=boom):
        with pytest.raises(discord.WebhookError) as caught:
            discord.send(_URL, "hello")
    assert _SECRET not in str(caught.value)
    assert "discord.com" in str(caught.value)


def test_the_raised_error_does_not_chain_the_original_leaking_one() -> None:
    """`raise ... from exc` would print the original message in the traceback,
    which puts the URL back into the log the scrubbing just removed it from."""
    boom = requests.HTTPError(f"404 Client Error: Not Found for url: {_URL}")
    with patch("quantpulse.alerting.discord.http.post_for_status", side_effect=boom):
        with pytest.raises(discord.WebhookError) as caught:
            discord.send(_URL, "hello")
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None or caught.value.__suppress_context__


def test_a_non_2xx_status_raises_without_the_token() -> None:
    with patch("quantpulse.alerting.discord.http.post_for_status", return_value=403):
        with pytest.raises(discord.WebhookError) as caught:
            discord.send(_URL, "hello")
    assert _SECRET not in str(caught.value)
    assert "403" in str(caught.value)


def test_nothing_logged_while_sending_carries_the_token(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with patch("quantpulse.alerting.discord.http.post_for_status", return_value=204):
            discord.send(_URL, "hello")
    assert _SECRET not in caplog.text


def test_redaction_keeps_enough_to_debug_with() -> None:
    """Scrubbed to uselessness is its own failure: "which webhook" has to stay
    answerable, or a misconfigured secret is undiagnosable."""
    redacted = discord.redact(_URL)
    assert _SECRET not in redacted
    assert "123456789" not in redacted
    assert redacted.startswith("https://discord.com/api/webhooks/")


def test_redaction_survives_a_url_that_is_not_a_url() -> None:
    """A pasted-wrong secret must produce a diagnosable error, not a crash
    inside the error handler."""
    assert _SECRET not in discord.redact(f"not a url at all {_SECRET}")
