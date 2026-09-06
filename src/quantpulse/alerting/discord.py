"""Delivering a digest to a Discord webhook (Section 10).

Discord over Gmail SMTP, and the reason is the size of the secret rather than
the elegance of the API. A webhook is **one opaque URL**: create it in a
channel's settings, paste it into a repo secret, done -- and revoking it is one
click in that same menu. Gmail SMTP needs a host, a port, a from-address, a
to-address and an app password, and that app password is a credential against
the whole mailbox rather than against one channel. Five secrets instead of one,
with the widest blast radius of the five being the one that must be stored.

The URL is a credential in its own right -- anyone holding it can post into that
channel -- so Section 18's "never hardcoded, never logged" covers it. `requests`
writes the full URL into the message of the `HTTPError` it raises, and that
message is what a public CI log records, so every error out of this module is
re-raised with the URL scrubbed and deliberately *not* chained to the original.
"""

import logging

from quantpulse.ingestion import http

__all__ = ["MESSAGE_LIMIT", "WebhookError", "redact", "send", "split_message"]

logger = logging.getLogger(__name__)

#: Discord's documented per-message `content` limit, in characters.
MESSAGE_LIMIT = 2000


class WebhookError(RuntimeError):
    """A webhook post failed. Never carries the webhook URL's secret path."""


def redact(webhook_url: str) -> str:
    """The webhook URL with its id and token removed, but still recognisable.

    Scrubbed to uselessness is its own failure -- a misconfigured secret has to
    stay diagnosable -- so the scheme and host survive and everything that
    identifies (or authenticates to) the channel does not. Anything that is not
    a URL at all collapses to a constant rather than raising: this runs inside
    an error handler, and a crash there would replace a diagnosable failure with
    an undiagnosable one.
    """
    marker = "://"
    if marker not in webhook_url:
        return "<not a url>"
    scheme, _, rest = webhook_url.partition(marker)
    host = rest.split("/", 1)[0]
    if not host or " " in scheme or " " in host:
        return "<not a url>"
    return f"{scheme}://{host}/api/webhooks/…"


def _describe(error: Exception) -> str:
    """An error in words, carrying its HTTP status and never its URL.

    `requests` puts the full URL in `str(error)`, so only the type name and the
    status code -- read off the attached response, not off the message -- may
    be repeated.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        return type(error).__name__
    return f"{type(error).__name__} (HTTP {status})"


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """`text` as a list of chunks that each fit in one Discord message.

    Splits on line boundaries so a chunk never ends mid-word, and hard-splits a
    single line that is longer than the whole limit rather than dropping it.
    `build_digest` caps its sections precisely so this stays a safety net --
    but a net that silently discarded the line it caught would be worse than
    none, given nothing renders these messages in a test.
    """
    if not text.strip():
        return []
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send(webhook_url: str, text: str) -> int:
    """Post `text` to `webhook_url`, returning how many messages it took.

    Zero for an empty digest, which is the normal quiet-night outcome and not
    an error. Raises `WebhookError` on anything else, with the URL scrubbed.
    """
    chunks = split_message(text)
    if not chunks:
        return 0
    destination = redact(webhook_url)
    for index, chunk in enumerate(chunks, start=1):
        try:
            status = http.post_for_status(webhook_url, json_body={"content": chunk})
        except Exception as error:
            # `from None`, not `from error`. Chaining prints the original
            # exception's message in the traceback, and that message is where
            # `requests` put the full URL -- which would undo the redaction one
            # line above it.
            #
            # The status code is dug out of the exception rather than left in
            # it. `post_for_status` raises on a 4xx, so this -- not the check
            # below -- is the path a revoked or deleted webhook takes, and
            # "failed: HTTPError" does not tell its reader whether the URL is
            # wrong (404), the webhook was revoked (401), or Discord is rate
            # limiting (429). Redaction that costs diagnosability is a bad
            # trade; this keeps both.
            raise WebhookError(
                f"posting message {index}/{len(chunks)} to {destination} failed: {_describe(error)}"
            ) from None
        if not 200 <= status < 300:
            # A 3xx, in practice: `post_for_status` raises on 4xx/5xx, so this
            # catches the redirect that `requests` did not follow rather than
            # an error status. Kept because "the webhook answered something
            # other than success" must never read as delivered.
            raise WebhookError(
                f"posting message {index}/{len(chunks)} to {destination} returned HTTP {status}"
            )
    logger.info("Alert delivered to %s in %d message(s)", destination, len(chunks))
    return len(chunks)
