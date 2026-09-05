import random
import time
from typing import Any

import requests

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_DEFAULT_BACKOFF_CAP_SECONDS = 30.0


def _parse_retry_after(response: requests.Response) -> float | None:
    """Return the `Retry-After` delay in seconds, if the server sent one as an integer.

    Free-tier APIs that set this header use the integer-seconds form; the
    HTTP-date form is ignored (falls back to computed backoff) rather than
    pulling in date parsing for a case these providers don't use.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def compute_backoff(
    attempt: int,
    base_seconds: float,
    *,
    cap_seconds: float = _DEFAULT_BACKOFF_CAP_SECONDS,
    retry_after: float | None = None,
) -> float:
    """Delay before the next retry.

    Honors a server-supplied `Retry-After` when present; otherwise uses
    exponential backoff (`base * 2**attempt`) with "full jitter" — a uniform
    random draw in [0, backoff] — which spreads retries out so 500 tickers
    failing at once don't all wake up and re-hit a struggling source in
    lockstep. Capped so a late attempt can't sleep for minutes.
    """
    if retry_after is not None:
        return min(retry_after, cap_seconds)
    ceiling = min(cap_seconds, base_seconds * (2**attempt))
    return random.uniform(0.0, ceiling)


def _request_with_retries(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    raise_on_status: bool = True,
) -> requests.Response:
    """Request `url`, retrying network errors and 429/5xx with exponential backoff.

    A 429 or 503 carrying a `Retry-After` header waits exactly that long
    instead of guessing — the polite behavior that keeps a free-tier key
    from being escalated to an outright ban (Section 19).

    `method` is "GET" (the ingestion clients), "POST" (the LLM providers in
    `quantpulse.llm`, whose APIs take a JSON request body) or "HEAD" (asking
    whether a large file exists without downloading it). They are dispatched
    explicitly rather than through `requests.request` so the GET path still
    calls `requests.get` — the retry/backoff behavior every ingestion client
    depends on is unchanged by the other two.

    Retrying a POST is safe for the only POSTs this project makes: an LLM
    completion is a pure function call with no server-side state to duplicate,
    unlike a POST that creates a resource.

    `raise_on_status=False` returns 4xx responses to the caller instead of
    raising. That is for the one case where a 404 is an *answer* rather than a
    failure — asking whether a file has been published yet — and it deliberately
    does not extend to 429/5xx, which still retry above and then raise, because
    those genuinely are failures however the caller means to read them.
    """
    for attempt in range(max_retries + 1):
        try:
            if method == "POST":
                response = requests.post(
                    url, params=params, headers=headers, json=json_body, timeout=timeout
                )
            elif method == "HEAD":
                response = requests.head(
                    url, params=params, headers=headers, timeout=timeout, allow_redirects=True
                )
            else:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(compute_backoff(attempt, backoff_seconds))
                continue
            raise

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
            time.sleep(
                compute_backoff(attempt, backoff_seconds, retry_after=_parse_retry_after(response))
            )
            continue

        if raise_on_status:
            response.raise_for_status()
        return response
    raise RuntimeError("unreachable")  # pragma: no cover


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Any:
    """GET `url` and parse the response as JSON. See `_request_with_retries` for retry behavior."""
    response = _request_with_retries(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    return response.json()


def get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> str:
    """GET `url` as raw text (e.g. RSS/XML). See `_request_with_retries` for retry behavior."""
    response = _request_with_retries(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    return response.text


def post_json(
    url: str,
    *,
    json_body: Any,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Any:
    """POST `json_body` to `url` and parse the response as JSON.

    Same retry/backoff/`Retry-After` behavior as the GET helpers (see
    `_request_with_retries`) — which is exactly why the LLM providers
    (`quantpulse.llm.providers`) use this rather than calling `requests`
    directly: a free-tier LLM endpoint rate-limits like any other free-tier
    API, and there is no reason for it to have its own retry semantics.
    """
    response = _request_with_retries(
        url,
        method="POST",
        params=params,
        headers=headers,
        json_body=json_body,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    return response.json()


def resource_exists(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 2,
    backoff_seconds: float = 1.0,
) -> bool:
    """Whether `url` exists, asked with a HEAD so nothing large is transferred.

    For deciding *which* of several large files to download — a 404 here means
    "not published yet", which is a normal answer, so it comes back as `False`
    rather than as an exception. A 429/5xx still retries and then raises: the
    server being broken is not the same answer as the file being absent, and
    collapsing the two would report a published file as missing during an
    outage.

    Callers should keep this outside their circuit breaker. A breaker exists to
    stop hammering a failing source, and a run of honest 404s is not a failing
    source — counting them would open the circuit and then block the download
    of the file that *was* found.
    """
    response = _request_with_retries(
        url,
        method="HEAD",
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        raise_on_status=False,
    )
    return response.status_code < 400


def get_bytes(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> bytes:
    """GET `url` as raw bytes (e.g. a ZIP download). See `_request_with_retries` for retry behavior.

    Callers fetching a large file (tens/hundreds of MB) should pass a much
    larger `timeout` than the default -- this default matches `get_json`/
    `get_text`'s small-payload assumption, not this function's own use case.
    """
    response = _request_with_retries(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    return response.content
