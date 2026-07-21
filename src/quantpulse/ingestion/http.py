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


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Any:
    """GET `url` as JSON, retrying network errors and 429/5xx with exponential backoff.

    A 429 or 503 carrying a `Retry-After` header waits exactly that long
    instead of guessing — the polite behavior that keeps a free-tier key
    from being escalated to an outright ban (Section 19).
    """
    for attempt in range(max_retries + 1):
        try:
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

        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")  # pragma: no cover
