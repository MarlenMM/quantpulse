import time
from typing import Any

import requests

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Any:
    """GET `url` as JSON, retrying with linear backoff on network errors or a 429/5xx."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            raise
        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
            time.sleep(backoff_seconds * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")  # pragma: no cover
