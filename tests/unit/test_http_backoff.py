from unittest.mock import Mock, patch

from quantpulse.ingestion.http import _parse_retry_after, compute_backoff, get_json


def _response(
    status_code: int, headers: dict[str, str] | None = None, json_data: object = None
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def test_compute_backoff_is_bounded_by_exponential_ceiling() -> None:
    for _ in range(50):
        assert 0.0 <= compute_backoff(0, 1.0) <= 1.0
        assert 0.0 <= compute_backoff(3, 1.0) <= 8.0  # 1 * 2**3


def test_compute_backoff_respects_the_cap() -> None:
    for _ in range(50):
        assert compute_backoff(10, 1.0, cap_seconds=5.0) <= 5.0


def test_retry_after_overrides_computed_backoff() -> None:
    assert compute_backoff(0, 1.0, retry_after=7.0) == 7.0


def test_retry_after_is_still_capped() -> None:
    assert compute_backoff(0, 1.0, retry_after=1000.0, cap_seconds=30.0) == 30.0


def test_parse_retry_after_reads_integer_seconds() -> None:
    assert _parse_retry_after(_response(429, {"Retry-After": "12"})) == 12.0


def test_parse_retry_after_returns_none_when_absent_or_unparseable() -> None:
    assert _parse_retry_after(_response(429, {})) is None
    assert (
        _parse_retry_after(_response(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None
    )


@patch("quantpulse.ingestion.http.time.sleep")
@patch("quantpulse.ingestion.http.requests.get")
def test_get_json_waits_exactly_the_retry_after_on_429(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.side_effect = [
        _response(429, {"Retry-After": "3"}),
        _response(200, json_data={"ok": True}),
    ]

    assert get_json("http://example.com", max_retries=2) == {"ok": True}
    mock_sleep.assert_called_once_with(3.0)
