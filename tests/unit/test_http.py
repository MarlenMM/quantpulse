from unittest.mock import Mock, patch

import pytest
import requests

from quantpulse.ingestion.http import get_json


def _response(status_code: int, json_data: object = None) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_returns_parsed_json_on_success(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.return_value = _response(200, {"ok": True})

    assert get_json("http://example.com") == {"ok": True}
    mock_get.assert_called_once()


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_retries_on_429_then_succeeds(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.side_effect = [_response(429), _response(200, {"ok": True})]

    assert get_json("http://example.com", max_retries=2) == {"ok": True}
    assert mock_get.call_count == 2


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_raises_after_exhausting_retries_on_5xx(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.return_value = _response(500)

    with pytest.raises(requests.HTTPError):
        get_json("http://example.com", max_retries=2)

    assert mock_get.call_count == 3


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_does_not_retry_non_retryable_4xx(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.return_value = _response(404)

    with pytest.raises(requests.HTTPError):
        get_json("http://example.com", max_retries=2)

    mock_get.assert_called_once()


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_retries_on_connection_error(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.side_effect = [requests.ConnectionError("boom"), _response(200, {"ok": True})]

    assert get_json("http://example.com", max_retries=2) == {"ok": True}
    assert mock_get.call_count == 2
