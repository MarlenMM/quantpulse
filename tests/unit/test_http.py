from unittest.mock import Mock, patch

import pytest
import requests

from quantpulse.ingestion import http
from quantpulse.ingestion.http import get_bytes, get_json, get_text


def _response(
    status_code: int, json_data: object = None, text: str = "", content: bytes = b""
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = text
    response.content = content
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


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_get_text_returns_raw_body_on_success(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.return_value = _response(200, text="<rss><channel/></rss>")

    assert get_text("http://example.com") == "<rss><channel/></rss>"
    mock_get.assert_called_once()


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_get_text_retries_on_429_then_succeeds(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.side_effect = [_response(429), _response(200, text="ok")]

    assert get_text("http://example.com", max_retries=2) == "ok"
    assert mock_get.call_count == 2


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_get_bytes_returns_raw_content_on_success(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.return_value = _response(200, content=b"PK\x03\x04binary-zip-bytes")

    assert get_bytes("http://example.com") == b"PK\x03\x04binary-zip-bytes"
    mock_get.assert_called_once()


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
def test_get_bytes_retries_on_429_then_succeeds(mock_get: Mock, mock_sleep: Mock) -> None:
    mock_get.side_effect = [_response(429), _response(200, content=b"ok")]

    assert get_bytes("http://example.com", max_retries=2) == b"ok"
    assert mock_get.call_count == 2


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.post")
def test_post_json_sends_body_and_parses_response(mock_post: Mock, _sleep: Mock) -> None:
    mock_post.return_value = _response(200, {"ok": True})
    result = http.post_json("https://example.com/v1", json_body={"prompt": "hi"})

    assert result == {"ok": True}
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"prompt": "hi"}


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.post")
def test_post_json_retries_429_like_the_get_helpers(mock_post: Mock, _sleep: Mock) -> None:
    mock_post.side_effect = [_response(429, {}), _response(200, {"ok": True})]
    assert http.post_json("https://example.com/v1", json_body={}) == {"ok": True}
    assert mock_post.call_count == 2


@patch("quantpulse.ingestion.http.time.sleep", return_value=None)
@patch("quantpulse.ingestion.http.requests.get")
@patch("quantpulse.ingestion.http.requests.post")
def test_get_helpers_still_use_requests_get(mock_post: Mock, mock_get: Mock, _sleep: Mock) -> None:
    # The POST addition must not have rerouted the GET path (e.g. via
    # requests.request), which every ingestion client depends on.
    mock_get.return_value = _response(200, {"ok": True})
    http.get_json("https://example.com/data")
    mock_get.assert_called_once()
    mock_post.assert_not_called()
