from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import edgar_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edgar_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    settings.sec_edgar_user_agent = "test-agent test@example.com"
    return settings


def test_get_cik_for_ticker_looks_up_and_pads(tmp_path: Path) -> None:
    lookup_response = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.edgar_client.get_json", return_value=lookup_response),
    ):
        cik = edgar_client.get_cik_for_ticker("aapl")

    assert cik == "0000320193"


def test_get_cik_for_ticker_raises_for_unknown_ticker(tmp_path: Path) -> None:
    lookup_response = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.edgar_client.get_json", return_value=lookup_response),
    ):
        with pytest.raises(ValueError):
            edgar_client.get_cik_for_ticker("NOPE")


def test_fetch_company_facts_uses_padded_cik_in_url(tmp_path: Path) -> None:
    lookup_response = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    facts_response = {"cik": 320193, "entityName": "Apple Inc.", "facts": {}}

    with (
        patch(
            "quantpulse.ingestion.edgar_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.edgar_client.get_json",
            side_effect=[lookup_response, facts_response],
        ) as mock_get_json,
    ):
        result = edgar_client.fetch_company_facts("AAPL")

    assert result["entityName"] == "Apple Inc."
    called_url = mock_get_json.call_args_list[1].args[0]
    assert "CIK0000320193.json" in called_url
