from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import gdelt_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def test_fetch_articles_normalizes_response(tmp_path: Path) -> None:
    raw_response = {
        "articles": [
            {
                "title": "AI export controls tighten",
                "url": "https://example.com/a1",
                "domain": "example.com",
                "seendate": "20260722T120000Z",
                "sourcecountry": "United States",
                "language": "English",
            }
        ]
    }
    with (
        patch(
            "quantpulse.ingestion.gdelt_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.gdelt_client.get_json", return_value=raw_response
        ) as mock_get_json,
    ):
        df = gdelt_client.fetch_articles("semiconductor export controls")

    assert list(df.columns) == gdelt_client._ARTICLE_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["title"] == "AI export controls tighten"
    assert df.iloc[0]["query"] == "semiconductor export controls"
    _, kwargs = mock_get_json.call_args
    assert kwargs["params"]["query"] == "semiconductor export controls"
    assert kwargs["params"]["mode"] == "artlist"


def test_fetch_articles_caps_max_records_at_api_ceiling(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.gdelt_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch(
            "quantpulse.ingestion.gdelt_client.get_json", return_value={"articles": []}
        ) as mock_get_json,
    ):
        gdelt_client.fetch_articles("ai regulation", max_records=10_000)

    _, kwargs = mock_get_json.call_args
    assert kwargs["params"]["maxrecords"] == "250"


def test_fetch_articles_handles_missing_articles_key(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.gdelt_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.gdelt_client.get_json", return_value={}),
    ):
        df = gdelt_client.fetch_articles("ai regulation")

    assert df.empty
    assert list(df.columns) == gdelt_client._ARTICLE_COLUMNS


def test_fetch_tone_timeline_normalizes_response(tmp_path: Path) -> None:
    raw_response = {
        "timeline": [
            {
                "data": [
                    {"date": "20260701000000", "value": -1.25},
                    {"date": "20260702000000", "value": -0.5},
                ]
            }
        ]
    }
    with (
        patch(
            "quantpulse.ingestion.gdelt_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.gdelt_client.get_json", return_value=raw_response),
    ):
        df = gdelt_client.fetch_tone_timeline("fed policy")

    assert list(df.columns) == gdelt_client._TONE_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["date"] == date(2026, 7, 1)
    assert df.iloc[0]["tone"] == -1.25
    assert df.iloc[0]["query"] == "fed policy"


def test_fetch_tone_timeline_handles_missing_timeline(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.gdelt_client.get_settings", return_value=_fake_settings(tmp_path)
        ),
        patch("quantpulse.ingestion.gdelt_client.get_json", return_value={}),
    ):
        df = gdelt_client.fetch_tone_timeline("fed policy")

    assert df.empty
    assert list(df.columns) == gdelt_client._TONE_COLUMNS


def test_parse_gdelt_datetime_handles_both_observed_formats() -> None:
    assert gdelt_client._parse_gdelt_datetime("20260722T120000Z") is not None
    assert gdelt_client._parse_gdelt_datetime("20260722120000") is not None
    assert gdelt_client._parse_gdelt_datetime("not-a-date") is None
    assert gdelt_client._parse_gdelt_datetime(None) is None
