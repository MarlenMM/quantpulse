from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import fred_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path, api_key: str | None = "test-key") -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    settings.fred_api_key = api_key
    return settings


def test_fetch_series_raises_without_api_key(tmp_path: Path) -> None:
    with patch(
        "quantpulse.ingestion.fred_client.get_settings",
        return_value=_fake_settings(tmp_path, api_key=None),
    ):
        with pytest.raises(ValueError):
            fred_client.fetch_series("FEDFUNDS")


def test_fetch_series_normalizes_observations_and_drops_missing(tmp_path: Path) -> None:
    raw_response = {
        "observations": [
            {"date": "2026-06-01", "value": "5.25"},
            {"date": "2026-07-01", "value": "."},
        ]
    }
    with (
        patch(
            "quantpulse.ingestion.fred_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.fred_client.get_json", return_value=raw_response),
    ):
        df = fred_client.fetch_series("FEDFUNDS")

    assert list(df.columns) == ["date", "indicator_name", "value"]
    assert len(df) == 1
    assert df.iloc[0]["date"] == date(2026, 6, 1)
    assert df.iloc[0]["value"] == 5.25
    assert df.iloc[0]["indicator_name"] == "FEDFUNDS"


def test_named_convenience_wrappers_use_correct_series_id(tmp_path: Path) -> None:
    raw_response = {"observations": [{"date": "2026-07-01", "value": "3.5"}]}
    with (
        patch(
            "quantpulse.ingestion.fred_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.fred_client.get_json", return_value=raw_response
        ) as mock_get_json,
    ):
        fred_client.fetch_cpi()

    _, kwargs = mock_get_json.call_args
    assert kwargs["params"]["series_id"] == "CPIAUCSL"
