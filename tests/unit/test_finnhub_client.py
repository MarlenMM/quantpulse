from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quantpulse.ingestion import finnhub_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(finnhub_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path, api_key: str | None = "test-key") -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    settings.finnhub_api_key = api_key
    return settings


def test_fetch_quote_raises_without_api_key(tmp_path: Path) -> None:
    with patch(
        "quantpulse.ingestion.finnhub_client.get_settings",
        return_value=_fake_settings(tmp_path, api_key=None),
    ):
        with pytest.raises(ValueError):
            finnhub_client.fetch_quote("AAPL")


def test_fetch_quote_returns_parsed_json(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.finnhub_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.finnhub_client.get_json", return_value={"c": 150.0}
        ) as mock_get_json,
    ):
        result = finnhub_client.fetch_quote("AAPL")

    assert result == {"c": 150.0}
    _, kwargs = mock_get_json.call_args
    assert kwargs["params"]["symbol"] == "AAPL"
    assert kwargs["params"]["token"] == "test-key"


def test_fetch_company_news_passes_date_range(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.finnhub_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.finnhub_client.get_json", return_value=[{"headline": "x"}]
        ) as mock_get_json,
    ):
        result = finnhub_client.fetch_company_news("AAPL", date(2026, 7, 1), date(2026, 7, 21))

    assert result == [{"headline": "x"}]
    _, kwargs = mock_get_json.call_args
    assert kwargs["params"]["from"] == "2026-07-01"
    assert kwargs["params"]["to"] == "2026-07-21"


def test_fetch_earnings_calendar_extracts_list(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.finnhub_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.finnhub_client.get_json",
            return_value={"earningsCalendar": [{"symbol": "AAPL"}]},
        ),
    ):
        result = finnhub_client.fetch_earnings_calendar(date(2026, 7, 1), date(2026, 7, 21))

    assert result == [{"symbol": "AAPL"}]


def test_fetch_basic_financials_extracts_metric_dict(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.finnhub_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch(
            "quantpulse.ingestion.finnhub_client.get_json",
            return_value={"metric": {"peRatio": 30.0}},
        ),
    ):
        result = finnhub_client.fetch_basic_financials("AAPL")

    assert result == {"peRatio": 30.0}
