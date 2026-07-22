from unittest.mock import patch

from quantpulse.ingestion import short_interest_client


def test_fetch_short_interest_extracts_first_matching_candidate_keys() -> None:
    metrics = {
        "peBasicExclExtraTTM": 30.0,  # unrelated noise field
        "shortInterestSharePercent": 2.5,
        "shortInterestRatio": 3.1,
    }
    with patch.object(
        short_interest_client.finnhub_client, "fetch_basic_financials", return_value=metrics
    ):
        result = short_interest_client.fetch_short_interest("AAPL")

    assert result == {"symbol": "AAPL", "pct_float_short": 2.5, "days_to_cover": 3.1}


def test_fetch_short_interest_falls_back_through_candidate_keys_in_order() -> None:
    metrics = {"shortFloatPercent": 4.0, "daysToCover": 1.8}
    with patch.object(
        short_interest_client.finnhub_client, "fetch_basic_financials", return_value=metrics
    ):
        result = short_interest_client.fetch_short_interest("MSFT")

    assert result == {"symbol": "MSFT", "pct_float_short": 4.0, "days_to_cover": 1.8}


def test_fetch_short_interest_returns_none_when_no_candidate_key_present() -> None:
    metrics = {"peBasicExclExtraTTM": 30.0}
    with patch.object(
        short_interest_client.finnhub_client, "fetch_basic_financials", return_value=metrics
    ):
        result = short_interest_client.fetch_short_interest("ZZZZ")

    assert result == {"symbol": "ZZZZ", "pct_float_short": None, "days_to_cover": None}


def test_fetch_short_interest_ignores_non_numeric_values() -> None:
    metrics = {"shortInterestSharePercent": "N/A", "shortInterestRatio": None}
    with patch.object(
        short_interest_client.finnhub_client, "fetch_basic_financials", return_value=metrics
    ):
        result = short_interest_client.fetch_short_interest("AAPL")

    assert result["pct_float_short"] is None
    assert result["days_to_cover"] is None


def test_first_present_tries_keys_in_order() -> None:
    metrics = {"a": None, "b": 5.0, "c": 6.0}
    assert short_interest_client._first_present(metrics, ("a", "b", "c")) == 5.0


def test_first_present_returns_none_when_all_absent() -> None:
    assert short_interest_client._first_present({}, ("a", "b")) is None
