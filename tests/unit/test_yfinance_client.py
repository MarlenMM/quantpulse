from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from quantpulse.ingestion import yfinance_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfinance_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def test_fetch_price_history_normalizes_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.5],
            "Volume": [1000],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        },
        index=pd.DatetimeIndex(["2026-07-20"], name="Date", tz="America/New_York"),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        df = yfinance_client.fetch_price_history("AAPL", period="5d")

    assert list(df.columns) == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert df.iloc[0]["symbol"] == "AAPL"
    mock_ticker.history.assert_called_once_with(period="5d", auto_adjust=False)


def test_fetch_fundamentals_maps_info_fields(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "trailingPE": 30.0,
        "priceToBook": 40.0,
        "priceToSalesTrailing12Months": 10.0,
        "pegRatio": 2.5,
        "trailingEps": 8.0,
        "revenueGrowth": 0.1,
        "debtToEquity": 70.0,
        "returnOnEquity": 1.2,
        "returnOnAssets": 0.2,
        "freeCashflow": 1000000,
        "dividendYield": 0.3,
    }

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_fundamentals("AAPL")

    assert result == {
        "symbol": "AAPL",
        "pe": 30.0,
        "pb": 40.0,
        "ps": 10.0,
        "peg": 2.5,
        "eps": 8.0,
        "revenue_growth": 0.1,
        "debt_equity": 70.0,
        "roe": 1.2,
        "roa": 0.2,
        "fcf": 1000000,
        "div_yield": 0.3,
    }


def test_fetch_ffo_inputs_reads_cashflow_and_market_cap(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.cashflow = pd.DataFrame(
        {"col": [1_000_000_000.0, 2_500_000_000.0]},
        index=["Net Income From Continuing Operations", "Depreciation And Amortization"],
    )
    mock_ticker.info = {"marketCap": 60_000_000_000}

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_ffo_inputs("O")

    assert result == {
        "symbol": "O",
        "net_income": 1_000_000_000.0,
        "depreciation_amortization": 2_500_000_000.0,
        "market_cap": 60_000_000_000,
    }


def test_fetch_ffo_inputs_handles_missing_cashflow_rows(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.cashflow = pd.DataFrame({"col": [1.0]}, index=["Some Other Row"])
    mock_ticker.info = {"marketCap": 100}

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_ffo_inputs("XYZ")

    assert result["net_income"] is None
    assert result["depreciation_amortization"] is None


def test_fetch_ffo_inputs_handles_empty_cashflow(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.cashflow = pd.DataFrame()
    mock_ticker.info = {"marketCap": 100}

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_ffo_inputs("XYZ")

    assert result["net_income"] is None
    assert result["depreciation_amortization"] is None
    assert result["market_cap"] == 100


def test_fetch_analyst_consensus_uses_current_month_row(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.recommendations = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 6, "buy": 23, "hold": 14, "sell": 2, "strongSell": 2},
            {"period": "-1m", "strongBuy": 5, "buy": 20, "hold": 15, "sell": 1, "strongSell": 2},
        ]
    )
    mock_ticker.info = {"targetMeanPrice": 300.0}

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_analyst_consensus("AAPL")

    assert result == {
        "symbol": "AAPL",
        "strong_buy": 6,
        "buy": 23,
        "hold": 14,
        "sell": 2,
        "strong_sell": 2,
        "mean_price_target": 300.0,
    }


def test_fetch_analyst_consensus_handles_missing_recommendations(tmp_path: Path) -> None:
    mock_ticker = MagicMock()
    mock_ticker.recommendations = None
    mock_ticker.info = {"targetMeanPrice": None}

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        result = yfinance_client.fetch_analyst_consensus("AAPL")

    assert result["strong_buy"] == 0
    assert result["mean_price_target"] is None
