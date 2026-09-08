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


def test_fetch_price_history_returns_an_empty_frame_for_a_symbol_with_no_data(
    tmp_path: Path,
) -> None:
    # yfinance signals "delisted / throttled / 404" with an EMPTY frame whose
    # index is a plain Index, not a DatetimeIndex -- so `.tz_localize(None)`
    # raised AttributeError. The cold-start backfill walks every symbol that was
    # ever in the index, 756 of which are delisted, so "no history" has to be a
    # clean empty answer rather than an exception per symbol.
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with (
        patch(
            "quantpulse.ingestion.yfinance_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=mock_ticker),
    ):
        df = yfinance_client.fetch_price_history("ZZZZ", period="max")

    assert df.empty
    # Shape-identical to a populated result, so every consumer behaves the same.
    assert list(df.columns) == list(yfinance_client._PRICE_COLUMNS)


def test_fetch_price_history_empty_frame_columns_match_the_populated_path(
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.5],
            "Volume": [1000],
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
        populated = yfinance_client.fetch_price_history("AAPL", period="5d")

    assert list(populated.columns) == list(yfinance_client._PRICE_COLUMNS)


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


def test_fetch_price_history_drops_non_positive_adjusted_closes(tmp_path: Path) -> None:
    # Free sources emit adj_close = 0 for some delisted names (DEC ships 732
    # such bars while its raw close is $1.44). One of them turns the
    # equal-weight benchmark into `inf`, so they never leave ingestion.
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [2.0, 2.0, 2.0],
            "Low": [0.5, 0.5, 0.5],
            "Close": [1.44, 1.44, 1.44],
            "Adj Close": [0.0, 1.44, -1.0],
            "Volume": [100, 100, 100],
        },
        index=pd.DatetimeIndex(
            ["2026-07-20", "2026-07-21", "2026-07-22"], name="Date", tz="America/New_York"
        ),
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
        df = yfinance_client.fetch_price_history("DEC", period="max")

    assert len(df) == 1
    assert df.iloc[0]["adj_close"] == 1.44


def test_fetch_price_history_keeps_zero_volume_bars(tmp_path: Path) -> None:
    # Zero VOLUME is a legitimate untraded day, unlike a zero adjusted close --
    # the guard above must not quietly discard those.
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.5],
            "Volume": [0],
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
        df = yfinance_client.fetch_price_history("QUIET", period="5d")

    assert len(df) == 1


class TestFetchDividends:
    """Cash dividends per share, by ex-date (Section 13's dividend tracking).

    yfinance returns a `Series` here rather than a frame, indexed by a
    timezone-aware `DatetimeIndex`, and returns an *empty* one for the many
    names that have never paid — which is a normal answer, not a failure.
    """

    def _patched(self, dividends: pd.Series):
        ticker = MagicMock()
        ticker.dividends = dividends
        return patch("quantpulse.ingestion.yfinance_client.yf.Ticker", return_value=ticker)

    def test_it_normalizes_to_the_stored_row_shape(self, tmp_path) -> None:
        series = pd.Series(
            [0.24, 0.25],
            index=pd.DatetimeIndex(["2026-02-06", "2026-05-08"], tz="America/New_York"),
        )
        with patch.object(yfinance_client, "_cache_dir", return_value=tmp_path):
            with self._patched(series):
                frame = yfinance_client.fetch_dividends("AAPL")

        assert list(frame.columns) == ["symbol", "ex_date", "amount"]
        assert list(frame["symbol"]) == ["AAPL", "AAPL"]
        assert list(frame["amount"]) == [0.24, 0.25]

    def test_the_zone_is_dropped_rather_than_converted(self, tmp_path) -> None:
        """An ex-date is a fact about a trading day on an exchange, so the
        wall-clock date in the exchange's timezone is the answer.

        The timestamp here is deliberately late in the day. At midnight -- how
        yfinance actually stamps these -- dropping the zone and converting to
        UTC agree, so a midnight fixture passes with either and proves nothing.
        Past 19:00 New York they diverge, and converting would push the ex-date
        onto the following calendar day, paying the dividend to holders who
        bought the morning after.
        """
        series = pd.Series(
            [0.24],
            index=pd.DatetimeIndex(["2026-02-06 20:30:00"], tz="America/New_York"),
        )
        with patch.object(yfinance_client, "_cache_dir", return_value=tmp_path):
            with self._patched(series):
                frame = yfinance_client.fetch_dividends("AAPL")
        assert frame["ex_date"].iloc[0] == pd.Timestamp("2026-02-06")

    def test_a_name_that_has_never_paid_is_an_empty_frame(self, tmp_path) -> None:
        """Most of the index. An exception here would make "pays no dividend"
        indistinguishable from "the fetch broke"."""
        with patch.object(yfinance_client, "_cache_dir", return_value=tmp_path):
            with self._patched(pd.Series(dtype=float)):
                frame = yfinance_client.fetch_dividends("GOOGL")
        assert frame.empty
        assert list(frame.columns) == ["symbol", "ex_date", "amount"]

    def test_a_non_datetime_index_is_treated_as_no_data(self, tmp_path) -> None:
        """yfinance signals a throttled or 404 response with an empty frame
        carrying a plain `Index` — the same shape that once made
        `fetch_price_history` raise `AttributeError` on `tz_localize`."""
        with patch.object(yfinance_client, "_cache_dir", return_value=tmp_path):
            with self._patched(pd.Series([0.1], index=pd.Index(["nonsense"]))):
                frame = yfinance_client.fetch_dividends("AAPL")
        assert frame.empty

    def test_non_positive_amounts_are_dropped(self, tmp_path) -> None:
        """A zero or negative "dividend" is not one, and it would show up as a
        pay date that paid nothing."""
        series = pd.Series(
            [0.24, 0.0, -0.1],
            index=pd.DatetimeIndex(["2026-02-06", "2026-05-08", "2026-08-07"]),
        )
        with patch.object(yfinance_client, "_cache_dir", return_value=tmp_path):
            with self._patched(series):
                frame = yfinance_client.fetch_dividends("AAPL")
        assert list(frame["amount"]) == [0.24]
