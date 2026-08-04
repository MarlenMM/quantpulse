from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantpulse.ingestion import options_client


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(options_client._rate_limiter, "wait", lambda: None)


def _fake_settings(tmp_path: Path) -> Mock:
    settings = Mock()
    settings.ingestion_cache_dir = str(tmp_path)
    return settings


def _fake_chain(
    call_volumes: list[float],
    call_ivs: list[float],
    call_strikes: list[float],
    put_volumes: list[float],
    put_ivs: list[float],
    put_strikes: list[float],
    underlying_price: float,
) -> Mock:
    chain = Mock()
    chain.calls = pd.DataFrame(
        {"strike": call_strikes, "volume": call_volumes, "impliedVolatility": call_ivs}
    )
    chain.puts = pd.DataFrame(
        {"strike": put_strikes, "volume": put_volumes, "impliedVolatility": put_ivs}
    )
    chain.underlying = {"regularMarketPrice": underlying_price}
    return chain


# --- expiration selection ------------------------------------------------------


def test_select_expiration_skips_near_dated_contracts(tmp_path: Path) -> None:
    today = date(2026, 7, 22)
    near = (today + timedelta(days=2)).isoformat()
    far_enough = (today + timedelta(days=10)).isoformat()
    with (
        patch(
            "quantpulse.ingestion.options_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch("quantpulse.ingestion.options_client.date") as mock_date,
    ):
        mock_date.today.return_value = today
        with patch.object(options_client, "_fetch_expirations", return_value=(near, far_enough)):
            selected = options_client._select_expiration("AAPL", min_days_out=7)
    assert selected == far_enough


def test_select_expiration_returns_none_when_all_too_near(tmp_path: Path) -> None:
    today = date(2026, 7, 22)
    near = (today + timedelta(days=1)).isoformat()
    with patch("quantpulse.ingestion.options_client.date") as mock_date:
        mock_date.today.return_value = today
        with patch.object(options_client, "_fetch_expirations", return_value=(near,)):
            selected = options_client._select_expiration("AAPL", min_days_out=7)
    assert selected is None


def test_select_expiration_skips_unparseable_dates() -> None:
    today = date(2026, 7, 22)
    far_enough = (today + timedelta(days=10)).isoformat()
    with patch("quantpulse.ingestion.options_client.date") as mock_date:
        mock_date.today.return_value = today
        with patch.object(
            options_client, "_fetch_expirations", return_value=("not-a-date", far_enough)
        ):
            selected = options_client._select_expiration("AAPL", min_days_out=7)
    assert selected == far_enough


# --- _atm_iv --------------------------------------------------------------------


def test_atm_iv_averages_closest_strikes() -> None:
    df = pd.DataFrame(
        {"strike": [90, 100, 110, 200], "impliedVolatility": [0.20, 0.30, 0.40, 0.99]}
    )
    # underlying at 100 -> closest 3 strikes are 90, 100, 110 -> mean(0.20,0.30,0.40)=0.30
    assert options_client._atm_iv(df, 100.0) == pytest.approx(0.30)


def test_atm_iv_returns_none_for_empty_chain_side() -> None:
    df = pd.DataFrame({"strike": [], "impliedVolatility": []})
    assert options_client._atm_iv(df, 100.0) is None


def test_atm_iv_returns_none_for_non_positive_price() -> None:
    df = pd.DataFrame({"strike": [100], "impliedVolatility": [0.3]})
    assert options_client._atm_iv(df, 0.0) is None


# --- fetch_options_signals ------------------------------------------------------


def test_fetch_options_signals_computes_put_call_ratio_and_atm_iv(tmp_path: Path) -> None:
    chain = _fake_chain(
        call_volumes=[100, 50],
        call_ivs=[0.25, 0.30],
        call_strikes=[95, 105],
        put_volumes=[40, 60],
        put_ivs=[0.35, 0.45],
        put_strikes=[95, 105],
        underlying_price=100.0,
    )
    with (
        patch(
            "quantpulse.ingestion.options_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch.object(options_client, "_select_expiration", return_value="2026-08-15"),
        patch("quantpulse.ingestion.options_client.yf.Ticker") as mock_ticker_cls,
    ):
        mock_ticker_cls.return_value.option_chain.return_value = chain
        result = options_client.fetch_options_signals("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["expiration"] == "2026-08-15"
    # put volume 100, call volume 150 -> ratio = 100/150
    assert result["put_call_ratio"] == pytest.approx(100 / 150)
    assert result["atm_implied_volatility"] is not None


def test_fetch_options_signals_returns_none_fields_when_no_expiration(tmp_path: Path) -> None:
    with (
        patch(
            "quantpulse.ingestion.options_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch.object(options_client, "_select_expiration", return_value=None),
    ):
        result = options_client.fetch_options_signals("ILLIQUID")

    assert result == {
        "symbol": "ILLIQUID",
        "expiration": None,
        "put_call_ratio": None,
        "atm_implied_volatility": None,
    }


def test_fetch_options_signals_handles_zero_call_volume(tmp_path: Path) -> None:
    # Puts clear the volume floor so this exercises the zero-denominator path
    # specifically, not the thin-chain guard below.
    chain = _fake_chain(
        call_volumes=[0, 0],
        call_ivs=[0.25, 0.30],
        call_strikes=[95, 105],
        put_volumes=[30, 25],
        put_ivs=[0.35, 0.45],
        put_strikes=[95, 105],
        underlying_price=100.0,
    )
    with (
        patch(
            "quantpulse.ingestion.options_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch.object(options_client, "_select_expiration", return_value="2026-08-15"),
        patch("quantpulse.ingestion.options_client.yf.Ticker") as mock_ticker_cls,
    ):
        mock_ticker_cls.return_value.option_chain.return_value = chain
        result = options_client.fetch_options_signals("AAPL")

    assert result["put_call_ratio"] is None  # no division by zero


# --- plausibility guards -------------------------------------------------------
#
# These reproduce what the nightly job actually persisted into the committed
# demo database: 96% of stored ATM implied volatilities were below 5% (Apple at
# 0.46%), and put/call ratios ran as high as 141 off a near-empty call side.
# Both are artifacts of quoting outside regular trading hours, and both were
# stored and ranked as if they were market readings.


def _signals_from(tmp_path: Path, chain: Mock) -> dict:
    with (
        patch(
            "quantpulse.ingestion.options_client.get_settings",
            return_value=_fake_settings(tmp_path),
        ),
        patch.object(options_client, "_select_expiration", return_value="2026-08-15"),
        patch("quantpulse.ingestion.options_client.yf.Ticker") as mock_ticker_cls,
    ):
        mock_ticker_cls.return_value.option_chain.return_value = chain
        return options_client.fetch_options_signals("AAPL")


def test_implausibly_low_atm_iv_is_abstained_not_reported(tmp_path: Path) -> None:
    # The exact shape of the real bad data: a stale overnight chain whose
    # derived IV collapses toward zero. 0.46% annualized is not a market.
    result = _signals_from(
        tmp_path,
        _fake_chain(
            call_volumes=[500, 400],
            call_ivs=[0.0046, 0.0047],
            call_strikes=[95, 105],
            put_volumes=[300, 200],
            put_ivs=[0.0045, 0.0046],
            put_strikes=[95, 105],
            underlying_price=100.0,
        ),
    )
    assert result["atm_implied_volatility"] is None
    # The put/call ratio is independently fine and must survive the IV guard.
    assert result["put_call_ratio"] == pytest.approx(500 / 900)


def test_implausibly_high_atm_iv_is_abstained(tmp_path: Path) -> None:
    result = _signals_from(
        tmp_path,
        _fake_chain(
            call_volumes=[500, 400],
            call_ivs=[8.0, 9.0],
            call_strikes=[95, 105],
            put_volumes=[300, 200],
            put_ivs=[8.5, 9.5],
            put_strikes=[95, 105],
            underlying_price=100.0,
        ),
    )
    assert result["atm_implied_volatility"] is None


def test_a_genuine_low_volatility_reading_still_passes(tmp_path: Path) -> None:
    # The guard must not swallow real quiet-market readings -- a calm mega-cap
    # at 11% annualized is ordinary, not an artifact.
    result = _signals_from(
        tmp_path,
        _fake_chain(
            call_volumes=[500, 400],
            call_ivs=[0.11, 0.12],
            call_strikes=[95, 105],
            put_volumes=[300, 200],
            put_ivs=[0.11, 0.12],
            put_strikes=[95, 105],
            underlying_price=100.0,
        ),
    )
    assert result["atm_implied_volatility"] == pytest.approx(0.115)


def test_put_call_ratio_from_a_near_empty_chain_is_abstained(tmp_path: Path) -> None:
    # One call contract against 98 puts is arithmetic, not positioning -- this
    # is how the demo DB ended up with a "put/call ratio" of 141.
    result = _signals_from(
        tmp_path,
        _fake_chain(
            call_volumes=[1, 0],
            call_ivs=[0.25, 0.30],
            call_strikes=[95, 105],
            put_volumes=[50, 48],
            put_ivs=[0.35, 0.45],
            put_strikes=[95, 105],
            underlying_price=100.0,
        ),
    )
    assert result["put_call_ratio"] is None
    # IV is unaffected by the volume guard.
    assert result["atm_implied_volatility"] is not None


# --- compute_iv_rank -------------------------------------------------------------


def test_compute_iv_rank_percentile() -> None:
    assert options_client.compute_iv_rank(0.3, [0.1, 0.2, 0.25, 0.35, 0.4]) == 60.0


def test_compute_iv_rank_empty_history_returns_none() -> None:
    assert options_client.compute_iv_rank(0.3, []) is None


def test_compute_iv_rank_all_below_is_100() -> None:
    assert options_client.compute_iv_rank(0.5, [0.1, 0.2, 0.3]) == 100.0


def test_compute_iv_rank_all_above_is_0() -> None:
    assert options_client.compute_iv_rank(0.05, [0.1, 0.2, 0.3]) == 0.0
