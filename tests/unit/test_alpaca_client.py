"""The paper-trading client, and the one guard that matters more than the rest.

This is the only module in the project that can move money-shaped things. It is
pointed at Alpaca's *paper* endpoint, where the money is simulated -- and the
failure mode worth engineering against is not a bug in the strategy, it is a
single mistyped host turning a simulation into a live-money bot.

So the endpoint is not configurable. There is no setting, no environment
variable and no argument that can point this at `api.alpaca.markets`; the base
URL is a module constant, and every request re-checks it. The only other host
that can ever be reached is loopback, which exists so these tests and the local
end-to-end check can run, and which can never be Alpaca.
"""

import logging
from unittest.mock import patch

import pytest

from quantpulse.config import Settings
from quantpulse.execution import alpaca

_KEY = "PKTESTKEYID000000000"
_SECRET = "s3cr3t-alpaca-signing-key-must-not-leak"


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        **{
            "alpaca_api_key_id": _KEY,
            "alpaca_api_secret_key": _SECRET,
            **overrides,
        },
    )


# --------------------------------------------------------------------------- #
# The paper-only guard
# --------------------------------------------------------------------------- #


class TestPaperEndpointGuard:
    def test_the_paper_host_is_allowed(self) -> None:
        alpaca.assert_paper_endpoint("https://paper-api.alpaca.markets/v2/account")

    def test_the_live_trading_host_is_refused(self) -> None:
        """The whole point. `api.alpaca.markets` is real money."""
        with pytest.raises(alpaca.NotPaperEndpointError):
            alpaca.assert_paper_endpoint("https://api.alpaca.markets/v2/account")

    def test_a_lookalike_host_is_refused(self) -> None:
        """A suffix check would pass every one of these."""
        for url in (
            "https://paper-api.alpaca.markets.evil.test/v2/account",
            "https://evil.test/paper-api.alpaca.markets/v2/account",
            "https://notpaper-api.alpaca.markets/v2/account",
            "https://paper-api.alpaca.markets@evil.test/v2/account",
        ):
            with pytest.raises(alpaca.NotPaperEndpointError):
                alpaca.assert_paper_endpoint(url)

    def test_plain_http_to_the_paper_host_is_refused(self) -> None:
        """Credentials in a header over cleartext."""
        with pytest.raises(alpaca.NotPaperEndpointError):
            alpaca.assert_paper_endpoint("http://paper-api.alpaca.markets/v2/account")

    def test_loopback_is_allowed_so_this_can_be_tested_end_to_end(self) -> None:
        alpaca.assert_paper_endpoint("http://127.0.0.1:8931/v2/account")
        alpaca.assert_paper_endpoint("http://localhost:8931/v2/account")

    def test_a_non_loopback_private_address_is_refused(self) -> None:
        """ "Not the internet" is not the same claim as "not Alpaca live"."""
        with pytest.raises(alpaca.NotPaperEndpointError):
            alpaca.assert_paper_endpoint("http://10.0.0.5:8931/v2/account")

    def test_garbage_is_refused_rather_than_passed_through(self) -> None:
        for url in ("", "not a url", "ftp://paper-api.alpaca.markets/v2"):
            with pytest.raises(alpaca.NotPaperEndpointError):
                alpaca.assert_paper_endpoint(url)

    def test_the_module_constant_is_the_paper_host(self) -> None:
        assert alpaca.PAPER_BASE_URL == "https://paper-api.alpaca.markets"
        alpaca.assert_paper_endpoint(alpaca.PAPER_BASE_URL + "/v2/account")


class TestEveryCallIsGuarded:
    """The guard is worthless if one entry point skips it.

    Asserted by pointing the module's base URL at the live host and requiring
    every public call to refuse -- rather than by reading the source and
    trusting that each one remembered.
    """

    @pytest.mark.parametrize(
        "call",
        [
            lambda: alpaca.get_account(),
            lambda: alpaca.list_positions(),
            lambda: alpaca.submit_order("AAPL", qty=1, side="buy"),
        ],
        ids=["get_account", "list_positions", "submit_order"],
    )
    def test_no_entry_point_can_reach_the_live_host(self, call) -> None:
        with patch.object(alpaca, "_BASE_URL", "https://api.alpaca.markets"):
            with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
                with patch("quantpulse.execution.alpaca.http") as http:
                    with pytest.raises(alpaca.NotPaperEndpointError):
                        call()
        http.get_json.assert_not_called()
        http.post_json.assert_not_called()


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


class TestCredentials:
    def test_missing_credentials_raise_before_any_request(self) -> None:
        with patch(
            "quantpulse.execution.alpaca.get_settings",
            return_value=_settings(alpaca_api_key_id=None, alpaca_api_secret_key=None),
        ):
            with patch("quantpulse.execution.alpaca.http") as http:
                with pytest.raises(alpaca.AlpacaCredentialsError):
                    alpaca.get_account()
        http.get_json.assert_not_called()

    def test_a_key_without_its_secret_is_not_usable(self) -> None:
        with patch(
            "quantpulse.execution.alpaca.get_settings",
            return_value=_settings(alpaca_api_secret_key=None),
        ):
            with pytest.raises(alpaca.AlpacaCredentialsError):
                alpaca.get_account()

    def test_credentials_travel_in_headers_not_in_the_url(self) -> None:
        """A query-string credential lands in every proxy and server log."""
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch(
                "quantpulse.execution.alpaca.http.get_json", return_value=_ACCOUNT
            ) as get_json:
                alpaca.get_account()
        url = get_json.call_args.args[0]
        assert _SECRET not in url and _KEY not in url
        headers = get_json.call_args.kwargs["headers"]
        assert headers["APCA-API-KEY-ID"] == _KEY
        assert headers["APCA-API-SECRET-KEY"] == _SECRET

    def test_the_secret_is_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
                with patch("quantpulse.execution.alpaca.http.get_json", return_value=_ACCOUNT):
                    alpaca.get_account()
        assert _SECRET not in caplog.text

    def test_a_failed_request_does_not_carry_the_secret(self) -> None:
        import requests

        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch(
                "quantpulse.execution.alpaca.http.get_json",
                side_effect=requests.HTTPError(f"401 for url with {_SECRET}"),
            ):
                with pytest.raises(alpaca.AlpacaError) as caught:
                    alpaca.get_account()
        assert _SECRET not in str(caught.value)
        assert caught.value.__cause__ is None


# --------------------------------------------------------------------------- #
# Reading the account
# --------------------------------------------------------------------------- #

_ACCOUNT = {
    "status": "ACTIVE",
    "equity": "100123.45",
    "cash": "25000.00",
    "buying_power": "50000.00",
    "currency": "USD",
}


class TestAccount:
    def test_account_numbers_are_parsed_from_alpacas_strings(self) -> None:
        """Alpaca returns every number as a JSON string."""
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch("quantpulse.execution.alpaca.http.get_json", return_value=_ACCOUNT):
                account = alpaca.get_account()
        assert account.equity == pytest.approx(100123.45)
        assert account.cash == pytest.approx(25000.00)
        assert account.status == "ACTIVE"

    def test_a_non_usd_account_is_refused(self) -> None:
        """Every price in this database is USD; sizing against a EUR account
        would silently buy the wrong number of shares."""
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch(
                "quantpulse.execution.alpaca.http.get_json",
                return_value={**_ACCOUNT, "currency": "EUR"},
            ):
                with pytest.raises(alpaca.AlpacaError, match="USD"):
                    alpaca.get_account()

    def test_positions_come_back_as_symbol_to_quantity(self) -> None:
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch(
                "quantpulse.execution.alpaca.http.get_json",
                return_value=[
                    {"symbol": "AAPL", "qty": "10"},
                    {"symbol": "MSFT", "qty": "-3"},
                ],
            ):
                positions = alpaca.list_positions()
        assert positions == {"AAPL": 10.0, "MSFT": -3.0}

    def test_no_positions_is_an_empty_mapping_not_an_error(self) -> None:
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch("quantpulse.execution.alpaca.http.get_json", return_value=[]):
                assert alpaca.list_positions() == {}


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


class TestSubmitOrder:
    def _submit(self, **kwargs):
        with patch("quantpulse.execution.alpaca.get_settings", return_value=_settings()):
            with patch(
                "quantpulse.execution.alpaca.http.post_json", return_value={"id": "abc"}
            ) as post:
                alpaca.submit_order(**kwargs)
        return post.call_args.kwargs["json_body"]

    def test_orders_are_whole_share_market_day_orders(self) -> None:
        """Whole shares, not notional.

        Alpaca's own documentation states the fractional rule two ways -- that
        fractional trading extends into pre/post-market, and that fractional
        shares "can only be bought or sold with market orders during normal
        market hours". Relying on a rule the vendor states twice, differently,
        in a job that runs after the close is a bad trade for an exactness
        nobody needs; a real investor buys whole shares anyway.
        """
        body = self._submit(symbol="AAPL", qty=7, side="buy")
        assert body == {
            "symbol": "AAPL",
            "qty": "7",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }

    def test_a_fractional_quantity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="whole"):
            alpaca.submit_order("AAPL", qty=1.5, side="buy")

    def test_a_zero_or_negative_quantity_is_refused(self) -> None:
        """Side carries the direction; a signed quantity would mean it twice,
        and the two could disagree."""
        for qty in (0, -3):
            with pytest.raises(ValueError, match="positive"):
                alpaca.submit_order("AAPL", qty=qty, side="buy")

    def test_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="side"):
            alpaca.submit_order("AAPL", qty=1, side="hodl")
