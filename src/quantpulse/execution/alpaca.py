"""A small, deliberately narrow client for Alpaca's **paper** trading API.

This is the only module in the project that submits an order, so it is written
around one question: what stops this from ever reaching real money?

**The endpoint is not configurable.** There is no setting, no environment
variable and no argument that can point this at `api.alpaca.markets`. The base
URL is a module constant and every request re-validates it, so adding a new
call cannot quietly skip the check. The only other host that can be reached is
loopback -- which exists so the client can be exercised end-to-end against a
local stand-in, and which can never be Alpaca.

That is a stronger guarantee than a `paper=True` flag, and the failure it
prevents is not hypothetical: an env var with `api.` where `paper-api.` was
meant is one keystroke, produces no error, and would trade real money on a
schedule with nobody watching.

The API key and secret travel in headers, never in a URL, and neither is ever
logged or repeated into an exception -- `requests` puts the request URL into
its own error messages, so every failure here is re-raised scrubbed and
unchained, exactly as `alerting.discord` does for a webhook.
"""

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from quantpulse.config import get_settings
from quantpulse.ingestion import http

__all__ = [
    "Account",
    "AlpacaCredentialsError",
    "AlpacaError",
    "NotPaperEndpointError",
    "PAPER_BASE_URL",
    "assert_paper_endpoint",
    "get_account",
    "list_positions",
    "submit_order",
]

logger = logging.getLogger(__name__)

#: Alpaca's paper-trading host. Simulated money, real market data.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"

#: The live-money host, named here only so the guard below can be read against
#: it. Nothing in this project may send it a request.
_LIVE_HOST = "api.alpaca.markets"

_PAPER_HOST = "paper-api.alpaca.markets"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Module-level rather than a parameter, so no caller can pass a different one.
# Tests and the local end-to-end check patch this attribute; the guard still
# runs afterwards and still permits only loopback.
_BASE_URL = PAPER_BASE_URL

_SIDES = frozenset({"buy", "sell"})


class AlpacaError(RuntimeError):
    """A request failed. Never carries the API secret or the request URL."""


class AlpacaCredentialsError(AlpacaError):
    """No API key/secret configured, so nothing was attempted."""


class NotPaperEndpointError(AlpacaError):
    """A request was about to go somewhere other than the paper endpoint."""


@dataclass(frozen=True)
class Account:
    """The paper account's headline numbers. Alpaca sends all of them as strings."""

    status: str
    equity: float
    cash: float
    buying_power: float


def assert_paper_endpoint(url: str) -> None:
    """Raise unless `url` is the paper endpoint (or loopback, for testing).

    Parsed with `urlsplit` and compared on the **host**, not with a substring or
    a suffix test. Every one of these passes a naive check and none of them is
    Alpaca's paper endpoint:

        https://paper-api.alpaca.markets.evil.test/v2/account   (subdomain)
        https://evil.test/paper-api.alpaca.markets/v2/account   (path)
        https://paper-api.alpaca.markets@evil.test/v2/account   (userinfo)

    HTTPS is required for the real host because the credentials ride in a
    header; loopback may be plain HTTP, since a local stand-in has no
    credentials worth protecting and no network to expose them on.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        raise NotPaperEndpointError("refusing a request to an unparseable URL") from None
    host = parts.hostname  # userinfo and port stripped by urlsplit, unlike netloc
    if host in _LOOPBACK_HOSTS and parts.scheme in ("http", "https"):
        return
    if parts.scheme != "https" or host != _PAPER_HOST:
        raise NotPaperEndpointError(
            f"refusing to send a trading request to host {host!r} over {parts.scheme!r}: "
            f"this client may only reach https://{_PAPER_HOST} (paper, simulated money) "
            f"or loopback. It must never reach https://{_LIVE_HOST}, which is real money."
        )


def _headers() -> dict[str, str]:
    settings = get_settings()
    key, secret = settings.alpaca_api_key_id, settings.alpaca_api_secret_key
    if not (key and key.strip()) or not (secret and secret.strip()):
        raise AlpacaCredentialsError(
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are not both set; "
            "paper trading is disabled and nothing was requested"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"}


def _url(path: str) -> str:
    """Build and immediately validate the target. Every call goes through here."""
    url = f"{_BASE_URL}{path}"
    assert_paper_endpoint(url)
    return url


def _scrub(error: Exception) -> str:
    """An error's type and HTTP status, never its message.

    `requests` writes the full request URL into `str(error)`. That URL does not
    carry the secret (it rides in a header), but the same reasoning as
    `alerting.discord` applies to keeping error text built rather than repeated:
    the day a credential does end up in a URL, this is what stops it reaching a
    log, and it costs one function.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    return type(error).__name__ if status is None else f"{type(error).__name__} (HTTP {status})"


def _get(path: str) -> Any:
    url = _url(path)
    try:
        return http.get_json(url, headers=_headers())
    except AlpacaError:
        raise
    except Exception as error:
        raise AlpacaError(f"GET {path} failed: {_scrub(error)}") from None


def _post(path: str, body: dict[str, Any]) -> Any:
    url = _url(path)
    try:
        return http.post_json(url, json_body=body, headers=_headers())
    except AlpacaError:
        raise
    except Exception as error:
        raise AlpacaError(f"POST {path} failed: {_scrub(error)}") from None


def get_account() -> Account:
    """The paper account's status, equity, cash and buying power.

    A non-USD account is refused rather than converted. Every price in this
    database is USD, so sizing an order against an account denominated in
    anything else would silently buy the wrong number of shares -- a wrong
    answer that looks exactly like a right one.
    """
    payload = _get("/v2/account")
    currency = str(payload.get("currency", "USD"))
    if currency != "USD":
        raise AlpacaError(
            f"paper account is denominated in {currency}, not USD; every price in this "
            "database is USD, so order sizing would be silently wrong"
        )
    return Account(
        status=str(payload["status"]),
        equity=float(payload["equity"]),
        cash=float(payload["cash"]),
        buying_power=float(payload["buying_power"]),
    )


def list_positions() -> dict[str, float]:
    """Open positions as `{symbol: quantity}`. Empty when flat, which is not an error."""
    return {str(row["symbol"]): float(row["qty"]) for row in _get("/v2/positions")}


def submit_order(symbol: str, *, qty: int | float, side: str) -> dict[str, Any]:
    """Submit one whole-share market order, good for the day.

    **Whole shares, not notional.** Alpaca's documentation states the fractional
    rule two ways -- that fractional trading extends into pre- and post-market,
    and that fractional shares "can only be bought or sold with market orders
    during normal market hours". This job runs after the close, so relying on a
    rule the vendor states twice and differently would buy an exactness nobody
    needs at the price of a silent no-trade. A real investor buys whole shares.

    `time_in_force="day"` submitted after 16:00 ET is **queued for the next
    trading day** rather than rejected (Alpaca's documented behaviour), so the
    decision is made on one day's closing scores and filled at the next open.
    That one-session lag is real and is recorded with the run rather than
    modelled away.

    `qty` is positive and `side` carries the direction. A signed quantity would
    state the direction twice and the two could disagree.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {sorted(_SIDES)}, got {side!r}")
    if qty != int(qty):
        raise ValueError(f"qty must be a whole number of shares, got {qty!r}")
    if int(qty) <= 0:
        raise ValueError(f"qty must be positive (side carries the direction), got {qty!r}")
    logger.info("Submitting paper order: %s %d %s", side, int(qty), symbol)
    return dict(
        _post(
            "/v2/orders",
            {
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
        )
    )
