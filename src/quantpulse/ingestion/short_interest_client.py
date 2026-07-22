"""Short-interest signal — Finnhub basic-financials fields (Section 24).

Section 24: "Short interest (% of float sold short, days-to-cover). Available
from Finnhub's basic-financials endpoint for many names." This is a thin,
semantically-named wrapper over `finnhub_client.fetch_basic_financials` (same
endpoint, same cache, same rate limiter/circuit breaker) rather than a
separate HTTP client — the point of this module is the normalized short-
interest shape, not a new network path.

VERIFICATION GAP, stated plainly rather than silently assumed correct:
Finnhub's own public API specification declares the `/stock/metric` response's
`metric` field as a bare, undocumented key-value map (checked directly against
Finnhub's published OpenAPI/swagger document — it does not enumerate field
names anywhere), and no `FINNHUB_API_KEY` was available in this environment to
inspect a real response. The candidate key names below are a best-effort,
UNVERIFIED guess at Finnhub's actual short-interest field names, based on
common naming conventions across financial-data vendors — not confirmed
against a live response. Verify against `finnhub_client.fetch_basic_financials
(symbol)` with a real key before relying on this signal, and update
`_PCT_FLOAT_SHORT_KEYS` / `_DAYS_TO_COVER_KEYS` to match whatever the real
field names turn out to be — the same "seed data, confirm before trusting it"
honesty as `ingestion/economic_calendar.py`'s placeholder FOMC/CPI dates.
"""

from typing import Any

from quantpulse.ingestion import finnhub_client

# UNVERIFIED candidates, tried in order -- see module docstring.
_PCT_FLOAT_SHORT_KEYS = (
    "shortInterestSharePercent",
    "shortFloatPercent",
    "shortPercentOfFloat",
    "shortInterest",
)
_DAYS_TO_COVER_KEYS = ("shortInterestRatio", "daysToCover", "shortRatio")


def _first_present(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def fetch_short_interest(symbol: str) -> dict[str, Any]:
    """`symbol`'s short-interest snapshot, normalized toward the `short_interest` table.

    Section 13.

    Returns `{"symbol", "pct_float_short", "days_to_cover"}`. Either numeric
    field may be `None` if Finnhub doesn't cover the name, or if the real
    field names differ from the unverified candidates this tries (module
    docstring) -- a missing value here is not distinguishable from "not
    covered" from this function alone.
    """
    metrics = finnhub_client.fetch_basic_financials(symbol)
    return {
        "symbol": symbol,
        "pct_float_short": _first_present(metrics, _PCT_FLOAT_SHORT_KEYS),
        "days_to_cover": _first_present(metrics, _DAYS_TO_COVER_KEYS),
    }
