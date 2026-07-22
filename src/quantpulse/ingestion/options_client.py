"""Options-positioning signal — yfinance options chain (Section 24).

Two pieces:

1. **`fetch_options_signals`**: today's put/call ratio (volume-based, the
   conventional CBOE-style definition — elevated means more put volume than
   call volume, read as more hedging/bearish positioning) and an
   at-the-money implied-volatility snapshot, both computed from the nearest
   expiration at least `min_days_out` days away (default a week) rather than
   a front-week/0DTE contract, whose volume and IV are dominated by
   near-term noise the plan isn't asking for.

2. **`compute_iv_rank`**: Section 13's `options_signals.iv_rank` is "today's
   IV relative to its own trailing range" — a percentile that needs a
   *history* of daily IV snapshots to rank against. A single API call
   fundamentally cannot produce that on day one, so this module separates
   the two concerns cleanly: `fetch_options_signals` supplies the raw daily
   snapshot (an ingestion concern), and `compute_iv_rank` is a pure function
   over whatever history has already accumulated (several weeks of stored
   snapshots, once something persists `options_signals` over time) — the
   same disclosed cold-start limitation this project already applies
   elsewhere (Section 22), not a silent gap.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_json
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_SOURCE = "yfinance_options"
_DEFAULT_MIN_DAYS_OUT = 7
_ATM_STRIKE_COUNT = 3  # strikes closest to the underlying price, averaged for one side's ATM IV

# Same unofficial-endpoint caution as yfinance_client (Section 5).
_rate_limiter = SimpleRateLimiter(min_interval_seconds=0.5)


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "options" / subdir


def _fetch_expirations(symbol: str) -> tuple[str, ...]:
    """All available option expiration dates for `symbol`, nearest first."""

    def _fetch() -> list[str]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            return list(yf.Ticker(symbol).options)

    return tuple(
        cached_json(
            f"expirations_{symbol}", _fetch, _cache_dir("expirations"), ttl=timedelta(hours=6)
        )
    )


def _select_expiration(symbol: str, *, min_days_out: int) -> str | None:
    today = date.today()
    for expiration in _fetch_expirations(symbol):
        try:
            expiration_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        except ValueError:
            continue
        if (expiration_date - today).days >= min_days_out:
            return expiration
    return None


def _atm_iv(chain_side: pd.DataFrame, underlying_price: float) -> float | None:
    """Average implied volatility of the strikes closest to `underlying_price`."""
    if chain_side.empty or underlying_price <= 0:
        return None
    nearest = chain_side.assign(
        _distance=(chain_side["strike"] - underlying_price).abs()
    ).nsmallest(_ATM_STRIKE_COUNT, "_distance")
    implied_vols = nearest["impliedVolatility"].dropna()
    return float(implied_vols.mean()) if not implied_vols.empty else None


def fetch_options_signals(
    symbol: str, *, min_days_out: int = _DEFAULT_MIN_DAYS_OUT
) -> dict[str, Any]:
    """Put/call ratio + at-the-money implied volatility for `symbol` (Section 24).

    Returns `{"symbol", "expiration", "put_call_ratio", "atm_implied_volatility"}`.
    All three signal fields are `None` if `symbol` has no options chain or no
    expiration at least `min_days_out` days out. `atm_implied_volatility` is
    a raw snapshot, not an IV-rank — see `compute_iv_rank` and the module
    docstring.
    """
    expiration = _select_expiration(symbol, min_days_out=min_days_out)
    if expiration is None:
        return {
            "symbol": symbol,
            "expiration": None,
            "put_call_ratio": None,
            "atm_implied_volatility": None,
        }

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            chain = yf.Ticker(symbol).option_chain(expiration)

        call_volume = float(chain.calls["volume"].fillna(0).sum())
        put_volume = float(chain.puts["volume"].fillna(0).sum())
        put_call_ratio = put_volume / call_volume if call_volume > 0 else None

        underlying_price = float(chain.underlying.get("regularMarketPrice") or 0.0)
        side_ivs = [
            iv
            for iv in (
                _atm_iv(chain.calls, underlying_price),
                _atm_iv(chain.puts, underlying_price),
            )
            if iv is not None
        ]
        atm_implied_volatility = sum(side_ivs) / len(side_ivs) if side_ivs else None

        return {
            "symbol": symbol,
            "expiration": expiration,
            "put_call_ratio": put_call_ratio,
            "atm_implied_volatility": atm_implied_volatility,
        }

    return dict(
        cached_json(
            f"signals_{symbol}_{expiration}", _fetch, _cache_dir("signals"), ttl=timedelta(hours=6)
        )
    )


def compute_iv_rank(current_iv: float, historical_ivs: list[float]) -> float | None:
    """Percentile rank (0-100) of `current_iv` within `historical_ivs`.

    This is the actual Section-24/13 "IV-rank" computation — but it's a pure
    function over *already-accumulated* history, not something a single API
    call can produce (module docstring). Returns `None` for empty
    `historical_ivs` rather than a misleading default like 0 or 50.
    """
    if not historical_ivs:
        return None
    at_or_below = sum(1 for iv in historical_ivs if iv <= current_iv)
    return 100.0 * at_or_below / len(historical_ivs)
