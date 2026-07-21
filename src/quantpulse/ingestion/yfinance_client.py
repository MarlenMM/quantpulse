from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe, cached_json
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

# yfinance is an unofficial wrapper with no documented rate limit, but Section 5
# says it "can break/rate-limit without notice" -- self-throttle rather than hammer it.
_rate_limiter = SimpleRateLimiter(min_interval_seconds=0.5)

_FUNDAMENTAL_FIELDS = {
    "pe": "trailingPE",
    "pb": "priceToBook",
    "ps": "priceToSalesTrailing12Months",
    "peg": "pegRatio",
    "eps": "trailingEps",
    "revenue_growth": "revenueGrowth",
    "debt_equity": "debtToEquity",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "fcf": "freeCashflow",
    "div_yield": "dividendYield",
}


def _cache_dir(subdir: str) -> Path:
    return Path(get_settings().ingestion_cache_dir) / "yfinance" / subdir


def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """OHLCV + adjusted close for `symbol`, normalized to the `price_history` schema."""

    def _fetch() -> pd.DataFrame:
        _rate_limiter.wait()
        raw = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        df = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "adj_close", "volume"]].copy()
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        df.insert(0, "symbol", symbol)
        return df.reset_index()

    return cached_dataframe(
        f"price_history_{symbol}", _fetch, _cache_dir("price_history"), ttl=timedelta(hours=12)
    )


def fetch_fundamentals(symbol: str) -> dict[str, Any]:
    """Sector-agnostic fundamental ratios for `symbol`, normalized to `fundamentals_snapshot`.

    Sector-specific substitutes (FFO for REITs, etc. -- Section 7.2) are added
    in Phase 3, on top of this common set.
    """

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        info = yf.Ticker(symbol).info
        return {"symbol": symbol, **{k: info.get(v) for k, v in _FUNDAMENTAL_FIELDS.items()}}

    return cached_json(
        f"fundamentals_{symbol}", _fetch, _cache_dir("fundamentals"), ttl=timedelta(days=7)
    )


def fetch_analyst_consensus(symbol: str) -> dict[str, Any]:
    """Current-month analyst rating counts + mean price target for `symbol`."""

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        ticker = yf.Ticker(symbol)
        recs = ticker.recommendations
        current = None
        if recs is not None and not recs.empty:
            this_month = recs[recs["period"] == "0m"]
            if not this_month.empty:
                current = this_month.iloc[0]
        info = ticker.info
        return {
            "symbol": symbol,
            "strong_buy": int(current["strongBuy"]) if current is not None else 0,
            "buy": int(current["buy"]) if current is not None else 0,
            "hold": int(current["hold"]) if current is not None else 0,
            "sell": int(current["sell"]) if current is not None else 0,
            "strong_sell": int(current["strongSell"]) if current is not None else 0,
            "mean_price_target": info.get("targetMeanPrice"),
        }

    return cached_json(
        f"analyst_consensus_{symbol}",
        _fetch,
        _cache_dir("analyst_consensus"),
        ttl=timedelta(days=7),
    )
