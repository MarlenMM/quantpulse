from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from quantpulse.config import get_settings
from quantpulse.ingestion.cache import cached_dataframe, cached_json
from quantpulse.ingestion.circuit_breaker import get_breaker
from quantpulse.ingestion.rate_limit import SimpleRateLimiter

_SOURCE = "yfinance"

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
        with get_breaker(_SOURCE).guard():
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

    # `period` must be in the cache key: a 5d nightly pull and a max/10y seed
    # backfill for the same symbol are different data and must not collide.
    return cached_dataframe(
        f"price_history_{symbol}_{period}",
        _fetch,
        _cache_dir("price_history"),
        ttl=timedelta(hours=12),
    )


def fetch_fundamentals(symbol: str) -> dict[str, Any]:
    """Sector-agnostic fundamental ratios for `symbol`, normalized to `fundamentals_snapshot`.

    Sector-specific substitutes (FFO for REITs, etc. -- Section 7.2) are added
    in Phase 3, on top of this common set.
    """

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            info = yf.Ticker(symbol).info
        return {"symbol": symbol, **{k: info.get(v) for k, v in _FUNDAMENTAL_FIELDS.items()}}

    return cached_json(
        f"fundamentals_{symbol}", _fetch, _cache_dir("fundamentals"), ttl=timedelta(days=7)
    )


def fetch_ffo_inputs(symbol: str) -> dict[str, Any]:
    """Net income, D&A, and market cap -- the inputs to an FFO proxy for REITs (Section 7.2).

    FFO = Net Income + Depreciation & Amortization is the standard, simplified
    NAREIT approximation; the full definition also excludes gains/losses on
    property sales, which needs additional line-item data kept out of scope
    here. Only meaningful for the Real Estate sector's P/FFO substitution --
    other sectors' scoring never calls this.
    """

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            ticker = yf.Ticker(symbol)
            cashflow = ticker.cashflow
            market_cap = ticker.info.get("marketCap")

        net_income = None
        depreciation_amortization = None
        if cashflow is not None and not cashflow.empty:
            if "Net Income From Continuing Operations" in cashflow.index:
                net_income = float(cashflow.loc["Net Income From Continuing Operations"].iloc[0])
            if "Depreciation And Amortization" in cashflow.index:
                depreciation_amortization = float(
                    cashflow.loc["Depreciation And Amortization"].iloc[0]
                )

        return {
            "symbol": symbol,
            "net_income": net_income,
            "depreciation_amortization": depreciation_amortization,
            "market_cap": market_cap,
        }

    return cached_json(
        f"ffo_inputs_{symbol}", _fetch, _cache_dir("ffo_inputs"), ttl=timedelta(days=7)
    )


def fetch_analyst_consensus(symbol: str) -> dict[str, Any]:
    """Current-month analyst rating counts + mean price target for `symbol`."""

    def _fetch() -> dict[str, Any]:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            ticker = yf.Ticker(symbol)
            recs = ticker.recommendations
            info = ticker.info
        current = None
        if recs is not None and not recs.empty:
            this_month = recs[recs["period"] == "0m"]
            if not this_month.empty:
                current = this_month.iloc[0]
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
