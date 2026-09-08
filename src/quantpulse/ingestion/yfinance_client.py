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


# Exactly the columns, in the order, the populated path below emits, so an
# empty result is shape-identical to a real one for every downstream consumer.
_PRICE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "adj_close", "volume")


def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """OHLCV + adjusted close for `symbol`, normalized to the `price_history` schema.

    Returns an **empty, correctly-columned** frame when the symbol has no data
    for `period` -- a delisted ticker, or a throttled/404 response. yfinance
    signals that with an empty frame carrying a plain `Index` rather than a
    `DatetimeIndex`, so the timezone normalization below used to raise
    `AttributeError: 'Index' object has no attribute 'tz_localize'`. That
    matters most exactly where it hurts: the cold-start backfill walks every
    symbol that was EVER in the index, 756 of which are delisted, so the
    fast path for "this name has no history" must be a clean empty answer,
    not an exception traceback per symbol.
    """

    def _fetch() -> pd.DataFrame:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            raw = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if raw.empty or not isinstance(raw.index, pd.DatetimeIndex):
            return pd.DataFrame({column: pd.Series(dtype="object") for column in _PRICE_COLUMNS})
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
        # Drop bars missing any OHLCV field at the source. Every one of these
        # columns is NOT NULL in `price_history`, and yfinance emits partial
        # bars for halted sessions and around listing changes -- deep history
        # for a long-dead ticker is full of them. Both writers (the nightly and
        # the cold-start seed) keep their own guard as well, but doing it here
        # means a bar that cannot be stored never leaves the ingestion layer,
        # so a third consumer can't reintroduce the same crash.
        clean = df.dropna(subset=list(_PRICE_COLUMNS[2:]))
        # A non-positive adjusted close is not a price -- free sources emit
        # zeros for some delisted names (DEC ships 732 such bars while its raw
        # close is $1.44). One of them turns the equal-weight benchmark into
        # `inf`, so drop them at the source instead of letting each consumer
        # discover it. `volume` is legitimately 0 on an untraded day and is
        # deliberately not screened here.
        return clean[clean["adj_close"] > 0].reset_index()

    # `period` must be in the cache key: a 5d nightly pull and a max/10y seed
    # backfill for the same symbol are different data and must not collide.
    return cached_dataframe(
        f"price_history_{symbol}_{period}",
        _fetch,
        _cache_dir("price_history"),
        ttl=timedelta(hours=12),
    )


_DIVIDEND_COLUMNS = ("symbol", "ex_date", "amount")


def fetch_dividends(symbol: str) -> pd.DataFrame:
    """Cash dividends per share by ex-date, normalized to the `dividends` schema.

    yfinance answers with a `Series` here rather than a frame, and answers with
    an **empty** one for most of the index -- a name that has never paid a
    dividend is the common case, not a failure, so that comes back as a clean
    empty frame with the right columns.

    The index is timezone-aware, and the zone is *dropped* rather than
    converted. An ex-date is a fact about a trading day on an exchange, so the
    wall-clock date in the exchange's own timezone is the answer; converting to
    UTC first pushes any timestamp later than 19:00 New York onto the following
    calendar day, and entitlement to a dividend is decided by exactly which date
    this is. yfinance stamps these at midnight today, where the two agree --
    which is precisely why it needs a test rather than a glance.

    Guards against the same shape that once made `fetch_price_history` raise:
    a throttled or 404 response arrives as an empty object carrying a plain
    `Index`, which has no `tz_localize`.
    """

    def _fetch() -> pd.DataFrame:
        _rate_limiter.wait()
        with get_breaker(_SOURCE).guard():
            raw = yf.Ticker(symbol).dividends
        empty = pd.DataFrame({column: pd.Series(dtype="object") for column in _DIVIDEND_COLUMNS})
        if raw is None or len(raw) == 0 or not isinstance(raw.index, pd.DatetimeIndex):
            return empty
        index = raw.index
        if index.tz is not None:
            index = index.tz_localize(None)
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "ex_date": index.normalize(),
                "amount": pd.to_numeric(raw.to_numpy(), errors="coerce"),
            }
        )
        # A zero or negative "dividend" is not one; it would render as a pay
        # date that paid nothing.
        return frame[frame["amount"] > 0].reset_index(drop=True)

    return cached_dataframe(
        f"dividends_{symbol}",
        _fetch,
        _cache_dir("dividends"),
        ttl=timedelta(days=7),
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
