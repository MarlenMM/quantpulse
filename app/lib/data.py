"""Cached database access for the Streamlit pages (Sections 12, 29).

Section 29 calls out `@st.cache_data`/`@st.cache_resource` specifically:
Streamlit re-runs the whole script on every interaction, so without caching,
every slider drag would re-query the database. That caching belongs *here*, in
the presentation layer, and is deliberately distinct from the ingestion-layer
response cache of Section 6.

Every function is a thin wrapper: open a session, call one
`storage.persistence` reader, return. No SQL lives in `app/` (Section 14), and
because the readers are plain functions over a `Session`, they stay testable
without Streamlit even though the wrappers here are not.

`TTL_SECONDS` is short enough that data landing mid-session shows up without a
restart, and long enough that clicking around a page doesn't re-query on every
rerun. It is a backstop rather than the main path: a refresh started from the
Settings page clears these caches the moment it finishes (`lib/refresh.py`), so
the wait only applies to rows that arrived some other way. `st.cache_data` also
keys on arguments, so the per-symbol readers cache per symbol automatically.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from quantpulse.analysis import risk
from quantpulse.config import get_settings
from quantpulse.storage import persistence
from quantpulse.storage.db import get_session

logger = logging.getLogger(__name__)

TTL_SECONDS = 300


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def screener_rows(profile: str = "balanced") -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_screener_rows(session, profile=profile)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def rating_changes(profile: str = "balanced", limit: int = 25) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_rating_changes(session, profile=profile, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def ohlcv(symbol: str, lookback_days: int = 400) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_symbol_ohlcv(session, symbol, lookback_days=lookback_days)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def forecasts(symbol: str) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_symbol_forecasts(session, symbol)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def patterns(symbol: str, lookback_days: int = 120) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_symbol_patterns(session, symbol, lookback_days=lookback_days)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def analyst_consensus(symbol: str) -> dict[str, Any] | None:
    with get_session() as session:
        return persistence.read_latest_analyst_consensus(session, symbol)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def short_interest(symbol: str) -> dict[str, Any] | None:
    """One symbol's latest short-interest snapshot, or None if none is stored.

    Section 24 requires both readings (bearish conviction vs squeeze setup) be
    presented rather than collapsed into one directional signal, which is
    exactly why `smart_money` keeps them out of its blended score. That only
    works if a page actually shows them.
    """
    from datetime import date as _date

    with get_session() as session:
        frame = persistence.read_latest_short_interest(session, as_of=_date.today())
    if frame.empty:
        return None
    row = frame[frame["symbol"] == symbol]
    return None if row.empty else dict(row.iloc[0])


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def options_signals(symbol: str) -> dict[str, Any] | None:
    """One symbol's latest options snapshot (put/call ratio, ATM implied volatility).

    The implied-volatility half of Section 7.7's "historical & implied where
    available" risk block -- ingested nightly, and until now read by nothing but
    the smart-money blend.
    """
    from datetime import date as _date

    with get_session() as session:
        frame = persistence.read_latest_options(session, as_of=_date.today())
    if frame.empty:
        return None
    row = frame[frame["symbol"] == symbol]
    return None if row.empty else dict(row.iloc[0])


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def sentiment_history(symbol: str, limit: int = 2) -> list[tuple[date, float, int | None]]:
    """One symbol's latest sentiment readings, newest first — the "what moved it" input."""
    with get_session() as session:
        return persistence.read_symbol_sentiment_history(session, symbol, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def filing_excerpt(symbol: str) -> dict[str, Any] | None:
    """The latest 10-K/10-Q excerpt for `symbol`, fetched on demand from SEC EDGAR.

    Not part of the nightly job on purpose: a 10-K document is several megabytes
    and a reader opens one company at a time (see
    `edgar_client.fetch_filing_excerpt`).
    """
    from quantpulse.ingestion import edgar_client

    try:
        return edgar_client.fetch_filing_excerpt(symbol)
    except Exception:  # noqa: BLE001 - a filing fetch failing must not take the page down
        return None


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def symbol_news(symbol: str, limit: int = 10) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_symbol_news(session, symbol, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def market_moving_news(limit: int = 8) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_market_moving_news(session, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def market_regime(limit: int = 90) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_recent_market_regime(session, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def backtest_history(limit: int = 20) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_backtest_history(session, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def refresh_log(limit: int = 20) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_refresh_log(session, limit=limit)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def data_freshness() -> dict[str, date | None]:
    with get_session() as session:
        return persistence.read_data_freshness(session)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def universe() -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_ticker_universe(session)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def latest_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    # Takes a tuple rather than a list so Streamlit can hash the argument for
    # the cache key -- a list is unhashable and would raise at call time.
    with get_session() as session:
        return persistence.read_latest_prices(session, list(symbols))


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def macro_series(name: str, lookback_days: int = 60) -> list[float]:
    """A cross-asset series (VIX, oil, gold, the dollar index), oldest first.

    Feeds Section 28's targeted sector overlay, which had no consumer at all --
    the series were ingested nightly and nothing ever computed the adjustment.
    """
    from datetime import date as _date

    with get_session() as session:
        return persistence.read_macro_series(
            session, name, as_of=_date.today(), lookback_days=lookback_days
        )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def universe_panel(lookback_days: int = 150) -> pd.DataFrame:
    """The whole active universe's adjusted-close panel, for market-wide views.

    Used by the Dashboard's sector-rotation table. Deliberately a shorter window
    than the portfolio panel: rotation is a one-month relative-strength read, so
    years of history would be cost without benefit.
    """
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    end = _date.today()
    with get_session() as session:
        symbols = list(persistence.read_ticker_universe(session)["symbol"])
        return persistence.read_adj_close_panel(
            session, start=end - _timedelta(days=lookback_days), end=end, symbols=symbols
        )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def benchmark_closes(lookback_days: int) -> pd.Series:
    """The market index's stored adjusted closes over the trailing window.

    Its own reader rather than a column pulled out of `universe_panel`: the
    index is deliberately not in the universe (see
    `persistence.upsert_benchmark_ticker`), so no panel this app builds
    contains it.
    """
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    end = _date.today()
    with get_session() as session:
        return persistence.read_benchmark_closes(
            session,
            symbol=risk.MARKET_INDEX_SYMBOL,
            start=end - _timedelta(days=lookback_days),
            end=end,
        )


def market_series(lookback_days: int = risk.MARKET_PANEL_DAYS) -> risk.MarketSeries:
    """The market return series every Streamlit beta is regressed against.

    Deliberately **not** cached itself — both of its inputs are, and a
    `MarketSeries` wraps a Series that `st.cache_data` would have to pickle for
    no gain. What matters is that the Stock Detail page and the Portfolio page
    call *this*, not two hand-rolled equivalents: those two, plus the API, are
    the three copies that once published three different betas.
    """
    return risk.resolve_market_returns(
        benchmark_closes(lookback_days), universe_panel(lookback_days)
    )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def adj_close_panel(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    with get_session() as session:
        return persistence.read_adj_close_panel(
            session, start=start, end=end, symbols=list(symbols)
        )


def has_any_data() -> bool:
    """Whether the pipeline has ever produced scores -- drives the empty-state banner."""
    freshness = data_freshness()
    return any(value is not None for value in freshness.values())


def portfolio_backend() -> str:
    """Which storage backend the config selects (ADR 4.5): `sqlite` or `session`."""
    return get_settings().portfolio_backend


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def catalogue() -> pd.DataFrame:
    """Every listed symbol a visitor may search for, ranked or merely catalogued.

    Cached like the rest of this module, and it matters more here: the search
    box re-runs on every keystroke, and this is ~13,000 rows.
    """
    try:
        with get_session() as session:
            return persistence.read_searchable_catalogue(session)
    except Exception:
        # A database written before the `coverage` migration has no catalogue.
        # That is a stale file, not a broken app: the caller falls back to the
        # ranked universe and the page keeps working, which is a far better
        # outcome than a traceback where the search box should be. The next
        # nightly run migrates and re-commits the database.
        logger.warning("Catalogue unavailable; falling back to the ranked universe.")
        return pd.DataFrame(
            columns=["symbol", "name", "sector", "asset_type", "exchange", "coverage"]
        )


# On-demand results are cached for far longer than the rest of this module.
# Each one is a live fetch of three endpoints plus a forecast, so re-running it
# on every Streamlit rerun -- which is every widget interaction -- would make
# the page feel broken and hammer a free data source for no new information.
# Fifteen minutes sits comfortably inside the life of a daily bar.
ON_DEMAND_TTL_SECONDS = 900


@st.cache_data(ttl=ON_DEMAND_TTL_SECONDS, show_spinner=False)
def on_demand_analysis(symbol: str, sector: str | None) -> Any:
    """Analyse a symbol the nightly job has never scored. None if it has no prices.

    The stored composites and the stored fundamentals go in as context: the
    first places this stock against the ranking, and the second is the peer
    group its fundamentals are ranked inside. Without that peer group the
    fundamental score would be a company ranked against itself, which is
    always 100 -- see `on_demand._fundamental_score`.
    """
    from datetime import date as _date

    from quantpulse import on_demand

    with get_session() as session:
        ranked = persistence.read_screener_rows(session)
        peers = persistence.read_latest_fundamentals(session, as_of=_date.today())
    return on_demand.analyse(
        symbol,
        sector=sector,
        ranked_composites=ranked["composite_score"] if not ranked.empty else None,
        sector_peers=peers if not peers.empty else None,
    )
