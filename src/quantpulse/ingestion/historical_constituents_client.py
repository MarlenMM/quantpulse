"""Point-in-time S&P 500 membership, for survivorship-bias-aware backtests.

Sections 5 & 22: today's Wikipedia constituent list silently excludes every
company that was removed (bankruptcy, acquisition, demotion), which is exactly
the data a backtest must keep. This module loads an interval-format historical
dataset (`ticker, start_date, end_date`) that includes those removed names.

If no such dataset is configured or reachable, the seed script falls back to
`build_current_only_membership()` — today's survivors only — which is honest
but survivorship-biased, and must be surfaced as a documented limitation
rather than presented as complete history.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantpulse.config import get_settings
from quantpulse.ingestion import wikipedia_client
from quantpulse.ingestion.cache import cached_dataframe

logger = logging.getLogger(__name__)

INDEX_NAME = "S&P 500"
_MEMBERSHIP_COLUMNS = ["index_name", "symbol", "added_date", "removed_date"]


class HistoricalMembershipUnavailable(RuntimeError):
    """No point-in-time membership source is configured or reachable."""


def _normalize_symbol(symbol: str) -> str:
    # Match the '.'-> '-' convention the price/fundamental providers use.
    return str(symbol).strip().upper().replace(".", "-")


def _parse_interval_frame(raw: pd.DataFrame, *, today: date | None = None) -> pd.DataFrame:
    expected = {"ticker", "start_date", "end_date"}
    missing = expected - set(raw.columns)
    if missing:
        raise HistoricalMembershipUnavailable(
            f"historical constituents dataset missing columns: {sorted(missing)}"
        )

    cutoff = today or date.today()
    added = pd.to_datetime(raw["start_date"], errors="coerce")
    removed = pd.to_datetime(raw["end_date"], errors="coerce")

    df = pd.DataFrame(
        {
            "index_name": INDEX_NAME,
            "symbol": raw["ticker"].map(_normalize_symbol),
            "added_date": [d.date() if pd.notna(d) else None for d in added],
            "removed_date": [_removed_or_open(d, cutoff) for d in removed],
        }
    )
    # A row with no parseable start_date can't be placed in time; drop it.
    df = df[df["added_date"].notna()].copy()
    df = df[df["symbol"].astype(bool)]
    return df[_MEMBERSHIP_COLUMNS].reset_index(drop=True)


def _removed_or_open(parsed_end: pd.Timestamp, cutoff: date) -> date | None:
    """A real removal date, or `None` when the interval is still open.

    Datasets disagree on how they mark a *current* member's open interval: some
    leave `end_date` empty (parses to NaT), others use a far-future sentinel
    (e.g. `2059-12-31`). Both must resolve to `None`, or every current member
    would be recorded with a `removed_date` and wrongly flagged inactive -- and
    the survivorship-aware backtest would treat a live name as long gone
    (Sections 5, 22). A removal dated after today is, as of today, no removal.
    """
    if pd.isna(parsed_end):
        return None
    end = parsed_end.date()
    return None if end > cutoff else end


def fetch_historical_membership() -> pd.DataFrame:
    """Load point-in-time membership as [index_name, symbol, added_date, removed_date].

    Prefers a local file (`historical_constituents_path`) for reproducibility,
    else the configured URL. Raises `HistoricalMembershipUnavailable` if neither
    is set or the download/parse fails, so the caller can fall back explicitly.
    """
    settings = get_settings()

    local_path = settings.historical_constituents_path
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise HistoricalMembershipUnavailable(f"local dataset not found: {path}")
        logger.info("Loading historical S&P 500 membership from local file %s", path)
        return _parse_interval_frame(pd.read_csv(path))

    url = settings.historical_constituents_url
    if not url:
        raise HistoricalMembershipUnavailable("no historical_constituents_path or _url configured")

    def _fetch() -> pd.DataFrame:
        logger.info("Downloading historical S&P 500 membership from %s", url)
        raw = pd.read_csv(url, storage_options={"User-Agent": wikipedia_client._USER_AGENT})
        return _parse_interval_frame(raw)

    cache_dir = Path(settings.ingestion_cache_dir) / "historical_constituents"
    try:
        return cached_dataframe("sp500_membership", _fetch, cache_dir, ttl=timedelta(days=30))
    except Exception as exc:  # network error, HTTP error, malformed CSV
        raise HistoricalMembershipUnavailable(str(exc)) from exc


def build_current_only_membership(fallback_added_date: date | None = None) -> pd.DataFrame:
    """Degraded, survivorship-biased membership: today's constituents only.

    Uses real Wikipedia add-dates where available; any name whose add-date
    can't be parsed is seeded with `fallback_added_date` (default: a fixed
    floor). All `removed_date`s are null because this source has no record of
    removals — which is the entire survivorship-bias limitation (Section 22).
    """
    floor = fallback_added_date or date(1990, 1, 1)
    logger.warning(
        "SURVIVORSHIP-BIAS LIMITATION: no historical membership dataset available; "
        "seeding index_membership_history with today's constituents only. Removed "
        "companies are absent, so any backtest over this universe is survivorship-biased. "
        "See Section 22; set historical_constituents_path/_url to correct this."
    )

    current = wikipedia_client.fetch_sp500_constituents()[["symbol"]].copy()
    dates = wikipedia_client.fetch_sp500_date_added()
    merged = current.merge(dates, on="symbol", how="left")
    merged["added_date"] = merged["added_date"].where(merged["added_date"].notna(), floor)
    merged["index_name"] = INDEX_NAME
    merged["removed_date"] = None
    return merged[_MEMBERSHIP_COLUMNS].reset_index(drop=True)
