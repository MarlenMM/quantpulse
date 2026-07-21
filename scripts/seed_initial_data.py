"""One-time cold-start historical backfill (Section 6.2).

Deliberately a *separate* script from the nightly `refresh_data.py`: this is
the big, slow, run-once job that (1) reconstructs survivorship-bias-aware index
membership and (2) pulls years of daily price history for every symbol that was
ever in the index — not just today's survivors. It is:

- **Survivorship-bias-aware** (Sections 5, 22): membership comes from a
  point-in-time dataset that includes removed companies; the union of all
  historical symbols is what gets price history, so a later backtest can see
  the losers, not only the winners.
- **Resumable** (Section 6.2): progress is inferred from the database itself —
  a symbol that already has deep history is skipped — so a rate-limit
  interruption after 300 tickers just continues from 301 on the next run,
  rather than starting over.
- **Gently paced**: prices are pulled sequentially through the same
  rate-limited, circuit-broken yfinance client the nightly uses. Speed is not
  the goal here; not getting the unofficial endpoint to throttle us is.
- **Staged-rollout friendly** (Section 6.13): `--limit`/`--symbols` let you
  prove the pipeline on 10-20 tickers before committing to the full universe.

Run once, locally:  ``uv run python scripts/seed_initial_data.py``
"""

import argparse
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from quantpulse.config import get_settings
from quantpulse.ingestion import historical_constituents_client as hist
from quantpulse.ingestion import wikipedia_client, yfinance_client
from quantpulse.ingestion.historical_constituents_client import HistoricalMembershipUnavailable
from quantpulse.storage.db import get_session
from quantpulse.storage.models import IndexMembershipHistory, PriceHistory, RefreshLog, Ticker
from quantpulse.utils.log import configure_logging

logger = logging.getLogger(__name__)

# A symbol with price data at least this old is treated as already backfilled
# (the nightly only ever stores a handful of recent days), which is what makes
# re-running the seed resume instead of re-fetching everything.
_BACKFILL_STALENESS_DAYS = 300

SessionFactory = Callable[[], AbstractContextManager[Session]]


def resolve_membership() -> tuple[pd.DataFrame, str]:
    """Return (membership_frame, mode). Falls back to current-only if needed."""
    try:
        membership = hist.fetch_historical_membership()
        return membership, "historical"
    except HistoricalMembershipUnavailable as exc:
        logger.warning("Historical membership unavailable (%s); falling back.", exc)
        return hist.build_current_only_membership(), "current_only"


def seed_tickers(session: Session, membership: pd.DataFrame) -> int:
    """Ensure a `tickers` row exists for every symbol in `membership`.

    Removed names get minimal placeholder rows (is_active=False) so the
    membership/price foreign keys resolve; current names are enriched with
    Wikipedia sector/industry where available. Existing rows are never
    clobbered (on-conflict-do-nothing), so a prior nightly's richer data and
    its is_active bookkeeping survive a re-seed.
    """
    try:
        current = wikipedia_client.fetch_sp500_constituents()
        meta = {r.symbol: r for r in current.itertuples(index=False)}
    except Exception:
        logger.exception("Could not load current constituent metadata; using placeholders only")
        meta = {}

    # A symbol counts as active if any of its membership intervals is still open.
    active_symbols = set(membership.loc[membership["removed_date"].isna(), "symbol"])

    rows = []
    for symbol in sorted(membership["symbol"].unique()):
        info = meta.get(symbol)
        rows.append(
            {
                "symbol": symbol,
                "name": info.name if info is not None else symbol,
                "sector": info.sector if info is not None else None,
                "industry": info.industry if info is not None else None,
                "exchange": None,
                "asset_type": "equity",
                "is_active": symbol in active_symbols,
            }
        )

    stmt = sqlite_insert(Ticker).values(rows).on_conflict_do_nothing(index_elements=["symbol"])
    session.execute(stmt)
    return len(rows)


def seed_index_membership(session: Session, membership: pd.DataFrame) -> int:
    """Replace this index's membership rows with the resolved point-in-time set.

    Membership is authoritative reference data, so re-seeding from an updated
    dataset legitimately replaces it — distinct from the never-overwrite rule
    for price/score *history* (Section 6.8/6.9), which this never touches.
    """
    session.query(IndexMembershipHistory).filter(
        IndexMembershipHistory.index_name == hist.INDEX_NAME
    ).delete()

    records = [
        {
            "index_name": row.index_name,
            "symbol": row.symbol,
            "added_date": row.added_date,
            "removed_date": row.removed_date,
        }
        for row in membership.itertuples(index=False)
    ]
    session.execute(sqlite_insert(IndexMembershipHistory).values(records))
    return len(records)


def _needs_backfill(session: Session, symbol: str, cutoff: date) -> bool:
    oldest = session.scalar(
        select(func.min(PriceHistory.date)).where(PriceHistory.symbol == symbol)
    )
    return oldest is None or oldest > cutoff


def _upsert_price_history(session: Session, df: pd.DataFrame) -> int:
    # Mirrors refresh_data._upsert_price_history; kept local so the two one-off
    # scripts stay independent. Point-in-time rows are keyed by (symbol, date).
    if df.empty:
        return 0
    records = df.to_dict("records")
    stmt = sqlite_insert(PriceHistory).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={c: stmt.excluded[c] for c in ("open", "high", "low", "close", "adj_close", "volume")},
    )
    session.execute(stmt)
    return len(records)


def backfill_prices(
    symbols: list[str],
    period: str,
    session_factory: SessionFactory = get_session,
    today: date | None = None,
) -> tuple[int, int]:
    """Fetch and store full history for each symbol needing it. Returns (rows, skipped)."""
    cutoff = (today or date.today()) - timedelta(days=_BACKFILL_STALENESS_DAYS)
    rows_written = 0
    skipped = 0

    for index, symbol in enumerate(symbols, start=1):
        with session_factory() as session:
            if not _needs_backfill(session, symbol, cutoff):
                skipped += 1
                continue

        try:
            df = yfinance_client.fetch_price_history(symbol, period=period)
        except Exception:
            logger.exception("Backfill failed for %s (continuing)", symbol)
            continue

        with session_factory() as session:
            written = _upsert_price_history(session, df)
        rows_written += written
        logger.info("[%d/%d] %s: stored %d price rows", index, len(symbols), symbol, written)

    return rows_written, skipped


def _select_symbols(
    membership: pd.DataFrame, override: list[str] | None, limit: int | None
) -> list[str]:
    if override:
        return [hist._normalize_symbol(s) for s in override]
    symbols = sorted(membership["symbol"].unique())
    return symbols[:limit] if limit else symbols


def run(
    limit: int | None = None,
    symbols: list[str] | None = None,
    period: str | None = None,
    skip_prices: bool = False,
    skip_membership: bool = False,
    session_factory: SessionFactory = get_session,
) -> None:
    run_id = configure_logging(get_settings().log_level)
    started_at = datetime.now()
    period = period or get_settings().seed_history_period
    status = "success"
    rows_updated = 0

    logger.info("seed_initial_data starting (run_id=%s)", run_id)

    try:
        membership, mode = resolve_membership()
        logger.info(
            "Resolved %d membership rows in '%s' mode (%d symbols)",
            len(membership),
            mode,
            membership["symbol"].nunique(),
        )
        if mode == "current_only":
            status = "partial_survivorship_biased"

        if not skip_membership:
            with session_factory() as session:
                rows_updated += seed_tickers(session, membership)
                rows_updated += seed_index_membership(session, membership)

        if not skip_prices:
            targets = _select_symbols(membership, symbols, limit)
            logger.info(
                "Backfilling price history for %d symbols (period=%s)", len(targets), period
            )
            written, skipped = backfill_prices(targets, period, session_factory)
            rows_updated += written
            logger.info("Backfill complete: %d rows written, %d symbols skipped", written, skipped)

    except Exception:
        logger.exception("seed_initial_data failed")
        status = "failed"

    with session_factory() as session:
        session.add(
            RefreshLog(
                job_name="seed_initial_data",
                run_timestamp=started_at,
                status=status,
                rows_updated=rows_updated,
            )
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-time cold-start historical backfill")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N symbols")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol override")
    parser.add_argument(
        "--period", type=str, default=None, help="yfinance history period (e.g. max, 10y)"
    )
    parser.add_argument("--skip-prices", action="store_true", help="Seed membership only")
    parser.add_argument("--skip-membership", action="store_true", help="Backfill prices only")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    run(
        limit=args.limit,
        symbols=symbols,
        period=args.period,
        skip_prices=args.skip_prices,
        skip_membership=args.skip_membership,
    )


if __name__ == "__main__":
    main()
