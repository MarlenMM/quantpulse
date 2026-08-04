"""Portfolio storage: one interface, two backends (ADR 4.5, Sections 25, 30).

ADR 4.5 is the reason this module exists in this shape. On the public demo,
holdings must live in the visitor's own browser session -- a shared SQLite file
would let every visitor see and overwrite every other visitor's portfolio. On
your own machine, holdings must persist across restarts like any normal app.
The ADR's decision is explicit that this should be "one code path, not two,"
switched by a single config flag, so the difference is confined to a
`PortfolioStore` implementation and nothing above it knows which one it has.

**The transaction log is the source of truth** (Section 30). Both backends
store `Transaction` records and *derive* positions through
`transactions.build_lot_book`; neither lets a caller edit a holdings row
directly. The `portfolio_holdings` table is a rewritten-from-the-log cache, not
independent state, which is what makes it structurally impossible for the
snapshot and the log to disagree.

**No Streamlit import anywhere here.** `SessionPortfolioStore` takes a plain
mutable dict, and the UI hands it `st.session_state`. That keeps the
presentation dependency in `app/` where Section 14 says it belongs, and means
the session backend is testable with an ordinary dict.

Cash is held as a scalar rather than as a synthetic "CASH" transaction: cash has
no cost basis, no tax lot, and no holding period, so routing it through the FIFO
lot machinery would mean special-casing it at every step to suppress concepts
that don't apply to it.
"""

from __future__ import annotations

import csv
import io
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from quantpulse.config import get_settings
from quantpulse.portfolio.transactions import Transaction, build_lot_book, positions
from quantpulse.storage.models import PortfolioHolding, PortfolioTransaction, WatchlistEntry

__all__ = [
    "ASSET_TYPES",
    "CSV_COLUMNS",
    "EXAMPLE_PORTFOLIO",
    "PortfolioState",
    "PortfolioStore",
    "SessionPortfolioStore",
    "SqlitePortfolioStore",
    "get_store",
    "to_csv",
    "from_csv",
    "example_state",
]

# Section 9: ETFs are first-class but skip company-fundamental scoring (a fund
# has no P/E in the way a company does); cash is tracked separately as a scalar.
ASSET_TYPES = ("equity", "etf")

CSV_COLUMNS = ("symbol", "action", "shares", "price", "date")

# Section 25's "Load example portfolio" button: a first-time visitor should see
# a populated, working page instead of an empty form. Deliberately boring,
# well-known, multi-sector names -- this is scaffolding to demo the analytics,
# not a suggested allocation.
EXAMPLE_PORTFOLIO: tuple[tuple[str, float, float, date], ...] = (
    ("AAPL", 25.0, 180.50, date(2024, 3, 14)),
    ("MSFT", 15.0, 405.20, date(2024, 5, 2)),
    ("JNJ", 30.0, 152.10, date(2023, 11, 8)),
    ("XOM", 40.0, 104.75, date(2024, 1, 22)),
    ("JPM", 20.0, 188.30, date(2024, 7, 9)),
)


@dataclass
class PortfolioState:
    """Everything a portfolio is: a trade log, a cash balance, and a watchlist.

    `asset_types` maps a symbol to `"equity"`/`"etf"`; anything absent is an
    equity. Kept per-symbol rather than per-transaction because an asset's type
    is a property of the instrument, not of the trade that bought it.
    """

    transactions: list[Transaction] = field(default_factory=list)
    cash: float = 0.0
    watchlist: list[str] = field(default_factory=list)
    asset_types: dict[str, str] = field(default_factory=dict)

    def symbols(self) -> list[str]:
        """Every symbol that has ever been traded, in first-seen order."""
        seen: dict[str, None] = {}
        for tx in self.transactions:
            seen.setdefault(tx.symbol, None)
        return list(seen)

    def asset_type(self, symbol: str) -> str:
        return self.asset_types.get(symbol, "equity")


class PortfolioStore(ABC):
    """Read/write portfolio state, backed by either a browser session or SQLite."""

    backend: str

    @abstractmethod
    def load(self) -> PortfolioState:
        """The current state. Always returns a state, never `None` (empty if new)."""

    @abstractmethod
    def save(self, state: PortfolioState) -> None:
        """Persist `state` wholesale, replacing whatever was there."""

    # -- convenience mutations, expressed in terms of load/save so a backend
    # only ever has to implement the two primitives above ------------------- #

    def add_transaction(self, transaction: Transaction, *, asset_type: str = "equity") -> None:
        """Append one buy/sell to the log (Section 30: log it, don't edit a snapshot).

        The prospective log is replayed through `build_lot_book` *before* it is
        saved, so an impossible transaction (selling more shares than are held,
        a non-positive price) is rejected at the point of entry and the stored
        log is never left in a state that cannot be interpreted.

        This validation lives here, in the shared base method, rather than in
        either backend: the SQLite store happens to replay the log inside
        `save` to rebuild its holdings snapshot and would have caught this
        incidentally, while the session store would have accepted the bad row
        and then failed on every subsequent render. Two backends disagreeing
        about what is valid is precisely what ADR 4.5's "one code path, not
        two" is meant to prevent.
        """
        if asset_type not in ASSET_TYPES:
            raise ValueError(f"asset_type must be one of {ASSET_TYPES}, got {asset_type!r}")
        state = self.load()
        candidate = [*state.transactions, transaction]
        build_lot_book(candidate)  # raises on an impossible log; nothing saved
        state.transactions.append(transaction)
        state.asset_types[transaction.symbol] = asset_type
        self.save(state)

    def set_cash(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"cash must be >= 0, got {amount}")
        state = self.load()
        state.cash = float(amount)
        self.save(state)

    def add_to_watchlist(self, symbol: str) -> None:
        state = self.load()
        cleaned = symbol.strip().upper()
        if cleaned and cleaned not in state.watchlist:
            state.watchlist.append(cleaned)
            self.save(state)

    def remove_from_watchlist(self, symbol: str) -> None:
        state = self.load()
        cleaned = symbol.strip().upper()
        if cleaned in state.watchlist:
            state.watchlist.remove(cleaned)
            self.save(state)

    def clear(self) -> None:
        self.save(PortfolioState())

    def current_positions(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        """Derived positions from the log (Section 30), priced where possible.

        Delegates to `transactions.build_lot_book` + `positions`, so FIFO lots,
        fractional shares and the graceful handling of a delisted holding all
        behave identically no matter which backend stored the log.
        """
        state = self.load()
        book = build_lot_book(state.transactions)
        return dict(positions(book, current_prices=prices))


class SessionPortfolioStore(PortfolioStore):
    """In-memory, per-browser-session state -- the public demo backend (ADR 4.5).

    Takes any mutable mapping (the UI passes Streamlit's `st.session_state`),
    so nothing in `src/` imports Streamlit and the backend is testable with a
    plain dict. State is *not* copied on load: mutating the returned
    `PortfolioState` and calling `save` is the normal flow, and there is no
    cross-session sharing to protect against because the mapping itself is
    per-session.
    """

    backend = "session"

    def __init__(self, container: MutableMapping[str, Any], key: str = "quantpulse_portfolio"):
        self._container = container
        self._key = key

    def load(self) -> PortfolioState:
        state = self._container.get(self._key)
        if not isinstance(state, PortfolioState):
            state = PortfolioState()
            self._container[self._key] = state
        return state

    def save(self, state: PortfolioState) -> None:
        self._container[self._key] = state


class SqlitePortfolioStore(PortfolioStore):
    """Persistent local backend (ADR 4.5's `PORTFOLIO_BACKEND=sqlite`).

    `save` rewrites the transaction log, the derived holdings snapshot and the
    watchlist together in one transaction. Rewriting wholesale rather than
    diffing is the right trade at this size (a personal portfolio is tens of
    rows, not millions) and removes any possibility of the derived
    `portfolio_holdings` cache drifting from the log it is derived from.

    **The store owns its transaction.** It takes a session *factory* (a callable
    returning a context manager, e.g. `db.get_session`) and opens/commits one
    per operation, rather than borrowing a long-lived session from its caller.
    That is not a style preference: Streamlit's `st.rerun()` raises immediately
    to restart the script, so any commit a page deferred until after a mutation
    would simply never execute and the write would be silently lost. Owning the
    transaction here makes a save durable the moment it returns, whatever the
    caller does next.
    """

    backend = "sqlite"

    def __init__(self, session_factory: Callable[[], AbstractContextManager[Session]]):
        self._session_factory = session_factory

    def load(self) -> PortfolioState:
        with self._session_factory() as session:
            return self._load(session)

    def _load(self, session: Session) -> PortfolioState:
        rows = session.scalars(
            select(PortfolioTransaction).order_by(
                PortfolioTransaction.date, PortfolioTransaction.id
            )
        ).all()
        transactions = [
            Transaction(
                symbol=row.symbol,
                action="buy" if row.action == "buy" else "sell",
                shares=row.shares,
                price=row.price,
                date=row.date,
            )
            for row in rows
        ]
        holdings = session.scalars(select(PortfolioHolding)).all()
        cash = next((h.shares for h in holdings if h.asset_type == "cash"), 0.0)
        asset_types = {h.symbol: h.asset_type for h in holdings if h.asset_type in ASSET_TYPES}
        watchlist = list(
            session.scalars(select(WatchlistEntry.symbol).order_by(WatchlistEntry.added_date))
        )
        return PortfolioState(
            transactions=transactions,
            cash=float(cash),
            watchlist=watchlist,
            asset_types=asset_types,
        )

    def save(self, state: PortfolioState) -> None:
        with self._session_factory() as session:
            self._save(session, state)

    def _save(self, session: Session, state: PortfolioState) -> None:
        session.execute(delete(PortfolioTransaction))
        session.execute(delete(PortfolioHolding))
        session.execute(delete(WatchlistEntry))

        for tx in state.transactions:
            session.add(
                PortfolioTransaction(
                    symbol=tx.symbol,
                    action=tx.action,
                    shares=tx.shares,
                    price=tx.price,
                    date=tx.date,
                )
            )

        book = build_lot_book(state.transactions)
        for symbol, position in positions(book).items():
            lots = book.open_lots[symbol]
            session.add(
                PortfolioHolding(
                    symbol=symbol,
                    asset_type=state.asset_type(symbol),
                    shares=position.shares,
                    cost_basis=position.cost_basis,
                    purchase_date=min(lot.purchase_date for lot in lots),
                )
            )
        if state.cash > 0:
            # Cash rides in the snapshot as a pseudo-position (Section 9/13's
            # `asset_type`) so a restored portfolio keeps its cash sleeve; it
            # has no transaction log because it has no cost basis or tax lot.
            session.add(
                PortfolioHolding(
                    symbol="CASH",
                    asset_type="cash",
                    shares=state.cash,
                    cost_basis=state.cash,
                    purchase_date=None,
                )
            )

        for symbol in state.watchlist:
            session.add(WatchlistEntry(symbol=symbol, added_date=date.today()))
        session.flush()


def get_store(
    session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
    container: MutableMapping[str, Any] | None = None,
) -> PortfolioStore:
    """The store the `PORTFOLIO_BACKEND` flag selects (ADR 4.5).

    Pass whichever backing the configured backend needs -- a session factory
    (e.g. `db.get_session`) for `sqlite`, a mutable `container` (Streamlit's
    `st.session_state`) for `session`. Raises if the one the config asks for
    wasn't supplied, rather than silently falling back to the other: quietly
    writing a local portfolio into an in-memory store that evaporates on refresh
    (or, worse, quietly writing a demo visitor's holdings into a shared file) is
    exactly the failure ADR 4.5 exists to prevent.
    """
    backend = get_settings().portfolio_backend
    if backend == "sqlite":
        if session_factory is None:
            raise ValueError("PORTFOLIO_BACKEND=sqlite requires a database session factory")
        return SqlitePortfolioStore(session_factory)
    if container is None:
        raise ValueError("PORTFOLIO_BACKEND=session requires a session-state container")
    return SessionPortfolioStore(container)


# --------------------------------------------------------------------------- #
# CSV round-trip (Section 25's "download my portfolio" / "upload to restore")
# --------------------------------------------------------------------------- #


def to_csv(state: PortfolioState) -> str:
    """Serialize the transaction log as CSV.

    The log, not the holdings snapshot: exporting derived positions would lose
    the cost-basis history that makes FIFO lots and realized P/L reconstructible
    (Section 30). Cash and the watchlist are deliberately not in this file --
    it is a *transaction* export, and keeping the format identical to the import
    format is what makes the round-trip honest.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for tx in state.transactions:
        writer.writerow([tx.symbol, tx.action, tx.shares, tx.price, tx.date.isoformat()])
    return buffer.getvalue()


def from_csv(text: str) -> list[Transaction]:
    """Parse a transaction CSV back into `Transaction` records.

    Accepts the header this module writes, case-insensitively, and raises
    `ValueError` naming the offending row on bad input -- an upload that is
    silently half-imported is worse than one that is rejected with a reason.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV appears to be empty")
    normalized = [name.strip().lower() for name in reader.fieldnames]
    missing = [column for column in CSV_COLUMNS if column not in normalized]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    parsed: list[Transaction] = []
    for line_number, raw in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not any(row.values()):
            continue
        try:
            action = row["action"].lower()
            if action not in ("buy", "sell"):
                raise ValueError(f"action must be 'buy' or 'sell', got {row['action']!r}")
            parsed.append(
                Transaction(
                    symbol=row["symbol"].upper(),
                    action="buy" if action == "buy" else "sell",
                    shares=_positive_number(row["shares"], "shares"),
                    price=_positive_number(row["price"], "price"),
                    date=date.fromisoformat(row["date"]),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"row {line_number} is invalid: {exc}") from exc
    return parsed


def _positive_number(text: str, field: str) -> float:
    """A finite, strictly-positive float from a CSV cell, or `ValueError` naming the field.

    `float()` happily parses "nan" and "inf", and every comparison against NaN
    is False -- so a plain `value <= 0` check lets both through. This function's
    caller promises to name the offending *row*, which is only useful if the bad
    value is rejected here rather than several layers down in `build_lot_book`.
    """
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a finite number > 0, got {text!r}")
    return value


def example_state(cash: float = 5_000.0) -> PortfolioState:
    """The "Load example portfolio" state (Section 25) -- a populated demo page."""
    return PortfolioState(
        transactions=[
            Transaction(symbol=symbol, action="buy", shares=shares, price=price, date=bought)
            for symbol, shares, price, bought in EXAMPLE_PORTFOLIO
        ],
        cash=cash,
        watchlist=["NVDA", "GOOGL"],
    )


def replace_transactions(
    store: PortfolioStore, transactions: Iterable[Transaction], *, cash: float | None = None
) -> PortfolioState:
    """Replace the whole log (a CSV restore), keeping cash/watchlist unless overridden.

    The replacement log is replayed through `build_lot_book` **before** anything
    is stored, for exactly the reason `PortfolioStore.add_transaction` does it:
    otherwise the two backends disagree about what is valid, which is what ADR
    4.5's "one code path, not two" exists to prevent. Uploading a CSV that sells
    more shares than it ever bought (or carries a NaN share count) used to be
    accepted outright by the session backend, and the Portfolio page then raised
    on *every* subsequent render, because the first thing it does is replay the
    log it just stored. The SQLite backend happened to catch the same file
    inside `save`, but only after issuing its DELETEs, and reported it as an
    unhandled traceback rather than a message about the file.
    """
    candidate = list(transactions)
    build_lot_book(candidate)  # raises on an impossible log; nothing saved
    state = store.load()
    updated = replace(state, transactions=candidate)
    if cash is not None:
        updated.cash = float(cash)
    store.save(updated)
    return updated


def sector_weights(
    position_values: dict[str, float], sectors: dict[str, str | None]
) -> dict[str, float]:
    """Fraction of invested value per sector, for the allocation chart (Section 9).

    Symbols with no known sector (ETFs, cash) are grouped under "Unclassified"
    rather than dropped, so the chart's slices always sum to the whole
    portfolio -- a pie that quietly omits a third of the value is worse than
    one with an honest catch-all slice.
    """
    total = sum(value for value in position_values.values() if value > 0)
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for symbol, value in position_values.items():
        if value <= 0:
            continue
        sector = sectors.get(symbol) or "Unclassified"
        weights[sector] = weights.get(sector, 0.0) + value / total
    return weights


def load_state_from_rows(rows: Sequence[dict[str, Any]]) -> PortfolioState:
    """Build a state from plain dict rows (the manual-entry form's output)."""
    return PortfolioState(
        transactions=[
            Transaction(
                symbol=str(row["symbol"]).upper(),
                action="buy" if str(row.get("action", "buy")).lower() == "buy" else "sell",
                shares=float(row["shares"]),
                price=float(row["price"]),
                date=row["date"]
                if isinstance(row["date"], date)
                else date.fromisoformat(str(row["date"])),
            )
            for row in rows
        ]
    )
