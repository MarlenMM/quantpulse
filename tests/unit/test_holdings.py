from collections.abc import Iterator
from contextlib import nullcontext
from datetime import date

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.portfolio import holdings as hl
from quantpulse.portfolio.transactions import Transaction
from quantpulse.storage.models import Base


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'portfolio.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        yield s


@pytest.fixture
def sqlite_store(session: Session) -> hl.SqlitePortfolioStore:
    # nullcontext hands the store the same open session every time, so writes
    # stay inside the test's transaction and are visible on reload.
    return hl.SqlitePortfolioStore(lambda: nullcontext(session))


def _buy(symbol: str, shares: float, price: float, day: date) -> Transaction:
    return Transaction(symbol=symbol, action="buy", shares=shares, price=price, date=day)


STORES = ["session", "sqlite"]


@pytest.fixture
def store(request, sqlite_store: hl.SqlitePortfolioStore) -> hl.PortfolioStore:
    """Both backends, so every behavior below is asserted against each (ADR 4.5)."""
    if request.param == "session":
        return hl.SessionPortfolioStore({})
    return sqlite_store


class TestPortfolioState:
    def test_symbols_are_first_seen_order(self) -> None:
        state = hl.PortfolioState(
            transactions=[
                _buy("MSFT", 1, 10, date(2024, 1, 1)),
                _buy("AAPL", 1, 10, date(2024, 1, 2)),
                _buy("MSFT", 1, 11, date(2024, 1, 3)),
            ]
        )
        assert state.symbols() == ["MSFT", "AAPL"]

    def test_asset_type_defaults_to_equity(self) -> None:
        state = hl.PortfolioState(asset_types={"SPY": "etf"})
        assert state.asset_type("SPY") == "etf"
        assert state.asset_type("AAPL") == "equity"


@pytest.mark.parametrize("store", STORES, indirect=True)
class TestBothBackends:
    def test_starts_empty(self, store: hl.PortfolioStore) -> None:
        state = store.load()
        assert state.transactions == []
        assert state.cash == 0.0
        assert state.watchlist == []

    def test_add_transaction_round_trips(self, store: hl.PortfolioStore) -> None:
        store.add_transaction(_buy("AAPL", 10, 100.0, date(2024, 1, 5)))
        state = store.load()
        assert len(state.transactions) == 1
        assert state.transactions[0].symbol == "AAPL"
        assert state.transactions[0].shares == pytest.approx(10.0)

    def test_positions_are_derived_from_the_log(self, store: hl.PortfolioStore) -> None:
        store.add_transaction(_buy("AAPL", 10, 100.0, date(2024, 1, 5)))
        store.add_transaction(_buy("AAPL", 10, 120.0, date(2024, 2, 5)))
        store.add_transaction(
            Transaction(symbol="AAPL", action="sell", shares=5, price=130.0, date=date(2024, 3, 5))
        )
        held = store.current_positions({"AAPL": 140.0})
        assert held["AAPL"].shares == pytest.approx(15.0)
        # FIFO: the 5 sold came from the $100 lot, leaving 5 @ $100 + 10 @ $120.
        assert held["AAPL"].cost_basis == pytest.approx(5 * 100 + 10 * 120)

    def test_cash_round_trips(self, store: hl.PortfolioStore) -> None:
        store.set_cash(2500.0)
        assert store.load().cash == pytest.approx(2500.0)

    def test_negative_cash_rejected(self, store: hl.PortfolioStore) -> None:
        with pytest.raises(ValueError, match="cash must be >= 0"):
            store.set_cash(-1.0)

    def test_watchlist_add_remove_and_dedupe(self, store: hl.PortfolioStore) -> None:
        store.add_to_watchlist("nvda")
        store.add_to_watchlist("NVDA")  # same symbol, different case
        assert store.load().watchlist == ["NVDA"]
        store.add_to_watchlist("GOOGL")
        assert store.load().watchlist == ["NVDA", "GOOGL"]
        store.remove_from_watchlist("NVDA")
        assert store.load().watchlist == ["GOOGL"]

    def test_clear_wipes_everything(self, store: hl.PortfolioStore) -> None:
        store.save(hl.example_state())
        assert store.load().transactions
        store.clear()
        state = store.load()
        assert state.transactions == []
        assert state.cash == 0.0
        assert state.watchlist == []

    def test_example_portfolio_loads(self, store: hl.PortfolioStore) -> None:
        store.save(hl.example_state())
        state = store.load()
        assert len(state.transactions) == len(hl.EXAMPLE_PORTFOLIO)
        assert state.cash > 0
        assert set(store.current_positions()) == {s for s, _, _, _ in hl.EXAMPLE_PORTFOLIO}

    def test_rejects_unknown_asset_type(self, store: hl.PortfolioStore) -> None:
        with pytest.raises(ValueError, match="asset_type must be one of"):
            store.add_transaction(_buy("AAPL", 1, 10, date(2024, 1, 1)), asset_type="crypto")

    def test_overselling_is_rejected_at_entry_by_both_backends(
        self, store: hl.PortfolioStore
    ) -> None:
        store.add_transaction(_buy("AAPL", 5, 100.0, date(2024, 1, 1)))
        with pytest.raises(ValueError, match="cannot sell"):
            store.add_transaction(
                Transaction(
                    symbol="AAPL", action="sell", shares=50, price=110.0, date=date(2024, 2, 1)
                )
            )
        # ...and the rejected row must not have been persisted.
        assert len(store.load().transactions) == 1

    def test_invalid_transaction_is_rejected_before_saving(self, store: hl.PortfolioStore) -> None:
        with pytest.raises(ValueError, match="price must be > 0"):
            store.add_transaction(_buy("AAPL", 5, 0.0, date(2024, 1, 1)))
        assert store.load().transactions == []


class TestSqliteBackendSpecifics:
    def test_derived_holdings_snapshot_is_rewritten_from_the_log(
        self, sqlite_store: hl.SqlitePortfolioStore, session: Session
    ) -> None:
        from quantpulse.storage.models import PortfolioHolding

        sqlite_store.add_transaction(_buy("AAPL", 10, 100.0, date(2024, 1, 5)))
        rows = session.query(PortfolioHolding).all()
        assert {r.symbol for r in rows} == {"AAPL"}
        assert rows[0].shares == pytest.approx(10.0)

        # Selling everything must leave no snapshot row behind.
        sqlite_store.add_transaction(
            Transaction(symbol="AAPL", action="sell", shares=10, price=110.0, date=date(2024, 2, 5))
        )
        assert session.query(PortfolioHolding).filter_by(asset_type="equity").count() == 0

    def test_cash_is_stored_as_a_pseudo_position(
        self, sqlite_store: hl.SqlitePortfolioStore, session: Session
    ) -> None:
        from quantpulse.storage.models import PortfolioHolding

        sqlite_store.set_cash(1234.0)
        cash_row = session.query(PortfolioHolding).filter_by(asset_type="cash").one()
        assert cash_row.shares == pytest.approx(1234.0)
        assert cash_row.purchase_date is None  # cash has no cost basis or holding period

    def test_asset_types_survive_a_round_trip(self, sqlite_store: hl.SqlitePortfolioStore) -> None:
        sqlite_store.add_transaction(_buy("SPY", 5, 400.0, date(2024, 1, 1)), asset_type="etf")
        assert sqlite_store.load().asset_type("SPY") == "etf"


class TestSessionBackendSpecifics:
    def test_state_lives_in_the_supplied_container(self) -> None:
        container: dict[str, object] = {}
        store = hl.SessionPortfolioStore(container)
        store.add_transaction(_buy("AAPL", 1, 10.0, date(2024, 1, 1)))
        assert "quantpulse_portfolio" in container

    def test_two_containers_are_isolated(self) -> None:
        # The whole point of ADR 4.5's session backend: one visitor's holdings
        # must never be visible to another.
        first = hl.SessionPortfolioStore({})
        second = hl.SessionPortfolioStore({})
        first.add_transaction(_buy("AAPL", 1, 10.0, date(2024, 1, 1)))
        assert second.load().transactions == []


class TestGetStore:
    def test_sqlite_requires_a_session_factory(self, monkeypatch) -> None:
        from quantpulse import config

        monkeypatch.setattr(config, "get_settings", config.get_settings)
        settings = config.get_settings()
        monkeypatch.setattr(settings, "portfolio_backend", "sqlite", raising=False)
        with pytest.raises(ValueError, match="requires a database session factory"):
            hl.get_store(None, {})

    def test_session_requires_a_container(self, monkeypatch) -> None:
        from quantpulse import config

        settings = config.get_settings()
        monkeypatch.setattr(settings, "portfolio_backend", "session", raising=False)
        with pytest.raises(ValueError, match="requires a session-state container"):
            hl.get_store(None, None)


class TestCsvRoundTrip:
    def test_round_trip_preserves_the_log(self) -> None:
        state = hl.example_state()
        parsed = hl.from_csv(hl.to_csv(state))
        assert len(parsed) == len(state.transactions)
        assert parsed[0].symbol == state.transactions[0].symbol
        assert parsed[0].shares == pytest.approx(state.transactions[0].shares)
        assert parsed[0].date == state.transactions[0].date

    def test_header_is_case_insensitive(self) -> None:
        csv_text = "Symbol,Action,Shares,Price,Date\nAAPL,BUY,10,100.5,2024-01-05\n"
        parsed = hl.from_csv(csv_text)
        assert parsed[0].symbol == "AAPL"
        assert parsed[0].action == "buy"

    def test_blank_rows_are_skipped(self) -> None:
        csv_text = "symbol,action,shares,price,date\nAAPL,buy,1,10,2024-01-01\n,,,,\n"
        assert len(hl.from_csv(csv_text)) == 1

    def test_missing_column_is_rejected_with_a_reason(self) -> None:
        with pytest.raises(ValueError, match="missing required column"):
            hl.from_csv("symbol,shares\nAAPL,10\n")

    def test_bad_row_names_the_line(self) -> None:
        csv_text = "symbol,action,shares,price,date\nAAPL,buy,notanumber,10,2024-01-01\n"
        with pytest.raises(ValueError, match="row 2 is invalid"):
            hl.from_csv(csv_text)

    def test_bad_action_is_rejected(self) -> None:
        csv_text = "symbol,action,shares,price,date\nAAPL,short,1,10,2024-01-01\n"
        with pytest.raises(ValueError, match="row 2 is invalid"):
            hl.from_csv(csv_text)

    def test_empty_csv_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            hl.from_csv("")


class TestSectorWeights:
    def test_weights_sum_to_one(self) -> None:
        weights = hl.sector_weights(
            {"AAPL": 100.0, "MSFT": 100.0, "XOM": 200.0},
            {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"},
        )
        assert weights["Tech"] == pytest.approx(0.5)
        assert weights["Energy"] == pytest.approx(0.5)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_unknown_sector_becomes_an_explicit_catch_all(self) -> None:
        # A pie that silently omits a third of the value is worse than one with
        # an honest "Unclassified" slice.
        weights = hl.sector_weights({"AAPL": 100.0, "SPY": 100.0}, {"AAPL": "Tech"})
        assert weights["Unclassified"] == pytest.approx(0.5)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_no_value_yields_no_weights(self) -> None:
        assert hl.sector_weights({}, {}) == {}
        assert hl.sector_weights({"AAPL": 0.0}, {"AAPL": "Tech"}) == {}


class TestReplaceTransactions:
    def test_replaces_log_but_keeps_cash_and_watchlist(self) -> None:
        store = hl.SessionPortfolioStore({})
        store.save(hl.example_state())
        before = store.load()
        new_log = [_buy("TSLA", 3, 200.0, date(2025, 1, 1))]
        updated = hl.replace_transactions(store, new_log)
        assert len(updated.transactions) == 1
        assert updated.cash == pytest.approx(before.cash)
        assert updated.watchlist == before.watchlist

    def test_cash_override_applies(self) -> None:
        store = hl.SessionPortfolioStore({})
        store.save(hl.example_state())
        updated = hl.replace_transactions(store, [], cash=42.0)
        assert updated.cash == pytest.approx(42.0)
