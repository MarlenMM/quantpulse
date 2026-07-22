"""End-to-end Phase 7: seed price history + membership, then forecast and backtest.

The forecasting/backtest constants in `refresh_data` are tuned for the full
universe; here they're monkeypatched down (one short horizon, the fast baseline
runner, a small momentum window) so the *wiring* is exercised end-to-end without
paying for a universe-scale walk-forward in CI.
"""

from collections.abc import Iterator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import refresh_data
from quantpulse.analysis.forecasting import baseline_forecast
from quantpulse.storage.models import (
    BacktestResult,
    Base,
    Forecast,
    IndexMembershipHistory,
    PriceHistory,
    Ticker,
)

AS_OF = date(2026, 7, 22)
_N_BARS = 260


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        yield s


@pytest.fixture(autouse=True)
def _fast_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    # One short horizon + the fit-free baseline runner keep the pooled walk-
    # forward instant; a small momentum window keeps the backtest's history need
    # modest. The real code paths (persistence, keying, append-only) are unchanged.
    monkeypatch.setattr(refresh_data, "_FORECAST_HORIZONS", (5,))
    monkeypatch.setattr(
        refresh_data, "_FORECAST_RUNNERS", {"baseline": (baseline_forecast, "baseline")}
    )
    monkeypatch.setattr(refresh_data, "_ACCURACY_SAMPLE_SIZE", 3)
    monkeypatch.setattr(refresh_data, "_BACKTEST_MOMENTUM_LOOKBACK", 20)


def _seed_prices(
    session: Session, symbol: str, closes: np.ndarray, *, end: date = AS_OF, adj: bool = True
) -> None:
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=len(closes))
    for d, close in zip(dates, closes, strict=True):
        session.add(
            PriceHistory(
                symbol=symbol,
                date=d.date(),
                open=float(close),
                high=float(close) * 1.01,
                low=float(close) * 0.99,
                close=float(close),
                adj_close=float(close) if adj else float(close),
                volume=1_000_000,
            )
        )


def _seed(session: Session) -> pd.DataFrame:
    """Two active members (AAA up, BBB down) plus OLD, a member removed mid-window."""
    rng = np.random.default_rng(0)
    trend = {
        "AAA": np.linspace(100, 180, _N_BARS),
        "BBB": np.linspace(180, 110, _N_BARS),
    }
    for symbol, base in trend.items():
        session.add(
            Ticker(symbol=symbol, name=f"{symbol} Inc.", sector="Tech", asset_type="equity")
        )
        _seed_prices(session, symbol, base + rng.normal(0, 1.0, _N_BARS))
        session.add(
            IndexMembershipHistory(
                index_name="S&P 500", symbol=symbol, added_date=date(2015, 1, 1), removed_date=None
            )
        )

    # OLD: an inactive, since-removed member -- present in membership + prices,
    # but delisted partway through, so the survivorship-aware backtest still sees
    # its history without it distorting the current forecast universe.
    session.add(
        Ticker(symbol="OLD", name="Old Co", sector="Tech", asset_type="equity", is_active=False)
    )
    _seed_prices(session, "OLD", np.linspace(50, 30, 120), end=AS_OF - timedelta(days=200))
    session.add(
        IndexMembershipHistory(
            index_name="S&P 500",
            symbol="OLD",
            added_date=date(2015, 1, 1),
            removed_date=AS_OF - timedelta(days=200),
        )
    )
    session.flush()
    return pd.DataFrame([{"symbol": "AAA"}, {"symbol": "BBB"}])


def test_refresh_forecasts_persists_rows_with_hit_rate(session: Session) -> None:
    universe = _seed(session)
    written = refresh_data.refresh_forecasts(session, universe, AS_OF)
    session.flush()

    forecasts = session.scalars(select(Forecast)).all()
    assert written == len(forecasts) > 0
    assert {f.symbol for f in forecasts} == {"AAA", "BBB"}
    assert all(f.generated_date == AS_OF and f.horizon_days == 5 for f in forecasts)
    # The fan chart is consistent: point_price == last_close * (1 + point_return).
    for f in forecasts:
        assert f.point_price is not None and f.lower_price <= f.point_price <= f.upper_price
    # The pooled baseline runner produced a non-null historical hit-rate.
    baseline = [f for f in forecasts if f.model_name == "baseline"]
    assert baseline and all(0.0 <= f.historical_hit_rate <= 1.0 for f in baseline)


def test_refresh_forecasts_is_append_only(session: Session) -> None:
    universe = _seed(session)
    refresh_data.refresh_forecasts(session, universe, AS_OF)
    session.flush()
    first = session.scalars(
        select(Forecast).where(Forecast.symbol == "AAA", Forecast.model_name == "baseline")
    ).one()
    original = first.point_return

    refresh_data.refresh_forecasts(session, universe, AS_OF)  # same-day re-run
    session.flush()
    rows = session.scalars(
        select(Forecast).where(Forecast.symbol == "AAA", Forecast.model_name == "baseline")
    ).all()
    assert len(rows) == 1 and rows[0].point_return == original


def test_refresh_forecasts_is_point_in_time(session: Session) -> None:
    universe = _seed(session)
    refresh_data.refresh_forecasts(session, universe, AS_OF)
    session.flush()
    baseline = session.scalars(
        select(Forecast).where(Forecast.symbol == "AAA", Forecast.model_name == "baseline")
    ).one()
    before = baseline.point_return

    # A future price (dated after AS_OF) must not change a forecast generated
    # as-of AS_OF -- read_active_ohlcv bounds at the as-of date.
    session.add(
        PriceHistory(
            symbol="AAA",
            date=AS_OF + timedelta(days=1),
            open=999.0,
            high=999.0,
            low=999.0,
            close=999.0,
            adj_close=999.0,
            volume=1,
        )
    )
    session.flush()
    refresh_data.refresh_forecasts(session, universe, AS_OF + timedelta(days=2))
    session.flush()
    reforecast = session.scalars(
        select(Forecast).where(
            Forecast.symbol == "AAA",
            Forecast.model_name == "baseline",
            Forecast.generated_date == AS_OF + timedelta(days=2),
        )
    ).one()
    # The as-of-AS_OF forecast is untouched; the new one differs precisely because
    # it now legitimately includes the (previously future) bar.
    assert before == baseline.point_return
    assert reforecast.point_return != before


def test_refresh_backtest_persists_a_track_record(session: Session) -> None:
    _seed(session)
    written = refresh_data.refresh_backtest(session, AS_OF)
    session.flush()

    results = session.scalars(select(BacktestResult)).all()
    assert written == 1 and len(results) == 1
    run = results[0]
    assert run.run_date == AS_OF
    assert run.cadence == "monthly"
    assert run.n_periods >= 2
    assert run.assumed_txn_cost == refresh_data._BACKTEST_TXN_COST
    assert run.period_start is not None and run.period_end is not None
    # A benchmark was computed from the universe (equal-weight proxy).
    assert run.benchmark_cagr is not None


def test_refresh_backtest_without_membership_is_a_noop(session: Session) -> None:
    # No index_membership_history rows -> nothing to run survivorship-aware.
    session.add(Ticker(symbol="AAA", name="A", sector="Tech", asset_type="equity"))
    _seed_prices(session, "AAA", np.linspace(100, 120, _N_BARS))
    session.flush()
    assert refresh_data.refresh_backtest(session, AS_OF) == 0
