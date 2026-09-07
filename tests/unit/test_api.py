"""Route tests for the FastAPI read API (ADR 4.1's stretch goal).

Runs against a real temporary SQLite database rather than mocked readers: the
whole point of the API layer is that it faithfully translates what
`storage.persistence` returns, so mocking the readers would test only that the
mock was wired up.
"""

import json
import logging
from collections.abc import Iterator
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.analysis import risk
from quantpulse.api.main import app, db_session
from quantpulse.storage import persistence
from quantpulse.storage.models import (
    AnalystConsensus,
    BacktestResult,
    Base,
    CompositeScore,
    Forecast,
    MarketRegime,
    NewsEvent,
    PatternSignal,
    PriceHistory,
    Ticker,
)

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


def _seed(session: Session) -> None:
    session.add(Ticker(symbol="AAPL", name="Apple Inc.", sector="Tech", asset_type="equity"))
    session.add(Ticker(symbol="XOM", name="Exxon Mobil", sector="Energy", asset_type="equity"))
    for day, rating, score in ((YESTERDAY, "hold", 50.0), (TODAY, "buy", 75.0)):
        session.add(
            CompositeScore(
                symbol="AAPL",
                date=day,
                profile="balanced",
                composite_score=score,
                technical_score=80.0,
                fundamental_score=None,
                rating=rating,
                percentile_rank=score,
                data_confidence=90.0,
                # Absolute mode re-scores from the stored *raw* category values;
                # without them it correctly declines, so they have to be seeded
                # for that path to be exercised at all.
                technical_raw=0.8,
                fundamental_raw=0.6,
            )
        )
        session.add(
            CompositeScore(
                symbol="XOM",
                date=day,
                profile="balanced",
                composite_score=40.0,
                technical_score=30.0,
                technical_raw=0.3,
                fundamental_raw=0.2,
                rating="sell",
                percentile_rank=20.0,
                data_confidence=55.0,
            )
        )
    for i in range(6):
        session.add(
            PriceHistory(
                symbol="AAPL",
                date=TODAY - timedelta(days=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0 + i,
                adj_close=100.0 + i,
                volume=1000,
            )
        )
    session.add(
        Forecast(
            symbol="AAPL",
            generated_date=TODAY,
            horizon_days=5,
            model_name="gbr",
            point_return=0.02,
            point_price=103.0,
            lower_price=99.0,
            upper_price=107.0,
            historical_hit_rate=0.55,
        )
    )
    session.add(
        PatternSignal(
            symbol="AAPL",
            date=TODAY,
            pattern_type="cup_and_handle",
            direction="bullish",
            confidence=0.8,
        )
    )
    session.add(
        AnalystConsensus(
            symbol="AAPL",
            as_of_date=TODAY,
            strong_buy=5,
            buy=3,
            hold=1,
            sell=0,
            strong_sell=0,
            mean_price_target=120.0,
        )
    )
    session.add(MarketRegime(date=TODAY, vix_level=18.0, regime_score=62.0, regime_label="risk_on"))
    session.add(
        NewsEvent(
            article_id="t3",
            tier=3,
            title="Fed holds",
            published_at=datetime.now(),
            event_type="macro",
            sentiment_score=0.1,
        )
    )
    session.add(
        NewsEvent(
            article_id="t1",
            tier=1,
            title="Apple ships",
            published_at=datetime.now(),
            matched_symbols=["AAPL"],
            sentiment_score=0.5,
        )
    )
    session.add(
        BacktestResult(
            run_date=TODAY,
            cadence="monthly",
            n_periods=40,
            sharpe=0.8,
            sharpe_ci_low=0.2,
            sharpe_ci_high=1.3,
            ci_confidence_level=0.9,
            assumed_txn_cost=0.001,
            win_rate=0.6,
            payoff_ratio=1.5,
            # Deliberately contradictory: healthy in absolute terms, no edge at
            # all against the benchmark. A Kelly built from the first pair sizes
            # a confident bet; one built from the second declines. That is what
            # makes `TestKellyIsSizedOnExcessReturn` able to tell them apart.
            excess_win_rate=0.25,
            excess_payoff_ratio=0.5,
            signal_name="momentum_category",
        )
    )
    session.commit()


def _client(tmp_path, seed: bool = True, extra=None) -> Iterator[TestClient]:
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    if seed:
        with factory() as s:
            _seed(s)
    if extra is not None:
        with factory() as s:
            extra(s)

    def override() -> Iterator[Session]:
        with factory() as s:
            yield s

    app.dependency_overrides[db_session] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    yield from _client(tmp_path)


@pytest.fixture
def empty_client(tmp_path) -> Iterator[TestClient]:
    yield from _client(tmp_path, seed=False)


@pytest.fixture
def short_interest_client(tmp_path) -> Iterator[TestClient]:
    """A seeded client that also has a short-interest reading for AAPL.

    Kept separate from `client` so the "no reading stored" case stays a real
    assertion instead of being seeded out of existence.
    """
    yield from _client(tmp_path, extra=_seed_short_interest)


def _seed_short_interest(session: Session) -> None:
    from quantpulse.storage.models import ShortInterest

    session.add(
        ShortInterest(symbol="AAPL", as_of_date=TODAY, pct_float_short=22.0, days_to_cover=7.5)
    )
    session.commit()


class TestHealth:
    def test_reports_data_present_and_freshness(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["has_data"] is True
        assert body["freshness"]["composite_scores"] == TODAY.isoformat()

    def test_never_populated_dataset_is_null_not_missing(self, client: TestClient) -> None:
        # The client must be able to tell "stale" from "never ran".
        freshness = client.get("/api/health").json()["freshness"]
        assert "fundamentals" in freshness
        assert freshness["fundamentals"] is None

    def test_empty_database_reports_no_data_rather_than_erroring(
        self, empty_client: TestClient
    ) -> None:
        body = empty_client.get("/api/health").json()
        assert body["has_data"] is False


class TestScreener:
    def test_returns_ranked_rows_with_context(self, client: TestClient) -> None:
        body = client.get("/api/screener").json()
        assert body["as_of"] == TODAY.isoformat()
        assert body["profile"] == "balanced"
        assert body["count"] == 2
        assert [row["symbol"] for row in body["rows"]] == ["AAPL", "XOM"]

    def test_rating_mode_travels_with_the_rows(self, client: TestClient) -> None:
        # So a client cannot present a relative ranking as an absolute judgment
        # just because it forgot to ask which scheme produced it (Section 22).
        assert client.get("/api/screener").json()["rating_mode"] == "relative"

    def test_missing_subscore_serializes_as_null_not_zero(self, client: TestClient) -> None:
        row = next(r for r in client.get("/api/screener").json()["rows"] if r["symbol"] == "AAPL")
        assert row["fundamental_score"] is None
        assert row["technical_score"] == pytest.approx(80.0)

    def test_response_is_strictly_valid_json(self, client: TestClient) -> None:
        # A pandas NaN would serialize as a bare `NaN` literal, which is not
        # valid JSON and which JSON.parse rejects outright.
        raw = client.get("/api/screener").text
        assert "NaN" not in raw
        json.loads(raw)

    def test_unknown_profile_is_empty_not_an_error(self, client: TestClient) -> None:
        body = client.get("/api/screener", params={"profile": "growth"}).json()
        assert body["count"] == 0
        assert body["as_of"] is None

    def test_empty_database_returns_an_empty_table(self, empty_client: TestClient) -> None:
        body = empty_client.get("/api/screener").json()
        assert body["count"] == 0

    def test_rating_changes(self, client: TestClient) -> None:
        changes = client.get("/api/screener/changes").json()
        assert [c["symbol"] for c in changes] == ["AAPL"]
        assert changes[0]["previous_rating"] == "hold"
        assert changes[0]["rating"] == "buy"

    def test_limit_is_validated(self, client: TestClient) -> None:
        assert client.get("/api/screener/changes", params={"limit": 0}).status_code == 422


class TestStockDetail:
    def test_bundles_everything_in_one_round_trip(self, client: TestClient) -> None:
        body = client.get("/api/stocks/AAPL").json()
        assert body["symbol"] == "AAPL"
        assert body["summary"]["name"] == "Apple Inc."
        assert body["score"]["rating"] == "buy"
        assert len(body["prices"]) == 6
        assert len(body["forecasts"]) == 1
        assert len(body["patterns"]) == 1
        assert body["analyst_consensus"]["strong_buy"] == 5
        assert [n["title"] for n in body["news"]] == ["Apple ships"]

    def test_forecast_carries_its_own_track_record(self, client: TestClient) -> None:
        # Section 7.6: a forecast must be shown next to its own hit-rate, so it
        # ships in the same payload rather than behind a second request.
        forecast = client.get("/api/stocks/AAPL").json()["forecasts"][0]
        assert forecast["historical_hit_rate"] == pytest.approx(0.55)

    def test_symbol_is_case_insensitive(self, client: TestClient) -> None:
        assert client.get("/api/stocks/aapl").json()["symbol"] == "AAPL"

    def test_unknown_symbol_is_404(self, client: TestClient) -> None:
        response = client.get("/api/stocks/NOPE")
        assert response.status_code == 404
        assert "NOPE" in response.json()["detail"]

    def test_known_symbol_without_analysis_is_not_404(self, client: TestClient) -> None:
        # "We track this but haven't computed it" and "this doesn't exist" are
        # different answers; the client must be able to tell which it got.
        body = client.get("/api/stocks/XOM").json()
        assert body["symbol"] == "XOM"
        assert body["prices"] == []
        assert body["forecasts"] == []
        assert body["analyst_consensus"] is None
        assert body["score"] is not None  # XOM is scored, just not priced


class TestMarketRoutes:
    def test_regime(self, client: TestClient) -> None:
        points = client.get("/api/regime").json()
        assert points[-1]["regime_label"] == "risk_on"

    def test_market_news_is_tier_2_and_3_only(self, client: TestClient) -> None:
        titles = [item["title"] for item in client.get("/api/news").json()]
        assert titles == ["Fed holds"]

    def test_backtest_includes_confidence_bounds(self, client: TestClient) -> None:
        runs = client.get("/api/backtest").json()
        assert runs[0]["sharpe"] == pytest.approx(0.8)
        assert runs[0]["sharpe_ci_low"] == pytest.approx(0.2)
        assert runs[0]["ci_confidence_level"] == pytest.approx(0.9)

    def test_absent_confidence_bounds_stay_null(self, client: TestClient) -> None:
        # A run too short to bootstrap stores nulls rather than a fake interval;
        # the client must be able to tell that from a zero-width one.
        runs = client.get("/api/backtest").json()
        assert runs[0]["cagr_ci_low"] is None

    def test_prices(self, client: TestClient) -> None:
        bars = client.get("/api/prices/AAPL").json()
        assert len(bars) == 6
        assert bars[0]["date"] < bars[-1]["date"]  # oldest first

    def test_price_range(self, client: TestClient) -> None:
        body = client.get("/api/prices/AAPL/range", params={"days": 30}).json()
        assert body["symbol"] == "AAPL"
        assert body["change"] is not None

    def test_price_range_for_unpriced_symbol_is_null_not_zero(self, client: TestClient) -> None:
        body = client.get("/api/prices/XOM/range").json()
        assert body["change"] is None


class TestReferenceRoutes:
    def test_glossary_serves_the_shared_terms(self, client: TestClient) -> None:
        from quantpulse.glossary import TERMS

        entries = client.get("/api/glossary").json()
        assert len(entries) == len(TERMS)
        sharpe = next(e for e in entries if e["term"] == "Sharpe ratio")
        assert sharpe["category"] == "Risk"
        assert "volatility" in sharpe["definition"]

    def test_universe(self, client: TestClient) -> None:
        symbols = {row["symbol"] for row in client.get("/api/universe").json()}
        assert symbols == {"AAPL", "XOM"}


class TestApiContract:
    def test_openapi_schema_is_generated(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "QuantPulse API"
        assert "/api/screener" in schema["paths"]

    def test_api_is_read_only(self, client: TestClient) -> None:
        # Deliberate: portfolio state is per-user and ADR 4.5 splits it between
        # a browser session and a local file, neither of which maps onto an
        # unauthenticated REST API (Section 18). No write route should exist.
        schema = client.get("/openapi.json").json()
        methods = {
            method.upper()
            for path in schema["paths"].values()
            for method in path
            if method != "parameters"
        }
        assert methods == {"GET"}

    def test_cors_is_an_explicit_allow_list_not_a_wildcard(self) -> None:
        from quantpulse.api.main import _DEV_ORIGINS

        assert "*" not in _DEV_ORIGINS
        assert all(origin.startswith("http://") for origin in _DEV_ORIGINS)


class TestParityWithStreamlit:
    """Sections that existed only in the Streamlit app until now.

    The React client rendered 7 of the 12 sections its sibling did. Short
    interest is the one that mattered most: Section 24 explicitly requires both
    readings be surfaced and never collapsed into a single directional verdict,
    so a front end that omits it entirely is a spec gap, not a styling choice.
    Each of these is computed by the very same analysis function the Streamlit
    page calls, so the two front ends cannot disagree about a number.
    """

    def test_short_interest_reports_both_readings_never_a_verdict(
        self, short_interest_client: TestClient
    ) -> None:
        payload = short_interest_client.get("/api/stocks/AAPL").json()["short_interest"]
        assert payload is not None
        assert payload["pct_float_short"] == 22.0
        assert payload["days_to_cover"] == 7.5
        # A flag, deliberately not a direction: the same figure supports both a
        # bearish reading and a squeeze setup (Section 24).
        assert payload["elevated"] is True
        assert "rating" not in payload and "signal" not in payload

    def test_absent_short_interest_is_null_rather_than_zeroed(self, client: TestClient) -> None:
        assert client.get("/api/stocks/AAPL").json()["short_interest"] is None

    def test_risk_block_declines_per_estimator_on_thin_history(self, client: TestClient) -> None:
        # The seed has six price bars, so every estimator with a data floor
        # should abstain rather than emit a confident-looking number.
        risk_block = client.get("/api/stocks/AAPL").json()["risk"]
        assert risk_block is not None
        assert risk_block["sharpe"] is None
        assert risk_block["sortino"] is None
        assert risk_block["value_at_risk"] is None
        # ...and the client is told the threshold so it can explain the absence.
        assert risk_block["ratio_min_observations"] > risk_block["n_observations"]

    def test_monte_carlo_is_absent_without_enough_history(self, client: TestClient) -> None:
        assert client.get("/api/stocks/AAPL").json()["monte_carlo"] is None

    def test_macro_overlay_is_targeted_not_universal(self, client: TestClient) -> None:
        # Section 28: a sector with no configured commodity sensitivity gets
        # nothing at all, never a small meaningless nudge. AAPL is seeded as
        # "Tech", which is not a configured sector name.
        assert client.get("/api/stocks/AAPL").json()["macro_overlay"] is None

    def test_sector_rotation_is_served(self, client: TestClient) -> None:
        response = client.get("/api/sectors/rotation")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_sector_rotation_on_an_empty_database_is_empty_not_an_error(
        self, empty_client: TestClient
    ) -> None:
        response = empty_client.get("/api/sectors/rotation")
        assert response.status_code == 200
        assert response.json() == []

    def test_every_new_route_is_still_read_only(self, client: TestClient) -> None:
        # The read-only guarantee is asserted over the whole OpenAPI schema
        # elsewhere; this pins that the routes added for parity did not quietly
        # introduce the first write path.
        paths = client.get("/openapi.json").json()["paths"]
        methods = {m.upper() for spec in paths.values() for m in spec}
        assert methods == {"GET"}


class TestParitySurface:
    """Endpoints added so the React client can reach what Streamlit already could.

    Each of these existed as a capability with no route: the investor-profile
    presets (the API accepted a `profile` but never told a client which ones
    exist), absolute-mode ratings, and the Kelly position size -- whose
    `payoff_ratio` was a stored column the API simply never sent.
    """

    def test_profiles_lists_every_preset_with_its_weights(self, client: TestClient) -> None:
        profiles = client.get("/api/profiles").json()
        names = [p["name"] for p in profiles]
        assert names[0] == "balanced", "balanced is the onboarding default and must lead"
        assert set(names) == {
            "balanced",
            "value",
            "growth",
            "income",
            "momentum_active",
            "conservative",
        }
        for profile in profiles:
            assert profile["weights"], f"{profile['name']} carries no weights"
            assert abs(sum(profile["weights"].values()) - 1.0) < 1e-9

    def test_only_the_two_rescoring_profiles_are_flagged(self, client: TestClient) -> None:
        """The flag a client needs to know whether it may re-weight locally.

        Income and conservative genuinely re-score a category, so their rankings
        must be *fetched*; the other four differ by weights alone and can be
        applied to rows already in memory. A client that treated all six alike
        would show balanced sub-scores under an income label.
        """
        flags = {p["name"]: p["rescores"] for p in client.get("/api/profiles").json()}
        assert flags == {
            "balanced": False,
            "value": False,
            "growth": False,
            "income": True,
            "momentum_active": False,
            "conservative": True,
        }

    def test_absolute_ratings_are_keyed_by_real_symbols(self, client: TestClient) -> None:
        """`build_composite` returns a positional index, so the symbol is a column.

        Reading it from `iterrows()`'s index instead yielded "0", "1", "2" --
        which type-checks, serializes cleanly, and is wrong. Only looking at the
        response against real data caught it.
        """
        payload = client.get("/api/screener/absolute").json()
        assert payload["available"] is True
        assert payload["rating_mode"] == "absolute"
        symbols = {row["symbol"] for row in payload["rows"]}
        assert symbols == {"AAPL", "XOM"}

    def test_absolute_mode_declines_when_raw_values_are_absent(
        self, empty_client: TestClient
    ) -> None:
        """An absolute rating cannot be recovered from a percentile.

        Saying so is the honest answer; returning relative ratings under an
        "absolute" label would be the exact mislabelling the mode exists to stop.
        """
        payload = empty_client.get("/api/screener/absolute").json()
        assert payload["available"] is False
        assert payload["rows"] == []

    def test_backtest_carries_payoff_ratio_and_a_server_computed_kelly(
        self, client: TestClient
    ) -> None:
        """Kelly is derived server-side so both front ends size the bet identically.

        `payoff_ratio` was stored on every run and exposed by no route, which is
        why the React Track Record had no position-sizing section at all.

        The arithmetic below moved once, deliberately: the fraction is now built
        from the **excess** metrics rather than the absolute ones. On p=0.6,
        b=1.5 it was a confident quarter-Kelly 8.3%; on this run's edge over the
        benchmark (p=0.25, b=0.5) full Kelly is negative, which in a long-only
        tool is reported as 0.0 -- "do not take this bet".
        """
        run = client.get("/api/backtest").json()[0]
        assert run["payoff_ratio"] == pytest.approx(1.5), "the absolute figure is still served"
        assert run["kelly_fraction"] == pytest.approx(0.0)

    def test_every_new_route_is_still_a_GET(self, client: TestClient) -> None:
        """The read-only guarantee has to survive each addition, not just the first."""
        paths = client.get("/openapi.json").json()["paths"]
        for path, methods in paths.items():
            assert set(methods) <= {"get"}, f"{path} exposes a non-GET method"


class TestMarketIndexIsNotATradableSymbol:
    """`^GSPC` is stored in `price_history`, and must be a price series only.

    It needs a `tickers` row because `price_history.symbol` is a foreign key.
    That row is the one thing standing between "a benchmark the beta regression
    can read" and "a 504th stock in the universe" -- so every surface that
    enumerates symbols is checked here, in one place, rather than trusting each
    reader's own filter to keep holding.
    """

    @pytest.fixture
    def index_client(self, tmp_path) -> Iterator[TestClient]:
        def _seed_index(session: Session) -> None:
            persistence.upsert_benchmark_ticker(
                session, symbol=risk.MARKET_INDEX_SYMBOL, name=risk.MARKET_INDEX_NAME
            )
            for offset in range(120):
                level = 4000.0 * (1.0005**offset)
                session.add(
                    PriceHistory(
                        symbol=risk.MARKET_INDEX_SYMBOL,
                        date=TODAY - timedelta(days=120 - offset),
                        open=level,
                        high=level,
                        low=level,
                        close=level,
                        adj_close=level,
                        volume=0,
                    )
                )
            session.commit()

        yield from _client(tmp_path, extra=_seed_index)

    def test_it_is_not_in_the_universe(self, index_client: TestClient) -> None:
        symbols = {row["symbol"] for row in index_client.get("/api/universe").json()}
        assert risk.MARKET_INDEX_SYMBOL not in symbols
        assert symbols, "the fixture's real tickers vanished too -- the filter is too broad"

    def test_it_is_not_in_the_screener(self, index_client: TestClient) -> None:
        body = index_client.get("/api/screener").json()
        assert risk.MARKET_INDEX_SYMBOL not in {row["symbol"] for row in body["rows"]}

    def test_it_has_no_stock_page(self, index_client: TestClient) -> None:
        """404, not a company profile of an index with every field empty."""
        assert index_client.get(f"/api/stocks/{risk.MARKET_INDEX_SYMBOL}").status_code == 404
        # The guard is about this row's asset_type, not about the symbol being
        # unusual -- an ordinary ticker in the same fixture still resolves.
        assert index_client.get("/api/stocks/AAPL").status_code == 200

    def test_it_does_not_become_a_sector(self, index_client: TestClient) -> None:
        rotation = index_client.get("/api/sectors/rotation").json()
        assert all(row["sector"] for row in rotation)


class TestKellyIsSizedOnExcessReturn:
    """The position size must come from the edge over the benchmark, not raw return.

    On absolute returns the Track Record page recommended betting 6.2% of
    capital on a run whose CAGR trailed its own buy-and-hold benchmark: it was
    measuring the market's return and reporting it as the strategy's edge. The
    server computes the fraction (one implementation for both front ends), so
    the assertion belongs here.
    """

    def test_the_fraction_uses_the_excess_metrics(self, client: TestClient) -> None:
        run = client.get("/api/backtest").json()[0]
        from quantpulse.portfolio.optimization import kelly_position_fraction

        on_excess = kelly_position_fraction(run["excess_win_rate"], run["excess_payoff_ratio"])
        on_absolute = kelly_position_fraction(run["win_rate"], run["payoff_ratio"])
        assert on_absolute is not None and on_absolute > 0, (
            "the fixture must recommend a bet on the old basis, or this proves nothing"
        )
        assert run["kelly_fraction"] == pytest.approx(on_excess)
        assert run["kelly_fraction"] != pytest.approx(on_absolute)

    def test_a_run_with_no_edge_is_sized_at_zero_or_less(self, client: TestClient) -> None:
        run = client.get("/api/backtest").json()[0]
        assert run["kelly_fraction"] is not None
        assert run["kelly_fraction"] <= 0, (
            "a strategy beating its benchmark in a quarter of periods at a payoff below "
            "1.0 was handed a positive position size"
        )

    def test_the_run_says_what_it_ranked(self, client: TestClient) -> None:
        """The page cannot name the signal if the row does not carry it."""
        run = client.get("/api/backtest").json()[0]
        assert run["signal_name"] == "momentum_category"
        assert run["composite_history_days"] >= 0


class TestForecastGradingReachesTheClient:
    """React splits its forecast table on `is_graded`, so the server must send it.

    The rule lives in `forecasting.is_graded` and is sent rather than re-derived
    in TypeScript: it decides how prominently a number is displayed, and a null
    check written separately on each surface is how two front ends come to grade
    the same forecast differently.
    """

    def test_a_graded_row_is_flagged_graded(self, client: TestClient) -> None:
        rows = client.get("/api/stocks/AAPL").json()["forecasts"]
        graded = [r for r in rows if r["historical_hit_rate"] is not None]
        assert graded, "the fixture has no graded forecast to assert on"
        assert all(r["is_graded"] for r in graded)

    def test_an_ungraded_row_is_flagged_ungraded(self, tmp_path) -> None:
        def _seed_ungraded(session: Session) -> None:
            session.add(
                Forecast(
                    symbol="AAPL",
                    generated_date=TODAY,
                    horizon_days=252,
                    model_name="baseline",
                    point_return=0.822,
                    point_price=379.77,
                    lower_price=180.53,
                    upper_price=798.91,
                    historical_hit_rate=None,
                    baseline_hit_rate=None,
                    hit_rate_windows=None,
                )
            )
            session.commit()

        for c in _client(tmp_path, extra=_seed_ungraded):
            rows = c.get("/api/stocks/AAPL").json()["forecasts"]
            long_horizon = [r for r in rows if r["horizon_days"] == 252]
            assert long_horizon, "the fixture stored no 252-day forecast"
            assert long_horizon[0]["is_graded"] is False, (
                "a one-year forecast with no measured accuracy was sent to the client "
                "flagged as graded -- React would render it in the default table"
            )
            # The point of the split: it is also the biggest number in the set.
            assert long_horizon[0]["point_return"] > max(
                r["point_return"] for r in rows if r["is_graded"]
            )
            break


class TestRegimeCoverageReachesTheClient:
    """React prints the coverage sentence, so the server must compose and send it.

    Both front ends described the index as "built from four inputs" while the
    yield-curve spread had been null since the project began. The sentence is
    server-side for the same reason `beta_benchmark` is: it is a claim about how
    the number beside it was computed.
    """

    def test_each_point_carries_its_own_note(self, client: TestClient) -> None:
        points = client.get("/api/regime").json()
        assert points, "the fixture stored no regime rows"
        assert all(p["coverage_note"] for p in points)

    def test_a_missing_input_is_reported_as_missing(self, tmp_path) -> None:
        def _seed_partial(session: Session) -> None:
            session.add(
                MarketRegime(
                    date=TODAY - timedelta(days=2),
                    vix_level=14.5,
                    breadth_pct_above_200dma=66.4,
                    macro_news_tone=-0.5,
                    yield_curve_spread=None,
                    regime_score=73.6,
                    regime_label="risk_on",
                )
            )
            session.commit()

        for c in _client(tmp_path, extra=_seed_partial):
            point = next(p for p in c.get("/api/regime").json() if p["yield_curve_spread"] is None)
            assert "Three of its four inputs are live" in point["coverage_note"]
            assert "the yield-curve spread is missing" in point["coverage_note"]
            break

    def test_the_note_is_per_row_not_per_series(self, tmp_path) -> None:
        """A key added tomorrow makes tomorrow a four-input reading, not the history.

        Rewriting older points to match today's coverage would be exactly the
        retroactive edit the append-only score storage exists to prevent.
        """

        def _seed_mixed(session: Session) -> None:
            session.add(
                MarketRegime(
                    date=TODAY - timedelta(days=3),
                    vix_level=15.0,
                    breadth_pct_above_200dma=60.0,
                    macro_news_tone=-0.2,
                    yield_curve_spread=None,
                    regime_score=70.0,
                    regime_label="risk_on",
                )
            )
            session.add(
                MarketRegime(
                    date=TODAY - timedelta(days=2),
                    vix_level=15.0,
                    breadth_pct_above_200dma=60.0,
                    macro_news_tone=-0.2,
                    yield_curve_spread=0.4,
                    regime_score=71.0,
                    regime_label="risk_on",
                )
            )
            session.commit()

        for c in _client(tmp_path, extra=_seed_mixed):
            notes = {
                p["date"]: p["coverage_note"]
                for p in c.get("/api/regime").json()
                if p["date"]
                in {
                    str(TODAY - timedelta(days=3)),
                    str(TODAY - timedelta(days=2)),
                }
            }
            assert len(notes) == 2
            assert len(set(notes.values())) == 2, (
                "both points got the same coverage note, so it is describing the series "
                "rather than the row"
            )
            break


class TestForwardTestEndpoint:
    """The forward test's own endpoint (Sections 10, 32)."""

    def test_an_untraded_record_is_an_empty_record_not_a_404(self, client) -> None:
        """The normal state until someone sets the two Alpaca secrets. "We track
        this and it has not started" and "this endpoint does not exist" are
        different answers, and the page needs the first."""
        response = client.get("/api/forward-test")
        assert response.status_code == 200
        payload = response.json()
        assert payload["n_snapshots"] == 0
        assert payload["total_return"] is None
        assert payload["is_meaningful"] is False
        assert payload["points"] == []

    def test_the_threshold_is_served_so_the_client_needs_no_copy_of_it(self, client) -> None:
        """Otherwise "8 of 20 days" needs the 20 in TypeScript as well, and the
        two can disagree the moment one is changed."""
        from quantpulse.execution import record

        payload = client.get("/api/forward-test").json()
        assert payload["min_days_for_meaning"] == record.MIN_FORWARD_TEST_DAYS

    @pytest.fixture
    def traded_client(self, tmp_path) -> Iterator[TestClient]:
        def _seed_forward(session: Session) -> None:
            for offset, equity, close in ((1, 100_000.0, 5000.0), (0, 102_000.0, 5050.0)):
                persistence.upsert_paper_snapshot(
                    session,
                    {
                        "run_date": date(2026, 9, 7) - timedelta(days=offset),
                        "equity": equity,
                        "cash": 0.0,
                        "positions_held": 20,
                        "benchmark_close": close,
                        "rebalanced": offset == 1,
                        "orders_submitted": 21 if offset == 1 else 0,
                        "orders_rejected": 0,
                        "turnover": 1.0 if offset == 1 else None,
                        "signal_name": "composite_rating",
                        "profile": "balanced",
                        "target_positions": 20,
                    },
                )
            session.commit()

        yield from _client(tmp_path, extra=_seed_forward)

    def test_it_serves_the_curve_and_the_summary_together(self, traded_client) -> None:
        payload = traded_client.get("/api/forward-test").json()
        assert payload["n_snapshots"] == 2
        assert payload["total_return"] == pytest.approx(0.02)
        assert payload["benchmark_total_return"] == pytest.approx(0.01)
        assert payload["signal_name"] == "composite_rating"
        assert payload["rebalances"] == 1
        assert [p["run_date"] for p in payload["points"]] == ["2026-09-06", "2026-09-07"]

    def test_nothing_it_serves_is_annualised(self, traded_client) -> None:
        """A record that starts one day long would spend months turning a good
        week into a headline CAGR."""
        payload = traded_client.get("/api/forward-test").json()
        assert not any("cagr" in key.lower() or "annual" in key.lower() for key in payload)


class TestRatingExplanation:
    """Section 10's "why is this one rated Buy?" on the stock endpoint."""

    @pytest.fixture
    def explained_client(self, tmp_path) -> Iterator[TestClient]:
        """A scored row whose stored composite matches its own sub-scores.

        The shared fixture's does not -- it stores `technical_score=80` with
        every other category null beside a `composite_score` of 75, which no run
        of `build_composite` could produce. That inconsistency is what surfaced
        the guard tested at the bottom of this class; the happy path needs a
        coherent row.
        """

        def _seed_consistent(session: Session) -> None:
            from sqlalchemy import delete as sa_delete

            from quantpulse.storage.models import CompositeScore

            # The shared seed already wrote an AAPL row for this date; replace
            # it rather than colliding with its primary key.
            session.execute(sa_delete(CompositeScore).where(CompositeScore.symbol == "AAPL"))
            session.add(
                CompositeScore(
                    symbol="AAPL",
                    date=TODAY,
                    profile="balanced",
                    fundamental_score=90.0,
                    technical_score=70.0,
                    analyst_score=60.0,
                    sentiment_score=20.0,
                    momentum_score=55.0,
                    smart_money_score=50.0,
                    # 0.25*90 + 0.20*70 + 0.10*60 + 0.10*20 + 0.15*55 + 0.10*50,
                    # over a covered weight of 0.90 (industry/macro absent).
                    composite_score=(22.5 + 14.0 + 6.0 + 2.0 + 8.25 + 5.0) / 0.90,
                    percentile_rank=88.0,
                    rating="buy",
                    data_confidence=90.0,
                )
            )
            session.commit()

        yield from _client(tmp_path, extra=_seed_consistent)

    def test_a_scored_stock_carries_its_explanation(self, explained_client) -> None:
        payload = explained_client.get("/api/stocks/AAPL").json()
        assert payload["explanation"] is not None
        assert payload["explanation"]["sentence"].startswith("Rated Buy")

    def test_the_contributions_add_up_to_the_composite(self, explained_client) -> None:
        """The identity the decomposition rests on, asserted through the wire
        rather than only in the analysis module -- a serialisation that dropped
        or reordered a field would still look like a valid explanation."""
        payload = explained_client.get("/api/stocks/AAPL").json()
        explanation = payload["explanation"]
        total = sum(c["contribution"] for c in explanation["contributions"])
        assert total == pytest.approx(
            payload["score"]["composite_score"] - explanation["baseline"], abs=1e-6
        )

    def test_effective_weights_sum_to_one(self, explained_client) -> None:
        explanation = explained_client.get("/api/stocks/AAPL").json()["explanation"]
        weights = [c["effective_weight"] for c in explanation["contributions"]]
        assert sum(weights) == pytest.approx(1.0)

    def test_missing_categories_are_named_on_the_wire(self, explained_client) -> None:
        """484 of 503 names had no industry/macro reading. A client that could
        not tell "absent" from "average" would draw the second."""
        explanation = explained_client.get("/api/stocks/AAPL").json()["explanation"]
        assert explanation["missing"] == ["industry_macro"]
        assert all(
            c["category"] not in explanation["missing"] for c in explanation["contributions"]
        )

    def test_a_row_whose_composite_disagrees_with_its_parts_is_not_explained(
        self, client, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Refusing to explain a number that is not the one on the page.

        The shared fixture stores a composite of 75 beside sub-scores implying
        80. A confident sentence accounting for 80, printed next to a displayed
        75, is worse than no sentence -- so the explanation is withheld and the
        disagreement logged.
        """
        with caplog.at_level(logging.WARNING):
            payload = client.get("/api/stocks/AAPL").json()
        assert payload["score"] is not None, "the row itself is still served"
        assert payload["explanation"] is None
        assert "stored composite" in caplog.text

    def test_an_unscored_stock_has_no_explanation_rather_than_an_empty_one(
        self, empty_client
    ) -> None:
        """ "We track this and have not scored it" must not render as a
        confident-looking sentence about nothing."""
        response = empty_client.get("/api/stocks/AAPL")
        if response.status_code == 200:
            assert response.json()["explanation"] is None


class TestResponseModelsAreComplete:
    """No response model should carry an unresolved forward reference.

    `StockDetail.explanation` was first declared as `RatingExplanation | None`
    with `RatingExplanation` defined 170 lines *below* it, leaving
    `__pydantic_complete__` False until something forced a rebuild. Pydantic
    resolves the annotation lazily on first use, so the field was in fact always
    serialized correctly — that was checked directly against the old ordering
    rather than assumed.

    The guard is here anyway, because "it happens to work because something else
    triggers a rebuild first" is a property of import order rather than of this
    module, and import order is exactly what changes when a new caller appears.
    Declaring the classes in dependency order costs nothing and removes the
    question.
    """

    def test_every_schema_resolved_its_annotations(self) -> None:
        """Checked in a **subprocess importing only `schemas`**.

        Importing `api.main` first rebuilds the models as a side effect of route
        registration, so an in-process assertion passes or fails depending on
        which test ran before it -- it caught the broken ordering when it
        happened to run first and missed it otherwise. A guard that depends on
        test ordering is not a guard.
        """
        import subprocess
        import sys
        import textwrap

        program = textwrap.dedent(
            """
            import inspect
            from pydantic import BaseModel
            from quantpulse.api import schemas

            bad = [
                name
                for name, obj in inspect.getmembers(schemas, inspect.isclass)
                if issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == schemas.__name__
                and not obj.__pydantic_complete__
            ]
            print(",".join(bad))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=True
        )
        incomplete = [name for name in result.stdout.strip().split(",") if name]
        assert incomplete == [], (
            f"unresolved forward references in {incomplete} — a field annotated with a "
            f"class defined lower in the module is silently dropped from the response "
            f"payload, with no error and no failing build"
        )

    def test_the_stock_payload_actually_carries_the_explanation_key(self, client) -> None:
        """`explanation: null` and no `explanation` key at all are different
        answers to a client, and only the second is a bug -- so this checks for
        the key rather than for a truthy value."""
        payload = client.get("/api/stocks/AAPL").json()
        assert "explanation" in payload
