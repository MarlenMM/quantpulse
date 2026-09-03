"""Route tests for the FastAPI read API (ADR 4.1's stretch goal).

Runs against a real temporary SQLite database rather than mocked readers: the
whole point of the API layer is that it faithfully translates what
`storage.persistence` returns, so mocking the readers would test only that the
mock was wired up.
"""

import json
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
        """
        run = client.get("/api/backtest").json()[0]
        assert run["payoff_ratio"] == pytest.approx(1.5)
        # Quarter-Kelly on p=0.6, b=1.5: (1.5*0.6 - 0.4) / 1.5 = 1/3, quartered.
        assert run["kelly_fraction"] == pytest.approx(0.25 * (1.0 / 3.0))

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
