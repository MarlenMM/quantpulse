"""Searching for a stock the nightly job has never scored, from the actual page.

The engine that analyses an uncovered symbol is unit-tested; this is about the
wiring. The page has to (a) find a catalogue symbol at all -- the picker used to
offer only scored names, so anything else was unreachable -- and (b) route it to
the live path rather than trying to read stored analysis that does not exist.

Both fail silently in opposite directions if wired wrongly: an unreachable
symbol looks like the app not knowing about the company, and a catalogue symbol
sent down the stored path looks like a crash.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.storage.models import Base, CompositeScore, PriceHistory, Ticker

REPO = Path(__file__).resolve().parents[2]
PAGE = str(REPO / "app" / "pages" / "2_Stock_Detail.py")

RANKED = "AAA"
CATALOGUED = "ZZQQ"


def _prices(symbol: str, n: int = 300) -> list[PriceHistory]:
    start = date(2026, 1, 1)
    level = 100.0
    bars = []
    for i in range(n):
        level *= 1.001
        bars.append(
            PriceHistory(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=level,
                high=level * 1.01,
                low=level * 0.99,
                close=level,
                adj_close=level,
                volume=1_000_000,
            )
        )
    return bars


@pytest.fixture
def app(tmp_path: Path):
    """The page, pointed at a database holding one ranked and one catalogued symbol.

    `lib.data.get_session` is patched rather than `DATABASE_URL` set, because
    `storage.db` binds its engine at *import* time -- once any other test module
    has imported it, an environment variable cannot move it, and this page then
    reads whatever database that first import chose. That failure only appears
    when the whole suite runs, which is the worst time to discover it.

    The cached readers are cleared for the same class of reason: `@st.cache_data`
    outlives a single test, so without this the second test answers from the
    first one's temporary database.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'ondemand.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(
            Ticker(
                symbol=RANKED,
                name="Alpha Corp",
                sector="Industrials",
                asset_type="equity",
                is_active=True,
                coverage=Ticker.RANKED,
            )
        )
        session.add(
            Ticker(
                symbol=CATALOGUED,
                name="Zeta Quantum",
                sector="Information Technology",
                asset_type="equity",
                is_active=False,
                coverage=Ticker.CATALOGUE,
            )
        )
        session.add_all(_prices(RANKED))
        session.add(
            CompositeScore(
                symbol=RANKED,
                date=date(2026, 8, 7),
                profile="balanced",
                technical_score=70.0,
                momentum_score=65.0,
                composite_score=68.0,
                percentile_rank=80.0,
                rating="buy",
                data_confidence=45.0,
            )
        )
        session.commit()

    @contextmanager
    def fake_get_session():
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    import lib.data as lib_data

    for name in dir(lib_data):
        reader = getattr(lib_data, name)
        if callable(reader) and hasattr(reader, "clear"):
            reader.clear()

    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.data.portfolio_backend", return_value="session"),
    ):
        yield AppTest.from_file(PAGE, default_timeout=120)


def _search(app: AppTest, symbol: str) -> AppTest:
    search = next(w for w in app.text_input if "Search" in (w.label or ""))
    return search.set_value(symbol).run()


def test_a_catalogue_symbol_is_findable_at_all(app: AppTest) -> None:
    """The picker must reach beyond the scored universe.

    Before the catalogue existed this symbol simply could not be selected, which
    reads to a visitor as "this app has never heard of that company".
    """
    app.run()
    assert not app.exception
    _search(app, CATALOGUED)

    options = next(w for w in app.selectbox if w.label == "Symbol").options
    assert any(CATALOGUED in str(option) for option in options)


def test_a_catalogue_symbol_routes_to_the_live_path(app: AppTest) -> None:
    """And is analysed rather than looked up.

    `on_demand.analyse` is stubbed: this asserts the *routing*, not the
    analysis, which has its own tests. What matters is that the page called it
    and rendered its honest framing.
    """
    from quantpulse import on_demand

    stub = on_demand.OnDemandAnalysis(
        symbol=CATALOGUED,
        name="Zeta Quantum",
        sector="Information Technology",
        as_of=date(2026, 8, 7),
        computed_at=pd.Timestamp("2026-08-08 12:00").to_pydatetime(),
        prices=pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-01", periods=60),
                "open": np.linspace(10, 12, 60),
                "high": np.linspace(10, 12, 60) * 1.01,
                "low": np.linspace(10, 12, 60) * 0.99,
                "close": np.linspace(10, 12, 60),
                "adj_close": np.linspace(10, 12, 60),
                "volume": 1_000_000,
            }
        ),
        category_raw={"technical": 71.0, "momentum": 64.0},
        composite_score=67.5,
        absolute_rating="buy",
        data_confidence=35.0,
        patterns=[],
        forecasts=[],
        fundamentals=None,
        analyst=None,
        percentile_vs_ranked=82.0,
        ranked_universe_size=503,
        notes=["News sentiment is not computed for a live lookup."],
    )

    with patch("quantpulse.on_demand.analyse", return_value=stub) as analysed:
        app.run()
        _search(app, CATALOGUED)

    assert not app.exception
    assert analysed.called, "the page never asked for a live analysis"

    rendered = " ".join(
        str(element.value)
        for group in (app.markdown, app.caption, app.info, app.subheader)
        for element in group
    )
    # The framing is the point: a visitor must not read this as a ranking entry.
    assert "not part of the ranking" in rendered
    assert "News sentiment is not computed" in rendered

    # The placement is a metric, and its label must not say "percentile" --
    # this stock has no rank within the universe, only a position against it.
    metrics = {m.label: str(m.value) for m in app.metric}
    assert metrics["Would place above"] == "82%"
    assert metrics["Rating (absolute)"].endswith("Buy")


def test_a_ranked_symbol_still_uses_the_stored_analysis(app: AppTest) -> None:
    """Mutation guard: routing everything to the live path would also pass the
    test above, while silently discarding the nightly pipeline's work."""
    with patch("quantpulse.on_demand.analyse") as analysed:
        app.run()
        _search(app, RANKED)

    assert not app.exception
    assert not analysed.called, "a scored symbol was re-analysed live instead of read"
