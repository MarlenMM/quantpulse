"""Streamlit and React must publish the same beta for the same stock.

They did not. The Streamlit Stock Detail page borrowed
`app/lib/data.universe_panel()`, whose 150-day default exists for the
Dashboard's one-month sector-rotation read (its own docstring says as much),
while the API used its own `_RISK_PANEL_DAYS = 420`. Reading the committed demo
database on the same day, the two pages reported AIZ at **beta 0.46, R² 0.06
over 103 shared bars** and **beta 0.57, R² 0.07 over 288** respectively.

Neither number was arithmetically wrong, which is why every existing gate
passed: it was a third copy of a constant that had quietly drifted. The window
now lives once, in `risk.MARKET_PANEL_DAYS`, and this test asserts the property
that actually matters -- the two front ends agree on the number -- rather than
asserting the constant is shared, which a future refactor could satisfy while
still disagreeing.

**A window is only half of "the same beta".** The other half is *which market*,
and that is a second thing three surfaces could each decide for themselves. So
the fixture stores a real index series (`risk.MARKET_INDEX_SYMBOL`) and this
test pins the benchmark too: the Streamlit caption and the API's
`beta_benchmark` must name the same series. Without the index row the fixture
would silently exercise only `resolve_market_returns`' fallback, and a
regression in which one surface regressed against the index while the other
used the proxy would pass -- which is the same shape of bug as the original,
one level along.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis import risk
from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage import persistence
from quantpulse.storage.models import Base, CompositeScore, PriceHistory, Ticker

APP_DIR = Path(__file__).resolve().parents[2] / "app"
STOCK_DETAIL = str(APP_DIR / "pages" / "2_Stock_Detail.py")
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
# Enough history that a 150-day and a 420-day window genuinely differ; with a
# short series both would truncate to the same bars and the bug would hide.
DAYS = 500
# The top-scored name, so the page defaults to it.
TARGET = "AAA"
# ...and its sensitivity to the market *changes partway through the history*.
# A stock with one stable beta measures the same over any window, which makes a
# window mismatch invisible -- the first version of this test seeded exactly
# that and passed with the bug reintroduced. A regime change is what separates
# "the last five months" from "the last fourteen".
RECENT_DAYS = 150
BETA_RECENT = 2.5
BETA_OLDER = 0.1


def _seed(session: Session) -> None:
    rng = np.random.default_rng(3)
    market = rng.normal(0.0004, 0.010, DAYS)

    # The index itself, from the same return draws every stock is built from --
    # so `BETA_RECENT`/`BETA_OLDER` are the stock's *true* betas against the
    # series both front ends are supposed to regress against, not an
    # approximation of it.
    persistence.upsert_benchmark_ticker(
        session, symbol=risk.MARKET_INDEX_SYMBOL, name=risk.MARKET_INDEX_NAME
    )
    index_level = 4000.0
    for offset, m in zip(range(DAYS, 0, -1), market, strict=True):
        index_level *= float(np.exp(m))
        session.add(
            PriceHistory(
                symbol=risk.MARKET_INDEX_SYMBOL,
                date=date(2026, 7, 27) - timedelta(days=offset),
                open=index_level,
                high=index_level * 1.01,
                low=index_level * 0.99,
                close=index_level,
                adj_close=index_level,
                volume=0,
            )
        )

    for index, symbol in enumerate(SYMBOLS):
        session.add(
            Ticker(
                symbol=symbol,
                name=f"{symbol} Inc",
                sector="Information Technology",
                asset_type="equity",
                is_active=True,
            )
        )
        noise = rng.normal(0.0, 0.008, DAYS)
        level = 100.0
        offsets = range(DAYS, 0, -1)
        for offset, m, e in zip(offsets, market, noise, strict=True):
            if symbol == TARGET:
                beta_true = BETA_RECENT if offset <= RECENT_DAYS else BETA_OLDER
            else:
                beta_true = 0.5 + 0.3 * index
            level *= float(np.exp(beta_true * m + e))
            session.add(
                PriceHistory(
                    symbol=symbol,
                    date=date(2026, 7, 27) - timedelta(days=offset),
                    open=level,
                    high=level * 1.01,
                    low=level * 0.99,
                    close=level,
                    adj_close=level,
                    volume=1_000_000,
                )
            )
        # TARGET scores highest so the page opens on it.
        score = 99.0 if symbol == TARGET else 50.0 + index
        session.add(
            CompositeScore(
                symbol=symbol,
                date=date(2026, 7, 27),
                profile="balanced",
                composite_score=score,
                percentile_rank=score,
                rating="hold",
                data_confidence=80.0,
                **{f"{category}_score": score for category in CATEGORIES},
            )
        )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'beta_parity.db'}")
    Base.metadata.create_all(eng)
    with sessionmaker(bind=eng)() as session:
        _seed(session)
        session.commit()
    return eng


@contextmanager
def _streamlit_wired(eng: Engine) -> Iterator[None]:
    factory = sessionmaker(bind=eng)

    @contextmanager
    def fake_get_session() -> Iterator[object]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    from lib import data as lib_data

    for name in dir(lib_data):
        if name.startswith("_"):
            continue
        reader = getattr(lib_data, name)
        if callable(reader) and hasattr(reader, "clear"):
            reader.clear()
    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.data.portfolio_backend", return_value="session"),
        patch("quantpulse.llm.providers.get_provider", return_value=None),
    ):
        yield


def _streamlit_beta(eng: Engine) -> tuple[str, str, float, str]:
    """The symbol the page chose, the Beta it printed, and its R².

    Reading the symbol back off the page rather than assuming one matters: the
    page defaults to the top-ranked name, not the first alphabetically, so
    hardcoding a symbol silently compares two different stocks.
    """
    with _streamlit_wired(eng):
        at = AppTest.from_file(STOCK_DETAIL, default_timeout=180)
        at.run()
    assert not at.exception, f"Stock Detail raised: {at.exception}"

    headings = [str(element.value) for element in at.markdown if "—" in str(element.value)]
    assert headings, "the page printed no symbol heading"
    symbol = headings[0].split("###")[-1].split("—")[0].strip()

    betas = [str(m.value) for m in at.metric if m.label == "Beta"]
    assert betas, f"the page showed no Beta metric; metrics were {[m.label for m in at.metric]}"

    captions = [str(element.value) for element in at.caption]
    # Matched on the phrase that is *structural* rather than on the benchmark's
    # name: an assertion that hunted for "equal-weight proxy" would have gone on
    # passing after the index landed, by silently matching the fallback wording.
    matches = [c for c in captions if "R² =" in c and "Beta is measured against" in c]
    assert matches, f"the page reported no beta caption; captions were {captions}"
    caption = matches[0]
    benchmark = caption.split("Beta is measured against ")[1].split(", R² =")[0]
    return (
        symbol,
        betas[0],
        float(caption.split("R² = ")[1].split()[0].rstrip(".")),
        benchmark,
    )


def _api_beta(eng: Engine, symbol: str) -> tuple[float, float, str]:
    from quantpulse.api import main as api_main

    factory = sessionmaker(bind=eng)
    with factory() as session:
        # Same call the `/api/stocks/{symbol}` handler makes, same default window.
        bars = persistence.read_symbol_ohlcv(session, symbol, lookback_days=400)
        profile = api_main._risk_profile(session, symbol, bars)
    assert profile is not None and profile.beta is not None
    return (
        float(profile.beta),
        float(profile.beta_r_squared or 0.0),
        str(profile.beta_benchmark),
    )


def test_both_front_ends_regress_beta_over_the_same_window(engine: Engine) -> None:
    """The window is shared, so the two surfaces cannot drift apart again."""
    from quantpulse.api import main as api_main

    assert api_main._RISK_PANEL_DAYS == risk.MARKET_PANEL_DAYS


def test_streamlit_and_api_report_the_same_beta(engine: Engine) -> None:
    """The property that matters: the same stock, the same day, the same number.

    Both surfaces print to two decimals, so the comparison is made there -- far
    tighter than the 0.46-vs-0.57 gap the drifted windows produced on real data.
    """
    symbol, page_beta, page_r2, page_benchmark = _streamlit_beta(engine)
    api_beta, api_r2, api_benchmark = _api_beta(engine, symbol)

    assert page_beta == f"{api_beta:.2f}", (
        f"Stock Detail shows beta {page_beta} but the API reports {api_beta:.4f} for "
        f"{symbol} -- the two front ends are regressing beta over different windows"
    )
    assert page_r2 == pytest.approx(round(api_r2, 2), abs=0.005), (
        f"Stock Detail shows R² {page_r2} but the API reports {api_r2:.4f} for {symbol}"
    )
    assert page_benchmark == api_benchmark, (
        f"Stock Detail says beta is measured against {page_benchmark!r} but the API "
        f"reports {api_benchmark!r} -- the same number described as two different markets"
    )
    # And it must be the real index, not the fallback: the fixture stores one,
    # so a run that lands on the proxy means the resolver stopped finding it.
    assert risk.MARKET_INDEX_SYMBOL in page_benchmark, (
        f"both front ends agree, but on the equal-weight fallback ({page_benchmark!r}) "
        f"even though the fixture seeded {risk.MARKET_INDEX_SYMBOL}"
    )


def test_beta_is_measured_against_the_index_not_the_universe(engine: Engine) -> None:
    """The fix itself: the stock's true beta is recovered, which the proxy could not do.

    `_seed` builds TARGET with a true beta of `BETA_RECENT` against the index
    series it also stores, so this asserts against a known answer rather than
    against whatever the code currently prints. Regressed on an equal-weight
    basket of the six fixture names instead, the same stock reads far lower --
    the synthetic version of NVDA reading 0.70 against the proxy and 1.86
    against `^GSPC` on the real database.
    """
    factory = sessionmaker(bind=engine)
    end = date.today()
    start = end - timedelta(days=risk.MARKET_PANEL_DAYS)
    with factory() as session:
        index = persistence.read_benchmark_closes(
            session, symbol=risk.MARKET_INDEX_SYMBOL, start=start, end=end
        )
        panel = persistence.read_adj_close_panel(session, start=start, end=end)
        target = risk.to_returns(panel[TARGET])

    resolved = risk.resolve_market_returns(index, panel)
    assert resolved.source == "index"
    assert risk.MARKET_INDEX_SYMBOL not in panel.columns, (
        "the index leaked into the universe panel; it must stay out of every "
        "screener, search and scoring read (see persistence.upsert_benchmark_ticker)"
    )

    proxy = risk.equal_weight_market_returns(panel)

    # Over the recent regime alone the fixture's true beta is exactly
    # `BETA_RECENT`, so this is a known answer rather than a snapshot of what
    # the code happens to print. Over the full 420-day window it would be a
    # blend of the two regimes -- which is correct, and untestable against a
    # constant, which is why the slice is taken.
    recent = target.tail(RECENT_DAYS - 10)
    against_index = risk.beta(recent, resolved.returns)
    against_proxy = risk.beta(recent, proxy)
    assert against_index is not None and against_proxy is not None
    assert against_index.beta == pytest.approx(BETA_RECENT, rel=0.15), (
        f"beta against the index came out at {against_index.beta:.2f}, but over this "
        f"window the fixture built this stock with a true beta of {BETA_RECENT}"
    )
    assert abs(against_proxy.beta - BETA_RECENT) > abs(against_index.beta - BETA_RECENT), (
        f"the equal-weight proxy recovered the true beta as well as the index did "
        f"({against_proxy.beta:.2f} vs {against_index.beta:.2f}) -- if that is really "
        f"true, this test is not exercising the difference the fix exists for"
    )
    # Deliberately NOT asserted here: that the index also wins on R². It does on
    # real data by a wide margin (NVDA 0.42 vs 0.05 on the committed demo
    # database), but it does not on this fixture, and the reason is the
    # fixture's own construction rather than the code's: all six names are
    # `beta_i * market + independent noise`, so averaging them cancels ~1/sqrt(6)
    # of the noise and yields a *cleaner* market factor than the index series
    # itself carries. Asserting it anyway would mean tuning the fixture until a
    # real-data property held in a synthetic world -- which is how a test ends up
    # measuring its own setup. The beta comparison above is the real claim.


def test_stock_detail_shows_sharpe_beside_sortino(engine: Engine) -> None:
    """Sharpe was computed, typed, served over the API -- and rendered nowhere.

    `stock_risk_profile` produces both ratios under one shared floor, the
    Portfolio and Track Record pages both show a Sharpe, `RiskProfileModel`
    carries it, and the React page's own caption told the reader it was being
    withheld for sample-size reasons. Only the metric itself was missing, on
    both front ends.
    """
    with _streamlit_wired(engine):
        at = AppTest.from_file(STOCK_DETAIL, default_timeout=180)
        at.run()
    assert not at.exception

    labels = [m.label for m in at.metric]
    assert "Sharpe" in labels, f"no Sharpe metric on Stock Detail; showed {labels}"
    assert "Sortino" in labels, "Sortino disappeared"

    # Both share one floor, so with a year of history neither may be a dash.
    values = {m.label: str(m.value) for m in at.metric}
    assert values["Sharpe"] != "—", "Sharpe rendered as a dash despite ample history"
