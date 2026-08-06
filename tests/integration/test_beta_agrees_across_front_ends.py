"""Streamlit and React must publish the same beta for the same stock.

They did not. The Streamlit Stock Detail page borrowed
`app/lib/data.universe_panel()`, whose 150-day default exists for the
Dashboard's one-month sector-rotation read (its own docstring says as much),
while the API used its own `_RISK_PANEL_DAYS = 420`. Reading the committed demo
database on the same day, the two pages reported AIZ at **beta 0.46, R² 0.06
over 103 shared bars** and **beta 0.57, R² 0.07 over 288** respectively -- both
captioned "against an equal-weight proxy for the market".

Neither number was arithmetically wrong, which is why every existing gate
passed: it was a third copy of a constant that had quietly drifted. The window
now lives once, in `risk.MARKET_PANEL_DAYS`, and this test asserts the property
that actually matters -- the two front ends agree on the number -- rather than
asserting the constant is shared, which a future refactor could satisfy while
still disagreeing.
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


def _streamlit_beta(eng: Engine) -> tuple[str, str, float]:
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
    matches = [c for c in captions if "R² =" in c and "equal-weight proxy" in c]
    assert matches, f"the page reported no beta caption; captions were {captions}"
    return symbol, betas[0], float(matches[0].split("R² = ")[1].split()[0].rstrip("."))


def _api_beta(eng: Engine, symbol: str) -> tuple[float, float]:
    from quantpulse.api import main as api_main
    from quantpulse.storage import persistence

    factory = sessionmaker(bind=eng)
    with factory() as session:
        # Same call the `/api/stocks/{symbol}` handler makes, same default window.
        bars = persistence.read_symbol_ohlcv(session, symbol, lookback_days=400)
        profile = api_main._risk_profile(session, symbol, bars)
    assert profile is not None and profile.beta is not None
    return float(profile.beta), float(profile.beta_r_squared or 0.0)


def test_both_front_ends_regress_beta_over_the_same_window(engine: Engine) -> None:
    """The window is shared, so the two surfaces cannot drift apart again."""
    from quantpulse.api import main as api_main

    assert api_main._RISK_PANEL_DAYS == risk.MARKET_PANEL_DAYS


def test_streamlit_and_api_report_the_same_beta(engine: Engine) -> None:
    """The property that matters: the same stock, the same day, the same number.

    Both surfaces print to two decimals, so the comparison is made there -- far
    tighter than the 0.46-vs-0.57 gap the drifted windows produced on real data.
    """
    symbol, page_beta, page_r2 = _streamlit_beta(engine)
    api_beta, api_r2 = _api_beta(engine, symbol)

    assert page_beta == f"{api_beta:.2f}", (
        f"Stock Detail shows beta {page_beta} but the API reports {api_beta:.4f} for "
        f"{symbol} -- the two front ends are regressing beta over different windows"
    )
    assert page_r2 == pytest.approx(round(api_r2, 2), abs=0.005), (
        f"Stock Detail shows R² {page_r2} but the API reports {api_r2:.4f} for {symbol}"
    )
