"""Numbers a page puts in a table must be formatted the way the page's own metrics are.

Both bugs this file pins were invisible to every existing gate. Lint, mypy and
`test_ui_pages_render.py` all pass on an unformatted table -- it renders, it just
renders wrongly -- and `AppTest`'s `.value` returns the *underlying* frame, so a
missing `Styler.format` looks identical to a present one there. They were found
by screenshotting the running app and reading the numbers.

What they were:

* Stock Detail's forecast table printed dollar prices as raw float64:
  "282.7719" next to "287.08" (Streamlit drops trailing zeros, so the columns
  did not even align), with no currency marker -- while the React page rendered
  the very same stored row as "$282.77" through `formatPrice`. Two front ends
  disagreeing about one number is the specific failure the shared-reader design
  exists to prevent.
* The Screener's Compare table printed a sub-score as "98.80715705765407" and a
  missing category as "<NA>". Its columns hold scores *and* the Rating string,
  so the dtype is object and neither a Styler nor a NumberColumn ever applies --
  pandas falls back to `repr`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage.models import (
    Base,
    CompositeScore,
    Forecast,
    PriceHistory,
    Ticker,
)

APP_DIR = Path(__file__).resolve().parents[2] / "app"
AS_OF = date(2026, 7, 27)


def _seed(engine: Engine) -> None:
    """Two names with enough history to score, and one priced forecast row each."""
    rng = np.random.default_rng(0)
    with sessionmaker(bind=engine)() as session:
        for index, symbol in enumerate(("AAA", "BBB")):
            session.add(
                Ticker(
                    symbol=symbol,
                    name=f"{symbol} Inc",
                    sector="Information Technology",
                    asset_type="equity",
                    is_active=True,
                )
            )
            level = 100.0
            for offset in range(400, 0, -1):
                level *= float(np.exp(rng.normal(0.0004, 0.014)))
                session.add(
                    PriceHistory(
                        symbol=symbol,
                        date=AS_OF - timedelta(days=offset),
                        open=level,
                        high=level * 1.01,
                        low=level * 0.99,
                        close=level,
                        adj_close=level,
                        volume=1_000_000,
                    )
                )
            session.add(
                CompositeScore(
                    symbol=symbol,
                    date=AS_OF,
                    profile="balanced",
                    composite_score=60.0 + index * 5,
                    percentile_rank=60.0 + index * 5,
                    rating="buy",
                    data_confidence=75.0,
                    # Deliberately only two categories carry a score: a missing
                    # one is what used to render as "<NA>".
                    technical_score=98.80715705765407,
                    momentum_score=91.31313131313131,
                    **{f"{category}_raw": 0.5 for category in CATEGORIES},
                )
            )
            # Prices chosen so a correct render is unambiguous: 282.771949 must
            # become "$282.77" and never "282.7719".
            session.add(
                Forecast(
                    symbol=symbol,
                    generated_date=AS_OF,
                    horizon_days=5,
                    model_name="baseline",
                    point_return=0.005,
                    point_price=282.771949,
                    lower_price=267.841062,
                    upper_price=298.535162,
                )
            )
        session.commit()


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'fmt.db'}")
    Base.metadata.create_all(eng)
    _seed(eng)
    return eng


@contextmanager
def _wired(eng: Engine) -> Iterator[list[Any]]:
    """Run a page against `eng`, capturing every object handed to `st.dataframe`.

    Capturing the argument is the only way to assert on a `Styler`: the format
    is applied when Streamlit serializes it for the browser, so `AppTest` sees
    the raw frame either way.
    """
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

    captured: list[Any] = []

    def record(frame: Any, *args: Any, **kwargs: Any) -> None:
        captured.append(frame)

    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.data.portfolio_backend", return_value="session"),
        patch("streamlit.dataframe", record),
        patch("quantpulse.llm.providers.get_provider", return_value=None),
    ):
        yield captured


def _rendered(obj: Any) -> str:
    """Every display string a captured table would show the user."""
    if isinstance(obj, pd.io.formats.style.Styler):
        return obj.to_html()
    return pd.DataFrame(obj).to_string()


def test_forecast_prices_render_as_money_not_raw_floats(engine: Engine) -> None:
    with _wired(engine) as captured:
        at = AppTest.from_file(str(APP_DIR / "pages" / "2_Stock_Detail.py"), default_timeout=180)
        at.run()
    assert not at.exception

    forecast_tables = [t for t in captured if "$282.77" in _rendered(t)]
    assert forecast_tables, (
        "the forecast table never rendered its point price as money -- "
        f"captured tables: {[_rendered(t)[:200] for t in captured]}"
    )
    html = _rendered(forecast_tables[0])
    # The band bounds must be formatted too, not just the headline price.
    assert "$267.84" in html and "$298.54" in html
    # ...and the unrounded float must be gone entirely.
    assert "282.771949" not in html
    assert "267.841062" not in html


def test_compare_table_rounds_subscores_and_dashes_missing_ones(engine: Engine) -> None:
    with _wired(engine) as captured:
        at = AppTest.from_file(str(APP_DIR / "pages" / "1_Screener.py"), default_timeout=180)
        at.run()
    assert not at.exception

    compare = [t for t in captured if "98.8" in _rendered(t)]
    assert compare, "the Compare table never rendered the technical sub-score"
    text = _rendered(compare[0])
    assert "98.80715705765407" not in text, "sub-score rendered at full float precision"
    assert "<NA>" not in text, "a missing category rendered as <NA> rather than an em dash"
    assert "—" in text
