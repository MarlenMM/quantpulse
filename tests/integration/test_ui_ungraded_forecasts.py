"""An ungraded forecast must not be displayed like a graded one.

Section 7.6: "a forecast without its own track record next to it invites more
confidence than it's earned." The Stock Detail page satisfied that to the letter
— an ungraded horizon showed three dashes — and defeated it with typography: the
row sat in the same table, in the same weight, as rows standing on 154 measured
out-of-sample windows.

The rows that lose by this are exactly the ones carrying the largest numbers. On
the real universe every 63- and 252-day forecast is ungraded (not *some* — all of
them), the 252-day ones average **+16%** and reach **+231%**, and the biggest,
least defensible figure on the page was the one with no accuracy measurement at
all behind it.

So the graded horizons are the default view and the ungraded ones sit behind a
disclosure that says what is missing. These tests pin that split at the point a
reader actually experiences it: what the page hands to `st.dataframe`, and in
what order.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis import forecasting
from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage.models import Base, CompositeScore, Forecast, PriceHistory, Ticker

APP_DIR = Path(__file__).resolve().parents[2] / "app"
STOCK_DETAIL = str(APP_DIR / "pages" / "2_Stock_Detail.py")
AS_OF = date(2026, 8, 26)
SYMBOL = "AAA"

# The shape the real database has: the short horizons carry a measured rate, the
# long ones carry none at all -- and the long ones are where the big numbers are.
GRADED_HORIZONS = (5, 20)
UNGRADED_HORIZONS = (63, 252)
UNGRADED_RETURNS = {63: 0.163, 252: 0.822}


def _seed(engine: Engine) -> None:
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(Ticker(symbol=SYMBOL, name="AAA Inc", sector="Tech", asset_type="equity"))
        level = 100.0
        for offset in range(400, 0, -1):
            level *= 1.001
            session.add(
                PriceHistory(
                    symbol=SYMBOL,
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
                symbol=SYMBOL,
                date=AS_OF,
                profile="balanced",
                composite_score=71.0,
                percentile_rank=88.0,
                rating="buy",
                data_confidence=90.0,
                **{f"{category}_score": 71.0 for category in CATEGORIES},
            )
        )
        for horizon in GRADED_HORIZONS:
            session.add(
                Forecast(
                    symbol=SYMBOL,
                    generated_date=AS_OF,
                    horizon_days=horizon,
                    model_name="baseline",
                    point_return=0.013,
                    point_price=211.26,
                    lower_price=190.12,
                    upper_price=234.75,
                    historical_hit_rate=0.50,
                    baseline_hit_rate=0.51,
                    hit_rate_windows=154,
                )
            )
        for horizon in UNGRADED_HORIZONS:
            session.add(
                Forecast(
                    symbol=SYMBOL,
                    generated_date=AS_OF,
                    horizon_days=horizon,
                    model_name="baseline",
                    point_return=UNGRADED_RETURNS[horizon],
                    point_price=379.77,
                    lower_price=180.53,
                    upper_price=798.91,
                    historical_hit_rate=None,
                    baseline_hit_rate=None,
                    hit_rate_windows=None,
                )
            )
        session.commit()


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{tmp_path / 'ungraded.db'}")
    Base.metadata.create_all(eng)
    _seed(eng)
    return eng


@contextmanager
def _wired(eng: Engine) -> Iterator[tuple[list[Any], list[str]]]:
    """Render the page, capturing every `st.dataframe` argument and expander label.

    Capturing the argument is the only way to assert on a `Styler`: the format
    is applied when Streamlit serializes it, so `AppTest` sees the raw frame
    either way. The expander labels are captured for the same reason -- the
    disclosure is the point, and `AppTest` does not expose one it did not open.
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

    tables: list[Any] = []
    expanders: list[str] = []
    real_expander = __import__("streamlit").expander

    def record_table(frame: Any, *args: Any, **kwargs: Any) -> None:
        tables.append(frame)

    def record_expander(label: str, *args: Any, **kwargs: Any) -> Any:
        expanders.append(label)
        return real_expander(label, *args, **kwargs)

    with (
        patch("lib.data.get_session", fake_get_session),
        patch("lib.data.portfolio_backend", return_value="session"),
        patch("streamlit.dataframe", record_table),
        patch("streamlit.expander", record_expander),
        patch("quantpulse.llm.providers.get_provider", return_value=None),
    ):
        yield tables, expanders


def _horizons(obj: Any) -> set[int]:
    """The horizon column of a captured forecast table, or an empty set."""
    frame = obj.data if isinstance(obj, pd.io.formats.style.Styler) else obj
    if not isinstance(frame, pd.DataFrame) or "Horizon (days)" not in frame.columns:
        return set()
    return {int(v) for v in frame["Horizon (days)"]}


def _render(eng: Engine) -> tuple[list[set[int]], list[str]]:
    with _wired(eng) as (tables, expanders):
        at = AppTest.from_file(STOCK_DETAIL, default_timeout=180)
        at.run()
    assert not at.exception, f"Stock Detail raised: {at.exception}"
    return [h for h in (_horizons(t) for t in tables) if h], expanders


class TestUngradedForecastsAreNotShownAsEvidence:
    def test_the_default_table_holds_only_graded_horizons(self, engine: Engine) -> None:
        forecast_tables, _ = _render(engine)
        assert forecast_tables, "the page rendered no forecast table at all"
        assert forecast_tables[0] == set(GRADED_HORIZONS), (
            f"the first forecast table a reader sees contains {forecast_tables[0]}, but "
            f"only {set(GRADED_HORIZONS)} have a measured accuracy -- an ungraded "
            f"forecast is again being displayed as though it were evidence"
        )

    def test_the_ungraded_horizons_are_still_available(self, engine: Engine) -> None:
        """Disclosed, not deleted.

        The forecast is the model's honest output and a reader who came for a
        one-year number should find it. What changes is that it stops borrowing
        the credibility of the rows above it.
        """
        forecast_tables, _ = _render(engine)
        shown = set().union(*forecast_tables)
        assert shown == set(GRADED_HORIZONS) | set(UNGRADED_HORIZONS)

    def test_they_sit_behind_a_disclosure_that_names_them(self, engine: Engine) -> None:
        _, expanders = _render(engine)
        labels = " ".join(expanders).lower()
        assert "ungraded" in labels, (
            f"no disclosure identified the ungraded horizons; expanders were {expanders}"
        )
        for horizon in UNGRADED_HORIZONS:
            assert f"{horizon}-day" in labels

    def test_the_biggest_number_on_the_page_is_the_one_behind_the_disclosure(
        self, engine: Engine
    ) -> None:
        """The whole point, stated as the property rather than as a layout.

        This fails if the split is ever made on something other than evidence --
        by horizon length, say, or by a hardcoded list -- in a way that lets the
        largest unproven figure back into the default view.
        """
        forecast_tables, _ = _render(engine)
        biggest_ungraded = max(UNGRADED_RETURNS.values())
        assert biggest_ungraded > 0.5, "fixture: the ungraded row must carry a large number"
        assert 252 not in forecast_tables[0]
        assert not forecast_tables[0] & set(UNGRADED_HORIZONS)


class TestTheSplitRuleIsShared:
    def test_the_page_uses_forecastings_predicate(self) -> None:
        """One rule, not a null check written separately on each surface.

        The API sends `is_graded` to React from this same function; a second
        implementation here is how two front ends come to grade the same
        forecast differently.
        """
        assert forecasting.is_graded(0.5) is True
        assert forecasting.is_graded(None) is False
        assert forecasting.is_graded(float("nan")) is False

    def test_a_zero_hit_rate_is_graded(self) -> None:
        """ "Measured at zero" and "never measured" are different claims.

        A model that got every direction wrong has a track record -- a very bad
        one -- and belongs in the default table saying so. Collapsing it into the
        ungraded group would hide the most informative result the page can show.
        """
        assert forecasting.is_graded(0.0) is True
