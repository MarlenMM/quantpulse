"""The Screener's sidebar filters must survive every rating scheme.

Absolute rating mode re-scores from the stored raw category values, and rows
written before those columns existed genuinely cannot be re-rated that way --
so the page falls back to the relative ranking and says so. The bug this file
pins is what it fell back *to*: the whole universe, discarding the sector,
rating, coverage and search choices the user had made. Picking "Absolute" then
silently getting every symbol back is a different answer to a different
question, and nothing on screen said the filters had been dropped.

Runs the real page through `streamlit.testing.v1.AppTest`, following
`test_ui_orphan_sections.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.storage.models import Base, CompositeScore, Ticker

SCREENER = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Screener.py")
AS_OF = date(2026, 7, 27)

# (symbol, sector, composite) -- two sectors so a sector filter has something to
# exclude, and distinct scores so the ordering is unambiguous.
_UNIVERSE = (
    ("NVDA", "Information Technology", 91.0),
    ("MSFT", "Information Technology", 74.0),
    ("XOM", "Energy", 55.0),
    ("CVX", "Energy", 41.0),
)


@pytest.fixture
def legacy_engine(tmp_path: Path) -> Engine:
    """Scores written before the `*_raw` columns existed -- absolute mode can't use them."""
    engine = create_engine(f"sqlite:///{tmp_path / 'screener.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        for symbol, sector, composite in _UNIVERSE:
            session.add(
                Ticker(
                    symbol=symbol,
                    name=f"{symbol} Inc",
                    sector=sector,
                    asset_type="equity",
                    is_active=True,
                )
            )
            session.add(
                CompositeScore(
                    symbol=symbol,
                    date=AS_OF,
                    profile="balanced",
                    composite_score=composite,
                    percentile_rank=composite,
                    rating="buy",
                    data_confidence=80.0,
                    # Normalized sub-scores present, raw ones deliberately NULL.
                    **{f"{category}_score": composite for category in CATEGORIES},
                )
            )
        session.commit()
    return engine


@contextmanager
def _wired(engine: Engine) -> Iterator[None]:
    factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_session() -> Iterator[object]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    from lib import data as lib_data

    for reader in (lib_data.screener_rows, lib_data.universe, lib_data.market_regime):
        reader.clear()
    with patch("lib.data.get_session", fake_get_session):
        yield


def _run_with_energy_filter(engine: Engine, *, absolute: bool) -> AppTest:
    with _wired(engine):
        at = AppTest.from_file(SCREENER, default_timeout=120)
        at.run()
        # Addressed by label, not index: AppTest lists main-body widgets before
        # sidebar ones, so positional indexing here silently drives the wrong
        # control (the Compare picker) and the test passes for the wrong reason.
        next(m for m in at.multiselect if m.label == "Sector").set_value(["Energy"])
        if absolute:
            next(r for r in at.radio if r.label.startswith("How should")).set_value("absolute")
        at.run()
    return at


def _offered_tickers(at: AppTest) -> list[str]:
    """Symbols the Compare picker offers -- the page builds them from the filtered rows."""
    compare = next(m for m in at.multiselect if m.label == "Tickers")
    # Options are rendered "SYMBOL — Company Name" by the autocomplete formatter.
    return sorted(str(option).split(" ")[0] for option in compare.options)


class TestSectorFilterSurvivesTheRatingScheme:
    def test_relative_mode_respects_the_sector_filter(self, legacy_engine: Engine) -> None:
        at = _run_with_energy_filter(legacy_engine, absolute=False)
        assert not at.exception
        assert _offered_tickers(at) == ["CVX", "XOM"]

    def test_absolute_mode_fallback_keeps_the_same_filter(self, legacy_engine: Engine) -> None:
        at = _run_with_energy_filter(legacy_engine, absolute=True)
        assert not at.exception
        # The fallback warning must fire -- otherwise this test would pass
        # vacuously by never exercising the fallback path at all.
        assert any("absolute rating cannot be derived" in w.value for w in at.warning)
        assert _offered_tickers(at) == ["CVX", "XOM"]
        assert "NVDA" not in _offered_tickers(at)


class TestProfileTiltsReachTheScreener:
    """Picking Income/Conservative must serve their own stored sub-scores.

    Those two profiles re-SCORE two categories rather than re-weighting them
    (Section 23), so the nightly stores separate rows for them. Until this
    wiring existed the page always read the balanced ranking and applied the
    profile's weights to it, which is a different -- and quietly wrong -- answer.
    """

    @pytest.fixture
    def two_profile_engine(self, tmp_path: Path) -> Engine:
        engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
        Base.metadata.create_all(engine)
        with sessionmaker(bind=engine)() as session:
            for symbol, sector, composite in _UNIVERSE:
                session.add(
                    Ticker(
                        symbol=symbol,
                        name=f"{symbol} Inc",
                        sector=sector,
                        asset_type="equity",
                        is_active=True,
                    )
                )
                for profile, offset in (("balanced", 0.0), ("conservative", 100.0)):
                    # The conservative rows deliberately invert the ranking, so
                    # "which profile's rows am I looking at" is unmistakable.
                    score = composite if profile == "balanced" else offset - composite
                    session.add(
                        CompositeScore(
                            symbol=symbol,
                            date=AS_OF,
                            profile=profile,
                            composite_score=score,
                            percentile_rank=score,
                            rating="buy",
                            data_confidence=80.0,
                            **{f"{category}_score": score for category in CATEGORIES},
                        )
                    )
            session.commit()
        return engine

    @staticmethod
    def _run(engine: Engine, profile: str) -> AppTest:
        with _wired(engine):
            at = AppTest.from_file(SCREENER, default_timeout=120)
            at.run()
            next(s for s in at.selectbox if s.label == "Start from profile").set_value(profile)
            at.run()
        return at

    @staticmethod
    def _top_symbol(at: AppTest) -> str:
        """The Compare picker defaults to the top two rows; take the first."""
        compare = next(m for m in at.multiselect if m.label == "Tickers")
        return str(compare.value[0]).split(" ")[0]

    def test_balanced_and_conservative_rank_differently(self, two_profile_engine: Engine) -> None:
        balanced = self._run(two_profile_engine, "balanced")
        conservative = self._run(two_profile_engine, "conservative")
        assert not balanced.exception and not conservative.exception
        assert self._top_symbol(balanced) == "NVDA"  # highest balanced score
        assert self._top_symbol(conservative) == "CVX"  # highest conservative score

    def test_a_profile_with_no_stored_rows_says_so(self, legacy_engine: Engine) -> None:
        # `legacy_engine` stores only the balanced profile.
        at = self._run(legacy_engine, "conservative")
        assert not at.exception
        assert any("re-scores two categories" in w.value for w in at.warning)


class TestCompareRadarsWithThinCoverage:
    """Two tickers that cannot be plotted must not take the page down.

    Streamlit derives an element's internal id from its type and contents, so
    two charts that happen to be *identical* collide and the whole script dies
    with `StreamlitDuplicateElementId` -- a red traceback where the Screener
    should be, not a missing chart.

    Two Compare radars are identical exactly when neither can be drawn:
    `subscore_radar` needs three scored categories and otherwise returns the
    same "not enough scored categories" placeholder for every symbol. That is
    the deployed demo's normal state -- five weeks of history means only the
    price-derived categories have data -- so the live Screener page was down,
    and every test seeded enough categories to miss it.
    """

    @pytest.fixture
    def thin_engine(self, tmp_path: Path) -> Engine:
        engine = create_engine(f"sqlite:///{tmp_path / 'thin.db'}")
        Base.metadata.create_all(engine)
        with sessionmaker(bind=engine)() as session:
            for symbol, sector, composite in _UNIVERSE:
                session.add(
                    Ticker(
                        symbol=symbol,
                        name=f"{symbol} Inc",
                        sector=sector,
                        asset_type="equity",
                        is_active=True,
                    )
                )
                session.add(
                    CompositeScore(
                        symbol=symbol,
                        date=AS_OF,
                        profile="balanced",
                        composite_score=composite,
                        percentile_rank=composite,
                        rating="buy",
                        data_confidence=30.0,
                        # Only two categories have data -- one short of what a
                        # radar needs, which is what makes the charts identical.
                        technical_score=composite,
                        smart_money_score=composite,
                    )
                )
            session.commit()
        return engine

    def test_the_page_survives_two_unplottable_radars(self, thin_engine: Engine) -> None:
        with _wired(thin_engine):
            at = AppTest.from_file(SCREENER, default_timeout=120)
            at.run()
        assert not at.exception, (
            "identical placeholder charts collided on their auto-generated element id"
        )
        assert "Compare" in [element.value for element in at.subheader]
        # Anti-vacuity: the loop draws a chart and *then* captions it with the
        # symbol, so the collision killed the script partway through and the
        # later symbols' captions never appeared. Seeing every compared symbol
        # captioned proves the loop ran to completion rather than the section
        # merely being skipped.
        compare = next(m for m in at.multiselect if m.label == "Tickers")
        picked = [str(option).split(" ")[0] for option in compare.value]
        assert len(picked) >= 2
        captions = [str(element.value) for element in at.caption]
        for symbol in picked:
            assert symbol in captions, f"{symbol}'s radar caption never rendered"
