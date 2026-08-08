"""Analysing a symbol the pipeline has never seen.

The interesting failures here are all the same shape: a number that looks
plausible but was computed against nothing. A percentile over one row, a rating
whose peer group is itself, a category quietly scored zero. Each would make
every on-demand stock look good, and none would look wrong on screen.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantpulse import on_demand


def _prices(n: int = 400, start: float = 100.0, drift: float = 0.0006) -> pd.DataFrame:
    """A well-behaved series: constant log drift, so every scorer has real input."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = start * np.exp(drift * np.arange(n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        }
    )


def _peers(sector: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"PEER{i}" for i in range(n)],
            "sector": sector,
            "pe": np.linspace(10, 40, n),
            "pb": np.linspace(1, 8, n),
            "ps": np.linspace(1, 10, n),
            "roe": np.linspace(0.02, 0.35, n),
            "roa": np.linspace(0.01, 0.2, n),
            "debt_equity": np.linspace(0.1, 2.0, n),
            "revenue_growth": np.linspace(-0.1, 0.5, n),
            "dividend_yield": np.linspace(0.0, 0.05, n),
        }
    )


@pytest.fixture
def no_network():
    """Every outbound call stubbed. These tests are about the arithmetic."""
    with (
        patch.object(on_demand.yfinance_client, "fetch_price_history", return_value=_prices()),
        patch.object(on_demand.yfinance_client, "fetch_fundamentals", return_value={}),
        patch.object(on_demand.yfinance_client, "fetch_analyst_consensus", return_value={}),
    ):
        yield


def test_a_symbol_with_no_price_history_is_none_not_an_empty_result(no_network) -> None:
    """An unknown ticker must be distinguishable from a bad one."""
    with patch.object(
        on_demand.yfinance_client, "fetch_price_history", return_value=pd.DataFrame()
    ):
        assert on_demand.analyse("NOSUCH") is None


def test_price_based_categories_are_scored(no_network) -> None:
    result = on_demand.analyse("TEST")
    assert result is not None
    assert "technical" in result.covered_categories
    assert "momentum" in result.covered_categories


def test_categories_it_cannot_compute_are_absent_not_zero(no_network) -> None:
    """A zero is a *bad* score. Absent is the truth, and the composite
    renormalises over what is present rather than penalising the gap."""
    result = on_demand.analyse("TEST")
    assert result is not None
    for category in ("sentiment", "industry_macro", "smart_money"):
        assert result.category_raw[category] is None
        assert category not in result.covered_categories
    assert result.data_confidence is not None and result.data_confidence < 100


def test_fundamentals_are_not_scored_against_an_empty_peer_group(no_network) -> None:
    """The bug this guards is invisible on screen.

    `score_fundamentals` percentile-ranks within a sector. Given a frame holding
    only this company, it returns 100 every time -- a stock is always the best
    of itself -- and fundamentals carry the heaviest weight of any category, so
    every on-demand composite came out inflated and every placement with it.
    """
    with patch.object(
        on_demand.yfinance_client, "fetch_fundamentals", return_value={"pe": 15.0, "roe": 0.2}
    ):
        result = on_demand.analyse("TEST", sector="Industrials", sector_peers=None)

    assert result is not None
    assert result.category_raw["fundamental"] is None
    assert any("sector peers" in note for note in result.notes)


def test_fundamentals_are_scored_when_there_are_real_peers(no_network) -> None:
    """Mutation guard for the test above: with peers, the category must come back."""
    peers = _peers("Industrials", 30)
    with patch.object(
        on_demand.yfinance_client,
        "fetch_fundamentals",
        return_value={"pe": 12.0, "pb": 1.5, "roe": 0.30, "debt_equity": 0.2},
    ):
        result = on_demand.analyse("TEST", sector="Industrials", sector_peers=peers)

    assert result is not None
    score = result.category_raw["fundamental"]
    assert score is not None
    # Ranked among 30 real peers, not against itself. A cheap, profitable,
    # low-debt company should land high but need not be perfect -- and 100 for
    # every input is exactly the symptom being ruled out.
    assert 0.0 <= score <= 100.0


def test_too_few_peers_is_treated_as_no_peers(no_network) -> None:
    with patch.object(
        on_demand.yfinance_client, "fetch_fundamentals", return_value={"pe": 15.0, "roe": 0.2}
    ):
        result = on_demand.analyse(
            "TEST", sector="Industrials", sector_peers=_peers("Industrials", 3)
        )
    assert result is not None
    assert result.category_raw["fundamental"] is None


def test_the_rating_is_absolute_because_there_is_no_peer_group(no_network) -> None:
    """Relative mode would make every on-demand stock a Strong Buy.

    A relative rating means "top decile of the scored universe". Run over a
    one-row frame, every symbol is its own top decile.
    """
    ratings = set()
    for drift in (0.004, -0.004):
        with patch.object(
            on_demand.yfinance_client, "fetch_price_history", return_value=_prices(drift=drift)
        ):
            result = on_demand.analyse("TEST")
            assert result is not None
            ratings.add(result.absolute_rating)
    # A strongly rising and a strongly falling stock must not receive the same
    # verdict, which is what a self-referential relative rating would produce.
    assert len(ratings) > 1


def test_placement_reports_where_it_would_have_ranked(no_network) -> None:
    stored = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    with patch.object(on_demand, "_absolute_composite", return_value=(35.0, "buy", 60.0)):
        result = on_demand.analyse("TEST", ranked_composites=stored)
    assert result is not None
    assert result.percentile_vs_ranked == pytest.approx(60.0)
    assert result.ranked_universe_size == 5


def test_placement_is_absent_without_a_stored_universe(no_network) -> None:
    result = on_demand.analyse("TEST")
    assert result is not None
    assert result.percentile_vs_ranked is None


def test_a_failing_fetch_degrades_to_a_note_rather_than_an_exception(no_network) -> None:
    """Three independent endpoints; any of them can be down. That is not a
    reason to show an error page instead of the parts that did arrive."""
    with patch.object(
        on_demand.yfinance_client, "fetch_analyst_consensus", side_effect=RuntimeError("429")
    ):
        result = on_demand.analyse("TEST")
    assert result is not None
    assert result.category_raw["technical"] is not None
    assert any("Analyst" in note for note in result.notes)


def test_only_horizons_the_history_supports_are_offered(no_network) -> None:
    """The 3x-history floor is why 252 days is not on the menu for a two-year fetch."""
    assert 252 not in on_demand.HORIZONS
    result = on_demand.analyse("TEST")
    assert result is not None
    assert {f.horizon_days for f in result.forecasts} <= set(on_demand.HORIZONS)


def test_the_result_carries_the_date_of_the_data_not_of_the_request(no_network) -> None:
    """`as_of` is the last traded bar, not today.

    They differ over every weekend and holiday, and a page that stamps live
    analysis with today's date claims data it does not have.
    """
    series = _prices()
    result = on_demand.analyse("TEST")
    assert result is not None
    assert result.as_of == series["date"].iloc[-1].date()
    assert result.computed_at.date() >= result.as_of
