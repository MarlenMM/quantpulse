import pandas as pd
import pytest

from quantpulse.analysis.fundamental import (
    SECTOR_CONFIGS,
    SectorFundamentalConfig,
    compute_p_ffo,
    get_sector_config,
    score_fundamentals,
)


class TestSectorConfigs:
    def test_every_configured_sector_weight_sums_to_one(self) -> None:
        for config in SECTOR_CONFIGS.values():
            assert sum(config.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_rejects_a_config_whose_weights_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            SectorFundamentalConfig("Bogus", {"pe": 0.5, "pb": 0.2})

    def test_financials_drops_leverage_and_price_to_sales(self) -> None:
        config = get_sector_config("Financials")
        assert "debt_equity" not in config.weights
        assert "ps" not in config.weights
        assert config.weights["roe"] > SECTOR_CONFIGS["_default"].weights["roe"]

    def test_real_estate_substitutes_p_ffo_for_pe(self) -> None:
        config = get_sector_config("Real Estate")
        assert "p_ffo" in config.weights
        assert "pe" not in config.weights
        assert "roe" not in config.weights

    def test_utilities_upweights_dividend_yield(self) -> None:
        config = get_sector_config("Utilities")
        assert config.weights["div_yield"] > SECTOR_CONFIGS["_default"].weights["div_yield"]

    def test_unknown_sector_falls_back_to_default(self) -> None:
        assert get_sector_config("Some Made Up Sector").sector == "_default"

    def test_none_sector_falls_back_to_default(self) -> None:
        assert get_sector_config(None).sector == "_default"


class TestComputePFfo:
    def test_computes_a_plausible_multiple(self) -> None:
        p_ffo = compute_p_ffo(
            market_cap=60_000_000_000,
            net_income=1_000_000_000,
            depreciation_amortization=2_500_000_000,
        )
        assert p_ffo == pytest.approx(60_000_000_000 / 3_500_000_000)

    def test_none_when_any_input_missing(self) -> None:
        assert compute_p_ffo(None, 1, 2) is None
        assert compute_p_ffo(100, None, 2) is None
        assert compute_p_ffo(100, 1, None) is None

    def test_none_when_ffo_is_non_positive(self) -> None:
        assert compute_p_ffo(100, -50, 10) is None
        assert compute_p_ffo(100, 0, 0) is None


class TestScoreFundamentals:
    def test_cheaper_higher_quality_stock_scores_higher(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "CHEAP",
                    "sector": "Information Technology",
                    "pe": 10,
                    "pb": 2,
                    "revenue_growth": 0.30,
                    "roe": 0.35,
                    "div_yield": 0.02,
                },
                {
                    "symbol": "MID",
                    "sector": "Information Technology",
                    "pe": 20,
                    "pb": 5,
                    "revenue_growth": 0.15,
                    "roe": 0.20,
                    "div_yield": 0.01,
                },
                {
                    "symbol": "PRICEY",
                    "sector": "Information Technology",
                    "pe": 50,
                    "pb": 15,
                    "revenue_growth": 0.05,
                    "roe": 0.10,
                    "div_yield": 0.00,
                },
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        assert result.loc["CHEAP", "fundamental_score"] > result.loc["MID", "fundamental_score"]
        assert result.loc["MID", "fundamental_score"] > result.loc["PRICEY", "fundamental_score"]

    def test_negative_pe_excluded_not_ranked_as_cheapest(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "LOSS",
                    "sector": "Information Technology",
                    "pe": -5,
                    "pb": 3,
                    "roe": -0.2,
                },
                {
                    "symbol": "OK",
                    "sector": "Information Technology",
                    "pe": 15,
                    "pb": 3,
                    "roe": 0.15,
                },
                {
                    "symbol": "EXPENSIVE",
                    "sector": "Information Technology",
                    "pe": 80,
                    "pb": 3,
                    "roe": 0.05,
                },
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        # LOSS's pe is excluded (undefined), so its score comes only from pb/roe
        # (both worse than OK's) -- it must not be treated as the "cheapest" pe.
        assert result.loc["LOSS", "fundamental_score"] < result.loc["OK", "fundamental_score"]
        assert result.loc["LOSS", "coverage"] < 1.0
        assert result.loc["OK", "coverage"] == pytest.approx(1.0)

    def test_negative_roe_is_scored_not_excluded(self) -> None:
        # Unlike a negative P/E, a negative ROE is a real (bad) signal and
        # must still be ranked -- not treated as missing data.
        df = pd.DataFrame(
            [
                {"symbol": "LOSING_MONEY", "sector": "Information Technology", "roe": -0.30},
                {"symbol": "PROFITABLE", "sector": "Information Technology", "roe": 0.25},
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        assert result.loc["LOSING_MONEY", "coverage"] == pytest.approx(
            result.loc["PROFITABLE", "coverage"]
        )
        assert (
            result.loc["LOSING_MONEY", "fundamental_score"]
            < result.loc["PROFITABLE", "fundamental_score"]
        )

    def test_missing_metric_lowers_coverage_but_still_scores(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "FULL",
                    "sector": "Information Technology",
                    "pe": 15,
                    "pb": 3,
                    "roe": 0.15,
                    "div_yield": 0.02,
                    "revenue_growth": 0.1,
                },
                {
                    "symbol": "SPARSE",
                    "sector": "Information Technology",
                    "pe": 15,
                    "pb": None,
                    "roe": 0.15,
                    "div_yield": 0.02,
                    "revenue_growth": 0.1,
                },
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        assert result.loc["SPARSE", "coverage"] < result.loc["FULL", "coverage"]
        assert pd.notna(result.loc["SPARSE", "fundamental_score"])

    def test_scores_are_sector_relative_not_universe_wide(self) -> None:
        # A P/E of 25 is middling for tech but would be expensive for a bank;
        # each stock must only be compared within its own sector.
        df = pd.DataFrame(
            [
                {"symbol": "TECH_A", "sector": "Information Technology", "pe": 10},
                {"symbol": "TECH_B", "sector": "Information Technology", "pe": 25},
                {"symbol": "TECH_C", "sector": "Information Technology", "pe": 50},
                {"symbol": "BANK_A", "sector": "Financials", "pe": 25},
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        # TECH_B (middle of its group) and BANK_A (alone in its group) both
        # have a raw pe of 25, but must not necessarily match -- BANK_A, as
        # the only Financials row, is trivially its own top percentile.
        assert result.loc["BANK_A", "fundamental_score"] == pytest.approx(100.0)
        assert result.loc["TECH_B", "fundamental_score"] < 100.0

    def test_real_estate_uses_p_ffo_and_ignores_pe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "REIT_CHEAP",
                    "sector": "Real Estate",
                    "pe": 999,
                    "p_ffo": 10,
                    "div_yield": 0.05,
                },
                {
                    "symbol": "REIT_PRICEY",
                    "sector": "Real Estate",
                    "pe": 1,
                    "p_ffo": 30,
                    "div_yield": 0.02,
                },
            ]
        )
        result = score_fundamentals(df).set_index("symbol")
        # REIT_CHEAP has a terrible raw pe but the best p_ffo/div_yield --
        # since pe isn't even in the Real Estate config, it must win.
        assert (
            result.loc["REIT_CHEAP", "fundamental_score"]
            > result.loc["REIT_PRICEY", "fundamental_score"]
        )

    def test_raises_without_required_columns(self) -> None:
        with pytest.raises(ValueError, match="symbol"):
            score_fundamentals(pd.DataFrame({"pe": [10, 20]}))

    def test_empty_input_returns_empty_with_correct_columns(self) -> None:
        result = score_fundamentals(pd.DataFrame(columns=["symbol", "sector", "pe"]))
        assert list(result.columns) == ["symbol", "sector", "fundamental_score", "coverage"]
        assert len(result) == 0

    def test_lone_stock_in_a_sector_scores_at_the_top(self) -> None:
        df = pd.DataFrame([{"symbol": "ONLY", "sector": "Energy", "pe": 40, "roe": 0.05}])
        result = score_fundamentals(df)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(100.0)
