import math

import pandas as pd
import pytest

from quantpulse.analysis import smart_money as sm


def _transactions(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """rows: (transaction_code, shares, insider_name)."""
    return pd.DataFrame(rows, columns=["transaction_code", "shares", "insider_name"])


# --- score_insider_activity ----------------------------------------------------


def test_insider_balanced_buy_and_sell_is_neutral() -> None:
    df = _transactions([("P", 1000, "A"), ("S", 1000, "B")])
    result = sm.score_insider_activity(df)
    assert result.score == 50.0
    assert result.net_shares == 0.0


def test_insider_pure_buying_saturates_at_100() -> None:
    df = _transactions([("P", 1000, "A")])
    result = sm.score_insider_activity(df)
    assert result.score == 100.0


def test_insider_pure_selling_saturates_at_0() -> None:
    df = _transactions([("S", 1000, "A")])
    result = sm.score_insider_activity(df)
    assert result.score == 0.0


def test_insider_cluster_of_distinct_buyers_scores_higher_than_solo_buyer() -> None:
    # Same net shares (+50), but one case has 3 distinct buyers vs. 1.
    clustered = _transactions([("P", 100, "A"), ("P", 100, "B"), ("P", 100, "C"), ("S", 250, "D")])
    solo = _transactions([("P", 300, "A"), ("S", 250, "D")])
    clustered_result = sm.score_insider_activity(clustered)
    solo_result = sm.score_insider_activity(solo)
    assert clustered_result.net_shares == solo_result.net_shares == 50
    assert clustered_result.score > solo_result.score
    assert clustered_result.distinct_buyers == 3
    assert solo_result.distinct_buyers == 1


def test_insider_excludes_non_open_market_codes() -> None:
    # Grant (A), tax-withholding (F), option exercise (M) -- none are
    # discretionary open-market decisions and must not count as signal.
    df = _transactions([("A", 500, "A"), ("F", 500, "A"), ("M", 500, "A")])
    result = sm.score_insider_activity(df)
    assert result.score is None


def test_insider_empty_dataframe_returns_none_score() -> None:
    df = pd.DataFrame(columns=["transaction_code", "shares", "insider_name"])
    result = sm.score_insider_activity(df)
    assert result.score is None
    assert result.net_shares == 0.0


def test_insider_missing_transaction_code_column_returns_none() -> None:
    df = pd.DataFrame({"shares": [100]})
    result = sm.score_insider_activity(df)
    assert result.score is None


def test_insider_distinct_buyer_and_seller_counts() -> None:
    df = _transactions([("P", 100, "A"), ("P", 100, "A"), ("S", 50, "B")])
    result = sm.score_insider_activity(df)
    assert result.distinct_buyers == 1  # same insider buying twice is still one buyer
    assert result.distinct_sellers == 1
    assert result.buy_shares == 200
    assert result.sell_shares == 50


# --- score_institutional_trend --------------------------------------------------


def test_institutional_trend_full_swing_growth_saturates_at_100() -> None:
    row = {"total_shares_held": 11000, "change_from_prior_quarter": 1000, "num_filers": 5}
    result = sm.score_institutional_trend(row)
    assert result.score == 100.0
    assert result.pct_change_from_prior_quarter == pytest.approx(10.0)


def test_institutional_trend_full_swing_decline_saturates_at_0() -> None:
    row = {"total_shares_held": 9000, "change_from_prior_quarter": -1000, "num_filers": 5}
    result = sm.score_institutional_trend(row)
    assert result.score == 0.0


def test_institutional_trend_no_change_is_neutral() -> None:
    row = {"total_shares_held": 10000, "change_from_prior_quarter": 0, "num_filers": 3}
    result = sm.score_institutional_trend(row)
    assert result.score == 50.0


def test_institutional_trend_partial_swing_is_proportional() -> None:
    # +5% change with a 10% full-swing threshold -> half the range -> 75.
    row = {"total_shares_held": 10500, "change_from_prior_quarter": 500, "num_filers": 4}
    result = sm.score_institutional_trend(row)
    assert result.score == pytest.approx(75.0)


def test_institutional_trend_none_row_returns_none() -> None:
    result = sm.score_institutional_trend(None)
    assert result.score is None
    assert result.total_shares_held is None


def test_institutional_trend_nan_change_returns_none_but_keeps_total() -> None:
    row = {"total_shares_held": 5000.0, "change_from_prior_quarter": float("nan"), "num_filers": 2}
    result = sm.score_institutional_trend(row)
    assert result.score is None
    assert result.total_shares_held == 5000.0


def test_institutional_trend_zero_prior_shares_returns_none() -> None:
    # prior = current - change = 1000 - 1000 = 0 -> can't compute a % change.
    row = {"total_shares_held": 1000, "change_from_prior_quarter": 1000, "num_filers": 1}
    result = sm.score_institutional_trend(row)
    assert result.score is None


# --- score_options_positioning ---------------------------------------------------


def test_options_neutral_ratio_is_50() -> None:
    result = sm.score_options_positioning({"put_call_ratio": 1.0, "atm_implied_volatility": 0.3})
    assert result.score == 50.0


def test_options_log_scale_is_symmetric() -> None:
    bearish = sm.score_options_positioning({"put_call_ratio": 2.0, "atm_implied_volatility": 0.3})
    bullish = sm.score_options_positioning({"put_call_ratio": 0.5, "atm_implied_volatility": 0.3})
    assert bearish.score == pytest.approx(0.0)
    assert bullish.score == pytest.approx(100.0)
    # Symmetric distance from neutral (50), not a raw linear-ratio distance.
    assert (50.0 - bearish.score) == pytest.approx(bullish.score - 50.0)


def test_options_partial_ratio_between_neutral_and_saturation() -> None:
    result = sm.score_options_positioning({"put_call_ratio": 1.4, "atm_implied_volatility": 0.3})
    assert 0.0 < result.score < 50.0  # more puts than calls -> below neutral, not yet saturated


def test_options_zero_ratio_does_not_crash_and_saturates_bullish() -> None:
    result = sm.score_options_positioning({"put_call_ratio": 0.0, "atm_implied_volatility": 0.3})
    assert result.score == pytest.approx(100.0)


def test_options_none_ratio_returns_none_score() -> None:
    result = sm.score_options_positioning({"put_call_ratio": None, "atm_implied_volatility": 0.3})
    assert result.score is None


def test_options_iv_rank_is_carried_as_context_never_affects_score() -> None:
    without = sm.score_options_positioning({"put_call_ratio": 1.0, "atm_implied_volatility": 0.3})
    with_high_iv_rank = sm.score_options_positioning(
        {"put_call_ratio": 1.0, "atm_implied_volatility": 0.3}, iv_rank=95.0
    )
    assert without.score == with_high_iv_rank.score == 50.0
    assert with_high_iv_rank.iv_rank == 95.0
    assert without.iv_rank is None


# --- read_short_interest: never directional -------------------------------------


def test_short_interest_low_is_not_elevated() -> None:
    result = sm.read_short_interest({"pct_float_short": 2.0, "days_to_cover": 1.0})
    assert result.elevated is False
    assert result.pct_float_short == 2.0
    assert result.days_to_cover == 1.0


def test_short_interest_high_is_elevated() -> None:
    result = sm.read_short_interest({"pct_float_short": 25.0, "days_to_cover": 8.0})
    assert result.elevated is True


def test_short_interest_missing_pct_is_not_elevated() -> None:
    result = sm.read_short_interest({"pct_float_short": None, "days_to_cover": None})
    assert result.elevated is False
    assert result.pct_float_short is None


def test_short_interest_reading_has_no_score_field() -> None:
    # Structural guarantee that short interest cannot be mistaken for a
    # directional score -- the dataclass simply has no such field.
    result = sm.read_short_interest({"pct_float_short": 50.0, "days_to_cover": 20.0})
    assert not hasattr(result, "score")


# --- compute_smart_money_score: the blend ---------------------------------------


@pytest.fixture
def full_insider_buy() -> pd.DataFrame:
    return _transactions([("P", 100, "A")])


@pytest.fixture
def full_institutional_growth() -> dict:
    return {"total_shares_held": 11000, "change_from_prior_quarter": 1000, "num_filers": 5}


def test_blend_with_full_coverage_matches_documented_weights(
    full_insider_buy: pd.DataFrame, full_institutional_growth: dict
) -> None:
    result = sm.compute_smart_money_score(
        "AAPL",
        insider_transactions=full_insider_buy,
        institutional_trend_row=full_institutional_growth,
        options_signals={"put_call_ratio": 1.0, "atm_implied_volatility": 0.3},
        short_interest={"pct_float_short": 25.0, "days_to_cover": 8.0},
    )
    assert result.coverage == 1.0
    expected = 0.45 * 100 + 0.25 * 100 + 0.30 * 50
    assert result.score == pytest.approx(expected)


def test_blend_extremely_high_short_interest_does_not_move_the_score(
    full_insider_buy: pd.DataFrame, full_institutional_growth: dict
) -> None:
    common_kwargs = dict(
        insider_transactions=full_insider_buy,
        institutional_trend_row=full_institutional_growth,
        options_signals={"put_call_ratio": 1.0, "atm_implied_volatility": 0.3},
    )
    low_si = sm.compute_smart_money_score(
        "AAPL", short_interest={"pct_float_short": 1.0, "days_to_cover": 0.5}, **common_kwargs
    )
    high_si = sm.compute_smart_money_score(
        "AAPL", short_interest={"pct_float_short": 40.0, "days_to_cover": 15.0}, **common_kwargs
    )
    # The whole point of Section 24's instruction: identical score regardless
    # of how extreme short interest is, because it never enters the blend.
    assert low_si.score == high_si.score
    assert low_si.short_interest.elevated is False
    assert high_si.short_interest.elevated is True


def test_blend_renormalizes_when_one_signal_is_missing(
    full_insider_buy: pd.DataFrame, full_institutional_growth: dict
) -> None:
    result = sm.compute_smart_money_score(
        "MSFT",
        insider_transactions=full_insider_buy,
        institutional_trend_row=full_institutional_growth,
        options_signals={"put_call_ratio": None, "atm_implied_volatility": None},
        short_interest={"pct_float_short": None, "days_to_cover": None},
    )
    # coverage = (0.45 + 0.25) / 1.0
    assert result.coverage == pytest.approx(0.70)
    # Both remaining signals saturate at 100 -> renormalized average still 100.
    assert result.score == pytest.approx(100.0)


def test_blend_returns_none_score_and_zero_coverage_when_nothing_available() -> None:
    result = sm.compute_smart_money_score(
        "ZZZZ",
        insider_transactions=pd.DataFrame(columns=["transaction_code", "shares", "insider_name"]),
        institutional_trend_row=None,
        options_signals={"put_call_ratio": None, "atm_implied_volatility": None},
        short_interest={"pct_float_short": None, "days_to_cover": None},
    )
    assert result.score is None
    assert result.coverage == 0.0


def test_blend_preserves_symbol_and_sub_results(
    full_insider_buy: pd.DataFrame, full_institutional_growth: dict
) -> None:
    result = sm.compute_smart_money_score(
        "AAPL",
        insider_transactions=full_insider_buy,
        institutional_trend_row=full_institutional_growth,
        options_signals={"put_call_ratio": 1.0, "atm_implied_volatility": 0.3},
        short_interest={"pct_float_short": 25.0, "days_to_cover": 8.0},
    )
    assert result.symbol == "AAPL"
    assert isinstance(result.insider, sm.InsiderActivityScore)
    assert isinstance(result.institutional, sm.InstitutionalTrendScore)
    assert isinstance(result.options, sm.OptionsPositioningScore)
    assert isinstance(result.short_interest, sm.ShortInterestReading)


def test_sub_score_weights_sum_to_one() -> None:
    assert sum(sm._SUB_SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_blend_iv_rank_is_forwarded_to_options_context(
    full_insider_buy: pd.DataFrame, full_institutional_growth: dict
) -> None:
    result = sm.compute_smart_money_score(
        "AAPL",
        insider_transactions=full_insider_buy,
        institutional_trend_row=full_institutional_growth,
        options_signals={"put_call_ratio": 1.0, "atm_implied_volatility": 0.3},
        short_interest={"pct_float_short": None, "days_to_cover": None},
        iv_rank=72.0,
    )
    assert result.options.iv_rank == 72.0
    assert math.isclose(result.score, 0.45 * 100 + 0.25 * 100 + 0.30 * 50)
