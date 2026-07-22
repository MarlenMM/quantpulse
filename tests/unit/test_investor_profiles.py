import pytest

from quantpulse.analysis import investor_profiles as ip


class TestProfileValidity:
    def test_every_profile_covers_the_seven_categories_and_sums_to_one(self) -> None:
        for name in ip.profile_names():
            profile = ip.get_profile(name)
            assert set(profile.weights) == set(ip.CATEGORIES)
            assert sum(profile.weights.values()) == pytest.approx(1.0)
            assert all(w >= 0 for w in profile.weights.values())

    def test_bad_weight_sum_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ip.InvestorProfile("bad", dict.fromkeys(ip.CATEGORIES, 0.5))

    def test_missing_category_is_rejected(self) -> None:
        weights = {c: 1 / 6 for c in ip.CATEGORIES[:-1]}  # drops one category
        with pytest.raises(ValueError, match="must cover exactly"):
            ip.InvestorProfile("bad", weights)


class TestProfileShifts:
    def test_value_raises_fundamental_and_lowers_momentum(self) -> None:
        base, value = ip.get_profile("balanced"), ip.get_profile("value")
        assert value.weights["fundamental"] > base.weights["fundamental"]
        assert value.weights["momentum"] < base.weights["momentum"]

    def test_growth_raises_momentum_and_technical_lowers_fundamental(self) -> None:
        base, growth = ip.get_profile("balanced"), ip.get_profile("growth")
        assert growth.weights["momentum"] > base.weights["momentum"]
        assert growth.weights["technical"] > base.weights["technical"]
        assert growth.weights["fundamental"] < base.weights["fundamental"]

    def test_income_raises_fundamental_lowers_technical_and_sets_tilt(self) -> None:
        base, income = ip.get_profile("balanced"), ip.get_profile("income")
        assert income.weights["fundamental"] > base.weights["fundamental"]
        assert income.weights["technical"] < base.weights["technical"]
        assert income.income_tilt is True

    def test_momentum_active_raises_technical_and_news_lowers_fundamental(self) -> None:
        base, active = ip.get_profile("balanced"), ip.get_profile("momentum_active")
        assert active.weights["technical"] > base.weights["technical"]
        assert active.weights["sentiment"] > base.weights["sentiment"]
        assert active.weights["industry_macro"] > base.weights["industry_macro"]
        assert active.weights["fundamental"] < base.weights["fundamental"]

    def test_conservative_lowers_smart_money_and_sentiment_and_sets_low_vol(self) -> None:
        base, cons = ip.get_profile("balanced"), ip.get_profile("conservative")
        assert cons.weights["smart_money"] < base.weights["smart_money"]
        assert cons.weights["sentiment"] < base.weights["sentiment"]
        assert cons.prefer_low_volatility is True


class TestLookup:
    def test_unknown_and_none_fall_back_to_balanced(self) -> None:
        assert ip.get_profile(None).name == "balanced"
        assert ip.get_profile("does-not-exist").name == "balanced"

    def test_lookup_is_case_insensitive(self) -> None:
        assert ip.get_profile("VALUE").name == "value"

    def test_profile_names_lists_balanced_first(self) -> None:
        assert ip.profile_names()[0] == "balanced"
        assert set(ip.profile_names()) == set(ip.PROFILES)
