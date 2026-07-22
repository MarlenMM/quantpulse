"""Investor profile presets — named starting points on the composite weight table (Section 23).

The composite scoring table (Section 7.5) is one reasonable configuration among
many. Rather than making a user fiddle with seven weight sliders, each profile
picks a sensible set of category weights that emphasizes what that kind of
investor cares about. Crucially (Section 23), a profile *reweights the same
seven-category table* -- it is not a separate scoring system -- so the exact
same underlying sub-scores produce visibly different, sensible rankings
depending on stated goals, and the sliders stay editable underneath.

The seven categories are the columns of `composite_scores` (Section 13):

    fundamental, technical, analyst, sentiment (Tier-1 company news/social),
    momentum (momentum / risk-adjusted return),
    industry_macro (Tier-2 industry + Tier-3 macro news), smart_money

Every profile's weights must cover exactly these seven and sum to 1.0; the
shifts from `balanced` are documented per profile and mirror Section 23's
table. Two profiles also carry a non-weight tilt that changes how a category is
*scored* (not just weighted), so the plan's intent isn't lost when a shift is
about metric emphasis rather than category weight:

- `income` sets `income_tilt` -- the Screener additionally favors higher-yield
  names, and the fundamental sub-score leans on dividend metrics (Section 23's
  "Fundamental reweighted toward dividend/payout-ratio metrics ... also filters
  the Screener toward higher-yield names").
- `conservative` sets `prefer_low_volatility` -- the momentum/risk-adjusted
  sub-score is scored toward *lower* volatility rather than raw momentum
  (Section 23's "Momentum/Risk-adjusted reweighted toward low-volatility").

These weights are a documented, tunable starting point, not fit to any
backtest (Section 22's overfitting-the-weights caution applies).
"""

from dataclasses import dataclass

# The seven composite categories, in the order they appear in Section 7.5's
# weight table. Every profile's weights are keyed by exactly these.
CATEGORIES: tuple[str, ...] = (
    "fundamental",
    "technical",
    "analyst",
    "sentiment",
    "momentum",
    "industry_macro",
    "smart_money",
)
_CATEGORY_SET = frozenset(CATEGORIES)

DEFAULT_PROFILE = "balanced"


@dataclass(frozen=True)
class InvestorProfile:
    """A named set of composite category weights, plus any non-weight scoring tilts."""

    name: str
    weights: dict[str, float]
    description: str = ""
    # Non-weight tilts (Section 23): these change how a category is *scored*,
    # not just how much it counts.
    income_tilt: bool = False  # favor higher yield in screener + dividend-leaning fundamentals
    prefer_low_volatility: bool = False  # momentum sub-score rewards lower volatility

    def __post_init__(self) -> None:
        keys = frozenset(self.weights)
        if keys != _CATEGORY_SET:
            missing = _CATEGORY_SET - keys
            extra = keys - _CATEGORY_SET
            raise ValueError(
                f"{self.name} weights must cover exactly {sorted(_CATEGORY_SET)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        total = sum(self.weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"{self.name} weights must sum to 1.0, got {total}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError(f"{self.name} weights must be non-negative")


# Balanced (default): Section 7.5's table exactly.
_BALANCED = {
    "fundamental": 0.25,
    "technical": 0.20,
    "analyst": 0.10,
    "sentiment": 0.10,
    "momentum": 0.15,
    "industry_macro": 0.10,
    "smart_money": 0.10,
}

# Value: cheap-relative-to-fundamentals. Fundamental up, momentum down (Section 23).
_VALUE = {
    "fundamental": 0.35,
    "technical": 0.20,
    "analyst": 0.10,
    "sentiment": 0.10,
    "momentum": 0.05,
    "industry_macro": 0.10,
    "smart_money": 0.10,
}

# Growth: revenue/earnings growth + momentum. Momentum and technical up,
# valuation-heavy fundamental down (Section 23).
_GROWTH = {
    "fundamental": 0.12,
    "technical": 0.28,
    "analyst": 0.10,
    "sentiment": 0.10,
    "momentum": 0.25,
    "industry_macro": 0.08,
    "smart_money": 0.07,
}

# Income: dividend safety/yield. Fundamental up (dividend-leaning via
# income_tilt), technical down; screener also favors higher yield (Section 23).
_INCOME = {
    "fundamental": 0.40,
    "technical": 0.10,
    "analyst": 0.10,
    "sentiment": 0.10,
    "momentum": 0.10,
    "industry_macro": 0.10,
    "smart_money": 0.10,
}

# Momentum/Active: short-term technical + news-driven moves. Technical and
# company/industry news up, fundamental down (Section 23).
_MOMENTUM_ACTIVE = {
    "fundamental": 0.10,
    "technical": 0.28,
    "analyst": 0.10,
    "sentiment": 0.15,
    "momentum": 0.15,
    "industry_macro": 0.15,
    "smart_money": 0.07,
}

# Conservative: lower volatility, lower concentration. Momentum/risk-adjusted
# reweighted toward low-volatility (prefer_low_volatility); the noisier,
# shorter-horizon smart-money and sentiment signals down (Section 23).
_CONSERVATIVE = {
    "fundamental": 0.28,
    "technical": 0.18,
    "analyst": 0.12,
    "sentiment": 0.05,
    "momentum": 0.17,
    "industry_macro": 0.12,
    "smart_money": 0.08,
}


PROFILES: dict[str, InvestorProfile] = {
    "balanced": InvestorProfile(
        "balanced", dict(_BALANCED), "Section 7.5's default table, unchanged."
    ),
    "value": InvestorProfile(
        "value", dict(_VALUE), "Cheap relative to fundamentals: fundamental up, momentum down."
    ),
    "growth": InvestorProfile(
        "growth",
        dict(_GROWTH),
        "Growth + momentum: technical and momentum up, valuation-heavy fundamental down.",
    ),
    "income": InvestorProfile(
        "income",
        dict(_INCOME),
        "Dividend safety/yield: fundamental up (dividend-leaning), technical down; "
        "screener favors higher yield.",
        income_tilt=True,
    ),
    "momentum_active": InvestorProfile(
        "momentum_active",
        dict(_MOMENTUM_ACTIVE),
        "Short-term technical + news-driven: technical and company/industry news up, "
        "fundamental down.",
    ),
    "conservative": InvestorProfile(
        "conservative",
        dict(_CONSERVATIVE),
        "Lower volatility: momentum scored toward low volatility; smart-money and sentiment down.",
        prefer_low_volatility=True,
    ),
}


def get_profile(name: str | None) -> InvestorProfile:
    """The `InvestorProfile` for `name` (case-insensitive), or the balanced default.

    An unknown name falls back to `balanced` rather than raising -- a bad
    profile string should degrade to the sensible default, not break scoring.
    """
    if name is None:
        return PROFILES[DEFAULT_PROFILE]
    return PROFILES.get(name.strip().lower(), PROFILES[DEFAULT_PROFILE])


def profile_names() -> list[str]:
    """All available profile names, balanced first (the onboarding default)."""
    ordered = [DEFAULT_PROFILE] + [n for n in PROFILES if n != DEFAULT_PROFILE]
    return ordered
