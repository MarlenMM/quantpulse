"""The regime label's cutoffs are one public pair of numbers (finding 29).

Both gauges drew their zones from literals (35 / 65) while the label switched at
35 / 60, so the picture and the word disagreed for every score from 60 to 65 --
2 of the 32 days in the published database. The gauges now draw from these, so
what is asserted here is that they are the label's own boundaries.
"""

from quantpulse.news_intelligence import market_regime


def test_the_label_changes_exactly_at_the_published_cutoffs() -> None:
    assert market_regime.label_for(market_regime.RISK_ON_AT) == "risk_on"
    assert market_regime.label_for(market_regime.RISK_ON_AT - 0.01) == "neutral"
    assert market_regime.label_for(market_regime.RISK_OFF_AT) == "risk_off"
    assert market_regime.label_for(market_regime.RISK_OFF_AT + 0.01) == "neutral"


def test_the_cutoffs_are_ordered_inside_the_scale() -> None:
    assert 0 < market_regime.RISK_OFF_AT < market_regime.RISK_ON_AT < 100
