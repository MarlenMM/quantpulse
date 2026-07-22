"""Smart Money Signals: insider + institutional + options + short interest (Section 24).

Combines the four raw ingestion clients from Phase 5's first Sonnet part
(`edgar_client.fetch_insider_transactions`, `edgar_13f_client.
fetch_institutional_ownership_trend`, `options_client.fetch_options_signals`,
`short_interest_client.fetch_short_interest`) into one per-symbol result,
matching every other category-scoring module's shape (`fundamental.py`,
`analyst_consensus.py`): a pure function over already-fetched data, producing
this category's own 0-100 raw score (Section 7.5 step 1) -- the later,
separate cross-sectional percentile pass across all seven categories
(Section 7.5 step 2) is Phase 6's job, not this module's.

Three sub-signals are genuinely directional and combine into `score`:

- **Insider activity**: net open-market buy/sell share volume, restricted to
  transaction codes P (open-market purchase) and S (open-market sale) --
  deliberately excluding grants ("A"), option exercises ("M"), and
  tax-withholding dispositions ("F"): those are compensation mechanics an
  insider doesn't choose, not the discretionary "put my own money on this"
  decision Section 24 is actually asking about, and counting them would add
  noise, not signal. "Weighted more heavily when several distinct insiders
  act together" (Section 24) is a multiplier on the net buy/sell tilt, not
  an additive bonus, so a cluster of buyers can push an already-positive
  signal further, not manufacture one from nothing.
- **Institutional ownership trend**: this quarter's `change_from_prior_quarter`
  as a % of the prior quarter's total, since a 10,000-share swing means very
  different things for a mega-cap and a micro-cap.
- **Options positioning**: the put/call ratio, scored in *log*-space --
  a ratio of 2.0 (2x more puts) and 0.5 (2x more calls) are equally extreme
  in opposite directions, but only symmetric on a log scale, not a linear
  one. IV / IV-rank are informative about *how much the market expects to
  move*, not which direction, so they're returned as context (`iv_rank` is
  itself only meaningful once several weeks of `options_client.
  compute_iv_rank` history exist -- a disclosed cold-start limitation) and
  never pushed into the directional blend.

**Short interest is deliberately excluded from `score`'s directional math.**
Section 24 is explicit and emphatic about this: "read this one carefully
rather than treating it as simply bullish or bearish... present both
readings in the UI rather than collapsing it into a single directional
signal, since collapsing it would misrepresent what the data actually
tells you." Giving it any directional pull in the blended score would be
exactly that collapse, one level removed. Its `pct_float_short` and
`days_to_cover` readings are always returned in full on `ShortInterestReading`
so a caller has everything needed to present both interpretations (bearish
conviction vs. squeeze setup) rather than a single blended verdict.

Like `fundamental.py`, `score` is renormalized over whichever of the three
directional sub-signals actually had usable data for this symbol (a company
with no options chain isn't punished with a phantom zero), and `coverage`
(0-1, the fraction of configured weight that had usable data) is returned
alongside it -- the same data-completeness idea Section 7.5 calls for.
"""

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Only genuinely discretionary, open-market transactions count as signal
# (module docstring) -- everything else is compensation mechanics.
_OPEN_MARKET_BUY_CODE = "P"
_OPEN_MARKET_SELL_CODE = "S"

# Each additional distinct insider acting the SAME direction as the net tilt
# amplifies it by this much, capped at _MAX_CLUSTER_MULTIPLIER -- e.g. 3
# distinct buyers (2 beyond the first) -> 1 + 2*0.15 = 1.30x. Deliberately a
# multiplier, not an additive bonus: a cluster can't manufacture a signal
# out of net-zero activity, only amplify a real net tilt (module docstring).
_CLUSTER_STEP = 0.15
_MAX_CLUSTER_MULTIPLIER = 2.0

# A 10% QoQ swing in institutional shares held is already a substantial move
# for a rough, quarterly-lagging proxy (Section 24 calls it exactly that) --
# scaled so +/-10% saturates the full 0-100 range, not an empirically fit
# threshold.
_INSTITUTIONAL_FULL_SWING_PCT = 10.0

# A put/call ratio of 2.0 (twice as many puts as calls) saturates the score;
# log(2.0) so the log-space scaling (module docstring) is symmetric.
_OPTIONS_FULL_SWING_LOG_RATIO = math.log(2.0)
# Floor for a zero put/call ratio (no puts at all) so log() doesn't blow up --
# corresponds to a ~150:1 call:put skew, already far past saturation.
_MIN_PUT_CALL_RATIO_FOR_LOG = 1.0 / 150.0

# A commonly-used "notably high" short-interest threshold (10% of float) --
# flagged for context, not scored directionally (module docstring).
_ELEVATED_SHORT_INTEREST_PCT = 10.0

# Insider activity is the most confidently-framed signal in Section 24
# ("clusters are meaningful... much stronger signal"); institutional trend is
# explicitly called "a rough... proxy" updating only quarterly, so it counts
# for less; options positioning sits in between. These three weights (must
# sum to 1.0) are a documented, tunable starting point, not derived from data.
_SUB_SCORE_WEIGHTS = {"insider": 0.45, "institutional": 0.25, "options": 0.30}


@dataclass(frozen=True)
class InsiderActivityScore:
    """Net open-market insider buy/sell activity. `score` is None with no qualifying trades."""

    score: float | None
    net_shares: float
    buy_shares: float
    sell_shares: float
    distinct_buyers: int
    distinct_sellers: int


@dataclass(frozen=True)
class InstitutionalTrendScore:
    """This quarter's institutional-ownership change vs. the prior quarter."""

    score: float | None
    pct_change_from_prior_quarter: float | None
    total_shares_held: float | None
    num_filers: int | None


@dataclass(frozen=True)
class OptionsPositioningScore:
    """Put/call-ratio-derived score, plus IV context (not blended directionally)."""

    score: float | None
    put_call_ratio: float | None
    atm_implied_volatility: float | None
    iv_rank: float | None


@dataclass(frozen=True)
class ShortInterestReading:
    """Both short-interest readings, deliberately not reduced to one score (module docstring)."""

    pct_float_short: float | None
    days_to_cover: float | None
    elevated: bool


@dataclass(frozen=True)
class SmartMoneyScore:
    """One symbol's Smart Money Signals result (Section 24), feeding the composite (Section 7.5)."""

    symbol: str
    score: float | None
    coverage: float
    insider: InsiderActivityScore
    institutional: InstitutionalTrendScore
    options: OptionsPositioningScore
    short_interest: ShortInterestReading


def score_insider_activity(transactions: pd.DataFrame) -> InsiderActivityScore:
    """Net open-market insider buy/sell activity from a `fetch_insider_transactions` DataFrame.

    `score` (0-100, 50=balanced) is `None` if there are no open-market (P/S)
    transactions at all -- an insider-quiet stock isn't "neutral," it's
    "no signal," and the two shouldn't look the same to a caller.
    """
    if transactions.empty or "transaction_code" not in transactions.columns:
        return InsiderActivityScore(None, 0.0, 0.0, 0.0, 0, 0)

    relevant = transactions[
        transactions["transaction_code"].isin([_OPEN_MARKET_BUY_CODE, _OPEN_MARKET_SELL_CODE])
    ]
    if relevant.empty:
        return InsiderActivityScore(None, 0.0, 0.0, 0.0, 0, 0)

    buys = relevant[relevant["transaction_code"] == _OPEN_MARKET_BUY_CODE]
    sells = relevant[relevant["transaction_code"] == _OPEN_MARKET_SELL_CODE]

    buy_shares = float(buys["shares"].fillna(0).sum())
    sell_shares = float(sells["shares"].fillna(0).sum())
    net_shares = buy_shares - sell_shares
    total_shares = buy_shares + sell_shares
    distinct_buyers = int(buys["insider_name"].nunique())
    distinct_sellers = int(sells["insider_name"].nunique())

    if total_shares <= 0:
        return InsiderActivityScore(
            None, net_shares, buy_shares, sell_shares, distinct_buyers, distinct_sellers
        )

    net_ratio = net_shares / total_shares  # in [-1, 1]: +1 all-buying, -1 all-selling
    cluster_count = distinct_buyers if net_shares >= 0 else distinct_sellers
    cluster_multiplier = min(
        1.0 + _CLUSTER_STEP * max(cluster_count - 1, 0), _MAX_CLUSTER_MULTIPLIER
    )
    tilt = max(-1.0, min(1.0, net_ratio * cluster_multiplier))

    return InsiderActivityScore(
        score=50.0 + 50.0 * tilt,
        net_shares=net_shares,
        buy_shares=buy_shares,
        sell_shares=sell_shares,
        distinct_buyers=distinct_buyers,
        distinct_sellers=distinct_sellers,
    )


def score_institutional_trend(
    row: "pd.Series[Any] | dict[str, Any] | None",
) -> InstitutionalTrendScore:
    """Institutional-ownership QoQ trend score from one `fetch_institutional_ownership_trend` row.

    `row` may be `None` (symbol absent from that quarter's matched holdings
    entirely) or a mapping with `total_shares_held` / `change_from_prior_quarter`.
    `score` is `None` when there's no comparable prior-quarter figure -- an
    honestly unknown trend, not a guessed flat 0% (Section 22).
    """
    if row is None:
        return InstitutionalTrendScore(None, None, None, None)

    total_shares_held = row.get("total_shares_held")
    change = row.get("change_from_prior_quarter")
    num_filers = row.get("num_filers")

    if total_shares_held is None or change is None or pd.isna(total_shares_held) or pd.isna(change):
        return InstitutionalTrendScore(
            None,
            None,
            float(total_shares_held)
            if total_shares_held is not None and pd.notna(total_shares_held)
            else None,
            int(num_filers) if num_filers is not None and pd.notna(num_filers) else None,
        )

    prior_shares_held = total_shares_held - change
    if prior_shares_held <= 0:
        return InstitutionalTrendScore(
            None,
            None,
            float(total_shares_held),
            int(num_filers) if num_filers is not None else None,
        )

    pct_change = change / prior_shares_held * 100.0
    scaled = max(-1.0, min(1.0, pct_change / _INSTITUTIONAL_FULL_SWING_PCT))

    return InstitutionalTrendScore(
        score=50.0 + 50.0 * scaled,
        pct_change_from_prior_quarter=pct_change,
        total_shares_held=float(total_shares_held),
        num_filers=int(num_filers) if num_filers is not None else None,
    )


def score_options_positioning(
    options_signals: dict[str, Any], *, iv_rank: float | None = None
) -> OptionsPositioningScore:
    """Put/call-ratio score (log-scaled, module docstring) from a `fetch_options_signals` dict.

    `iv_rank`, when the caller has enough accumulated history to supply one
    (`options_client.compute_iv_rank`), is carried as context only -- IV
    level says how much movement is expected, not which direction, so it
    never contributes to `score`.
    """
    put_call_ratio = options_signals.get("put_call_ratio")
    atm_iv = options_signals.get("atm_implied_volatility")

    if put_call_ratio is None or put_call_ratio < 0:
        return OptionsPositioningScore(None, put_call_ratio, atm_iv, iv_rank)

    log_ratio = math.log(max(put_call_ratio, _MIN_PUT_CALL_RATIO_FOR_LOG))
    scaled = max(-1.0, min(1.0, -log_ratio / _OPTIONS_FULL_SWING_LOG_RATIO))

    return OptionsPositioningScore(
        score=50.0 + 50.0 * scaled,
        put_call_ratio=put_call_ratio,
        atm_implied_volatility=atm_iv,
        iv_rank=iv_rank,
    )


def read_short_interest(short_interest: dict[str, Any]) -> ShortInterestReading:
    """Both short-interest readings, unreduced (module docstring). Never contributes to `score`."""
    pct_float_short = short_interest.get("pct_float_short")
    days_to_cover = short_interest.get("days_to_cover")
    elevated = pct_float_short is not None and pct_float_short >= _ELEVATED_SHORT_INTEREST_PCT
    return ShortInterestReading(pct_float_short, days_to_cover, elevated)


def compute_smart_money_score(
    symbol: str,
    *,
    insider_transactions: pd.DataFrame,
    institutional_trend_row: "pd.Series[Any] | dict[str, Any] | None",
    options_signals: dict[str, Any],
    short_interest: dict[str, Any],
    iv_rank: float | None = None,
) -> SmartMoneyScore:
    """`symbol`'s combined Smart Money Signals result (Section 24), from the four raw ingestion
    client outputs.

    `score` (0-100) blends `insider`/`institutional`/`options` (module
    docstring explains why `short_interest` never contributes to it),
    renormalized over whichever of the three had usable data -- exactly
    `fundamental.py`'s pattern, so a symbol missing one signal (e.g. no
    options chain) isn't penalized with a phantom neutral/zero for it.
    `coverage` (0-1) is the fraction of the three's configured weight that
    had usable data.
    """
    insider = score_insider_activity(insider_transactions)
    institutional = score_institutional_trend(institutional_trend_row)
    options = score_options_positioning(options_signals, iv_rank=iv_rank)
    short_interest_reading = read_short_interest(short_interest)

    sub_scores = {
        "insider": insider.score,
        "institutional": institutional.score,
        "options": options.score,
    }
    available_weight = sum(
        _SUB_SCORE_WEIGHTS[name] for name, score in sub_scores.items() if score is not None
    )
    total_weight = sum(_SUB_SCORE_WEIGHTS.values())

    if available_weight <= 0:
        blended_score = None
    else:
        weighted_sum = sum(
            _SUB_SCORE_WEIGHTS[name] * score
            for name, score in sub_scores.items()
            if score is not None
        )
        blended_score = weighted_sum / available_weight

    coverage = available_weight / total_weight

    return SmartMoneyScore(
        symbol=symbol,
        score=blended_score,
        coverage=coverage,
        insider=insider,
        institutional=institutional,
        options=options,
        short_interest=short_interest_reading,
    )
