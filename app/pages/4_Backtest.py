"""Backtest / Track Record — the algorithm's own honest history (Sections 7.6, 10, 12).

Section 20 calls this "your strongest talking point in an interview," and
Section 7.6 is specific about what makes it credible rather than promotional:
the strategy is compared against a benchmark, costs and a realistic rebalance
cadence are stated rather than assumed away, and the headline Sharpe/CAGR carry
bootstrap confidence intervals so a small-sample lucky streak can't masquerade
as an edge.

So this page leads with the caveats rather than burying them: every metric is
rendered next to its interval where one exists, and where an interval straddles
zero the page says so in words. A track record page that only shows the good
number is worse than no track record page.
"""

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lib import charts, data
from lib.brand import PAGE_ICON
from lib.format import format_percent, format_ratio
from lib.glossary import tip
from quantpulse.portfolio.optimization import kelly_position_fraction

st.set_page_config(page_title="QuantPulse — Track Record", page_icon=PAGE_ICON, layout="wide")


def interval_caption(low: float | None, high: float | None, level: float | None) -> str:
    """ "90% CI [0.30, 1.30]" — or an honest note that the run was too short."""
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "no confidence interval — the run was too short to bootstrap honestly"
    confidence = f"{level:.0%}" if level and not pd.isna(level) else "CI"
    excludes_zero = low > 0 or high < 0
    verdict = (
        "excludes zero" if excludes_zero else "**straddles zero — not distinguishable from luck**"
    )
    return f"{confidence} CI [{low:.2f}, {high:.2f}] — {verdict}"


def render_position_sizing(latest: pd.Series) -> None:
    """Section 27's fractional-Kelly sizing, derived from this run's own record.

    The rest of this page answers "did the strategy work?". This answers "how
    much would it have been rational to bet on it?", which is the question a
    track record is actually *for*. Both inputs come from the same stored run --
    its realized win rate and payoff ratio -- so the number cannot be more
    confident than the history behind it.

    Shown only when the backtest genuinely supports it: a run with no losing
    periods has an undefined payoff ratio, and Kelly on an apparently
    can't-lose bet would return a maximal position.
    """
    hit_rate = latest.get("win_rate")
    payoff = latest.get("payoff_ratio")
    if pd.isna(hit_rate) or pd.isna(payoff):
        return

    fraction = kelly_position_fraction(float(hit_rate), float(payoff))
    if fraction is None:
        return

    st.divider()
    st.subheader("How much to bet", help=tip("Kelly fraction"))
    columns = st.columns(3)
    columns[0].metric("Suggested position", format_percent(fraction))
    columns[1].metric("Win rate used", format_percent(float(hit_rate)))
    columns[2].metric("Payoff ratio used", format_ratio(float(payoff)))

    if fraction <= 0:
        st.info(
            "The Kelly criterion says **do not take this bet at all** — at this win "
            "rate and payoff ratio the strategy has no positive edge to size, so any "
            "position is a losing proposition on average."
        )
    else:
        st.caption(
            f"A **quarter-Kelly** size: the growth-optimal bet given this run's own "
            f"{format_percent(float(hit_rate))} win rate and "
            f"{format_ratio(float(payoff))} payoff ratio, then cut to a quarter because "
            "full Kelly is famously too volatile to live with and is exquisitely "
            "sensitive to an over-estimated edge. Treat it as an upper bound, not a "
            "recommendation — it assumes the future resembles this backtest, which is "
            "exactly the assumption the confidence intervals above tell you to doubt."
        )


def _render_interval(
    column: DeltaGenerator,
    point: float | None,
    low: float | None,
    high: float | None,
    level: float | None,
) -> None:
    """The interval, drawn and then spelled out.

    Both, on purpose. The sentence is the one that has to be right — a reader
    who cannot see the drawing loses nothing. What the drawing buys is that a
    reader who is skimming cannot skim past the interval, which is the failure
    this page exists to prevent.
    """
    whisker = charts.interval_whisker(point, low, high)
    if whisker is not None:
        column.plotly_chart(whisker, width="stretch", config={"displayModeBar": False})
    column.caption(interval_caption(low, high, level))


def main() -> None:
    st.title("What this would have returned")
    runs = data.backtest_history(limit=20)
    if runs.empty:
        st.info(
            "No backtest has been stored yet. `scripts/refresh_data.py` runs the "
            "survivorship- and cost-aware strategy backtest on its weekly cadence."
        )
        st.markdown(
            """
            When it does run, this page shows what Section 7.6 asks for:

            - A "followed the algorithm's ratings" strategy vs a buy-and-hold benchmark
            - Sharpe, CAGR and max drawdown **with bootstrap confidence intervals**
            - The assumed transaction cost and rebalance cadence, stated explicitly
            """
        )
        return

    latest = runs.iloc[0]
    st.caption(
        f"Most recent run **{latest['run_date']}** covering "
        f"{latest['period_start']} → {latest['period_end']} · "
        f"{latest['cadence']} rebalancing · {int(latest['n_periods'])} periods"
    )

    # `st.header`, not another subheader: the estimate and its interval are the
    # page, and everything below is supporting material for reading them.
    st.header("The estimate, and how sure it is")

    columns = st.columns(4)
    columns[0].metric("Sharpe", format_ratio(latest["sharpe"]), help=tip("Sharpe ratio"))
    _render_interval(
        columns[0],
        latest["sharpe"],
        latest["sharpe_ci_low"],
        latest["sharpe_ci_high"],
        latest["ci_confidence_level"],
    )
    columns[1].metric("CAGR", format_percent(latest["cagr"]), help=tip("CAGR"))
    _render_interval(
        columns[1],
        latest["cagr"],
        latest["cagr_ci_low"],
        latest["cagr_ci_high"],
        latest["ci_confidence_level"],
    )
    columns[2].metric(
        "Max drawdown", format_percent(latest["max_drawdown"]), help=tip("Max drawdown")
    )
    columns[2].caption(
        "deliberately not bootstrapped — a path-dependent extremum has no meaningful "
        "resampled interval"
    )
    columns[3].metric(
        "Win rate",
        format_percent(latest["win_rate"]),
        help=tip("Turnover", "Win rate is the share of rebalance periods that ended positive."),
    )
    columns[3].caption(f"average turnover {format_percent(latest['avg_turnover'])} per rebalance")

    st.caption(
        "The bar under Sharpe and CAGR is that bootstrap interval, and the hairline "
        "crossing it is zero. A bar that overlaps the hairline is a result the data has "
        "not separated from luck; it is drawn grey rather than in the accent to say so."
    )

    st.divider()
    st.subheader("Versus benchmark")
    benchmark_columns = st.columns(2)
    benchmark_columns[0].metric(
        "Strategy CAGR",
        format_percent(latest["cagr"]),
        delta=(
            None
            if pd.isna(latest["benchmark_cagr"]) or pd.isna(latest["cagr"])
            else format_percent(latest["cagr"] - latest["benchmark_cagr"])
        ),
    )
    benchmark_columns[1].metric(
        "Benchmark CAGR", format_percent(latest["benchmark_cagr"]), help=tip("Benchmark")
    )
    st.caption(
        f"Transaction cost assumed: **{format_percent(latest['assumed_txn_cost'], digits=2)}** "
        "per unit of turnover — a conservative bid-ask stand-in. The benchmark is the "
        "**equal-weight** universe held buy-and-hold, which is the comparison that isolates "
        "the signal: this strategy equal-weights the names it picks, so measuring it against "
        "an equal-weight version of the whole universe asks *did ranking help*, with the "
        "weighting scheme held fixed. Against the cap-weighted index it would instead be "
        "measuring the ranking and the equal-weight tilt together, as one number."
    )

    render_position_sizing(latest)

    st.warning(
        "**Read this honestly.** These are backtested, hypothetical results on a "
        "survivorship-aware universe with assumed costs — not realized returns, and not "
        "a prediction. A confidence interval that straddles zero means the result has "
        "not been distinguished from luck."
    )

    st.divider()
    st.subheader("Run history")
    display = runs.copy()
    st.dataframe(
        display[
            [
                "run_date",
                "period_start",
                "period_end",
                "n_periods",
                "sharpe",
                "cagr",
                "max_drawdown",
                "benchmark_cagr",
                "assumed_txn_cost",
            ]
        ]
        .rename(
            columns={
                "run_date": "Run",
                "period_start": "From",
                "period_end": "To",
                "n_periods": "Periods",
                "sharpe": "Sharpe",
                "cagr": "CAGR",
                "max_drawdown": "Max DD",
                "benchmark_cagr": "Bench CAGR",
                "assumed_txn_cost": "Cost",
            }
        )
        .style.format(
            {
                "Sharpe": "{:.2f}",
                "CAGR": "{:.1%}",
                "Max DD": "{:.1%}",
                "Bench CAGR": "{:.1%}",
                "Cost": "{:.2%}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download run history as CSV",
        runs.to_csv(index=False).encode("utf-8"),
        file_name="quantpulse_backtest_history.csv",
        mime="text/csv",
    )


main()
