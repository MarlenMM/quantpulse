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

from lib import data
from lib.format import format_percent, format_ratio
from lib.glossary import tip

st.set_page_config(page_title="QuantPulse — Track Record", page_icon="📊", layout="wide")


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


def main() -> None:
    st.title("📊 Backtest / Track Record")
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

    columns = st.columns(4)
    columns[0].metric("Sharpe", format_ratio(latest["sharpe"]), help=tip("Sharpe ratio"))
    columns[0].caption(
        interval_caption(
            latest["sharpe_ci_low"], latest["sharpe_ci_high"], latest["ci_confidence_level"]
        )
    )
    columns[1].metric("CAGR", format_percent(latest["cagr"]), help=tip("CAGR"))
    columns[1].caption(
        interval_caption(
            latest["cagr_ci_low"], latest["cagr_ci_high"], latest["ci_confidence_level"]
        )
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
        "per unit of turnover — a conservative bid-ask stand-in. The benchmark is an "
        "equal-weight universe proxy (no S&P 500 price series is ingested)."
    )

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
