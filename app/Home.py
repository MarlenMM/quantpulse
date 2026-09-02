"""QuantPulse Dashboard -- the app entrypoint (Section 12, page 1).

Run with: `streamlit run app/Home.py`

Streamlit puts this file's directory on `sys.path`, which is what makes
`from lib import ...` work here and in every module under `pages/`.
"""

import streamlit as st

from lib import charts, data
from lib.brand import PAGE_ICON
from lib.format import (
    format_pct_already_scaled,
    format_score,
    freshness_label,
    humanize,
    rating_label,
)
from lib.glossary import tip
from quantpulse.analysis import risk, technical

st.set_page_config(
    page_title="QuantPulse — Dashboard",
    page_icon=PAGE_ICON,
    layout="wide",
    # "auto", not "expanded": Streamlit overlays the sidebar on narrow screens,
    # so pinning it open means a phone visitor lands on a nav menu covering the
    # whole page. "auto" expands on desktop and collapses on mobile, which is
    # the responsive behavior Section 31's mobile pass is asking for.
    initial_sidebar_state="auto",
)

DISCLAIMER = (
    "**Educational/research tool. Not financial advice. Not a registered investment "
    "advisor.** Past backtested performance does not guarantee future results."
)
# Section 31 asks for "a brief first-visit onboarding note (a dismissible info
# box, not a full guided tour)". Dismissal lives in session state, so it stays
# gone while you browse and returns for the next visitor -- which is the right
# behavior for a shared demo link.
_ONBOARDING_KEY = "onboarding_dismissed"


def render_onboarding() -> None:
    if st.session_state.get(_ONBOARDING_KEY):
        return
    with st.container(border=True):
        st.markdown(
            "**New here?** QuantPulse ranks stocks with transparent statistics — no "
            "black box. Start on the **Screener** for the ranked list, click through to "
            "**Stock Detail** for one company, or open **Portfolio** to analyse holdings "
            "you enter yourself. Unfamiliar term? Every number has an ⓘ tooltip, and the "
            "**Glossary** page explains all of them in plain English."
        )
        if st.button("Got it", key="dismiss_onboarding"):
            st.session_state[_ONBOARDING_KEY] = True
            st.rerun()


def render_empty_state() -> None:
    """What a freshly-cloned repo actually shows -- and how to fix it.

    A brand-new checkout has an empty database, so the honest first-run screen
    is instructions rather than a grid of blank charts. Section 6.2's
    cold-start/incremental split is the thing a new user most needs pointed out.
    """
    st.info("No analysis data yet — the pipeline hasn't been run against this database.")
    st.markdown(
        """
        **To populate it:**

        ```bash
        uv run alembic upgrade head                  # create the schema
        uv run python scripts/seed_initial_data.py   # one-time historical backfill
        ```

        The cold-start backfill is a separate, resumable job from the incremental
        refresh (Section 6.2) — it pulls years of history for the whole universe
        and can take a while on the first run. It is also the only step that has
        to happen at a terminal: after it, keep the data current from
        **Settings → Run a refresh**. Nothing runs on a schedule, so a refresh
        happens when you ask for one.
        """
    )


def render_regime(regime: "object") -> None:
    st.subheader("Market Regime", help=tip("Market Regime Index"))
    regime_df = data.market_regime(limit=90)
    if regime_df.empty:
        st.plotly_chart(charts.regime_gauge(None), width="stretch")
        return

    latest = regime_df.iloc[-1]
    st.plotly_chart(
        charts.regime_gauge(latest["regime_score"], latest["regime_label"]),
        width="stretch",
    )
    # Two-by-two rather than one-by-four: this sits in the narrow right-hand
    # column, where four metrics truncate their own values to "2…".
    top = st.columns(2)
    top[0].metric("VIX", format_score(latest["vix_level"], digits=1), help=tip("VIX"))
    top[1].metric(
        "Breadth >200DMA",
        format_pct_already_scaled(latest["breadth_pct_above_200dma"]),
        help=tip("Market breadth"),
    )
    bottom = st.columns(2)
    bottom[0].metric(
        "10Y-2Y spread",
        format_score(latest["yield_curve_spread"], digits=2),
        help=tip("Yield curve spread"),
    )
    bottom[1].metric(
        "Macro tone",
        format_score(latest["macro_news_tone"], digits=2),
        help=tip("Tier 1 / 2 / 3 news"),
    )
    st.caption(
        "Built in-house from VIX percentile, index breadth, macro news tone and the "
        "yield-curve spread (Section 5) — not scraped from a paywalled index."
    )


def render_sector_rotation() -> None:
    """Section 7.1's sector-rotation read -- which sectors money has moved into.

    `technical.compute_sector_rotation` and the `compute_relative_strength` it
    builds on were written, tested and called by nothing. This is the natural
    home for a market-wide view, and it answers a question the top-ranked-names
    table cannot: whether the leaders are concentrated in one corner of the
    market.

    Measured against an equal-weight proxy for the market, because no S&P 500
    price series is ingested anywhere -- the same honest stand-in the beta
    calculation and the backtest benchmark already use.
    """
    panel = data.universe_panel()
    if panel.empty or panel.shape[1] < 2:
        return
    benchmark = risk.equal_weight_market_returns(panel)
    if benchmark.empty:
        return
    # `compute_relative_strength` wants a price *level*, not returns.
    benchmark_level = (1.0 + benchmark).cumprod()

    universe = data.universe()
    sectors = {
        row.symbol: row.sector for row in universe.itertuples() if isinstance(row.sector, str)
    }
    rotation = technical.compute_sector_rotation(
        {column: panel[column].dropna() for column in panel.columns},
        sectors,
        benchmark_level,
    )
    if rotation.empty:
        return

    st.subheader("Sector rotation", help=tip("Sector rotation"))
    display = rotation.rename(
        columns={"sector": "Sector", "relative_strength_change_pct": "vs market (1m)"}
    )
    st.dataframe(
        display.style.format({"vs market (1m)": "{:+.1f}%"}),
        hide_index=True,
        width="stretch",
        height=260,
    )
    st.caption(
        "Change in each sector's strength *relative to the market* over the last month — "
        "top row is where money has been rotating in. A sector can appear here with a "
        "positive number while falling in absolute terms, if it simply fell less than "
        "everything else. This describes what already happened; it is not a forecast."
    )


def main() -> None:
    st.title("QuantPulse")
    st.caption(
        "Self-hosted, $0-cost stock research. The statistics do the thinking; "
        "the optional LLM layer only narrates numbers that already exist."
    )

    render_onboarding()

    if not data.has_any_data():
        render_empty_state()
        st.divider()
        st.caption(DISCLAIMER)
        return

    freshness = data.data_freshness()
    stale_bits = [
        f"{humanize(name)}: {freshness_label(value)}" for name, value in freshness.items()
    ]
    st.caption("Data freshness — " + " · ".join(stale_bits))

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Today's top-ranked names")
        rows = data.screener_rows()
        if rows.empty:
            st.info("No composite scores stored yet — run `scripts/refresh_data.py`.")
        else:
            top = rows.head(10).copy()
            top["Rating"] = top["rating"].map(rating_label)
            st.dataframe(
                top[["symbol", "name", "sector", "Rating", "composite_score", "data_confidence"]]
                .rename(
                    columns={
                        "symbol": "Symbol",
                        "name": "Company",
                        "sector": "Sector",
                        "composite_score": "Score",
                        "data_confidence": "Coverage",
                    }
                )
                .style.format({"Score": "{:.1f}", "Coverage": "{:.0f}%"}),
                hide_index=True,
                width="stretch",
                column_config={
                    "Rating": st.column_config.TextColumn("Rating", help=tip("Rating")),
                    "Score": st.column_config.NumberColumn("Score", help=tip("Composite score")),
                    "Coverage": st.column_config.NumberColumn(
                        "Coverage", help=tip("Data coverage")
                    ),
                },
            )
            st.page_link("pages/1_Screener.py", label="Open the full Screener →")

        st.subheader("What changed since the last refresh")
        changes = data.rating_changes(limit=10)
        if changes.empty:
            st.caption(
                "No rating changes to show — this needs at least two stored scoring "
                "snapshots (the point-in-time schema makes this view free once they exist)."
            )
        else:
            changes = changes.copy()
            changes["From"] = changes["previous_rating"].map(rating_label)
            changes["To"] = changes["rating"].map(rating_label)
            st.dataframe(
                changes[["symbol", "From", "To", "score_change"]].rename(
                    columns={"symbol": "Symbol", "score_change": "Score Δ"}
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Score Δ": st.column_config.NumberColumn(
                        "Score Δ",
                        help="Change in composite score since the previous snapshot.",
                        format="%.1f",
                    )
                },
            )

    with right:
        render_regime(None)

    st.divider()
    render_sector_rotation()

    st.divider()
    st.subheader("Today's market-moving news")
    st.caption("Tier-2 (industry/thematic) and Tier-3 (macro) stories the news module flagged.")
    news = data.market_moving_news(limit=8)
    if news.empty:
        st.caption("No Tier-2/3 stories ingested in the last few days.")
    else:
        for row in news.itertuples():
            tier = f"Tier {row.tier}"
            theme = f" · {humanize(row.matched_theme)}" if row.matched_theme else ""
            event = f" · {humanize(row.event_type)}" if row.event_type else ""
            title = row.title or "(untitled)"
            headline = f"[{title}]({row.source_url})" if row.source_url else title
            sentiment = format_score(row.sentiment_score, digits=2)
            st.markdown(f"**{headline}**")
            st.caption(f"{tier}{theme}{event} · sentiment {sentiment}")

    st.divider()
    st.caption(DISCLAIMER)


main()
