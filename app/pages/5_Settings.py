"""Settings / About — freshness, methodology, disclaimers, provider config (Section 12).

Section 12's second "easy to skip and worth not skipping" rule is the reason
this page leads with data freshness rather than burying it: if a refresh was
skipped or a source's quota ran out, the app must say so visibly instead of
presenting every number with the same implied confidence. The freshness table
distinguishes "never run" from "stale," because an empty pipeline and a
day-old one are very different situations that a blank cell would conflate.

Secrets are never displayed — only whether each one is set (Section 18).
"""

import streamlit as st

from lib import data
from lib.format import freshness_label, humanize
from quantpulse.config import get_settings
from quantpulse.llm.providers import get_provider

st.set_page_config(page_title="QuantPulse — Settings", page_icon="⚙️", layout="wide")


def render_freshness() -> None:
    st.subheader("Data freshness")
    freshness = data.data_freshness()
    rows = [
        {
            "Dataset": humanize(name),
            "Latest": "—" if value is None else value.isoformat(),
            "Age": freshness_label(value),
        }
        for name, value in freshness.items()
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if all(value is None for value in freshness.values()):
        st.info(
            "Nothing has been ingested yet. Run `scripts/seed_initial_data.py` once, "
            "then `scripts/refresh_data.py` nightly."
        )


def render_pipeline_health() -> None:
    st.subheader("Pipeline health")
    log = data.refresh_log(limit=15)
    if log.empty:
        st.caption("No refresh runs logged yet.")
        return
    st.dataframe(
        log.rename(
            columns={
                "job_name": "Job",
                "run_timestamp": "Run at",
                "status": "Status",
                "rows_updated": "Rows",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "A `partial` status means one source failed and was skipped rather than "
        "aborting the whole run (Section 6.5's graceful degradation)."
    )


def render_configuration() -> None:
    st.subheader("Configuration")
    settings = get_settings()
    left, right = st.columns(2)

    with left:
        st.markdown("**Portfolio backend** (ADR 4.5)")
        st.code(settings.portfolio_backend)
        st.caption(
            "`sqlite` persists holdings locally; `session` keeps them in the browser "
            "session only, which is what the public demo uses so visitors never see or "
            "overwrite each other's portfolios."
        )

    with right:
        st.markdown("**LLM narration** (Sections 4.3, 11)")
        provider = get_provider()
        if provider is None:
            st.code("disabled / not configured")
            st.caption(
                "Every number in the app is still computed and displayed — the LLM only "
                "ever narrates results that already exist, so turning it off removes "
                "paragraphs of prose and nothing else."
            )
        else:
            st.code(f"{provider.name} · {settings.llm_provider}")
            st.caption("Used only to narrate already-computed numbers, never to produce them.")

    st.markdown("**Credentials configured** (values are never displayed — Section 18)")
    secrets = {
        "FINNHUB_API_KEY": settings.finnhub_api_key,
        "FRED_API_KEY": settings.fred_api_key,
        "REDDIT_CLIENT_ID": settings.reddit_client_id,
        "GEMINI_API_KEY": settings.gemini_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
    }
    st.dataframe(
        [{"Secret": name, "Set": "✅" if value else "—"} for name, value in secrets.items()],
        hide_index=True,
        width="stretch",
    )


def render_methodology() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        **The math does the thinking; the LLM does the talking.** Every score, forecast
        and risk metric is computed from transparent statistics you can read in
        `src/quantpulse/`. The optional LLM layer only narrates numbers that already
        exist — it never produces them.

        - **Composite score** — seven categories (fundamental, technical, analyst,
          sentiment, momentum/risk-adjusted, industry & macro news, smart money),
          each percentile-normalized within the universe, then weighted and
          renormalized over whichever categories actually had data.
        - **Ratings are relative, not absolute.** The top 10% of the scored universe is
          Strong Buy however the market as a whole looks. A high rating means "ranks
          well against peers right now," not "cheap" or "safe."
        - **Data-completeness score** — shown next to every stock, because a
          thinly-covered small-cap should not look as trustworthy as a mega-cap.
        - **Backtests are survivorship- and cost-aware**, run at a realistic rebalance
          cadence, and reported with bootstrap confidence intervals.
        - **Point-in-time storage** — scores are append-only and never rewritten, so
          "what did the algorithm say on June 3rd" always returns what it actually said.

        **Known limitations, stated rather than hidden:** beta is measured against an
        equal-weight universe proxy (no S&P 500 price series is ingested); news models
        are English-language and Western-media weighted; free data sources are
        best-effort and can be stale or incomplete.
        """
    )


def main() -> None:
    st.title("⚙️ Settings & About")
    render_freshness()
    st.divider()
    render_pipeline_health()
    st.divider()
    render_configuration()
    st.divider()
    render_methodology()
    st.divider()
    st.error(
        "**Educational/research tool. Not financial advice. Not a registered investment "
        "advisor. Past backtested performance does not guarantee future results.** "
        "QuantPulse never connects to a brokerage, never executes trades, and never asks "
        "for credentials."
    )


main()
