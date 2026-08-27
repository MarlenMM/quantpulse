"""Settings / About — freshness, refresh control, methodology, disclaimers (Section 12).

Section 12's second "easy to skip and worth not skipping" rule is the reason
this page leads with data freshness rather than burying it: if a refresh was
skipped or a source's quota ran out, the app must say so visibly instead of
presenting every number with the same implied confidence. The freshness table
distinguishes "never run" from "stale," because an empty pipeline and a
day-old one are very different situations that a blank cell would conflate.

That rule is also why the refresh *control* sits directly under the freshness
table. Nothing runs on a schedule — this button is the trigger — so "sentiment
is nine days old" and "here is how to fix that" belong on one screen, in that
order, rather than one being a table and the other being a command in the
README.

Secrets are never displayed — only whether each one is set (Section 18).
"""

from datetime import datetime

import streamlit as st

from lib import data, refresh
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
            "Nothing has been ingested yet. Run `scripts/seed_initial_data.py` once for the "
            "historical backfill, then use **Run a refresh** below to keep it current."
        )


def _elapsed(since: datetime, until: datetime | None = None) -> str:
    total = int(((until or datetime.now()) - since).total_seconds())
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds:02d}s"


@st.fragment(run_every=2)
def render_running_refresh() -> None:
    """Poll a refresh that is already running, and tail its log.

    A fragment rather than a `sleep`/`st.rerun` loop over the whole page: only
    this block re-renders every couple of seconds, so the freshness table and
    the methodology text below it aren't rebuilt sixty times a minute. The
    polling stops on its own — once the run ends this reruns the whole app,
    which renders the idle branch instead and never calls back in here.
    """
    runner = refresh.get_runner()
    state = runner.state()

    if not state.running:
        # New rows just landed. `lib.data`'s readers cache for TTL_SECONDS, so
        # without this every page would keep serving pre-refresh numbers for
        # another five minutes — the exact "stale but presented confidently"
        # failure the freshness table above exists to prevent.
        st.cache_data.clear()
        st.rerun(scope="app")
        return

    assert state.started_at is not None
    st.info(
        f"Refresh running — started {state.started_at:%H:%M:%S}, "
        f"{_elapsed(state.started_at)} elapsed. Leaving this page does not stop it."
    )
    st.code("\n".join(state.log[-20:]) or "Starting…", language="text")
    if st.button("Stop this refresh", key="stop_refresh"):
        runner.stop()
        st.rerun(scope="app")


def render_last_run(state: refresh.RefreshState) -> None:
    """The outcome of the most recent run started from this page, if any."""
    if state.never_run or state.finished_at is None or state.started_at is None:
        return
    took = _elapsed(state.started_at, state.finished_at)
    finished = f"{state.finished_at:%Y-%m-%d %H:%M:%S}"
    if state.succeeded:
        # Deliberately not the word "success": the script exits 0 for a `partial`
        # run (one flaky source skipped) and for `skipped_non_trading_day` too,
        # and calling either a success would undo the point of the status column
        # in the table below. The exit code says "did not fail"; the
        # pipeline-health row says what actually happened.
        st.success(f"Refresh finished at {finished} after {took}. See its status below.")
    else:
        st.error(
            f"Refresh failed at {finished} after {took} "
            f"(exit code {state.returncode}). The log below has the traceback."
        )
    with st.expander("Refresh log", expanded=not state.succeeded):
        st.code("\n".join(state.log) or "(no output)", language="text")


def render_manual_refresh() -> None:
    st.subheader("Run a refresh")
    if not get_settings().manual_refresh_allowed():
        st.caption(
            "Refreshing from the UI is turned off here (`MANUAL_REFRESH_ENABLED`), which is "
            "the default for a hosted demo. A refresh is a long job against rate-limited "
            "free-tier APIs and is not something an anonymous visitor should be able to "
            "start — and this host installs `requirements.txt`, which deliberately omits "
            "the model stack the refresh needs, so it could not run here anyway. The "
            "shared demo database is refreshed by dispatching "
            "`.github/workflows/refresh_data.yml`."
        )
        return

    runner = refresh.get_runner()
    state = runner.state()
    if state.running:
        render_running_refresh()
        return

    st.caption(
        "Nothing runs on a schedule — the data above changes only when you refresh it here. "
        "Best done after the US close, when the day's closing prices and option chain have "
        "actually been published. It runs in the background, so you can keep using the app "
        "while it works."
    )
    weekly = st.checkbox(
        "Include the weekly steps",
        help=(
            "Fundamentals, analyst consensus, 13F holdings, forecasts, the backtest, news "
            "and sentiment. These normally run only on Mondays — and with no schedule left "
            "to come round, a database only ever refreshed on other days would never see "
            "them again. Much slower: hours rather than minutes."
        ),
    )
    ignore_calendar = st.checkbox(
        "Run even though the market is closed today",
        help=(
            "A refresh is normally a deliberate no-op on a weekend or holiday, because "
            "there is no new session to pull. Tick this when catching a database up, where "
            "\u201cit is Saturday\u201d is not a reason to do nothing."
        ),
    )
    if st.button("🔄 Refresh now", type="primary"):
        if runner.start(weekly=weekly, ignore_market_calendar=ignore_calendar):
            st.rerun()
        else:
            # The runner is process-wide, so this is another tab or another
            # browser, not a double-click: say so rather than doing nothing.
            st.warning("A refresh is already running — it was started from another session.")
    render_last_run(state)


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
    render_manual_refresh()
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
