"""Stock Detail — the per-ticker deep dive (Sections 8, 12 page 3).

Section 8 specifies the contents: price chart with indicators and detected
patterns, a radar of the sub-scores, the forecast fan chart, the
analyst-vs-algorithm comparison, the "what's driving this" news feed, and an
optional LLM narrative. Section 10's chatbot ("answer questions like 'why is
NVDA rated Buy?'") lands here too, for the same reason its own example is
phrased per-stock: the question is about one company, and this is the page
where that company's computed numbers are already on screen.

Three deliberate choices:

* **Every forecast is shown next to its own track record.** Section 7.6:
  "Show the model's own historical hit-rate/accuracy stat *alongside every
  individual forecast* in the UI, not just on a separate backtest page — a
  forecast without its own track record next to it invites more confidence
  than it's earned." So `historical_hit_rate` is a column of the forecast
  table, not a footnote elsewhere.
* **The LLM narrative is strictly optional and visibly so.** With no provider
  configured the section simply doesn't render; nothing else on the page
  changes, which is Section 11's "the core engine has zero dependency on it"
  made visible rather than merely claimed. The chat box below it follows the
  same rule.
* **The summary and the chat box share one grounding.** Both are built from
  the `RatingNarrative` this page assembles once (plus the selected model's
  forecasts for the chat), so the paragraph and the answers cannot cite
  different numbers for the same stock on the same screen.
"""

import pandas as pd
import streamlit as st

from lib import charts, data
from lib.format import (
    confidence_label,
    format_percent,
    format_price,
    format_ratio,
    format_score,
    format_signed_percent,
    freshness_label,
    humanize,
    rating_label,
)
from lib.glossary import tip
from lib.search import format_choice
from quantpulse.analysis import forecasting, macro, risk, smart_money, technical
from quantpulse.analysis.investor_profiles import CATEGORIES
from quantpulse.llm import chatbot
from quantpulse.llm import narrative as llm_narrative
from quantpulse.llm.providers import get_provider

st.set_page_config(page_title="QuantPulse — Stock Detail", page_icon="🔬", layout="wide")

SCORE_COLUMNS = {category: f"{category}_score" for category in CATEGORIES}

# Monte Carlo settings. A quarter ahead is long enough for the fan to visibly
# widen without running past the point the calibration means anything; the path
# count is `forecasting`'s own default, kept explicit so the caption can state it.
_MONTE_CARLO_HORIZON = 63
_MONTE_CARLO_PATHS = 10_000


def render_analyst_comparison(symbol: str, row: pd.Series) -> None:
    """Section 7.4's "our algorithm says X, Wall Street says Y" panel."""
    st.subheader("Algorithm vs Wall Street")
    consensus = data.analyst_consensus(symbol)
    left, right = st.columns(2)
    left.metric("QuantPulse rating", rating_label(row["rating"]), help=tip("Rating"))
    left.caption(
        f"Composite {format_score(row['composite_score'])} · "
        f"{confidence_label(row['data_confidence'])}"
    )
    if consensus is None:
        right.metric("Analyst consensus", "—")
        right.caption("No analyst coverage stored for this symbol.")
        return

    counts = {
        "Strong Buy": consensus["strong_buy"],
        "Buy": consensus["buy"],
        "Hold": consensus["hold"],
        "Sell": consensus["sell"],
        "Strong Sell": consensus["strong_sell"],
    }
    total = sum(counts.values())
    leading = max(counts, key=lambda k: counts[k]) if total else "—"
    right.metric("Analyst consensus", leading)
    target = consensus["mean_price_target"]
    right.caption(
        f"{total} analysts · mean target {format_price(target)} · as of {consensus['as_of_date']}"
    )
    st.plotly_chart(charts.sector_bar(counts, title="Analyst rating counts"), width="stretch")


def render_short_interest(symbol: str) -> None:
    """Section 24's short-interest panel — deliberately two readings, not one.

    Section 24 is explicit and emphatic that this signal must not be collapsed
    into a single directional verdict: heavy shorting can mean sophisticated
    money is betting against the company, *or* it can set up a squeeze if
    sentiment turns. `smart_money.py` honours that by keeping short interest out
    of its blended score entirely and returning both readings intact — which
    only means anything if a page shows them, and until now none did.
    """
    reading_row = data.short_interest(symbol)
    if reading_row is None:
        return

    reading = smart_money.read_short_interest(reading_row)
    if reading.pct_float_short is None and reading.days_to_cover is None:
        return

    st.subheader("Short interest", help=tip("Short interest"))
    left, right = st.columns(2)
    left.metric(
        "% of float short",
        format_percent(reading.pct_float_short / 100.0)
        if reading.pct_float_short is not None
        else "—",
        help=tip("Short interest"),
    )
    right.metric(
        "Days to cover",
        format_ratio(reading.days_to_cover) if reading.days_to_cover is not None else "—",
        help=tip("Days to cover"),
    )
    if reading.elevated:
        st.warning(
            "**Elevated short interest — and that cuts both ways.** It can mean "
            "informed investors are betting against this company. It can equally "
            "set up a **short squeeze**: a crowded short position that has to buy "
            "back quickly if the story improves, which pushes the price *up*. "
            "QuantPulse does not score this as bullish or bearish, because the same "
            "number genuinely supports both readings."
        )
    else:
        st.caption(
            "Short interest is not elevated. Shown as context only — it is "
            "deliberately excluded from the Smart Money score, since the same "
            "figure can be read as bearish conviction or as squeeze potential."
        )


def render_monte_carlo(symbol: str, bars: pd.DataFrame) -> None:
    """Section 7.6's Monte Carlo fan chart — simulated paths, not a point forecast.

    Deliberately separate from the stored-forecast fan above, and labelled as a
    different *kind* of answer. The models above each commit to one number per
    horizon; this simulates thousands of random-walk paths calibrated to this
    stock's own history and reports the range they spread into. It is the same
    random-walk-with-drift model `baseline_forecast` uses, executed by
    simulation instead of closed form, which is exactly why it is NOT entered as
    a fourth competing model in the table above — it would be grading the
    baseline against itself under simulation noise.
    """
    if bars.empty or len(bars) < 60:
        return

    prices = bars.rename(columns=str.lower)
    fan = forecasting.monte_carlo_fan_chart(prices, _MONTE_CARLO_HORIZON)
    if fan is None:
        return

    st.subheader("Simulated price paths", help=tip("Monte Carlo fan chart"))
    st.caption(
        f"{_MONTE_CARLO_PATHS:,} random-walk paths over the next "
        f"{_MONTE_CARLO_HORIZON} trading days, calibrated to this stock's own "
        "historical drift and volatility. The shaded band is the middle 90% of "
        "simulated outcomes; it widens with time because uncertainty compounds — "
        "that widening is the message. This is a range of possibilities, not a "
        "prediction, and it assumes the future resembles the past."
    )
    st.plotly_chart(charts.monte_carlo_fan_chart(bars, fan), width="stretch")
    st.caption(
        f"Calibrated on {fan.n_train:,} daily returns "
        f"(drift {fan.mu * 100:.3f}%/day, volatility {fan.sigma * 100:.2f}%/day)."
    )


def render_risk_profile(symbol: str, bars: pd.DataFrame) -> None:
    """Section 7.7's per-name risk block -- volatility, beta, Sortino, VaR.

    `risk.stock_risk_profile` computes all of it in one call and had no caller:
    the Portfolio page shows portfolio-level risk, so a single stock's risk
    numbers existed and were displayed nowhere. Each estimator declines
    independently when its own data floor isn't met, so a short-history name
    yields a partly-filled block rather than fabricated numbers.
    """
    if bars.empty:
        return
    closes = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["date"])))["close"]
    returns = risk.to_returns(closes)
    if returns.empty:
        return

    market = risk.equal_weight_market_returns(data.universe_panel())
    stored_iv = data.options_signals(symbol)
    profile = risk.stock_risk_profile(
        returns,
        market_returns=market if not market.empty else None,
        implied_volatility=None if stored_iv is None else stored_iv.get("atm_implied_volatility"),
    )

    st.subheader("Risk profile", help=tip("Volatility"))
    columns = st.columns(5)
    columns[0].metric(
        "Volatility (ann.)", format_percent(profile.volatility.historical), help=tip("Volatility")
    )
    columns[1].metric(
        "Implied vol",
        format_percent(profile.volatility.implied),
        help=tip("Implied volatility"),
    )
    columns[2].metric(
        "Beta",
        format_ratio(profile.beta.beta) if profile.beta else "—",
        help=tip("Beta"),
    )
    columns[3].metric("Sortino", format_ratio(profile.sortino), help=tip("Sortino ratio"))
    columns[4].metric(
        "Daily VaR 95%",
        format_percent(profile.value_at_risk.var) if profile.value_at_risk else "—",
        help=tip("Value at Risk"),
    )
    notes = [f"Measured on {profile.n_observations} daily returns."]
    # An absent Sortino needs the same plain explanation VaR already gets, or a
    # dash reads as a glitch rather than as "the sample cannot support this".
    ratio_floor = risk.min_ratio_observations(risk.TRADING_DAYS_PER_YEAR)
    if profile.sortino is None and profile.n_observations < ratio_floor:
        notes.append(
            f"Sortino needs about a year of history ({ratio_floor} daily returns) before it "
            "means anything — a ratio of average return to downside risk is far noisier than "
            "either number on its own, so it is left blank rather than shown as noise."
        )
    if profile.beta is not None and profile.beta.r_squared is not None:
        notes.append(
            f"Beta is against an equal-weight proxy for the market (no S&P 500 price series "
            f"is ingested), R² = {profile.beta.r_squared:.2f}."
        )
    if profile.volatility.implied_premium is not None:
        direction = "more" if profile.volatility.implied_premium > 0 else "less"
        notes.append(
            f"Options are pricing **{direction}** movement than this stock has recently "
            f"delivered ({format_signed_percent(profile.volatility.implied_premium)} spread) — "
            "a spread between two volatilities, not a direction."
        )
    st.caption(" ".join(notes))


def render_macro_overlay(sector: str | None) -> None:
    """Section 28's targeted commodity/currency overlay for this stock's sector.

    Oil for Energy, gold for Materials, the dollar for the sectors dominated by
    multinationals earning abroad -- and a flat zero for every other sector, on
    purpose ("a small biotech doesn't care about oil prices"). The series were
    ingested nightly from the start and `commodity_overlay_adjustment` was
    called by nothing, so the overlay existed only as a function.
    """
    sensitivities = macro.SECTOR_COMMODITY_SENSITIVITY.get(sector or "")
    if not sensitivities:
        return

    moves = {
        series: change
        for series in sensitivities
        if (change := macro.pct_change(data.macro_series(series))) is not None
    }
    if not moves:
        return
    adjustment = macro.commodity_overlay_adjustment(sector, moves)

    st.subheader("Macro overlay", help=tip("Sector macro overlay"))
    tone = "tailwind" if adjustment > 0 else "headwind" if adjustment < 0 else "neutral"
    st.markdown(
        f"**{sector}** is exposed to "
        + ", ".join(f"{humanize(series)} ({change:+.1f}%)" for series, change in moves.items())
        + f" over the last ~3 months — a **{tone}** of {adjustment:+.2f} on a -1 to +1 scale."
    )
    st.caption(
        "Applied only to the sectors these series genuinely move (Section 28): oil for "
        "Energy, metals for Materials, the dollar for sectors dominated by multinationals "
        "earning abroad. Every other sector gets exactly zero rather than a small "
        "meaningless nudge. This is context, not part of the composite score."
    )


def _none_if_nan(value: object) -> float | None:
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def rating_narrative(symbol: str, row: pd.Series) -> llm_narrative.RatingNarrative:
    """The rating context both the summary paragraph and the chat box are grounded in.

    Built once and shared deliberately: if the chat could see a different set of
    numbers than the paragraph above it, the two could contradict each other on
    the same screen.
    """
    return llm_narrative.RatingNarrative(
        symbol=symbol,
        rating=row["rating"],
        composite_score=float(row["composite_score"]),
        sub_scores={
            category: _none_if_nan(row[SCORE_COLUMNS[category]]) for category in CATEGORIES
        },
        percentile_rank=_none_if_nan(row["percentile_rank"]),
        data_confidence=_none_if_nan(row["data_confidence"]),
        as_of=row["date"],
    )


def forecast_narrative(
    symbol: str, forecast_rows: pd.DataFrame, model_name: str, last_close: float | None
) -> llm_narrative.ForecastNarrative:
    """The forecast context for `model_name` — the same rows the table above renders."""
    selected = forecast_rows[forecast_rows["model_name"] == model_name]
    return llm_narrative.ForecastNarrative(
        symbol=symbol,
        model_name=model_name,
        last_close=last_close,
        horizons=tuple(
            llm_narrative.ForecastHorizon(
                horizon_days=int(item.horizon_days),
                point_return=float(item.point_return),
                point_price=_none_if_nan(item.point_price),
                lower_price=_none_if_nan(item.lower_price),
                upper_price=_none_if_nan(item.upper_price),
                historical_hit_rate=_none_if_nan(item.historical_hit_rate),
            )
            for item in selected.sort_values("horizon_days").itertuples()
        ),
    )


def render_narrative(context: llm_narrative.RatingNarrative) -> None:
    """Section 11's optional "why this rating" paragraph — absent when no LLM is set up."""
    if get_provider() is None:
        return
    st.subheader("Plain-English summary")
    st.caption(
        "Generated by the optional LLM layer from the numbers above — it narrates "
        "them, it does not compute or add any."
    )
    with st.spinner("Narrating…"):
        text = llm_narrative.explain_rating(context)
    if text:
        st.info(text)
    else:
        st.caption("The LLM layer is configured but did not return a summary this time.")


def render_sentiment_narrative(symbol: str, row: pd.Series, news: pd.DataFrame) -> None:
    """Section 11 use 2 -- plain English on what moved this stock's sentiment score.

    `narrative.explain_sentiment_move` was built and called by nothing. It sits
    directly under the headline feed it summarizes, so a reader can check the
    summary against the same articles rather than taking it on trust, and it
    reports the polarities FinBERT already assigned -- Section 11 forbids the
    model re-scoring them, and the prompt says so as well as the grounding
    instruction enforcing it.
    """
    if get_provider() is None or news.empty:
        return
    history = data.sentiment_history(symbol)
    current = _none_if_nan(row.get("sentiment_raw"))
    if current is None and history:
        current = history[0][1]
    if current is None:
        return

    context = llm_narrative.SentimentNarrative(
        symbol=symbol,
        current_score=float(current),
        previous_score=history[1][1] if len(history) > 1 else None,
        mention_volume=history[0][2] if history else int(len(news)),
        as_of=history[0][0] if history else None,
        drivers=tuple(
            llm_narrative.SentimentDriver(
                title=str(item.title or ""),
                event_type=item.event_type,
                sentiment_score=_none_if_nan(item.sentiment_score),
                published_at=item.published_at,
                source=item.source,
            )
            for item in news.itertuples()
            if item.title
        ),
    )
    with st.spinner("Summarizing the coverage…"):
        text = llm_narrative.explain_sentiment_move(context)
    if text:
        st.info(text)
        st.caption(
            "Summarizes the headlines above and the polarity already assigned to each — "
            "the model never scores them itself."
        )


def render_filing_summary(symbol: str) -> None:
    """Section 11 use 4 -- a plain-English read of the latest 10-K/10-Q.

    `narrative.summarize_filing_excerpt` existed with nothing to summarize: no
    part of the pipeline stored filing prose. The document is fetched on demand
    rather than nightly, because a 10-K is several megabytes and a reader opens
    one company at a time.

    The excerpt is the only free-text payload anywhere in the LLM layer, so it
    is the only place source material could contain instruction-like text; the
    context builder frames it as quoted material and the prompt tells the model
    to ignore anything inside it that reads like an instruction.
    """
    if get_provider() is None:
        return
    st.subheader("Latest SEC filing", help=tip("10-K / 10-Q"))
    if not st.button("Summarize the latest 10-K/10-Q", key=f"filing_{symbol}"):
        st.caption(
            "Fetches the most recent annual or quarterly report from SEC EDGAR and "
            "summarizes its Management's Discussion section. Not fetched until asked — "
            "these documents are several megabytes each."
        )
        return

    with st.spinner("Fetching the filing…"):
        excerpt = data.filing_excerpt(symbol)
    if excerpt is None:
        st.caption("No 10-K or 10-Q found for this symbol in the last year.")
        return

    with st.spinner("Summarizing…"):
        text = llm_narrative.summarize_filing_excerpt(llm_narrative.FilingExcerpt(**excerpt))
    if not text:
        st.caption("The LLM layer is configured but did not return a summary this time.")
        return
    st.info(text)
    section = excerpt["section"] or "the opening pages"
    st.caption(
        f"From {symbol}'s **{excerpt['form_type']}** filed {excerpt['filed_date']}, "
        f"{section}. A summary of an excerpt, not of the whole filing — "
        f"[read the original]({excerpt['source_url']})."
    )


def render_chat(symbol: str, context_blocks: list[str]) -> None:
    """Section 10's grounded chat box — absent entirely when no LLM is configured.

    Two things worth knowing about the wiring:

    * **The transcript is keyed per symbol.** `context_blocks` describe exactly
      one company, so carrying an AAPL conversation into NVDA's context is how
      you get a model confidently answering about the wrong stock. Switching the
      symbol picker starts a fresh thread rather than silently re-grounding an
      existing one.
    * **What the box can see is stated, not implied.** `chatbot.answer` is
      instructed to say the app doesn't show something rather than guess, so a
      question about, say, this stock's dividend history will honestly come back
      unanswered — the caption sets that expectation up front instead of leaving
      the user to discover the edges by hitting them.
    """
    if get_provider() is None:
        return
    st.subheader("Ask about this stock", help=tip("Composite score"))
    st.caption(
        f"Grounded strictly in {symbol}'s rating breakdown and stored forecasts above — "
        "it reports those numbers, it never computes new ones, and it will say so rather "
        "than guess when you ask for something the app doesn't hold."
    )

    history_key = f"chat_history_{symbol}"
    conversation: chatbot.Conversation = st.session_state.get(history_key, chatbot.Conversation())

    for turn in conversation.turns:
        with st.chat_message(turn.role):
            st.markdown(turn.content)

    question = st.chat_input(
        f"Ask about {symbol}…",
        key=f"chat_input_{symbol}",
        max_chars=chatbot.MAX_QUESTION_CHARS,
    )
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # `conversation` here is the history *before* this question --
                # `answer` takes the new question separately from the transcript.
                reply = chatbot.answer(question, context_blocks, history=conversation)
            answered = reply or (
                "The LLM layer is configured but did not return an answer this time."
            )
            st.markdown(answered)
        st.session_state[history_key] = conversation.with_turn("user", question).with_turn(
            "assistant", answered
        )

    st.caption(chatbot.ADVICE_DISCLAIMER)


def main() -> None:
    st.title("🔬 Stock Detail")
    rows = data.screener_rows()
    if rows.empty:
        st.info("No scored symbols yet — run `scripts/refresh_data.py` first.")
        return

    symbols = rows["symbol"].tolist()
    preselected = st.session_state.get("detail_symbol")
    index = symbols.index(preselected) if preselected in symbols else 0
    # Labels carry the company name so a picker confirms you chose the company
    # you meant, not a similarly-spelled ticker (Section 31).
    labels = rows[["symbol", "name"]]
    symbol = st.selectbox(
        "Symbol",
        symbols,
        index=index,
        format_func=lambda s: format_choice(labels, s),
    )
    st.session_state["detail_symbol"] = symbol

    row = rows[rows["symbol"] == symbol].iloc[0]
    header = st.columns([2, 1, 1, 1])
    header[0].markdown(f"### {symbol} — {row['name'] or ''}")
    header[0].caption(row["sector"] or "Sector unknown")
    header[1].metric("Rating", rating_label(row["rating"]), help=tip("Rating"))
    header[2].metric("Composite", format_score(row["composite_score"]), help=tip("Composite score"))
    header[3].metric(
        "Percentile", format_score(row["percentile_rank"], digits=0), help=tip("Percentile rank")
    )
    st.caption(
        f"{confidence_label(row['data_confidence'])} · scored {freshness_label(row['date'])}"
    )

    bars = data.ohlcv(symbol, lookback_days=400)
    overlays: dict[str, pd.Series] = {}
    if not bars.empty and len(bars) >= 50:
        overlays["SMA 50"] = bars["close"].rolling(50).mean()
    if not bars.empty and len(bars) >= 200:
        overlays["SMA 200"] = bars["close"].rolling(200).mean()
    # Section 8's "price chart with indicators": the support/resistance detector
    # was built and drawn nowhere, so the chart carried two moving averages and
    # nothing else. Only levels the market has actually turned at twice or more
    # are returned, so this stays sparse rather than striping the chart.
    levels = (
        technical.find_support_resistance_levels(bars)
        if not bars.empty and len(bars) >= 30
        else None
    )
    st.plotly_chart(charts.price_chart(bars, overlays=overlays, levels=levels), width="stretch")
    if levels is not None and not levels.empty:
        st.caption(
            f"Dotted lines are **support/resistance** — {len(levels)} price level(s) this "
            "stock has repeatedly turned at, annotated with how many times. A level tested "
            "once is a data point, not a level, so those are not drawn."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Sub-scores", help=tip("Composite score"))
        sub_scores = {
            category: (
                None
                if pd.isna(row[SCORE_COLUMNS[category]])
                else float(row[SCORE_COLUMNS[category]])
            )
            for category in CATEGORIES
        }
        st.plotly_chart(charts.subscore_radar(sub_scores, name=symbol), width="stretch")
        st.caption(
            "Categories with no data are omitted rather than plotted at zero — "
            "a missing score is not a bad score."
        )
    with right:
        st.subheader(
            "Detected patterns",
            help=tip(
                "Support and resistance",
                "Chart and candlestick patterns detected geometrically, each with a "
                "confidence score rather than a yes/no verdict.",
            ),
        )
        found = data.patterns(symbol)
        if found.empty:
            st.caption("No chart or candlestick patterns detected in the last 120 days.")
        else:
            display = found.copy()
            display["Pattern"] = display["pattern_type"].map(humanize)
            display["Direction"] = display["direction"].map(humanize)
            st.dataframe(
                display[["date", "Pattern", "Direction", "confidence"]]
                .rename(columns={"date": "Date", "confidence": "Confidence"})
                .style.format({"Confidence": "{:.2f}"}),
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.subheader("Forecast", help=tip("Monte Carlo simulation"))
    forecast_rows = data.forecasts(symbol)
    forecast_context: llm_narrative.ForecastNarrative | None = None
    if forecast_rows.empty:
        st.caption("No forecasts generated for this symbol yet.")
    else:
        models = sorted(forecast_rows["model_name"].unique())
        model = st.selectbox("Model", models, index=0)
        st.caption(
            "**Hit rate** is this model's own out-of-sample directional accuracy; "
            "**vs naive** is the same measure for a naive random-walk forecast over "
            "the same periods. Read them together — a model at or below the naive "
            "column has not demonstrated any skill. Note `arima` and `baseline` are "
            "near-duplicates by construction (a drifting ARIMA converges to the "
            "random-walk-with-drift null), so the two agreeing is not corroboration."
        )
        # Built from the model the user actually selected, so the chat box below
        # is grounded in the same forecasts the table and fan chart are showing.
        last_close = None if bars.empty else float(bars["close"].iloc[-1])
        forecast_context = forecast_narrative(symbol, forecast_rows, model, last_close)
        st.plotly_chart(
            charts.forecast_fan_chart(bars, forecast_rows, model_name=model), width="stretch"
        )
        table = forecast_rows[forecast_rows["model_name"] == model].copy()
        table["Return"] = table["point_return"].map(format_signed_percent)
        # Always paired with the naive null's rate. A hit rate alone is not a
        # skill measure -- on real history the baseline's rate was exactly the
        # fraction of periods that happened to be up, so a bare "53%" reads as
        # modest skill when it is a coin flip, and can even hide a model doing
        # worse than the null it exists to beat (Section 7.6).
        table["Hit rate"] = table["historical_hit_rate"].map(
            lambda v: "—" if pd.isna(v) else format_percent(v, digits=0)
        )
        table["vs naive"] = table["baseline_hit_rate"].map(
            lambda v: "—" if pd.isna(v) else format_percent(v, digits=0)
        )
        # The sample size behind those two percentages. A rate over 40 distinct
        # out-of-sample windows and one over 3 are different claims, and without
        # this column they looked identical.
        table["Windows"] = table.get(
            "hit_rate_windows", pd.Series(index=table.index, dtype="float")
        ).map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        st.dataframe(
            table[
                [
                    "horizon_days",
                    "Return",
                    "point_price",
                    "lower_price",
                    "upper_price",
                    "Hit rate",
                    "vs naive",
                    "Windows",
                ]
            ]
            .rename(
                columns={
                    "horizon_days": "Horizon (days)",
                    "point_price": "Target",
                    "lower_price": "Low",
                    "upper_price": "High",
                }
            )
            # These three are dollar prices and must be printed like every other
            # price in either front end -- `format_price` here, `formatPrice` in
            # React. Without this they render as raw float64 (282.7719 beside
            # 287.08, since Streamlit drops trailing zeros), which is four
            # decimals of precision a forecast does not have, ragged column
            # alignment, no currency marker, and a visible disagreement with the
            # React page showing "$282.77" for the very same stored row.
            .style.format(
                {"Target": "${:,.2f}", "Low": "${:,.2f}", "High": "${:,.2f}"}, na_rep="—"
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Hit rate": st.column_config.TextColumn("Hit rate", help=tip("Hit rate")),
                "Low": st.column_config.NumberColumn(
                    "Low", help="Lower bound of the forecast range, not a floor."
                ),
                "High": st.column_config.NumberColumn(
                    "High", help="Upper bound of the forecast range, not a target."
                ),
                "Windows": st.column_config.TextColumn(
                    "Windows",
                    help="How many distinct out-of-sample periods the two hit rates were "
                    "measured over. More is better; a rate from a handful of windows is "
                    "an anecdote, not a track record.",
                ),
            },
        )
        st.caption(
            "**Hit rate** is this model's own out-of-sample directional accuracy at that "
            "horizon (Section 7.6) — shown next to the forecast, not hidden on another page. "
            "**Windows** is how many separate historical periods it was measured over. A "
            "dash means the model has not been graded over enough distinct windows for a "
            "rate to mean anything, so none is shown rather than a flattering one: at the "
            "one-year horizon in particular there simply are not many independent years to "
            "test against."
        )

    render_monte_carlo(symbol, bars)

    st.divider()
    render_risk_profile(symbol, bars)

    st.divider()
    render_analyst_comparison(symbol, row)
    render_short_interest(symbol)
    render_macro_overlay(row["sector"])

    st.divider()
    st.subheader("What's driving this", help=tip("Sentiment score"))
    news = data.symbol_news(symbol, limit=10)
    if news.empty:
        st.caption("No Tier-1 articles matched this symbol in the last three weeks.")
    else:
        for item in news.itertuples():
            title = item.title or "(untitled)"
            headline = f"[{title}]({item.source_url})" if item.source_url else title
            st.markdown(f"**{headline}**")
            event = humanize(item.event_type) if item.event_type else "unclassified"
            st.caption(
                f"{event} · sentiment {format_score(item.sentiment_score, digits=2)} · "
                f"{item.source or 'unknown source'}"
            )
        render_sentiment_narrative(symbol, row, news)

    st.divider()
    render_filing_summary(symbol)

    st.divider()
    rating_context = rating_narrative(symbol, row)
    render_narrative(rating_context)

    st.divider()
    blocks = [llm_narrative.build_rating_context(rating_context)]
    if forecast_context is not None:
        blocks.append(llm_narrative.build_forecast_context(forecast_context))
    render_chat(symbol, blocks)


main()
