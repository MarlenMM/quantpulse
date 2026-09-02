"""Plotly figure builders shared across the pages (Section 12).

Section 12 picks Plotly because it is free, interactive, native in Streamlit,
and would survive a React migration via `react-plotly.js`. Each builder here
takes already-computed data and returns a `go.Figure` -- no Streamlit import,
no database access, no computation beyond arranging numbers into traces. That
keeps them unit-testable (assert on the traces) and keeps the "analysis engine
is UI-agnostic" boundary intact in the other direction too: charts don't
compute, they draw.

Colors follow `format.RATING_DISPLAY` so a rating's color means the same thing
in a table cell and in a chart -- and, per Section 12, color is never the only
channel: every chart that encodes a rating also labels it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from lib.format import RATING_DISPLAY, humanize
from quantpulse.utils.market_calendar import trading_days_between

__all__ = [
    "empty_figure",
    "price_chart",
    "subscore_radar",
    "horizon_dates",
    "forecast_fan_chart",
    "monte_carlo_fan_chart",
    "allocation_pie",
    "equity_curve_chart",
    "regime_gauge",
    "correlation_heatmap",
    "sector_bar",
    "interval_whisker",
]

# One accent, used wherever a series has no semantic color of its own — the same
# ink blue the two front ends use for every interactive affordance.
#
# It is a mid-tone rather than either theme's own accent because a Plotly trace
# takes a literal: the figure has to stay legible on the warm paper of the light
# theme and the warm slate of the dark one, and `.streamlit/config.toml` cannot
# vary a series colour per scheme. Everything Streamlit *can* theme per scheme —
# axis text, gridlines, the paper — it does, because the figures below set
# transparent backgrounds and let it through.
_ACCENT = "#3a72b8"
_MUTED = "#7a756b"
_BAND = "rgba(58, 114, 184, 0.18)"


def _base_layout(fig: go.Figure, *, height: int = 340, title: str | None = None) -> go.Figure:
    """Shared layout: transparent background so the figure inherits the app theme.

    Letting Streamlit's theme (light or dark, Section 31) show through means one
    set of figures works in both, rather than hardcoding a background that looks
    wrong in one of them.
    """
    # Only set `title` when there is one: passing None makes Plotly render the
    # literal string "undefined" above the figure.
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if title:
        fig.update_layout(title=title)
    return fig


def empty_figure(message: str = "No data yet") -> go.Figure:
    """A placeholder figure carrying an explanation.

    Used wherever a chart has nothing to draw. An explicit "why it's empty"
    beats a blank rectangle, which reads as a broken page rather than an
    un-run pipeline -- the state a freshly-cloned repo is actually in.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=_MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=240)


def price_chart(
    ohlcv: pd.DataFrame,
    *,
    overlays: dict[str, pd.Series] | None = None,
    levels: pd.DataFrame | None = None,
    title: str | None = None,
) -> go.Figure:
    """Candlestick price chart with optional moving-average overlays and price levels.

    `levels` is `technical.find_support_resistance_levels`' output (`level`,
    `touches`) -- prices the market has repeatedly turned at. Section 8 asks for
    "price chart with indicators and detected patterns"; the chart drew two
    moving averages and nothing else, while the support/resistance detector sat
    unused. Drawn as dashed horizontal lines annotated with the touch count,
    because a level tested five times and one tested twice are not the same
    claim and the number is the only thing that distinguishes them.
    """
    if ohlcv.empty:
        return empty_figure("No price history stored for this symbol yet")

    fig = go.Figure(
        go.Candlestick(
            x=ohlcv["date"],
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="Price",
            # The same green and red a Buy and a Sell are, so a rising candle
            # and a Buy rating are one colour vocabulary rather than two.
            increasing_line_color=RATING_DISPLAY["buy"][2],
            decreasing_line_color=RATING_DISPLAY["sell"][2],
        )
    )
    # Overlays are all drawn in the accent and separated by dash pattern, not by
    # hue. Two reasons, and the first is the important one: green and red mean
    # *direction* on this figure, and a moving average given the next colour in
    # a categorical palette came out green -- a line that means nothing of the
    # kind, in the colour that means "up", crossing candles that use it
    # literally. The second is the rule the ratings already follow: a series
    # that is distinguishable without colour perception stays distinguishable.
    dashes = ("solid", "dash", "dot", "dashdot")
    for index, (label, series) in enumerate((overlays or {}).items()):
        fig.add_trace(
            go.Scatter(
                x=ohlcv["date"],
                y=series,
                name=label,
                mode="lines",
                line=dict(width=1.4, color=_ACCENT, dash=dashes[index % len(dashes)]),
                opacity=1.0 if index == 0 else 0.75,
            )
        )
    if levels is not None and not levels.empty:
        for row in levels.itertuples(index=False):
            fig.add_hline(
                y=float(row.level),
                line=dict(color=_MUTED, width=1, dash="dot"),
                annotation_text=f"{int(row.touches)} touches",
                annotation_position="right",
                annotation_font=dict(size=10, color=_MUTED),
            )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _base_layout(fig, height=420, title=title)


def subscore_radar(sub_scores: dict[str, float | None], *, name: str = "") -> go.Figure:
    """Radar/spider chart of the seven category sub-scores (Section 8).

    Categories with no data are dropped rather than plotted at zero: a missing
    sentiment score is not a *bad* sentiment score, and a radar that shows it
    pinched to the origin says the opposite of what the data says (Section 7.5's
    coverage discipline, rendered visually).
    """
    present = {k: v for k, v in sub_scores.items() if v is not None}
    if len(present) < 3:
        return empty_figure("Not enough scored categories to plot a radar")

    labels = [humanize(k) for k in present]
    values = list(present.values())
    fig = go.Figure(
        go.Scatterpolar(
            r=[*values, values[0]],  # close the polygon
            theta=[*labels, labels[0]],
            fill="toself",
            name=name or "Sub-scores",
            line=dict(color=_ACCENT),
            fillcolor=_BAND,
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return _base_layout(fig, height=380)


# A generous calendar-days-per-trading-day ratio for sizing the session lookup
# window: the real figure is ~1.45 (365 / 252), so 1.6 plus a fortnight always
# spans far enough, including across a holiday-heavy stretch.
_CALENDAR_SPAN_PER_TRADING_DAY = 1.6


def horizon_dates(anchor: pd.Timestamp, offsets: Sequence[int]) -> list[pd.Timestamp]:
    """Real calendar dates `offsets` **trading** days after `anchor`.

    Every horizon in this project is counted in trading days -- `forecasting.
    DEFAULT_HORIZONS` is `(5, 20, 63, 252)`, i.e. a week, a month, a quarter and
    a year of *sessions* -- so plotting them by adding that many *calendar* days
    puts every forecast in the wrong place on a date axis, and increasingly so
    with distance: measured against the NYSE calendar from 2026-08-04, the
    20-day point landed 8 days early, the 63-day point 27 days early, and the
    252-day point **114 days early**. The one-year forecast was drawn at roughly
    the eight-month mark, next to eight-month-old price history, which is the
    kind of quietly-wrong picture Section 22 is about.

    Falls back to a proportional calendar estimate only if the exchange calendar
    somehow cannot reach the requested horizon, so a chart still draws.
    """
    if not offsets:
        return []
    steps = [int(offset) for offset in offsets]
    furthest = max(steps)
    window_end = anchor + pd.Timedelta(days=int(furthest * _CALENDAR_SPAN_PER_TRADING_DAY) + 21)
    sessions = [
        pd.Timestamp(day)
        for day in trading_days_between(anchor.date(), window_end.date())
        if pd.Timestamp(day) > anchor
    ]
    return [
        sessions[step - 1]
        if 0 < step <= len(sessions)
        else anchor + pd.Timedelta(days=round(step * 365.0 / 252.0))
        for step in steps
    ]


def forecast_fan_chart(
    history: pd.DataFrame, forecasts: pd.DataFrame, *, model_name: str | None = None
) -> go.Figure:
    """Recent price history plus the stored forecast point + band per horizon.

    Section 7.6 calls the fan chart "the most honest way to visually communicate
    'possible price forecast'" -- the band is the point of the chart, not
    decoration, so a forecast row lacking bounds is drawn as a point without a
    band rather than as a confident-looking line.
    """
    if forecasts.empty:
        return empty_figure("No forecasts generated for this symbol yet")

    selected = forecasts
    if model_name:
        selected = forecasts[forecasts["model_name"] == model_name]
    if selected.empty:
        return empty_figure(f"No stored forecasts from model '{model_name}'")

    selected = selected.sort_values("horizon_days")
    generated = pd.Timestamp(selected["generated_date"].iloc[0])
    # `horizon_days` counts trading sessions, not calendar days (see `horizon_dates`).
    future_dates = horizon_dates(generated, list(selected["horizon_days"]))

    fig = go.Figure()
    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["close"],
                name="Close",
                mode="lines",
                line=dict(color=_MUTED, width=1.5),
            )
        )

    upper = selected["upper_price"].tolist()
    lower = selected["lower_price"].tolist()
    if any(pd.notna(v) for v in upper) and any(pd.notna(v) for v in lower):
        fig.add_trace(
            go.Scatter(
                x=[*future_dates, *future_dates[::-1]],
                y=[*upper, *lower[::-1]],
                fill="toself",
                fillcolor=_BAND,
                line=dict(width=0),
                name="Forecast range",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=selected["point_price"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color=_ACCENT, dash="dash"),
        )
    )
    return _base_layout(fig, height=400)


def allocation_pie(weights: dict[str, float], *, title: str | None = None) -> go.Figure:
    """Portfolio allocation by sector or position (Section 9)."""
    positive = {k: v for k, v in weights.items() if v > 0}
    if not positive:
        return empty_figure("No priced holdings to allocate")
    fig = go.Figure(
        go.Pie(
            labels=list(positive),
            values=list(positive.values()),
            hole=0.45,
            textinfo="label+percent",
            sort=True,
        )
    )
    return _base_layout(fig, height=380, title=title)


def equity_curve_chart(strategy: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    """Backtest equity curve against the benchmark (Section 7.6's track record)."""
    if strategy.empty:
        return empty_figure("No backtest has been run yet")
    fig = go.Figure(
        go.Scatter(
            x=strategy.index,
            y=strategy.to_numpy(),
            name="Strategy",
            mode="lines",
            line=dict(color=_ACCENT, width=2),
        )
    )
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.to_numpy(),
                name="Benchmark",
                mode="lines",
                line=dict(color=_MUTED, width=1.6, dash="dot"),
            )
        )
    return _base_layout(fig, height=380)


def regime_gauge(regime_score: float | None, label: str | None = None) -> go.Figure:
    """The Market Regime Index as a 0-100 gauge (Sections 5, 12's homepage widget).

    Higher is risk-on. The band colors are a reading aid; the numeric value and
    the `label` text carry the meaning on their own.
    """
    if regime_score is None:
        return empty_figure("Market Regime Index has not been computed yet")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=regime_score,
            title={"text": humanize(label) if label else "Market Regime"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _ACCENT},
                "steps": [
                    {"range": [0, 35], "color": "rgba(207, 34, 46, 0.25)"},
                    {"range": [35, 65], "color": "rgba(154, 103, 0, 0.20)"},
                    {"range": [65, 100], "color": "rgba(45, 164, 78, 0.25)"},
                ],
            },
        )
    )
    return _base_layout(fig, height=260)


def correlation_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Holdings correlation matrix (Section 9's "are you diversified?" view)."""
    if matrix.empty:
        return empty_figure("Not enough overlapping price history to correlate holdings")
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="ρ"),
        )
    )
    return _base_layout(fig, height=380)


def sector_bar(counts: dict[str, Any], *, title: str | None = None) -> go.Figure:
    """Horizontal bar of a per-sector figure (rating counts, average score, ...)."""
    if not counts:
        return empty_figure("No sector data to summarize")
    ordered = dict(sorted(counts.items(), key=lambda kv: kv[1]))
    fig = go.Figure(
        go.Bar(
            x=list(ordered.values()),
            y=list(ordered),
            orientation="h",
            marker_color=_ACCENT,
        )
    )
    return _base_layout(fig, height=max(260, 26 * len(ordered) + 80), title=title)


def rating_distribution(counts: dict[str, int]) -> go.Figure:
    """Screener rating mix, in Strong Buy -> Strong Sell order with labelled bars."""
    present = {r: counts.get(r, 0) for r in RATING_DISPLAY}
    if not any(present.values()):
        return empty_figure("No ratings computed yet")
    labels = [f"{RATING_DISPLAY[r][0]} {RATING_DISPLAY[r][1]}" for r in present]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=list(present.values()),
            marker_color=[RATING_DISPLAY[r][2] for r in present],
            text=list(present.values()),
            textposition="outside",
        )
    )
    return _base_layout(fig, height=300)


def monte_carlo_fan_chart(history: pd.DataFrame, fan: Any) -> go.Figure:
    """A simulated-path fan: percentile price bands widening day by day.

    Distinct from `forecast_fan_chart`, which draws a stored model's point and
    band at each *stored horizon*. This draws a `forecasting.MonteCarloFanChart`
    -- a percentile price at every trading day out to the horizon -- so the
    uncertainty widens continuously the way a random walk's actually does,
    rather than stepping between a handful of horizons. Section 7.6 calls that
    the most honest way to show a possible price path.

    The median is drawn as a line; each symmetric percentile pair around it is
    drawn as a filled band. Nothing here computes anything: the simulation is
    `forecasting.monte_carlo_fan_chart`'s job, this only draws it.
    """
    if fan is None:
        return empty_figure("Not enough price history to simulate paths")

    generated = (
        pd.Timestamp(history["date"].iloc[-1]) if not history.empty else pd.Timestamp.today()
    )
    # `fan.days` are trading-day offsets (1..horizon_days), so they have to be
    # walked along the exchange calendar rather than added as calendar days --
    # otherwise the fan is drawn compressed into two-thirds of the time it
    # actually covers. See `horizon_dates`.
    future_dates = horizon_dates(generated, list(fan.days))

    fig = go.Figure()
    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["close"],
                name="Close",
                mode="lines",
                line=dict(color=_MUTED, width=1.5),
            )
        )

    levels = sorted(fan.percentiles)
    # Pair the outermost percentiles inward, so a 5/95 band sits behind a 25/75.
    for low, high in zip(levels, reversed(levels), strict=False):
        if low >= high:
            break
        fig.add_trace(
            go.Scatter(
                x=[*future_dates, *future_dates[::-1]],
                y=[*fan.percentiles[high], *fan.percentiles[low][::-1]],
                fill="toself",
                fillcolor=_BAND,
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{low:g}-{high:g}%",
            )
        )

    median = min(levels, key=lambda p: abs(p - 50.0))
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=fan.percentiles[median],
            name=f"{median:g}th percentile",
            mode="lines",
            line=dict(color=_ACCENT, width=2, dash="dot"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def interval_whisker(
    point: float | None,
    low: float | None,
    high: float | None,
    *,
    height: int = 44,
) -> go.Figure | None:
    """A bootstrap interval, drawn against zero.

    The one piece of drawing in this app that is about the *methodology* rather
    than about the data, and the counterpart of `IntervalWhisker` on the React
    Track Record page — the two front ends draw the same picture.

    Every headline figure on the Track Record page is a bootstrap estimate, and
    the interval around it is what decides whether the figure means anything: a
    Sharpe of 0.8 whose 90% interval runs from -0.4 to 2.0 has not been
    distinguished from luck, and the number alone cannot say so. The caption
    beside it says exactly that in words, and has since the page was written.
    This says it in one glance, so a reader who is skimming cannot skim past it.

    The bar is the interval, the tick inside it is the point estimate, and the
    hairline is zero. An interval that overlaps zero is drawn in the muted grey
    rather than the accent — redundantly with the caption, never instead of it.

    Returns None when there is nothing honest to draw, so the caller renders the
    caption alone rather than a bar with invented ends.
    """
    if point is None or low is None or high is None:
        return None
    if any(pd.isna(value) for value in (point, low, high)):
        return None

    # The drawn domain always contains zero, because zero is the whole point of
    # the picture. 12% padding keeps an endpoint off the edge, where it would
    # read as "continues off-screen".
    raw_low, raw_high = min(low, 0.0), max(high, 0.0)
    pad = (raw_high - raw_low or abs(point) or 1.0) * 0.12
    straddles_zero = low <= 0.0 <= high
    color = _MUTED if straddles_zero else _ACCENT

    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=raw_low - pad,
        x1=raw_high + pad,
        y0=0,
        y1=0,
        line=dict(color=_MUTED, width=1),
        opacity=0.35,
    )
    fig.add_shape(
        type="line",
        x0=low,
        x1=high,
        y0=0,
        y1=0,
        line=dict(color=color, width=7),
        opacity=0.4,
    )
    fig.add_shape(
        type="line",
        x0=0.0,
        x1=0.0,
        y0=-1,
        y1=1,
        line=dict(color=_MUTED, width=1),
    )
    fig.add_shape(
        type="line",
        x0=point,
        x1=point,
        y0=-0.8,
        y1=0.8,
        line=dict(color=color, width=2),
    )
    # An invisible trace carrying the whole story as one hover, since the shapes
    # above have no hover of their own.
    verdict = "includes zero" if straddles_zero else "excludes zero"
    fig.add_trace(
        go.Scatter(
            x=[point],
            y=[0],
            mode="markers",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            hovertemplate=(
                f"{point:.2f}<br>interval {low:.2f} to {high:.2f}<br>{verdict}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_xaxes(visible=False, range=[raw_low - pad, raw_high + pad], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], fixedrange=True)
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig
