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

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from lib.format import RATING_DISPLAY, humanize

__all__ = [
    "empty_figure",
    "price_chart",
    "subscore_radar",
    "forecast_fan_chart",
    "allocation_pie",
    "equity_curve_chart",
    "regime_gauge",
    "correlation_heatmap",
    "sector_bar",
]

# One neutral accent used wherever a series has no semantic color of its own.
_ACCENT = "#3b82f6"
_MUTED = "#6e7781"
_BAND = "rgba(59, 130, 246, 0.18)"


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
    ohlcv: pd.DataFrame, *, overlays: dict[str, pd.Series] | None = None, title: str | None = None
) -> go.Figure:
    """Candlestick price chart with optional moving-average overlays."""
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
            increasing_line_color="#2da44e",
            decreasing_line_color="#cf222e",
        )
    )
    for label, series in (overlays or {}).items():
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=series, name=label, mode="lines", line=dict(width=1.4))
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
    future_dates = [generated + pd.Timedelta(days=int(h)) for h in selected["horizon_days"]]

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
