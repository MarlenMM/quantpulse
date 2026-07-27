"""Screener — the ranked, filterable stock table + Compare mode (Sections 8, 12).

Two things here are doing real work rather than just displaying a table:

* **The re-weighting sliders recompute the composite client-side.** Section 8
  asks for "custom score-weight sliders... recomputed client-side from stored
  sub-scores — no need to re-run the whole pipeline," which is possible only
  because the stored sub-scores are weight-INDEPENDENT by design (Section 7.5,
  and why the nightly job stores just the `balanced` profile). The recomputation
  here mirrors `scoring.build_composite`'s coverage rule exactly: renormalize
  over the categories that actually have data rather than treating a missing
  one as zero.
* **Compare mode** (Section 12) puts 2–4 tickers' sub-scores side by side,
  which is the cheapest possible way to make the ranking legible: a single
  score of 78 means little until you see what a 62 looks like next to it.
"""

import pandas as pd
import streamlit as st

from lib import charts, data
from lib.format import (
    RATING_ORDER,
    confidence_label,
    format_score,
    humanize,
    rating_label,
)
from quantpulse.analysis.investor_profiles import CATEGORIES, get_profile, profile_names

st.set_page_config(page_title="QuantPulse — Screener", page_icon="🔎", layout="wide")

SCORE_COLUMNS = {category: f"{category}_score" for category in CATEGORIES}


def reweight(rows: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Recompute the composite from stored sub-scores under caller-supplied weights.

    Deliberately identical in behavior to `scoring.build_composite`'s weighting
    step: the weighted sum is divided by the weight that actually had data, so a
    stock missing a category is neither penalized with a phantom zero nor
    silently boosted. Getting this wrong here would make the sliders quietly
    disagree with the stored ranking, which is worse than not having sliders.
    """
    weight_series = pd.Series(weights, dtype=float)
    sub = rows[[SCORE_COLUMNS[c] for c in weight_series.index]].copy()
    sub.columns = list(weight_series.index)
    present = sub.notna()
    available = present.mul(weight_series, axis=1).sum(axis=1)
    weighted = sub.fillna(0.0).mul(weight_series, axis=1).sum(axis=1)
    return weighted.where(available > 0).div(available.where(available > 0))


def main() -> None:
    st.title("🔎 Screener")
    rows = data.screener_rows()
    if rows.empty:
        st.info(
            "No composite scores stored yet. Run `scripts/refresh_data.py` to populate "
            "the ranking, then reload."
        )
        return

    st.caption(
        f"Ranking as of **{rows['date'].iloc[0]}** · {len(rows)} symbols scored. "
        "Ratings are *relative* — the top decile is Strong Buy however the market as a "
        "whole looks (Section 22)."
    )

    with st.sidebar:
        st.header("Filters")
        sectors = sorted({s for s in rows["sector"].dropna().unique()})
        chosen_sectors = st.multiselect("Sector", sectors, default=[])
        chosen_ratings = st.multiselect(
            "Rating", list(RATING_ORDER), default=[], format_func=rating_label
        )
        min_confidence = st.slider(
            "Minimum data coverage",
            0,
            100,
            0,
            help="Section 7.5's data-completeness score. Raise it to exclude thinly-covered names.",
        )
        search = st.text_input("Search symbol or company").strip().lower()

        st.header("Re-weight categories")
        st.caption(
            "Recomputed instantly from stored sub-scores — no pipeline re-run "
            "(the stored sub-scores are weight-independent by design)."
        )
        profile_name = st.selectbox(
            "Start from profile", profile_names(), format_func=humanize, index=0
        )
        base = get_profile(profile_name)
        weights = {
            category: st.slider(humanize(category), 0.0, 1.0, float(base.weights[category]), 0.05)
            for category in CATEGORIES
        }

    filtered = rows.copy()
    if chosen_sectors:
        filtered = filtered[filtered["sector"].isin(chosen_sectors)]
    if chosen_ratings:
        filtered = filtered[filtered["rating"].isin(chosen_ratings)]
    if min_confidence:
        filtered = filtered[filtered["data_confidence"].fillna(0) >= min_confidence]
    if search:
        mask = filtered["symbol"].str.lower().str.contains(search) | filtered["name"].fillna(
            ""
        ).str.lower().str.contains(search)
        filtered = filtered[mask]

    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized = {k: v / total_weight for k, v in weights.items()}
        filtered = filtered.assign(custom_score=reweight(filtered, normalized))
        filtered = filtered.sort_values("custom_score", ascending=False)
    else:
        st.warning("All category weights are zero — showing the stored ranking instead.")
        filtered = filtered.assign(custom_score=filtered["composite_score"])

    if filtered.empty:
        st.warning("No symbols match these filters.")
        return

    display = filtered.copy()
    display["Rating"] = display["rating"].map(rating_label)
    display["Coverage"] = display["data_confidence"].map(confidence_label)
    columns = {
        "symbol": "Symbol",
        "name": "Company",
        "sector": "Sector",
        "Rating": "Rating",
        "custom_score": "Score",
        "composite_score": "Stored",
        "Coverage": "Coverage",
    }
    st.dataframe(
        display[list(columns)]
        .rename(columns=columns)
        .style.format({"Score": "{:.1f}", "Stored": "{:.1f}"}),
        hide_index=True,
        width="stretch",
        height=460,
    )

    export, _ = st.columns([1, 4])
    export.download_button(
        "Download as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="quantpulse_screener.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Rating mix")
    st.plotly_chart(
        charts.rating_distribution(filtered["rating"].value_counts().to_dict()),
        width="stretch",
    )

    st.divider()
    st.subheader("Compare")
    st.caption("Pick 2–4 tickers to see every sub-score side by side (Section 12).")
    picks = st.multiselect(
        "Tickers",
        filtered["symbol"].tolist(),
        default=filtered["symbol"].head(2).tolist(),
        max_selections=4,
    )
    if len(picks) < 2:
        st.caption("Select at least two tickers to compare.")
        return

    compare = filtered[filtered["symbol"].isin(picks)].set_index("symbol")
    table = pd.DataFrame(
        {
            symbol: {
                humanize(category): compare.loc[symbol, SCORE_COLUMNS[category]]
                for category in CATEGORIES
            }
            | {
                "Composite": compare.loc[symbol, "custom_score"],
                "Rating": rating_label(compare.loc[symbol, "rating"]),
                "Coverage": format_score(compare.loc[symbol, "data_confidence"], digits=0),
            }
            for symbol in picks
        }
    )
    st.dataframe(table, width="stretch")

    radar_columns = st.columns(len(picks))
    for column, symbol in zip(radar_columns, picks, strict=True):
        sub_scores = {
            category: compare.loc[symbol, SCORE_COLUMNS[category]] for category in CATEGORIES
        }
        cleaned = {k: (None if pd.isna(v) else float(v)) for k, v in sub_scores.items()}
        column.plotly_chart(charts.subscore_radar(cleaned, name=symbol), width="stretch")
        column.caption(symbol)


main()
