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
from lib.glossary import tip
from lib.search import format_choice, search_symbols
from quantpulse.analysis import scoring
from quantpulse.analysis.investor_profiles import (
    CATEGORIES,
    InvestorProfile,
    get_profile,
    profile_names,
)

st.set_page_config(page_title="QuantPulse — Screener", page_icon="🔎", layout="wide")

SCORE_COLUMNS = {category: f"{category}_score" for category in CATEGORIES}
# The pre-normalization inputs absolute mode re-scores from (see `rescore_absolute`).
RAW_COLUMNS = {category: f"{category}_raw" for category in CATEGORIES}


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


def rescore_absolute(
    rows: pd.DataFrame, weights: dict[str, float], profile_name: str
) -> pd.DataFrame | None:
    """Re-rate the filtered rows against a fixed bar, from the stored raw values.

    Delegates to `scoring.build_composite(..., rating_mode="absolute")` rather
    than reimplementing the mapping, so the page cannot drift from the engine --
    the same discipline `reweight` follows for the relative path.

    Returns `None` when the rows predate the raw columns, because an absolute
    rating genuinely cannot be recovered from a percentile. Saying so is the
    honest option; quietly showing relative ratings under an "absolute" label
    would be the exact mislabelling this mode was fixed to stop.
    """
    raw_columns = [RAW_COLUMNS[category] for category in CATEGORIES]
    if not set(raw_columns).issubset(rows.columns):
        return None
    raw = rows[raw_columns].copy()
    raw.columns = list(CATEGORIES)
    raw.index = rows["symbol"]
    if raw.notna().to_numpy().sum() == 0:
        return None

    profile = get_profile(profile_name)
    custom = InvestorProfile(
        name=profile.name,
        weights={c: w / sum(weights.values()) for c, w in weights.items()}
        if sum(weights.values()) > 0
        else profile.weights,
        income_tilt=profile.income_tilt,
        prefer_low_volatility=profile.prefer_low_volatility,
    )
    scored = scoring.build_composite(raw, profile=custom, rating_mode="absolute").scores
    if scored.empty:
        return None
    scored = scored.set_index("symbol")
    merged = rows.set_index("symbol").copy()
    merged["custom_score"] = scored["composite_score"]
    merged["rating"] = scored["rating"]
    return merged.reset_index()


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
            "Rating",
            list(RATING_ORDER),
            default=[],
            format_func=rating_label,
            help=tip("Rating"),
        )
        min_confidence = st.slider(
            "Minimum data coverage",
            0,
            100,
            0,
            help=tip(
                "Data coverage",
                "Raise this to exclude thinly-covered names from the table.",
            ),
        )
        search = st.text_input(
            "Search symbol or company",
            placeholder="e.g. GOOGL, Alphabet, or a typo like 'Alphabt'",
            help=(
                "Fuzzy-matches tickers and company names, so you don't have to remember "
                "that Alphabet is GOOGL."
            ),
        ).strip()

        st.header("Rating scheme")
        absolute_mode = (
            st.radio(
                "How should ratings be decided?",
                ("relative", "absolute"),
                format_func=lambda m: (
                    "Relative — rank against today's peers"
                    if m == "relative"
                    else "Absolute — judge against a fixed bar"
                ),
                help=tip("Rating"),
            )
            == "absolute"
        )
        st.caption(
            "**Relative** always names a top decile Strong Buy, however the whole "
            "market looks — that is Section 22's warning, not a bug. **Absolute** "
            "measures every category against a fixed bar instead, so a broadly "
            "falling market genuinely produces fewer Strong Buys (and can produce "
            "none). Needs the stored raw category values; rows written before those "
            "existed fall back to relative."
        )

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
        # Fuzzy rank across symbol AND company name (Section 31), then keep the
        # table in that relevance order rather than re-sorting alphabetically.
        matches = search_symbols(filtered[["symbol", "name"]], search, limit=len(filtered))
        filtered = filtered[filtered["symbol"].isin(matches)]
        if not filtered.empty:
            order = {symbol: rank for rank, symbol in enumerate(matches)}
            filtered = filtered.assign(_rank=filtered["symbol"].map(order)).sort_values("_rank")
            filtered = filtered.drop(columns="_rank")

    total_weight = sum(weights.values())
    if absolute_mode:
        rescored = rescore_absolute(filtered, weights, profile_name)
        if rescored is None:
            st.warning(
                "These rows were scored before raw category values were stored, so an "
                "absolute rating cannot be derived from them. Showing the relative "
                "ranking — the next nightly run will populate them."
            )
            # Fall back on the *filtered* rows, not the whole universe: the
            # sector/rating/coverage/search choices in the sidebar are still the
            # user's, and silently re-showing every symbol because the rating
            # scheme couldn't be applied is a different answer to a different
            # question.
            filtered = filtered.assign(custom_score=filtered["composite_score"])
        else:
            filtered = rescored.sort_values("custom_score", ascending=False)
    elif total_weight > 0:
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
        column_config={
            "Rating": st.column_config.TextColumn("Rating", help=tip("Rating")),
            "Score": st.column_config.NumberColumn(
                "Score",
                help=tip(
                    "Composite score",
                    "Recomputed live from the sliders in the sidebar.",
                ),
            ),
            "Stored": st.column_config.NumberColumn(
                "Stored", help="The composite score as computed and stored by the nightly job."
            ),
            "Coverage": st.column_config.TextColumn("Coverage", help=tip("Data coverage")),
        },
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
    universe_for_labels = filtered[["symbol", "name"]]
    picks = st.multiselect(
        "Tickers",
        filtered["symbol"].tolist(),
        default=filtered["symbol"].head(2).tolist(),
        max_selections=4,
        format_func=lambda symbol: format_choice(universe_for_labels, symbol),
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
