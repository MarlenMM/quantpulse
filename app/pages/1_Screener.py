"""Screener — the ranked, filterable stock table + Compare mode (Sections 8, 12).

Two things here are doing real work rather than just displaying a table:

* **The re-weighting sliders recompute the composite client-side.** Section 8
  asks for "custom score-weight sliders... recomputed client-side from stored
  sub-scores — no need to re-run the whole pipeline," which is possible only
  because the stored sub-scores are weight-INDEPENDENT by design (Section 7.5,
  and why the nightly job stores just the `balanced` profile). Both re-scoring
  paths delegate to `scoring.build_composite` rather than reimplementing the
  weighting, so a slider cannot quietly disagree with the engine that produced
  the stored ranking — and the **rating** moves with the score, ranked against
  the whole scored universe rather than against whatever the filters left.
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


def _custom_profile(weights: dict[str, float], profile_name: str) -> InvestorProfile:
    """The slider weights as an `InvestorProfile`, renormalized to sum to 1."""
    profile = get_profile(profile_name)
    total = sum(weights.values())
    return InvestorProfile(
        name=profile.name,
        weights={c: w / total for c, w in weights.items()} if total > 0 else profile.weights,
        income_tilt=profile.income_tilt,
        prefer_low_volatility=profile.prefer_low_volatility,
    )


def _apply(rows: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Attach a `build_composite` result's score and rating back onto `rows`."""
    indexed = scored.set_index("symbol")
    merged = rows.set_index("symbol").copy()
    merged["custom_score"] = indexed["composite_score"]
    merged["rating"] = indexed["rating"]
    return merged.reset_index()


def rescore_relative(
    rows: pd.DataFrame,
    weights: dict[str, float],
    profile_name: str,
    *,
    regime_score: float | None = None,
) -> pd.DataFrame:
    """Re-score AND re-rate every row under the slider weights (Section 7.5 step 4).

    Delegates to `scoring.build_composite` rather than reimplementing the
    weighting, so the sliders cannot drift from the engine that produced the
    stored ranking -- the same discipline `rescore_absolute` follows. Passing
    the already-normalized sub-scores back in is safe and deliberate: they are
    cross-sectional percentiles, and percentiling a percentile is monotonic, so
    the ranking is untouched and only the weighting changes.

    **The rating is recomputed too, and that is the fix.** Dragging a slider
    used to change the Score column while the Rating column and the Rating-mix
    chart kept showing the *stored* balanced-profile verdict -- so a name could
    sit at the top of a re-weighted table labelled "Sell", and the rating
    histogram never moved however hard the weights were pushed. Absolute mode
    already re-rated, so the two modes also disagreed about whether a slider
    means anything.

    **Called on the whole scored universe, before any filtering.** A relative
    rating is defined against the peer group the nightly ranks -- "top decile of
    the market", not "top decile of what I happen to be looking at" -- so
    filtering to one sector must not promote its best name to Strong Buy.
    """
    sub = rows[[SCORE_COLUMNS[category] for category in CATEGORIES]].copy()
    sub.columns = list(CATEGORIES)
    sub.index = rows["symbol"]
    scored = scoring.build_composite(
        sub,
        profile=_custom_profile(weights, profile_name),
        rating_mode="relative",
        regime_score=regime_score,
    ).scores
    if scored.empty:
        return rows.assign(custom_score=rows["composite_score"])
    return _apply(rows, scored)


def rescore_absolute(
    rows: pd.DataFrame, weights: dict[str, float], profile_name: str
) -> pd.DataFrame | None:
    """Re-rate every row against a fixed bar, from the stored raw values.

    Delegates to `scoring.build_composite(..., rating_mode="absolute")` rather
    than reimplementing the mapping, so the page cannot drift from the engine --
    the same discipline `rescore_relative` follows.

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

    scored = scoring.build_composite(
        raw, profile=_custom_profile(weights, profile_name), rating_mode="absolute"
    ).scores
    if scored.empty:
        return None
    return _apply(rows, scored)


def latest_regime_score() -> float | None:
    """The most recent Market Regime Index reading, or None if none is stored.

    Fed into the relative re-rating so the risk-off Strong-Buy dampener (Section
    7.3 Tier 3) applies to a slider-driven rating exactly as it did to the
    stored one.
    """
    regime = data.market_regime(limit=1)
    if regime.empty:
        return None
    value = regime["regime_score"].iloc[-1]
    return None if pd.isna(value) else float(value)


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
            "Score **and rating** are recomputed instantly from stored sub-scores — "
            "no pipeline re-run, because the stored sub-scores are weight-independent "
            "by design. Ratings are always ranked against the whole scored universe, "
            'not against whatever the filters above left, so "top decile" keeps '
            "meaning the same thing."
        )
        profile_name = st.selectbox(
            "Start from profile", profile_names(), format_func=humanize, index=0
        )
        base = get_profile(profile_name)
        st.caption(base.description)
        weights = {
            category: st.slider(humanize(category), 0.0, 1.0, float(base.weights[category]), 0.05)
            for category in CATEGORIES
        }

    # Two profiles don't just re-weight the seven categories, they re-SCORE two
    # of them (Section 23): income ranks fundamentals against a dividend-leaning
    # sector config, conservative scores momentum toward low volatility. Neither
    # can be recovered by re-weighting a finished sub-score, so the nightly
    # stores their own rows and this reads them instead of the balanced ones.
    # Until that existed, picking either profile silently applied only its
    # weights while the sidebar described a scoring change.
    if base.income_tilt or base.prefer_low_volatility:
        tilted_rows = data.screener_rows(profile=profile_name)
        if tilted_rows.empty:
            st.warning(
                f"The **{humanize(profile_name)}** profile re-scores two categories rather "
                "than just re-weighting them, and no ranking has been stored for it yet — "
                "the next nightly run will write one. Showing the balanced sub-scores under "
                f"{humanize(profile_name)}'s weights until then, which is not the same thing."
            )
        else:
            rows = tilted_rows
            st.caption(
                f"Scored under the **{humanize(profile_name)}** profile — its sub-scores are "
                "genuinely different, not the balanced ones re-weighted."
            )

    # Re-score and re-rate the WHOLE universe first, then filter. Both orders
    # give the same Score, but only this one gives a rating that still means
    # "top decile of the market" rather than "top decile of this sector"; and
    # filtering by Rating afterwards then filters on what the table shows.
    total_weight = sum(weights.values())
    if total_weight <= 0:
        st.warning("All category weights are zero — showing the stored ranking instead.")
        scored = rows.assign(custom_score=rows["composite_score"])
    elif absolute_mode:
        rescored = rescore_absolute(rows, weights, profile_name)
        if rescored is None:
            st.warning(
                "These rows were scored before raw category values were stored, so an "
                "absolute rating cannot be derived from them. Showing the relative "
                "ranking — the next nightly run will populate them."
            )
            scored = rescore_relative(
                rows, weights, profile_name, regime_score=latest_regime_score()
            )
        else:
            scored = rescored
    else:
        scored = rescore_relative(rows, weights, profile_name, regime_score=latest_regime_score())

    filtered = scored.copy()
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

    if not search:
        # A search already ordered the rows by relevance; leave that alone.
        filtered = filtered.sort_values("custom_score", ascending=False)

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
            "Rating": st.column_config.TextColumn(
                "Rating",
                help=tip(
                    "Rating",
                    "Recomputed live from the sliders, ranked against the whole scored "
                    "universe rather than the filtered rows.",
                ),
            ),
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
    # Every cell goes through `format_score`, including the numeric sub-scores.
    # Each column here holds scores *and* the Rating/Coverage strings, so the
    # column dtype is object and neither a Styler format nor a NumberColumn
    # applies -- pandas falls back to repr, which printed a sub-score as
    # "98.80715705765407" (fourteen decimals of precision the score does not
    # have) and a missing category as "<NA>" instead of the em dash used
    # everywhere else in both front ends.
    table = pd.DataFrame(
        {
            symbol: {
                humanize(category): format_score(compare.loc[symbol, SCORE_COLUMNS[category]])
                for category in CATEGORIES
            }
            | {
                "Composite": format_score(compare.loc[symbol, "custom_score"]),
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
        # The `key` is load-bearing, not tidiness. Streamlit derives an
        # element's internal id from its type and contents, so two charts that
        # happen to be identical collide and the whole page dies with
        # `StreamlitDuplicateElementId` -- not a blank chart, a red traceback
        # where the Screener should be. Two radars are identical exactly when
        # neither can be drawn: `subscore_radar` returns the same "not enough
        # scored categories" placeholder for every symbol, and it takes no
        # `name`. That is not a hypothetical -- it is the deployed demo's normal
        # state, where only the price-derived categories have data, so the live
        # Screener page was down.
        column.plotly_chart(
            charts.subscore_radar(cleaned, name=symbol),
            width="stretch",
            key=f"compare_radar_{symbol}",
        )
        column.caption(symbol)


main()
