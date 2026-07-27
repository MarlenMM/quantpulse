"""Portfolio & Watchlist — Function 2's front end (Sections 9, 25, 27, 30).

This page is where the whole Phase 8 engine finally has a user: holdings come
from `portfolio.holdings` (session or SQLite per ADR 4.5), positions are derived
FIFO from the transaction log by `portfolio.transactions`, risk comes from
`analysis.risk`, guidance from `portfolio.recommendations`, and the optional
target allocation + trade list from `portfolio.optimization` +
`portfolio.rebalancing`. The page itself computes nothing — it arranges.

ADR 4.5 in practice: the backend is chosen by `PORTFOLIO_BACKEND` and surfaced
in the UI, because "your holdings are saved to disk" and "your holdings vanish
when you close this tab" are facts a user must not have to guess between. The
session backend gets the CSV download/upload pair and the "Load example
portfolio" button Section 25 asks for so a demo visitor can still keep their
work and still see a populated page on arrival.

Per Section 9, the disclaimer banner lives on this page specifically — it is
the one making the most direct per-holding suggestions.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from lib import charts, data
from lib.format import (
    action_label,
    confidence_label,
    format_money,
    format_percent,
    format_ratio,
    format_signed_percent,
    humanize,
    rating_label,
)
from quantpulse.analysis import risk
from quantpulse.portfolio import holdings as holdings_lib
from quantpulse.portfolio import recommendations as recs
from quantpulse.portfolio.transactions import Transaction, build_lot_book, holding_term, positions
from quantpulse.storage.db import get_session

st.set_page_config(page_title="QuantPulse — Portfolio", page_icon="💼", layout="wide")

DISCLAIMER = (
    "⚠️ **Educational tool, not financial advice — neither you nor this app is a "
    "licensed advisor.** Per-holding suggestions below are mechanical outputs of the "
    "scoring model. Holding-period labels are descriptive only, never tax advice "
    "(consult a professional)."
)
RISK_LOOKBACK_DAYS = 420


def get_store() -> holdings_lib.PortfolioStore:
    """The configured store (ADR 4.5), with Streamlit's session state as the container.

    The SQLite store opens and commits its own session per operation. That
    matters here specifically: `st.rerun()` raises immediately to restart the
    script, so a commit deferred to the end of this page would never run and
    every write would be silently discarded.
    """
    if data.portfolio_backend() == "sqlite":
        return holdings_lib.SqlitePortfolioStore(get_session)
    return holdings_lib.SessionPortfolioStore(st.session_state)


def render_entry_form(store: holdings_lib.PortfolioStore) -> None:
    with st.expander("Add a transaction", expanded=False):
        with st.form("add_transaction", clear_on_submit=True):
            columns = st.columns(6)
            symbol = columns[0].text_input("Symbol").strip().upper()
            action = columns[1].selectbox("Action", ["buy", "sell"])
            shares = columns[2].number_input("Shares", min_value=0.0, step=1.0, format="%.4f")
            price = columns[3].number_input("Price", min_value=0.0, step=1.0, format="%.2f")
            traded_on = columns[4].date_input("Date", value=date.today())
            asset_type = columns[5].selectbox("Type", list(holdings_lib.ASSET_TYPES))
            submitted = st.form_submit_button("Add")
        if submitted:
            if not symbol or shares <= 0 or price <= 0:
                st.error("Symbol, shares and price are all required (shares/price must be > 0).")
            else:
                try:
                    store.add_transaction(
                        Transaction(
                            symbol=symbol,
                            action="buy" if action == "buy" else "sell",
                            shares=float(shares),
                            price=float(price),
                            date=traded_on,
                        ),
                        asset_type=asset_type,
                    )
                    st.success(f"Recorded {action} {shares:g} {symbol}.")
                    st.rerun()
                except ValueError as exc:
                    # e.g. selling more than is held -- surfaced, never silently absorbed.
                    st.error(str(exc))

        st.caption(
            "Enter cost basis in **post-split** terms — stored price history is "
            "split-adjusted, so a pre-split basis will not line up (Section 30)."
        )


def render_data_controls(store: holdings_lib.PortfolioStore) -> None:
    state = store.load()
    columns = st.columns(4)

    if columns[0].button("Load example portfolio"):
        store.save(holdings_lib.example_state())
        st.rerun()

    columns[1].download_button(
        "Download CSV",
        holdings_lib.to_csv(state).encode("utf-8"),
        file_name="quantpulse_transactions.csv",
        mime="text/csv",
        disabled=not state.transactions,
    )

    uploaded = columns[2].file_uploader(
        "Restore from CSV", type="csv", label_visibility="collapsed"
    )
    if uploaded is not None:
        try:
            parsed = holdings_lib.from_csv(uploaded.getvalue().decode("utf-8"))
        except ValueError as exc:
            st.error(f"Could not read that CSV: {exc}")
        else:
            holdings_lib.replace_transactions(store, parsed)
            st.success(f"Restored {len(parsed)} transactions.")
            st.rerun()

    if columns[3].button("Clear portfolio"):
        store.clear()
        st.rerun()


def render_positions(state: holdings_lib.PortfolioState) -> pd.DataFrame:
    """The holdings table; returns the priced frame the rest of the page reuses."""
    book = build_lot_book(state.transactions)
    symbols = tuple(book.open_lots)
    prices = data.latest_prices(symbols) if symbols else {}
    held = positions(book, current_prices=prices)
    if not held:
        st.info("No open positions yet — add a transaction or load the example portfolio.")
        return pd.DataFrame()

    universe = data.universe()
    sectors = dict(zip(universe["symbol"], universe["sector"], strict=False))
    scores = data.screener_rows().set_index("symbol")

    records = []
    for symbol, position in held.items():
        lots = book.open_lots[symbol]
        earliest = min(lot.purchase_date for lot in lots)
        score_row = scores.loc[symbol] if symbol in scores.index else None
        records.append(
            {
                "Symbol": symbol,
                "Shares": position.shares,
                "Avg cost": position.average_cost,
                "Price": position.current_price,
                "Value": position.market_value,
                "Unrealized": position.unrealized_gain,
                "Return": (
                    None
                    if position.unrealized_gain is None or position.cost_basis <= 0
                    else position.unrealized_gain / position.cost_basis
                ),
                "Sector": sectors.get(symbol) or "Unclassified",
                "Rating": rating_label(score_row["rating"]) if score_row is not None else "—",
                "Term": humanize(holding_term(earliest, as_of=date.today())),
                "Stale": "⚠️ no price" if position.is_stale else "",
            }
        )
    frame = pd.DataFrame(records)

    st.dataframe(
        frame.style.format(
            {
                "Shares": "{:.4g}",
                "Avg cost": "${:,.2f}",
                "Price": "${:,.2f}",
                "Value": "${:,.0f}",
                "Unrealized": "${:,.0f}",
                "Return": "{:+.1%}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    if (frame["Stale"] != "").any():
        st.warning(
            "One or more holdings have no stored price (delisted, acquired, or simply "
            "never ingested). They're shown with their cost basis and excluded from "
            "market-value totals rather than erroring out the page (Section 30)."
        )
    st.caption(
        "**Term** is a descriptive short/long holding-period flag based on purchase "
        "date — informational only, not tax advice."
    )
    return frame


def render_summary(frame: pd.DataFrame, cash: float) -> None:
    invested = float(frame["Value"].fillna(0).sum())
    cost = float((frame["Value"].fillna(0) - frame["Unrealized"].fillna(0)).sum())
    total = invested + cash
    columns = st.columns(4)
    columns[0].metric("Total value", format_money(total))
    columns[1].metric("Invested", format_money(invested))
    columns[2].metric("Cash", format_money(cash))
    columns[3].metric(
        "Unrealized P/L",
        format_money(invested - cost),
        delta=None if cost <= 0 else format_signed_percent((invested - cost) / cost),
    )


def render_risk(frame: pd.DataFrame, cash: float) -> None:
    st.subheader("Risk & diversification")
    priced = frame[frame["Value"].notna() & (frame["Value"] > 0)]
    if priced.empty:
        st.caption("No priced holdings to analyze.")
        return

    total = float(priced["Value"].sum()) + cash
    weights = {row.Symbol: float(row.Value) / total for row in priced.itertuples()}
    symbols = tuple(weights)
    end = date.today()
    panel = data.adj_close_panel(symbols, end - timedelta(days=RISK_LOOKBACK_DAYS), end)

    if panel.empty or panel.shape[1] < 1:
        st.caption("Not enough stored price history to compute portfolio risk yet.")
        return

    returns = risk.returns_panel(panel)
    usable = {s: w for s, w in weights.items() if s in returns.columns}
    if not usable:
        st.caption("None of the holdings have usable return history yet.")
        return

    market = risk.equal_weight_market_returns(panel)
    summary = risk.portfolio_risk(
        returns,
        usable,
        cash_weight=cash / total if total > 0 else 0.0,
        market_returns=market if not market.empty else None,
    )

    columns = st.columns(5)
    columns[0].metric("Volatility (ann.)", format_percent(summary.volatility))
    columns[1].metric("Sharpe", format_ratio(summary.sharpe))
    columns[2].metric("Sortino", format_ratio(summary.sortino))
    columns[3].metric("Max drawdown", format_percent(summary.max_drawdown))
    columns[4].metric("Beta", format_ratio(summary.beta.beta) if summary.beta else "—")
    if summary.value_at_risk is not None:
        var = summary.value_at_risk
        st.caption(
            f"Daily Value-at-Risk ({var.confidence:.0%}, {var.method}): "
            f"**{format_percent(var.var)}** — on the worst {1 - var.confidence:.0%} of days this "
            f"mix lost at least that much. Expected shortfall beyond it: "
            f"{format_percent(var.expected_shortfall)}. Based on {var.n_observations} observations."
        )
    else:
        st.caption(
            "Not enough history for an honest Value-at-Risk estimate yet "
            "(a 95% historical VaR needs ~100 observations)."
        )
    if summary.beta is not None and summary.beta.r_squared is not None:
        st.caption(
            f"Beta is measured against an equal-weight proxy for the market "
            f"(no S&P 500 price series is ingested), R² = {summary.beta.r_squared:.2f} "
            f"over {summary.beta.n_observations} days."
        )

    left, right = st.columns(2)
    with left:
        position_values = {row.Symbol: float(row.Value) for row in priced.itertuples()}
        sectors = {row.Symbol: row.Sector for row in priced.itertuples()}
        left.plotly_chart(
            charts.allocation_pie(
                holdings_lib.sector_weights(position_values, sectors), title="By sector"
            ),
            width="stretch",
        )
    with right:
        right.plotly_chart(charts.correlation_heatmap(summary.correlations), width="stretch")
        if summary.average_correlation is not None:
            st.caption(
                f"Average pairwise correlation: **{summary.average_correlation:.2f}**. "
                + (
                    "Most correlated pair: "
                    + ", ".join(
                        f"{a}/{b} at {rho:.2f}" for a, b, rho in summary.most_correlated[:1]
                    )
                    if summary.most_correlated
                    else ""
                )
            )


def render_recommendations(frame: pd.DataFrame, cash: float) -> None:
    st.subheader("Recommendations")
    scores = data.screener_rows().set_index("symbol")
    priced = frame[frame["Value"].notna() & (frame["Value"] > 0)]
    if priced.empty:
        st.caption("No priced holdings to advise on.")
        return

    total = float(priced["Value"].sum()) + cash
    contexts = {}
    for row in priced.itertuples():
        if row.Symbol not in scores.index:
            continue
        contexts[row.Symbol] = recs.HoldingContext(
            weight=float(row.Value) / total,
            rating=str(scores.loc[row.Symbol, "rating"]),
            sector=None if row.Sector == "Unclassified" else row.Sector,
        )
    if not contexts:
        st.caption(
            "None of your holdings have a stored composite score yet — the screener "
            "has to have scored a symbol before the app can suggest anything about it."
        )
        return

    universe = data.universe()
    candidates: dict[str, list[str]] = {}
    if not scores.empty:
        ranked = scores.reset_index().merge(
            universe[["symbol", "sector"]], on="symbol", how="left", suffixes=("", "_u")
        )
        for sector, group in ranked.dropna(subset=["sector"]).groupby("sector"):
            candidates[str(sector)] = (
                group.sort_values("composite_score", ascending=False)["symbol"].head(5).tolist()
            )

    result = recs.recommend(contexts, sector_candidates=candidates)

    table = pd.DataFrame(
        [
            {
                "Symbol": rec.symbol,
                "Action": action_label(rec.action),
                "Weight": rec.weight,
                "Why": rec.reason,
            }
            for rec in result.holdings
        ]
    )
    st.dataframe(table.style.format({"Weight": "{:.1%}"}), hide_index=True, width="stretch")

    concentration = result.concentration
    columns = st.columns(2)
    columns[0].metric("Position HHI", format_ratio(concentration.position_hhi))
    columns[0].caption(
        "As diversified as "
        f"{format_ratio(concentration.position_effective_count, digits=1)} equal-weighted "
        "positions."
        if concentration.position_effective_count
        else ""
    )
    columns[1].metric(
        "Sector HHI",
        format_ratio(concentration.sector_hhi) if concentration.sector_hhi else "—",
    )

    for warning in concentration.warnings:
        st.warning(warning.message)
    for gap in result.sector_gaps:
        st.info(gap.message)

    if result.rebalance.triggered:
        st.caption("Rebalance worth considering — " + "; ".join(result.rebalance.reasons) + ".")


def render_watchlist(store: holdings_lib.PortfolioStore) -> None:
    st.subheader("Watchlist")
    st.caption("Tracked but not owned (Section 9) — same analysis, no shares or cost basis.")
    state = store.load()
    columns = st.columns([3, 1])
    new_symbol = columns[0].text_input("Add symbol", key="watch_add").strip().upper()
    if columns[1].button("Add to watchlist") and new_symbol:
        store.add_to_watchlist(new_symbol)
        st.rerun()

    if not state.watchlist:
        st.caption("Watchlist is empty.")
        return

    scores = data.screener_rows().set_index("symbol")
    rows = []
    for symbol in state.watchlist:
        score_row = scores.loc[symbol] if symbol in scores.index else None
        rows.append(
            {
                "Symbol": symbol,
                "Rating": rating_label(score_row["rating"]) if score_row is not None else "—",
                "Score": None if score_row is None else float(score_row["composite_score"]),
                "Coverage": (
                    "—" if score_row is None else confidence_label(score_row["data_confidence"])
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format({"Score": "{:.1f}"}, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    to_remove = st.selectbox("Remove", ["—", *state.watchlist])
    if to_remove != "—":
        store.remove_from_watchlist(to_remove)
        st.rerun()


def main() -> None:
    st.title("💼 Portfolio & Watchlist")
    st.warning(DISCLAIMER)

    store = get_store()
    backend = store.backend
    if backend == "session":
        st.caption(
            "**Session mode** (ADR 4.5) — holdings live in this browser session only and "
            "are gone on refresh. Download the CSV to keep them."
        )
    else:
        st.caption("**Local mode** — holdings persist in your SQLite database.")

    render_data_controls(store)
    render_entry_form(store)

    state = store.load()
    cash_value = st.number_input(
        "Cash balance", min_value=0.0, value=float(state.cash), step=100.0, format="%.2f"
    )
    if cash_value != state.cash:
        store.set_cash(cash_value)
        state = store.load()

    if not state.transactions:
        st.divider()
        render_watchlist(store)
        return

    st.divider()
    frame = render_positions(state)
    if frame.empty:
        return

    render_summary(frame, state.cash)
    st.divider()
    render_risk(frame, state.cash)
    st.divider()
    render_recommendations(frame, state.cash)
    st.divider()
    render_watchlist(store)


main()
