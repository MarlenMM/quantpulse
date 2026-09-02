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

from collections.abc import MutableMapping
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd
import streamlit as st

from lib import charts, data
from lib.brand import PAGE_ICON
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
from lib.glossary import tip
from lib.search import format_choice, search_symbols
from quantpulse.analysis import clustering, risk
from quantpulse.portfolio import holdings as holdings_lib
from quantpulse.portfolio import optimization
from quantpulse.portfolio import recommendations as recs
from quantpulse.portfolio.rebalancing import (
    DEFAULT_TRANSACTION_COST,
    RebalancePlan,
    build_rebalance_plan,
)
from quantpulse.portfolio.transactions import Transaction, build_lot_book, holding_term, positions
from quantpulse.storage.db import get_session

st.set_page_config(page_title="QuantPulse — Portfolio", page_icon=PAGE_ICON, layout="wide")

DISCLAIMER = (
    "**Educational tool, not financial advice — neither you nor this app is a "
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
    # `SessionStateProxy` implements the MutableMapping protocol the store asks
    # for but doesn't declare it, so the cast states what's already true rather
    # than widening the store's own contract to accommodate Streamlit.
    return holdings_lib.SessionPortfolioStore(cast("MutableMapping[str, Any]", st.session_state))


def render_entry_form(store: holdings_lib.PortfolioStore) -> None:
    universe = data.universe()
    with st.expander("Add a transaction", expanded=False):
        # Section 31's autocomplete: type a company name, pick the ticker. The
        # lookup sits outside the form because a form only reruns on submit,
        # and suggestions have to update as you type.
        lookup = st.text_input(
            "Find a symbol",
            key="symbol_lookup",
            placeholder="Type a ticker or company name, e.g. 'Alphabet'",
            help="Fuzzy-matches company names as well as tickers.",
        )
        suggestions = search_symbols(universe, lookup) if lookup.strip() else []
        picked = ""
        if suggestions:
            picked = st.radio(
                "Matches",
                suggestions,
                horizontal=True,
                format_func=lambda s: format_choice(universe, s),
                key="symbol_suggestion",
            )
        elif lookup.strip():
            st.caption(
                f"No match for “{lookup}” in the scored universe — you can still type the "
                "ticker below; it just won't have analysis attached until the pipeline "
                "covers it."
            )

        with st.form("add_transaction", clear_on_submit=True):
            columns = st.columns(6)
            symbol = columns[0].text_input("Symbol", value=picked).strip().upper()
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
            try:
                # An unreadable file and an unreplayable one are both the file's
                # problem, not the app's -- e.g. a log that sells more shares
                # than it ever bought parses fine and then cannot be booked.
                # Surfaced here the same way the entry form surfaces its own
                # rejections, rather than escaping as a page traceback.
                holdings_lib.replace_transactions(store, parsed)
            except ValueError as exc:
                st.error(f"Could not restore that portfolio: {exc}")
            else:
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
                "Stale": "no price" if position.is_stale else "",
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
        column_config={
            "Avg cost": st.column_config.NumberColumn("Avg cost", help=tip("Cost basis")),
            "Unrealized": st.column_config.NumberColumn("Unrealized", help=tip("Unrealized P/L")),
            "Rating": st.column_config.TextColumn("Rating", help=tip("Rating")),
            "Term": st.column_config.TextColumn("Term", help=tip("Holding period")),
            "Stale": st.column_config.TextColumn(
                "Stale",
                help="Flagged when no current price is stored — delisted, acquired, or "
                "simply not yet ingested.",
            ),
        },
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
        help=tip("Unrealized P/L"),
    )


def render_risk(frame: pd.DataFrame, cash: float) -> pd.DataFrame:
    """Section 9's risk block; returns the price panel it read, for reuse below.

    The target-allocation section needs exactly the same panel, and reading it
    twice would both cost a second query and risk the two blocks describing
    different windows if either lookback were ever changed.
    """
    st.subheader("Risk & diversification", help=tip("Value at Risk"))
    empty = pd.DataFrame()
    priced = frame[frame["Value"].notna() & (frame["Value"] > 0)]
    if priced.empty:
        st.caption("No priced holdings to analyze.")
        return empty

    total = float(priced["Value"].sum()) + cash
    weights = {row.Symbol: float(row.Value) / total for row in priced.itertuples()}
    symbols = tuple(weights)
    end = date.today()
    panel = data.adj_close_panel(symbols, end - timedelta(days=RISK_LOOKBACK_DAYS), end)

    if panel.empty or panel.shape[1] < 1:
        st.caption("Not enough stored price history to compute portfolio risk yet.")
        return empty

    returns = risk.returns_panel(panel)
    usable = {s: w for s, w in weights.items() if s in returns.columns}
    if not usable:
        st.caption("None of the holdings have usable return history yet.")
        return panel

    # The market proxy must come from the WHOLE universe, not from `panel` --
    # `panel` holds only this portfolio's own symbols, so building the proxy
    # from it regressed the portfolio against an equal-weight version of
    # itself. For an equal-weight portfolio that is beta 1.0000 with R^2
    # 1.0000 *exactly*, and an R^2 of exactly 1 against "the market" is not a
    # number any real regression produces. Measured on five real names from the
    # demo database: 1.0000 (R^2 1.0000) the old way, 0.448 (R^2 0.238) against
    # the actual universe. Every other statistic here is correctly derived from
    # the holdings' own returns; only the proxy was wrong.
    market = risk.equal_weight_market_returns(data.universe_panel(risk.MARKET_PANEL_DAYS))
    summary = risk.portfolio_risk(
        returns,
        usable,
        cash_weight=cash / total if total > 0 else 0.0,
        market_returns=market if not market.empty else None,
    )

    columns = st.columns(5)
    columns[0].metric(
        "Volatility (ann.)", format_percent(summary.volatility), help=tip("Volatility")
    )
    columns[1].metric("Sharpe", format_ratio(summary.sharpe), help=tip("Sharpe ratio"))
    columns[2].metric("Sortino", format_ratio(summary.sortino), help=tip("Sortino ratio"))
    columns[3].metric(
        "Max drawdown", format_percent(summary.max_drawdown), help=tip("Max drawdown")
    )
    columns[4].metric(
        "Beta",
        format_ratio(summary.beta.beta) if summary.beta else "—",
        help=tip("Beta"),
    )
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
    # Sharpe and Sortino abstain on a short sample for the same reason VaR does
    # above; say so, rather than leaving two unexplained dashes side by side.
    ratio_floor = risk.min_ratio_observations(risk.TRADING_DAYS_PER_YEAR)
    if summary.sharpe is None and summary.n_observations < ratio_floor:
        st.caption(
            f"Sharpe and Sortino need about a year of shared history "
            f"({ratio_floor} daily returns; this mix has {summary.n_observations}). "
            "Both divide an average return by a measure of risk, which makes them much "
            "noisier than either part — on a few weeks of data they mostly report how "
            "recently the market went up, so they are left blank."
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

    render_correlation_clusters(summary.correlations)
    return panel


# At least this many holdings before clustering says anything a reader couldn't
# see by eye, and at least this many names per cluster on average so the answer
# isn't "every holding is its own group".
_MIN_HOLDINGS_TO_CLUSTER = 4
_NAMES_PER_CLUSTER = 2


def render_correlation_clusters(correlations: pd.DataFrame) -> None:
    """Group holdings that move together (Section 7.1's correlation clustering).

    The heatmap above shows every pairwise number; this answers the question a
    reader actually has, which is "how many genuinely different bets do I own?"
    Holding six names from one cluster is far less diversification than six
    holdings implies, and that is invisible in a grid of pairwise numbers.

    `risk.correlation_matrix` deliberately leaves a pair NaN when the two names
    share too little overlapping history, and `cluster_by_correlation` refuses
    to run on a matrix containing NaN rather than silently imputing. So thin
    names are dropped here first, and reported as dropped.
    """
    if correlations is None or correlations.empty:
        return

    usable = correlations.dropna(axis=0, how="any").dropna(axis=1, how="any")
    usable = usable.loc[usable.index, usable.index]
    dropped = [s for s in correlations.index if s not in usable.index]
    if len(usable) < _MIN_HOLDINGS_TO_CLUSTER:
        return

    n_clusters = max(2, min(len(usable) // _NAMES_PER_CLUSTER, 5))
    try:
        assignment = clustering.cluster_by_correlation(usable, n_clusters)
    except ValueError:
        return

    groups: dict[int, list[str]] = {}
    for symbol, cluster_id in assignment.items():
        groups.setdefault(cluster_id, []).append(symbol)

    st.markdown("**Correlation clusters**", help=tip("Correlation cluster"))
    for members in sorted(groups.values(), key=len, reverse=True):
        names = ", ".join(sorted(members))
        if len(members) > 1:
            st.markdown(f"- **{len(members)} names move together:** {names}")
        else:
            st.markdown(f"- **On its own:** {names}")
    st.caption(
        f"{len(groups)} distinct group(s) across {len(usable)} holdings. Names in the "
        "same group have tended to rise and fall together, so they are closer to one "
        "bet than to several — the count of holdings overstates diversification when "
        "they cluster."
        + (
            f" Excluded for too little shared history: {', '.join(sorted(dropped))}."
            if dropped
            else ""
        )
    )


# Section 27's three methods, in the order Section 27 itself argues for them:
# HRP first because it needs no expected-return estimate, Black-Litterman next
# because its views are anchored to an equilibrium, and plain mean-variance last
# with its own caveat attached. The labels say what each one actually uses, so
# the choice isn't between three opaque acronyms.
_OPTIMIZER_METHODS = {
    "Hierarchical Risk Parity — correlation structure only": "hrp",
    "Black-Litterman — equilibrium + your composite scores": "black_litterman",
    "Mean-variance (MPT) — minimum volatility": "min_volatility",
}


def _target_allocation(
    method: str, panel: pd.DataFrame, scores: pd.DataFrame
) -> tuple[optimization.OptimizedPortfolio | None, float | None]:
    """Run one optimizer over `panel`, returning `(result, relaxed_cap_or_None)`.

    The default 15% cap ties to Section 9's concentration threshold, but it
    cannot fill a portfolio of fewer than seven names -- `_validate_bounds`
    rejects that outright rather than letting the solver report an infeasible
    problem. A real personal portfolio is routinely smaller than seven names, so
    the cap is relaxed to the smallest feasible value and the caller is told,
    rather than the section simply refusing to appear.
    """
    n_assets = panel.shape[1]
    cap: float | None = optimization.DEFAULT_MAX_WEIGHT
    relaxed: float | None = None
    if cap is not None and cap * n_assets < 1.0:
        cap = relaxed = 1.0 / n_assets

    if method == "hrp":
        return optimization.hierarchical_risk_parity(panel), None
    if method == "black_litterman":
        held = [s for s in panel.columns if s in scores.index]
        composite = scores.loc[held, "composite_score"].astype(float)
        if len(composite) < panel.shape[1]:
            return None, relaxed
        return optimization.black_litterman_optimize(panel, composite, max_weight=cap), relaxed
    return optimization.mean_variance_optimize(
        panel, objective="min_volatility", max_weight=cap
    ), relaxed


def render_target_allocation(
    frame: pd.DataFrame, cash: float, panel: pd.DataFrame
) -> RebalancePlan | None:
    """Section 27: a target allocation and the concrete trades that reach it.

    The three optimizers and the trade-list generator were fully built, tested
    and reachable from no page at all -- the README advertised "3 optimization
    methods" that a user had no way to run. This is the seam Section 27
    describes: the same composite scores that drive the Screener become
    Black-Litterman's views, the optimizer proposes weights, and
    `build_rebalance_plan` turns the gap between those and what you hold into
    "sell 12 shares of X, buy 5 of Y" with its cost stated.

    Returns the plan so the recommendation block below can point at it rather
    than mentioning a rebalance in the abstract.
    """
    st.subheader("Target allocation & trades", help=tip("Efficient frontier"))
    priced = frame[frame["Value"].notna() & (frame["Value"] > 0)]
    if len(priced) < 2:
        st.caption("At least two priced holdings are needed to optimize an allocation.")
        return None
    held_panel = panel[[s for s in panel.columns if s in set(priced["Symbol"])]]
    # Every estimator needs a COMMON date window, so one recently-listed holding
    # truncates the sample for everything else -- on real data, a single 22-bar
    # name collapsed an eight-name 289-day panel to 22 usable rows and all three
    # optimizers (correctly) abstained. Naming the culprits and optimizing over
    # the rest beats an unexplained "no allocation could be computed".
    usable, excluded = optimization.usable_common_window(held_panel)
    if usable.shape[1] < 2:
        st.caption(
            "Not enough overlapping price history across your holdings to optimize yet — "
            "every method needs the same date window for all of them."
        )
        return None
    if excluded:
        st.caption(
            f"Excluded from the optimization for too little shared history: "
            f"**{', '.join(excluded)}**. The target weights below cover the remaining "
            f"{usable.shape[1]} holding(s); those names keep whatever you already hold."
        )

    label = st.radio("Method", list(_OPTIMIZER_METHODS), horizontal=False)
    st.caption(
        "**Hierarchical Risk Parity** never estimates expected returns — it clusters your "
        "holdings by how they move together and splits risk down that tree, which is why "
        "Section 27 prefers it. **Black-Litterman** starts from an equilibrium allocation "
        "and tilts it with this app's own composite scores. **Minimum-volatility** "
        "mean-variance uses only the covariance, avoiding the noisiest input of classic "
        "MPT. All three are descriptions of a model's beliefs, not forecasts — the "
        "Track Record page is where you find out whether any of it worked."
    )

    scores = data.screener_rows().set_index("symbol")
    with st.spinner("Optimizing…"):
        result, relaxed = _target_allocation(_OPTIMIZER_METHODS[label], usable, scores)
    if result is None:
        st.info(
            "No allocation could be computed from this portfolio — usually too little "
            "shared price history, or (for Black-Litterman) a holding the screener has "
            "not scored yet. Nothing is shown rather than a fabricated target."
        )
        return None
    if relaxed is not None:
        st.caption(
            f"Position cap relaxed to {relaxed:.0%} — the usual "
            f"{optimization.DEFAULT_MAX_WEIGHT:.0%} limit cannot fill a "
            f"{usable.shape[1]}-holding portfolio. A concentrated portfolio cannot be "
            "made diversified by an optimizer."
        )

    prices = {row.Symbol: float(row.Price) for row in priced.itertuples() if pd.notna(row.Price)}
    shares = {row.Symbol: float(row.Shares) for row in priced.itertuples()}
    plan = build_rebalance_plan(shares, prices, result.weights, cash=cash)
    if plan is None:
        st.caption("No portfolio value to reallocate.")
        return None

    comparison = pd.DataFrame(
        [
            {
                "Symbol": symbol,
                "Now": plan.current_weights.get(symbol, 0.0),
                "Target": plan.target_weights.get(symbol, 0.0),
                "After trades": plan.achieved_weights.get(symbol, 0.0),
            }
            for symbol in sorted(set(plan.current_weights) | set(plan.target_weights))
        ]
    )
    st.dataframe(
        comparison.style.format({"Now": "{:.1%}", "Target": "{:.1%}", "After trades": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )

    if not plan.trades:
        st.success("Already at target — no trades needed.")
    else:
        trades = pd.DataFrame(
            [
                {
                    "Action": trade.action.title(),
                    "Symbol": trade.symbol,
                    "Shares": trade.shares,
                    "Price": trade.price,
                    "Value": trade.trade_value,
                }
                for trade in plan.trades
            ]
        )
        st.dataframe(
            trades.style.format({"Shares": "{:.4g}", "Price": "${:,.2f}", "Value": "${:,.0f}"}),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        f"Sells first, then buys. Turnover {format_percent(plan.turnover)} of portfolio value; "
        f"estimated cost {format_money(plan.estimated_transaction_cost)} at "
        f"{format_percent(DEFAULT_TRANSACTION_COST, digits=2)} per unit of turnover — the same "
        "friction assumption the backtest charges itself. Cash "
        f"{format_money(plan.cash_before)} → {format_money(plan.cash_after)}. "
        "**Not an instruction to trade.**"
    )
    return plan


def render_recommendations(frame: pd.DataFrame, cash: float, plan: RebalancePlan | None) -> None:
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

    # The pointer passes through whatever plan the section above produced, so
    # "rebalance worth considering" can link to actual trades rather than to an
    # abstraction (`recommend` never computes one itself -- Section 27's split).
    result = recs.recommend(contexts, sector_candidates=candidates, rebalance_plan=plan)

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
    st.dataframe(
        table.style.format({"Weight": "{:.1%}"}),
        hide_index=True,
        width="stretch",
        column_config={
            "Action": st.column_config.TextColumn(
                "Action",
                help="Add / Hold / Trim / Sell, derived from the stock's current rating. "
                "An Add is downgraded to Hold when the position is already overweight.",
            ),
            "Weight": st.column_config.NumberColumn(
                "Weight", help="This position's share of total portfolio value."
            ),
        },
    )

    concentration = result.concentration
    columns = st.columns(2)
    columns[0].metric(
        "Position HHI", format_ratio(concentration.position_hhi), help=tip("Herfindahl index")
    )
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
        help=tip("Herfindahl index", "Computed over sector weights rather than positions."),
    )

    for warning in concentration.warnings:
        st.warning(warning.message)
    for gap in result.sector_gaps:
        st.info(gap.message)

    if result.rebalance.triggered:
        pointer = "Rebalance worth considering — " + "; ".join(result.rebalance.reasons) + "."
        if result.rebalance.plan is not None:
            pointer += (
                f" The target allocation above proposes {len(result.rebalance.plan.trades)} "
                "trade(s)."
            )
        st.caption(pointer)


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
    st.title("Portfolio & Watchlist")
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
    panel = render_risk(frame, state.cash)
    st.divider()
    plan = render_target_allocation(frame, state.cash, panel) if not panel.empty else None
    st.divider()
    render_recommendations(frame, state.cash, plan)
    st.divider()
    render_watchlist(store)


main()
