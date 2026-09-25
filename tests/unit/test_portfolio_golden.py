"""The demo's client-side portfolio must compute what the full app computes (finding 33).

The React demo gains a portfolio kept in the visitor's browser
(`frontend/src/lib/portfolio.ts`), a TypeScript port of the engine's FIFO lots,
per-holding recommendations, concentration checks and portfolio risk. Two
implementations of the same rules is how this project's front ends have
disagreed before -- three betas from three copies of one window -- so both are
pinned to one file: `frontend/tests/fixtures/portfolio-golden.json`.

This test recomputes every expected figure in that file with the engine's own
functions and requires them to match; `frontend/tests/portfolio.spec.ts` runs the
TypeScript over the same inputs and requires the same. The file changes only
when the rules do:

    uv run python tests/unit/test_portfolio_golden.py --write

The inputs are synthetic and seeded, so the test does not move with the data.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantpulse.analysis import risk
from quantpulse.portfolio import holdings as holdings_lib
from quantpulse.portfolio import recommendations as recs
from quantpulse.portfolio.transactions import Transaction, build_lot_book, holding_term, positions

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "tests"
    / "fixtures"
    / "portfolio-golden.json"
)


# --------------------------------------------------------------------- inputs


def _prices(seed: int, symbols: dict[str, float], days: int, start: str) -> dict[str, list]:
    """Correlated synthetic daily bars: {symbol: [[date, close, adj_close], ...]}."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    market = rng.normal(0.0004, 0.011, days)
    out: dict[str, list] = {}
    for i, (symbol, start_price) in enumerate(symbols.items()):
        beta = 0.6 + 0.3 * i
        daily = beta * market + rng.normal(0.0002, 0.012, days)
        close = start_price * np.cumprod(1 + daily)
        rows = []
        for d, c in zip(dates, close, strict=True):
            # A dividend-adjusted series for one name: adj_close != close before
            # the "ex-date", so a port that values holdings at adj_close, or
            # measures returns on close, fails.
            adj = c * (0.97 if symbol == "DDD" and d < dates[days // 2] else 1.0)
            rows.append([d.date().isoformat(), round(float(c), 4), round(float(adj), 4)])
        out[symbol] = rows
    return out


def _close_on(prices: dict[str, list], symbol: str, day: str) -> float:
    return round(next(c for d, c, _ in prices[symbol] if d == day), 2)


def _book_scenario() -> dict[str, Any]:
    prices = _prices(7, {"AAA": 120.0, "BBB": 45.0, "CCC": 210.0, "DDD": 60.0}, 300, "2025-06-02")
    # A few missing days for one name: returns across a gap stay missing, and
    # portfolio returns keep only complete cross-sections.
    prices["DDD"] = [row for i, row in enumerate(prices["DDD"]) if i not in {40, 41, 120, 200, 260}]
    day = [row[0] for row in prices["AAA"]]

    def tx(symbol: str, action: str, shares: float, i: int) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "price": _close_on(prices, symbol, day[i]),
            "date": day[i],
        }

    return {
        "name": "a year of trading",
        "cash": 5000.0,
        "transactions": [
            tx("AAA", "buy", 10, 5),
            tx("BBB", "buy", 20, 30),
            tx("CCC", "buy", 8, 100),
            tx("DDD", "buy", 50, 150),
            tx("AAA", "buy", 5, 200),
            tx("CCC", "sell", 3, 250),
            # Spans both AAA lots: the first is held more than a year (long),
            # two shares of the second are not (short).
            tx("AAA", "sell", 12, 290),
        ],
        "prices": prices,
        "sectors": {
            "AAA": "Information Technology",
            "BBB": "Information Technology",
            "CCC": "Health Care",
            "DDD": "Financials",
        },
        "ratings": {"AAA": "strong_buy", "BBB": "hold", "CCC": "sell", "DDD": "buy"},
        "sector_candidates": {"Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX"]},
    }


def _short_history_scenario() -> dict[str, Any]:
    prices = _prices(11, {"EEE": 30.0, "FFF": 80.0}, 25, "2026-06-01")
    day = [row[0] for row in prices["EEE"]]
    return {
        "name": "too little history to measure risk",
        "cash": 0.0,
        "transactions": [
            {
                "symbol": "EEE",
                "action": "buy",
                "shares": 100,
                "price": _close_on(prices, "EEE", day[0]),
                "date": day[0],
            },
            {
                "symbol": "FFF",
                "action": "buy",
                "shares": 10,
                "price": _close_on(prices, "FFF", day[0]),
                "date": day[0],
            },
        ],
        "prices": prices,
        "sectors": {"EEE": "Energy", "FFF": None},
        "ratings": {"EEE": "strong_buy", "FFF": "strong_sell"},
        "sector_candidates": {},
    }


# ------------------------------------------------------------------- expected


def _num(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def expected(scenario: dict[str, Any]) -> dict[str, Any]:
    """Every figure the demo shows, computed with the engine's own functions."""
    transactions = [
        Transaction(
            symbol=t["symbol"],
            action=t["action"],
            shares=float(t["shares"]),
            price=float(t["price"]),
            date=date.fromisoformat(t["date"]),
        )
        for t in scenario["transactions"]
    ]
    book = build_lot_book(transactions)
    last_close = {s: rows[-1][1] for s, rows in scenario["prices"].items()}
    held = positions(book, current_prices=last_close)
    cash = float(scenario["cash"])
    total = sum(p.market_value or 0.0 for p in held.values()) + cash
    weights = {s: (p.market_value or 0.0) / total for s, p in held.items()}

    sectors = scenario["sectors"]
    contexts = {
        s: recs.HoldingContext(
            weight=weights[s], rating=scenario["ratings"][s], sector=sectors.get(s)
        )
        for s in held
    }
    result = recs.recommend(contexts, sector_candidates=scenario["sector_candidates"])

    panel = pd.DataFrame(
        {
            s: pd.Series({pd.Timestamp(d): adj for d, _, adj in rows})
            for s, rows in scenario["prices"].items()
            if s in held
        }
    ).sort_index()
    returns = risk.returns_panel(panel)
    summary = risk.portfolio_risk(returns, weights, cash_weight=cash / total)
    var = summary.value_at_risk
    corr = summary.correlations

    return {
        "positions": {
            s: {
                "shares": p.shares,
                "cost_basis": p.cost_basis,
                "average_cost": p.average_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_gain": p.unrealized_gain,
                "weight": weights[s],
            }
            for s, p in sorted(held.items())
        },
        "open_lots": {
            s: [
                {
                    "purchase_date": lot.purchase_date.isoformat(),
                    "shares": lot.shares,
                    "cost_basis": lot.cost_basis,
                }
                for lot in lots
            ]
            for s, lots in sorted(book.open_lots.items())
        },
        "realized": [
            {
                "symbol": g.symbol,
                "sale_date": g.sale_date.isoformat(),
                "purchase_date": g.purchase_date.isoformat(),
                "shares": g.shares,
                "proceeds": g.proceeds,
                "cost_basis": g.cost_basis,
                "gain": g.gain,
                "term": g.term,
            }
            for g in book.realized_gains
        ],
        "total_value": total,
        "sector_weights": holdings_lib.sector_weights(
            {s: p.market_value or 0.0 for s, p in held.items()}, sectors
        ),
        "recommendations": {
            r.symbol: {"action": r.action, "reason": r.reason} for r in result.holdings
        },
        "concentration": {
            "position_hhi": result.concentration.position_hhi,
            "position_effective_count": result.concentration.position_effective_count,
            "sector_hhi": result.concentration.sector_hhi,
            "sector_effective_count": result.concentration.sector_effective_count,
            "warnings": [w.message for w in result.concentration.warnings],
        },
        "sector_gaps": [g.message for g in result.sector_gaps],
        "rebalance_reasons": result.rebalance.reasons,
        "risk": {
            "n_observations": summary.n_observations,
            "volatility": _num(summary.volatility),
            "max_drawdown": _num(summary.max_drawdown),
            "var": _num(var.var) if var else None,
            "expected_shortfall": _num(var.expected_shortfall) if var else None,
            "average_correlation": _num(summary.average_correlation),
            "correlations": {
                a: {b: _num(corr.loc[a, b]) for b in corr.columns} for a in corr.index
            },
        },
    }


#: The one-year rule's edges: the anniversary itself is still short-term ("more
#: than one year"), and a Feb 29 purchase's anniversary is Feb 28 in a common year.
_TERM_CASES = [
    ("2025-03-10", "2026-03-10"),
    ("2025-03-10", "2026-03-11"),
    ("2024-02-29", "2025-02-28"),
    ("2024-02-29", "2025-03-01"),
    ("2023-02-28", "2024-02-28"),
    ("2023-02-28", "2024-02-29"),
]


def build() -> dict[str, Any]:
    scenarios = [_book_scenario(), _short_history_scenario()]
    example = holdings_lib.example_state()
    return {
        # The full app's "Load example portfolio" and its CSV format, so the
        # demo's copies of both are pinned to the engine's.
        "example": {
            "cash": example.cash,
            "transactions": [
                {
                    "symbol": tx.symbol,
                    "action": tx.action,
                    "shares": tx.shares,
                    "price": tx.price,
                    "date": tx.date.isoformat(),
                }
                for tx in example.transactions
            ],
        },
        "csv_columns": list(holdings_lib.CSV_COLUMNS),
        "engine_csv": holdings_lib.to_csv(
            holdings_lib.PortfolioState(
                transactions=[
                    Transaction(
                        symbol=t["symbol"],
                        action=t["action"],
                        shares=float(t["shares"]),
                        price=float(t["price"]),
                        date=date.fromisoformat(t["date"]),
                    )
                    for t in _book_scenario()["transactions"]
                ],
                cash=0.0,
                watchlist=[],
            )
        ),
        "holding_terms": [
            [
                purchase,
                as_of,
                holding_term(date.fromisoformat(purchase), as_of=date.fromisoformat(as_of)),
            ]
            for purchase, as_of in _TERM_CASES
        ],
        "about": (
            "Inputs and the engine's own results for frontend/src/lib/portfolio.ts. "
            "Regenerate with: uv run python tests/unit/test_portfolio_golden.py --write"
        ),
        "scenarios": [{"inputs": s, "expected": expected(s)} for s in scenarios],
    }


# ---------------------------------------------------------------------- tests


def _close(actual: Any, wanted: Any, path: str) -> None:
    if isinstance(wanted, dict):
        assert isinstance(actual, dict) and set(actual) == set(wanted), path
        for key in wanted:
            _close(actual[key], wanted[key], f"{path}.{key}")
    elif isinstance(wanted, list):
        assert isinstance(actual, list) and len(actual) == len(wanted), path
        for i, (a, w) in enumerate(zip(actual, wanted, strict=True)):
            _close(a, w, f"{path}[{i}]")
    elif isinstance(wanted, float):
        assert actual == pytest.approx(wanted, rel=1e-9, abs=1e-12), path
    else:
        assert actual == wanted, path


def test_the_golden_file_is_what_the_engine_computes() -> None:
    committed = json.loads(GOLDEN.read_text())
    _close(build(), committed, "golden")


def test_the_scenarios_exercise_what_they_claim() -> None:
    """A golden file of easy cases would pin nothing."""
    book, short = (s["expected"] for s in build()["scenarios"])
    terms = {g["term"] for g in book["realized"] if g["symbol"] == "AAA"}
    assert terms == {"long", "short"}, "the AAA sale must span a long and a short lot"
    assert {r["action"] for r in book["recommendations"].values()} >= {"add", "hold", "trim"}
    assert book["concentration"]["warnings"], "no concentration warning fired"
    assert any("Energy" in g and "XOM" in g for g in book["sector_gaps"])
    assert book["risk"]["var"] is not None and book["risk"]["n_observations"] >= 100
    assert short["risk"]["var"] is None, "25 bars must be too few for historical VaR"
    assert short["risk"]["volatility"] is not None, "24 returns still clear the volatility floor"
    assert short["risk"]["average_correlation"] is None


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {GOLDEN}")
