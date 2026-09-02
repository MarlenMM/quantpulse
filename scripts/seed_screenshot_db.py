"""Build the throwaway database the README screenshots are taken against.

`docs/screenshots/README.md` has always described this recipe -- synthetic price
series for fifteen well-known tickers, run through the *real* scoring,
forecasting and backtest pipeline -- but only in prose, so refreshing the images
meant reconstructing it by hand and hoping the result resembled the last one.
This is that paragraph as a script. Same inputs every run (one seeded RNG, dates
anchored to a `--today` you pass in), so two people who run it get the same
pictures.

**Synthetic inputs, real pipeline.** Everything this file writes directly is a
made-up *input*: prices, fundamentals, analyst counts, sentiment, filings, macro
series. Every *output* on screen -- the sub-scores, the composite, the rating,
the forecasts, the backtest and its confidence intervals -- is computed by
importing the nightly job's own stages from `scripts/refresh_data.py` and
running them over those inputs. Hand-typing a plausible-looking Sharpe into the
table would make the screenshots fiction, and a screenshot of fiction is the
"fake product evidence" that makes a landing page untrustworthy. A number in
these images is a number this code produced.

Never point this at `quantpulse.db` or `quantpulse_demo.db`. It writes a scratch
file and refuses to touch either of those by name.

    uv run python scripts/seed_screenshot_db.py --out /tmp/shots.db
    DATABASE_URL=sqlite:////tmp/shots.db uv run streamlit run app/Home.py

Offline by design: the one stage of the nightly job that reaches the network for
the Market Regime Index's macro tone is replaced here with a stored reading, so
this runs on a plane and produces the same gauge either way.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from quantpulse.analysis import macro  # noqa: E402
from quantpulse.ingestion import fred_client  # noqa: E402
from quantpulse.news_intelligence import market_regime  # noqa: E402
from quantpulse.storage import persistence  # noqa: E402
from quantpulse.storage.models import (  # noqa: E402
    AnalystConsensus,
    Base,
    FundamentalsSnapshot,
    IndexMembershipHistory,
    InsiderTransaction,
    InstitutionalOwnership,
    MacroIndicator,
    NewsEvent,
    OptionsSignal,
    PriceHistory,
    RefreshLog,
    SentimentScore,
    ShortInterest,
    Ticker,
)

#: Databases this script must never write to, however it is invoked.
PROTECTED = {"quantpulse.db", "quantpulse_demo.db"}

#: Roughly four years of daily bars. The backtest needs several years of history
#: before it has enough rebalance periods to bootstrap an interval that is not
#: simply "too short to say" -- and a Track Record page whose headline is
#: "no confidence interval" is a poor advertisement for the page.
TRADING_DAYS = 1_010

#: Fifteen well-known tickers across eight sectors.
#:
#: Real names, entirely invented numbers. The point of using recognisable
#: symbols is that a reader can see at a glance that the table is a plausible
#: shape; the point of inventing the series is that no screenshot should imply
#: this project has ever had a view on Apple.
#:
#: `drift` and `vol` are annualised, and are chosen to spread the fifteen across
#: the rating scale: a screenshot in which everything is a Strong Buy shows
#: nothing about how the ranking behaves.
TICKERS: tuple[tuple[str, str, str, float, float], ...] = (
    ("AAPL", "Apple Inc.", "Information Technology", 0.16, 0.26),
    ("MSFT", "Microsoft Corporation", "Information Technology", 0.19, 0.24),
    ("NVDA", "NVIDIA Corporation", "Information Technology", 0.34, 0.44),
    ("JPM", "JPMorgan Chase & Co.", "Financials", 0.12, 0.22),
    ("BAC", "Bank of America Corp.", "Financials", 0.04, 0.25),
    ("XOM", "Exxon Mobil Corporation", "Energy", 0.14, 0.28),
    ("CVX", "Chevron Corporation", "Energy", 0.07, 0.24),
    ("JNJ", "Johnson & Johnson", "Health Care", 0.05, 0.16),
    ("UNH", "UnitedHealth Group Inc.", "Health Care", 0.11, 0.21),
    ("PG", "Procter & Gamble Co.", "Consumer Staples", 0.06, 0.15),
    ("KO", "Coca-Cola Company", "Consumer Staples", 0.03, 0.14),
    ("CAT", "Caterpillar Inc.", "Industrials", 0.13, 0.25),
    ("HON", "Honeywell International", "Industrials", 0.02, 0.20),
    ("META", "Meta Platforms Inc.", "Communication Services", 0.22, 0.35),
    ("NEE", "NextEra Energy Inc.", "Utilities", -0.03, 0.23),
)


def _trading_days(end: date, count: int) -> list[date]:
    """`count` weekdays ending at `end`. Holidays are not modelled: a gap in a
    synthetic series would be indistinguishable from a data bug in the pictures."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def seed_universe(session: Session, today: date) -> None:
    for symbol, name, sector, _, _ in TICKERS:
        session.add(
            Ticker(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=sector,
                exchange="NASDAQ",
                asset_type="equity",
                is_active=True,
                coverage="full",
            )
        )
        # The backtest is survivorship-aware and asks which symbols were in the
        # index on each rebalance date, so an unpopulated membership table
        # leaves it with an empty universe and nothing to trade.
        session.add(
            IndexMembershipHistory(
                index_name="SP500",
                symbol=symbol,
                added_date=today - timedelta(days=TRADING_DAYS * 2),
                removed_date=None,
            )
        )


def seed_prices(session: Session, today: date, rng: np.random.Generator) -> dict[str, float]:
    """Geometric Brownian motion per ticker, plus one shared market factor.

    The shared factor is what makes the sector-rotation table, the equal-weight
    benchmark and the beta calculation produce something other than noise: with
    fifteen independent random walks, every beta is zero and every sector's
    relative strength is a coin flip. Correlation is the part of a real market
    those views are actually reading.
    """
    days = _trading_days(today, TRADING_DAYS)
    market = rng.normal(0.0, 0.008, size=len(days))
    closes: dict[str, float] = {}

    for symbol, _, _, drift, vol in TICKERS:
        daily_drift = drift / 252.0
        daily_vol = vol / np.sqrt(252.0)
        # Two-thirds market, one-third idiosyncratic -- close to the equity
        # average, and enough that betas land either side of 1.
        shocks = 0.65 * market + 0.75 * rng.normal(0.0, daily_vol, size=len(days))
        level = 40.0 + rng.uniform(0.0, 220.0)
        for day, shock in zip(days, shocks, strict=True):
            level *= float(np.exp(daily_drift - 0.5 * daily_vol**2 + shock))
            intraday = abs(float(rng.normal(0.0, daily_vol))) * level
            session.add(
                PriceHistory(
                    symbol=symbol,
                    date=day,
                    open=level - intraday * 0.3,
                    high=level + intraday,
                    low=level - intraday,
                    close=level,
                    adj_close=level,
                    volume=int(rng.uniform(2e6, 4e7)),
                )
            )
        closes[symbol] = level
    return closes


def seed_fundamentals(session: Session, today: date, rng: np.random.Generator) -> None:
    """Four quarters of fundamentals, so the score is sector-relative not absolute."""
    for quarter in range(4):
        as_of = today - timedelta(days=30 + quarter * 91)
        for symbol, _, _, drift, _ in TICKERS:
            # Tie the fundamentals loosely to the drift, so a name that reads as
            # a compounder in the price chart is not simultaneously the worst
            # balance sheet on the page. Screenshots that contradict themselves
            # are the ones readers notice.
            quality = 0.5 + drift
            session.add(
                FundamentalsSnapshot(
                    symbol=symbol,
                    as_of_date=as_of,
                    pe=float(rng.uniform(11.0, 38.0)),
                    pb=float(rng.uniform(1.1, 9.0)),
                    ps=float(rng.uniform(1.0, 12.0)),
                    peg=float(rng.uniform(0.7, 3.4)),
                    eps=float(rng.uniform(2.0, 14.0)),
                    revenue_growth=float(np.clip(quality * 0.25 + rng.normal(0, 0.03), -0.1, 0.4)),
                    debt_equity=float(rng.uniform(0.2, 2.1)),
                    roe=float(np.clip(quality * 0.35 + rng.normal(0, 0.04), 0.02, 0.55)),
                    roa=float(np.clip(quality * 0.16 + rng.normal(0, 0.02), 0.01, 0.28)),
                    fcf=float(rng.uniform(1e9, 9e10)),
                    div_yield=float(rng.uniform(0.0, 0.042)),
                    sector_specific_metrics=None,
                )
            )


def seed_analyst(
    session: Session, today: date, closes: dict[str, float], rng: np.random.Generator
) -> None:
    for symbol, _, _, drift, _ in TICKERS:
        for quarter in range(3):
            enthusiasm = float(np.clip(0.5 + drift * 1.6, 0.05, 0.95))
            total = int(rng.integers(18, 42))
            strong_buy = int(total * enthusiasm * 0.45)
            buy = int(total * enthusiasm * 0.4)
            sell = int(total * (1 - enthusiasm) * 0.28)
            session.add(
                AnalystConsensus(
                    symbol=symbol,
                    as_of_date=today - timedelta(days=7 + quarter * 45),
                    strong_buy=strong_buy,
                    buy=buy,
                    hold=max(0, total - strong_buy - buy - sell),
                    sell=sell,
                    strong_sell=max(0, int(total * (1 - enthusiasm) * 0.06)),
                    mean_price_target=closes[symbol] * float(1.0 + drift * 0.55),
                )
            )


def seed_sentiment_and_news(session: Session, today: date, rng: np.random.Generator) -> None:
    for symbol, name, _sector, drift, _ in TICKERS:
        for offset in (1, 4, 9):
            session.add(
                SentimentScore(
                    symbol=symbol,
                    date=today - timedelta(days=offset),
                    source="news",
                    sentiment_score=float(np.clip(drift * 2.0 + rng.normal(0, 0.2), -1.0, 1.0)),
                    mention_volume=int(rng.integers(5, 90)),
                    total_weight=float(rng.uniform(1.0, 9.0)),
                )
            )
        session.add(
            NewsEvent(
                article_id=f"tier1-{symbol}",
                tier=1,
                title=f"{name} reports quarterly results",
                published_at=datetime.combine(today - timedelta(days=2), time(13, 30)),
                matched_symbols=[symbol],
                matched_theme=None,
                event_type="earnings",
                sentiment_score=float(np.clip(drift * 2.0, -1.0, 1.0)),
                source="synthetic",
                source_url=None,
            )
        )

    # Tier 2 and 3 are what the Home page's market-moving list shows, and they
    # feed the industry/macro category rather than any one symbol.
    macro_stories = [
        (2, "Semiconductor demand outlook lifts equipment orders", "semiconductors", 0.42),
        (2, "Refining margins narrow as crude spreads compress", "energy_transition", -0.28),
        (2, "Health insurers guide costs above consensus", None, -0.19),
        (3, "Payrolls come in above forecast; yields tick higher", None, 0.11),
        (3, "Committee holds rates, signals patience on cuts", None, 0.05),
        (3, "Manufacturing survey slips back into contraction", None, -0.33),
    ]
    for index, (tier, title, theme, tone) in enumerate(macro_stories):
        session.add(
            NewsEvent(
                article_id=f"tier{tier}-{index}",
                tier=tier,
                title=title,
                published_at=datetime.combine(today - timedelta(days=index % 4), time(9, 0)),
                matched_symbols=[],
                matched_theme=theme,
                event_type="macro" if tier == 3 else "industry",
                sentiment_score=tone,
                source="synthetic",
                source_url=None,
            )
        )


def seed_smart_money(session: Session, today: date, rng: np.random.Generator) -> None:
    for symbol, _, _, drift, vol in TICKERS:
        session.add(
            InstitutionalOwnership(
                symbol=symbol,
                quarter_end_date=today - timedelta(days=45),
                total_shares_held=float(rng.uniform(2e8, 3e9)),
                total_value=float(rng.uniform(1e10, 6e11)),
                num_filers=int(rng.integers(180, 900)),
                change_from_prior_quarter=float(
                    np.clip(drift * 0.3 + rng.normal(0, 0.03), -0.2, 0.2)
                ),
            )
        )
        session.add(
            ShortInterest(
                symbol=symbol,
                as_of_date=today - timedelta(days=12),
                # One name deliberately carries elevated short interest, so the
                # Stock Detail page's "cuts both ways" panel is on screen in the
                # screenshots rather than being a branch nobody ever sees.
                pct_float_short=12.4 if symbol == "NEE" else float(rng.uniform(0.4, 3.8)),
                days_to_cover=6.1 if symbol == "NEE" else float(rng.uniform(0.5, 2.6)),
            )
        )
        session.add(
            OptionsSignal(
                symbol=symbol,
                date=today - timedelta(days=1),
                expiration=(today + timedelta(days=30)).isoformat(),
                put_call_ratio=float(np.clip(1.0 - drift, 0.4, 1.9)),
                atm_implied_volatility=float(vol * rng.uniform(0.9, 1.3)),
                iv_rank=float(rng.uniform(10.0, 90.0)),
            )
        )
        for which in range(2):
            buying = drift > 0.1
            session.add(
                InsiderTransaction(
                    symbol=symbol,
                    insider_name=f"{symbol} Officer {which + 1}",
                    insider_title="Chief Financial Officer" if which else "Chief Executive Officer",
                    filing_date=today - timedelta(days=18 + which * 9),
                    transaction_date=today - timedelta(days=20 + which * 9),
                    transaction_code="P" if buying else "S",
                    acquired_disposed_code="A" if buying else "D",
                    shares=float(rng.uniform(2_000, 40_000)),
                    price_per_share=float(rng.uniform(30.0, 400.0)),
                    shares_owned_after=float(rng.uniform(50_000, 900_000)),
                )
            )


def seed_macro(session: Session, today: date, rng: np.random.Generator) -> None:
    """The four series the Market Regime Index reads, plus the cross-asset ones.

    A year of VIX rather than one reading: the index scores today's VIX as a
    *percentile* of its own recent history, so a single point has no percentile
    and the gauge falls back to a neutral reading that says nothing.
    """
    days = _trading_days(today, 260)
    vix = 17.0
    for day in days:
        vix = float(np.clip(vix + rng.normal(0, 0.7), 10.5, 34.0))
        session.add(MacroIndicator(date=day, indicator_name=macro.VIX, value=vix))

    for day in days[-90:]:
        session.add(
            MacroIndicator(
                date=day,
                indicator_name=fred_client.TREASURY_YIELD_10Y,
                value=4.28 + rng.normal(0, 0.05),
            )
        )
        session.add(
            MacroIndicator(
                date=day,
                indicator_name=fred_client.TREASURY_YIELD_2Y,
                value=3.91 + rng.normal(0, 0.05),
            )
        )
        session.add(
            MacroIndicator(date=day, indicator_name=macro.OIL_WTI, value=74.0 + rng.normal(0, 1.6))
        )
        session.add(
            MacroIndicator(
                date=day, indicator_name=macro.DOLLAR_INDEX, value=103.0 + rng.normal(0, 0.6)
            )
        )
        session.add(
            MacroIndicator(date=day, indicator_name=macro.GOLD, value=2350.0 + rng.normal(0, 22.0))
        )


def seed_regime(session: Session, today: date) -> None:
    """Today's regime reading, computed by the real index over the seeded inputs.

    The nightly job's own `refresh_market_regime` is not reused here for one
    reason: its macro-tone input is a live GDELT pull, which would make this
    script need the network and make the gauge different on every run. The tone
    is passed in as a stored reading instead; every other input, and the whole
    of the scoring, is `market_regime`'s own.
    """
    vix_level = persistence.read_latest_macro_value(session, macro.VIX, as_of=today)
    vix_history = persistence.read_macro_series(session, macro.VIX, as_of=today, lookback_days=365)
    spread = macro.yield_curve_spread(
        persistence.read_latest_macro_value(session, fred_client.TREASURY_YIELD_10Y, as_of=today),
        persistence.read_latest_macro_value(session, fred_client.TREASURY_YIELD_2Y, as_of=today),
    )
    prices = persistence.read_active_price_history(session, as_of=today, lookback_days=400)
    reading = market_regime.compute_market_regime(
        today,
        vix_level=vix_level,
        vix_history=vix_history,
        breadth_pct=market_regime.compute_breadth(prices, today),
        macro_tone=0.62,
        yield_curve_spread_value=spread,
    )
    persistence.upsert_market_regime(session, market_regime.regime_to_record(reading))


def seed_refresh_log(session: Session, today: date) -> None:
    """A believable Pipeline-health table, and the ages the freshness strip shows.

    The daily jobs ran today and the weekly ones a few days ago, which is what a
    healthy pipeline actually looks like -- and it means the freshness strip in
    the screenshot demonstrates its stale marking on the weekly sources instead
    of showing nine identical "today"s that prove nothing.
    """
    jobs = [
        ("prices", 0, "success", len(TICKERS)),
        ("composite_scores", 0, "success", len(TICKERS) * 3),
        ("forecasts", 0, "success", len(TICKERS) * 4),
        ("market_regime", 0, "success", 1),
        ("tier1_news", 1, "success", len(TICKERS)),
        ("fundamentals", 5, "success", len(TICKERS) * 4),
        ("analyst_consensus", 5, "success", len(TICKERS) * 3),
        ("backtest", 5, "success", 1),
    ]
    for name, days_ago, status, rows in jobs:
        session.add(
            RefreshLog(
                job_name=name,
                run_timestamp=datetime.combine(today - timedelta(days=days_ago), time(4, 15)),
                status=status,
                rows_updated=rows,
            )
        )


def run_pipeline(session: Session, today: date) -> dict[str, int]:
    """The nightly job's own compute stages, over the synthetic inputs.

    Imported from `scripts/refresh_data.py` rather than reimplemented, which is
    the whole claim `docs/screenshots/README.md` makes about these images: the
    ratings, forecasts and backtest figures in them came out of the same code
    that produces them in the app.
    """
    import refresh_data

    universe = persistence.read_ticker_universe(session)
    written: dict[str, int] = {}

    # Two scoring dates, a week apart, because "what the model changed its mind
    # about" needs two snapshots to diff -- with one it renders its own empty
    # state, which is not what the Home page looks like in use.
    previous = today - timedelta(days=7)
    written["scores (prior week)"] = refresh_data.refresh_composite_scores(
        session, universe, previous
    )
    written["scores (today)"] = refresh_data.refresh_composite_scores(session, universe, today)
    written["patterns"] = refresh_data.refresh_pattern_signals(session, universe, today)
    written["forecasts"] = refresh_data.refresh_forecasts(session, universe, today)
    written["backtest"] = refresh_data.refresh_backtest(session, today)
    return written


def build(out: Path, today: date, *, seed: int = 20260903) -> dict[str, int]:
    if out.name in PROTECTED:
        raise SystemExit(
            f"refusing to write {out.name}: this script builds a throwaway database, "
            "and the demo and local databases are not throwaway"
        )
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{out}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    rng = np.random.default_rng(seed)
    with factory() as session:
        seed_universe(session, today)
        closes = seed_prices(session, today, rng)
        seed_fundamentals(session, today, rng)
        seed_analyst(session, today, closes, rng)
        seed_sentiment_and_news(session, today, rng)
        seed_smart_money(session, today, rng)
        seed_macro(session, today, rng)
        session.commit()

        seed_regime(session, today)
        seed_refresh_log(session, today)
        session.commit()

    # The pipeline stages bind their own engine from `DATABASE_URL` in places,
    # so they get a session on this database explicitly.
    with factory() as session:
        written = run_pipeline(session, today)
        session.commit()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "build" / "screenshots.db")
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="The 'as of' date every window is measured back from (YYYY-MM-DD).",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args(argv)

    written = build(args.out.resolve(), args.today, seed=args.seed)
    print(f"wrote {args.out}")
    for stage, rows in written.items():
        print(f"  {stage}: {rows} rows")
    print(f"\nnow:  DATABASE_URL=sqlite:///{args.out.resolve()} uv run streamlit run app/Home.py")
    return 0


__all__ = ["TICKERS", "build", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
