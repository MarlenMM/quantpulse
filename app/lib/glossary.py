"""Plain-English definitions for every term the UI shows (Sections 10, 12).

Section 10 asks for "inline hover-tooltips (or a glossary page) explaining P/E,
RSI, Sharpe ratio, beta, etc. — worth doing given a recruiter reviewing your
live demo may not be a finance person."

**One definition, two surfaces.** This module is the single source of truth:
the Glossary page renders `TERMS` in full, and every `help=` tooltip across the
app pulls from the same dict via `tip()`. Writing a definition inline on a
widget would guarantee that the tooltip and the glossary drift apart the first
time one of them is edited — and a glossary that contradicts the tooltip beside
it is worse than no glossary.

Definitions are written for someone who is *not* a finance person: what the
number means and how to read it, not its formula. Where a term carries a
caveat this project takes seriously (a relative rating is not an absolute
judgment; VaR says nothing about how bad the tail gets), the caveat is part of
the definition rather than a footnote elsewhere — the tooltip is exactly where
a reader is asking the question.
"""

from __future__ import annotations

__all__ = ["TERMS", "CATEGORIES", "tip", "define", "search_terms"]

# term -> (category, plain-English definition)
TERMS: dict[str, tuple[str, str]] = {
    # -- QuantPulse's own outputs ------------------------------------------ #
    "Composite score": (
        "QuantPulse scores",
        "The single 0–100 number this app ranks stocks by. It blends seven categories "
        "(fundamentals, technicals, analysts, news sentiment, momentum, industry/macro "
        "news and smart money), each scored against the rest of the universe, then "
        "weighted. Higher is more attractive by this model's reckoning.",
    ),
    "Rating": (
        "QuantPulse scores",
        "Strong Buy → Strong Sell, assigned by where a stock's composite score ranks "
        "against its peers: top 10% Strong Buy, next 20% Buy, middle 40% Hold, next 20% "
        "Sell, bottom 10% Strong Sell. Because it is *relative*, some stocks are always "
        "rated Strong Buy — even in a market that is broadly expensive or falling.",
    ),
    "Percentile rank": (
        "QuantPulse scores",
        "Where this stock's composite score sits among all scored stocks. A percentile "
        "of 90 means it scored higher than 90% of the universe today.",
    ),
    "Data coverage": (
        "QuantPulse scores",
        "How much of the underlying data was actually available for this stock, 0–100. "
        "A thinly-covered small-cap with no analyst estimates and little news deserves "
        "less confidence than a mega-cap with everything — this number is how the app "
        "says so instead of presenting both with equal certainty.",
    ),
    "Market Regime Index": (
        "QuantPulse scores",
        "An in-house 0–100 gauge of overall market risk appetite, built from the VIX, "
        "how many stocks trade above their 200-day average, the tone of macro news, and "
        "the yield-curve spread. High is risk-on (calm, broad participation); low is "
        "risk-off (stressed).",
    ),
    # -- Valuation & fundamentals ------------------------------------------ #
    "P/E": (
        "Fundamentals",
        "Price-to-earnings: the share price divided by earnings per share. Roughly "
        "'how many years of current earnings you are paying for one share.' Only "
        "meaningful against companies in the same sector — a bank and a software firm "
        "are not comparable on it.",
    ),
    "P/B": (
        "Fundamentals",
        "Price-to-book: share price against the accounting value of the company's net "
        "assets. Most useful for banks and asset-heavy businesses.",
    ),
    "P/S": (
        "Fundamentals",
        "Price-to-sales: share price against revenue per share. Useful for companies "
        "that are growing but not yet profitable, where P/E is undefined.",
    ),
    "PEG": (
        "Fundamentals",
        "P/E divided by the earnings growth rate. An attempt to say whether a high P/E "
        "is justified by fast growth. Below ~1 is conventionally considered cheap.",
    ),
    "ROE": (
        "Fundamentals",
        "Return on equity: profit as a percentage of shareholders' money in the "
        "business. A rough measure of how efficiently a company turns capital into "
        "profit.",
    ),
    "ROA": (
        "Fundamentals",
        "Return on assets: profit as a percentage of everything the company owns. Like "
        "ROE, but not flattered by borrowing.",
    ),
    "Debt/Equity": (
        "Fundamentals",
        "How much the company has borrowed relative to shareholders' money. High "
        "leverage is risky for an industrial company and completely normal for a bank, "
        "which is why this app weights it differently by sector.",
    ),
    "Dividend yield": (
        "Fundamentals",
        "The annual dividend as a percentage of the share price — the cash income you "
        "would receive per dollar invested, if the dividend is maintained.",
    ),
    "Free cash flow": (
        "Fundamentals",
        "Cash left over after the company pays for running and maintaining the "
        "business. Harder to manipulate than reported earnings.",
    ),
    "FFO": (
        "Fundamentals",
        "Funds from operations — the REIT equivalent of earnings. Property accounting "
        "charges large depreciation that is not a real cash cost, so REITs are valued "
        "on FFO rather than P/E.",
    ),
    # -- Technicals -------------------------------------------------------- #
    "SMA": (
        "Technicals",
        "Simple moving average: the average closing price over the last N days. Price "
        "above its 200-day average is a common shorthand for 'in an uptrend'.",
    ),
    "RSI": (
        "Technicals",
        "Relative Strength Index, 0–100: how one-sided recent price moves have been. "
        "Above 70 is conventionally called overbought, below 30 oversold — but a strong "
        "trend can stay at an extreme for a long time.",
    ),
    "MACD": (
        "Technicals",
        "The gap between a fast and a slow moving average, used to spot a shift in "
        "trend. A rising histogram means upward momentum is building.",
    ),
    "ADX": (
        "Technicals",
        "Measures how *strong* a trend is, not which direction it points. Above ~25 is "
        "usually read as a genuine trend rather than choppy drift.",
    ),
    "ATR": (
        "Technicals",
        "Average True Range: the typical size of a day's price swing, in dollars. A "
        "plain-language measure of how much this stock moves around.",
    ),
    "Relative strength": (
        "Technicals",
        "How a stock has performed *compared to* its sector or the wider market. A "
        "stock rising only because everything is rising is a very different signal from "
        "one outperforming its peers.",
    ),
    "Support and resistance": (
        "Technicals",
        "Price levels the stock has repeatedly failed to fall below (support) or rise "
        "above (resistance) in the past.",
    ),
    # -- Risk -------------------------------------------------------------- #
    "Volatility": (
        "Risk",
        "How much a price bounces around, annualized. 20% means a typical year's swings "
        "are on that order. Higher volatility is not automatically bad — it is the "
        "price of the returns you are chasing.",
    ),
    "Beta": (
        "Risk",
        "How much a stock moves when the market moves. Beta 1 tracks the market; 1.5 "
        "tends to move 50% more in both directions; 0.5 about half as much. Always read "
        "it with R², which says how much of the stock's movement the market explains "
        "at all.",
    ),
    "R²": (
        "Risk",
        "How much of a stock's movement is explained by the market. A beta of 1.4 with "
        "an R² of 0.05 mostly means 'this does its own thing' — the opposite of what "
        "'beta 1.4' sounds like.",
    ),
    "Sharpe ratio": (
        "Risk",
        "Return earned per unit of volatility. It answers 'was the return worth the "
        "bumpiness?' Roughly: under 1 is unremarkable, above 2 is very good — but a "
        "Sharpe from a short sample is mostly noise, which is why this app shows a "
        "confidence interval next to it.",
    ),
    "Sortino ratio": (
        "Risk",
        "Like the Sharpe ratio, but it only counts *downside* movement as risk. Upside "
        "volatility is not something most investors want penalized, and Sortino is the "
        "version that agrees.",
    ),
    "Max drawdown": (
        "Risk",
        "The worst peak-to-trough fall over the period — the largest loss you would "
        "have sat through if you had bought at the worst moment and held. Shown as a "
        "negative number.",
    ),
    "Value at Risk": (
        "Risk",
        "VaR: a loss level that only the worst few percent of days exceed. 'Daily VaR "
        "(95%) of 2%' means: on the worst 5% of days, this lost 2% or more. It says "
        "nothing about how bad those days get — that is what expected shortfall is for.",
    ),
    "Expected shortfall": (
        "Risk",
        "The *average* loss on the days that breach the VaR threshold. It answers the "
        "question VaR structurally cannot: when it goes wrong, how wrong?",
    ),
    "Correlation": (
        "Risk",
        "How closely two holdings move together, from -1 to +1. Near +1 means they rise "
        "and fall as one — owning both is closer to owning one position twice than to "
        "being diversified.",
    ),
    "Herfindahl index": (
        "Risk",
        "HHI: a single number for how concentrated a portfolio is. 1/HHI reads as 'this "
        "is about as diversified as N equal-sized positions', regardless of how many "
        "you actually hold.",
    ),
    # -- Forecasting & backtesting ----------------------------------------- #
    "CAGR": (
        "Track record",
        "Compound annual growth rate: the steady yearly return that would produce the "
        "same end result as the actual bumpy path.",
    ),
    "Hit rate": (
        "Track record",
        "How often a forecast got the *direction* right, out of sample. Around 50% is "
        "a coin flip — which is the honest bar any forecasting model has to clear.",
    ),
    "Confidence interval": (
        "Track record",
        "A range the true value plausibly sits in. '90% CI [0.3, 1.3]' around a Sharpe "
        "of 0.8 means the result is consistent with anything from mediocre to good. If "
        "the interval includes zero, the result has not been distinguished from luck.",
    ),
    "Turnover": (
        "Track record",
        "How much of the portfolio is bought and sold at each rebalance. High turnover "
        "means more trading costs eating into returns.",
    ),
    "Transaction cost": (
        "Track record",
        "An assumed cost charged on every simulated trade (0.1% here) standing in for "
        "the bid-ask spread. Backtests that skip it flatter themselves.",
    ),
    "Benchmark": (
        "Track record",
        "What the strategy is compared against — here, buying and holding the whole "
        "universe. Beating a benchmark is the only version of 'good returns' that means "
        "anything.",
    ),
    "Monte Carlo simulation": (
        "Track record",
        "Simulating thousands of possible future price paths to show a *range* of "
        "outcomes rather than a single confident-looking prediction.",
    ),
    "Look-ahead bias": (
        "Track record",
        "The error of letting a backtest use information that would not have been known "
        "at the time. It makes results look brilliant and mean nothing.",
    ),
    "Survivorship bias": (
        "Track record",
        "Testing a strategy only against companies that still exist today, quietly "
        "excluding every one that went bankrupt or was acquired. It flatters results "
        "for reasons that have nothing to do with the strategy.",
    ),
    "Point-in-time": (
        "Track record",
        "Storing what the app concluded on each date and never rewriting it, so asking "
        "'what did this say on June 3rd?' returns what it actually said.",
    ),
    # -- Smart money ------------------------------------------------------- #
    "Insider transaction": (
        "Smart money",
        "A company officer or director buying or selling their own company's shares, "
        "disclosed on SEC Form 4. Individual sales are noisy (people buy houses); "
        "several different insiders buying at once is the meaningful pattern.",
    ),
    "13F filing": (
        "Smart money",
        "A quarterly disclosure of what large institutions hold. It shows whether "
        "professional money is accumulating or trimming a name — but arrives quarterly "
        "and in arrears, so it is a slow signal.",
    ),
    "Put/call ratio": (
        "Smart money",
        "How many bearish options are being traded relative to bullish ones. Elevated "
        "readings mean more hedging or negative positioning.",
    ),
    "IV rank": (
        "Smart money",
        "Where today's option-implied volatility sits within its own past year's range. "
        "High IV rank means the market expects unusually big moves ahead — often before "
        "a known event like earnings.",
    ),
    "Implied volatility": (
        "Smart money",
        "How much movement the options market is pricing in for the future — as opposed "
        "to historical volatility, which measures what already happened.",
    ),
    "Short interest": (
        "Smart money",
        "The share of a company's tradable stock that has been sold short. Read it two "
        "ways, not one: it can mean sophisticated money betting against the company, or "
        "fuel for a squeeze if sentiment turns.",
    ),
    # -- Portfolio bookkeeping --------------------------------------------- #
    "Cost basis": (
        "Portfolio",
        "What you actually paid for the shares you still hold, including across several "
        "purchases at different prices.",
    ),
    "FIFO": (
        "Portfolio",
        "First in, first out: when you sell part of a position, the shares you bought "
        "earliest are treated as the ones sold. This app uses FIFO to work out gains.",
    ),
    "Tax lot": (
        "Portfolio",
        "One purchase of shares at one price on one date. Selling consumes lots oldest "
        "first, which is why a single sale can produce both a long-term and a "
        "short-term gain.",
    ),
    "Holding period": (
        "Portfolio",
        "How long you have held a position — short-term under a year, long-term over. "
        "Shown here purely as description. This app does not compute tax; consult a "
        "professional.",
    ),
    "Unrealized P/L": (
        "Portfolio",
        "Profit or loss on paper, for positions you still hold. It becomes realized "
        "only when you sell.",
    ),
    "Rebalancing": (
        "Portfolio",
        "Buying and selling to move a portfolio back to a target set of weights — for "
        "example trimming a position that has grown to dominate the account.",
    ),
    "Efficient frontier": (
        "Portfolio",
        "The set of portfolios giving the most return available for each level of risk. "
        "The classic mean-variance optimization result.",
    ),
    "Hierarchical risk parity": (
        "Portfolio",
        "HRP: a way of building a diversified portfolio that groups assets by how they "
        "move together, avoiding the fragile step of predicting returns.",
    ),
    "Black-Litterman": (
        "Portfolio",
        "A method that blends a neutral market-baseline allocation with your own views "
        "on specific stocks — here, the app's composite scores act as those views.",
    ),
    # -- News -------------------------------------------------------------- #
    "Sentiment score": (
        "News",
        "How positive or negative recent coverage of a stock has been, from -1 to +1, "
        "scored by a finance-specific language model. Recent articles count for more "
        "than old ones.",
    ),
    "Event type": (
        "News",
        "What *kind* of story an article is (earnings, M&A, regulation, macro policy…), "
        "classified automatically. A Fed decision and an earnings beat should not move a "
        "score the same way.",
    ),
    "Tier 1 / 2 / 3 news": (
        "News",
        "Tier 1 names a specific company; Tier 2 moves a whole industry or theme (for "
        "example AI export controls); Tier 3 moves the entire market (Fed decisions, "
        "inflation prints).",
    ),
    "VIX": (
        "News",
        "The market's expectation of near-term S&P 500 volatility, often called the "
        "'fear index'. It spikes when markets are stressed.",
    ),
    "Yield curve spread": (
        "News",
        "The 10-year Treasury yield minus the 2-year. When it goes negative (an "
        "'inversion'), short-term rates exceed long-term ones — historically one of the "
        "better-known recession warnings.",
    ),
    "Market breadth": (
        "News",
        "How many stocks are participating in a move — for example the share trading "
        "above their 200-day average. A rally carried by a handful of names is narrower "
        "than the index alone suggests.",
    ),
}

# Display order for the Glossary page; every category in TERMS must appear here.
CATEGORIES: tuple[str, ...] = (
    "QuantPulse scores",
    "Fundamentals",
    "Technicals",
    "Risk",
    "Track record",
    "Smart money",
    "Portfolio",
    "News",
)


def define(term: str) -> str | None:
    """The definition for `term`, or `None` if it isn't in the glossary."""
    entry = TERMS.get(term)
    return entry[1] if entry else None


def tip(term: str, extra: str | None = None) -> str:
    """A tooltip string for Streamlit's `help=`, built from the glossary.

    Unknown terms fall back to `extra` (or the term itself) rather than raising:
    a missing tooltip is a cosmetic gap, and taking a page down over one would
    be a far worse trade. `extra` is appended to a known definition when a
    specific widget needs context the general definition shouldn't carry.
    """
    definition = define(term)
    if definition is None:
        return extra or term
    return f"**{term}** — {definition}" + (f"\n\n{extra}" if extra else "")


def search_terms(query: str) -> list[str]:
    """Glossary terms whose name or definition contains `query`, case-insensitively."""
    needle = query.strip().lower()
    if not needle:
        return list(TERMS)
    return [
        term
        for term, (_, definition) in TERMS.items()
        if needle in term.lower() or needle in definition.lower()
    ]
