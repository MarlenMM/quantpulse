# How to use QuantPulse

Plain-English guide. No setup, no keys, no payment — none of this costs anything.

---

## Three ways in

**1. The public link — nothing to install**

**https://marlenmm.github.io/quantpulse/**

Open it on any device. It shows the research side of the app: rankings, per-stock
detail, charts, forecasts, the track record, the glossary, and a Portfolio page.
It refreshes itself every weekday evening, after the US market closes; the
Dashboard's freshness strip shows the date it is actually showing.

It is a set of files on a free host with nothing running behind it, so there is
nothing to log in to. Its **Portfolio page keeps your holdings in your own
browser** — nothing is sent anywhere — and it covers positions, profit and loss,
risk and suggestions. The optimisers, the rebalancing trade list and the history
need the full app.

**2. The full app, hosted — nothing to install either**

**https://quantpulse-demo.streamlit.app**

All seven pages, including the **Portfolio Manager** with its optimisers,
rebalancing and history. It sleeps when nobody has used it for a while, so the
first visit takes about a minute to wake it.

**3. On your own machine — the whole app**

```bash
git clone https://github.com/MarlenMM/quantpulse.git
cd quantpulse
./run.sh
```

Then open http://localhost:8501. That is the same seven-page app.

The first run takes a few minutes: it installs what it needs and downloads the
published demo database once. After that it starts in seconds. It needs Python
3.12 — if you don't have it, the script tells you the one line to run. No
account, no API key, no bill.

---

## What each page is for

The public link has six of these — everything except Settings, with a
browser-local Portfolio page. The full app has all seven.

| Page | What it's for | What you can do |
|---|---|---|
| **Dashboard** | The market at a glance | See today's top-ranked names, whether the market is risk-on or risk-off, which sectors money moved into this month, and which ratings changed since the last refresh |
| **Screener** | The ranked list of all 503 companies | Sort and filter the whole S&P 500 by score, rating or sector; search by name; switch investor profile (below); compare up to four names side by side; download it as a spreadsheet |
| **Stock Detail** | One company, in depth | Price chart, the category scores behind its rating, detected chart patterns, forecasts a week and a month ahead, each with its own track record, volatility/Sharpe/Sortino/beta, and a simulated range of future prices |
| **Backtest / Track Record** | Did the ranking actually work? | The historical result of following the **momentum** half of the ranking, with confidence intervals and a benchmark to compare against. The page names its own signal and says plainly why it is not yet the full Buy/Sell rating — five of the seven scoring categories have only weeks of stored history, and ranking a 2023 rebalance by 2026 fundamentals would be look-ahead bias |
| **Portfolio Manager** | Your own holdings | Enter what you own, see risk, correlation, sector concentration, add/trim/hold/sell suggestions, and three target allocations with a concrete trade list. On the public link: the browser-local subset described above |
| **Settings** *(full app only)* | What's switched on | Which data sources are configured and when each last ran |
| **Glossary** | Every term explained | 74 definitions in plain English; searchable by concept, not just by name |

**Investor profiles** (on the Screener) re-rank the same companies for a
different priority: balanced, value, growth, income, momentum, conservative.
Same data, different weighting — not a different opinion.

**Portfolio Manager tip:** there's a "Load example portfolio" button. Click it
to see everything the page can do before typing in anything of your own.
In the hosted app, holdings live in your browser session and vanish on refresh —
use "Download CSV" to keep them, and "Upload" to bring them back. A CSV exported
from the public link's Portfolio page imports unchanged.

---

## Which numbers to trust, and which are thin

This is the part worth reading. **The app is honest about its own limits, and it
says so on screen — you just need to know where to look.**

**Coverage.** Every stock shows a coverage figure, e.g. *"good coverage (80%)"*.
The composite score blends seven categories; coverage is how much of that
weighting actually had data. A category with no data is left out and the score
is re-weighted over what remains, rather than counted as zero. In the published
demo (late October 2026) most names have every category and none is below 80%; a score with low
coverage is a good read on what it covers and a poor read on the rest.

**"Never run", and "Q1 2026 filings".** The Dashboard's freshness strip, and the
Settings page, list each dataset and when it last updated; one that is behind is
marked. A dataset that has never been collected says *never run* — deliberate
honesty, not a bug: the app leaves it visibly empty rather than quietly scoring
it as zero, which would drag good companies down for no reason. Institutional
ownership is the exception to "how long ago": SEC publishes 13F filings once a
quarter, weeks after the quarter ends, so it is labelled by its quarter and by
whether that is still the newest SEC has published (the weekly run asks).

**Ratings are relative, not absolute.** "Strong Buy" means top 10% *of this
list*, whatever the market is doing. In a falling market the top 10% is still
labelled Strong Buy. The Screener has a switch for absolute ratings — fixed
bars instead of a ranking — and the two genuinely disagree for about half the
names.

**Forecasts come with their own report card.** Each horizon shows a hit rate,
the naive baseline's hit rate next to it, and how many independent windows it
was measured over. **When the app's model doesn't beat the naive baseline, it
shows you that** — and mostly it doesn't. There are no quarter- or year-ahead
forecasts: the stored history holds too few independent quarters and years to
grade them, and a forecast that can never be checked isn't published. If a model
is ever short of graded windows at a published horizon, its rows sit behind a
separate "ungraded" disclosure.

**Backtest numbers carry confidence intervals.** Sharpe 1.40 with a 90% interval
of [0.69, 2.35] is not the same claim as "Sharpe 1.40". If an interval crosses
zero, the result is not distinguishable from luck. The momentum strategy has so
far **trailed simply holding the whole universe equal-weight** — and the page
shows that rather than hiding it.

**Risk ratios go blank rather than lie.** Sharpe, Sortino and beta need about a
year of observations to mean anything. Below that the app shows a dash and
explains why, instead of printing an impressive number built from three weeks
of data.

**Rule of thumb:** trust the prices, the scores, the rankings, the patterns, and
anything with a stated sample size. Treat a composite score as partial when its
coverage is low. Treat any forecast without a hit rate as unproven. Treat the
backtest as one historical run, not a promise.

---

## Why some things are empty

Every weekday evening the refresh collects prices and recomputes the scores and
the market regime; Monday's run also collects fundamentals, analyst consensus,
news and sentiment, 13F filings, forecasts and the backtest. None of that needs
a key. Three optional datasets do:

- **Short interest** needs a Finnhub key and **macro rates** need a FRED key.
  Both have free tiers, but neither is set on the public demo, so those two stay
  empty there, and the market-regime index says it is scored on three of its
  four inputs.
- **The forward test** — paper-trading the published rating — needs Alpaca
  paper-trading keys, and its record can only start from the day they are added.

None of this stops the app working. It just narrows what it can see, and it
tells you so.

---

## If something looks wrong

- **The public link shows old data.** It republishes after each weekday
  evening's refresh. The Dashboard's freshness strip tells you the date it's
  actually showing; if prices are more than two trading days behind, a refresh
  has failed. Running the app yourself, ⚙️ Settings → Run a refresh updates
  your own copy.
- **The hosted app shows "This app has gone to sleep".** Press the button to
  wake it; the first page takes about a minute.
- **`./run.sh` says Python 3.12 is required.** Run the one-line install command
  it prints, then run `./run.sh` again.
- **A page says a section can't be computed.** That's usually the honest
  version — not enough history, or a dataset that hasn't run. The message says
  which.

---

*Educational and research tool. Not financial advice, not a registered
investment advisor. Past backtested performance does not guarantee future
results.*
