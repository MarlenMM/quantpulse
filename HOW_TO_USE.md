# How to use QuantPulse

Plain-English guide. No setup, no keys, no payment — none of this costs anything.

---

## Two ways in

**1. The public link — nothing to install**

**https://marlenmm.github.io/quantpulse/**

Open it on any device. It shows the research side of the app: rankings, per-stock
detail, charts, forecasts, the track record, the glossary. It refreshes itself
every weeknight.

It is **read-only**. It cannot hold a portfolio, because it is a set of files on
a free host with nothing running behind it — there is no server to save anything
to, and nothing to log in to.

**2. On your own machine — the whole app**

```bash
git clone https://github.com/MarlenMM/quantpulse.git
cd quantpulse
./run.sh
```

Then open http://localhost:8501. That is the full seven-page app, including the
**Portfolio Manager**, which the link cannot do.

The first run takes about a minute while it installs what it needs. After that
it starts in seconds. It needs Python 3.12 — if you don't have it, the script
tells you the one line to run.

Everything it shows comes from a database committed inside the repository. No
account, no API key, no bill.

---

## What each page is for

The public link has the first four. Your local copy has all seven.

| Page | What it's for | What you can do |
|---|---|---|
| **Dashboard** | The market at a glance | See today's top-ranked names, whether the market is risk-on or risk-off, which sectors money moved into this month, and which ratings changed since yesterday |
| **Screener** | The ranked list of all 503 companies | Sort and filter the whole S&P 500 by score, rating or sector; search by name; switch investor profile (below); compare up to four names side by side; download it as a spreadsheet |
| **Stock Detail** | One company, in depth | Price chart, the seven category scores behind its rating, detected chart patterns, forecasts at four horizons with their track record, risk numbers, short interest |
| **Backtest / Track Record** | Did the ranking actually work? | The historical result of following the ratings, with confidence intervals and a benchmark to compare against |
| **Portfolio Manager** *(local only)* | Your own holdings | Enter what you own, see risk, correlation, sector concentration, add/trim/hold/sell suggestions, and three target allocations with a concrete trade list |
| **Settings** *(local only)* | What's switched on | Which data sources are configured and when each last ran |
| **Glossary** | Every term explained | 71 definitions in plain English; searchable by concept, not just by name |

**Investor profiles** (on the Screener) re-rank the same companies for a
different priority: balanced, value, growth, income, momentum, conservative.
Same data, different weighting — not a different opinion.

**Portfolio Manager tip:** there's a "Load example portfolio" button. Click it
to see everything the page can do before typing in anything of your own.
Holdings live in your browser session and vanish on refresh — use "Download CSV"
to keep them, and "Upload" to bring them back.

---

## Which numbers to trust, and which are thin

This is the part worth reading. **The app is honest about its own limits, and it
says so on screen — you just need to know where to look.**

**Coverage.** Every stock shows a coverage figure, e.g. *"thin coverage (45%)"*.
The composite score blends seven categories; coverage is how many of them
actually had data. Right now the demo runs on the price-based ones — technicals,
momentum, smart money — because the categories that need company filings and
news haven't been collected yet (more below). **45% coverage means the score is
real but partial: it is a good read on price behaviour and a poor read on
whether the business is any good.**

**"Never run".** The Dashboard's freshness strip, and the Settings page, list
each dataset and when it last updated. Fundamentals, Analyst Consensus and
Sentiment currently say *never run*. That is deliberate honesty, not a bug — the
app leaves them visibly empty rather than quietly scoring them as zero, which
would drag good companies down for no reason.

**Ratings are relative, not absolute.** "Strong Buy" means top 10% *of this
list*, whatever the market is doing. In a falling market the top 10% is still
labelled Strong Buy. The Screener has a switch for absolute ratings — fixed
bars instead of a ranking — and the two genuinely disagree for about half the
names.

**Forecasts come with their own report card.** Each horizon shows a hit rate,
the naive baseline's hit rate next to it, and how many independent windows it
was measured over. **When the app's model doesn't beat the naive baseline, it
shows you that** — and mostly it doesn't. The 63-day and 252-day rows show "—"
for hit rate because there wasn't enough history to grade them honestly.
An ungraded forecast is a guess, and the app says so.

**Backtest numbers carry confidence intervals.** Sharpe 1.40 with a 90% interval
of [0.69, 2.35] is not the same claim as "Sharpe 1.40". If an interval crosses
zero, the result is not distinguishable from luck. The current run also **loses
to simply buying and holding the whole index** — that's shown rather than hidden.

**Risk ratios go blank rather than lie.** Sharpe, Sortino and beta need about a
year of observations to mean anything. Below that the app shows a dash and
explains why, instead of printing an impressive number built from three weeks
of data.

**Rule of thumb:** trust the prices, the technical and momentum scores, the
rankings, the patterns, and anything with a stated sample size. Treat a
composite score as partial while coverage is under ~60%. Treat any forecast
without a hit rate as unproven. Treat the backtest as one historical run, not a
promise.

---

## Why some things are empty

The nightly job collects everything it can from free sources. Four datasets need
credentials the project deliberately doesn't buy:

- **Fundamentals, analyst consensus, news sentiment** — these come from free
  sources but are collected on the *weekly* branch of the nightly job, which
  runs on Mondays. They should populate on the next Monday run.
- **Short interest** needs a Finnhub key; **macro rates** need a FRED key.
  Both have free tiers, but neither is set, so those two stay empty forever
  unless you add them. Nothing asks you to.

None of this stops the app working. It just narrows what it can see, and it
tells you so.

---

## If something looks wrong

- **The public link shows old data.** It republishes after each weeknight
  refresh. A weekend or a holiday means no new data — the Dashboard's freshness
  strip tells you the date it's actually showing.
- **`./run.sh` says Python 3.12 is required.** Run the one-line install command
  it prints, then run `./run.sh` again.
- **A page says a section can't be computed.** That's usually the honest
  version — not enough history, or a dataset that hasn't run. The message says
  which.

---

*Educational and research tool. Not financial advice, not a registered
investment advisor. Past backtested performance does not guarantee future
results.*
