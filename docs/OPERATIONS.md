# QuantPulse — deployment and operations

How the two public deployments are built and kept current, the nightly data
pipeline, the secrets that switch optional features on, and the engineering
notes behind several features. The [README](../README.md) is the overview; this
is the detail.

- [The demo database is a release asset](#the-demo-database-is-a-release-asset)
- [GitHub Pages](#1-github-pages--the-public-link-fully-automated)
- [Streamlit Community Cloud](#2-streamlit-community-cloud--the-full-app-one-manual-step)
- [Data and secrets](#data-and-secrets)
- [Alerts](#alerts--the-refresh-telling-you-when-to-open-the-app)
- [Forward test](#forward-test--trading-the-ratings-not-re-reading-them)
- [Portfolio history](#portfolio-history--value-benchmark-dividends)
- [The SPA reaches parity with Streamlit](#the-spa-reaches-parity-with-streamlit)
- [Charting dependency](#charting-dependency-handled-deliberately)
- [Keyboard and screen-reader budget](#keyboard-and-screen-reader-budget)

## The demo database is a release asset

> **The demo database is a release asset, not a committed file.** It is ~66 MB,
> and forty committed revisions of it had reached **309 MB of a 317 MB
> repository** — 97.5%, against 8 MB for every source file, test and document
> put together — growing about 7.7 MB per refresh, five refreshes a week. It now
> lives on the rolling [`demo-data`](https://github.com/MarlenMM/quantpulse/releases/tag/demo-data)
> release tag, and its old revisions were removed from history on 2026-09-13 —
> a full clone went from ~317 MB to 13 MB. `./run.sh` downloads it once on first run, and so do CI, the
> Pages build and the Streamlit app; nothing needs an API key. A clone is small
> again, and the file is fetched once rather than every version of it ever made
> arriving with the clone.

## Deployments

There are two deployments, and they are deliberately different shapes.

### 1. GitHub Pages — the public link, fully automated

**<https://marlenmm.github.io/quantpulse/>**

The React SPA, served as static files, free on public repos, always on, nothing
to sign into. Pages cannot run Python, so `scripts/build_static_site.py`
pre-renders the read API instead: it runs the real FastAPI app through
Starlette's `TestClient` over the committed demo database and writes every
response the client can ask for (524 files, ~29 MB). **The published numbers are
the API's own output**, not a second implementation — the same discipline that
keeps the two front ends agreeing.

Every route is a real file, not a fallback: the build writes an `index.html`
per page and per stock, so `…/stocks/NVDA` answers **200** with its own
`<title>` rather than the 404-with-the-app-in-the-body that a single
`404.html` gives you. `404.html` is still there, for paths that genuinely are
not found.

`.github/workflows/pages.yml` builds and publishes it on every push to `main`,
and the refresh workflow calls it directly once it has committed fresh data.
(It has to be an explicit call: the refresh's commit carries `[skip ci]`, which
suppresses every workflow a push would otherwise start.)

Before publishing, the workflow loads the finished bundle in a real browser and
reads real numbers off it. That check is load-bearing rather than ceremonial:
the generator and the client agree about filenames only by convention, they are
in different languages, and a one-character disagreement would 404 every request
while the type check, the build and the unit tests all stayed green.

**What Pages cannot host:** the full Portfolio Manager — the optimisers, the
rebalancing trade list and the history panel need the engine's solvers or data
the demo does not publish. The demo's Portfolio page is the browser-local part
of it; for the rest run `./run.sh`, or deploy the Streamlit app below. A CSV
exported from the demo imports into it unchanged.

### 2. Streamlit Community Cloud — the full app, one manual step

**Deployed 2026-09-25 at <https://quantpulse-demo.streamlit.app>** with exactly
the settings below (Python 3.12, `app/Home.py`, the two secrets). Verified after
a reboot: a cold container whose first request was a deep link to the Screener
downloaded the database and rendered all 503 names, and the build installed
`app/requirements.txt` — 80 packages, no torch/transformers/spaCy. One thing the
host does on its own: it replaces `pyarrow` 25.0.0 with 24.0.0, citing a known
segfault, so the hosted app runs a pyarrow the test suite does not.

The seven-page Streamlit app, including the Portfolio Manager and the LLM
narration layer. Connecting a repo needs an interactive GitHub sign-in, so this
is the one step that cannot be scripted. The repo is already prepared for it:

1. Go to **<https://share.streamlit.io>** and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in: Repository `MarlenMM/quantpulse`, Branch `main`, Main file path
   `app/Home.py`. Under **Advanced settings**, set Python version **3.12**.
4. In the same **Advanced settings** panel, paste this into **Secrets**:
   ```toml
   DATABASE_URL = "sqlite:///./quantpulse_demo.db"
   PORTFOLIO_BACKEND = "session"
   ```
5. **Deploy**. First build takes a few minutes.
6. Open the app. The first page view on a fresh (or woken) container downloads
   the demo database, about 83 MB, behind a spinner; later views are instant.
   Any page can be the first one — a shared link straight to the Screener works.

`app/requirements.txt` is what that host installs — a copy of the root
`requirements.txt` beside the entrypoint, because the host takes a dependency
file there ahead of the root, where it would otherwise pick `uv.lock` and install
everything (the first deploy did, 2026-09-25). It deliberately omits torch,
transformers and spaCy — the refresh job's models, roughly 2.5 GB of wheels,
which no page imports and the free tier cannot fit. It is generated from
`uv.lock` by `scripts/sync_requirements.py`, and every page render in the test
suite asserts none of the three ends up in `sys.modules`.

`PORTFOLIO_BACKEND=session` is the part that matters (ADR 4.5): it keeps every
visitor's holdings in their own browser session rather than the shared committed
file. An LLM key (Section 4.3) is optional — the app runs fine without one.

Streamlit Community Cloud auto-redeploys on every push to `main`. The data
itself no longer arrives that way — the database is a release asset, and
`app/Home.py` downloads it on first run through `quantpulse.demo_data`, which is
in the package precisely because that host puts only `app/` on `sys.path` and
could not import anything under `scripts/`. A restarted container picks up the
current data without a redeploy.

### Data and secrets

`.github/workflows/refresh_data.yml` rebuilds the demo database
(`quantpulse_demo.db` — distinct from your own local `quantpulse.db`, see
`.gitignore`) and publishes it to the `demo-data` release, so neither deployment
needs API keys of its own (ADR 4.4).

**There are two database assets, and the split is deliberate.** `demo-data` is
rolling — replaced every refresh, read by the demo, the Pages build and
`./run.sh`, because those want the current data. [`ci-fixture`](https://github.com/MarlenMM/quantpulse/releases/tag/ci-fixture)
is pinned, verified on download against `demo_data.CI_FIXTURE_SHA256`, and read
only by the CI test job. CI used to read the rolling one, so a refresh that
published a gutted database on 2026-09-15 turned three unrelated frontend pull
requests red on a Streamlit test about effective weights — a test job whose
inputs change underneath it is not testing the diff. Rolling the fixture forward
is a deliberate commit: upload a newer database to that release and put its
digest in the same change. The current data is still gated by the publish
workflow's static-site suite, which runs before the demo updates.

The refresh runs
**weekday evenings at 22:00 UTC**, one to two hours after the New York close, and
can also be dispatched by hand (Actions → Data Refresh → Run workflow) to catch a
database up or to test a fix without waiting for the next run. Monday's run
carries the weekly branch, so the slow-moving datasets come round by themselves.

The schedule was off between 2026-08-27 and 2026-09-06, and the demo visibly
aged: its own freshness strip read "24 days ago" for sentiment. Worth recording
why, because the cause was not the timer. From 2026-08-10 the scheduled run went
red thirteen nights running, and every failure was the *Pages publish*, not the
refresh — a Playwright strict-mode violation in the site check, where
`getByText("AIZ")` matched six elements. The data kept updating and the site
stopped republishing. The selector is fixed; the timer is back.

The in-app refresh button stays off in hosted `session` mode
(`MANUAL_REFRESH_ENABLED`): a shared URL is no place to let a visitor start an
hours-long job on rate-limited quota, and that host omits the model stack the
refresh needs anyway.

That database is already seeded and published — `./run.sh` downloads it on first
run, which is why a fresh clone needs no setup. To rebuild it from scratch
(after a long gap, or to change the history depth):

```bash
DATABASE_URL=sqlite:///./quantpulse_demo.db uv run alembic upgrade head
DATABASE_URL=sqlite:///./quantpulse_demo.db uv run python scripts/seed_initial_data.py
gh release upload demo-data quantpulse_demo.db --clobber
```

The schedule then keeps it current, publishing an updated `quantpulse_demo.db`
to the `demo-data` release and republishing the Pages site against it. One
thing keeps the schedule itself alive: GitHub switches off a public repository's
scheduled workflows after 60 days without activity, and since the database is a
release asset the refresh commits nothing. So the nightly calls
`.github/workflows/keepalive.yml`, which commits a short `docs/refresh_status.md`
only when `main` has had no commit for 30 days. If the schedule is ever disabled
anyway, `gh workflow enable refresh_data.yml` turns it back on. The
refresh job fetches that asset before it migrates or writes anything: skipping
that step is how it once rebuilt an empty database from scratch and clobbered
three years of history with four weeks of it, so the upload is now gated by
`scripts/demo_db_guard.py`, which refuses to publish a database that came out of
a run with fewer rows than it went in with.

Fresher data needs **repo secrets** (Settings → Secrets and variables →
Actions). **Without them the job still runs, but some datasets stay
permanently empty**, and the app shows them as "never run" rather than
pretending otherwise. Worth knowing which cost what:

   | Secret | What it unlocks — and what stays empty without it | Where to get it |
   |---|---|---|
   | `FINNHUB_API_KEY` | Short interest (Section 24's two readings: % of float short, days to cover). **Caveat:** the field names read from Finnhub's `/stock/metric` response are an unverified guess (`ingestion/short_interest_client.py` says so) — no key has ever been available to check a real response, so the first run with a key may store rows with empty values. Unset, the weekly run logs one line and skips it (it used to log one warning per ticker, 503 a run — finding 37) | Free key at [finnhub.io/register](https://finnhub.io/register) |
   | `FRED_API_KEY` | Fed funds, CPI, unemployment, GDP, and the 10Y/2Y Treasury series — so the yield-curve spread, one of the Market Regime Index's four inputs. Unset, the weekly run logs one line naming the six series it skipped, and the regime is scored without that input (and says so) | Free with a FRED account: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |
   | `SEC_EDGAR_USER_AGENT` | Insider (Form 4) and 13F institutional ownership. This one is **not an API key** — SEC only asks for a contact string like `"Your Name your@email.com"`, so it costs nothing but a repo secret | Any contact string you choose ([SEC's fair-access policy](https://www.sec.gov/os/accessing-edgar-data)) |
   | `ALERT_DISCORD_WEBHOOK_URL` | Every alert the project sends: the nightly data digest (below), a notice when a refresh or Pages run **fails** (which job and step, with the run's link), a notice when a refresh finishes **partial** (its reasons, with the run's link), and a notice when the public demo's prices fall more than two trading sessions behind. Unset, each logs one line and sends nothing; a stale demo still turns the nightly run red, which GitHub can email you about if your notification settings allow | A channel's *Edit Channel → Integrations → Webhooks → New Webhook → Copy Webhook URL* in Discord |
   | `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` | The forward test (below): the Monday run paper-trades the published rating and every run records the account's value in `paper_trading_snapshots`. Unset, the refresh logs one line, nothing is traded, and the Track Record page says so. **The one with a clock on it** — the record can only accrue from the day the keys are added | [alpaca.markets](https://alpaca.markets) → switch to **Paper Trading** → generate an API key; both halves |

   Everything else — prices, options, news, fundamentals, analyst consensus,
   the index constituent list — comes from sources that need no credential at
   all, which is why the composite score still computes without any of the
   above (at a lower `data_confidence`, which every page displays).

   As of 2026-09-25 (`gh secret list`) only `SEC_EDGAR_USER_AGENT` is set, so
   short interest, the FRED macro series and the forward test are empty in the
   published demo (`paper_trading_snapshots` 0 rows, `short_interest` 0 rows, the
   regime's 10Y−2Y input null on every one of its 32 days), and no alert of any
   kind is sent. Nothing scores them as
   zero, and — since this was the point of the audit item — nothing hides them
   either: the Market Regime Index states that three of its four inputs are live
   and that the score is renormalized over those, and a stock with no
   short-interest reading says the source is absent rather than dropping the
   section, which would read as "this stock has no short interest".

### Alerts — the refresh telling you when to open the app

The refresh already computes every rating change and completed chart formation.
With a Discord webhook in `ALERT_DISCORD_WEBHOOK_URL`, it posts the ones worth
interrupting you about:

* **every** rating change on a name you hold or watch,
* a new chart formation on one of those names at confidence ≥ 70, completed
  within the last 30 days,
* and, from the rest of the universe, only a rating that **lands on** Strong Buy
  or Strong Sell.

The scoping is not squeamishness. Measured against the committed demo database,
"any rating change" is 96–195 a night — at ~45 characters a line that is roughly
twice Discord's 2,000-character message limit, so the naive version would fail
rather than annoy. The published rules produce one message of about 800
characters. A night on which nothing qualifies sends **nothing at all**; an
alert that arrives every night regardless stops being read.

To set it up: in Discord, *Edit Channel → Integrations → Webhooks → New Webhook
→ Copy Webhook URL*, then paste it into a repo secret (or `.env` locally).
Treat the URL as a password — anyone holding it can post into that channel; it
is never written to a log, and revoking it is one click in the same menu.

**The same webhook also carries notices about the pipeline itself.** A refresh
or Pages run that fails posts which job and step failed, with the run's link
(`scripts/pipeline_alert.py failure`, from an `if: failure()` job in each
workflow — one message per failure, not one per workflow). And before each
nightly refresh, a separate job reads the *published* `health.json`: if its
newest prices are more than two trading sessions behind — counted on the NYSE
calendar, so a Monday holiday is not an outage — it posts that and fails the
job. Two was chosen from the run history: it fires on the second night of every
real outage the demo has had, and on no isolated missed night.

Discord rather than Gmail SMTP because of the size of the secret: a webhook is
one opaque URL, where SMTP needs a host, a port, a from-address, a to-address
and an app password that authenticates against a whole mailbox.

### Forward test — trading the ratings, not re-reading them

Every backtest invites "you fitted that", and the one on the Track Record page
has a sharper version of the problem: it ranks the **momentum category**, not
the published Buy/Sell rating, because five of the composite's seven categories
hold weeks of stored history rather than years. The rating the whole app
publishes has never been testable against the past.

So the refresh paper-trades it forward. With Alpaca credentials set, the Monday
run buys the top 20 names rated Buy or Strong Buy, equal-weight in whole shares,
and every run records what the account was worth. Simulated money, real market
data, real fills — and every position chosen before its outcome was known, which
is the one thing a backtest can never claim.

Two things worth knowing about how it is built:

* **The endpoint is not configurable.** There is no setting, no environment
  variable and no argument that can point it at `api.alpaca.markets`. The paper
  host is a module constant, re-validated on every request, and the only other
  host it will talk to is loopback. An env var carrying `api.` where
  `paper-api.` was meant is one keystroke, raises nothing, and would trade real
  money on a schedule with nobody watching.
* **Nothing is annualised.** A record that starts one day long and grows by one
  a day would spend months turning a good week into a headline CAGR — the exact
  trap `backtest.MIN_TRACK_RECORD_PERIODS` exists for, where two monthly periods
  spanning 35 days were once published as 26.6%. The page shows a total return
  over a stated window, and says "8 of 20 trading days" until the record is long
  enough to be called one.

To set it up: sign up at [alpaca.markets](https://alpaca.markets), switch to
Paper Trading, generate an API key, and put both halves in repo secrets. Unset,
the step logs one line and the Track Record page explains what would fill it.

### Portfolio history — value, benchmark, dividends

The Portfolio Manager knows tax lots, three optimisers, correlation clusters,
VaR and sector gaps, and until now knew all of it only as of *now*. A History
section adds the time axis, built entirely from the transaction ledger:
what the holdings have been worth, how they have done against the S&P 500, and
what they have paid in dividends.

The benchmark comparison is a **time-weighted return**, which is the whole
point rather than a detail. Laying portfolio value beside an index level is the
obvious approach and it is wrong: buying £10,000 of stock raises the value by
£10,000 and has earned nothing. On the example portfolio, adding one purchase
mid-window makes a naive value ratio report **+171.9%** where the time-weighted
return is **+33.5%** — and flips the verdict from "beat the index" to "lagged
it". Chaining each period's return with that period's cash flow removed leaves
only what the holdings did.

Dividends are counted on the shares held on each **ex-date**, not the shares
held now — buying the day after an ex-date earns nothing from it. Cash is
excluded throughout, and the page says so: `PortfolioState` stores one current
balance rather than a history of deposits, so there is no honest way to say what
was uninvested on a past date.

### The SPA reaches parity with Streamlit

Three things the Streamlit app had and the public demo — the front end most
visitors actually see — did not:

- **CSV export** on the Screener and the Track Record. The screener export is
  the rows on screen, with your weights, filters and rating scheme applied, not
  the stored balanced ranking. The track-record export carries `signal_name` and
  `assumed_txn_cost`, which the table has no room for and without which a row is
  not interpretable.
- **Compare mode** — 2–4 names' sub-scores side by side. Nothing is fetched; the
  sub-scores are already in the screener payload and are weight-independent by
  design. A category with no data shows an em dash, never a zero.
- **A watchlist**, as a star on each row and on every stock page, plus a
  "watchlist only" filter.

The watchlist lives in `localStorage`, and every surface says so. It cannot be
anything else: the API is read-only by design (ADR 4.5) and the site is static
files on GitHub Pages, so there is nowhere on a server to put it. It is
per-browser, not synced to another device, and cleared with site data — a list
that silently failed to follow you would be worse than one you knew was local.

The SPA's Portfolio page is local in exactly the same way, and says so. Its
figures come from `frontend/src/lib/portfolio.ts`, a port of the engine's FIFO,
recommendation and risk rules; both implementations are pinned to one golden
file (`frontend/tests/fixtures/portfolio-golden.json`), regenerated from the
engine with `uv run python tests/unit/test_portfolio_golden.py --write`, so
the two cannot drift apart silently. It measures risk over the ~13 months of
prices the demo publishes, where the full app uses 420 days.

### Charting dependency, handled deliberately

Plotly has silently blanked every chart in this app twice — once on Vite 7→8's
CJS interop, once on `react-plotly.js` 2→4 shipping a `forwardRef` object — and
both times TypeScript, the build and CI stayed green throughout. So the standing
rule is that any charting or bundler upgrade is verified by loading a chart page
in a real browser and reading the console, and `components/Chart.tsx` resolves
the component by asking what a React element type looks like rather than by
pattern-matching one version's packaging.

Taking plotly to 4.1.0 also retired `plotly.js-dist-min`, which had been a
direct dependency without ever being used: `react-plotly.js` resolves its peer
`plotly.js`, and removing the dist package left the largest built chunk
byte-identical. `plotly.js` is now an explicit dependency, which is what the
code imports its types from anyway.

### Keyboard and screen-reader budget

The Screener ranks 503 names, and rendering all of them put **1,535 focusable
elements and 7,695 DOM nodes** on one page — 1,509 of those controls inside the
table body, so a keyboard user leaving it had that many stops to get past, with
no skip link to avoid the header either.

It now renders 50 rows a page: **178 focusable elements, 962 DOM nodes**. The
rank column stays absolute, so paginating a *ranked* table does not lose the one
thing the ordering means, and the CSV still exports every row the filters match
rather than the page on screen.

Paginated rather than virtualised, deliberately. Virtualising keeps the feel of
one long list, but it hides rows from the browser's own find-in-page and
misreports the table's size to a screen reader unless `aria-rowcount` is handled
carefully — so the usual fix for an accessibility problem would have traded one
for two.
