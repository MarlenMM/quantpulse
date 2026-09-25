# QuantPulse — improvement backlog and session handoff

An 18-point audit was run on **2026-09-03** against the live demo, the committed
demo database, and the pipeline's own upstream sources. **All eighteen points are fixed and live.**. This file is the handoff: what was done, what is
left, and the things a new session would otherwise rediscover the hard way.

**State at time of writing:** 1,824 tests, 14 Alembic migrations, CI and Pages
green, working tree clean.

---

## 1. What this project is, in one screen

QuantPulse is a solo, zero-cost stock research and portfolio-management engine.
Statistics and ML do the ranking and forecasting; a free-tier LLM only narrates
numbers that already exist. Phases 0–12 of `PROJECT_PLAN.md` are complete.

| | |
|---|---|
| Engine | `src/quantpulse/` — ingestion, analysis, storage, read-only API |
| Front ends | Streamlit (`app/`, 7 pages, the full app) and React+TS (`frontend/`, 5 pages, the public demo) |
| Database | SQLite, 23 tables, Alembic-migrated. `quantpulse_demo.db` is **committed** |
| Public demo | <https://marlenmm.github.io/quantpulse/> — pre-rendered API JSON + the SPA |
| Refresh | `.github/workflows/refresh_data.yml`, weekdays 22:00 UTC; Monday carries the weekly branch |

**The rule everything serves:** an honestly-labelled limitation beats a silently
inflated number (Section 22). Most of the nine fixes below were not broken
maths — they were the app describing itself more confidently than its data
supported.

### Commands

```bash
uv run pytest                 # 1,824 tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src scripts       # pre-commit checks scripts/ too, not just src/
cd frontend && npm run test:e2e     # Playwright, stubbed API from a fixture
cd frontend && npm run test:static  # Playwright, the real pre-rendered dist/
./run.sh                      # the whole app locally, no keys
```

---

## 2. Traps that have cost real time

Read this before debugging anything. Each was learned by losing an hour to it.

- **`vite preview` cannot verify HTTP status codes.** It has its own SPA
  fallback and returns 200 for everything, including `/totally-made-up`. To
  model GitHub Pages, serve `dist/` with `python3 -m http.server` — directory
  indexes, 301s and real 404s.
- **`npm run test:e2e` leaves `dist/` in the wrong state for `test:static`.**
  `playwright.config.ts` rebuilds with `npm run build` and no `VITE_STATIC_API`,
  so running the stubbed suite then the static one hands the second an API-mode
  bundle: all fourteen content tests fail with "element not found" and none says
  why. `tests-static/assert-static-build.ts` now refuses to run and says so.
  `reuseExistingServer` makes a stale dev server on :4177 do the same thing.
- **There are two Playwright suites and they test different things.**
  `test:e2e` stubs the API from `frontend/tests/fixtures/stock-AIZ.json`;
  `test:static` runs against the real pre-rendered `dist/`. Running one is not
  running the other. CI runs both.
- **Recapture API fixtures from the real API, never hand-patch them.** A fixture
  edited until the client is happy has stopped being evidence about the server.
- **A mutation must compile before its result means anything.** A React mutation
  once "passed" because it left a variable unused, `tsc` failed, and Playwright
  ran against a stale `dist/`.
- **Assert the caller, not just the helper.** Unit tests for a resolver pass
  whether or not the pipeline calls it. Two separate fixes shipped with a
  mutation that changed nothing detectable until a caller-level test was added.
- **Rendering several Streamlit pages in one process SIGSEGVs** inside libarrow.
  One page per subprocess (`tests/support/render_one_page.py`).
- **Streamlit's dataframe paints to a canvas.** `[role="gridcell"]` innerText
  carries raw underlying values, not what the user sees. To assert on a Styler,
  patch `streamlit.dataframe` and capture the argument.
- **Duplicated constants across `app/`, `api/` and `analysis/` are the recurring
  root cause of the two front ends disagreeing.** Collapse them into one
  exported value; three separate copies of a market window once produced three
  different betas.
- **A guard shadowed by an earlier guard is an untested guard.** Three tests
  written for point 10 passed with the thing they were named for deleted: a
  `KeyError` came from a rank lookup two lines below the label lookup being
  tested, a "these scores are not today's" check sat behind a "there are not
  two snapshots" check that fired first, and an unlimited read was asserted
  against a one-row fixture where a cap of 25 is invisible. Mutating found all
  three; nothing else would have.
- **Render the real database before believing the fixtures.** The alert's
  formation lines looked right in every test and read "UNP new double bottom"
  three times against real rows, and a 403 from a real socket reported
  `HTTPError` with no status code where the mocked test had asserted a status.
  Both were only visible by running it against `quantpulse_demo.db`.
- **A test that reads the real weekday is not a test.**
  `test_run_end_to_end_with_tiny_mocked_universe` derived `is_weekly` from
  `date.today()`, so it exercised the daily path Tue–Sun and the weekly path on
  Mondays — where a mocked-empty step correctly downgrades the run and the
  assertion failed for an unrelated reason. It went red on its own on
  2026-09-07, the first Monday after it was written. Pin the clock.
- **`frontend/dist/data/` is Vite's *copy* of `frontend/public/data/`.**
  `scripts/build_static_site.py` writes to `public/`; only `npm run build`
  refreshes `dist/`. Reading `dist/` after re-running just the Python build
  gives a stale file, and doing that produced a completely wrong diagnosis
  (a schema field looked "silently dropped from the published payload" when it
  had always been present). Check `public/data/`, or rebuild both.
- **A variance-share "effective weights" chart is a trap.** Every sub-score is a
  percentile, so every category has identical dispersion and the shares come
  back equal to the stated weights. Rank correlation with the composite is the
  measure that actually differs.
- **Don't assert on a word that appears elsewhere on the same page.** A test for
  the zero-coverage warning passed with the whole sentence rewritten, because
  "renormalized" also appears in the Methodology section below it.
- **A step that accumulates and persists once loses everything to a timeout.**
  `tier2_news` spent thirty minutes fetching and scoring, was killed on its
  budget, and stored zero rows. Take a deadline, stop before the alarm, and keep
  what you have — `tier1_news` already did.
- **Rebuild `dist/` between Playwright mutation runs.** A restored source file
  with a stale build is the trap from §2 in a new costume: the suite failed with
  "element not found" for a mutation that had already been reverted.
- **US market holidays cluster on Mondays**, so anything pinned to a weekday
  silently skips roughly one week in ten. MLK, Presidents' Day, Memorial Day and
  Labor Day are each "the nth Monday of" a month; four of 2026's Mondays are
  closed. Ask the calendar ("the week's first trading day"), never the weekday.
- **A mutation harness that hides build failures proves nothing.** Redirecting
  `npm run build` to /dev/null meant an invalid mutation left the previous
  `dist/` in place and the suite passed against unmutated code — two mutations
  "survived" that way before the harness was made to fail loudly.
- **Adjust state during render, not in an effect, when a filter resets a page.**
  An effect runs after the commit, so there is one painted frame where the new
  result set is sliced by the old page index. The clamp that hid it was
  untestable and survived every mutation.
- **A `-dist-min` package can be dead weight.** `plotly.js-dist-min` was a direct
  dependency for months while `react-plotly.js` resolved its peer `plotly.js`
  instead. Removing it left the largest built chunk byte-identical — that
  comparison is how to tell, not reading imports.
- **`[skip ci]` anywhere in a commit message skips every push-triggered
  workflow** — including when the message is only *describing* a comment that
  mentions it. On 2026-09-25 two consecutive pushes (`9f79591`, `c0f00dc`)
  created no CI or Pages run, with GitHub reporting every component
  operational, because their messages quoted the stale `[skip ci]` comment they
  were removing. The next push, without the string, ran normally. `ci.yml` has
  no `workflow_dispatch`, so the only recovery is another push. Grep the message
  before pushing; "a push produced zero runs" has a cause worth checking first.
- **`gh run list` first.** A workflow that was cancelled or never created writes
  nothing the app can see, and that has been the single biggest bug twice.

---

## 3. Points 1–9 — fixed and live

Recorded because several of them explain why the code looks the way it does now.

| # | Was | Fix | Commit |
|---|---|---|---|
| 1 | Beta regressed against an equal-weight proxy; NVDA read **0.78 (R² 0.06)** | Ingest `^GSPC`; one resolver (`risk.resolve_market_returns`) shared by all three surfaces. NVDA now **1.88 (R² 0.44)** | `29edf62` |
| 2 | 13F had **never ingested a row** — it asked for the in-progress quarter, which SEC does not publish | Ask which window exists (`latest_published_window`). 489/503 names landed. A second bug (blank `NAMEOFISSUER` → float NaN) was hiding behind the first | `2e75b13` |
| 3 | `industry_macro` weighed 10% and covered **24/503 names** | Sectors became first-class baskets with their own GDELT triggers. Membership alone was a half-fix that measured as no fix | `3c2632c` |
| 4 | Track Record claimed to backtest "the algorithm's ratings"; it ranked a hand-rolled trailing return | Signal is now `scoring.score_momentum`; run records `signal_name`; Kelly sizes on **excess over benchmark** (0.0 = "do not take this bet") | `72f47d6` |
| 5 | Ungraded forecasts sat beside graded ones — and carried the biggest numbers (252d max **+231%**) | Graded is the default table; ungraded behind a disclosure. `forecasting.is_graded` is the one rule | `5f3b2e5`, `7018be2` |
| 6 | Demo announced its own staleness ("SENTIMENT 24 days ago") | Schedule restored. **The timer was never the problem** — 13 nights of red were a Playwright strict-mode bug in the *publish* job | `c194cb6` |
| 7 | Whole document scrolled sideways at 375px (Dashboard 619px) | `minmax(0, 1fr)` in the base `.split`/`.split-even` rules | `63045a3` |
| 8 | Regime said "built from four inputs" while one had always been null; short interest vanished silently | Server-composed coverage note, per row; short interest explains its absence | `d96cb02` |
| 9 | Every deep link answered **HTTP 404** with the app in the body | A real `index.html` per route (508 files), each stock page with its own title | `f12dab1` |
| 10 | No alerting of any kind, while the refresh already computed every input for it | Discord webhook from the refresh job. Triggers chosen by counting: everything on a held/watchlisted name, a new formation there at confidence ≥ 70 completed within 30 days, and universe-wide only a rating *landing on* Strong Buy/Sell. "Any change" measured 96–195 a night — about twice Discord's 2,000-char limit | `c94470e`, `e15c808`, `077e6b4` |
| 11 | The Track Record page ranked the momentum category, never the published rating, and could not — 22 days of stored composite history against a 1,183-day window | Alpaca paper trading, forward. Top 20 Buy/Strong Buy, equal-weight, whole shares, weekly rebalance, daily equity snapshot. The endpoint is a module constant with no setting that can reach live money; nothing is annualised | `66a01c9`, `4d697e4`, `840feca` |
| 12 | The radar showed seven sub-scores and never said which moved the rating | An exact decomposition: `composite - 50 = Σ (w/A)(s-50)`, verified to sum for all 503 names. One sentence built server-side, naming the drivers **and** the largest opposing category (that clause fires for 75% of names) | `e5fb095` + this |
| 13 | The Portfolio Manager was entirely as-of-now | A History section: value over time, a **time-weighted** comparison against the S&P 500, and dividend income counted on the shares held at each ex-date. On the example portfolio a naive value ratio reports **+171.9%** where the time-weighted return is **+33.5%** — and inverts the verdict against the index | `ba157ae`, `2cef83a` + this |
| 14 | The SPA lacked CSV export, Compare mode and the watchlist | All three, client-side over data the screener payload already carries. The watchlist is `localStorage` — the API is read-only by design, so it is per-browser and says so. A parity test names which Streamlit feature each mirrors | this |
| 15 | Four of seven categories barely moved the ranking, and nothing said so | An **Effective weights** panel in Settings: stated weight beside realised influence (rank correlation with the composite), plus the pair that causes it. Technical is 2nd by weight and **1st** by influence; fundamental is 1st by weight and 3rd. The obvious measure — variance share — would have recovered the weights exactly and shown nothing | this |
| 16 | Three open Dependabot PRs, one the twice-breaking charting library | Routine three taken together (one CI run, no rebase churn). `plotly.js-dist-min` turned out to be **unused** — removing it left the largest chunk byte-identical — so it is gone and `plotly.js` is explicit at **4.1.0**, verified in a real browser across all four trace types | `ae8154e`, `a0edfc0` |
| 17 | 503 rows, no skip link — and point 14's star + checkbox had tripled the focusable count to **1,535** | Skip link that actually moves focus, and the table paginated at 50 with the rank kept absolute. **1,535 → 178 focusable, 7,695 → 962 DOM nodes.** Paginated rather than virtualised: virtualising hides rows from find-in-page and misreports table size to a screen reader, trading one accessibility problem for two | this |
| 18 | 40 revisions of a 66 MB binary were **309 MB of a 317 MB repo** (97.5%), growing ~7.7 MB per refresh | Moved to the rolling `demo-data` release asset, fetched by `run.sh`, both workflows and the Streamlit app. Repo stops growing; **the 309 MB already in history is not reclaimed** — that needs a force-push, see below | this |

### Two things from those fixes that still need watching

1. **Point 3 is code-complete but not yet visible.** `industry_macro` still
   covers **19/503** because the sector-tagged Tier-2 news has not been fetched
   yet — GDELT rate-limited the dev machine, and the fix landed after the last
   weekly run. **Monday's scheduled run is the test.** Check on Tuesday:
   `select count(distinct matched_theme) from news_events where tier=2` should
   be ~17, not 5, and `industry_macro_raw` should be non-null for ~all 503.

   **Re-checked 2026-09-13: STILL OPEN, third distinct reason, and it briefly
   got worse.** The cadence fix worked — the 2026-09-08 run took 3h34m, the
   weekly-branch signature. But `tier2_news` then exceeded its 1800s budget and
   was killed, and because it accumulated every article and persisted only after
   the loop, the timeout discarded all of it: tier-2 themes stayed at 5 and
   `industry_macro` fell from 19/503 to **0/503**. The step's budget had been
   set from a cost model that counted model inference and ignored the network —
   17 serial queries against a rate-limited free API is what spends the half
   hour. Fixed in `032a5ce` (deadline + keep what you fetched + rotate baskets
   by staleness, budget 50 min). **The next weekly run is the test**; a slow
   week now degrades to partial coverage rather than none.

   **Re-checked 2026-09-08 (Tuesday): open, and the reason was new.**
   Monday 2026-09-07 was **Labor Day**. The scheduled run resolved the trading
   day in exchange time, found the market shut and logged
   `skipped_non_trading_day` — so the weekly branch did not run at all. That is
   now fixed (`05688b6`): the weekly branch runs on the week's first *trading*
   day, so Tuesday carries it when Monday is closed. **Tonight's 22:00 UTC run
   is the test**, and it is the first one that can be.

   **Re-checked 2026-09-06 (Sunday): still open, and not for a new reason.**
   `matched_theme` is still the same 5 curated themes and `industry_macro_raw`
   is still 19/503 on the 2026-09-05 snapshot. The manual dispatch on
   2026-09-05 does not count as the test — it ran 17m29s without
   `force_weekly`, and news/sentiment are on the weekly branch, so it never
   fetched a Tier-2 article. The 11 GICS sector baskets *are* in
   `thematic_baskets` (503 names covered), so the missing half really is only
   the fetch. **Monday 2026-09-07 22:00 UTC is still the test.**
2. **Three repo secrets are still unset, and they are the user's to add** —
   never offer to obtain or enter a key or a webhook URL. `FRED_API_KEY`
   (fred.stlouisfed.org) fills the yield curve and macro series;
   `FINNHUB_API_KEY` (finnhub.io) fills short interest;
   `ALERT_DISCORD_WEBHOOK_URL` (point 10) is where the nightly alert goes; and
   `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` (point 11) are the paper
   account the forward test trades in. Until each is set the matching step logs
   one line and does nothing, by design. All go in Settings → Secrets and
   variables → Actions.

---

## 4. Points 10–18 — the remaining backlog

Ordered as the audit ranked them. 18 is
methodology and hygiene. The numbering is the audit's and is kept stable, so
14 stays 14 now that 10-13 are done.

---

## 5. The history rewrite, done 2026-09-13

Point 18 stopped the growth. This reclaimed what was already there.

```bash
uvx git-filter-repo --invert-paths --path quantpulse_demo.db --force
git push --force origin main
git push --force origin refs/tags/demo-data
```

| | before | after |
|---|---|---|
| fresh full clone | ~317 MB | **13 MB** |
| local `.git` | 321 MB | 29 MB |
| commits | 207 | 175 |

**The 32 dropped commits were exactly the pure data-refresh ones** — 24 "Nightly
data refresh", 6 "Data refresh", 2 reseeds — whose entire content was the file
that no longer exists, so `filter-repo` pruned them as empty. Every commit that
touched code survived: the non-data subject lists match exactly, 175 to 175, and
every tracked file at HEAD has an identical content hash to before. The dates
those refreshes ran are still in `refresh_log` and in the Actions history.

Three things worth knowing if this is ever done again:

- **The tag had to move too.** `demo-data` pointed at a pre-rewrite commit, and
  leaving it would have kept all 309 MB reachable through it — the rewrite would
  have reclaimed nothing. The GitHub Release survived the tag being force-moved,
  because a release references the tag by *name*: asset still attached, still
  downloadable, still byte-identical.
- **GitHub's reported repository size did not change** (84 MB before and after).
  The old objects are unreachable but not yet garbage-collected; that is on
  GitHub's schedule and can be hurried with a support request. What a reader
  actually pays — the clone — dropped immediately.
- **`filter-repo` removes the `origin` remote** on purpose, so it has to be
  re-added before pushing.

Safe here because there were **0 forks and 0 open PRs**; a full mirror backup
was taken first. With either of those non-zero this is a different decision.

## 6. How these fixes have been done

Nine for nine, the pattern that worked:

1. **Reproduce and measure first.** Every fix above started with a number —
   619px, 24/503, 13 red nights, +231%. A fix whose "before" was never measured
   cannot be shown to have worked.
2. **Look at the running thing.** Every bug that mattered was invisible to lint,
   types, unit tests and the build.
3. **Write the test before the fix and watch it fail with the real numbers.**
4. **Mutation-check every new guard.** Revert the fix, confirm the test fails,
   restore. Three separate mutations passed on the first attempt and exposed a
   vacuous test each time.
5. **Verify on the live site after the deploy**, not just in CI.
6. **Say what was not done.** Point 8's two API keys, point 4's impossible
   composite backtest, point 3's pending Monday run — all stated rather than
   quietly dropped.

Commit and push each scoped unit of work automatically; no confirmation needed.

---

## 7. Point 19 — the refresh published an empty database over the real one

Found 2026-09-16 in a second audit, fixed the same day. **This is the failure
mode point 18 created**, and it is worth reading before touching either
workflow again.

Point 18 moved `quantpulse_demo.db` out of git and onto the rolling
`demo-data` release. `ci.yml` and `pages.yml` got a fetch step. The refresh
job — the one that *republishes* the asset — did not. So on 2026-09-15
`alembic upgrade head` created an empty database, the run filled it with about
four weeks of data fetched from scratch, and `gh release upload --clobber`
wrote that over three years of history.

| | published after the run | what it replaced |
|---|---|---|
| `price_history` | 9,577 | 417,993 |
| `forecasts` | 0 | 33,320 |
| `composite_scores` | 1,509 | 39,234 |
| `backtest_results` | 0 | 7 |

**Four things let it through, and each is now closed:**

1. **No fetch step.** Added, after `uv sync` (the fetch imports `quantpulse`)
   and before `alembic upgrade head` (which is what silently creates the empty
   file).
2. **`pragma integrity_check` was the only guard on the upload.** A small
   database is still a well-formed one. `scripts/demo_db_guard.py` now takes a
   row census at fetch time and refuses the upload if any table came out with
   fewer rows than it went in with — the pipeline's own append-only invariant
   (Section 6.8), which is a sharper test than any threshold on file size. A
   missing baseline is refused too, because that means the fetch never ran.
3. **The ordering test named two workflows literally**, so the only workflow
   that can destroy the asset was the one workflow nothing checked. It now
   reads the directory and applies to anything that runs pytest, the static
   build, the refresh script, or `gh release upload`.
4. **The tag kept no history.** The publish step now uploads the pre-run copy
   as `quantpulse_demo.previous.db` before clobbering, so one bad upload is
   recoverable from the release alone. Recovery this time depended on a copy
   that happened to be on a laptop.

**Recovery was a merge, not a restore.** Monday's run had ingested real data
the laptop copy lacked — news through 09-15, sentiment through 09-14 (the local
copy stopped at 08-10), prices, insider and options through 09-14. The
ingested tables were merged in with `INSERT OR IGNORE` and the **derived**
tables were left alone, because that run computed them against four weeks of
history: composite scores, patterns, forecasts and the backtest all came from
the good copy. `insider_transactions` and `index_membership_history` have
surrogate integer primary keys, so `SELECT *` would have collided on `id` and
dropped real rows — the first was merged by explicit column list, the second
skipped entirely (no unique key to dedupe on, and the local copy was already
the superset).

**It happened again the next night, before the fix landed.** The 2026-09-16
00:02 UTC scheduled run rebuilt from empty a second time and published a
**1.5 MB** database over the 11.7 MB one — smaller again, because a weekday run
carries only the daily branch. The site did not rebuild on it: the publish
job's fetch refused the download with `downloaded only 1507328 bytes … that is
not the database`. **`demo_data.MIN_BYTES` is the consumer-side half of this
guard and it worked**; what was missing was anything on the producer side, which
is what `demo_db_guard.py` now supplies.

**The first fix did not run at all, and this is the trap to remember.** The
job-level `env:` it added used `${{ runner.temp }}`. The `runner` context exists
only inside a step, and GitHub answers a job-level `env` that reads one by
refusing to parse **the whole file**: the run is created, fails in zero seconds
with "this run likely failed because of a workflow file issue", and is listed
under `.github/workflows/refresh_data.yml` — the path — because an unparseable
workflow has no name to show. `yaml.safe_load` reads that file happily, so
`test_every_workflow_is_parseable_yaml` was green throughout and the only
feedback was a red run after a push. `test_no_job_level_env_uses_a_step_only_context`
now rejects `runner.`, `steps.`, `job.` and `env.` in any job-level `env` value;
mutation-checked by putting the expression back, which fails it with the job and
key named while the YAML-parse test still passes.

### Point 21 — CI read a file that changed nightly

Fixed 2026-09-16, in the same session. The symptom was the three Dependabot
pull requests going red on
`test_settings_reports_effective_weights_against_the_real_ranking`: a Streamlit
test about the ranking, failing because of a database none of those pull
requests had touched. **A test job whose inputs change underneath it is not
testing the diff**, and nothing on screen said which of the two it was.

There are two release assets now, and the split is the fix:

| asset | who reads it | changes |
|---|---|---|
| `demo-data` | the public demo, the Pages build, `./run.sh` | every refresh |
| `ci-fixture` | the CI test job | only when someone rolls it forward |

`ci-fixture` is verified against `demo_data.CI_FIXTURE_SHA256` on download, so
a bad night cannot reach a pull request, and a fixture replaced without its
digest being updated fails *at the fetch step*, naming the database rather than
a test.

**To roll the fixture forward** (worth doing when the data shapes the tests care
about have genuinely moved): upload a newer database to the `ci-fixture`
release, then put its `shasum -a 256` in `demo_data.CI_FIXTURE_SHA256` in the
same commit. Both halves in one commit is the point — the change to what CI
tests against becomes reviewable, with a reason attached.

CI also migrates the fixture to head before running pytest. A pinned snapshot is
a schema snapshot, so a migration landing in the pull request under test would
otherwise run against a database that predates it — something the rolling asset
never exposed, because the refresh migrates it nightly.

**What this deliberately does not do** is stop testing the current data. The
publish workflow still runs the static-site suite against the rolling asset
before the demo updates, which is precisely what kept the gutted database off
the public site on both nights it was published.

### Point 33 — half the product had no public link

Done 2026-09-25; the owner chose **both** routes.

**(a) Streamlit Community Cloud — the owner's sign-in.** Re-checked as that host
runs it (clean checkout, `requirements.txt`-only venv, the README's secrets,
driven in a browser). That check found point 44 (a deep link on a cold container
left the app empty), now fixed and re-verified. The README's click list is
correct and gains the first-visit download note. The one remaining step is the
owner's OAuth sign-in at share.streamlit.io.

**(b) A Portfolio page in the public demo** (`/portfolio`). A transaction log
kept in the browser's `localStorage` (and saying so), and everything the engine
derives from one that needs only published data: FIFO lots and realized gains,
value and P/L, the per-holding Add/Hold/Trim/Sell with its reason, concentration
(position and sector HHI, 15% warnings), sector gaps with the screener's top
names, the "rebalance worth considering" reasons, and the risk of today's mix
(volatility, 1-day 95% VaR and expected shortfall, max drawdown, correlations).
Trades take that day's published close when the price is blank; oversells are
refused; CSV export/import uses the full app's format; an example portfolio
(the engine's own) loads in one click. Not included, and said on the page: the
optimisers, the rebalancing trade list, the history panel, splits after purchase.

**How two implementations are kept from drifting.** `frontend/src/lib/portfolio.ts`
ports `transactions.py`, `recommendations.py`, `holdings.sector_weights` and
`risk.portfolio_risk`. Both are pinned to `frontend/tests/fixtures/portfolio-golden.json`:
`tests/unit/test_portfolio_golden.py` recomputes it with the engine, and
`frontend/tests/portfolio.spec.ts` requires `analyse()` — exactly what the page
renders — to match it to 1e-9. It holds two seeded scenarios (a year of trades
spanning a long and a short lot, a dividend-adjusted series, a name with missing
days, cash, every action, warnings, gaps; and a book too short for VaR or
correlation), the one-year rule's edge dates, the engine's example portfolio and
its CSV. A static-suite test then pins the page to `analyse()` on the real
published files.

Details the port had to get exactly right: numpy's linear quantile, sample (n−1)
standard deviation, a return touching a missing day staying missing (pandas 3
does not forward-fill — measured), Python's round-half-even in `{:.0%}` messages
(`percent0`), holdings valued at `close` while returns use `adj_close`.

**Mutation-checked twenty ways** across the port, the store and the page — FIFO→LIFO,
half-up rounding, `>=` for the one-year rule (survived at first: no sale fell on
an anniversary, so the golden file gained edge cases), nearest-rank quantile,
population std, forward-filled gaps, returns on `close`, a sixth gap candidate,
no concentration cap, a 20-day correlation floor, the headline total without
cash, a blank price taking the wrong bar, an oversell accepted — each caught.
Two were equivalent and said so: valuing at `adj_close` (equal to `close` on the
latest bar) and the Feb-29 special case (string comparison already draws the
engine's boundary), which was removed as dead code.

**Found on the way:** the Symbol field's `<datalist>` sat inside its `<label>`,
so its accessible name was "Symbol" followed by 503 company names; moved out.
axe-core: 0 violations on the page, empty and filled, both themes.

**A recorded decision, superseded by the owner:** `test_the_spa_does_not_grow_a_portfolio_manager`
asserted no SPA page was called Portfolio — a proxy for ADR 4.5's read-only API.
It is replaced by the invariant itself (`test_the_spa_never_writes_to_the_api`:
the client only issues plain GETs, nothing under `src/` sets a method or beacons)
plus a check that the page says it is local.

### Point 44 — the hosted Streamlit app would have stayed empty after a deep link

Found 2026-09-25 while re-checking finding 33's "the repo is prepared for
Community Cloud". **Reproduced as that host runs it**: a clean checkout (no
database — it is a release asset), a venv built from `requirements.txt` alone
(564 MB, Python 3.12), `streamlit run app/Home.py` with the README's two secrets,
driven in a browser.

- Home opened first: the 82.8 MB download completed and every page worked.
- **A deep link opened first** (`/Screener`): only `Home.py` called
  `ensure_demo_database()`, so the page connected first and SQLite created a
  **0-byte** `quantpulse_demo.db`. From then on "the file exists" skipped the
  download — Home included — and every page showed
  `OperationalError: no such table` for the life of the container.

Community Cloud sleeps idle apps, and a shared link is exactly how a cold
container gets its first visitor.

**Fix:** every read in the app goes through `lib.data.get_session`, which
ensures the database first (the Portfolio page used the engine's own
`get_session` and now does not); and a **zero-byte** file counts as missing —
in `ensure_demo_database` *and* in `demo_data.fetch`. Only zero bytes: anything
larger may be someone's small real database, and is never replaced.

**The first version of this fix did not work, and the reason is a recorded
trap** ("assert the caller, not just the helper"): the unit tests mocked
`fetch`, which hid `fetch`'s own `if target.exists(): return False`. The
simulated host stayed broken with every test green. A caller-level test now runs
the real `fetch` (HTTP stubbed) through `ensure_demo_database`. Re-verified on
the simulated host: deep link first → database downloaded, Screener, Portfolio
and Stock Detail all render; a leftover 0-byte file → repaired on the next read.

**Hermeticity:** every app read can now download, so the full suite ran under a
plugin that fails any test reaching `fetch` via `ensure_demo_database` (proved
non-vacuous on a probe test): none does. Mutation-checked four ways (the
session skipping ensure, "exists" meaning present in either function, a page
importing the engine's session) — each caught by name. The first mutation round
hit the zsh word-splitting trap (`$T` holding two paths → "no tests ran"); the
harness printed the real tail, so it was not mistaken for a pass.

**Noticed, not changed:** the Streamlit Stock Detail page reads nothing from
the URL (`?symbol=NVDA` opens the top-ranked name), so the full app cannot be
deep-linked per stock.

### Point 32 — 508 real pages, and a shared link arrived as a bare URL

Fixed 2026-09-25. Measured on the live site: `/stocks/NVDA/` answered 200 with
its own `<title>` and description, but carried **no Open Graph or Twitter tags**;
`/quantpulse/sitemap.xml` and `/quantpulse/robots.txt` were 404.

`scripts/emit_route_pages.py` now writes, into every page it emits (the root,
`404.html`, the four fixed routes, every stock): `og:type`, `og:site_name`,
`og:title` and `og:description` (the page's own), `og:url` and a canonical link
(absolute; 404 claims none), one `og:image` with its size and alt text, and
`twitter:card=summary_large_image`. It also writes `sitemap.xml` from the same
route list (508 URLs; `lastmod` is the data's newest price date). Absolute URLs
come from `actions/configure-pages`' `base_url`, which now runs before the
emitter, so a fork's cards point at the fork.

The image is `frontend/public/og-image.png`, 1200×630 PNG (unfurlers do not
render SVG): the shared mark and the name on the light paper, rendered by
`scripts/render_og_image.mjs` — re-run it if the mark or the name changes.

**Not done, deliberately: `robots.txt`.** Crawlers read it only at the host
root, and on a project site that is `marlenmm.github.io/robots.txt` — a URL only
a separate `MarlenMM.github.io` repository could serve (it is 404 today). A file
under `/quantpulse/` would be read by nothing; without one, everything is
allowed already. The sitemap can be submitted in Google Search Console, which is
an account action for the owner.

**A trap the emitter had to handle:** the root `index.html` is both the shell
every page is read from and a page itself, so a second run would have stacked a
second set of tags on all 508 pages. The emitter strips its own tags before
writing; a test runs it twice and requires identical output (also checked on the
real `dist/`). Mutation-checked: `og:url` dropped, the strip removed, stocks
missing from the sitemap, a relative image URL, `configure-pages` after the
emitter — each caught by name.

### Point 31 — the charts failed an accessibility check, and so did the palette

Fixed 2026-09-25. Measured with axe-core 4.10 on the local static build, every
page, both themes:

| | before | after |
|---|---|---|
| `nested-interactive` (serious) | Stock Detail ×3 (Dashboard's went with 29) | 0 |
| `color-contrast` (serious), light theme | **13–116 per page, every page** | 0 |
| dark theme | Strong Sell chip below AA (not on screen during the run) | 0 |

**The charts.** Plotly figures sat in `role="img"` wrappers while the modebar put
8, 3 and 8 focusable buttons inside them. They are `role="figure"` now (a figure
may contain controls); the modebar stays, because it is the only visible way out
of a zoom. A static test checks every page for any `role="img"` with a focusable
descendant — it caught a link planted in the SVG dial — and that the three
charts are labelled figures.

**The palette — the audit missed this, probably because it ran in dark mode.**
The light `--muted` (#78746a) was 4.39:1 on the page and 4.06 on sunken panels,
under AA's 4.5. With the user's approval it is **#706c63** (4.93 / 4.56 / 5.23),
changed in all four copies (both light blocks, the chart fallback, Streamlit's
light `grayColor`). Then the rating chips: text on its own 12% tint was Buy
3.27–3.72, Hold 3.81–4.32, Sell 4.12–4.71 in light, and Strong Sell 3.37–3.80 in
dark. Also with approval, chips now take **chip-only inks** (`--up-ink`,
`--flat-ink`, `--down-ink`, `--down-strong-ink`: light #177241 / #7b5e19 /
#b33025, dark Strong Sell #d86f65), same hues; the tints and the chart colours
are unchanged.

`tests/unit/test_text_contrast.py` computes WCAG ratios from the stylesheet's
own tokens — every text token on every background, every chip ink on its tint
composited over every background, in all three theme blocks — and checks the
four copies of `--muted` agree. It found the dark Strong Sell failure that axe
could not, because axe only sees what is rendered.

**Traps:** (1) **axe only judges what is on screen** — the dark run said "0"
while a Strong Sell chip would have failed; compute token pairs instead. (2)
**`npm run test:e2e` leaves an API-mode `dist/`** (§2) — the first axe run here
measured empty pages and reported "no h1" everywhere.

### Point 30 — the browser tab never changed as you moved through the app

Fixed 2026-09-25. Reproduced in a browser before the change: clicking from the
Dashboard to the Screener left the tab on "QuantPulse — S&P 500 research"; so did
every stock. Every route's emitted `index.html` had its own `<title>` (a hard load
was right) — but nothing set `document.title`, and the three other fixed routes
were verbatim copies of the shell, so four pages shared one title even on a hard
load.

- `scripts/emit_route_pages.py` gains `FIXED_TITLES`: the Dashboard keeps the
  site title; Screener, Track Record and Glossary name themselves.
- `frontend/src/lib/title.ts` holds the same strings, `stockTitle()` (the
  emitter's "NVDA — Nvidia") and a `useDocumentTitle` hook. The router sets the
  fixed routes' and the not-found page's titles; Stock Detail sets its own once
  the payload names the company.

**How "they can't drift" is enforced**, since Python and TypeScript cannot share
a constant: the static suite navigates *inside the app* — Dashboard, each nav
link, a stock from the table, and Back — and asserts `document.title` equals the
`<title>` in that route's emitted page. Mutation-checked four ways (a Screener
title with a different separator, an en dash in the stock title, the stock page
setting none, two Python routes sharing a title), each failing with the two
strings shown. Static suite 29/29, e2e 6/6, pytest 1,949.

**Trap for local runs:** the title test reads `dist/<route>/index.html`, which
only `emit_route_pages.py` writes — and every `npm run build` wipes `dist/`. Run
the emitter after each build (the Pages workflow always does).

### Point 35 — two of the four horizons could never be graded

Measured on the published database: 5- and 20-day forecasts are graded on every
row (hit-rate windows up to 156 and 39); **all 15,504** stored 63- and 252-day
rows are ungraded. A hit rate needs 30 non-overlapping windows; the weekly run's
~3.5-year read window allows at most 164 / 39 / **13** / **0** at h = 5 / 20 / 63 /
252, and the log reports 9 at h=63 once GBR's training floor is counted. That gap
does not close with time on this history (≈7.5 more years at 63, ≈30 at 252), so
every quarter and year forecast was destined for the ungraded drawer, carried the
largest numbers on the page, and cost ~5 of the forecast step's ~20 minutes.

**User's call: drop them, say why.** Grading them against the deeper local history
(1972–2026) was rejected: that history covers only today's survivors.

**Fix:** `forecasting.DEFAULT_HORIZONS = (5, 20)`, used by the weekly run and, now,
by on-demand lookups (which had offered 63). The reader serves only published
horizons, taking the latest date over them too, so the 63/252 rows kept by the
append-only table (and still at the latest date of a symbol whose weekly run
failed) leave both front ends and the static site with this push, not with the
next weekly run, and nothing is deleted. `forecasting.HORIZON_SCOPE_NOTE` says why
on both Stock Detail pages (server-composed, printed verbatim). The ungraded
drawer stays for a published horizon a model is short of windows at. The read
window was deliberately **not** shortened: it is also what the hit rates are graded
over. README / HOW_TO_USE updated.

**Tests:** every published horizon can reach 30 windows on the read window (the
bound is checked against a real `walk_forward_accuracy` run), and 63/252 cannot;
stored 63/252 rows are not served, and a newer date holding only them doesn't
hide the published set; the note reaches the API and the Streamlit page; on-demand
horizons equal the published ones; the Streamlit drawer test now grades h=5,
leaves h=20 ungraded and seeds leftover 63/252 rows; e2e makes its ungraded row
explicitly from the recaptured fixture; the static test reads the published
payload instead of expecting a 252-day row. Mutations (re-add 63; drop either
reader filter; drop the Streamlit note; on-demand back to 63) each fail a test.

### Point 46 — the hosted app served new pages on top of old modules

Found 2026-09-26, the first code push after the Community Cloud deploy that
changed a shared module. The host applies each push to the running app's files
and keeps the process; Streamlit drops changed modules only for sessions
connected when the files change. After finding 34's push, the hosted Stock
Detail page raised `ImportError: cannot import name 'format_edge_cell' from
'lib.format'` — the page was new, the module was not — until the app was
rebooted (done at once; the page then rendered, including point 45's migration of
the container's old database).

**Fix:** `app/lib/code_reload.refresh_changed_code()`, called by every page before
its other `lib` imports: it compares source modification times under `app/lib`
and `src/quantpulse` with its last snapshot and drops the modules whose source
changed (and the package attribute `from pkg import mod` would otherwise hand
back). A function call, not an import side effect, because a module cannot remove
itself from `sys.modules` mid-import; pages carry a file-level E402 waiver
saying so.

**Reproduced and verified on a real server** with a copy of the repository: visit
Stock Detail, disconnect, "deploy" a new function in `lib/format.py` used by the
page, visit again — *without* the call: `ImportError`, as in production; *with*
it: the new code served. Tests cover the drop, the no-change path, the
record-then-compare sequence, and that every page calls it first.
**Limit:** the first call in a process only records the baseline, so a process
already stale before this existed needed one reboot.

### Point 34 — a forecast was shown beside a hit rate that was not evidence for it

Done 2026-09-26; the owner chose **the interval plus a plausibility flag**.

**Re-measured first, and it changed the question.** The audit's +90.7% SNDK row
had moved; the largest graded forecast was GBR's 20-day +46.8%. The proposed
per-symbol plausibility bound would have fired on **0 of ~2,940** graded
forecasts outside the stock's own range (1 outside its 1st–99th percentile): SNDK
rose 47× in 19 months, so +46.8% was its 72nd percentile. The real problem was
the hit rate. Reproducing the nightly's pooled walk-forward exactly (GBR 20-day
51.8%, 34 windows) and bootstrapping **by window** — twenty names in one window
are one piece of evidence — **no model's edge over naive excluded zero at any
graded horizon**: GBR 20d +3.5 pts [−0.9, +7.8], GBR 5d +0.6 [−1.7, +2.9], ARIMA
20d +0.4 [−0.3, +1.2], ARIMA 5d −0.5 [−1.2, +0.1].

**A second defect under the first:** "vs naive" was the naive forecast's rate over
*its own* 39 windows while GBR is graded on 34; over GBR's windows naive was 48.2%,
not the 49.4% printed beside it, although the page says "over the same periods".

What changed:
- `backtest.paired_edge_ci`: model minus naive over the same pairs, 90% interval
  from resampling whole windows.
- `refresh_data._pooled_hit_rates` → `PooledAccuracy(rate, windows, baseline_rate,
  edge)`: naive measured on each model's own pairs; the edge and interval stored
  per row (migration `a34e1d9c2b70`: five nullable columns).
- `forecasting.own_history_position`: the forecast's percentile among the stock's
  past h-day moves over the series the model was fitted on, and whether it is
  beyond them all (≥60 moves, else nothing).
- `forecasting.describe_edge` / `describe_history_position`: one sentence each,
  sent by the API (`edge_note`, `history_note`) and printed verbatim by both
  front ends. React: an *Edge vs naive* column with the Track Record's interval
  whisker, a "beyond its history" marker, and the sentences under the table.
  Streamlit: the same column as text, the marker, the same sentences.

**Verified on the real thing:** the nightly's own `refresh_forecasts` run on a
migrated copy of the published database (30 min, 5,534 rows) stored GBR 20d
**+2.75 pts [−1.9, +7.0]** on 09-23 data — consistent with the independent 09-21
measurement — with 52.2% vs 49.4% naive over the same 34 windows, now subtracting
exactly; every interval straddles zero; the only "beyond its history" rows were
three 63-day ones (a horizon finding 35 drops). Rendered in both front ends, both
themes, 375px; axe unchanged.

**Tests and mutations:** paired-CI unit tests (including "copies inside a window
buy no precision"), sentence tests, pooled-accuracy tests, a caller-level
`refresh_forecasts` test, an API round trip, and a data-driven static test.
Mutation-checked eight ways; one — naive over its own pairs — **survived at
first** because every fixture graded the model and naive on identical pairs;
a stand-in model that declines to call on alternate windows now reproduces the
production case and catches it. The e2e fixture was recaptured from the real API
output, not patched. **Until the next weekly run (2026-09-28) the live rows are
the old ones**, so the pages show "—" in the new column; nothing was backfilled
by hand.

### Point 45 — the published database was read at whatever schema it had

Found 2026-09-26 while preparing finding 34, the first new database column since
the demo database became a release asset. The nightly migrates before it writes,
but **nothing that only reads the published file migrated it**: the Pages build
pre-rendered the asset as downloaded, and the hosted Streamlit app used its
downloaded copy as-is. With 34's columns the pre-render would have failed on its
first push, and the hosted Stock Detail page would have raised — and kept raising
on a warm container, which never downloads again, even after the nightly had
published a migrated file.

- `pages.yml` gains **Migrate it to head** between the fetch and the pre-render
  (CI already did this for its pinned fixture). A test requires that order in any
  workflow that pre-renders the database.
- `demo_data.ensure_schema_current(url)` upgrades a SQLite database to head;
  `lib.data.get_session` calls it once per process (`ensure_schema`, a cached
  resource) after the download check. It leaves alone any database without an
  `alembic_version` table (a test's `create_all`), and builds Alembic's `Config`
  without the ini file so `env.py` doesn't reconfigure the host's logging.
- **A trap found on the way:** `env.py` overwrote any URL with the settings'
  `DATABASE_URL`, so migrating a named file would have migrated whatever the
  settings pointed at. It now honours an explicit URL; the CLI's ini placeholder
  still falls back to settings.

Mutation-checked four ways (migrate step removed, session not migrating, the
unconditional URL override restored, the managed-database guard removed). Run
alone on `HEAD` plus these files against the published, not-yet-migrated
database, the suite passed but for two Streamlit tests that assert a category is
missing on the older local database — true there, not on the published one.

### Point 29 — the landing page downloaded Plotly to draw one dial

Fixed 2026-09-25. **Measured on the built site** (raw bytes; Pages gzips):

| page | before | after |
|---|---|---|
| Dashboard (landing) | 4.70 MB, of which Plotly 4,113 KB | **0.59 MB**, no chart library |
| Stock Detail | 4.48 MB (Plotly 4,113 KB / 1,239 KB gz) | Plotly **1,149 KB / 391 KB gz** |
| Screener, Track Record, Glossary | 0.30–0.57 MB | unchanged |

Two changes. **The regime dial is hand-drawn SVG** (`components/RegimeGauge.tsx`),
the same decision the interval whisker made: a semicircle, three bands, an arc
and a number, coloured from the page's own custom properties, one `role="img"`
wrapper with nothing focusable inside. **Stock Detail's Plotly is trimmed**:
`lib/plotly.ts` registers only the three trace types the app draws (candlestick,
scatter, scatterpolar) and `Chart.tsx` builds the component with
`react-plotly.js/factory` instead of the default entry, which bundles every
trace type Plotly has.

**A second bug the gauge was hiding.** `market_regime` labels risk-on from
**60**, but both gauges (React and Streamlit) drew the green band from a literal
**65** — so a 64.7 on 2026-09-16 read "Risk On" with its arc ending in the
neutral band (2 of the 32 published days). The cutoffs are now public
(`market_regime.RISK_ON_AT` / `RISK_OFF_AT`, `label_for`), sent with every
`/api/regime` point (`risk_on_at`, `risk_off_at`), and both gauges draw from them.

**Two traps from the trim:**
- **Plotly's *source* modules read Node's `global`**; the prebuilt bundle does
  not. The first trimmed build threw `global is not defined` and mounted no
  chart — invisible to `tsc` and the build, caught by the static suite's stock
  page test. `lib/plotly-global.ts` defines it, imported first.
- **An unregistered trace type fails silently.** Plotly resolves it to an empty
  `scatter` and logs nothing: unregistering `scatterpolar` blanked the radar with
  all 28 static tests green. The stock page test now asserts every figure's
  resolved trace types equal the ones it asked for.

**Tests and checks:** static suite 28/28 with two new tests (the Dashboard's
JavaScript stays under 400 KB with no Plotly figure; the dial's bands are the
API's cutoffs and the band holding the score is the one the label names), the
trace-type assertion, `test_regime_cutoffs.py`, an API test and a Streamlit
test. **Mutation-checked** eight ways: an eager `react-plotly.js` import (caught
by size alone, 4,374,680 bytes), a stray `<Chart>`, the band hard-coded at 65
in the SVG and in Streamlit, the labeller on a literal, an unregistered trace
type, and — before the fix — the missing `global`. Screenshots checked in both
themes and at 375px (the first cut clipped the "100" tick; the viewBox was
widened).

### Point 43 — my fix for 27 stopped the nightly from starting at all

Found and fixed 2026-09-25, the morning after 27 shipped. The scheduled run
`36076360382` ended in **`startup_failure`**: no job ran, 2026-09-24 was never
fetched or published, and the demo stayed on 2026-09-23. GitHub's annotation:
*"Error calling workflow … pages.yml … The nested job 'notify' is requesting
'actions: read', but is only allowed 'actions: none'."*

27 gave `pages.yml` a `notify` job with `actions: read`. The nightly's `publish`
job calls `pages.yml` and granted only contents/pages/id-token. **GitHub checks
every job of a called workflow against the caller's grant when the calling run
starts — including a job whose `if:` would skip it.** On a push `pages.yml` is
top-level and may ask for anything, so CI and Pages were green on every commit
and only the scheduled run broke. And the `notify` job that should have reported
it could not start either — 27's own stated limit, met on its first night.

**Fix** (`4b78f59`): `actions: read` on the nightly's `publish` job.
**Guard:** `test_a_called_workflow_never_asks_for_more_than_its_caller_grants`
checks every caller of a local reusable workflow against every callee job's
request, and requires the caller to have an explicit block; on the pre-fix file
it fails with GitHub's own wording (two further mutations caught).

**Verified on GitHub without running anything.** A refresh dispatched before the
US open would store a pre-market options chain (the reason the cron waits for
the close), so instead two throwaway branches carried the workflow with every
job `if: false` — GitHub still validates the whole call graph at startup. The
pre-fix copy: `startup_failure` (run `36127186443`); the fixed copy: started,
every job skipped (run `36127189875`). Branches deleted afterwards.

**What 09-24 cost:** prices backfill on the next run (the fetch is incremental
from the last stored date); that day's options snapshot and its regime and
composite rows cannot be recovered.

**Traps:** (1) a called workflow's permissions are checked at the caller's
startup, for skipped jobs too; (2) a workflow change that only a *schedule* runs
is untested by every push — the `if: false` branch dispatch is a cheap,
side-effect-free way to make GitHub validate it before the night does.

### Point 28 — five finished features wait on secrets only the owner can add

Re-verified 2026-09-25. **Nothing to activate:** `gh secret list` still shows
only `SEC_EDGAR_USER_AGENT`, as on 2026-09-23, so there was no new run in which
a feature could have switched on. What each unset path does, from the latest
logs (weekday run `35937083808`, weekly run `35671993244`) and the published
database:

| secret | unset path, as logged | in the published database |
|---|---|---|
| `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` | one line: *"Paper trading is not configured (no Alpaca credentials); skipping"* | `paper_trading_snapshots` **0** rows |
| `ALERT_DISCORD_WEBHOOK_URL` | one line: *"Alerting is not configured (no webhook URL); sending nothing"* (and, since point 27, one line per failure/staleness notice) | — |
| `FRED_API_KEY` | six warnings, *"Skipping macro series fetch_… : FRED_API_KEY not set"* (weekly) | `market_regime.yield_curve_spread` null on all **32** days |
| `FINNHUB_API_KEY` | **503** warnings, one per ticker (point 37, still open) | `short_interest` **0** rows |

**The README table was incomplete, and is fixed.** It named every secret
exactly, but gave no source for `FINNHUB_API_KEY` or `FRED_API_KEY`, said the
webhook unlocked only the data digest (it now carries point 27's notices too),
and dated its status line 2026-09-06. It now has a *Where to get it* column, the
unset behaviour of each, the Alpaca key's clock, and today's verified state.
`tests/unit/test_readme_secrets.py` fails if any secret a workflow reads has no
row, or a row lacks what it unlocks or where to get it (the pre-fix README fails
it; two mutations checked).

**A caveat the table now states, found while checking it:** the Finnhub
short-interest field names are an *unverified guess* —
`ingestion/short_interest_client.py` says so in its own docstring, since no key
was ever available to inspect a real `/stock/metric` response, and Finnhub's
published specification does not enumerate that map. So adding
`FINNHUB_API_KEY` may produce `short_interest` rows with empty values. The first
weekly run after the key is added is the check; if the values are null, the
field names need correcting against one real response.

### Point 27 — nothing told anyone when the pipeline broke

Fixed 2026-09-25. On 2026-09-15 the nightly went red and published an empty
database; on 2026-09-24 the refresh succeeded, the publish gate failed (point
42) and the demo froze on the previous day. Both were findable only by opening
the Actions tab. Point 10's alerting is about the *data*; nothing had the job
itself as its subject.

**Failure notices.** `src/quantpulse/alerting/pipeline.py` +
`scripts/pipeline_alert.py failure`: read this run's jobs with
`gh run view --json jobs`, name each failed job and its first failed step, and
post that with the run URL through the existing `discord.send`. On the real
09-24 run it reads *"publish / build — step 'Check the built site actually
serves its data'"*. `refresh_data.yml` has a `notify` job (`needs: [refresh,
publish, keepalive]`, `if: failure()`); `pages.yml` has one for push runs, and
stays quiet when the nightly called it (new string input `failure_notice:
caller`) — one failure, one message. A string rather than a boolean because
GitHub's `==` is loose and an absent boolean compares equal to `false`.

**The unset path is the normal one.** `ALERT_DISCORD_WEBHOOK_URL` is unset (as
in every fork): one log line, exit 0 — run locally against the real 09-24 jobs
JSON it logs exactly that. The URL reaches the script through `env` only; a test
refuses it in any `run:` line. A configured webhook that *refuses* exits 1 with
`discord.send`'s already-redacted error, because a revoked webhook would
otherwise make every later notice vanish silently.

**Staleness — the half a failure notice cannot see.** A job in the nightly with
no `needs` (so it runs on exactly the nights the refresh fails), first, a day
after the last publish (so Pages' cache cannot show it a stale copy of a fresh
deploy). It reads the *published* `health.json` and counts completed NYSE
sessions newer than its price date — the market calendar, and today only after
the 16:00 ET close. A healthy site reads **1** at that moment: tonight's.

`STALE_AFTER_SESSIONS = 2`, **from the record**. Replaying every scheduled run
from 2026-07-27 to 2026-09-24, the site was one missed night behind five times —
two isolated (08-03, 09-24) and three the first night of a real outage, the
longest thirteen sessions and unnoticed for two weeks. Alerting past two fires
on the second night of all three outages and on neither isolated miss (those
already get the failure notice). It is `STALE_AFTER_DAYS`' rule — "roughly twice
the cadence" — counted in sessions, so Labor Day and Thanksgiving weeks raise
nothing (both tested: weekday counting would have alarmed on each). A stale site
fails the job whether or not the webhook is set: the missing secret is not an
error, the frozen site is, and a red run is something GitHub can email about
(depending on notification settings).

**Verified:** CI green on `bcb8450`; the next Pages run evaluated `notify` and
skipped it on success; run locally, the live site (2026-09-23 prices) was within
bounds. **Mutation-checked sixteen ways**, each failing its own test.

**Limits, stated:** the staleness check rides on the nightly schedule — if that
stops, so does it (point 26 is what stops that). The notice jobs need
`uv sync`; a run that failed *because* PyPI was down may not be able to report
it, and then only the red run remains.

### Point 26 — the schedule would have switched itself off after 60 idle days

Fixed 2026-09-25, with **point 41** (the comments that hid it). GitHub disables
a public repository's scheduled workflows after 60 days without repository
activity. Until point 18 the refresh committed the database every weeknight, so
this could not happen; since the database became a release asset only a person
commits. Measured: the repository was 66 days old, its longest gap between
commits 18.7 days — the exposure is the quiet period after active work stops,
which is exactly when nobody would notice the demo freezing. The workflow's own
comment said the caveat *"no longer applies … there is none left to keep
alive"*, written while the cron was removed and left behind when it came back.

**What counts as "activity" is not documented**, and it decided the mechanism.
GitHub's page says only "no repository activity"; community reports disagree
about releases and about the workflow-enable API (one maintainer in GitHub's
own discussion forum was waiting two months to find out). A commit on the
default branch is the one thing every account agrees counts, and its effect is
checkable: the last commit's date *is* the clock.

- `scripts/keep_schedule_alive.py` (standard library only) writes
  `docs/refresh_status.md` — the last refresh's result and run URL, and when the
  demo database was last published — **only when `main` has been idle for 30
  days**. At most one commit a month; none in any month with another commit, so
  point 18's reason for leaving git is not undone. Thresholds at or past 60 are
  refused.
- `.github/workflows/keepalive.yml` has **no schedule of its own**: the rule
  disables every scheduled workflow in a repository at once, so a separate timer
  would stop at the same moment as the refresh. The refresh calls it as a job
  (`needs: refresh`, `if: always()` — a run of failed nights is when nobody may
  be committing). `workflow_dispatch` with `max_idle_days: 0` forces a
  heartbeat.
- The heartbeat is pushed with `GITHUB_TOKEN`, which by design starts no other
  workflow — no CI run, no Pages rebuild.

**Verified end to end** rather than in a month: run `36034421003` (forced)
committed `172e223` as `github-actions[bot]`, and no CI or Pages run followed
it. **What remains unproven** is GitHub's side: that a bot commit resets the
60-day clock is what every report says, not what GitHub documents. The limit is
stated in `keepalive.yml`: if the schedule has already been disabled, nothing in
the repository can run to fix it — `gh workflow enable refresh_data.yml`.

**Point 41** is closed by the same commit: the publish step's comment now
describes the release asset and what moving it cost; the publish job and
`pages.yml` no longer explain a `[skip ci]` commit that stopped existing; and a
test refuses those phrases in *every* workflow file (it caught `pages.yml` on
its first mutation — a literal file list would not have).

Mutation-checked eleven ways, each failing its own test by name.

**Trap, seen again:** the push of `9f79591` created no workflow runs at all,
with GitHub reporting every component operational — the quirk recorded in
memory. `ci.yml` has no `workflow_dispatch`, so the only recovery is another
push.

### Point 42 — the publish gate held the demo hostage to the market

Found and fixed 2026-09-25, while starting on 26. The 2026-09-24 scheduled run
(`35937083808`) went red: the refresh succeeded, then `publish / build` failed
one static-site test of 26 and the demo stayed on 2026-09-22 data. The test
asserted the Dashboard shows the literal text **"Risk On"** — a claim about the
market. On 2026-09-23 the regime scored 55.7 (76.1 the day before) and came out
*neutral*, so the gate would have held the site until the market happened to
turn risk-on again. Nobody was told (that is point 27).

The gate now reads the newest label from the generated `regime__limit-90.json`
and asserts its humanized form through the client's own `humanize()` — still a
value that can only have come from the database. Reproduced first on the
published 2026-09-23 database (same failure), 26/26 after, and mutation-checked
two ways (the Dashboard rendering the oldest row; an empty regime file). Live
`health.json` moved from 2026-09-22 to 2026-09-23 on the next publish.

**Trap:** a browser gate that asserts a *data value* by literal is a test of
the data, and data moves. Assert that the page shows what the generated file
says, not what it said the day the test was written.

**An observation, not investigated:** the regime's 76.1 → 55.7 drop coincides
with the macro-tone input (GDELT, handoff §4 item A) returning after two days
absent, at −0.57, its most negative reading that week — so part of the move may
be renormalization over four inputs instead of three rather than the market.
Not attributed; worth a look alongside item A.

### Point 25 — the catalogue failed exactly on the nights new tickers appeared

Fixed 2026-09-23. It first showed on the empty-database night (2026-09-15) and
looked like an artefact of that night. **It recurred on 2026-09-21 on a normal,
populated database** — the Monday S&P's quarterly rebalance took effect — with
the same `sqlite3.OperationalError: database is locked`, this time while
inserting new ETF listings.

**The mechanism.** `sync_universe` rewrites every constituent's name, sector and
active flag each night and flushes through the run's shared session. SQLAlchemy
only emits SQL for values that changed, so on an ordinary night nothing is
written and no lock is taken. On a night the index changes, the flush writes,
and SQLite holds that write lock until the shared session commits at the end of
its block. The catalogue ran *inside* that block, in a session of its own, and
every night has something new to catalogue (8–15 listings) — so it waited out
the 5-second busy timeout and failed. It failed precisely when new tickers
appear, which is when the catalogue matters most. Reproduced on the real
functions against a SQLite file in 0.34 s before anything was changed.

**The lock was covering an ordering bug.** A session of its own cannot see the
shared session's uncommitted new constituents, so without the lock the
catalogue would have inserted them as catalogue rows, and the shared commit
would then have collided on the primary key and rolled back the universe sync
itself. `sync_catalogue`'s own promise — *a ranked symbol is never demoted* —
only holds if the universe is committed first.

**The fix is ordering, not waiting.** The catalogue now runs just after the
shared block commits. Nothing left in that block depended on it: the queries
after it filter on `is_active`, and catalogue rows are inactive. The audit's
first suggestion, a longer busy timeout, was wrong — against this bug it would
only have made the failure arrive later.

Two tests. A **structural** one parses `run()` and refuses any `_in_session`
call inside an open `get_session()` block: SQLite has one writer, so that shape
is a deadlock waiting for a night when both sessions write, and refusing the
shape covers the next instance rather than only this one. A **behavioural** one
drives `run()` through a rebalance night on a real SQLite file with a 0.3 s busy
timeout, with a new constituent that is also in the listing directory: the
catalogue must not fail, the ETF must land as catalogue, and the new constituent
must stay ranked. Moving the step back inside the block fails both.

**What the same 2026-09-21 run confirmed about earlier points**, since it was
the first weekly run to carry them:

- **24:** *"13F: the 2026-03-01 to 2026-05-31 window reports the quarter ending
  2026-03-31, which is already stored"* — no download, no false "partial".
- **23:** *"GDELT refused Tier-2 basket oil_gas; every remaining basket this run
  comes from Google News"*, then all 17 baskets scored, and the next night
  closed `finished success` with industry/macro populated — the first clean
  night in weeks.
- **22:** tier-1 classification measured **5.0 s per article** on that runner
  (2,485 s for 500), worse than the 3.1 s the cap was sized from. At the cap of
  500 the step took ~50 of its 90 minutes; at the old 1,500 classification alone
  would have needed about two hours.

### Point 24 — every weekly run said "partial" for a step doing the right thing

Fixed 2026-09-17. The 2026-09-08 run's closing line read *"step(s) that wrote
nothing: institutional_ownership"*, and so would roughly eleven Mondays in
twelve. SEC publishes one 13F window a quarter, so most weeks the newest window
is one already ingested: the step re-downloaded the ~100 MB file, correctly
inserted nothing, and `_STEPS_EXPECTED_TO_WRITE` — added to catch the months
this step silently stored nothing — marked the run degraded. **Two good fixes
composed into a false alarm**, and a status that says "partial" every week is
one nobody reads.

**Ask before downloading.** `edgar_13f_client.quarter_end_for_window` maps a
window to the quarter it reports on, and that mapping comes from **Rule 13f-1,
not a guess**: filings are due within 45 days of quarter end, and the windows
are offset one month from calendar quarters, so each quarter is due inside the
window that opens the month it ends. A test checks that premise — quarter end
plus 45 days lands inside the assigned window — for all twelve months. If
`persistence.has_institutional_quarter` says the quarter is stored, the step
returns 0 without downloading. The quarter actually stored is still read from
the file's dominant `PERIODOFREPORT`; if the file ever disagrees with the rule,
the file wins and the log says so.

**Leaving the guard without reopening the silence.** The three ways this step
used to fail quietly now **raise** and fail the step by name: no published
window, a failed download (previously caught and turned into 0), and a window
matching no holdings. So `institutional_ownership` left
`_STEPS_EXPECTED_TO_WRITE`, which now holds `benchmark_prices` alone.

**Measured against live SEC on 2026-09-17**, on a copy of the published
database: the newest published window is still March–May (June–August had not
appeared 17 days after it closed), the rule maps it to 2026-03-31, which is
stored — and **two consecutive calls returned 0 rows in 1.5 s and 0.6 s with no
download**. Before, each weekly run spent the ~100 MB download to reach the same
answer and then reported it as a degradation.

**One consequence, stated rather than hidden:** a constituent added mid-quarter
gets its 13F reading when the next window publishes, not the next Monday.
Section 24 already treats this as a slow quarterly overlay, and re-downloading
the same file never reliably picked such names up — issuer-name matching
decides that.

**Four test classes were reaching SEC live** — `TestPaperTrading`,
`TestWeeklyBranchCadence`, `TestDividendRefresh` and `TestTier2NewsDeadline`, all
through `TestAlerting._driven_run`, which ran the weekly branch without stubbing
13F. Their failures had been swallowed as "wrote nothing"; once the step failed by
name, the one test asserting an exact failed-steps string broke, which is how
they surfaced. The shared harness now stubs 13F as already current. Proven with
a diagnostic plugin that records and raises on SEC's HEAD probe and bulk
download: 38 tests clean with the stub, 12 errors without it. The refresh test
module went from 90 s to 40 s.

**Two traps worth keeping from this one:**

- **A fallback triggered by an *empty* primary result reaches every test that
  mocks the primary as empty.** Point 23's Google News fallback did exactly
  that: a dozen existing tests fetched live headlines and ran the real models,
  taking CI's test step past 18 minutes. The new behaviour's own tests all
  patched the fallback, so none of them — and none of the eight mutations run
  against them — could see it. The cost showed up only as *other* tests'
  runtime. When CI suddenly takes three times as long, believe it.
- **A step that swallows exceptions hides live network calls in tests.** The SEC
  probes above ran for weeks unnoticed because `step()` catches `Exception`. A
  guard that only raises proves nothing there; it has to record the call and
  assert at teardown.

Mutation-checked eight ways, including removing the harness stub to show the
SEC guard is not vacuous.

### Point 23 — industry/macro had no data source that answered

Fixed 2026-09-17. **Backlog point 3 had stayed open through four separate
causes**, and the fourth was that GDELT could not be reached reliably at all:
on every weekly run from 2026-09-08 it throttled the shared GitHub runner, no
Tier-2 article landed after 2026-08-18, `read_tier2_news`'s 21-day window
emptied, and `industry_macro` fell to 0 of 503 names.

**Measured before choosing anything.** GDELT answered a single, unpaced request
from a developer machine with HTTP 429 — so this was not the pipeline being
impolite, or only a shared-IP problem. The same seventeen basket queries sent to
Google News RSS — the keyless endpoint Tier-1 already calls from the runner,
with no recorded failure — returned **50–100 articles each, spanning the full
week**. Relevance is mostly on-topic ("Retail sales rise a better-than-expected
1.2%", "Canada's Steel Tariff Doubles to 50%") with some government and
press-release noise, comparable to GDELT's own keyword matching.

Four changes:

1. **Fall back, and stay fallen back.** On the first GDELT refusal the run stops
   asking GDELT and serves every remaining basket from
   `news_client.fetch_google_news_query`. Each refusal had been costing about a
   minute of 429 backoff *per basket*. An empty GDELT week also asks the
   fallback — a whole sector with no coverage for seven days is not a plausible
   answer — but does not make the switch sticky.
2. **Fetch the week, not the day.** Tier-2 runs weekly and requested
   `timespan="1d"` for its whole life, so even a perfect Monday captured one day
   in seven of a 21-day window. Now `_TIER2_WINDOW_DAYS = 7` for both sources,
   bounded by Google's own `when:7d` operator rather than trimmed afterwards.
   Overlap with last week costs nothing: `upsert_news_events` is append-only on
   `article_id`.
3. **Record the source that answered.** `source` was hardcoded `"gdelt"`. A
   sentiment reading from a different corpus is a different claim, and a run log
   line now says how many baskets came from each.
4. **Classify only what can be seen.** `_MAX_CLASSIFIED_TIER2_PER_BASKET` 40 →
   8, and 8 is provably sufficient rather than chosen: the most any surface
   displays is 8, and an article among the 8 newest *overall* is necessarily
   among its own basket's 8 newest. At the runner's measured 3.1 s per article,
   40 × 17 baskets was ~35 minutes of a 50-minute budget.

**Google News is scored on titles only**, because its RSS summaries are HTML
markup and GDELT supplies titles alone; mixing the two inside one basket's
average would compare markup against headlines.

**Real-data before/after**, on a copy of the published database with live
network and the real FinBERT and BART models:

| | before | after |
|---|---|---|
| names with an industry tilt | **0 / 503** | **503 / 503** |
| Tier-2 articles in the 21-day window | 0 | 1,555 |
| baskets with articles | 0 | all 17 |
| articles classified | — | 136 (8 × 17) |
| step runtime | killed at its 1800 s budget | 73 s |

**A second GDELT failure shape turned up in that run** that no mock had
modelled: not a 429 but an empty body, raised as `JSONDecodeError: Expecting
value: line 1 column 1`. The fallback treats any exception as a refusal, so it
held — but it is the reason the fix catches exceptions broadly rather than
checking for status 429.

**Not fixed here:** the Market Regime Index's macro-news-tone input is also a
GDELT query (`fetch_tone_timeline`) and fails the same way. Google News has no
tone timeline to fall back to, and `describe_regime_coverage` already says when
that input is missing.

Mutation-checked eight ways — refusal never recorded, empty week not falling
back, a one-day window, summaries scored, source hardcoded, cap back to 40,
oldest-first before the cap, `when:Nd` dropped — each caught by the test written
for it.

### Point 22 — the composite scored without a whole category for five weeks

Fixed 2026-09-16. Between **2026-08-10 and 2026-09-14** every one of 503 names
was scored with no news sentiment at all, and nothing anywhere said so.

The chain: `tier1_news` died on its 90-minute budget each weekly run, so no
sentiment rows were written; `read_latest_sentiment` looks back thirty days, so
past that the category simply stopped arriving; `build_composite` renormalized
over the six categories that remained, which is correct behaviour; and the only
visible trace was `data_confidence` falling from 90 to 80, rendered as "good
coverage (80%)". **No step raised, so `step()` saw nothing. No step wrote zero
rows, so `_STEPS_EXPECTED_TO_WRITE` saw nothing. Every run logged success.**

Three parts, because the silence had three causes:

1. **Say which categories are behind a score.** `scoring.describe_composite_coverage`
   composes one sentence server-side and both front ends print it verbatim — the
   same pattern as `market_regime.describe_regime_coverage`, for the same
   reason. On the current database it reads: *"Five of the seven categories are
   behind this score — news sentiment and industry/macro are missing, so the
   weights are renormalized over the rest rather than counting them as
   neutral."* It deliberately does not guess *why*: a stock with no analyst
   coverage and a step that died last night look identical from there.
2. **Make a universe-wide absence loud.** `scoring.zero_coverage_categories`
   plus a new `degrade()` beside `step()` in the refresh: a category empty for
   every name now marks the run `partial` and names itself in the closing line.
   `degrade()` is a third kind of signal — every step ran, every step wrote, and
   the output is still wrong.
3. **Stop sentiment being what a slow week drops.** `_MAX_CLASSIFIED_ARTICLES`
   1,500 → 500, from a runner measurement: BART classification cost **3.1s per
   article** there (4,674s for 1,500) against the 192–347ms measured locally,
   leaving the step finishing 5,327s into a 5,400s budget. Section 7.3's
   priority is unchanged and is what makes this the right thing to cut:
   `event_type` is read by one surface, `sentiment_score` feeds every tilt.

**`MIN_UNIVERSE_FOR_COVERAGE_CLAIM = 20`, and it is not tidiness.** On a
one-name universe "absent from every row" and "absent from this row" are the
same statement, and without the floor the refresh's end-to-end test — which
scores a single mocked symbol with two price bars — reported six categories as
an outage and turned a healthy run "partial".

**Two traps worth keeping from this one:**

- **`degrade()` logs its own reason, so asserting on `caplog.text` proved
  nothing about the summary.** The mutation that dropped `degraded_reasons`
  from the closing line's guard *survived* the first version of the test. Assert
  the run's last line, which is what a reader of a three-hour log actually sees.
- **zsh does not word-split unquoted parameters.** `run "$1"` with `"a.py b.py"`
  hands pytest one nonexistent path and prints "no tests ran in 0.00s", which
  reads like a passing mutation if you are grepping for failures. It cost two
  full mutation rounds here and a broken merge loop earlier the same day. Use
  `"$@"`, and make the harness print the real pytest tail rather than a grep.

**A trap worth keeping.** The new test failed on correct YAML at first:
`text.index("alembic upgrade head")` matched the *comment* in the fetch step
that explains this bug, several hundred bytes before the step that runs it.
Match commands in their `run:` form. The same trap is already recorded one
paragraph up in that test for `build_static_site.py` — it caught a second
victim within the hour.
