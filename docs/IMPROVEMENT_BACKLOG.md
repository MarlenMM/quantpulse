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
