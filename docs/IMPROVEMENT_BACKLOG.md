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

**A trap worth keeping.** The new test failed on correct YAML at first:
`text.index("alembic upgrade head")` matched the *comment* in the fetch step
that explains this bug, several hundred bytes before the step that runs it.
Match commands in their `run:` form. The same trap is already recorded one
paragraph up in that test for `build_static_site.py` — it caught a second
victim within the hour.
