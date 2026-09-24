# QuantPulse — session handoff, 2026-09-16 → 2026-09-23

A second product audit (23 findings, numbered **19–41** so they continue
`docs/IMPROVEMENT_BACKLOG.md`), followed by fixes for **19–25**. This file is the
complete record: what was found, what was fixed and how it was verified, every
open finding in full, and the prompt for the next session.

- **Repository HEAD at start:** `875a23f` · **at end of fixes:** `92a87ad` (this file is committed after it)
- **Tests:** 1,844 passed / 1 skipped at the first fix → **1,895 passed / 1 skipped** at `92a87ad`
- **CI:** green on every fix commit (`92a87ad`: test + frontend, 5m38s)
- **Findings page (artifact, private, version 5):** <https://claude.ai/artifact/HokUNu4qH7eg5jgxeTcByL>
- **Previous (2026-09-03) audit page, points 1–18:** <https://claude.ai/code/artifact/daf2285a-ae37-4964-a239-976ad0a84116>
- **Deeper per-fix write-ups** are in `docs/IMPROVEMENT_BACKLOG.md` §7 (points 19–25), and each commit message carries the full reasoning.

**Continued 2026-09-25** (HEAD `e4303d1`, **1,943 passed / 1 skipped**, CI green on
`bcb8450`): **26, 27, 41 fixed; 42 found and fixed; 28's README fixed** (the keys are
still the user's). Per-finding status is marked in §3; detail in backlog §7. Findings
page version 8. **Not yet observed:** the first scheduled run carrying the new jobs
(2026-09-25 ~00:00 UTC). It should show `keepalive` logging "nothing to do",
`staleness` passing (site at 2026-09-24 data after that night's publish, or 09-23
before it), `notify` skipped, and `publish / notify` skipped. Check it with
`gh run list --workflow refresh_data.yml -L 1` first thing. Next work: 29–40
(33, 34, 35 need the user's call). One new environment trap: never put the
skip-ci marker in a commit message, even quoted (backlog §2).

---

## 1. Environment facts a new session must know first

These cost real time in this session. Read them before running anything.

1. **`/usr/bin/git` is broken on this Mac.** Since ~2026-09-16 it fails with
   *"You have not agreed to the Xcode license agreements"*. Fixing that needs
   `sudo xcodebuild -license accept`, i.e. the user's password — never attempt it.
   **Use `/opt/homebrew/bin/git`** (2.55.0) for every git command.
2. **`gh` shells out to git**, so it fails the same way. Either
   `export PATH=/opt/homebrew/bin:$PATH` first, or pass `-R MarlenMM/quantpulse`.
3. **The scratchpad directory is not durable.** Between 2026-09-17 and 2026-09-23
   it was wiped: a pre-merge backup of the demo database and the findings page's
   source were lost. Anything that must survive goes in the repository or a release.
4. **zsh does not word-split unquoted parameters.** A helper `run() { pytest $1; }`
   called as `run "a.py b.py"` passes ONE nonexistent path and prints
   `no tests ran in 0.00s`. This voided two mutation rounds and silently skipped
   13 tables in a database merge. Use `"$@"` and literal lists.
5. **Never run two pytest processes at once.** A concurrent pair stalled
   indefinitely (CPU frozen) once this session.
6. **Commit attribution:** use the line the session's system reminder specifies
   (this session ended on `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
7. **Commit AND push each fix automatically** (the user's standing convention —
   see memory `quantpulse_commit_push_convention.md`).
8. **The findings artifact** must be updated from its published copy: `Artifact`
   `action: "read"` with the URL saves the full HTML to a local file. That file
   includes the host's `<!doctype>/<head>/<body>` wrapper — strip everything
   before `<title>QuantPulse September Findings</title>` and any trailing
   `</body></html>`, edit, then republish **passing `url`**.

### Release assets (GitHub Releases on MarlenMM/quantpulse)

| tag | asset | role |
|---|---|---|
| `demo-data` | `quantpulse_demo.db` | **rolling** — replaced by every refresh; read by the Pages build, `./run.sh`, the Streamlit app |
| `demo-data` | `quantpulse_demo.previous.db` | one-deep rollback, uploaded by each refresh **before** it clobbers the current one |
| `ci-fixture` | `quantpulse_ci.db` | **pinned** — the only database CI's test job reads; SHA-256 `58480632d3cf613b28fa626bd3799a31c4f2e26c95c324fc91dd7fd031f0fb28` (73,998,336 bytes), recorded in `src/quantpulse/demo_data.py::CI_FIXTURE_SHA256` |

To roll the CI fixture forward: upload a newer database to the `ci-fixture`
release and put its `shasum -a 256` in `CI_FIXTURE_SHA256` **in the same commit**.

### Secrets (verified 2026-09-23)

Only `SEC_EDGAR_USER_AGENT` is set. `FRED_API_KEY`, `FINNHUB_API_KEY`,
`ALERT_DISCORD_WEBHOOK_URL`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` are
**unset**. They are the user's to add — never offer to obtain or enter a key or
webhook URL.

### Production state (verified 2026-09-23)

The last four scheduled Data Refresh runs all succeeded:
2026-09-17, 2026-09-18, 2026-09-21 (weekly, 2h16m), 2026-09-22.
The 2026-09-22 run closed `refresh_data finished success (3434 rows)` — the first
fully clean night in weeks.

---

## 2. What was fixed (19–25), precisely

### 19 · The refresh published an empty database over the real one — FIXED

**Commits:** `5d5bef3`, `4ea5dfd`, `3066644` (2026-09-16).

**What happened.** Point 18 had moved `quantpulse_demo.db` out of git onto the
rolling `demo-data` release. `ci.yml` and `pages.yml` got a fetch step;
`refresh_data.yml` — the one workflow that republishes the asset — did not. So on
2026-09-15 (run `34912965020`) `alembic upgrade head` created an **empty**
database, the refresh filled it with ~4 weeks of data from scratch, and
`gh release upload --clobber` overwrote the real asset. `pragma integrity_check`
passed, because a small database is still well-formed.

| table | published after the run | what it replaced |
|---|---|---|
| price_history | 9,577 | 417,993 |
| forecasts | 0 | 33,320 |
| composite_scores | 1,509 | 39,234 |
| pattern_signals | 164 | 7,320 |
| options_signals | 503 | 13,077 |
| insider_transactions | 20,899 | 32,695 |
| backtest_results | 0 | 7 |
| refresh_log | 1 | 30 |

The public site was saved by the `tests-static` gate (3 failing assertions on the
missing backtest and charts), so Pages kept serving 2026-09-11 data. **It
happened again** on the 2026-09-16 00:02 UTC run (`35038251888`), which published
a **1.5 MB** database; that time the publish job's fetch refused it via
`demo_data.MIN_BYTES` (`downloaded only 1507328 bytes … that is not the database`).

**Restore — a merge, not a rollback.** The bad runs had ingested real data the
good local copy (dated 2026-09-13, data to 09-11) lacked. Ingested tables were
merged from the gutted asset into the good copy with `INSERT OR IGNORE`
(natural-key tables: price_history, dividends, news_events, sentiment_scores,
options_signals, analyst_consensus, fundamentals_snapshot, macro_indicators,
economic_calendar, short_interest, tickers, institutional_ownership,
thematic_baskets). `insider_transactions` (surrogate `id` + unique business key)
was merged by explicit column list; `index_membership_history` (surrogate id, no
unique key; local copy already the superset) was skipped. **Derived tables were
not merged** (computed on 4 weeks of history). Result: price_history 418,412
through 2026-09-14; sentiment 1,003 rows through 09-14 (the good copy had stopped
at 08-10); news_events 15,754 through 09-15; insider 33,673; options 13,580.
Uploaded as the `demo-data` asset (73,998,336 bytes, 2026-09-16T02:07:49Z) and
copied over the repo's local `quantpulse_demo.db`.

**Code:**
- `refresh_data.yml`: a **Fetch the demo database** step after `uv sync --locked`
  (the fetch imports `quantpulse`) and **before** `alembic upgrade head` (which is
  what creates the empty file).
- `scripts/demo_db_guard.py` (new): `capture BASELINE_JSON [DATABASE]` at fetch
  time, `check BASELINE_JSON [DATABASE]` before upload. Refuses the upload if **any
  table has fewer rows** than at fetch (the pipeline's append-only invariant,
  Section 6.8), and refuses when the baseline is missing (= the fetch never ran).
  Exempt tables: `alembic_version`, `portfolio_holdings`, `portfolio_transactions`,
  `watchlist`. Against the real files it reports
  `price_history: 418412 rows before, 9577 after (-408835)` first.
- Job-level env: `DEMO_DB_BASELINE=/tmp/demo-db-baseline.json`,
  `DEMO_DB_PREVIOUS=/tmp/quantpulse_demo.previous.db`. The publish step uploads the
  previous copy **first**, then the new one.
- **Trap hit by the first attempt (`4ea5dfd`):** `${{ runner.temp }}` in a
  job-level `env:` is a whole-file parse error (the `runner` context is
  step-scoped). GitHub created the run, failed it in 0 s, and listed it under the
  file path. `yaml.safe_load` accepts it. New test
  `test_no_job_level_env_uses_a_step_only_context` rejects `runner.`, `steps.`,
  `job.`, `env.` in job-level env.

**Tests:** `tests/unit/test_demo_db_guard.py` (10), additions to
`tests/unit/test_deploy_requirements.py`. **Mutation-checked** five ways (remove
fetch, move it after migrations, drop the census call, break the comparison,
restore `runner.temp`).

**Production verification:** 2026-09-16 23:59 run — `census of quantpulse_demo.db:
26 tables, 632,747 rows`, `census check passed: no table shrank, +3,359 rows`,
published 71 MB.

### 20 · The ordering test checked two workflows out of three — FIXED

**Commit:** `5d5bef3`. `test_every_workflow_that_reads_the_database_fetches_it_first`
now reads `.github/workflows/*.yml` instead of the literal `("ci.yml", "pages.yml")`,
applies to any workflow running pytest, the static build, `refresh_data.py` or
`gh release upload`, and asserts: install < fetch; fetch < `run: uv run alembic
upgrade head`; fetch < `uv run python scripts/refresh_data.py`; fetch < each reader;
`demo_db_guard.py check` < `gh release upload`. **Trap:** `text.index("alembic
upgrade head")` first matched the *comment* explaining the bug — commands are
matched in their `run:` form.

### 21 · CI read a database that changed nightly — FIXED

**Commit:** `f3c6755`. Three Dependabot PRs had gone red on
`test_settings_reports_effective_weights_against_the_real_ranking` because CI
fetched the (gutted) rolling asset. Now:
- New release `ci-fixture` / asset `quantpulse_ci.db`, pinned by SHA-256 (see §1).
- `demo_data.fetch(target, *, url=, expected_sha256=, timeout=)` hashes while
  streaming; a mismatch raises a message saying it is *not about the code being
  tested*. `scripts/fetch_demo_db.py --ci-fixture` selects it.
- `ci.yml`: **Fetch the pinned test database** (`./scripts/fetch_demo_db.sh
  --ci-fixture`) then **Migrate it to head** (`uv run alembic upgrade head`) before
  pytest — a pinned snapshot is also a schema snapshot.
- `pages.yml` and `refresh_data.yml` keep the rolling asset (asserted by a test).
- README "Data and secrets" updated.

**Tests:** `TestPinnedFixture`, `TestCli` in `tests/unit/test_demo_data.py`;
`test_ci_reads_the_pinned_database_and_the_demo_reads_the_rolling_one`.
**Mutation-checked** five ways. **A test bug caught by mutation:** the CLI tests
originally patched `quantpulse.demo_data.fetch`, which the CLI never reads after
import — they asserted nothing (patch where used: `fetch_demo_db.fetch`). CI log
verified: *"Downloading the pinned test database (~70 MB)"* then the migration.

### 22 · The composite scored without sentiment for five weeks — FIXED

**Commit:** `357e6c9`. Between 2026-08-10 and 2026-09-14 every name was scored
with no news sentiment: `tier1_news` died on its 5,400 s budget (2026-09-08 run
failed inside `classify_articles`), `read_latest_sentiment(lookback_days=30)`
stopped returning rows, `build_composite` renormalized, and the only trace was
`data_confidence` 90 → 80 ("good coverage (80%)"). No step raised, none wrote zero
rows, every run said success.

**Code:**
- `scoring.missing_categories(row)`, `scoring.describe_composite_coverage(row)`
  (one sentence, server-side, e.g. *"Five of the seven categories are behind this
  score — news sentiment and industry/macro are missing, so the weights are
  renormalized over the rest rather than counting them as neutral. The
  dashboard's data-freshness strip shows when each source last updated."*),
  `scoring.zero_coverage_categories(scores, *, min_symbols=MIN_UNIVERSE_FOR_COVERAGE_CLAIM)`
  with `MIN_UNIVERSE_FOR_COVERAGE_CLAIM = 20`.
- API: `ScreenerRow.coverage_note`; `api/main.py::_screener_row()` used by both
  `/api/screener` and `/api/stocks/{symbol}`.
- Streamlit: caption on Stock Detail; universe-level caption on the Screener
  ("Missing from every row below: …").
- React: `coverage_note` in `types.ts`; rendered on `StockDetail.tsx`.
- `refresh_data.py`: new `degrade(reason)` beside `step()`, `degraded_reasons`
  list, closing-line guard `if failed_steps or empty_steps or degraded_reasons`;
  after the composite step the run reads back `read_screener_rows(profile=
  "balanced")` through the shared session and degrades per empty category.
- `_MAX_CLASSIFIED_ARTICLES` 1,500 → **500** (BART measured 3.1 s/article on a
  runner).

**Tests:** `tests/unit/test_scoring_coverage.py`, API assertions (identical
sentence on both surfaces), parametrized degrade test via `run()`, two Streamlit
render tests on the real database. **Mutation-checked** six ways — one (dropping
`degraded_reasons` from the summary guard) **survived** at first because
`degrade()` logs its own reason; the test now asserts the run's *closing line*.
Verified rendered on the live NVDA page.

**Production:** the 2026-09-21 run measured tier-1 classification at **5.0 s per
article** (2,485 s for 500); the step took ~50 of its 90 minutes.

### 23 · Industry/macro had no data source that answered — FIXED

**Commits:** `32add7f`, plus the regression fix `254e7dc`.

GDELT answered a **single unpaced request** with HTTP 429 (and later with an
empty body → `JSONDecodeError`). Google News RSS returned **50–100 articles for
each of the 17 baskets**, spanning a full week. Tier-2 also fetched
`timespan="1d"` on a weekly job against a 21-day read window.

**Code:**
- `news_client.fetch_google_news_query(query, *, days=7)` — bounded with Google's
  `when:Nd`; cache key is a slug (not `hash()`).
- `refresh_data._TIER2_WINDOW_DAYS = 7` for both sources;
  `_google_tier2_articles(basket)` returns **titles only** (Google summaries are
  HTML; GDELT gives titles).
- First GDELT refusal → `gdelt_refused = True`, every remaining basket from Google
  News; an **empty** GDELT week also falls back (not sticky).
- `source` recorded as whichever answered; log line
  `Tier-2 baskets by source: …`.
- `_MAX_CLASSIFIED_TIER2_PER_BASKET` 40 → **8** — provably sufficient: no surface
  shows more than 8 stories (Streamlit Home 8, API default 8, SPA 6), and an article
  among the 8 newest overall is necessarily among its basket's 8 newest.

**Real-data measurement** (copy of published DB, live network, real FinBERT/BART):
industry tilt **0/503 → 503/503**, 1,555 articles, all 17 baskets, 136 classified,
**73 s**. **Mutation-checked** eight ways.

**Regression I introduced, fixed in `254e7dc`:** many existing refresh tests mock
GDELT as empty and run the weekly branch, so they started fetching live Google
News and running real models. CI's test step passed 18 minutes (run
`35193283147`, cancelled). Autouse fixture `_google_news_fallback_is_offline` in
`tests/integration/test_refresh_data.py`: one test 23.5 s → 4.7 s, module 265 s →
90 s, suite 328 s → 167 s.

**Production:** 2026-09-21 run — *"GDELT refused Tier-2 basket oil_gas; every
remaining basket this run comes from Google News instead"*, all 17 scored,
`Tier-2 baskets by source: google_news 17`; 2026-09-22 closed `success` with
industry/macro populated.

**Not fixed here:** the Market Regime Index's macro-tone input
(`gdelt_client.fetch_tone_timeline`) also fails — see §4 item A.

### 24 · Every weekly run said "partial" for a 13F step doing the right thing — FIXED

**Commit:** `c4f5575`. SEC publishes one 13F window a quarter; the step
re-downloaded the same ~100 MB file weekly, correctly inserted nothing, and
`_STEPS_EXPECTED_TO_WRITE` marked the run degraded.

**Code:**
- `edgar_13f_client.quarter_end_for_window(window)` — the quarter ending in the
  window's **first** month, from **SEC Rule 13f-1** (due within 45 days of quarter
  end; windows are offset one month). A test checks the premise for all 12 months.
  The file's dominant `PERIODOFREPORT` remains the source of truth; disagreement is
  logged and the file wins.
- `persistence.has_institutional_quarter(session, quarter_end)`.
- `refresh_institutional_ownership`: returns 0 **without downloading** when the
  quarter is stored; **raises** (fails the step by name) on no published window,
  a failed download, or a window matching no holdings.
- `_STEPS_EXPECTED_TO_WRITE = frozenset({"benchmark_prices"})`.
- Trade-off stated: a constituent added mid-quarter gets 13F data at the next
  window.

**Test hermeticity found on the way:** `TestPaperTrading`,
`TestWeeklyBranchCadence`, `TestDividendRefresh`, `TestTier2NewsDeadline` (all via
`TestAlerting._driven_run`) had been reaching SEC live. The harness now stubs
`refresh_institutional_ownership`. Proven with a record-and-raise plugin: 38 tests
clean; removing the stub → 12 errors. Refresh module 90 s → 40 s.

**Live SEC check (2026-09-17):** newest window still March–May → quarter
2026-03-31 stored → two calls returned 0 rows in **1.5 s and 0.6 s, no download**.
**Production (2026-09-21):** *"13F: … already stored; nothing new until SEC
publishes the next window"*. **Mutation-checked** eight ways.

### 25 · The catalogue failed exactly on the nights new tickers appeared — FIXED

**Commit:** `92a87ad` (2026-09-23). `database is locked` on 2026-09-15 **and again
on 2026-09-21** — the Monday S&P's quarterly rebalance took effect.
`sync_universe` flushes through the shared session; SQLAlchemy only writes changed
values, so the write lock is taken only when the index changes, and is held until
that session commits. The catalogue ran **inside** that block in its own session
and hit the 5 s busy timeout. The lock also hid an ordering bug: the catalogue's
own session cannot see uncommitted new constituents, so without the lock it would
insert them as catalogue rows and the shared commit would collide and roll back
the universe sync.

**Fix:** the catalogue step now runs **after** the shared block commits.
**Deliberately not done:** a longer busy timeout (would only delay the same
failure). **Tests:** `TestTheCatalogueWaitsForTheUniverse` —
`test_no_step_opens_its_own_session_inside_an_open_shared_one` (AST over `run()`)
and `test_a_rebalance_night_catalogues_without_locking_or_demoting` (real SQLite
file, 0.3 s timeout). Reproduced beforehand in 0.34 s. Mutation (move it back)
fails both. CI green.

### Other actions taken this session

- Re-ran CI on Dependabot PR #28 (vite) after the restore → green.
- Cancelled the stuck CI run `35193283147`.
- Created the `ci-fixture` release.
- Memory: `quantpulse_review_techniques.md` gained two sections (shell traps
  that void mutation checks; Homebrew git).
- Findings artifact published as versions 1–5.

---

## 3. Open findings 26–41, in full

Text as audited on 2026-09-16, with **dated updates** where facts have changed.
Severity: **S1** data lost or published wrong · **S2** visible to anyone who
opens the demo · **S3** built, and idle or unreachable · **S4** hygiene.

### 26 · The schedule will switch itself off, and point 18 is why (S2) — FIXED 2026-09-25

**Status:** fixed in `9f79591` (with 41). `keepalive.yml`, called by the refresh
every night, commits `docs/refresh_status.md` when `main` has been idle 30 days;
it has no schedule of its own because the rule disables every scheduled workflow
at once. Proven by a forced dispatch (run `36034421003` → bot commit `172e223`,
no CI/Pages run triggered). Unproven: GitHub does not document that a bot
commit resets the clock; every report says it does. Full write-up in backlog §7.

GitHub's documentation: *"In a public repository, scheduled workflows are
automatically disabled when no repository activity has occurred in 60 days."*
Until point 18, the nightly committed a database — the repository had activity
five nights a week whether or not the user touched it. Moving the database to a
release asset removed that, and nothing replaced it.

Evidence (as audited): last commit 2026-09-13 → 60 days → around 12 Nov 2026 if
nobody pushes. `refresh_data.yml` comment: *"that only ever affected
`schedule:` triggers, and there is none left to keep alive"* — written before the
cron was restored.

The workflow's own comment, left over from the period when the schedule had been
removed, argues that this cannot happen. That comment is now the thing most
likely to stop someone noticing.

**Fix as proposed:** a monthly job that touches the repository, or a monthly
commit of something real — a one-line refresh summary would do. Then fix the
comment: it describes a repository that no longer exists, including a commit step
that is now a release upload.

**Update 2026-09-23:** commits have continued, so the deadline moves with them —
60 days after the most recent push (≈ 2026-11-22 from `92a87ad`). The stale
comment is still present at `refresh_data.yml` lines ~175–188 (the ADR 4.4 block
mentioning "repo-committed file", `[skip ci]`, "none left to keep alive"). GitHub's
docs do not define "repository activity"; do not assume a bot push or an API call
counts without evidence.

### 27 · Nothing tells you the pipeline broke (S2) — FIXED 2026-09-25

**Status:** fixed in `bcb8450`. `notify` jobs (`if: failure()`) in `refresh_data.yml`
and `pages.yml` post the failed job + step and the run URL through
`alerting/pipeline.py` → `discord.send`; a `staleness` job reads the published
`health.json` and fails when prices are more than `STALE_AFTER_SESSIONS = 2` NYSE
sessions behind (chosen by replaying the run history). Unset webhook → one log line,
exit 0. CI green; Pages evaluated `notify` and skipped it. Backlog §7 has the detail.

The 2026-09-15 run went red at 02:57 UTC, published an empty database, and left
the site frozen. It was findable only by opening the Actions tab. The alerting
built in point 10 is about the *data* — rating changes, new formations on held
names — and it is also unconfigured. There is no alert whose subject is "the job
that makes all of this failed".

**Fix as proposed:** an `if: failure()` job on both workflows (`refresh_data.yml`,
`pages.yml`) that posts the run URL and the failed step to the same webhook, plus
a cheap external staleness check: if `health.json`'s newest price date is more
than a few trading days old, say so. The demo's own freshness strip already
computes this — it just has no way to tell anyone but a visitor who is already
looking.

**Update 2026-09-23:** still true. `ALERT_DISCORD_WEBHOOK_URL` is unset, so any
failure notice must follow the project convention: unset (or blank) → log one line
and exit 0, never an error. Staleness must be judged in **trading days** (US
holidays cluster on Mondays — see backlog §2 traps).

### 28 · Five finished features are idle, waiting on secrets only the user can add (S3)

**Status 2026-09-25: README fixed; the features still wait on the user.**
`gh secret list` → still only `SEC_EDGAR_USER_AGENT`, so nothing could activate.
The README's secrets table gained a *Where to get it* column, the webhook's new
notices, each unset path and today's state; `tests/unit/test_readme_secrets.py`
guards it. Caveat now stated there: Finnhub's short-interest field names are
unverified (`short_interest_client.py`), so that key may need a follow-up fix.

Evidence (as audited):
- `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` → the forward test ·
  `paper_trading_snapshots` **0 rows**
- `ALERT_DISCORD_WEBHOOK_URL` → "Alerting is not configured; sending nothing"
- `FRED_API_KEY` → the yield curve · live regime shows 10Y−2Y **—**
- `FINNHUB_API_KEY` → short interest · **0 rows**, and 503 warnings a run

The forward test is the one with a clock on it. Its whole argument is that the
record accrues in public from the day it starts, and it cannot be backdated —
every week without the two Alpaca keys is a week of track record that can never
exist. The live page reported `n_snapshots: 0`.

**Fix (user's to do):** all five go in Settings → Secrets and variables → Actions.
Never ask for, obtain or enter any of them.

**Update 2026-09-23:** `gh secret list` still shows only `SEC_EDGAR_USER_AGENT`.
The 2026-09-21 log still says *"Paper trading is not configured (no Alpaca
credentials); skipping"* and *"Alerting is not configured (no webhook URL);
sending nothing"*.

### 29 · The landing page downloads 1.25 MB of Plotly to draw one gauge (S2)

Evidence: Dashboard requests · `index-*.js` 81 KB gz · `dist-*.js` **1,250 KB**
gz (4.1 MB raw). What it draws there: the Market Regime dial — a semicircle, an
arc and a number.

Plotly is lazy-loaded per page, which is right, and Stock Detail genuinely needs
it — candlesticks, a fan chart, a radar. The Dashboard does not. It is the page a
recruiter opens on a phone, and it is fifteen times heavier than it needs to be.

**Fix:** draw the gauge as inline SVG — the interval whisker is already
hand-drawn, and this is a simpler shape. Plotly then loads only on the two pages
that plot real series. A trimmed Plotly bundle (candlestick, scatter,
scatterpolar) would cut those two as well. *(Not re-measured since 2026-09-16.
Note Dependabot PR #31 bumps plotly.js 4.1.0 → 4.1.1.)*

### 30 · The browser tab never changes as you move through the app (S2)

Evidence: open `/screener` → click Micron → URL `/stocks/MU` · h1 **MU — Micron
Technology** · tab title still **"QuantPulse — S&P 500 research"** ·
`document.title` occurrences in `frontend/src`: **0**.

Point 9's route pages give every stock a real title on a hard load — but
client-side navigation, which is how anyone actually browses, never updates it.
Three stocks in three tabs are indistinguishable; back/forward history is a column
of identical entries; a screen reader announces the same page name each time.

**Fix:** set the title in the router on every route change, from the same
`"NVDA — Nvidia"` string `scripts/emit_route_pages.py` already writes, so the two
can't drift.

### 31 · Every chart fails an accessibility check, and only the charts do (S2)

Evidence: axe-core 4.10 on the live site · Screener 0 · Track Record 0 · Glossary
0 · Stock Detail **3 × nested-interactive** (serious) · Dashboard **1** ·
`div[aria-label="Candlestick price chart for MU"]` — "element has focusable
descendants".

The wrapper is `role="img"`, which tells assistive tech "this is one picture, it
has no parts" — and then Plotly's modebar puts real focusable buttons inside it.
A keyboard user tabs into controls a screen reader has just promised aren't there.

**Fix:** `role="figure"` with the same label (allowed to have children) — or turn
the modebar off on charts that don't need it (the gauge, the radar) so the wrapper
really is a single image.

### 32 · 508 real pages, and no preview, no sitemap, no robots.txt (S2)

Evidence: `/stocks/NVDA/` → 200 with correct `<title>` and description ·
og:title / og:description / og:image / twitter:card → **none** · `/robots.txt`
**404** · `/sitemap.xml` **404**.

Point 9 made deep links real pages; the metadata that makes them *shareable*
didn't come with it. Pasted into Slack, LinkedIn or a message, a stock link
arrives as a bare URL or a title with no card.

**Fix:** emit og/twitter tags from `emit_route_pages.py` (it already has each
stock's title and description), plus one static preview image (the mark and the
name). Generate `sitemap.xml` from the same route list, and a three-line
`robots.txt` pointing at it.

### 33 · Half the product has no public link (S3) — needs the user's call

The pitch is two functions: a screener and a portfolio manager. Function 2 — FIFO
lots, three optimisers, correlation clusters, VaR, the rebalancing trade list, the
history panel — is reachable only by cloning the repo and running `./run.sh`.
Streamlit Community Cloud is still unconnected; the README carries the click list
for it.

**Your call:** either connect Community Cloud (one OAuth sign-in the user must do;
the repo is prepared), or give the SPA a client-side portfolio over the
pre-rendered data, the way the watchlist already works in `localStorage`. The
first is minutes and gives the real thing; the second is real work and gives a
demo that can't be taken down.

### 34 · A +90.7% twenty-day forecast is published wearing a graded badge (S1) — needs the user's call

Evidence: SNDK · h=20 · gbr · point return **+90.7%** · hit rate 52.4% vs naive
49.7% · 34 windows · STX +34.5% and WDAY +34.4% show **identical** 52.4 / 49.7 /
34 · SNDK history 395 bars, $36.00 (2025-02-13) → $1,692.59 (2026-09-10).

The hit rate is the model's, pooled across the universe — the same three figures
sit on every GBR row at that horizon. Beside +90.7% it reads as evidence for
*that number*, which it isn't. Point 5 sorted horizons by whether they are graded;
within a graded horizon there is still no bound on what a single extrapolation may
claim, and a name up 47× in nineteen months is where a gradient-boosted return
model runs away.

**Your call:** a per-symbol plausibility bound — abstain, or mark the row, when
the point estimate sits outside that symbol's own historical distribution of
h-day returns. Suppressing is cleaner; flagging is more honest to the project's
"disclose, don't delete" rule.

### 35 · Two of the four horizons can never be graded, and are recomputed every week (S2) — needs the user's call

Evidence: `MIN_GRADED_WINDOWS = 30` distinct evaluation windows · h=63 → **9**
windows available · h=252 → **0** · both published ungraded every run · 30
windows would need ≈ 7.5 years at h=63 and ≈ 30 years at h=252 · demo price
history begins 2023-05-01. (Confirmed again in the 2026-09-21 log: *"Not
publishing the … hit rate at h=63: 9 graded window(s), below the 30 needed"*.)

This isn't a threshold that will be met later; on three years of history it
cannot be met at all. The weekly job spends runner time producing two horizons
whose only destiny is the disclosure drawer — and those rows carry the largest
numbers on the page.

**Your call:** drop h=252 (and perhaps h=63) from the weekly run and say in the
methodology that the data cannot grade them — or grade them against the deeper
history in the local `quantpulse.db` (1972–2026, enough windows at both horizons).
The second is more work and would make the rows real.

### 36 · "Institutional ownership: 169 days ago" reads as neglect, and is correct (S4)

It is the most recent quarter SEC has published, pinned to the 31 March window
end, and it sits in the freshness strip beside prices measured in days. A reader
who doesn't know 13F's publication lag sees the app admitting to six months of
staleness.

**Fix:** label quarterly sources by their period, not their age: "Q1 2026 filings
— the newest SEC publishes". The row is computed server-side, so this is a
sentence, not a feature.

### 37 · One missing key writes 503 identical warnings (S4)

Evidence: `AOS: short_interest: FINNHUB_API_KEY is not set` × **503**, once per
ticker; FRED: 6 more of the same shape.

The design intent is written down — an unset key should log one line and do
nothing. Per-symbol it becomes the loudest thing in a three-hour log, and the run
that lost the database had its real errors in the middle of it.

**Fix:** check the credential once, before the loop, and skip the step with a
single line. *(Still present in the 2026-09-21 log.)*

### 38 · The documented counts have drifted again, in several directions (S4)

Evidence (as audited): migration files on disk 16 · README says 13 ·
ARCHITECTURE.md says 12 · backlog says 14.

Four passes have re-counted these by hand and they have drifted every time —
evidence that hand-counting is the wrong mechanism.

**Fix:** generate the "by the numbers" figures — migrations, tables, endpoints,
tests, glossary terms — and have a test fail when a documented figure disagrees
with the repository.

**Update 2026-09-23 (re-counted):** migration files **16**; README "13 Alembic
migrations"; ARCHITECTURE "12 revisions"; `docs/IMPROVEMENT_BACKLOG.md` header
still says **14 migrations and 1,824 tests** (actual: **1,895 passed, 1
skipped**); `src/quantpulse/api/main.py` has **15** `@app.get` routes against a
documented 14 — reconcile before trusting either number.

### 39 · `./run.sh` doesn't ship the one line that keeps Streamlit alive here (S4)

Navigating a few pages segfaults the Streamlit process inside libarrow's bundled
mimalloc — diagnosed on this machine, with `ARROW_DEFAULT_MEMORY_POOL=system`
established as the workaround. That line is in the notes and in no file. Anyone
following the README's "run the full app" path can hit a blank page with an idle
server and no error.

**Fix:** export it in `run.sh` with a comment saying what it works around.

### 40 · Dependabot PRs, including the libraries that have broken the charts before (S4)

As audited: vite 8.2.2 → 8.3.0, react 19 and react-dom, red for finding 21's
reason. The standing rule applies to vite: `tsc`, the build and CI were all green
through both previous chart breakages — load a chart page in a real browser and
read the console before merging.

**Update 2026-09-23 (current PR state):**

| PR | change | checks |
|---|---|---|
| #28 | vite 8.2.2 → 8.3.0 | test **pass**, frontend **pass** (re-run after the restore) |
| #29 | react-dom + @types/react-dom | test **fail** (stale 2026-09-15 run, broken-database era — re-run), frontend pass |
| #30 | react + @types/react | test **fail** and **frontend fail** — the frontend job never reads the database, so this one may be a genuine break; investigate |
| #31 | plotly.js 4.1.0 → 4.1.1 | both pass — charting library: browser check before merging |

Merge order matters (they all touch `package.json`/lockfile): rebase with
`gh pr comment N --body "@dependabot rebase"` after each merge.

### 41 · The refresh workflow's comments describe a repository that no longer exists (S4) — FIXED 2026-09-25

**Status:** closed by finding 26's fix (`9f79591`, plus `pages.yml` in the follow-up
docs commit): the blocks now describe the release asset, the restored cron and the
60-day rule, and `test_no_workflow_still_describes_a_committed_database` refuses
the stale phrases in every workflow file.

They explain why the database is committed, what `[skip ci]` is for on a
data-only commit, and that there is no schedule left to keep alive. All three
were true weeks ago. The comments in this project are how a future session learns
why something is the way it is — which is exactly why stale ones are expensive:
finding 26 is hiding behind one of them.

**Fix:** rewrite the block around the publish step to describe the release asset,
the restored cron, and the 60-day rule. **Update 2026-09-23:** still present at
`refresh_data.yml` lines ~175–188 and ~257. (Overlaps 26 — fixing 26 should
rewrite this block.)

---

### 42 · The publish gate held the demo hostage to the market (S2) — FIXED 2026-09-25

Found while starting on 26. Scheduled run `35937083808` (2026-09-24): refresh
green, `publish / build` red on one static test — it asserted the literal text
"Risk On", and the 2026-09-23 regime came out *neutral* (55.7). The demo stayed
on 2026-09-22 data and would have until the market turned risk-on. Fixed in
`f6f25a7`: the gate asserts the label the generated `regime__limit-90.json`
holds. Reproduced on the published database first; mutation-checked two ways;
live `health.json` moved to 2026-09-23 on the next publish.

## 4. Additional open items (not numbered in the audit)

- **A. The regime index's macro-tone input fails.** `gdelt_client.fetch_tone_timeline`
  is GDELT too; the 2026-09-21 run logged *"Failed to fetch GDELT macro tone"*.
  Google News has no tone timeline. `describe_regime_coverage` already discloses a
  missing input. No fix designed yet.
- **B. Test-harness quirk:** inside `TestAlerting._driven_run` (patched
  `refresh_data.datetime` mock), the backtest step fails with
  `isinstance() arg 2 must be a type` in `_coerce_date`. Test-only — production is
  unaffected, and no test asserts on it — but it adds a spurious
  `failed step(s): backtest` to those harness runs.
- **C.** `docs/IMPROVEMENT_BACKLOG.md`'s header statistics are stale (part of 38).

## 5. Technical judgement calls made this session (the user may overrule)

None were asked as methodology questions; each is recorded so it can be revisited.
- Tier-2 Google News articles are scored on **titles only** (23).
- Tier-2 classification cap **8** per basket, tier-1 cap **500** (22, 23).
- 13F skips the download when the rule-predicted quarter is stored; mid-quarter
  index additions wait for the next window (24).
- **No** longer SQLite busy timeout (25).
- CI reads a **pinned** fixture; current data is gated only by the publish
  workflow's static-site suite (21).

Methodology decisions from earlier sessions are in memory
(`quantpulse_review_passes.md`) — do not re-ask them.

## 6. Traps learned this session (also in `docs/IMPROVEMENT_BACKLOG.md` §7)

1. A green integrity check says the file is well-formed, not that it is the right file.
2. A literal list of files in a guard stops covering whatever is added next.
3. Match commands in their `run:` form — the comment explaining a bug contains the same words.
4. `${{ runner.* }}` in job-level `env` is a whole-file parse error; `yaml.safe_load` can't see it.
5. Patch where a name is *used*, not where it is defined.
6. A logger that logs its own reason makes `caplog.text` assertions vacuous — assert the closing line.
7. A fallback triggered by an *empty* primary result reaches every test that mocks the primary as empty. When CI suddenly takes 3× longer, believe it.
8. `step()` swallows exceptions, which hides live network calls in tests — a guard must record calls and assert at teardown.
9. zsh doesn't word-split unquoted parameters (see §1).
10. SQLite has one writer; never open an own-session write inside an open shared session.
11. A lock can be hiding an ordering bug — ask what would happen without it.

---

## 7. Prompt for the next session (fixes 26–28)

Paste this as the first message of a new session:

```text
You are continuing work on QuantPulse (/Users/marlenmelis/Documents/quantpulse).

Read first, in full, before doing anything else:
1. docs/SESSION_HANDOFF_2026-09-23.md — the complete record of the last session.
   Section 1 (environment facts) is mandatory: /usr/bin/git is broken (use
   /opt/homebrew/bin/git; for gh, export PATH=/opt/homebrew/bin:$PATH or pass
   -R MarlenMM/quantpulse), the scratchpad is not durable, zsh does not
   word-split unquoted parameters, and never run two pytest processes at once.
2. docs/IMPROVEMENT_BACKLOG.md §2 (traps) and §7 (points 19–25).
3. Your memory files for this project.

Then fix findings 26, 27 and 28 from the handoff's section 3, in that order.
Commit AND push each finding separately and automatically (the user's standing
convention), ending each commit message with the attribution line your system
reminder specifies.

Method — the same one that fixed 19–25:
- Measure and reproduce before changing anything; every fix starts from a number
  or an observed failure.
- Look at the running thing: run logs (gh run list / gh run view --log), the
  live site, the real database — not only unit tests.
- Write the test before the fix and watch it fail. Then mutation-check every new
  guard: revert or break the fix, confirm the specific test fails with the right
  message, restore. A mutation must compile (py_compile) and the harness must
  print the real pytest tail, with arguments passed as "$@".
- Keep tests hermetic: no live network (GDELT, Google News, SEC, Nasdaq, Discord,
  GitHub). If a new code path can reach the network, check that existing tests
  don't start reaching it.
- Run the full suite before each commit, then watch CI on the pushed commit.
- Say plainly what was not done and why.

Finding 26 — the schedule will be auto-disabled after 60 days without repository
activity:
- Verify the current risk: date of the most recent commit, and exactly what the
  refresh_data.yml comment block (~lines 175–188, and ~257) claims.
- Research what GitHub counts as "repository activity" for this rule before
  choosing a mechanism; do not assume a bot push or an API call counts. Prefer a
  mechanism whose effect you can verify (e.g. the workflow-enable API, or a real
  periodic commit of something meaningful), and state its limits.
- Implement it (a small scheduled workflow is the likely shape), with tests in the
  style of tests/unit/test_refresh_workflow.py and test_deploy_requirements.py.
- Rewrite the stale comment block in refresh_data.yml so it describes the release
  asset, the restored cron and the 60-day rule (this closes finding 41 too — say
  so in the handoff).

Finding 27 — nothing tells anyone when the pipeline breaks:
- Add failure notification to refresh_data.yml and pages.yml (an if: failure()
  job or step) that reports the run URL and what failed, through the existing
  alerting code in src/quantpulse/alerting/ and the ALERT_DISCORD_WEBHOOK_URL
  secret. That secret is UNSET: the unset/blank path must log one line and exit 0
  (the project's convention — see Settings.alerting_configured()). Never log or
  echo the webhook URL.
- Add a staleness check: if the published health.json's newest price date is more
  than N trading days old (use the market calendar, not weekdays — US holidays
  cluster on Mondays), alert. Decide where it runs and justify N from the data
  (STALE_AFTER_DAYS in app/lib/format.py is the existing vocabulary).
- Test both without the network: the workflow structure, the unset-secret path,
  and the staleness rule across a holiday week. Mutation-check.

Finding 28 — five features wait on secrets only the user can add:
- Never ask for, obtain, or enter any key or webhook URL.
- Re-verify with gh secret list which of FRED_API_KEY, FINNHUB_API_KEY,
  ALERT_DISCORD_WEBHOOK_URL, ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY are set.
- For each: confirm from the latest run log what the unset path does, confirm the
  README's secrets table names it exactly and says what it unlocks and where it is
  obtained, and fix the table if it is incomplete or wrong.
- If the user has added any since 2026-09-23, verify the feature actually
  activates on the next scheduled run (e.g. paper_trading_snapshots rows appear,
  the regime's 10Y−2Y input is non-null, short_interest rows appear, an alert is
  posted) and report the evidence.
- Finish with a short, exact list of what the user still has to do.

After each finding: update docs/IMPROVEMENT_BACKLOG.md (a new §7 entry), update
the status of that finding in docs/SESSION_HANDOFF_2026-09-23.md, and update the
findings artifact https://claude.ai/artifact/HokUNu4qH7eg5jgxeTcByL (read it
first with the Artifact tool, strip the host wrapper as described in the handoff,
mark the finding Fixed, republish with url).

If a genuine methodology or design decision comes up, ask — the user engages with
those. Do not re-ask decisions already recorded in memory. When offered a subset
or the complete scope, the user chooses the complete scope.
```
