# QuantPulse — improvement backlog and session handoff

An 18-point audit was run on **2026-09-03** against the live demo, the committed
demo database, and the pipeline's own upstream sources. Points **1–12 are fixed
and live**; **13–18 remain**. This file is the handoff: what was done, what is
left, and the things a new session would otherwise rediscover the hard way.

**State at time of writing:** 1,740 tests, 14 Alembic migrations, CI and Pages
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
uv run pytest                 # 1,740 tests
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

### Two things from those fixes that still need watching

1. **Point 3 is code-complete but not yet visible.** `industry_macro` still
   covers **19/503** because the sector-tagged Tier-2 news has not been fetched
   yet — GDELT rate-limited the dev machine, and the fix landed after the last
   weekly run. **Monday's scheduled run is the test.** Check on Tuesday:
   `select count(distinct matched_theme) from news_events where tier=2` should
   be ~17, not 5, and `industry_macro_raw` should be non-null for ~all 503.

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

Ordered as the audit ranked them. 13–14 are unbuilt features; 15–18 are
methodology and hygiene. The numbering is the audit's and is kept stable, so
13 stays 13 now that 10-12 are done.

### 13. The Portfolio Manager has no sense of time

FIFO tax lots, three optimisers, correlation clusters, VaR, concentration and
sector-gap warnings — all position-level, all as of now. Missing: a P/L-over-time
chart, a comparison of *your* portfolio against a benchmark, and dividend
tracking. The transaction ledger needed for all three is already stored.

### 14. React SPA is missing three things Streamlit has

CSV export (Streamlit has it on three pages), Compare mode, and the watchlist.
The Portfolio Manager's absence is a deliberate architectural decision (ADR 4.5,
the API is read-only) and should stay one — these three are not, they are just
unbuilt, and the SPA is the front end most visitors ever see.

### 15. The seven-category composite is, in practice, a price-trend ranking

Spearman correlation with the final composite, measured 2026-09-05 on all 503
balanced scores:

```
technical        0.716      fundamental      0.434
momentum         0.633      sentiment        0.335
industry_macro   0.566 *    smart_money      0.198
                            analyst          0.159

technical <-> momentum      0.646   (the two dominant inputs are largely one signal)
* over 19 covered names only; see the point-3 watch item above
```

The stated weights are honoured arithmetically — that was fixed in an earlier
pass. But four of seven categories barely move the ranking. Two honest options:
**surface it** (an "effective weights" panel in Settings showing each category's
realised influence, which is an unusual and creditable thing for a screener to
show), or **address it** (orthogonalise momentum against technical so the two
stop double-counting the same trend). Prefer surfacing first — it is the
cheaper, more informative half, and it makes the second measurable.

### 16. Three open Dependabot PRs, one of them the twice-breaking library

```
#23  plotly.js-dist-min  3.7.0 -> 4.0.0     <- major bump
#24  @types/react-dom    19.2.4 -> 19.2.5
#25  @vitejs/plugin-react 6.1.0 -> 6.1.1
```

Plotly has silently blanked every chart in this app **twice** — once on Vite
7→8's CJS interop, once on react-plotly.js 2→4 shipping a `forwardRef` object —
and both times `tsc`, the build and CI were green throughout. The standing rule:
after any charting or bundler upgrade, **load a chart page in a real browser and
read the console**. Merge the two routine ones first; do plotly on its own.

### 17. The Screener renders all 503 rows, and there is no skip link

525 focusable elements on one page, every row in the DOM whether visible or not.
Fine on a laptop, heavy on the phone that point 7 was about. Landmarks and image
alt text are in good shape; a skip-to-content link is the obvious gap.
Virtualising or paginating the table helps both at once.

### 18. The repository grows by a 60 MB binary per refresh

```
quantpulse_demo.db committed revisions   33
local .git                              143 MB
GitHub's packed size                     57 MB
```

The trade — repo size against a reader's first five minutes — is the right one,
and a fresh clone is still fine. But the refresh is back on a daily schedule
(point 6), so the number now moves five times a week. Worth deciding **now**
whether the demo database eventually moves to a release asset or an orphan
branch, before the decision is forced.

---

## 5. How these fixes have been done

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
