# QuantPulse

A self-hosted, $0-cost stock research & portfolio-management engine. Statistics and ML do the ranking/forecasting; a free-tier LLM only narrates results that already exist.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design doc (architecture, data sources, scoring methodology, roadmap).

**Status:** Analysis engine, Streamlit UI and the React + FastAPI stretch front end are all built — data layer, technical/fundamental/analyst/news/smart-money signals, the market-regime index, composite scoring, forecasting + backtesting, portfolio risk/optimization/rebalancing tools, the optional LLM narration layer, and all six app pages. Accessibility/glossary polish is the remaining pass. Nothing here makes trade or investment decisions.

The LLM layer is optional by design: with no API key set (or `LLM_ENABLED=false`), every number the app computes is still produced and displayed — you just don't get the plain-English paragraph next to it.

## Quickstart

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if you don't have it
brew install uv

# 2. Install dependencies (creates .venv automatically, pinned to Python 3.12)
uv sync

# 3. Configure environment
cp .env.example .env
# edit .env with your own API keys (all free-tier; see .env.example for where to get each one)

# 4. Apply database migrations (alembic.ini lives at the repo root)
uv run alembic upgrade head

# 5. Run the test suite
uv run pytest

# 6. Launch the app (works against an empty database — it tells you how to populate it)
uv run streamlit run app/Home.py
```

## Two front ends, one engine

The analysis engine never imports from a UI (Section 14), which is what makes
two front ends possible without duplicating a line of analysis:

| | Streamlit (`app/`) | React + FastAPI (`frontend/` + `src/quantpulse/api/`) |
|---|---|---|
| Role | The full app, including the Portfolio Manager | Stretch goal (ADR 4.1) — showcases the UI-agnostic engine |
| Hosting | Streamlit Community Cloud, free and always-on | Render/Fly.io + Vercel free tiers |
| Portfolio management | Yes | No — the API is read-only by design (see below) |

```bash
# React + FastAPI (two terminals)
uv run uvicorn quantpulse.api.main:app --reload   # API on :8000, docs at /docs
cd frontend && npm install && npm run dev          # SPA on :5173, proxies /api
```

**The API is deliberately read-only.** Portfolio state is per-user and ADR 4.5
splits it between a browser session and a local SQLite file; neither maps onto
a stateless REST API without the authentication the single-user MVP explicitly
doesn't have (Section 18). Portfolio management therefore stays in Streamlit,
where its storage backends already live.

### Populating it with real data

```bash
uv run python scripts/seed_initial_data.py   # one-time historical backfill (slow)
uv run python scripts/refresh_data.py        # nightly incremental refresh + scoring
```

## Pages

| Page | What it shows |
|---|---|
| **Dashboard** | Market Regime Index gauge, top-ranked names, what changed since the last refresh, market-moving Tier-2/3 news |
| **Screener** | The ranked, filterable table, with sliders that re-weight the seven score categories client-side and a 2–4 ticker Compare mode |
| **Stock Detail** | Price chart + patterns, sub-score radar, forecast fan chart with each model's own hit-rate, analyst-vs-algorithm, news feed |
| **Portfolio & Watchlist** | FIFO tax-lot positions, risk dashboard (vol/Sharpe/Sortino/beta/VaR/correlations), Add-Trim-Hold-Sell guidance, concentration + sector-gap warnings |
| **Backtest / Track Record** | Sharpe and CAGR with bootstrap confidence intervals, benchmark comparison, stated cost assumptions |
| **Settings / About** | Data freshness per dataset, pipeline health, configuration, methodology and limitations |

## Live Demo & Deployment

`.github/workflows/refresh_data.yml` keeps a small, repo-committed demo
database (`quantpulse_demo.db` — distinct from your own local
`quantpulse.db`, see `.gitignore`) up to date every weeknight, so Streamlit
Community Cloud can serve a public demo without the live app needing any API
keys of its own (ADR 4.4). Two one-time steps, both manual — neither can be
scripted from outside your own accounts:

1. **Seed the demo database once**, locally, with your own free-tier keys:
   ```bash
   DATABASE_URL=sqlite:///./quantpulse_demo.db uv run alembic upgrade head
   DATABASE_URL=sqlite:///./quantpulse_demo.db uv run python scripts/seed_initial_data.py
   git add quantpulse_demo.db && git commit -m "Seed the live-demo database" && git push
   ```
   After this, the nightly workflow takes over — it commits an updated
   `quantpulse_demo.db` back to `main` every weeknight (see the workflow
   file's own comments for why that's safe now that ADR 4.5's session-vs-
   sqlite split exists). Also add `FINNHUB_API_KEY` / `FRED_API_KEY` /
   `SEC_EDGAR_USER_AGENT` as **repo secrets** (Settings → Secrets and
   variables → Actions) so the nightly job itself can fetch fresh data.

2. **Connect the repo at [share.streamlit.io](https://share.streamlit.io)**:
   point it at this repo, branch `main`, main file `app/Home.py`, then set
   these under the *app's* own Settings → Secrets (a `.streamlit/secrets.toml`-
   format screen, separate from the GitHub repo secrets above):
   ```toml
   DATABASE_URL = "sqlite:///./quantpulse_demo.db"
   PORTFOLIO_BACKEND = "session"
   ```
   `PORTFOLIO_BACKEND=session` is the part that actually matters (ADR 4.5):
   it keeps every visitor's portfolio entries in their own browser session
   instead of the shared file, so the committed demo database only ever
   holds the same public screener/market data every visitor already sees.
   An LLM key (Section 4.3) is optional — the app runs fine without one.

Streamlit Community Cloud auto-redeploys on every push to `main`, so each
night's data commit above refreshes the live app automatically — no
separate redeploy step to remember.

## Development

- Lint/format: `uv run ruff check .` / `uv run ruff format .`
- Type-check: `uv run mypy src`
- Enable git hooks (runs ruff + mypy on every commit): `uv run pre-commit install`

## Project layout

See [Section 14 of the plan](PROJECT_PLAN.md#14-project-folder-structure) for the full intended structure. The `analysis/` package never imports from `app/`, so the analysis engine stays UI-agnostic.

## Disclaimer

Educational/research tool. Not financial advice. Not a registered investment advisor.
