# Architecture

A code-first tour for a contributor (including future-you) who wants to
understand how QuantPulse is actually laid out, without needing to run the
Streamlit app first. For the *why* behind these choices — data source
trade-offs, scoring methodology, the roadmap — see [PROJECT_PLAN.md](PROJECT_PLAN.md);
this file only covers the *what* and *where*. The system-level diagram lives
in [README.md's Architecture section](README.md#architecture) rather than
duplicated here.

## The one rule that shapes everything else

**`src/quantpulse/analysis/` (and `portfolio/`, `news_intelligence/`) never
import from `app/`, `frontend/`, or `src/quantpulse/api/`.** The analysis
engine is a plain Python library that takes DataFrames/dataclasses in and
returns DataFrames/dataclasses out — no Streamlit, no FastAPI, no UI
framework anywhere in its import graph. That's what makes two independent
front ends possible over one engine (`app/` and `frontend/` +
`src/quantpulse/api/`) without duplicating a single line of scoring,
forecasting, or risk logic, and it's asserted by the module layout itself,
not just a convention someone could quietly violate.

## Layers, outside in

```
external APIs  →  ingestion/  →  storage/ (SQLite)  →  analysis/ · news_intelligence/ · portfolio/  →  llm/ (optional)  →  app/ · api/+frontend/
```

| Directory | What lives here | Imports from |
|---|---|---|
| `src/quantpulse/ingestion/` | One client per external data source (`yfinance_client.py`, `finnhub_client.py`, `edgar_client.py`, `edgar_13f_client.py`, `fred_client.py`, `gdelt_client.py`, `reddit_client.py`, `news_client.py`, `wikipedia_client.py`, `options_client.py`, `short_interest_client.py`, `historical_constituents_client.py`, `economic_calendar.py`), plus the shared `rate_limit.py` / `circuit_breaker.py` / `cache.py` / `http.py` infrastructure every client uses. Pure I/O — no DB access. | Nothing else in `src/quantpulse/` |
| `src/quantpulse/storage/` | `models.py` (SQLAlchemy ORM, 23 tables), `migrations/` (Alembic, 8 revisions), `persistence.py` (point-in-time read/write helpers — the only place outside a migration that touches a `Session` directly for most callers), `db.py` (engine/session factory) | `config.py` |
| `src/quantpulse/analysis/` | The actual quant methodology, as pure functions: `technical.py`/`patterns.py` (indicators + geometric pattern detection), `fundamental.py` (sector-relative ratios), `analyst_consensus.py`, `smart_money.py`, `macro.py`, `scoring.py` (the composite-scoring core), `forecasting.py`, `backtest.py`, `risk.py`, `clustering.py`, `investor_profiles.py` | Nothing outside `analysis/` — every function takes plain DataFrames/dataclasses in, never a DB session; the only cross-module imports are within this package (e.g. `risk.py` re-exports `backtest.sharpe_ratio` rather than reimplementing it) |
| `src/quantpulse/news_intelligence/` | `entity_extraction.py`, `event_classifier.py`, `thematic_mapping.py`, `sentiment.py` (FinBERT), `market_regime.py` — the three-tier (company/industry/market) news pipeline that feeds `scoring.py`'s sentiment and industry/macro categories | Nothing outside this package either — pure functions over article DataFrames, same discipline as `analysis/`. `scripts/refresh_data.py` is what actually wires ingestion output into these |
| `src/quantpulse/portfolio/` | Holdings-specific logic that isn't general risk math: `transactions.py` (FIFO tax lots), `optimization.py` (MPT/HRP/Black-Litterman), `rebalancing.py` (trade-list generation) are all pure, self-contained modules, same discipline as `analysis/`; `recommendations.py` (Add/Trim/Hold/Sell + concentration/sector-gap analysis) and `holdings.py` (ADR 4.5's session-vs-sqlite backend switch) are the two files that actually touch anything else | `recommendations.py` → `analysis/scoring.py`; `holdings.py` → `storage/models.py`, `config.py` |
| `src/quantpulse/llm/` | `providers.py` (Gemini/Groq/Ollama abstraction, ADR 4.3), `narrative.py` (grounded explanation prompts), `chatbot.py`. Every provider shares one grounding instruction so swapping backends can't change the model's rules; the whole layer degrades to `None` with zero errors when no key is configured. | `ingestion/http.py` (for the HTTP calls), nothing from `analysis/` — it only ever narrates numbers it's handed |
| `src/quantpulse/api/` | `main.py` (FastAPI app, 12 endpoints, all GET — asserted by a test over the OpenAPI schema), `schemas.py` (Pydantic response models) | `storage/persistence.py` — the *same* readers `app/` uses, so the two front ends can't disagree |
| `src/quantpulse/glossary.py`, `config.py` | UI-agnostic shared content (63 term definitions) and settings (pydantic-settings, reads `.env`) | — |
| `app/` | The Streamlit app: `Home.py` + `pages/1_Screener.py` … `6_Glossary.py`, `lib/` (thin `@st.cache_data` wrappers over `storage/persistence.py`, plus `charts.py`/`format.py`/`search.py`/`glossary.py` display helpers — `lib/` never contains scoring logic, only presentation) | `storage/persistence.py`, `quantpulse.analysis.*`, `quantpulse.llm.*` |
| `frontend/` | React 19 + TypeScript SPA (Vite), 5 pages, a small custom History-API router (~110 lines) (no react-router — see the router file's own comment for why), Plotly lazy-loaded per-page | `src/quantpulse/api/` over HTTP only — no Python imports, obviously |
| `scripts/` | `seed_initial_data.py` (one-time, survivorship-bias-aware historical backfill) and `refresh_data.py` (the nightly incremental job: concurrent fetch phase via `ThreadPoolExecutor`, then a serial write phase — see the module's own docstring) | Everything above |
| `tests/` | `unit/` (one file per `src`/`app` module, fixed-input assertions), `integration/` (ingestion clients against fixtures, `refresh_data.py`/`seed_initial_data.py` end-to-end against a temp SQLite DB), `property/` (Hypothesis — coverage-renormalization, FIFO conservation, bootstrap-CI mechanics, and other invariants that should hold for *any* valid input, not just one hand-picked example) | Mirrors `src`/`scripts`/`app` |

## Design principles worth knowing before you change anything

These are covered in depth in [PROJECT_PLAN.md Section 22](PROJECT_PLAN.md#22-methodological-pitfalls-to-avoid);
the short version, because they're easy to accidentally break in a
refactor:

- **Point-in-time discipline.** Every scorer/forecaster/backtester reads only
  data dated on or before its `as_of` date. This is what makes the backtest
  honest rather than accidentally clairvoyant — it's enforced at read time
  (`storage/persistence.py`'s point-in-time readers), not by convention.
- **Coverage-based renormalization, never a phantom zero.** When a category
  or sub-score has no data for a symbol, it drops out of that symbol's
  weighting entirely (`scoring.build_composite`, the Screener's client-side
  `reweight`, `recommendations.py`) rather than being counted as zero and
  quietly dragging the result down.
- **Abstain rather than fabricate.** A statistic computed from too little
  data returns `None`, not a confident-looking number — VaR's minimum tail
  size, Sharpe/Sortino's degenerate-variance guard, the bootstrap's minimum
  observation floor all follow this.
- **A relative ranking is not an absolute judgment.** The default rating
  scheme ranks stocks against each other, so it always produces some "Strong
  Buy"s even in a falling market — the Market Regime Index exists specifically
  so a reader can tell the two apart.

## Testing philosophy

Three layers, each catching a different class of bug: `tests/unit/` pins
behavior against fixed, hand-checked inputs; `tests/integration/` proves the
pieces actually wire together (real ingestion fixtures, a real temp SQLite
DB, real `ThreadPoolExecutor` concurrency); `tests/property/` generalizes the
example-based tests' hand-picked cases across randomly generated inputs for
the modules where that meaningfully strengthens confidence — normalization
math, FIFO arithmetic conservation, bootstrap CI mechanics — without forcing
it onto code that doesn't benefit from it. `uv run pytest` runs all three.
