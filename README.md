# QuantPulse

A self-hosted, $0-cost stock research & portfolio-management engine. Statistics and ML do the ranking/forecasting; a free-tier LLM only narrates results that already exist.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design doc (architecture, data sources, scoring methodology, roadmap).

**Status:** Analysis engine in progress (data layer, technical/fundamental/analyst/news/smart-money signals, and the market-regime index are built; composite scoring, forecasting, portfolio tools, and the UI are still to come). Nothing here makes trade or investment decisions.

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
```

## Development

- Lint/format: `uv run ruff check .` / `uv run ruff format .`
- Type-check: `uv run mypy src`
- Enable git hooks (runs ruff + mypy on every commit): `uv run pre-commit install`

## Project layout

See [Section 14 of the plan](PROJECT_PLAN.md#14-project-folder-structure) for the full intended structure. The `analysis/` package never imports from `app/`, so the analysis engine stays UI-agnostic.

## Disclaimer

Educational/research tool. Not financial advice. Not a registered investment advisor.
