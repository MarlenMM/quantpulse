# Contributing

QuantPulse is primarily a solo portfolio project, built incrementally against
the roadmap in [PROJECT_PLAN.md](PROJECT_PLAN.md). That said, bug reports,
small fixes, and well-scoped pull requests are genuinely welcome — this guide
is for anyone who wants to send one.

## Before you start

- **Bug fix or small, self-contained feature?** Open a PR directly; no need
  to ask first.
- **Anything larger** (a new page, a new data source, a change to the
  scoring methodology) — please open an issue first describing what you want
  to do and why. [PROJECT_PLAN.md](PROJECT_PLAN.md) is the design doc this
  project follows; a change that conflicts with a documented decision there
  (an ADR in Section 4, the methodology in Section 7, the pitfalls in Section
  22) needs that discussion before code, not after.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) first if you're not sure where
  something belongs — the layering it describes (`analysis/` never imports
  from a UI, etc.) is enforced by convention, not by tooling, so a PR that
  crosses it will get asked to move code around.

## Dev setup

See the root [README's Quickstart](README.md#quickstart) — `uv sync`, copy
`.env.example` to `.env`, `alembic upgrade head`, `uv run pytest`. Nothing
in `.env` is required to run the test suite or migrations; only the
ingestion clients need real (free-tier) API keys.

Enable the git hook so lint/format/type-check issues get caught before you
even open a PR:

```bash
uv run pre-commit install
```

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src scripts
uv run pytest
```

All four need to pass clean — this is exactly what CI runs on every push and
PR (`.github/workflows/ci.yml`), so running it locally first means no
surprises. For the `frontend/` React app: `npm run typecheck` (no test suite
exists there yet).

## Tests

New behavior needs a test. Match the existing shape:

- **`tests/unit/`** — one file per `src`/`app` module, fixed/hand-checked
  inputs. This is most new tests.
- **`tests/integration/`** — ingestion clients against recorded fixtures
  (never live APIs — Section 19's rate-limit discipline applies to CI too),
  or anything that needs a real (temp) database.
- **`tests/property/`** — [Hypothesis](https://hypothesis.readthedocs.io/),
  reserved for the modules where a property genuinely strengthens confidence
  beyond another hand-picked example (normalization math, conservation laws,
  bootstrap mechanics). Don't reach for this by default — most new code is
  well served by a unit test with 2-3 well-chosen examples.

Look at a neighboring test file in the same directory before writing a new
one; this project is consistent about naming and structure on purpose.

## Code style

- Formatting/linting is `ruff` (config in `pyproject.toml`), not up for
  debate in review — run `ruff format .` and move on.
- Type hints everywhere in `src/`/`scripts/` (mypy-checked in CI); `tests/`
  is intentionally not type-checked (see the mypy config's scope).
- Default to **no comments** — code should read clearly from naming and
  structure. A comment is worth adding only when it explains a *why* that
  isn't obvious from the code itself (a subtle invariant, a workaround for a
  specific bug, a design decision someone will second-guess later). This
  codebase leans heavily on that pattern in its docstrings; match it.
- Don't add abstractions, config flags, or error handling for cases that
  can't happen. Section 22 ("Methodological Pitfalls to Avoid") is required
  reading before touching anything in `analysis/` — look-ahead bias and
  silent normalization bugs are the two easiest ways to quietly corrupt this
  project's actual output.

## Reporting a bug

Open an issue with: what you expected, what happened, and — if it's a
data/scoring issue rather than a crash — the exact symbol and date, since
almost everything here is point-in-time and reproducing "what the algorithm
said about AAPL on a given day" needs that context.
