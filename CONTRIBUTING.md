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

## Design

Both front ends share one system: the tokens live in
`frontend/src/styles.css` and are said again in Streamlit's vocabulary in
`.streamlit/config.toml`. Read the header comment in the first before changing
anything visual — it explains why each decision is what it is.

The short version, as constraints. These are the defaults the project has
already been through once and does not want back:

- **No gradients, glows, or glass panels.** Not in a hero, not behind a card,
  not as decoration on a status.
- **One accent, and it means "interactive".** Green, amber and red are the
  rating and severity vocabulary and are never used for anything else — not for
  a chart series, not for a nice-looking border. A moving average once came out
  green on a candlestick chart for exactly this reason.
- **Never colour alone.** Every rating and status carries a glyph and a word;
  colour is the third, redundant channel. Series in a figure are separated by
  dash pattern before hue.
- **Three type roles, not one.** Serif for display, the UI sans for chrome,
  monospace with `tabular-nums` for every figure. A sentence that merely
  contains a number is not a figure.
- **One dominant section per page.** In React that is a single `.card` with an
  `h2.h-lede`; in Streamlit it is the page's single `st.header`, with every
  other section an `st.subheader`. A page where every section takes the same
  heading weight has no subject.
- **Remove a layer before adding one.** A `.card` never contains another
  `.card` — the stylesheet flattens it if you try.
- **Emoji are not an icon set.** The one mark is `app/assets/mark.svg`, drawn
  by hand and shared by both front ends.
- **Motion only reports state.** The loading skeleton's sweep is the only
  animation in the app, and it stops under `prefers-reduced-motion`.

Copy is held to the same bar as the code. Write what the thing does, not that it
is powerful: a sentence that would fit another product unchanged is not doing
any work. Section numbers from `PROJECT_PLAN.md` belong in comments and
docstrings, never in a caption — a reader has not got that document.

## Reporting a bug

Open an issue with: what you expected, what happened, and — if it's a
data/scoring issue rather than a crash — the exact symbol and date, since
almost everything here is point-in-time and reproducing "what the algorithm
said about AAPL on a given day" needs that context.
