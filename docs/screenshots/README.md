# Screenshot & demo GIF provenance

These images are captured against a **throwaway** database of **synthetic price
series for 15 well-known tickers**, run through the *real* scoring, forecasting
and backtest pipeline (`analysis.scoring`, `analysis.forecasting`,
`analysis.backtest`) — not hand-typed numbers.

The distinction that matters: everything the seeder writes directly is an
*input* (prices, fundamentals, analyst counts, sentiment, filings, macro
series), and every *output* on screen — the sub-scores, the composite, the
rating, the forecasts, the backtest and its confidence intervals — is computed
by importing the nightly job's own stages from `scripts/refresh_data.py` and
running them over those inputs. So a ticker whose synthetic price trends up
scores and forecasts accordingly, and a Sharpe in these pictures is a Sharpe
this code produced. Hand-typing a plausible-looking number into a screenshot
would make it fiction, and a screenshot of fiction is exactly the fake product
evidence that makes a page untrustworthy.

Real ticker symbols, entirely invented series. Recognisable names make the table
a plausible shape at a glance; inventing the numbers means no screenshot ever
implies this project has had a view on Apple.

Never generated against `quantpulse.db` or `quantpulse_demo.db` — always a
separate scratch file, so this has no effect on your own local data or the
live-deployed demo database (see the root README's "Live Demo & Deployment"
section). `scripts/seed_screenshot_db.py` refuses to write to either by name.

## Refreshing them

This used to be a paragraph of instructions to follow by hand. It is now three
scripts, so two people who run it get the same pictures:

```bash
# 1. Build the scratch database. Slow -- 15-30 minutes -- because the
#    forecasting stage trains the walk-forward models the hit-rate columns are
#    measured on: three runners at four horizons for every symbol.
uv run python scripts/seed_screenshot_db.py --out build/screenshots.db

# 2. Capture all four pages. Starts and stops the app itself.
node scripts/capture_screenshots.mjs

# 3. Cross-fade them into demo.gif.
uv run python scripts/build_demo_gif.py
```

Step 2 is a script rather than four manual grabs because it is the step that
goes wrong silently: a page captured a second too early has a half-drawn chart
in it, and four pages captured at four slightly different window sizes cannot be
cross-faded at all. It waits for Streamlit's own status widget to stay gone
before it shoots, and it takes all four at 1440×900 at 1× — the size the
committed set uses.

**It also starts a fresh Streamlit per page, which is not fussiness.** Driving
all four pages through one server dies partway with no traceback and no log
line — the process is simply gone, and the next navigation gets
`ECONNREFUSED`. It is a SIGSEGV inside `libarrow`'s mimalloc allocator, a
native-level crash in this environment rather than a bug in any page: the pages
render fine one at a time. `tests/integration/test_ui_pages_real_data.py` hit
the same wall and solved it the same way.

Step 3 refuses to run on frames of differing sizes rather than resizing them: a
resized screenshot is a blurry screenshot, and these images exist to show the
type.

The seeder runs entirely offline. The one stage of the nightly job that reaches
the network — the GDELT macro-tone reading behind the Market Regime Index — is
replaced with a stored value, so the gauge is the same on every run instead of
depending on what the news looked like that morning.
