/**
 * Capture the four README screenshots from a running Streamlit app.
 *
 * The last third of the recipe in `docs/screenshots/README.md`. The other two
 * are scripts (`seed_screenshot_db.py`, `build_demo_gif.py`); this one existed
 * only as "screenshot each page", which is the step most likely to go wrong
 * silently -- a page grabbed a second too early has a half-drawn chart in it,
 * and four pages grabbed at four slightly different window sizes cannot be
 * cross-faded at all.
 *
 * So: one viewport for all four, and an explicit wait for Streamlit to go idle
 * rather than a hopeful sleep. Streamlit renders over a websocket and draws its
 * Plotly figures after the DOM settles, so `networkidle` alone is not enough --
 * this waits for the app's own "Running" status widget to stay gone, then lets
 * the figures paint.
 *
 * Usage, with the app already serving the scratch database:
 *
 *     uv run python scripts/seed_screenshot_db.py --out build/screenshots.db
 *     DATABASE_URL=sqlite:///build/screenshots.db uv run streamlit run app/Home.py
 *     node scripts/capture_screenshots.mjs
 *
 * Playwright is resolved out of `frontend/node_modules` -- it is already a
 * devDependency there for the front-end suites, and a second copy at the
 * repository root to take four pictures would be a poor trade.
 */

import { createRequire } from "node:module";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const { chromium } = require(
  require.resolve("@playwright/test", { paths: [join(REPO, "frontend")] }),
);

const BASE = process.env.STREAMLIT_URL ?? "http://localhost:8501";
const OUT = join(REPO, "docs", "screenshots");

/** The same 1440x900 the committed set uses. Changing it invalidates the GIF. */
const VIEWPORT = { width: 1440, height: 900 };

/**
 * In the order a reader meets them: the market-wide view, the ranking behind
 * it, one company in full, and the honest track record underneath.
 *
 * `settle` is extra time *after* Streamlit reports itself idle, for pages whose
 * Plotly figures are heavy enough to paint a frame late. The stock page draws
 * five.
 */
const PAGES = [
  { path: "/", file: "dashboard.png", settle: 2_500 },
  { path: "/Screener", file: "screener.png", settle: 3_000 },
  { path: "/Stock_Detail", file: "stock_detail.png", settle: 4_500 },
  { path: "/Backtest", file: "backtest.png", settle: 3_000 },
];

/**
 * Wait until Streamlit has stopped re-running.
 *
 * The status widget appears while a script run is in flight and is removed when
 * it finishes, so "absent" is the signal -- but it is also absent for the
 * moment between two runs, which is why this requires it to stay absent rather
 * than merely observing it once.
 */
async function waitForIdle(page, { quietMs = 1_500, timeoutMs = 120_000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let quietSince = null;
  while (Date.now() < deadline) {
    const busy = await page.locator('[data-testid="stStatusWidget"]').count();
    if (busy > 0) {
      quietSince = null;
    } else {
      quietSince ??= Date.now();
      if (Date.now() - quietSince >= quietMs) return;
    }
    await page.waitForTimeout(150);
  }
  throw new Error(`Streamlit never went idle within ${timeoutMs}ms`);
}

async function main() {
  await mkdir(OUT, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: VIEWPORT,
    // 1x, matching the committed set. A 2x capture is a sharper picture and a
    // four-times-larger GIF that GitHub scales back down anyway.
    deviceScaleFactor: 1,
    colorScheme: "dark",
    // Motion in a still is only ever a half-drawn transition.
    reducedMotion: "reduce",
  });
  const page = await context.newPage();

  const failures = [];
  for (const { path, file, settle } of PAGES) {
    const url = `${BASE}${path}`;
    process.stdout.write(`${file} <- ${url} ... `);
    try {
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 60_000 });
      await waitForIdle(page);
      await page.waitForTimeout(settle);
      await page.screenshot({ path: join(OUT, file) });
      console.log("ok");
    } catch (error) {
      console.log("FAILED");
      failures.push(`${file}: ${error.message}`);
    }
  }

  await browser.close();

  if (failures.length > 0) {
    console.error(`\n${failures.length} page(s) not captured:`);
    for (const failure of failures) console.error(`  ${failure}`);
    console.error(`\nIs the app serving the scratch database at ${BASE}?`);
    process.exitCode = 1;
    return;
  }
  console.log("\nnow: uv run python scripts/build_demo_gif.py");
}

await main();
