/**
 * Capture the four README screenshots.
 *
 * The last third of the recipe in `docs/screenshots/README.md`. The other two
 * are scripts (`seed_screenshot_db.py`, `build_demo_gif.py`); this one existed
 * only as "screenshot each page", which is the step most likely to go wrong
 * silently -- a page grabbed a second too early has a half-drawn chart in it,
 * and four pages grabbed at four slightly different window sizes cannot be
 * cross-faded at all.
 *
 * **One Streamlit process per page, and that is load-bearing.** Driving all
 * four pages through a single server dies partway with no traceback and no log
 * line: the server is simply gone, and the next `goto` gets ECONNREFUSED. It is
 * a SIGSEGV inside `libarrow`'s mimalloc allocator (`mi_thread_init` ->
 * `mi_heap_main`, reached from an Arrow hash kernel), which is a native-level
 * crash in this environment and not a bug in any page -- the pages render fine
 * one at a time. `tests/integration/test_ui_pages_real_data.py` hit the same
 * wall and solved it the same way, one subprocess per page; this comment exists
 * so the next person to "simplify" this back into one server finds out why in
 * less than the hour it cost the first time.
 *
 * So the script owns the server rather than asking you to run one: it starts
 * Streamlit, waits for it, captures one page, kills it, and repeats.
 *
 *     uv run python scripts/seed_screenshot_db.py --out build/screenshots.db
 *     node scripts/capture_screenshots.mjs
 *
 * Playwright is resolved out of `frontend/node_modules` -- it is already a
 * devDependency there for the front-end suites, and a second copy at the
 * repository root to take four pictures would be a poor trade.
 */

import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { mkdir, open } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const { chromium } = require(
  require.resolve("@playwright/test", { paths: [join(REPO, "frontend")] }),
);

const { values: args } = parseArgs({
  options: {
    database: { type: "string", default: join(REPO, "build", "screenshots.db") },
    out: { type: "string", default: join(REPO, "docs", "screenshots") },
    port: { type: "string", default: "8501" },
  },
});

const DATABASE = resolve(args.database);
const OUT = resolve(args.out);
const BASE_PORT = Number(args.port);

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

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

/** Start Streamlit on `port` against the scratch database, and wait for it. */
async function startServer(port, logPath) {
  const log = await open(logPath, "a");
  const child = spawn(
    join(REPO, ".venv", "bin", "streamlit"),
    [
      "run",
      join(REPO, "app", "Home.py"),
      "--server.port",
      String(port),
      "--server.headless",
      "true",
      // Nothing edits sources mid-capture, and the watcher is one more thread
      // in a process that is being restarted four times anyway.
      "--server.fileWatcherType",
      "none",
    ],
    {
      cwd: REPO,
      stdio: ["ignore", log.fd, log.fd],
      env: {
        ...process.env,
        DATABASE_URL: `sqlite:///${DATABASE}`,
        PORTFOLIO_BACKEND: "session",
        // The screenshots must show what the app computes, not what a language
        // model said about it -- and the narration panels need an API key the
        // person refreshing these images should not need.
        LLM_ENABLED: "false",
      },
    },
  );

  for (let attempt = 0; attempt < 90; attempt += 1) {
    if (child.exitCode !== null) {
      throw new Error(`Streamlit exited with ${child.exitCode}; see ${logPath}`);
    }
    try {
      const response = await fetch(`http://localhost:${port}/`, { signal: AbortSignal.timeout(2_000) });
      if (response.ok) {
        await log.close();
        return child;
      }
    } catch {
      // Not listening yet.
    }
    await sleep(1_000);
  }
  child.kill("SIGKILL");
  await log.close();
  throw new Error(`Streamlit never came up on port ${port}; see ${logPath}`);
}

async function stopServer(child) {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
  for (let attempt = 0; attempt < 20 && child.exitCode === null; attempt += 1) {
    await sleep(250);
  }
  if (child.exitCode === null) child.kill("SIGKILL");
}

/**
 * Wait until Streamlit has stopped re-running.
 *
 * The status widget is present while a script run is in flight and is removed
 * when it finishes, so "absent" is the signal -- but it is also absent for the
 * moment between two runs, which is why this requires it to stay absent rather
 * than merely observing it once. It reads "CONNECTING" until the websocket is
 * up, so this covers the connect too.
 */
async function waitForIdle(page, { quietMs = 1_500, timeoutMs = 90_000 } = {}) {
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

  const failures = [];
  for (const [index, { path, file, settle }] of PAGES.entries()) {
    // A distinct port per page, so a server that refuses to die does not make
    // the next page silently reuse it.
    const port = BASE_PORT + index;
    const logPath = join(OUT, `.capture-${file}.log`);
    process.stdout.write(`${file} <- :${port}${path} ... `);
    let server = null;
    try {
      server = await startServer(port, logPath);
      const page = await context.newPage();
      await page.goto(`http://localhost:${port}${path}`, {
        waitUntil: "domcontentloaded",
        timeout: 60_000,
      });
      await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 60_000 });
      await waitForIdle(page);
      await page.waitForTimeout(settle);
      await page.screenshot({ path: join(OUT, file) });
      await page.close();
      console.log("ok");
    } catch (error) {
      console.log("FAILED");
      failures.push(`${file}: ${error.message}`);
    } finally {
      if (server) await stopServer(server);
    }
  }

  await browser.close();

  if (failures.length > 0) {
    console.error(`\n${failures.length} page(s) not captured:`);
    for (const failure of failures) console.error(`  ${failure}`);
    console.error(`\nIs there a seeded database at ${DATABASE}?`);
    process.exitCode = 1;
    return;
  }
  console.log("\nnow: uv run python scripts/build_demo_gif.py");
}

await main();
