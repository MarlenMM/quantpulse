import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";
import screenerOneRow from "./fixtures/screener-one-row.json" with { type: "json" };
import stockAIZ from "./fixtures/stock-AIZ.json" with { type: "json" };

/**
 * The regression these tests exist for.
 *
 * `Chart.tsx` loads Plotly through `lazy(() => import("react-plotly.js"))`, and
 * resolving that component broke twice in two days, each time with the page
 * blank and the console reading "Element type is invalid. Received a promise
 * that resolves to: [object Object]":
 *
 * 1. **Vite 7 -> 8.** Rolldown's CommonJS interop yields
 *    `{ default: { default: Component } }` where Rollup gave the component.
 * 2. **react-plotly.js 2 -> 4.** v4 ships the component as a `forwardRef`
 *    object (`{ $$typeof, render }`), so a `typeof === "function"` check fell
 *    through and handed React `undefined`.
 *
 * Every static gate stayed green through both. A failed dynamic import is not
 * a type error and not a build error -- it is a runtime one, and nothing short
 * of a real browser loading the real bundle sees it.
 *
 * So these assert the two things that were actually false: that the figures
 * mount, and that the console is clean. An unhandled render error leaves both
 * a blank region and a console entry, and each check catches it independently.
 */

/** Console errors and uncaught exceptions, collected for the whole page life. */
function watchForErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error: Error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

/**
 * Serve the captured fixture for every API call.
 *
 * Captured from the real API against the committed demo database, so the shape
 * is the server's own rather than one hand-written to match the client's
 * assumptions -- a fixture that agrees with the client but not the server
 * would test nothing.
 */
async function stubApi(page: Page, stock: unknown = stockAIZ): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/stocks/")) {
      await route.fulfill({ json: stock });
      return;
    }
    // A real screener response cut to one row. Captured from the API, not
    // written by hand -- only rows were dropped, so every field is still the
    // server's own (the standing rule about fixtures). One row is the shape a
    // fork with a thin database sees, and the shape that has broken
    // selection logic here before.
    if (url.pathname === "/api/screener") {
      await route.fulfill({ json: screenerOneRow });
      return;
    }
    // Every other endpoint: an empty-but-valid body. The charts under test live
    // on the stock page, and a 404 here would surface as a console error and
    // fail the clean-console assertion for the wrong reason.
    await route.fulfill({ json: [] });
  });
}

test.describe("Stock Detail charts", () => {
  test("every Plotly figure mounts and the console stays clean", async ({ page }) => {
    const errors = watchForErrors(page);
    await stubApi(page);
    await page.goto("/stocks/AIZ");

    // Price, forecast fan, and Monte Carlo -- all three are `Chart`, so a
    // broken import takes out every one of them at once.
    await expect(page.locator(".js-plotly-plot")).toHaveCount(3, { timeout: 30_000 });

    // Plotly draws into an SVG it appends after mounting; a component that
    // mounted but never drew would still be a broken chart.
    await expect(page.locator(".js-plotly-plot .main-svg").first()).toBeVisible();

    expect(errors, `browser reported errors:\n${errors.join("\n")}`).toEqual([]);
  });

  test("the page renders its real numbers, not just its layout", async ({ page }) => {
    await stubApi(page);
    await page.goto("/stocks/AIZ");

    // A blank render still has a <main>, so assert on figures the fixture
    // actually carries. These are the values the Streamlit page shows for the
    // same stored row, which is the agreement the two front ends must keep.
    //
    // Recaptured whenever the API's shape changes, never hand-patched: this
    // fixture predated `is_graded`, so every forecast read as ungraded, the
    // whole table moved behind the disclosure, and the h=5 target below was
    // present in the DOM but hidden. A fixture edited to satisfy the client
    // stops being evidence about the server. Last recaptured 2026-09-26, for
    // point 35 (5- and 20-day horizons only, plus `horizon_note`), from the
    // pre-rendered API output.
    await expect(page.getByText("AIZ — Assurant")).toBeVisible();
    await expect(page.getByText("$283.67")).toBeVisible(); // arima h=5 target
    await expect(page.getByText("1.07")).toBeVisible(); // Sharpe
    await expect(page.getByText("1.54")).toBeVisible(); // Sortino
    await expect(page.getByText("0.16")).toBeVisible(); // beta, vs ^GSPC
  });

  test("a graded horizon is visible and an ungraded one is not", async ({ page }) => {
    // Since point 35 every row in the recaptured fixture is graded (only 5 and
    // 20 days are published, and both always are), so the ungraded case is made
    // here, in the open, rather than by editing the fixture: the 20-day rows
    // lose their grade exactly as the server would send them if a model were
    // short of windows. Every other field is still the server's own.
    const stock = structuredClone(stockAIZ) as typeof stockAIZ;
    for (const f of stock.forecasts) {
      if (f.horizon_days === 20) {
        Object.assign(f, {
          is_graded: false,
          historical_hit_rate: null,
          baseline_hit_rate: null,
          hit_rate_windows: null,
          edge_vs_naive: null,
          edge_ci_low: null,
          edge_ci_high: null,
          edge_note: null,
        });
      }
    }
    await stubApi(page, stock);
    await page.goto("/stocks/AIZ");

    const graded = page.locator("table:has(th:text-is('Horizon (days)'))").first();
    const rows = (horizon: string) =>
      graded.locator(`tbody tr:has(td:nth-child(1):text-is("${horizon}"))`);
    await expect(rows("5").first()).toBeVisible();
    await expect(rows("20")).toHaveCount(0);
    const disclosure = page.locator("details", { hasText: /ungraded horizon/i });
    await expect(disclosure).toBeVisible();
    await expect(disclosure.locator("tbody tr").first()).not.toBeVisible();
  });

  test("the page says why forecasts stop at twenty days", async ({ page }) => {
    await stubApi(page);
    await page.goto("/stocks/AIZ");
    await expect(page.getByText(stockAIZ.horizon_note)).toBeVisible();
    await expect(page.locator("details", { hasText: /ungraded horizon/i })).toHaveCount(0);
  });
});

test("the chart-free pages still render (isolates a chart break from an app break)", async ({
  page,
}) => {
  const errors = watchForErrors(page);
  await stubApi(page);
  await page.goto("/glossary");
  await expect(page.locator("#root")).not.toBeEmpty();
  expect(errors, `browser reported errors:\n${errors.join("\n")}`).toEqual([]);
});

/**
 * The watchlist against the stubbed fixture, where the universe is one stock.
 *
 * Not a duplicate of the static-site test. That one proves the feature works on
 * 503 real rows; this one proves it works when the table has a single row, which
 * is the shape that broke `.slice()`-based selection logic in this project
 * before, and is the only shape a fork with a thin database ever sees.
 */
test("a watchlist star toggles and persists on a single-row universe", async ({ page }) => {
  await stubApi(page);
  await page.goto("/screener");
  await expect(page.locator("tbody tr")).toHaveCount(1);

  const star = page.locator("tbody tr").first().getByRole("button", { name: /watchlist/ });
  await expect(star).toHaveAttribute("aria-pressed", "false");
  await star.click();
  await expect(star).toHaveAttribute("aria-pressed", "true");

  await page.reload();
  await expect(page.getByText("Watchlist only (1)")).toBeVisible();
});

test("compare mode asks for a second name rather than rendering one column", async ({ page }) => {
  await stubApi(page);
  await page.goto("/screener");
  await expect(page.locator("tbody tr")).toHaveCount(1);

  await page.locator("tbody tr").first().getByRole("checkbox", { name: /Compare/ }).check();
  // With one stock in the whole fixture there is no second name to tick, so the
  // panel must explain itself instead of drawing a comparison of one.
  await expect(page.getByText(/Tick a second name to compare/)).toBeVisible();
});
