import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";
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
async function stubApi(page: Page): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/stocks/")) {
      await route.fulfill({ json: stockAIZ });
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
    // stops being evidence about the server.
    await expect(page.getByText("AIZ — Assurant")).toBeVisible();
    await expect(page.getByText("$289.16")).toBeVisible(); // arima h=5 target
    await expect(page.getByText("1.78")).toBeVisible(); // Sharpe
    await expect(page.getByText("2.84")).toBeVisible(); // Sortino
    await expect(page.getByText("0.23")).toBeVisible(); // beta, vs ^GSPC
  });

  test("a graded horizon is visible and an ungraded one is not", async ({ page }) => {
    // The fixture carries both -- 6 graded rows and 6 ungraded -- so this is a
    // claim about the split, not about the data happening to be one-sided.
    await stubApi(page);
    await page.goto("/stocks/AIZ");

    const rows = (horizon: string) =>
      page.locator(`tbody tr:has(td:nth-child(1):text-is("${horizon}"))`);
    await expect(rows("5").first()).toBeVisible();
    await expect(rows("252").first()).not.toBeVisible();
    await expect(page.locator("details", { hasText: /ungraded horizon/i })).toBeVisible();
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
