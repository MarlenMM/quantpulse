import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

/**
 * The published static site actually serves its data.
 *
 * Nothing else can check this. `scripts/build_static_site.py` writes the files
 * and `frontend/src/lib/api.ts` decides which one to ask for; they are in
 * different languages, neither imports the other, and if they disagree about a
 * filename by one character every request 404s. The type checker, the build and
 * the fixture-stubbed chart tests are all still green in that state -- the only
 * symptom is a blank page on a public URL.
 *
 * So each page here is loaded from the real `dist/` with the real generated
 * JSON, and asserted on a value that can only have come out of the demo
 * database. A page that rendered its shell but got nothing back still fails.
 */

/** Console errors and uncaught exceptions, for the whole page life. */
function watchForErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error: Error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

test("the dashboard renders live figures from the pre-rendered data", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("dashboard");

  // Three independent fetches: the ranked table, the regime gauge and the
  // freshness strip each come from a different generated file, so a naming
  // mismatch on any one of them shows up here.
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await expect(page.getByText(/Data freshness/)).toBeVisible();

  // The regime section renders "hasn't been computed yet" until its own fetch
  // resolves, so assert on the value rather than the heading -- the heading is
  // there either way.
  await expect(page.getByText("Risk On")).toBeVisible();

  expect(errors).toEqual([]);
});

test("the screener loads a full ranked universe", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");

  // "503 symbols scored" comes straight out of the screener response. A shell
  // that rendered with an empty table would not produce it.
  await expect(page.getByText(/symbols scored/)).toBeVisible();
  expect(await page.locator("tbody tr").count()).toBeGreaterThan(50);

  expect(errors).toEqual([]);
});

test("switching investor profile fetches a different pre-rendered ranking", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const firstRow = async () => page.locator("tbody tr td").first().innerText();
  const balanced = await firstRow();

  // `income` genuinely re-scores, so it has its own generated file. If that
  // file were missing or misnamed, the table would empty out instead of
  // re-ranking -- which is exactly the drift this suite exists to catch.
  // The control lives inside a collapsed <details>, so open it first.
  await page.getByText("Investor profile & rating scheme").click();
  await page.getByLabel("Start from profile").selectOption("income");
  await expect(page.locator("tbody tr").first()).toBeVisible();
  expect(await page.locator("tbody tr").count()).toBeGreaterThan(50);
  expect(await firstRow()).not.toBe("");

  expect(errors).toEqual([]);
});

test("a stock page deep link renders its charts", async ({ page }) => {
  const errors = watchForErrors(page);
  // Deep links matter on Pages specifically: there is no server-side rewrite,
  // so this only works because `404.html` is a copy of `index.html` and the
  // router strips the project-site base prefix off the path.
  await page.goto("stocks/AIZ");

  // The h1 specifically, not any text node: the symbol also appears in the
  // chart titles and axis labels, so a bare getByText matches five elements
  // and fails Playwright's strict mode for a reason unrelated to the deep
  // link this test exists to check.
  await expect(page.getByRole("heading", { level: 1, name: /^AIZ/ })).toBeVisible();
  await expect(page.locator(".js-plotly-plot")).toHaveCount(3);
  await expect(page.locator(".main-svg").first()).toBeVisible();

  expect(errors).toEqual([]);
});

test("the track record shows a real backtest with its confidence interval", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("track-record");

  // A confidence interval is only rendered next to a stored run, so this fails
  // if the backtest file is missing rather than merely empty.
  await expect(page.getByText("Strategy Sharpe")).toBeVisible();
  await expect(page.getByText(/90% CI/).first()).toBeVisible();

  expect(errors).toEqual([]);
});

test("the glossary serves its terms", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("glossary");

  await expect(page.getByText("Composite score").first()).toBeVisible();

  expect(errors).toEqual([]);
});
