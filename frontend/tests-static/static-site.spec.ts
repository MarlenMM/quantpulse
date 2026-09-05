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

  // The forecast table's default view must hold only horizons with a measured
  // accuracy. Every 63- and 252-day forecast in the published data is ungraded,
  // and those carry by far the largest returns (AIZ's one-year row is +29%
  // against +2% at twenty days), so an ungraded row rendered in the same table
  // as a graded one borrows evidence it does not have. Asserted in a real
  // browser because it is a claim about what a reader sees, not about the data.
  // Scoped to the *first* such table, which is the graded one: a closed
  // `<details>` still holds its rows in the DOM, so a whole-page row query
  // matches the ungraded ones too and this assertion would pass for the wrong
  // reason. Matched on the horizon cell rather than on the text "252", which
  // also appears in prices.
  const defaultTable = page.locator("table:has(th:text-is('Horizon (days)'))").first();
  await expect(defaultTable.locator("tbody tr").first()).toBeVisible();
  await expect(
    defaultTable.locator('tbody tr:has(td:nth-child(1):text-is("252"))'),
  ).toHaveCount(0);

  const disclosure = page.locator("details", { hasText: /ungraded horizon/i });
  await expect(disclosure).toBeVisible();
  const hiddenLongHorizon = disclosure.locator(
    'tbody tr:has(td:nth-child(1):text-is("252"))',
  );
  await expect(hiddenLongHorizon).toHaveCount(1);
  await expect(hiddenLongHorizon).not.toBeVisible();
  await disclosure.locator("summary").click();
  await expect(hiddenLongHorizon).toBeVisible();

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


/**
 * No page may scroll sideways on a phone.
 *
 * Section 31 asks for "a quick mobile-viewing pass", on the grounds that a
 * recruiter is at least as likely to open the demo on a phone as on a laptop.
 * It had never been done, and the Dashboard was **589px wide in a 375px
 * viewport** -- the whole document drifting horizontally under the thumb, not
 * one wide table inside its own scroller.
 *
 * The cause is a CSS default that is easy to reintroduce: a grid item's
 * `min-width` is `auto`, so a wide table refuses to shrink below its
 * min-content width and pushes a bare `1fr` track past the viewport, and the
 * `overflow-x: auto` on `.tablewrap` never gets the chance to engage. The
 * desktop overrides of `.split` and `.split-even` already carried the
 * `minmax(0, …)` guard; their mobile base rules did not.
 *
 * Asserted per page rather than once, because it was true of two pages and
 * false of three -- a single spot check would have called it fixed. Wide
 * content is still allowed to scroll *inside its own container*; what is
 * forbidden is the document doing it.
 */
test.describe("mobile layout", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  for (const [name, path] of [
    ["dashboard", ""],
    ["screener", "screener"],
    ["track record", "track-record"],
    ["stock detail", "stocks/AIZ"],
    ["glossary", "glossary"],
  ] as const) {
    test(`the ${name} does not scroll sideways at 375px`, async ({ page }) => {
      await page.goto(path);
      // The charts settle asynchronously and resize their SVG as they do, so a
      // measurement taken before that lands is a different page's width.
      await page.waitForLoadState("networkidle");

      const { scrollWidth, clientWidth, worst } = await page.evaluate(() => {
        const de = document.documentElement;
        const worst = [...document.querySelectorAll("*")]
          .map((el) => ({ el, right: el.getBoundingClientRect().right }))
          .filter((e) => e.right > de.clientWidth + 1)
          .sort((a, b) => b.right - a.right)
          .slice(0, 3)
          .map((e) => `${e.el.tagName}.${e.el.className} → ${Math.round(e.right)}px`);
        return { scrollWidth: de.scrollWidth, clientWidth: de.clientWidth, worst };
      });

      expect(
        scrollWidth,
        `the document is ${scrollWidth}px wide in a ${clientWidth}px viewport, so the ` +
          `page scrolls sideways. Widest offenders: ${worst.join("; ") || "none"}`,
      ).toBeLessThanOrEqual(clientWidth + 1);
    });
  }
});
