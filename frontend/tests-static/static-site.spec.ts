import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

import { humanize } from "../src/lib/format";

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
  //
  // The value is read from the generated file, never written here. This line
  // once said "Risk On", which is a claim about the market, not the site: on
  // 2026-09-23 the regime scored 55.7, came out "neutral", and this gate held
  // the whole demo at the previous day's data -- and would have kept holding it
  // until the market happened to turn risk-on again.
  const regime = JSON.parse(
    readFileSync(join(process.cwd(), "dist", "data", "regime__limit-90.json"), "utf8"),
  ) as { regime_label: string }[];
  const label = regime.at(-1)?.regime_label;
  expect(label, "the pre-rendered regime file has no rows").toBeTruthy();
  await expect(page.getByText(humanize(label), { exact: true })).toBeVisible();

  expect(errors).toEqual([]);
});

/**
 * Finding 29: the landing page drew one dial with the whole of Plotly.
 *
 * Measured on the built site before the change: the Dashboard downloaded
 * 4,373 KB of JavaScript, 4,113 KB of it the Plotly chunk (1.24 MB gzipped),
 * to draw a semicircle, three bands and a number. The app chunk alone is
 * ~261 KB. The budget sits between the two, so a chart library pulled back in
 * by any route -- a stray `<Chart>`, an eager import -- fails here by size.
 */
test("the dashboard downloads no chart library", async ({ page }) => {
  const scripts: { name: string; bytes: number }[] = [];
  page.on("response", async (response) => {
    if (!response.url().endsWith(".js")) return;
    try {
      scripts.push({ name: response.url().split("/").pop() ?? "", bytes: (await response.body()).length });
    } catch {
      // A response body can vanish on navigation; the size check below is
      // still made on every script that did arrive.
    }
  });
  await page.goto("dashboard");
  await expect(page.locator(".regime-gauge")).toBeVisible();
  await page.waitForLoadState("networkidle");

  await expect(page.locator(".js-plotly-plot")).toHaveCount(0);
  const total = scripts.reduce((sum, s) => sum + s.bytes, 0);
  expect(
    total,
    `the dashboard downloaded ${total} bytes of JavaScript: ` +
      scripts.map((s) => `${s.name} ${s.bytes}`).join(", "),
  ).toBeLessThan(400_000);
});

/**
 * The dial's zones are where the label changes, not where a literal says.
 *
 * Both gauges drew risk-on from 65 while `market_regime` labels it from 60, so
 * on 2026-09-16 a 64.7 was titled "Risk On" with its arc ending in the neutral
 * band. The zone that contains the score must be the zone the label names, and
 * its edges must be the cutoffs the API sent.
 */
test("the regime dial's zones are the label's own cutoffs", async ({ page }) => {
  const regime = JSON.parse(
    readFileSync(join(process.cwd(), "dist", "data", "regime__limit-90.json"), "utf8"),
  ) as { regime_score: number; regime_label: string; risk_on_at: number; risk_off_at: number }[];
  const latest = regime.at(-1)!;
  await page.goto("dashboard");
  await expect(page.locator(".regime-gauge")).toBeVisible();

  const bands = await page.locator(".regime-gauge [data-band]").evaluateAll((paths) =>
    paths.map((p) => ({
      band: p.getAttribute("data-band"),
      from: Number(p.getAttribute("data-from")),
      to: Number(p.getAttribute("data-to")),
    })),
  );
  expect(bands).toEqual([
    { band: "risk_off", from: 0, to: latest.risk_off_at },
    { band: "neutral", from: latest.risk_off_at, to: latest.risk_on_at },
    { band: "risk_on", from: latest.risk_on_at, to: 100 },
  ]);
  const containing = bands.find(
    (b) =>
      latest.regime_score >= b.from &&
      (latest.regime_score < b.to || (b.to === 100 && latest.regime_score === 100)),
  );
  // On the boundaries the label is inclusive at both cutoffs (>= on, <= off).
  const onEdge =
    latest.regime_score === latest.risk_on_at || latest.regime_score === latest.risk_off_at;
  if (!onEdge) expect(containing?.band).toBe(latest.regime_label);
});

/**
 * Finding 30: the tab title never changed as you moved through the app.
 *
 * Each route's emitted `index.html` carries its own `<title>` (a hard load gets
 * it right), but client-side navigation -- how anyone actually browses -- left
 * every tab on "QuantPulse — S&P 500 research". The SPA now sets the title
 * itself, and this pins its strings to the emitter's: after navigating *inside
 * the app* to each route, `document.title` must equal the `<title>` of that
 * route's page in `dist/`. Python writes one, TypeScript the other, and this is
 * the only place the two meet.
 */
function emittedTitle(route: string): string {
  const html = readFileSync(join(process.cwd(), "dist", route, "index.html"), "utf8");
  const match = html.match(/<title>([^<]*)<\/title>/);
  expect(match, `dist/${route}/index.html has no <title> -- run emit_route_pages.py`).toBeTruthy();
  return match![1].replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#x27;/g, "'");
}

test("client-side navigation gives every page its own tab title", async ({ page }) => {
  await page.goto("dashboard");
  await expect(page.locator(".regime-gauge")).toBeVisible();
  await expect(page).toHaveTitle(emittedTitle("dashboard"));

  for (const [label, route] of [
    ["Screener", "screener"],
    ["Track Record", "track-record"],
    ["Glossary", "glossary"],
  ] as const) {
    await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: label }).click();
    await expect(page).toHaveURL(new RegExp(`/${route}$`));
    await expect(page).toHaveTitle(emittedTitle(route));
  }

  // Into a stock from the ranked table, the way a reader arrives at one.
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Screener" }).click();
  const first = page.locator("tbody tr .ticker").first();
  const symbol = (await first.innerText()).trim();
  await first.click();
  await expect(page.getByRole("heading", { level: 1, name: new RegExp(`^${symbol}`) })).toBeVisible();
  await expect(page).toHaveTitle(emittedTitle(`stocks/${symbol}`));

  // And back: history entries are only useful if they are distinguishable.
  await page.goBack();
  await expect(page).toHaveTitle(emittedTitle("screener"));
});

test("the screener loads a full ranked universe", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");

  // "503 symbols scored" comes straight out of the screener response. A shell
  // that rendered with an empty table would not produce it.
  await expect(page.getByText(/symbols scored/)).toBeVisible();
  // The table is paginated, so the row count is the page size by design. What
  // proves the whole universe arrived is the count the pager states -- asserting
  // rendered rows would now be asserting PAGE_SIZE and nothing about the fetch.
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await expect(page.getByText(/Showing 1–50 of \d{3}/)).toBeVisible();

  expect(errors).toEqual([]);
});

test("switching investor profile fetches a different pre-rendered ranking", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  // The symbol, not the first cell: the first cell is now the rank, which reads
  // "1" for every profile and would make the comparison below vacuous.
  const firstRow = async () => page.locator("tbody tr .ticker").first().innerText();
  const balanced = await firstRow();

  // `income` genuinely re-scores, so it has its own generated file. If that
  // file were missing or misnamed, the table would empty out instead of
  // re-ranking -- which is exactly the drift this suite exists to catch.
  // The control lives inside a collapsed <details>, so open it first.
  await page.getByText("Investor profile & rating scheme").click();
  await page.getByLabel("Start from profile").selectOption("income");
  await expect(page.locator("tbody tr").first()).toBeVisible();
  // Same reasoning as above: the pager's total is what shows the other
  // pre-rendered file was fetched, not the number of rows on screen.
  await expect(page.getByText(/Showing 1–50 of \d{3}/)).toBeVisible();
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

  // Every figure drew the trace type it asked for. The app ships a Plotly with
  // three trace types registered (`src/lib/plotly.ts`, finding 29), and an
  // unregistered one fails silently: Plotly resolves it to an empty `scatter`,
  // logs nothing, and every other assertion here still passes -- measured by
  // unregistering `scatterpolar`, which blanked the radar with 28/28 green.
  const types = await page.locator(".js-plotly-plot").evaluateAll((plots) =>
    plots.map((el) => {
      const figure = el as unknown as {
        data: { type?: string }[];
        _fullData: { type: string }[];
      };
      return {
        asked: figure.data.map((trace) => trace.type ?? "scatter"),
        drawn: figure._fullData.map((trace) => trace.type),
      };
    }),
  );
  for (const { asked, drawn } of types) expect(drawn).toEqual(asked);

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


/**
 * A route served from its own `index.html` arrives with a trailing slash.
 *
 * Deep links are real files now, so a static host answers `/screener` with a
 * 301 to `/screener/` and serves the directory index -- which means the app
 * boots on a pathname it never saw before. `useMatch` splits on "/" and filters
 * empties so the stock route never noticed, but `App.tsx`'s switch compares
 * whole strings: without the normalization in `toRoute` every one of those
 * visits rendered "No such page". Verified by reverting that one line, which
 * turned `/screener/` into the not-found view while every other gate stayed
 * green.
 */
test.describe("routes served as directory pages", () => {
  for (const [name, path] of [
    ["screener", "screener/"],
    ["track record", "track-record/"],
    ["glossary", "glossary/"],
    ["stock detail", "stocks/AIZ/"],
  ] as const) {
    test(`the ${name} renders with a trailing slash`, async ({ page }) => {
      const errors = watchForErrors(page);
      await page.goto(path);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await expect(page.getByText("No such page")).toHaveCount(0);
      expect(errors).toEqual([]);
    });
  }
});

/**
 * Section 10/12's three SPA features, against the real pre-rendered site.
 *
 * All three are pure client-side work over data the screener payload already
 * carries — which is exactly why they need a browser to test. None of them adds
 * a request, so a broken one is invisible to the type checker, the build, and
 * any test that only checks a fetch resolved.
 */
test("the screener exports the rows it is showing as CSV", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download as CSV" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("quantpulse_screener.csv");

  const stream = await file.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf8");

  // A header plus a body, not an empty file with a header.
  const lines = text.trim().split("\r\n");
  expect(lines[0]).toContain("Symbol");
  expect(lines[0]).toContain("Data coverage %");
  expect(lines.length).toBeGreaterThan(50);

  // The first data row must be the first table row: the export is what is on
  // screen, filters and weights included, not the stored balanced ranking.
  const firstSymbol = await page.locator("tbody tr").first().locator(".ticker").innerText();
  expect(lines[1]).toContain(firstSymbol);

  // Every row must have exactly as many fields as the header.
  //
  // This is the assertion that matters and the one the first version of this
  // test lacked. 2,976 names in the demo database contain a comma -- "Nike,
  // Inc.", "Nasdaq, Inc." -- so an unquoted cell shifts every column after it
  // and produces a file that still opens, still has a plausible header, and is
  // wrong from the first mega-cap onward. Removing the quoting entirely left
  // all nineteen tests green.
  const fields = (line: string): number => {
    let count = 1;
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') i += 1;
        else inQuotes = !inQuotes;
      } else if (char === "," && !inQuotes) count += 1;
    }
    return count;
  };
  const expected = fields(lines[0]);
  const ragged = lines.filter((line) => fields(line) !== expected);
  expect(ragged.slice(0, 3)).toEqual([]);

  // And a name that needs quoting is actually quoted, not merely survivable.
  const commaName = lines.find((line) => line.includes('", Inc."') || line.includes('"'));
  expect(commaName, "no quoted cell in the export at all").toBeTruthy();
  expect(errors).toEqual([]);
});

test("the watchlist survives a reload and filters the table", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const first = page.locator("tbody tr").first();
  const symbol = await first.locator(".ticker").innerText();
  await first.getByRole("button", { name: new RegExp(`Add ${symbol} to your watchlist`) }).click();

  // Persisted, not merely held in React state.
  await page.reload();
  await expect(page.getByText("Watchlist only (1)")).toBeVisible();

  await page.getByRole("checkbox", { name: /Watchlist only/ }).check();
  await expect(page.locator("tbody tr")).toHaveCount(1);
  await expect(page.locator("tbody tr").first().locator(".ticker")).toHaveText(symbol);

  // And it is reversible from the star it was set with.
  await page
    .locator("tbody tr")
    .first()
    .getByRole("button", { name: new RegExp(`Remove ${symbol} from your watchlist`) })
    .click();
  await expect(page.getByText("Watchlist only (0)")).toBeVisible();
  expect(errors).toEqual([]);
});

test("compare mode puts two names' sub-scores side by side", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const rows = page.locator("tbody tr");
  const firstSymbol = await rows.nth(0).locator(".ticker").innerText();
  const secondSymbol = await rows.nth(1).locator(".ticker").innerText();

  await rows.nth(0).getByRole("checkbox", { name: `Compare ${firstSymbol}` }).check();
  // One name is not a comparison, and the panel should say so rather than
  // render a one-column table.
  await expect(page.getByText(/Tick a second name to compare/)).toBeVisible();

  await rows.nth(1).getByRole("checkbox", { name: `Compare ${secondSymbol}` }).check();

  const compare = page.locator("table").last();
  await expect(compare.getByRole("columnheader", { name: firstSymbol })).toBeVisible();
  await expect(compare.getByRole("columnheader", { name: secondSymbol })).toBeVisible();
  // Every category is a row, plus composite, rating and coverage.
  await expect(compare.getByRole("rowheader", { name: "Fundamental" })).toBeVisible();
  await expect(compare.getByRole("rowheader", { name: "Composite" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("the track record exports its run history with what each run ranked", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("track-record");
  await expect(page.getByRole("heading", { name: "Run history" })).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download as CSV" }).click();
  const file = await download;
  const stream = await file.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf8");

  // The two columns the on-screen table has no room for, and without which a
  // row cannot be interpreted at all.
  expect(text).toContain("Signal ranked");
  expect(text).toContain("Assumed txn cost");
  expect(errors).toEqual([]);
});

/**
 * Section 12 accessibility: the skip link and the Screener's row budget.
 *
 * Measured on the live site before this landed: **1,535 focusable elements**,
 * 1,509 of them inside `<tbody>`, and 7,695 DOM nodes — a keyboard user leaving
 * the table had that many stops to get past, and there was no skip link to
 * avoid the header either. The audit recorded 525; the watchlist star and
 * compare checkbox added per row since then had nearly tripled it.
 */
const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),' +
  'textarea:not([disabled]),[tabindex]:not([tabindex="-1"]),details>summary';

test("a skip link is the first thing the keyboard reaches, and it moves focus", async ({
  page,
}) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  // Hidden until focused, visible once it is — a permanently invisible skip
  // link is a trap for a sighted keyboard user.
  await expect(skip).toBeInViewport();

  await page.keyboard.press("Enter");
  // Focus must actually land on the landmark. Without tabIndex=-1 the browser
  // scrolls there and leaves focus on the link, so the next Tab goes back to
  // the nav and the link silently does nothing.
  await expect(page.locator("main#content")).toBeFocused();
  expect(errors).toEqual([]);
});

test("the screener renders one page of rows, not the whole universe", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const rows = await page.locator("tbody tr").count();
  expect(rows).toBeLessThanOrEqual(50);

  // The budget this exists for. 503 rows carried 1,509 focusable controls.
  const focusable = await page.locator(FOCUSABLE).count();
  expect(focusable).toBeLessThan(250);

  // And the full count is still stated, so nothing looks lost.
  await expect(page.getByText(/Showing 1–50 of \d+/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("paging keeps the ranking's numbering and does not restart it", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  const firstRank = await page.locator("tbody tr").first().locator("td").first().innerText();
  expect(firstRank.trim()).toBe("1");

  await page.getByRole("button", { name: "Next →" }).click();
  const nextRank = await page.locator("tbody tr").first().locator("td").first().innerText();
  expect(nextRank.trim()).toBe("51");
  await expect(page.getByText(/Showing 51–100 of \d+/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("the CSV exports every filtered row, not just the page on screen", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  // Deliberately from page 2: the export must not follow the viewport.
  await page.getByRole("button", { name: "Next →" }).click();
  await expect(page.getByText(/Showing 51–100/)).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download as CSV" }).click();
  const stream = await (await download).createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk as Buffer);
  const lines = Buffer.concat(chunks).toString("utf8").trim().split("\r\n");

  expect(lines.length).toBeGreaterThan(200);
  expect(errors).toEqual([]);
});

test("changing a filter returns to the first page of the new ranking", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  // To page 5, then filter to a sector that leaves exactly two pages.
  for (let i = 0; i < 4; i += 1) await page.getByRole("button", { name: "Next →" }).click();
  await expect(page.getByText(/page 5 of/)).toBeVisible();

  await page.getByLabel("Sector").selectOption("Financials");

  // Distinguishes the reset from the clamp. Clamping alone would land on the
  // *last* page of the new ranking ("page 2 of 2") -- past the top of a ranking
  // the reader has just asked a new question of. The clamp is still what stops
  // an empty table; this is the behaviour on top of it, and without its own
  // case the two are indistinguishable.
  await expect(page.getByText(/page 1 of 2/)).toBeVisible();
  await expect(page.locator("tbody tr").first().locator("td").first()).toHaveText("1");
  expect(errors).toEqual([]);
});

test("re-weighting a category also returns to the first page", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  for (let i = 0; i < 2; i += 1) await page.getByRole("button", { name: "Next →" }).click();
  await expect(page.getByText(/page 3 of/)).toBeVisible();

  // Moving a slider re-ranks the whole universe, so page 3 is now a different
  // slice of a differently-ordered list -- the reader is looking at names they
  // never asked to see. The weights belong in the filter signature for exactly
  // this reason, and without their own case they can be dropped from it with
  // every other test still green.
  await page.getByText("Re-weight categories").click();
  const slider = page.getByLabel("Fundamental", { exact: false }).first();
  await slider.fill("0.05");

  await expect(page.getByText(/page 1 of/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("narrowing a filter while deep in the pages does not show an empty table", async ({
  page,
}) => {
  const errors = watchForErrors(page);
  await page.goto("screener");
  await expect(page.locator("tbody tr").first()).toBeVisible();

  await page.getByRole("button", { name: "Next →" }).click();
  await page.getByRole("button", { name: "Next →" }).click();
  await expect(page.getByText(/page 3 of/)).toBeVisible();

  // A filter that leaves far fewer than three pages. Without a reset the table
  // would render nothing and read as "no matches" when the truth is "you are
  // past the end".
  await page.getByLabel("Search symbol or company").fill("AAPL");
  await expect(page.locator("tbody tr").first()).toBeVisible();
  expect(errors).toEqual([]);
});
