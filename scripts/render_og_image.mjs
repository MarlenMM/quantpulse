/**
 * Render the link-preview image every page's card uses (finding 32).
 *
 * Slack, LinkedIn, Discord, iMessage and X build a card from `og:image`, and
 * none of them render SVG, so this is a PNG at the 1200x630 they crop to. It is
 * the project's own mark (`app/assets/mark.svg`, the same file Streamlit and the
 * favicon use) and name on the light theme's paper, set in the same three type
 * roles as the site. One image for every page: the card's title and description
 * are already per-page, which is what tells two stocks apart.
 *
 * Committed as `frontend/public/og-image.png` (Vite copies it to the site root)
 * rather than rendered in CI, because it changes only when the mark or the name
 * does. Re-run after either:
 *
 *     node scripts/render_og_image.mjs
 */
import { createRequire } from "node:module";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const { chromium } = require(
  require.resolve("@playwright/test", { paths: [join(REPO, "frontend")] }),
);

const mark = (await readFile(join(REPO, "app", "assets", "mark.svg"), "utf8"))
  .replace(/<!--[\s\S]*?-->/g, "")
  .replace(/width="26" height="24"/, 'width="208" height="192"');

const page = `<!doctype html><html><head><style>
  html, body { margin: 0; }
  body {
    width: 1200px; height: 630px; box-sizing: border-box;
    padding: 96px 104px; background: #faf8f4; color: #1b1a17;
    display: flex; flex-direction: column; justify-content: space-between;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .lockup { display: flex; align-items: center; gap: 44px; }
  h1 {
    margin: 0; font-size: 124px; font-weight: 600; letter-spacing: -0.01em;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Charter, Georgia, serif;
  }
  p { margin: 0; font-size: 38px; line-height: 1.3; color: #55524a; }
  .rule { border-top: 2px solid #b9b1a1; padding-top: 22px; display: flex; justify-content: space-between;
          font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 22px; color: #706c63;
          gap: 32px; white-space: nowrap; }
</style></head><body>
  <div class="lockup">${mark}<h1>QuantPulse</h1></div>
  <p>The S&amp;P 500, scored across seven categories of public data.</p>
  <div class="rule"><span>marlenmm.github.io/quantpulse</span><span>Educational research · not financial advice</span></div>
</body></html>`;

const browser = await chromium.launch();
const tab = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
await tab.setContent(page);
const out = join(REPO, "frontend", "public", "og-image.png");
await tab.screenshot({ path: out, clip: { x: 0, y: 0, width: 1200, height: 630 } });
await browser.close();
console.log(`wrote ${out}`);
