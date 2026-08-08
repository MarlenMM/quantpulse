import { defineConfig, devices } from "@playwright/test";

/**
 * Runtime check for the *published* build -- the static-data one.
 *
 * `playwright.config.ts` stubs the API from a fixture, which is right for the
 * charting regressions it guards: those are front-end bugs and the suite stays
 * fast and dependency-free. This config is the opposite trade. It serves the
 * exact `dist/` that gets uploaded to GitHub Pages, with the real pre-rendered
 * JSON in it, and no stubbing at all.
 *
 * That is the only thing that can catch the failure mode this deployment has:
 * a response the SPA asks for that the generator never wrote, or wrote under a
 * different name. Nothing static sees it -- the client and the generator are in
 * different languages and neither imports the other -- and the symptom is a
 * blank section on a public page.
 *
 * It deliberately does not build: `.github/workflows/pages.yml` builds with
 * `VITE_STATIC_API` and `VITE_BASE` set, and this must test that artifact
 * rather than a freshly-made different one. Run it the same way locally:
 *
 *   python scripts/build_static_site.py
 *   cd frontend
 *   VITE_STATIC_API=1 npm run build
 *   npm run test:static
 */
const BASE = process.env.VITE_BASE ?? "/";

export default defineConfig({
  testDir: "./tests-static",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://localhost:4177${BASE}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run preview -- --port 4177 --strictPort",
    url: `http://localhost:4177${BASE}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
