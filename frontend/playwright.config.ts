import { defineConfig, devices } from "@playwright/test";

/**
 * Runtime tests for the built SPA.
 *
 * These exist because two separate charting regressions shipped through every
 * other gate: `tsc --noEmit` passed, `vite build` passed, and both CI jobs were
 * green, while every chart page rendered a blank screen in an actual browser.
 * Both were module-resolution failures inside `lazy(() => import(...))`, which
 * only a real bundler feeding a real browser can reproduce.
 *
 * **Against the production build, not the dev server.** `vite preview` serves
 * exactly the bundle a visitor downloads, so the interop this is guarding is
 * the shipped one rather than the dev-mode dependency pre-bundle.
 *
 * The API is stubbed from a captured fixture, so the suite needs no Python, no
 * database and no network — it tests the front end, which is where these bugs
 * were.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:4173",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
