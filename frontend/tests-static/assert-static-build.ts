import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Refuse to run unless `dist/` is the *static-data* build.
 *
 * This config deliberately does not build — it exists to test the artifact that
 * gets published, not a freshly-made different one. The cost of that is that
 * `dist/` can be the wrong artifact, and when it is, all fourteen content tests
 * fail with "element not found" and none of them says why.
 *
 * It is not hypothetical, and it is not even unlikely: `playwright.config.ts`
 * runs `npm run build` **without** `VITE_STATIC_API`, so running the stubbed
 * suite and then this one — in that order, from a clean checkout — silently
 * leaves an API-mode bundle here for this suite to test. It requests `/api/...`,
 * `vite preview` has no API to proxy to, and every page renders empty.
 *
 * The discriminator is the bundle itself: with `VITE_STATIC_API=1` the constant
 * folds to `true`, the `/api` branch of `lib/api.ts` is eliminated, and the
 * literal disappears entirely. Measured both ways — 1 occurrence in an API
 * build, 0 in a static one.
 */
export default function assertStaticBuild(): void {
  const assets = join(process.cwd(), "dist", "assets");
  let files: string[];
  try {
    files = readdirSync(assets).filter((name) => name.endsWith(".js"));
  } catch {
    throw new Error(
      "frontend/dist/assets is missing — nothing has been built.\n" +
        "Build the published artifact first:\n" +
        "  uv run python scripts/build_static_site.py\n" +
        "  cd frontend && VITE_STATIC_API=1 npm run build\n" +
        "  uv run python scripts/emit_route_pages.py",
    );
  }

  // Matched as `/api` preceded by a quote character, which is how the template
  // literal survives minification (`` `/api${e}${t...` ``). An earlier version
  // looked for a double quote specifically and never matched, so the guard
  // silently passed on exactly the artifact it exists to reject.
  const apiMode = files.some((name) =>
    /["'`]\/api/.test(readFileSync(join(assets, name), "utf8")),
  );
  if (apiMode) {
    throw new Error(
      "frontend/dist holds an API-mode build, not the static one this suite tests.\n" +
        "Every content test would fail with 'element not found' and none would say why.\n" +
        "`npm run test:e2e` rebuilds dist without VITE_STATIC_API, so running it before\n" +
        "this suite leaves exactly this state. Rebuild:\n" +
        "  VITE_STATIC_API=1 npm run build\n" +
        "  uv run python scripts/emit_route_pages.py",
    );
  }
}
