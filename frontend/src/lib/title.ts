/**
 * The browser tab's title, per route (finding 30).
 *
 * Every route has a real `index.html` with its own `<title>`
 * (`scripts/emit_route_pages.py`), so a hard load was always right. Client-side
 * navigation -- how anyone actually browses -- never touched `document.title`,
 * so three stocks in three tabs were indistinguishable, back/forward history
 * was a column of identical entries, and a screen reader announced the same
 * page name on every navigation.
 *
 * These strings are the second copy of the emitter's `FIXED_TITLES` and
 * `_stock_metadata` title. Python and TypeScript cannot share them, so the
 * static-site suite navigates inside the app and asserts `document.title`
 * equals the `<title>` of each route's emitted page.
 */
import { useEffect } from "react";

export const SITE_TITLE = "QuantPulse — S&P 500 research";

export const ROUTE_TITLES: Record<string, string> = {
  "/dashboard": SITE_TITLE,
  "/screener": "Screener — QuantPulse",
  "/track-record": "Track Record — QuantPulse",
  "/glossary": "Glossary — QuantPulse",
};

export const NOT_FOUND_TITLE = "No such page — QuantPulse";

/** `emit_route_pages._stock_metadata`'s title: "NVDA — Nvidia", or the symbol alone. */
export function stockTitle(symbol: string, name: string | null | undefined): string {
  const upper = symbol.toUpperCase();
  return name ? `${upper} — ${name}` : upper;
}

/** Set the tab title; `null` means another component owns it on this route. */
export function useDocumentTitle(title: string | null): void {
  useEffect(() => {
    if (title !== null) document.title = title;
  }, [title]);
}
