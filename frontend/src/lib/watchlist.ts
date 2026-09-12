/**
 * The SPA's watchlist (Section 9).
 *
 * Streamlit persists a watchlist through `PortfolioStore`. The SPA cannot: the
 * API is read-only by design (ADR 4.5) and the site is static files on GitHub
 * Pages, so there is nowhere on a server to put it. It therefore lives in
 * `localStorage` — **this browser only**, not synced, and gone if site data is
 * cleared. Every surface that shows it says so, because a list that silently
 * fails to follow you to another device is worse than one you knew was local.
 *
 * Every access is wrapped: Safari's private mode throws on `localStorage`
 * rather than returning null, and a screener that crashes because someone
 * opened a private window is a worse bug than a watchlist that forgets.
 */

import { useCallback, useEffect, useState } from "react";

const KEY = "quantpulse.watchlist";

/** Read the stored list, tolerating absence, corruption and a throwing store. */
export function readWatchlist(): string[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Filter rather than trust: this value is editable by hand, and a non-string
    // in here would reach `.toUpperCase()` on render.
    return parsed.filter((s): s is string => typeof s === "string" && s.length > 0);
  } catch {
    return [];
  }
}

function writeWatchlist(symbols: string[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(symbols));
  } catch {
    // Quota exceeded or a blocked store. The in-memory list still works for
    // this page view, which is the graceful half of the failure.
  }
}

/**
 * `[symbols, toggle, isWatched]` for the current browser.
 *
 * Listens for `storage` events so two open tabs do not disagree — a star
 * toggled in one tab is a lie in the other until the page is reloaded
 * otherwise.
 */
export function useWatchlist(): {
  symbols: string[];
  toggle: (symbol: string) => void;
  isWatched: (symbol: string) => boolean;
} {
  const [symbols, setSymbols] = useState<string[]>(() => readWatchlist());

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === KEY || event.key === null) setSymbols(readWatchlist());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const toggle = useCallback((symbol: string) => {
    setSymbols((current) => {
      const next = current.includes(symbol)
        ? current.filter((s) => s !== symbol)
        : [...current, symbol];
      writeWatchlist(next);
      return next;
    });
  }, []);

  const isWatched = useCallback((symbol: string) => symbols.includes(symbol), [symbols]);

  return { symbols, toggle, isWatched };
}
