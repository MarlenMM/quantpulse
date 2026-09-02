/**
 * Theme tokens, read out of the stylesheet at runtime.
 *
 * **This exists because of a real bug.** Plotly takes literals — a trace cannot
 * hold `var(--accent)` — so the figures used to carry their own hardcoded
 * colours: `#c9d1d9` axis text and `rgba(255,255,255,0.08)` gridlines, both
 * chosen against the dark theme. On the light theme that is pale grey on white
 * and white on white: the published demo rendered every price chart with axis
 * labels you could barely read and gridlines you could not see at all. Nobody
 * noticed for the same reason it happened — the app was only ever looked at in
 * dark mode, which is exactly the failure mode of designing one theme and
 * declaring the other done.
 *
 * So the figures read the same custom properties the rest of the page uses,
 * once per theme change, and there is one definition of every colour again.
 *
 * `getComputedStyle` on `:root` resolves whichever branch of the media query
 * and `data-theme` cascade is currently winning, so this needs no knowledge of
 * how the theme is selected — only that the tokens exist. They are defined on
 * bare `:root` first, so a value is never missing.
 */
import { useEffect, useState } from "react";

export interface ThemeTokens {
  /** Axis labels, tick text, annotations. */
  ink: string;
  /** Gridlines and zero lines inside a figure. */
  grid: string;
  /** The single interactive accent — also the default series colour. */
  accent: string;
  /** A translucent fill of the accent, for confidence bands. */
  accentBand: string;
  muted: string;
  up: string;
  down: string;
}

const FALLBACK: ThemeTokens = {
  ink: "#3d3a34",
  grid: "rgba(27, 26, 23, 0.09)",
  accent: "#1d4e89",
  accentBand: "rgba(29, 78, 137, 0.16)",
  muted: "#78746a",
  up: "#1c8b4f",
  down: "#bf3327",
};

function read(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function currentTokens(): ThemeTokens {
  const accent = read("--accent", FALLBACK.accent);
  return {
    ink: read("--chart-ink", FALLBACK.ink),
    grid: read("--chart-grid", FALLBACK.grid),
    accent,
    // Plotly needs an rgba() literal for a band fill, and the accent token is a
    // hex. `color-mix` is not available to a JS string, so this converts.
    accentBand: withAlpha(accent, 0.18),
    muted: read("--muted", FALLBACK.muted),
    up: read("--up", FALLBACK.up),
    down: read("--down", FALLBACK.down),
  };
}

/** `#1d4e89` -> `rgba(29, 78, 137, 0.18)`. Passes anything else through. */
export function withAlpha(color: string, alpha: number): string {
  const hex = color.replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(hex)) return color;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * The resolved tokens, refreshed when the OS colour scheme changes.
 *
 * Re-reading on the media query rather than polling means switching the system
 * theme with the page open re-tints the figures, instead of leaving a dark
 * chart stranded on a light page until the next reload.
 */
export function useThemeTokens(): ThemeTokens {
  const [tokens, setTokens] = useState<ThemeTokens>(() =>
    typeof window === "undefined" ? FALLBACK : currentTokens(),
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const refresh = () => setTokens(currentTokens());
    // Re-read once on mount too: the first render can land before the
    // stylesheet has applied, in which case the initial read got the fallback.
    refresh();
    query.addEventListener("change", refresh);
    return () => query.removeEventListener("change", refresh);
  }, []);

  return tokens;
}
