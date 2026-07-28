/**
 * Display formatting — the TypeScript counterpart of `app/lib/format.py`.
 *
 * The two front ends must agree on what a rating looks like, so the same rule
 * holds here as there: **never encode a rating with color alone** (Section 12).
 * `ratingLabel` always returns an icon *and* a word; color is decoration layered
 * on top. Duplicating this small table in TypeScript is the price of a second
 * front end — the alternative would be shipping presentation strings through
 * the API, which would couple the API's contract to one client's rendering.
 */

export const RATING_ORDER = ["strong_buy", "buy", "hold", "sell", "strong_sell"] as const;

export const RATING_DISPLAY: Record<string, { icon: string; text: string; color: string }> = {
  strong_buy: { icon: "▲▲", text: "Strong Buy", color: "#1a7f37" },
  buy: { icon: "▲", text: "Buy", color: "#2da44e" },
  hold: { icon: "■", text: "Hold", color: "#9a6700" },
  sell: { icon: "▼", text: "Sell", color: "#cf222e" },
  strong_sell: { icon: "▼▼", text: "Strong Sell", color: "#a40e26" },
};

const NEUTRAL = "#6e7781";

export function humanize(value: string | null | undefined): string {
  if (!value) return "—";
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function ratingLabel(rating: string | null | undefined): string {
  if (!rating) return "—";
  const entry = RATING_DISPLAY[rating];
  return entry ? `${entry.icon} ${entry.text}` : humanize(rating);
}

export function ratingColor(rating: string | null | undefined): string {
  return rating && RATING_DISPLAY[rating] ? RATING_DISPLAY[rating].color : NEUTRAL;
}

/** A missing number is an em dash, never a zero — they mean different things. */
export function formatScore(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

/**
 * A value already on a 0-100 scale (62.0) as a percentage ("62.0%").
 * Distinct from formatPercent, which multiplies a 0-1 fraction by 100 --
 * using that on an already-0-100 value (e.g. breadth_pct_above_200dma, a
 * "share, 0-100" per compute_breadth's own contract) silently inflates it
 * 100x (62.0 -> "6200.0%").
 */
export function formatPctAlreadyScaled(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${value.toFixed(digits)}%`;
}

export function formatSignedPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

export function formatPrice(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString(undefined, { style: "currency", currency: "USD" });
}

/** Section 12's freshness badge, including an explicit "never run". */
export function freshnessLabel(isoDate: string | null | undefined): string {
  if (!isoDate) return "never run";
  const then = new Date(`${isoDate}T00:00:00`);
  const today = new Date();
  const days = Math.floor(
    (Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()) -
      Date.UTC(then.getFullYear(), then.getMonth(), then.getDate())) /
      86_400_000,
  );
  if (days < 0) return isoDate;
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  return `${days} days ago`;
}

export function confidenceLabel(value: number | null | undefined): string {
  if (value === null || value === undefined) return "coverage unknown";
  if (value >= 80) return `good coverage (${value.toFixed(0)}%)`;
  if (value >= 50) return `partial coverage (${value.toFixed(0)}%)`;
  return `thin coverage (${value.toFixed(0)}%)`;
}

/**
 * Fuzzy-ish symbol/company search — the TS counterpart of `app/lib/search.py`.
 *
 * Same staged ranking, and for the same reason: an exact ticker must never be
 * outranked by a close-but-wrong company name. Substring tiers only (no
 * edit-distance pass) because the client filters a list already in memory and
 * the user can see the results update as they type.
 */
export function searchSymbols<T extends { symbol: string; name: string | null }>(
  rows: T[],
  query: string,
): T[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return rows;
  const exact: T[] = [];
  const symbolPrefix: T[] = [];
  const namePrefix: T[] = [];
  const contains: T[] = [];
  for (const row of rows) {
    const symbol = row.symbol.toLowerCase();
    const name = (row.name ?? "").toLowerCase();
    if (symbol === needle) exact.push(row);
    else if (symbol.startsWith(needle)) symbolPrefix.push(row);
    else if (name.startsWith(needle)) namePrefix.push(row);
    else if (name.includes(needle) || symbol.includes(needle)) contains.push(row);
  }
  return [...exact, ...symbolPrefix, ...namePrefix, ...contains];
}
