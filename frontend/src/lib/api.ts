/**
 * Typed client for the QuantPulse read API.
 *
 * One `request` helper so every call has identical error semantics: a non-2xx
 * response throws an `ApiError` carrying the status, and the caller decides
 * what to render. Swallowing failures into empty arrays here would make an
 * unreachable backend look exactly like an un-run pipeline, which is the one
 * distinction this app works hardest everywhere else to preserve.
 */

import type {
  AbsoluteRatingResponse,
  InvestorProfile,
  BacktestRun,
  GlossaryTerm,
  Health,
  NewsItem,
  RatingChange,
  RegimePoint,
  ScreenerResponse,
  SectorStrength,
  StockDetail,
  TickerSummary,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Static-data mode: read pre-rendered JSON files instead of calling a server.
 *
 * The published demo lives on GitHub Pages, which serves static files and
 * cannot run Python. Every response this client can ask for is therefore
 * generated ahead of time by `scripts/build_static_site.py` — which produces
 * them by calling the real FastAPI app over the committed demo database, so
 * the files are the API's own output rather than a second implementation that
 * could disagree with it.
 *
 * Set at build time (`VITE_STATIC_API=1 npm run build`). Unset — development,
 * the local `npm run dev` proxy, and the Playwright suite — nothing changes and
 * the client talks to a live backend as before.
 */
const STATIC_DATA = import.meta.env.VITE_STATIC_API === "1";

/**
 * Where a pre-rendered response lives, given the request that would have
 * fetched it.
 *
 * Query strings cannot address a static file — a plain file server ignores
 * them — so the parameters are folded into the filename, sorted so the same
 * request always names the same file. `scripts/build_static_site.py` writes
 * exactly these names, and `tests/unit/test_static_site_layout.py` runs this
 * very function under node and compares, so the two cannot drift.
 */
export function staticPath(path: string, params?: Record<string, string | number>): string {
  const name = path.replace(/^\//, "").split("/").join("__");
  const suffix = params
    ? Object.keys(params)
        .sort()
        .map((key) => `__${key}-${params[key]}`)
        .join("")
    : "";
  return `${import.meta.env.BASE_URL}data/${name}${suffix}.json`;
}

async function request<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const query = params
    ? `?${new URLSearchParams(
        Object.entries(params).map(([k, v]) => [k, String(v)]),
      ).toString()}`
    : "";
  const url = STATIC_DATA ? staticPath(path, params) : `/api${path}${query}`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch (cause) {
    throw new ApiError(
      STATIC_DATA
        ? "Could not load the pre-rendered demo data for this page."
        : "Could not reach the QuantPulse API. Is the backend running?",
      0,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; the status text is the best we have */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  glossary: () => request<GlossaryTerm[]>("/glossary"),
  universe: () => request<TickerSummary[]>("/universe"),
  screener: (profile = "balanced") => request<ScreenerResponse>("/screener", { profile }),
  profiles: () => request<InvestorProfile[]>("/profiles"),
  // Absolute ratings are computed server-side: they need
  // `build_composite(rating_mode="absolute")` over the stored raw category
  // values, and a second copy of that mapping in TypeScript would drift.
  screenerAbsolute: (profile = "balanced") =>
    request<AbsoluteRatingResponse>("/screener/absolute", { profile }),
  ratingChanges: (limit = 10) => request<RatingChange[]>("/screener/changes", { limit }),
  stock: (symbol: string) => request<StockDetail>(`/stocks/${encodeURIComponent(symbol)}`),
  regime: (limit = 90) => request<RegimePoint[]>("/regime", { limit }),
  news: (limit = 8) => request<NewsItem[]>("/news", { limit }),
  backtest: (limit = 20) => request<BacktestRun[]>("/backtest", { limit }),
  sectorRotation: (lookbackDays = 21) =>
    request<SectorStrength[]>("/sectors/rotation", { lookback_days: lookbackDays }),
};
