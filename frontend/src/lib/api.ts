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

async function request<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const query = params
    ? `?${new URLSearchParams(
        Object.entries(params).map(([k, v]) => [k, String(v)]),
      ).toString()}`
    : "";
  let response: Response;
  try {
    response = await fetch(`/api${path}${query}`);
  } catch (cause) {
    throw new ApiError(
      "Could not reach the QuantPulse API. Is the backend running?",
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
  ratingChanges: (limit = 10) => request<RatingChange[]>("/screener/changes", { limit }),
  stock: (symbol: string) => request<StockDetail>(`/stocks/${encodeURIComponent(symbol)}`),
  regime: (limit = 90) => request<RegimePoint[]>("/regime", { limit }),
  news: (limit = 8) => request<NewsItem[]>("/news", { limit }),
  backtest: (limit = 20) => request<BacktestRun[]>("/backtest", { limit }),
  sectorRotation: (lookbackDays = 21) =>
    request<SectorStrength[]>("/sectors/rotation", { lookback_days: lookbackDays }),
};
