/**
 * TypeScript mirrors of the API's Pydantic schemas (`quantpulse/api/schemas.py`).
 *
 * Every field the Python side types `| None` is `| null` here, deliberately and
 * without a default. The project's coverage discipline — a missing sub-score is
 * not a zero sub-score (Section 7.5) — only survives the trip through JSON if
 * the client is forced to handle the absence rather than allowed to `?? 0` it
 * by accident. `strict` and `strictNullChecks` are what make that a compile
 * error instead of a silently fabricated number on screen.
 */

export interface Health {
  status: string;
  has_data: boolean;
  freshness: Record<string, string | null>;
}

export interface GlossaryTerm {
  term: string;
  category: string;
  definition: string;
}

export interface TickerSummary {
  symbol: string;
  name: string | null;
  sector: string | null;
  asset_type: string | null;
}

export interface ScreenerRow {
  symbol: string;
  name: string | null;
  sector: string | null;
  date: string;
  fundamental_score: number | null;
  technical_score: number | null;
  analyst_score: number | null;
  sentiment_score: number | null;
  momentum_score: number | null;
  industry_macro_score: number | null;
  smart_money_score: number | null;
  composite_score: number;
  percentile_rank: number | null;
  rating: string;
  data_confidence: number | null;
}

export interface ScreenerResponse {
  as_of: string | null;
  profile: string;
  rating_mode: string;
  // Percentile a name must reach to be a Strong Buy tonight: normally 90,
  // lifted toward 95 by the Market Regime Index in a risk-off market. Sent by
  // the API so client-side re-rating uses the same dampener the stored ratings
  // did, rather than a second implementation of it in TypeScript.
  strong_buy_cutoff: number;
  count: number;
  rows: ScreenerRow[];
}

export interface RatingChange {
  symbol: string;
  previous_rating: string;
  rating: string;
  previous_score: number;
  composite_score: number;
  score_change: number;
}

export interface PriceBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adj_close: number | null;
  volume: number | null;
}

export interface ForecastRow {
  model_name: string;
  horizon_days: number;
  point_return: number;
  point_price: number | null;
  lower_price: number | null;
  upper_price: number | null;
  historical_hit_rate: number | null;
  baseline_hit_rate: number | null;
  // Distinct out-of-sample windows the two rates above were measured over. A
  // rate from 40 windows and one from 3 are different claims; without this they
  // rendered identically.
  hit_rate_windows: number | null;
  generated_date: string;
}

export interface PatternRow {
  date: string;
  pattern_type: string;
  direction: string;
  confidence: number;
}

export interface AnalystConsensus {
  as_of_date: string;
  strong_buy: number;
  buy: number;
  hold: number;
  sell: number;
  strong_sell: number;
  mean_price_target: number | null;
}

export interface NewsItem {
  title: string | null;
  published_at: string | null;
  tier: number | null;
  matched_theme: string | null;
  event_type: string | null;
  sentiment_score: number | null;
  source: string | null;
  source_url: string | null;
}

export interface StockDetail {
  symbol: string;
  summary: TickerSummary;
  score: ScreenerRow | null;
  prices: PriceBar[];
  forecasts: ForecastRow[];
  patterns: PatternRow[];
  analyst_consensus: AnalystConsensus | null;
  news: NewsItem[];
  short_interest: ShortInterestReading | null;
  risk: RiskProfile | null;
  monte_carlo: MonteCarloFan | null;
  macro_overlay: MacroOverlay | null;
}

/**
 * Section 24's two readings, deliberately never collapsed into one verdict:
 * heavy shorting can mean informed money is betting against the company, or it
 * can set up a squeeze. `elevated` is a flag, not a direction.
 */
export interface ShortInterestReading {
  pct_float_short: number | null;
  days_to_cover: number | null;
  elevated: boolean;
}

/**
 * Every field is nullable because each estimator declines independently when
 * its own data floor is unmet. `ratio_min_observations` lets the client explain
 * *why* Sharpe/Sortino are absent instead of rendering two bare dashes.
 */
export interface RiskProfile {
  historical_volatility: number | null;
  implied_volatility: number | null;
  implied_premium: number | null;
  beta: number | null;
  beta_r_squared: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  value_at_risk: number | null;
  expected_shortfall: number | null;
  var_confidence: number | null;
  n_observations: number;
  ratio_min_observations: number;
}

export interface MonteCarloBand {
  day: number;
  lower: number;
  median: number;
  upper: number;
}

export interface MonteCarloFan {
  horizon_days: number;
  n_paths: number;
  n_train: number;
  mu: number;
  sigma: number;
  last_close: number;
  bands: MonteCarloBand[];
}

export interface MacroOverlayComponent {
  driver: string;
  sensitivity: number;
  move: number | null;
}

export interface MacroOverlay {
  sector: string;
  adjustment: number;
  components: MacroOverlayComponent[];
}

export interface SectorStrength {
  sector: string;
  relative_return: number;
  n_symbols: number;
}

export interface RegimePoint {
  date: string;
  vix_level: number | null;
  breadth_pct_above_200dma: number | null;
  macro_news_tone: number | null;
  yield_curve_spread: number | null;
  regime_score: number | null;
  regime_label: string | null;
}

export interface BacktestRun {
  run_date: string;
  period_start: string | null;
  period_end: string | null;
  cadence: string;
  n_periods: number;
  sharpe: number | null;
  sharpe_ci_low: number | null;
  sharpe_ci_high: number | null;
  cagr: number | null;
  cagr_ci_low: number | null;
  cagr_ci_high: number | null;
  ci_confidence_level: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  benchmark_cagr: number | null;
  benchmark_sharpe: number | null;
  avg_turnover: number | null;
  assumed_txn_cost: number;
}

/** The seven scoring categories, in the order the engine defines them. */
export const CATEGORIES = [
  "fundamental",
  "technical",
  "analyst",
  "sentiment",
  "momentum",
  "industry_macro",
  "smart_money",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const SUBSCORE_KEYS: Record<Category, keyof ScreenerRow> = {
  fundamental: "fundamental_score",
  technical: "technical_score",
  analyst: "analyst_score",
  sentiment: "sentiment_score",
  momentum: "momentum_score",
  industry_macro: "industry_macro_score",
  smart_money: "smart_money_score",
};
