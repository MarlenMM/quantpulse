/**
 * The demo's portfolio, computed in the visitor's browser (finding 33).
 *
 * The full Portfolio Manager is Streamlit; the public demo is static files, and
 * a per-visitor portfolio cannot be pre-rendered. So this is a TypeScript port
 * of the engine's rules, run over the stock files the demo already publishes:
 *
 * - FIFO tax lots and realized gains  -- `portfolio/transactions.py`
 * - per-holding Add/Hold/Trim/Sell, concentration, sector gaps
 *                                     -- `portfolio/recommendations.py`
 * - sector weights                    -- `portfolio/holdings.sector_weights`
 * - volatility, VaR, drawdown, correlation of a constant-weight portfolio
 *                                     -- `analysis/risk.portfolio_risk`
 *
 * **Two implementations of one set of rules is how this project's front ends
 * have disagreed before**, so both are pinned to one file,
 * `tests/fixtures/portfolio-golden.json`: `tests/unit/test_portfolio_golden.py`
 * recomputes it with the engine, `tests/portfolio.spec.ts` recomputes it with
 * `analyse()` below, and each must match it. `analyse()` is also exactly what
 * the Portfolio page renders, so the page's numbers are the tested ones.
 *
 * Every function here is pure: no fetches, no storage, no clock.
 */
import { RATING_DISPLAY } from "./format";

export type Action = "buy" | "sell";
export type Rating = "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";

export interface Transaction {
  symbol: string;
  action: Action;
  shares: number;
  price: number;
  /** ISO date, YYYY-MM-DD. */
  date: string;
}

/** One stock's bars, oldest first: [date, close, adj_close]. */
export type Bars = [string, number, number][];

export interface Inputs {
  cash: number;
  transactions: Transaction[];
  prices: Record<string, Bars>;
  sectors: Record<string, string | null>;
  ratings: Record<string, Rating>;
  sector_candidates: Record<string, string[]>;
}

// ------------------------------------------------------------------ constants

const MIN_LOT_SHARES = 1e-9; // transactions._MIN_LOT_SHARES
export const POSITION_THRESHOLD = 0.15; // recommendations.DEFAULT_POSITION_CONCENTRATION_THRESHOLD
export const SECTOR_THRESHOLD = 0.15; // recommendations.DEFAULT_SECTOR_CONCENTRATION_THRESHOLD
const MAX_GAP_CANDIDATES = 5; // recommendations._MAX_GAP_CANDIDATES
const TRADING_DAYS_PER_YEAR = 252; // backtest.TRADING_DAYS_PER_YEAR
export const VAR_CONFIDENCE = 0.95; // risk.DEFAULT_VAR_CONFIDENCE
const MIN_RETURN_OBS = 20; // risk._MIN_RETURN_OBS
const MIN_TAIL_OBS = 5; // risk._MIN_TAIL_OBS
const MIN_CORRELATION_OBS = 30; // risk._MIN_CORRELATION_OBS

export const KNOWN_GICS_SECTORS = [
  "Communication Services",
  "Consumer Discretionary",
  "Consumer Staples",
  "Energy",
  "Financials",
  "Health Care",
  "Industrials",
  "Information Technology",
  "Materials",
  "Real Estate",
  "Utilities",
] as const;

const ACTION_BY_RATING: Record<Rating, "add" | "hold" | "trim" | "sell"> = {
  strong_buy: "add",
  buy: "add",
  hold: "hold",
  sell: "trim",
  strong_sell: "sell",
};

/**
 * Python's `f"{x:.0%}"`: x*100 rounded half-to-even. `toFixed` rounds a tie up,
 * so 0.125 would read "13%" here and "12%" in the engine's messages.
 */
export function percent0(x: number): string {
  const scaled = x * 100;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let rounded: number;
  if (diff > 0.5) rounded = floor + 1;
  else if (diff < 0.5) rounded = floor;
  else rounded = floor % 2 === 0 ? floor : floor + 1;
  return `${rounded}%`;
}

// ----------------------------------------------------------------- FIFO lots

export interface Lot {
  purchase_date: string;
  shares: number;
  cost_basis: number;
}

export interface Realized {
  symbol: string;
  sale_date: string;
  purchase_date: string;
  shares: number;
  proceeds: number;
  cost_basis: number;
  gain: number;
  term: "short" | "long";
}

/**
 * transactions.holding_term: more than one *calendar* year after purchase.
 *
 * Compared as ISO strings, so a Feb 29 purchase needs no special case: the
 * engine moves its anniversary to Feb 28 in a common year, and here
 * "2025-02-29" -- not a real date -- sorts between 02-28 and 03-01, so the same
 * days count as long. The golden file's edge cases pin exactly that.
 */
export function holdingTerm(purchase: string, asOf: string): "short" | "long" {
  const year = Number(purchase.slice(0, 4)) + 1;
  const oneYearLater = `${String(year).padStart(4, "0")}${purchase.slice(4)}`;
  return asOf > oneYearLater ? "long" : "short";
}

/** transactions.build_lot_book, without splits (the demo publishes none). */
export function buildLotBook(transactions: Transaction[]): {
  open_lots: Record<string, Lot[]>;
  realized: Realized[];
} {
  const bySymbol = new Map<string, { tx: Transaction; order: number }[]>();
  transactions.forEach((tx, order) => {
    if (!Number.isFinite(tx.shares) || !(tx.shares > 0))
      throw new Error(`transaction shares must be > 0, got ${tx.shares} for ${tx.symbol}`);
    if (!Number.isFinite(tx.price) || !(tx.price > 0))
      throw new Error(`transaction price must be > 0, got ${tx.price} for ${tx.symbol}`);
    if (tx.action !== "buy" && tx.action !== "sell")
      throw new Error(`transaction action must be 'buy' or 'sell', got ${tx.action}`);
    const list = bySymbol.get(tx.symbol) ?? [];
    list.push({ tx, order });
    bySymbol.set(tx.symbol, list);
  });

  const openLots: Record<string, Lot[]> = {};
  const realized: Realized[] = [];
  for (const symbol of [...bySymbol.keys()].sort()) {
    const events = bySymbol.get(symbol)!.sort((a, b) =>
      a.tx.date < b.tx.date ? -1 : a.tx.date > b.tx.date ? 1 : a.order - b.order,
    );
    let lots: Lot[] = [];
    for (const { tx } of events) {
      if (tx.action === "buy") {
        lots.push({ purchase_date: tx.date, shares: tx.shares, cost_basis: tx.shares * tx.price });
        continue;
      }
      // _consume_fifo: oldest lots first.
      let remaining = tx.shares;
      const survivors: Lot[] = [];
      for (const lot of lots) {
        if (remaining <= MIN_LOT_SHARES) {
          survivors.push(lot);
          continue;
        }
        const consumed = Math.min(lot.shares, remaining);
        const consumedCost = lot.cost_basis * (consumed / lot.shares);
        const proceeds = consumed * tx.price;
        realized.push({
          symbol,
          sale_date: tx.date,
          purchase_date: lot.purchase_date,
          shares: consumed,
          proceeds,
          cost_basis: consumedCost,
          gain: proceeds - consumedCost,
          term: holdingTerm(lot.purchase_date, tx.date),
        });
        remaining -= consumed;
        const leftover = lot.shares - consumed;
        if (leftover > MIN_LOT_SHARES) {
          survivors.push({
            purchase_date: lot.purchase_date,
            shares: leftover,
            cost_basis: lot.cost_basis - consumedCost,
          });
        }
      }
      if (remaining > MIN_LOT_SHARES) {
        throw new Error(
          `cannot sell ${tx.shares} shares of ${symbol} on ${tx.date}: only ${
            tx.shares - remaining
          } were held`,
        );
      }
      lots = survivors;
    }
    const open = lots.filter((lot) => lot.shares > MIN_LOT_SHARES);
    if (open.length) openLots[symbol] = open;
  }
  realized.sort((a, b) =>
    a.sale_date !== b.sale_date
      ? a.sale_date < b.sale_date ? -1 : 1
      : a.symbol !== b.symbol
        ? a.symbol < b.symbol ? -1 : 1
        : a.purchase_date < b.purchase_date ? -1 : a.purchase_date > b.purchase_date ? 1 : 0,
  );
  return { open_lots: openLots, realized };
}

// ------------------------------------------------------------ recommendations

export interface Recommendation {
  action: "add" | "hold" | "trim" | "sell";
  reason: string;
}

function herfindahl(weights: number[]): number {
  return weights.reduce((sum, w) => (w > 0 ? sum + w * w : sum), 0);
}

function warningsFor(
  weights: [string, number][],
  noun: "position" | "sector",
  threshold: number,
): string[] {
  return weights
    .filter(([, w]) => w > threshold)
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => b.entry[1] - a.entry[1] || a.index - b.index)
    .map(
      ({ entry: [label, w] }) =>
        `${label} is ${percent0(w)} of your portfolio, above the ${percent0(threshold)} ` +
        `${noun}-concentration threshold.`,
    );
}

/** recommendations.recommend, without a rebalance plan (the demo builds none). */
export function recommend(
  holdings: [string, { weight: number; rating: Rating; sector: string | null }][],
  candidates: Record<string, string[]>,
) {
  const sectorWeights = new Map<string, number>();
  for (const [, ctx] of holdings) {
    if (ctx.sector) sectorWeights.set(ctx.sector, (sectorWeights.get(ctx.sector) ?? 0) + ctx.weight);
  }
  const positionHhi = herfindahl(holdings.map(([, ctx]) => ctx.weight));
  const sectorHhi = sectorWeights.size ? herfindahl([...sectorWeights.values()]) : null;
  const warnings = [
    ...warningsFor(
      holdings.map(([symbol, ctx]) => [symbol, ctx.weight]),
      "position",
      POSITION_THRESHOLD,
    ),
    ...(sectorWeights.size ? warningsFor([...sectorWeights.entries()], "sector", SECTOR_THRESHOLD) : []),
  ];

  const recommendations: Record<string, Recommendation> = {};
  for (const [symbol, ctx] of holdings) {
    let action = ACTION_BY_RATING[ctx.rating];
    const capped = action === "add" && ctx.weight >= POSITION_THRESHOLD;
    if (capped) action = "hold";
    let reason = `Rated ${RATING_DISPLAY[ctx.rating].text}.`;
    if (capped) {
      reason +=
        ` Already ${percent0(ctx.weight)} of the portfolio, at or above the ` +
        `${percent0(POSITION_THRESHOLD)} concentration guideline -- held rather than added to.`;
    }
    recommendations[symbol] = { action, reason };
  }

  const held = new Set([...sectorWeights.entries()].filter(([, w]) => w > 0).map(([s]) => s));
  const gaps = KNOWN_GICS_SECTORS.filter((sector) => !held.has(sector)).map((sector) => {
    const names = (candidates[sector] ?? []).slice(0, MAX_GAP_CANDIDATES);
    return `No ${sector} holdings.` + (names.length ? ` Top-ranked ${sector} names to consider: ${names.join(", ")}.` : "");
  });

  const reasons: string[] = [];
  if (warnings.some((w) => w.includes("position-concentration")))
    reasons.push("one or more positions exceed the concentration threshold");
  if (warnings.some((w) => w.includes("sector-concentration")))
    reasons.push("one or more sectors exceed the concentration threshold");
  if (Object.values(recommendations).some((r) => r.action === "sell" || r.action === "trim"))
    reasons.push("one or more holdings are rated Sell or Strong Sell");

  return {
    recommendations,
    concentration: {
      position_hhi: positionHhi,
      position_effective_count: positionHhi > 0 ? 1 / positionHhi : null,
      sector_hhi: sectorHhi,
      sector_effective_count: sectorHhi !== null && sectorHhi > 0 ? 1 / sectorHhi : null,
      warnings,
    },
    sector_gaps: gaps,
    rebalance_reasons: reasons,
  };
}

// ----------------------------------------------------------------------- risk

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sampleStd(values: number[]): number {
  const m = mean(values);
  return Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1));
}

/** numpy.quantile's default ("linear") interpolation. */
function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = q * (sorted.length - 1);
  const lo = Math.floor(position);
  const hi = Math.ceil(position);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}

/** Pearson over pairwise-complete observations; null below the overlap floor. */
function correlation(a: (number | null)[], b: (number | null)[]): number | null {
  const xs: number[] = [];
  const ys: number[] = [];
  a.forEach((x, i) => {
    const y = b[i];
    if (x !== null && y !== null) {
      xs.push(x);
      ys.push(y);
    }
  });
  if (xs.length < MIN_CORRELATION_OBS) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  const denominator = Math.sqrt(sxx * syy);
  return denominator > 0 ? sxy / denominator : null;
}

/**
 * risk.portfolio_risk over the held names' adjusted closes, at current weights.
 *
 * Returns are each name's move against its own previous *observed* bar on the
 * shared date axis, so a return touching a missing day stays missing
 * (`risk.returns_panel`); the portfolio keeps only dates where every held name
 * has one (`risk.portfolio_returns`). Like the engine, this is a hypothetical:
 * today's weights applied to the past, not the path the holdings took.
 */
export function portfolioRisk(
  prices: Record<string, Bars>,
  weights: [string, number][],
  cashWeight: number,
) {
  // The engine is handed only holdings with a price series (the page's
  // `usable`), and normalizes by their weights plus cash.
  const usable = weights.filter(([symbol]) => prices[symbol]?.length);
  const held = usable.map(([symbol]) => symbol);
  const dates = [...new Set(held.flatMap((symbol) => prices[symbol].map(([d]) => d)))].sort();
  const byDate = new Map(held.map((s) => [s, new Map(prices[s].map(([d, , adj]) => [d, adj]))]));
  const returns: Record<string, (number | null)[]> = {};
  for (const symbol of held) {
    const series = byDate.get(symbol)!;
    const r: (number | null)[] = [];
    for (let i = 1; i < dates.length; i += 1) {
      const now = series.get(dates[i]);
      const before = series.get(dates[i - 1]);
      r.push(now !== undefined && before !== undefined && now > 0 && before > 0 ? now / before - 1 : null);
    }
    returns[symbol] = r;
  }

  const total = usable.reduce((s, [, w]) => s + w, 0) + cashWeight;
  const portfolio: number[] = [];
  for (let i = 0; i < dates.length - 1; i += 1) {
    if (held.some((s) => returns[s][i] === null)) continue;
    portfolio.push(usable.reduce((s, [symbol, w]) => s + (returns[symbol][i] as number) * (w / total), 0));
  }

  const volatility =
    portfolio.length >= MIN_RETURN_OBS ? sampleStd(portfolio) * Math.sqrt(TRADING_DAYS_PER_YEAR) : null;

  let equity = 1;
  let peak = 1;
  let maxDrawdown = 0;
  for (const r of portfolio) {
    equity *= 1 + r;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.min(maxDrawdown, equity / peak - 1);
  }

  let valueAtRisk: number | null = null;
  let expectedShortfall: number | null = null;
  if (portfolio.length >= Math.ceil(MIN_TAIL_OBS / (1 - VAR_CONFIDENCE))) {
    const cut = quantile(portfolio, 1 - VAR_CONFIDENCE);
    const tail = portfolio.filter((r) => r <= cut);
    valueAtRisk = -cut;
    expectedShortfall = -(tail.length ? mean(tail) : cut);
  }

  const correlations: Record<string, Record<string, number | null>> = {};
  const pairs: number[] = [];
  held.forEach((a, i) => {
    correlations[a] = {};
    held.forEach((b, j) => {
      const value = correlation(returns[a], returns[b]);
      correlations[a][b] = value;
      if (j > i && value !== null) pairs.push(value);
    });
  });

  return {
    n_observations: portfolio.length,
    volatility,
    max_drawdown: maxDrawdown,
    var: valueAtRisk,
    expected_shortfall: expectedShortfall,
    average_correlation: pairs.length ? mean(pairs) : null,
    correlations,
  };
}

// ------------------------------------------------------------------ analyse

/** Everything the Portfolio page shows, in the golden file's shape. */
export function analyse(inputs: Inputs) {
  const book = buildLotBook(inputs.transactions);
  const lastClose = (symbol: string): number | null => {
    const bars = inputs.prices[symbol];
    const close = bars?.length ? bars[bars.length - 1][1] : null;
    return close !== null && Number.isFinite(close) && close > 0 ? close : null;
  };

  const symbols = Object.keys(book.open_lots).sort();
  const priced = symbols.map((symbol) => {
    const lots = book.open_lots[symbol];
    const shares = lots.reduce((s, lot) => s + lot.shares, 0);
    const cost = lots.reduce((s, lot) => s + lot.cost_basis, 0);
    const price = lastClose(symbol);
    const value = price !== null ? shares * price : null;
    return { symbol, shares, cost, price, value };
  });
  const total = priced.reduce((s, p) => s + (p.value ?? 0), 0) + inputs.cash;
  const weights: [string, number][] = priced.map((p) => [p.symbol, (p.value ?? 0) / total]);

  const positions: Record<string, unknown> = {};
  for (const p of priced) {
    positions[p.symbol] = {
      shares: p.shares,
      cost_basis: p.cost,
      average_cost: p.cost / p.shares,
      current_price: p.price,
      market_value: p.value,
      unrealized_gain: p.value !== null ? p.value - p.cost : null,
      weight: (p.value ?? 0) / total,
    };
  }

  // holdings.sector_weights: share of *invested* value, "Unclassified" kept.
  const invested = priced.reduce((s, p) => (p.value && p.value > 0 ? s + p.value : s), 0);
  const sectorWeights: Record<string, number> = {};
  if (invested > 0) {
    for (const p of priced) {
      if (!p.value || p.value <= 0) continue;
      const sector = inputs.sectors[p.symbol] || "Unclassified";
      sectorWeights[sector] = (sectorWeights[sector] ?? 0) + p.value / invested;
    }
  }

  const advice = recommend(
    weights
      .filter(([symbol]) => inputs.ratings[symbol] !== undefined)
      .map(([symbol, weight]) => [
        symbol,
        { weight, rating: inputs.ratings[symbol], sector: inputs.sectors[symbol] ?? null },
      ]),
    inputs.sector_candidates,
  );

  return {
    positions,
    open_lots: Object.fromEntries(symbols.map((s) => [s, book.open_lots[s]])),
    realized: book.realized,
    total_value: total,
    sector_weights: sectorWeights,
    ...advice,
    risk: portfolioRisk(inputs.prices, weights, total > 0 ? inputs.cash / total : 0),
  };
}
