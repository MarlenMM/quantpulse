import { useMemo, useState } from "react";
import { ErrorBox, Loading, RatingChip } from "../components/Common";
import { api } from "../lib/api";
import { Link } from "../lib/router";
import { CATEGORIES, SUBSCORE_KEYS, type ScreenerRow } from "../lib/types";
import { confidenceLabel, formatScore, humanize, searchSymbols } from "../lib/format";
import { useApi } from "../lib/useApi";

/** The balanced profile's default weights (Section 7.5's table). */
const DEFAULT_WEIGHTS: Record<string, number> = {
  fundamental: 0.25,
  technical: 0.2,
  analyst: 0.1,
  sentiment: 0.1,
  momentum: 0.15,
  industry_macro: 0.1,
  smart_money: 0.1,
};

/**
 * Recompute the composite from stored sub-scores under caller-chosen weights.
 *
 * Mirrors `scoring.build_composite`'s coverage rule exactly: divide by the
 * weight that actually had data, so a stock missing a category is neither
 * penalised with a phantom zero nor silently boosted. Getting this wrong would
 * make the sliders quietly disagree with the stored ranking — worse than having
 * no sliders at all.
 */
function reweight(row: ScreenerRow, weights: Record<string, number>): number | null {
  let weighted = 0;
  let available = 0;
  for (const category of CATEGORIES) {
    const value = row[SUBSCORE_KEYS[category]] as number | null;
    const weight = weights[category] ?? 0;
    if (value !== null && value !== undefined) {
      weighted += value * weight;
      available += weight;
    }
  }
  return available > 0 ? weighted / available : null;
}

/** Lower percentile bound of each rating below Strong Buy (Section 7.5 step 4). */
const RATING_CUTOFFS: [number, string][] = [
  [70, "buy"],
  [30, "hold"],
  [10, "sell"],
];

/**
 * Re-rate the re-weighted universe: top decile Strong Buy, next 20% Buy, and so on.
 *
 * Moving a slider changed the Score column while the Rating chip kept showing
 * the stored balanced-profile verdict, so a name could sit at the top of a
 * re-weighted table labelled "Sell". The rating has to follow the score it is
 * displayed next to.
 *
 * Two details that keep this agreeing with the Python that wrote the stored
 * ratings. Percentiles are computed over the **whole scored universe**, before
 * any sector or search filtering, because a relative rating means "top decile
 * of the market" and not "top decile of what I am looking at". And the
 * Strong-Buy cutoff comes from the API rather than being hardcoded at 90: the
 * Market Regime Index lifts it toward 95 in a risk-off market (Section 7.3
 * Tier 3), and re-deriving that here would be a second implementation of a
 * market-wide judgment call.
 */
function rateAll(
  scores: Map<string, number | null>,
  strongBuyCutoff: number,
): Map<string, string | null> {
  const scored = [...scores.entries()].filter(([, v]) => v !== null) as [string, number][];
  scored.sort((a, b) => a[1] - b[1]);
  const out = new Map<string, string | null>();
  for (const [symbol] of scores) out.set(symbol, null);
  // Average rank for ties, matching pandas' `rank(pct=True)` default, so a
  // block of equal scores cannot straddle a cutoff by array order.
  let i = 0;
  while (i < scored.length) {
    let j = i;
    while (j + 1 < scored.length && scored[j + 1][1] === scored[i][1]) j += 1;
    const percentile = ((i + j + 2) / 2 / scored.length) * 100;
    const rating =
      percentile >= strongBuyCutoff
        ? "strong_buy"
        : (RATING_CUTOFFS.find(([cutoff]) => percentile >= cutoff)?.[1] ?? "strong_sell");
    for (let k = i; k <= j; k += 1) out.set(scored[k][0], rating);
    i = j + 1;
  }
  return out;
}

export default function Screener() {
  const { data, error, loading } = useApi(() => api.screener(), []);
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("");
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);

  const sectors = useMemo(
    () => [...new Set((data?.rows ?? []).map((r) => r.sector).filter(Boolean))].sort() as string[],
    [data],
  );

  const rows = useMemo(() => {
    const all = data?.rows ?? [];
    // Score and rate the WHOLE universe first, then filter: a relative rating
    // is defined against the market, not against the current sector filter.
    const scores = new Map(all.map((row) => [row.symbol, reweight(row, weights)]));
    const ratings = rateAll(scores, data?.strong_buy_cutoff ?? 90);

    let result = all;
    if (sector) result = result.filter((r) => r.sector === sector);
    if (query.trim()) result = searchSymbols(result, query);
    const scored = result.map((row) => ({
      row,
      custom: scores.get(row.symbol) ?? null,
      rating: ratings.get(row.symbol) ?? row.rating,
    }));
    if (!query.trim()) scored.sort((a, b) => (b.custom ?? -Infinity) - (a.custom ?? -Infinity));
    return scored;
  }, [data, query, sector, weights]);

  if (loading) return <Loading what="the screener" />;
  if (error) return <ErrorBox error={error} />;
  if (!data || data.count === 0) {
    return <p className="muted">No composite scores stored yet — run the refresh job.</p>;
  }

  return (
    <>
      <h1>Screener</h1>
      <p className="muted small">
        Ranking as of <strong>{data.as_of}</strong> · {data.count} symbols scored. Ratings are{" "}
        <em>{data.rating_mode}</em> — the top decile is Strong Buy however the market as a
        whole looks.
      </p>

      <div className="controls">
        <label>
          Search symbol or company
          <input
            type="search"
            value={query}
            placeholder="e.g. GOOGL or Alphabet"
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label>
          Sector
          <select value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>

      <details className="panel">
        <summary>Re-weight categories</summary>
        <p className="muted small">
          Score <em>and rating</em> are recomputed instantly from stored sub-scores — no
          pipeline re-run, because the stored sub-scores are weight-independent by design.
          Ratings are always ranked against the whole scored universe, not against
          whatever the filters left, so "top decile" keeps meaning the same thing.
        </p>
        <div className="sliders">
          {CATEGORIES.map((category) => (
            <label key={category}>
              <span>{humanize(category)}</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={weights[category] ?? 0}
                onChange={(e) =>
                  setWeights({ ...weights, [category]: Number(e.target.value) })
                }
              />
              <output>{(weights[category] ?? 0).toFixed(2)}</output>
            </label>
          ))}
        </div>
        <button type="button" onClick={() => setWeights(DEFAULT_WEIGHTS)}>
          Reset to balanced
        </button>
      </details>

      <table>
        <thead>
          <tr>
            <th scope="col">Symbol</th>
            <th scope="col">Company</th>
            <th scope="col">Sector</th>
            <th scope="col">Rating</th>
            <th scope="col" className="num">Score</th>
            <th scope="col" className="num">Stored</th>
            <th scope="col">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ row, custom, rating }) => (
            <tr key={row.symbol}>
              <td><Link to={`/stocks/${row.symbol}`}>{row.symbol}</Link></td>
              <td>{row.name ?? "—"}</td>
              <td>{row.sector ?? "—"}</td>
              <td><RatingChip rating={rating} /></td>
              <td className="num">{formatScore(custom)}</td>
              <td className="num muted">{formatScore(row.composite_score)}</td>
              <td className="muted small">{confidenceLabel(row.data_confidence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 ? <p className="muted">No symbols match these filters.</p> : null}
    </>
  );
}
