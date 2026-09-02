import { useMemo, useState } from "react";
import { ErrorBox, LoadingTable, RatingChip } from "../components/Common";
import { Tip } from "../components/Tip";
import { api } from "../lib/api";
import { Link } from "../lib/router";
import { CATEGORIES, SUBSCORE_KEYS, type ScreenerRow } from "../lib/types";
import {
  RATING_ORDER,
  confidenceLabel,
  formatScore,
  humanize,
  searchSymbols,
} from "../lib/format";
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
  const [profile, setProfile] = useState("balanced");
  const [absolute, setAbsolute] = useState(false);
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("");
  const [ratingFilter, setRatingFilter] = useState<string[]>([]);
  const [minConfidence, setMinConfidence] = useState(0);
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);

  const { data: profiles } = useApi(() => api.profiles(), []);
  const selected = profiles?.find((p) => p.name === profile);

  // Only `income` and `conservative` need their own stored ranking; the other
  // four differ by weights alone, so they read the balanced rows and are
  // applied through the sliders below. Requesting a profile that has no stored
  // rows would return an empty table rather than the same names re-weighted.
  const fetchProfile = selected?.rescores ? profile : "balanced";
  const { data, error, loading } = useApi(() => api.screener(fetchProfile), [fetchProfile]);

  // Absolute ratings come from the server: they need
  // `build_composite(rating_mode="absolute")` over the stored raw category
  // values, and a second copy of that mapping here would drift from the engine.
  const { data: absoluteData } = useApi(
    () => (absolute ? api.screenerAbsolute(fetchProfile) : Promise.resolve(null)),
    [absolute, fetchProfile],
  );

  // Selecting a profile loads its weights into the sliders, so the table shows
  // that profile rather than the profile's name over balanced weights.
  const applyProfile = (name: string) => {
    setProfile(name);
    const next = profiles?.find((p) => p.name === name);
    if (next) setWeights({ ...next.weights });
  };

  const sectors = useMemo(
    () => [...new Set((data?.rows ?? []).map((r) => r.sector).filter(Boolean))].sort() as string[],
    [data],
  );

  const absoluteUnavailable = absolute && absoluteData !== null && !absoluteData?.available;

  // Whether the sliders still sit exactly where the chosen profile put them.
  //
  // This decides whether the table shows one score column or two. The Score
  // column is recomputed live from the sliders; "Stored" is the value the
  // nightly wrote. Until a slider moves they are the same number by
  // construction, so showing both by default gave every row two identical
  // columns, one of them labelled with a word that explains nothing about why
  // it is there. It earns its place the moment it disagrees, and not before.
  const reweighted = useMemo(() => {
    const base = selected?.weights;
    if (!base) return false;
    return CATEGORIES.some(
      (category) => Math.abs((weights[category] ?? 0) - (base[category] ?? 0)) > 1e-9,
    );
  }, [selected, weights]);

  const rows = useMemo(() => {
    const all = data?.rows ?? [];
    // Score and rate the WHOLE universe first, then filter: a relative rating
    // is defined against the market, not against the current sector filter.
    const scores = new Map(all.map((row) => [row.symbol, reweight(row, weights)]));
    const ratings = rateAll(scores, data?.strong_buy_cutoff ?? 90);

    const absoluteBySymbol = new Map(
      (absoluteData?.available ? absoluteData.rows : []).map((r) => [r.symbol, r]),
    );
    const useAbsolute = absolute && absoluteBySymbol.size > 0;

    let result = all;
    if (sector) result = result.filter((r) => r.sector === sector);
    if (minConfidence > 0) {
      result = result.filter((r) => (r.data_confidence ?? 0) >= minConfidence);
    }
    if (query.trim()) result = searchSymbols(result, query);

    let scored = result.map((row) => {
      const abs = absoluteBySymbol.get(row.symbol);
      return {
        row,
        custom: useAbsolute ? (abs?.composite_score ?? null) : (scores.get(row.symbol) ?? null),
        rating: useAbsolute
          ? (abs?.rating ?? row.rating)
          : (ratings.get(row.symbol) ?? row.rating),
      };
    });
    // Rating filter applies to what the table actually shows, so it follows
    // whichever scheme is active rather than the stored balanced verdict.
    if (ratingFilter.length) scored = scored.filter((s) => ratingFilter.includes(s.rating));
    if (!query.trim()) scored.sort((a, b) => (b.custom ?? -Infinity) - (a.custom ?? -Infinity));
    return scored;
  }, [data, absoluteData, absolute, query, sector, weights, ratingFilter, minConfidence]);

  if (loading) {
    return (
      <>
        <h1>Screener</h1>
        <LoadingTable what="the ranked universe" rows={12} columns={[12, 30, 20, 14, 10, 14]} />
      </>
    );
  }
  if (error) return <ErrorBox error={error} />;
  if (!data || data.count === 0) {
    return (
      <>
        <h1>Screener</h1>
        <p className="standfirst">
          No composite scores are stored yet. Run <code>scripts/refresh_data.py</code> and this
          table fills in with the universe it scores.
        </p>
      </>
    );
  }

  return (
    <>
      <h1>Rank the whole universe, your way</h1>
      <p className="standfirst">
        Every scored name, ordered by composite score. Move the category weights and the score{" "}
        <em>and</em> the rating recompute in the browser from stored sub-scores — no pipeline
        re-run, because the sub-scores are weight-independent by design.
      </p>
      <p className="muted small">
        Ranking as of <strong>{data.as_of}</strong> · {data.count} symbols scored · ratings are{" "}
        <em>{data.rating_mode}</em>, so the top decile is Strong Buy however the market as a
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
        <label>
          Rating
          <select
            multiple
            value={ratingFilter}
            onChange={(e) =>
              setRatingFilter([...e.target.selectedOptions].map((o) => o.value))
            }
          >
            {RATING_ORDER.map((r) => (
              <option key={r} value={r}>{humanize(r)}</option>
            ))}
          </select>
        </label>
        <label>
          Min coverage <output>{minConfidence}%</output>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
          />
        </label>
      </div>

      <details>
        <summary>Investor profile &amp; rating scheme</summary>
        <div className="controls">
          <label>
            Start from profile
            <select value={profile} onChange={(e) => applyProfile(e.target.value)}>
              {(profiles ?? []).map((p) => (
                <option key={p.name} value={p.name}>{humanize(p.name)}</option>
              ))}
            </select>
          </label>
          <label>
            Rating scheme
            <select
              value={absolute ? "absolute" : "relative"}
              onChange={(e) => setAbsolute(e.target.value === "absolute")}
            >
              <option value="relative">Relative</option>
              <option value="absolute">Absolute</option>
            </select>
          </label>
        </div>
        {selected && <p className="note">{selected.description}</p>}
        {selected?.rescores && (
          <p className="note">
            Scored under the <strong>{humanize(profile)}</strong> profile — its sub-scores
            are genuinely different, not the balanced ones re-weighted.
          </p>
        )}
        <p className="note">
          <strong>Relative</strong> always names a top decile Strong Buy, however the whole
          market looks — that is the plan's own warning, not a bug. <strong>Absolute</strong>{" "}
          measures every category against a fixed bar instead, so a broadly falling market
          genuinely produces fewer Strong Buys (and can produce none).
        </p>
        {absoluteUnavailable && (
          <p className="callout callout-warn">
            These rows were scored before raw category values were stored, so an absolute
            rating cannot be derived from them — showing the relative ranking instead. The
            next refresh will populate them.
          </p>
        )}
      </details>

      <details>
        <summary>Re-weight categories</summary>
        <p className="note">
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

      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Symbol</th>
              <th scope="col">Company</th>
              <th scope="col">Sector</th>
              <th scope="col">
                Rating
                <Tip term="Rating" />
              </th>
              <th scope="col" className="num">
                Score
                <Tip term="Composite score" />
              </th>
              {reweighted ? (
                <th scope="col" className="num">
                  Stored
                  <Tip
                    label="the stored score"
                    text="What the nightly job scored this stock at, under the profile's own weights. The Score column beside it is your slider settings applied to the same stored sub-scores — this column is what you are comparing against."
                  />
                </th>
              ) : null}
              <th scope="col">
                Coverage
                <Tip term="Data coverage" />
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ row, custom, rating }) => (
              <tr key={row.symbol}>
                <td>
                  <Link to={`/stocks/${row.symbol}`}>
                    <span className="ticker">{row.symbol}</span>
                  </Link>
                </td>
                <td>{row.name ?? "—"}</td>
                <td>{row.sector ?? "—"}</td>
                <td><RatingChip rating={rating} /></td>
                <td className="num">{formatScore(custom)}</td>
                {reweighted ? (
                  <td className="num muted">{formatScore(row.composite_score)}</td>
                ) : null}
                <td className="muted small">{confidenceLabel(row.data_confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 ? (
        <p className="muted" style={{ marginTop: "var(--s4)" }}>
          Nothing matches these filters. The coverage slider is the usual culprit — it hides
          any name whose data is thinner than the threshold.
        </p>
      ) : null}
    </>
  );
}
