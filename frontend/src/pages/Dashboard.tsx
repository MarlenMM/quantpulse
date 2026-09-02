import { Chart } from "../components/Chart";
import {
  EmptyState,
  ErrorBox,
  Loading,
  LoadingMetrics,
  LoadingTable,
  Metric,
  RatingChip,
} from "../components/Common";
import { Tip } from "../components/Tip";
import { api } from "../lib/api";
import { Link } from "../lib/router";
import {
  confidenceLabel,
  formatPctAlreadyScaled,
  formatScore,
  formatSignedPercent,
  freshnessLabel,
  humanize,
  isBehind,
} from "../lib/format";
import { useThemeTokens, withAlpha } from "../lib/theme";
import { useApi } from "../lib/useApi";

function freshnessTone(source: string, label: string): string {
  if (label === "never run") return "f-age is-never";
  return isBehind(source, label) ? "f-age is-stale" : "f-age";
}

/**
 * When each source last ran, as a scannable strip.
 *
 * This used to be one line of nine `name: age` pairs joined by middots, which
 * wrapped to three lines of grey text that nobody read — and it is the single
 * most important thing on the page for judging whether any other number here is
 * worth anything. As a row of label-over-value pairs it can be skimmed, and a
 * source that is behind is coloured so it can be found without reading all
 * nine.
 */
function Freshness({ freshness }: { freshness: Record<string, string | null> }) {
  const entries = Object.entries(freshness);
  if (entries.length === 0) return null;
  return (
    <section>
      <h2>Data freshness</h2>
      <ul className="freshness">
        {entries.map(([name, value]) => {
          const label = freshnessLabel(value);
          return (
            <li key={name}>
              <span className="f-name">{humanize(name)}</span>
              <span className={freshnessTone(name, label)}>{label}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default function Dashboard() {
  const health = useApi(() => api.health(), []);
  const screener = useApi(() => api.screener(), []);
  const regime = useApi(() => api.regime(90), []);
  const news = useApi(() => api.news(6), []);
  const changes = useApi(() => api.ratingChanges(8), []);
  const rotation = useApi(() => api.sectorRotation(), []);
  const theme = useThemeTokens();

  if (health.loading) return <Loading what="the dashboard" />;
  if (health.error) return <ErrorBox error={health.error} />;

  // A fresh clone has an empty database. Say so, and say how to fix it —
  // a grid of blank panels reads as a broken deploy, not an un-run pipeline.
  if (health.data && !health.data.has_data) {
    return (
      <EmptyState>
        <h3>Nothing has been scored yet</h3>
        <p>
          The database exists but the pipeline has never been run against it. Three commands,
          in this order:
        </p>
        <pre>
          uv run alembic upgrade head{"\n"}
          uv run python scripts/seed_initial_data.py{"\n"}
          uv run python scripts/refresh_data.py
        </pre>
        <p className="small muted">
          The middle one is the cold-start backfill — years of history for the whole universe,
          and the slow one. The third is the incremental refresh you run from then on.
        </p>
      </EmptyState>
    );
  }

  const latestRegime = regime.data?.at(-1) ?? null;

  return (
    <>
      <h1>Today's read</h1>
      <p className="standfirst">
        The S&amp;P 500, scored overnight across seven categories of public data — fundamentals,
        technicals, analyst estimates, news sentiment, momentum, macro and institutional
        filings. This page is the market-wide view: what ranks highest, what the model changed
        its mind about, and what regime it is all happening in.
      </p>

      {health.data ? <Freshness freshness={health.data.freshness} /> : null}

      <div className="split lede">
        {/* The subject of the page. It gets the raised surface, the serif
            heading and the width; everything else on this page is context for
            reading it. Giving all five sections the same card and the same
            heading weight is how a page ends up with no subject at all. */}
        <section className="card">
          <h2 className="h-lede">Today's top-ranked names</h2>
          {screener.loading ? <LoadingTable what="the ranking" rows={10} /> : null}
          {screener.error ? <ErrorBox error={screener.error} /> : null}
          {screener.data ? (
            <>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Symbol</th>
                      <th scope="col">Company</th>
                      <th scope="col">Rating</th>
                      <th scope="col" className="num">
                        Score
                        <Tip term="Composite score" />
                      </th>
                      <th scope="col">Coverage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {screener.data.rows.slice(0, 12).map((row) => (
                      <tr key={row.symbol}>
                        <td>
                          <Link to={`/stocks/${row.symbol}`}>
                            <span className="ticker">{row.symbol}</span>
                          </Link>
                        </td>
                        <td>{row.name ?? "—"}</td>
                        <td>
                          <RatingChip rating={row.rating} />
                        </td>
                        <td className="num">{formatScore(row.composite_score)}</td>
                        <td className="muted small">{confidenceLabel(row.data_confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note" style={{ marginTop: "var(--s4)" }}>
                Ranked by composite score out of {screener.data.count} scored symbols.{" "}
                <Link to="/screener">
                  Open the Screener to re-weight the categories and re-rank →
                </Link>
              </p>
            </>
          ) : null}
        </section>

        <section>
          <h2>
            Market regime
            <Tip term="Market Regime Index" label="the Market Regime Index" />
          </h2>
          {regime.loading ? <LoadingMetrics what="the regime index" count={4} /> : null}
          {latestRegime && latestRegime.regime_score !== null ? (
            <>
              <Chart
                ariaLabel={`Market regime score ${latestRegime.regime_score.toFixed(0)} of 100, ${humanize(latestRegime.regime_label)}`}
                height={200}
                data={[
                  {
                    type: "indicator",
                    mode: "gauge+number",
                    value: latestRegime.regime_score,
                    title: { text: humanize(latestRegime.regime_label) },
                    number: { font: { size: 30 } },
                    gauge: {
                      axis: { range: [0, 100], tickcolor: theme.grid },
                      bar: { color: theme.accent, thickness: 0.7 },
                      bgcolor: "rgba(0,0,0,0)",
                      borderwidth: 0,
                      // The three zones are the only decoration on this
                      // figure, and they are not decoration: they are the
                      // risk-off / neutral / risk-on bands the score is read
                      // against, in the same red/amber/green the ratings use.
                      steps: [
                        { range: [0, 35], color: withAlpha(theme.down, 0.16) },
                        { range: [35, 65], color: withAlpha(theme.muted, 0.12) },
                        { range: [65, 100], color: withAlpha(theme.up, 0.16) },
                      ],
                    },
                  } as never,
                ]}
                layout={{ margin: { l: 24, r: 24, t: 34, b: 0 } }}
              />
              <div className="metrics">
                <Metric label="VIX" value={formatScore(latestRegime.vix_level)} term="VIX" />
                <Metric
                  label="Breadth >200DMA"
                  value={formatPctAlreadyScaled(latestRegime.breadth_pct_above_200dma)}
                  term="Market breadth"
                />
                <Metric
                  label="10Y−2Y"
                  value={formatScore(latestRegime.yield_curve_spread, 2)}
                  term="Yield curve spread"
                />
                <Metric
                  label="Macro tone"
                  value={formatScore(latestRegime.macro_news_tone, 2)}
                  term="Sentiment score"
                />
              </div>
              <p className="note">
                Built here from four inputs — the VIX percentile, index breadth, macro news tone
                and the yield-curve spread — not read off a paywalled index. It moves the
                Strong Buy cutoff: in a risk-off market the top decile has to clear a higher
                bar before the model will call anything a Strong Buy.
              </p>
            </>
          ) : regime.loading ? null : (
            <p className="muted small">
              The regime index has not been computed yet. It needs the macro series, which the
              weekly refresh gathers.
            </p>
          )}
        </section>
      </div>

      <section className="block">
        <h2>Where money rotated this month</h2>
        {rotation.loading ? <LoadingTable what="sector rotation" rows={5} columns={[46, 28, 18]} /> : null}
        {rotation.data && rotation.data.length > 0 ? (
          <>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Sector</th>
                    <th scope="col" className="num">
                      vs market (1m)
                    </th>
                    <th scope="col" className="num">
                      Names
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rotation.data.map((row) => (
                    <tr key={row.sector}>
                      <td>{row.sector}</td>
                      <td className="num">{formatSignedPercent(row.relative_return / 100)}</td>
                      <td className="num">{row.n_symbols}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note" style={{ marginTop: "var(--s3)" }}>
              Change in each sector's strength <em>relative to the market</em> over the last
              month — the top row is where money has been rotating in. A sector can appear here
              with a positive number while falling in absolute terms, if it simply fell less
              than everything else. This describes what already happened; it is not a forecast.
            </p>
          </>
        ) : rotation.loading ? null : (
          <p className="muted small">
            Needs price history across several sectors before relative strength means anything.
          </p>
        )}
      </section>

      <section className="block">
        <h2>What the model changed its mind about</h2>
        {changes.loading ? <LoadingTable what="rating changes" rows={5} columns={[18, 26, 26, 16]} /> : null}
        {changes.data && changes.data.length > 0 ? (
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Symbol</th>
                  <th scope="col">From</th>
                  <th scope="col">To</th>
                  <th scope="col" className="num">
                    Score Δ
                  </th>
                </tr>
              </thead>
              <tbody>
                {changes.data.map((change) => (
                  <tr key={change.symbol}>
                    <td>
                      <Link to={`/stocks/${change.symbol}`}>
                        <span className="ticker">{change.symbol}</span>
                      </Link>
                    </td>
                    <td>
                      <RatingChip rating={change.previous_rating} />
                    </td>
                    <td>
                      <RatingChip rating={change.rating} />
                    </td>
                    <td className="num">{formatScore(change.score_change)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : changes.loading ? null : (
          <p className="muted small">
            Needs at least two stored scoring snapshots — the point-in-time schema makes this
            view free once they exist.
          </p>
        )}
      </section>

      <section className="block">
        <h2>Market-moving stories</h2>
        {news.loading ? <Loading what="the news feed" /> : null}
        {news.data && news.data.length > 0 ? (
          <ul className="newslist">
            {news.data.map((item, i) => (
              <li key={i}>
                {item.source_url ? (
                  <a href={item.source_url} target="_blank" rel="noreferrer noopener">
                    {item.title ?? "(untitled)"}
                  </a>
                ) : (
                  item.title ?? "(untitled)"
                )}
                <div className="newsmeta">
                  <span>Tier {item.tier ?? "—"}</span>
                  {item.event_type ? <span>{humanize(item.event_type)}</span> : null}
                  <span>
                    Sentiment <span className="n-value">{formatScore(item.sentiment_score, 2)}</span>
                  </span>
                </div>
              </li>
            ))}
          </ul>
        ) : news.loading ? null : (
          <p className="muted small">
            No Tier-2 or Tier-3 stories have been ingested recently. Tier 1 is company-specific
            and appears on each stock's own page instead.
          </p>
        )}
        <p className="note" style={{ marginTop: "var(--s3)" }}>
          Tier 2 is industry and thematic; Tier 3 is macro. Both feed the sentiment category of
          the composite score; neither is shown here as a recommendation.
        </p>
      </section>
    </>
  );
}
