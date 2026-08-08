import { Chart } from "../components/Chart";
import { EmptyState, ErrorBox, Loading, Metric, RatingChip } from "../components/Common";
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
} from "../lib/format";
import { useApi } from "../lib/useApi";

export default function Dashboard() {
  const health = useApi(() => api.health(), []);
  const screener = useApi(() => api.screener(), []);
  const regime = useApi(() => api.regime(90), []);
  const news = useApi(() => api.news(6), []);
  const changes = useApi(() => api.ratingChanges(8), []);
  const rotation = useApi(() => api.sectorRotation(), []);

  if (health.loading) return <Loading what="the dashboard" />;
  if (health.error) return <ErrorBox error={health.error} />;

  // A fresh clone has an empty database. Say so, and say how to fix it —
  // a grid of blank panels reads as a broken deploy, not an un-run pipeline.
  if (health.data && !health.data.has_data) {
    return (
      <EmptyState>
        <h2>No analysis data yet</h2>
        <p>The pipeline hasn't been run against this database.</p>
        <pre>
          uv run alembic upgrade head{"\n"}
          uv run python scripts/seed_initial_data.py{"\n"}
          uv run python scripts/refresh_data.py
        </pre>
      </EmptyState>
    );
  }

  const latestRegime = regime.data?.at(-1) ?? null;

  return (
    <>
      <h1>Dashboard</h1>
      {health.data ? (
        <p className="muted small">
          Data freshness —{" "}
          {Object.entries(health.data.freshness)
            .map(([name, value]) => `${humanize(name)}: ${freshnessLabel(value)}`)
            .join(" · ")}
        </p>
      ) : null}

      <section className="grid-2">
        <div className="panel">
          <h2>Today's top-ranked names</h2>
          {screener.loading ? <Loading /> : null}
          {screener.error ? <ErrorBox error={screener.error} /> : null}
          {screener.data ? (
            <table>
              <thead>
                <tr>
                  <th scope="col">Symbol</th>
                  <th scope="col">Company</th>
                  <th scope="col">Rating</th>
                  <th scope="col" className="num">Score</th>
                  <th scope="col">Coverage</th>
                </tr>
              </thead>
              <tbody>
                {screener.data.rows.slice(0, 10).map((row) => (
                  <tr key={row.symbol}>
                    <td>
                      <Link to={`/stocks/${row.symbol}`}>{row.symbol}</Link>
                    </td>
                    <td>{row.name ?? "—"}</td>
                    <td><RatingChip rating={row.rating} /></td>
                    <td className="num">{formatScore(row.composite_score)}</td>
                    <td className="muted small">{confidenceLabel(row.data_confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          <p className="small">
            <Link to="/screener">Open the full Screener →</Link>
          </p>
        </div>

        <div className="panel">
          <h2>
            Market Regime
            <Tip term="Market Regime Index" label="the Market Regime Index" />
          </h2>
          {regime.loading ? <Loading /> : null}
          {latestRegime && latestRegime.regime_score !== null ? (
            <>
              <Chart
                ariaLabel={`Market regime score ${latestRegime.regime_score.toFixed(0)} of 100, ${humanize(latestRegime.regime_label)}`}
                height={240}
                data={[
                  {
                    type: "indicator",
                    mode: "gauge+number",
                    value: latestRegime.regime_score,
                    title: { text: humanize(latestRegime.regime_label) },
                    gauge: {
                      axis: { range: [0, 100] },
                      bar: { color: "#3b82f6" },
                      steps: [
                        { range: [0, 35], color: "rgba(207,34,46,0.25)" },
                        { range: [35, 65], color: "rgba(154,103,0,0.20)" },
                        { range: [65, 100], color: "rgba(45,164,78,0.25)" },
                      ],
                    },
                  } as never,
                ]}
              />
              <div className="metrics">
                <Metric label="VIX" value={formatScore(latestRegime.vix_level)} term="VIX" />
                <Metric label="Breadth >200DMA" value={formatPctAlreadyScaled(latestRegime.breadth_pct_above_200dma)} term="Market breadth" />
                <Metric label="10Y-2Y" value={formatScore(latestRegime.yield_curve_spread, 2)} term="Yield curve spread" />
                <Metric label="Macro tone" value={formatScore(latestRegime.macro_news_tone, 2)} term="Sentiment score" />
              </div>
            </>
          ) : (
            <p className="muted">Market Regime Index hasn't been computed yet.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Sector rotation</h2>
        <p className="muted small">
          Change in each sector's strength <em>relative to the market</em> over the last month —
          the top row is where money has been rotating in. A sector can appear here with a
          positive number while falling in absolute terms, if it simply fell less than everything
          else. This describes what already happened; it is not a forecast.
        </p>
        {rotation.data && rotation.data.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th scope="col">Sector</th>
                <th scope="col" className="num">vs market (1m)</th>
                <th scope="col" className="num">Names</th>
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
        ) : (
          <p className="muted small">
            Needs price history across several sectors before relative strength means anything.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>What changed since the last refresh</h2>
        {changes.data && changes.data.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th scope="col">Symbol</th>
                <th scope="col">From</th>
                <th scope="col">To</th>
                <th scope="col" className="num">Score Δ</th>
              </tr>
            </thead>
            <tbody>
              {changes.data.map((change) => (
                <tr key={change.symbol}>
                  <td><Link to={`/stocks/${change.symbol}`}>{change.symbol}</Link></td>
                  <td><RatingChip rating={change.previous_rating} /></td>
                  <td><RatingChip rating={change.rating} /></td>
                  <td className="num">{formatScore(change.score_change)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted small">
            Needs at least two stored scoring snapshots — the point-in-time schema
            makes this view free once they exist.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>Today's market-moving news</h2>
        <p className="muted small">Tier-2 (industry) and Tier-3 (macro) stories.</p>
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
                <div className="muted small">
                  Tier {item.tier ?? "—"}
                  {item.event_type ? ` · ${humanize(item.event_type)}` : ""} · sentiment{" "}
                  {formatScore(item.sentiment_score, 2)}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted small">No Tier-2/3 stories ingested recently.</p>
        )}
      </section>
    </>
  );
}
