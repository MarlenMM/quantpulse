import { useState } from "react";
import { Chart } from "../components/Chart";
import { ErrorBox, Loading, Metric, RatingChip } from "../components/Common";
import { api } from "../lib/api";
import { CATEGORIES, SUBSCORE_KEYS } from "../lib/types";
import {
  confidenceLabel,
  formatPercent,
  formatPrice,
  formatScore,
  formatSignedPercent,
  humanize,
} from "../lib/format";
import { useApi } from "../lib/useApi";

export default function StockDetail({ symbol }: { symbol: string }) {
  const { data, error, loading } = useApi(() => api.stock(symbol), [symbol]);
  const [model, setModel] = useState<string | null>(null);

  if (loading) return <Loading what={symbol} />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  const score = data.score;
  const models = [...new Set(data.forecasts.map((f) => f.model_name))];
  const selectedModel = model ?? models[0] ?? null;
  const forecasts = data.forecasts.filter((f) => f.model_name === selectedModel);

  // A category with no data is omitted from the radar rather than plotted at
  // zero — a missing sentiment score is not a *bad* sentiment score.
  const radar = CATEGORIES.map((category) => ({
    label: humanize(category),
    value: score ? (score[SUBSCORE_KEYS[category]] as number | null) : null,
  })).filter((entry) => entry.value !== null);

  return (
    <>
      <h1>
        {data.symbol} <span className="muted">— {data.summary.name ?? ""}</span>
      </h1>
      <p className="muted small">{data.summary.sector ?? "Sector unknown"}</p>

      {score ? (
        <div className="metrics">
          <Metric label="Rating" value={<RatingChip rating={score.rating} />} title="Where this ranks against peers — relative, not absolute." />
          <Metric label="Composite" value={formatScore(score.composite_score)} title="The blended 0–100 score this app ranks by." />
          <Metric label="Percentile" value={formatScore(score.percentile_rank, 0)} title="Scored higher than this share of the universe." />
          <Metric label="Coverage" value={confidenceLabel(score.data_confidence)} title="How much underlying data was actually available." />
        </div>
      ) : (
        <p className="muted">This symbol is tracked but has no composite score yet.</p>
      )}

      <section className="panel">
        <h2>Price</h2>
        {data.prices.length > 0 ? (
          <Chart
            ariaLabel={`Candlestick price chart for ${data.symbol}`}
            height={400}
            data={[
              {
                type: "candlestick",
                x: data.prices.map((p) => p.date),
                open: data.prices.map((p) => p.open),
                high: data.prices.map((p) => p.high),
                low: data.prices.map((p) => p.low),
                close: data.prices.map((p) => p.close),
                name: "Price",
                increasing: { line: { color: "#2da44e" } },
                decreasing: { line: { color: "#cf222e" } },
              } as never,
            ]}
            layout={{ xaxis: { rangeslider: { visible: false } } }}
          />
        ) : (
          <p className="muted">No price history stored for this symbol yet.</p>
        )}
      </section>

      <section className="grid-2">
        <div className="panel">
          <h2>Sub-scores</h2>
          {radar.length >= 3 ? (
            <Chart
              ariaLabel={`Radar of ${data.symbol}'s category sub-scores`}
              height={340}
              data={[
                {
                  type: "scatterpolar",
                  r: [...radar.map((e) => e.value), radar[0].value],
                  theta: [...radar.map((e) => e.label), radar[0].label],
                  fill: "toself",
                  line: { color: "#3b82f6" },
                  fillcolor: "rgba(59,130,246,0.18)",
                  name: data.symbol,
                } as never,
              ]}
              layout={{ polar: { radialaxis: { visible: true, range: [0, 100] } } }}
            />
          ) : (
            <p className="muted">Not enough scored categories to plot a radar.</p>
          )}
          <p className="muted small">
            Categories with no data are omitted rather than plotted at zero — a missing
            score is not a bad score.
          </p>
        </div>

        <div className="panel">
          <h2>Detected patterns</h2>
          {data.patterns.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col">Pattern</th>
                  <th scope="col">Direction</th>
                  <th scope="col" className="num">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {data.patterns.map((p, i) => (
                  <tr key={i}>
                    <td>{p.date}</td>
                    <td>{humanize(p.pattern_type)}</td>
                    <td>{humanize(p.direction)}</td>
                    <td className="num">{p.confidence.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No patterns detected recently.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Forecast</h2>
        {models.length > 0 ? (
          <>
            <label>
              Model{" "}
              <select value={selectedModel ?? ""} onChange={(e) => setModel(e.target.value)}>
                {models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </label>
            <table>
              <thead>
                <tr>
                  <th scope="col" className="num">Horizon (days)</th>
                  <th scope="col" className="num">Return</th>
                  <th scope="col" className="num">Target</th>
                  <th scope="col" className="num">Low</th>
                  <th scope="col" className="num">High</th>
                  <th scope="col" className="num">Hit rate</th>
                  <th scope="col" className="num">vs naive</th>
                </tr>
              </thead>
              <tbody>
                {forecasts.map((f) => (
                  <tr key={`${f.model_name}-${f.horizon_days}`}>
                    <td className="num">{f.horizon_days}</td>
                    <td className="num">{formatSignedPercent(f.point_return)}</td>
                    <td className="num">{formatPrice(f.point_price)}</td>
                    <td className="num">{formatPrice(f.lower_price)}</td>
                    <td className="num">{formatPrice(f.upper_price)}</td>
                    <td className="num">{formatPercent(f.historical_hit_rate, 0)}</td>
                    <td className="num">{formatPercent(f.baseline_hit_rate, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              <strong>Hit rate</strong> is this model's own out-of-sample directional
              accuracy at that horizon — shown next to the forecast, not hidden on
              another page. <strong>vs naive</strong> is the same measure for a naive
              random-walk forecast over the same periods. Read them together: a hit
              rate only means something against that baseline, which on real history
              is close to "how often the market simply went up". A model at or below
              the naive column has not demonstrated any skill. Note that{" "}
              <strong>arima</strong> and <strong>baseline</strong> are near-duplicates
              by construction — once ARIMA has a drift term it converges to the
              random-walk-with-drift null — so the two agreeing is not corroboration.
            </p>
          </>
        ) : (
          <p className="muted">No forecasts generated for this symbol yet.</p>
        )}
      </section>

      <section className="grid-2">
        <div className="panel">
          <h2>Algorithm vs Wall Street</h2>
          {data.analyst_consensus ? (
            <div className="metrics">
              <Metric label="Strong Buy" value={data.analyst_consensus.strong_buy} />
              <Metric label="Buy" value={data.analyst_consensus.buy} />
              <Metric label="Hold" value={data.analyst_consensus.hold} />
              <Metric label="Sell" value={data.analyst_consensus.sell} />
              <Metric label="Mean target" value={formatPrice(data.analyst_consensus.mean_price_target)} />
            </div>
          ) : (
            <p className="muted">No analyst coverage stored.</p>
          )}
        </div>

        <div className="panel">
          <h2>What's driving this</h2>
          {data.news.length > 0 ? (
            <ul className="newslist">
              {data.news.map((item, i) => (
                <li key={i}>
                  {item.source_url ? (
                    <a href={item.source_url} target="_blank" rel="noreferrer noopener">
                      {item.title ?? "(untitled)"}
                    </a>
                  ) : (
                    item.title ?? "(untitled)"
                  )}
                  <div className="muted small">
                    {item.event_type ? humanize(item.event_type) : "unclassified"} · sentiment{" "}
                    {formatScore(item.sentiment_score, 2)}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No matching articles recently.</p>
          )}
        </div>
      </section>
    </>
  );
}
