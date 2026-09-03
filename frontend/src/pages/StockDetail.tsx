import { useState } from "react";
import { Chart } from "../components/Chart";
import {
  ErrorBox,
  LoadingChart,
  LoadingMetrics,
  Metric,
  RatingChip,
} from "../components/Common";
import { Tip } from "../components/Tip";
import { api } from "../lib/api";
import { Link } from "../lib/router";
import { CATEGORIES, SUBSCORE_KEYS } from "../lib/types";
import {
  confidenceLabel,
  formatPercent,
  formatPrice,
  formatScore,
  formatSignedPercent,
  humanize,
} from "../lib/format";
import { useThemeTokens } from "../lib/theme";
import { useApi } from "../lib/useApi";

export default function StockDetail({ symbol }: { symbol: string }) {
  const { data, error, loading } = useApi(() => api.stock(symbol), [symbol]);
  const [model, setModel] = useState<string | null>(null);
  const theme = useThemeTokens();

  // The whole page comes from one request, so a bare "Loading AIZ…" left the
  // viewport empty until it landed — on the published demo, long enough to read
  // as a dead link. The skeleton holds the page's actual shape instead.
  if (loading) {
    return (
      <>
        {/* A div, not an empty <h1>: a heading with no text is a real
            accessibility defect, and the announcement is carried by the
            visually-hidden status line inside the metrics below. */}
        <div className="skeleton" style={{ width: "18rem", height: "2.4rem" }} />
        <LoadingMetrics what={symbol} count={4} />
        <LoadingChart what={`${symbol}'s price history`} />
      </>
    );
  }
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
        <span className="ticker">{data.symbol}</span>
        {data.summary.name ? ` — ${data.summary.name}` : ""}
      </h1>
      <p className="standfirst">
        {data.summary.sector ?? "Sector unknown"} · every figure below is computed from stored
        public data, and each one links to its own definition.
      </p>

      {score ? (
        <div className="metrics">
          <Metric
            label="Rating"
            value={<RatingChip rating={score.rating} />}
            term="Rating"
            text
          />
          <Metric
            label="Composite"
            value={formatScore(score.composite_score)}
            term="Composite score"
          />
          <Metric
            label="Percentile"
            value={formatScore(score.percentile_rank, 0)}
            term="Percentile rank"
          />
          <Metric
            label="Coverage"
            value={confidenceLabel(score.data_confidence)}
            term="Data coverage"
            text
          />
        </div>
      ) : (
        <p className="muted">
          This symbol is in the universe but has never been scored. That happens to a
          recent index addition: the composite needs enough price history to compute the
          technical and momentum categories before it will rank a name at all.
        </p>
      )}

      {/* The price history is the one thing a reader opening a stock page came
          for, so it is the only card on the page. Everything after it is a
          plain section separated by a rule — nine identical rounded panels in a
          column is a stack with no first item. */}
      <section className="card lede">
        <h2 className="h-lede">Price</h2>
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
                increasing: { line: { color: theme.up } },
                decreasing: { line: { color: theme.down } },
              } as never,
            ]}
            layout={{ xaxis: { rangeslider: { visible: false } } }}
          />
        ) : (
          <p className="muted">
            No price history is stored for this symbol. Prices come in on the daily
            refresh; a name added to the index today has none until it next runs.
          </p>
        )}
      </section>

      <section className="split-even block">
        <div>
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
                  line: { color: theme.accent },
                  fillcolor: theme.accentBand,
                  name: data.symbol,
                } as never,
              ]}
              layout={{ polar: { radialaxis: { visible: true, range: [0, 100] } } }}
            />
          ) : (
            <p className="muted">
              Fewer than three categories have data, and a radar over two axes is a line.
              The Coverage figure above is the same fact as a number.
            </p>
          )}
          <p className="muted small">
            Categories with no data are omitted rather than plotted at zero — a missing
            score is not a bad score.
          </p>
        </div>

        <div>
          <h2>Detected patterns</h2>
          {data.patterns.length > 0 ? (
            <div className="tablewrap">
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
            </div>
          ) : (
            <p className="muted">
              The detector found none of the four pattern families in this stock&apos;s recent
              price action. That is the ordinary case, and it is not a bearish signal — the
              absence of a pattern says nothing about direction.
            </p>
          )}
        </div>
      </section>

      <section className="block">
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
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col" className="num">Horizon (days)</th>
                    <th scope="col" className="num">Return</th>
                    <th scope="col" className="num">Target</th>
                    <th scope="col" className="num">Low</th>
                    <th scope="col" className="num">High</th>
                    <th scope="col" className="num">
                      Hit rate
                      <Tip term="Hit rate" />
                    </th>
                    <th scope="col" className="num">
                      vs naive
                      <Tip
                        label="the naive comparison"
                        text="The same hit rate for the naive baseline — 'tomorrow looks like today'. A model only knows something the market does not if it beats this column, and mostly none of them do."
                      />
                    </th>
                    <th scope="col" className="num">
                      Windows
                      <Tip
                        label="windows"
                        text="How many distinct out-of-sample periods the two hit rates were measured over. A rate from a handful of windows is an anecdote, not a track record; below the minimum it is suppressed entirely and shows a dash."
                      />
                    </th>
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
                      <td className="num">{f.hit_rate_windows ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small">
              <strong>Hit rate</strong> is this model's own out-of-sample directional
              accuracy at that horizon — shown next to the forecast, not hidden on
              another page. <strong>vs naive</strong> is the same measure for a naive
              random-walk forecast over the same periods. Read them together: a hit
              rate only means something against that baseline, which on real history
              is close to "how often the market simply went up". A model at or below
              the naive column has not demonstrated any skill.{" "}
              <strong>Windows</strong> is how many separate historical periods those
              rates were measured over; a dash means too few for a rate to mean
              anything, so none is shown rather than a flattering one. Note that{" "}
              <strong>arima</strong> and <strong>baseline</strong> are near-duplicates
              by construction — once ARIMA has a drift term it converges to the
              random-walk-with-drift null — so the two agreeing is not corroboration.
            </p>
          </>
        ) : (
          <p className="muted">
            No forecasts are stored for this symbol. All three models need a couple of
            years of history before they will produce one.
          </p>
        )}
      </section>

      {data.risk ? (
        <section className="block">
          <h2>Risk profile</h2>
          <div className="metrics">
            <Metric
              label="Volatility (ann.)"
              value={formatPercent(data.risk.historical_volatility)}
              term="Volatility"
            />
            <Metric
              label="Implied vol"
              value={formatPercent(data.risk.implied_volatility)}
              term="Implied volatility"
            />
            <Metric
              label="Beta"
              value={formatScore(data.risk.beta, 2)}
              term="Beta"
            />
            <Metric
              label="Sharpe"
              value={formatScore(data.risk.sharpe, 2)}
              term="Sharpe ratio"
            />
            <Metric
              label="Sortino"
              value={formatScore(data.risk.sortino, 2)}
              term="Sortino ratio"
            />
            <Metric
              label={`Daily VaR ${data.risk.var_confidence ? `${(data.risk.var_confidence * 100).toFixed(0)}%` : ""}`}
              value={formatPercent(data.risk.value_at_risk)}
              term="Value at Risk"
            />
          </div>
          <p className="muted small">
            Measured on {data.risk.n_observations} daily returns.{" "}
            {data.risk.sortino === null &&
            data.risk.n_observations < data.risk.ratio_min_observations
              ? `Sharpe and Sortino need about a year of history (${data.risk.ratio_min_observations} daily
                 returns) before they mean anything — a ratio of average return to risk is far
                 noisier than either number alone, so they are left blank rather than shown as
                 noise. `
              : ""}
            {data.risk.beta !== null &&
            data.risk.beta_r_squared !== null &&
            data.risk.beta_benchmark !== null
              ? `Beta is measured against ${data.risk.beta_benchmark}, R² = ${data.risk.beta_r_squared.toFixed(2)}.`
              : ""}
          </p>
        </section>
      ) : null}

      {data.short_interest ? (
        <section className="block">
          <h2>Short interest</h2>
          <div className="metrics">
            <Metric
              label="% of float short"
              value={
                data.short_interest.pct_float_short !== null
                  ? formatPercent(data.short_interest.pct_float_short / 100)
                  : "—"
              }
            />
            <Metric
              label="Days to cover"
              value={formatScore(data.short_interest.days_to_cover, 2)}
            />
          </div>
          {data.short_interest.elevated ? (
            <p className="callout callout-warn small">
              <strong>Elevated short interest — and that cuts both ways.</strong> It can mean
              informed investors are betting against this company. It can equally set up a{" "}
              <strong>short squeeze</strong>: a crowded short position that has to buy back
              quickly if the story improves, which pushes the price <em>up</em>. QuantPulse does
              not score this as bullish or bearish, because the same number genuinely supports
              both readings.
            </p>
          ) : (
            <p className="muted small">
              Short interest is not elevated. Shown as context only — it is deliberately excluded
              from the Smart Money score, since the same figure can be read as bearish conviction
              or as squeeze potential.
            </p>
          )}
        </section>
      ) : null}

      {data.monte_carlo ? (
        <section className="block">
          <h2>Simulated price paths</h2>
          <p className="muted small">
            {data.monte_carlo.n_paths.toLocaleString()} random-walk paths over the next{" "}
            {data.monte_carlo.horizon_days} trading days, calibrated to this stock's own drift and
            volatility. The band is the middle 90% of simulated outcomes; it widens with time
            because uncertainty compounds — that widening is the message. This is a range of
            possibilities, not a prediction.
          </p>
          <Chart
            ariaLabel={`Monte Carlo simulated price fan for ${data.symbol}`}
            height={320}
            data={[
              {
                x: data.monte_carlo.bands.map((b) => b.day),
                y: data.monte_carlo.bands.map((b) => b.upper),
                type: "scatter",
                mode: "lines",
                line: { width: 0 },
                name: "95th percentile",
                hoverinfo: "skip",
              },
              {
                x: data.monte_carlo.bands.map((b) => b.day),
                y: data.monte_carlo.bands.map((b) => b.lower),
                type: "scatter",
                mode: "lines",
                fill: "tonexty",
                fillcolor: theme.accentBand,
                line: { width: 0 },
                name: "5th percentile",
                hoverinfo: "skip",
              },
              {
                x: data.monte_carlo.bands.map((b) => b.day),
                y: data.monte_carlo.bands.map((b) => b.median),
                type: "scatter",
                mode: "lines",
                line: { width: 2, color: theme.accent },
                name: "Median path",
              },
            ]}
            layout={{
              xaxis: { title: { text: "Trading days ahead" } },
              yaxis: { title: { text: "Simulated price" } },
            }}
          />
          <p className="muted small">
            Calibrated on {data.monte_carlo.n_train.toLocaleString()} daily returns (drift{" "}
            {(data.monte_carlo.mu * 100).toFixed(3)}%/day, volatility{" "}
            {(data.monte_carlo.sigma * 100).toFixed(2)}%/day).
          </p>
        </section>
      ) : null}

      {data.macro_overlay ? (
        <section className="block">
          <h2>Macro overlay</h2>
          <p>
            <strong>{data.macro_overlay.sector}</strong> is exposed to{" "}
            {data.macro_overlay.components
              .filter((c) => c.move !== null)
              .map((c) => `${humanize(c.driver)} (${formatSignedPercent((c.move ?? 0) / 100)})`)
              .join(", ")}{" "}
            over the last ~3 months — a{" "}
            <strong>
              {data.macro_overlay.adjustment > 0
                ? "tailwind"
                : data.macro_overlay.adjustment < 0
                  ? "headwind"
                  : "neutral"}
            </strong>{" "}
            of {data.macro_overlay.adjustment.toFixed(2)} on a −1 to +1 scale.
          </p>
          <p className="muted small">
            Applied only to the sectors these series genuinely move: oil for Energy, metals for
            Materials, the dollar for sectors dominated by multinationals earning abroad. Every
            other sector gets exactly zero rather than a small meaningless nudge. This is context,
            not part of the composite score.
          </p>
        </section>
      ) : null}

      <section className="split-even block">
        <div>
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
            <p className="muted small">
              No analyst estimates stored for this company. They are gathered on the weekly
              refresh — the freshness line on the <Link to="/dashboard">Dashboard</Link> says
              when that last completed, and reads "never run" if it has not.
            </p>
          )}
        </div>

        <div>
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
            // Say which of the two it is. "No articles" reads as a broken
            // feed, and a reader who cannot tell an un-run pipeline from a
            // failed one has no way to judge anything else on the page either.
            <p className="muted small">
              No articles from the last three weeks mention this company. Company news is
              gathered on the weekly refresh — the freshness line on the{" "}
              <Link to="/dashboard">Dashboard</Link> says when that last completed, and
              reads "never run" if it has not.
            </p>
          )}
        </div>
      </section>
    </>
  );
}
