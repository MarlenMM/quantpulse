import { ErrorBox, LoadingMetrics, LoadingTable, Metric } from "../components/Common";
import { IntervalWhisker } from "../components/IntervalWhisker";
import { api } from "../lib/api";
import { formatPercent, formatScore } from "../lib/format";
import { useApi } from "../lib/useApi";

/**
 * "90% CI [0.21, 1.34] — excludes zero", or an honest note that the run was
 * too short. Section 7.6 requires the interval next to the headline number, and
 * Section 22 requires saying plainly when it straddles zero: a result that
 * hasn't been distinguished from luck must not read like one that has.
 */
function intervalCaption(
  low: number | null,
  high: number | null,
  level: number | null,
): string {
  if (low === null || high === null) {
    return "no confidence interval — the run was too short to bootstrap honestly";
  }
  const confidence = level !== null ? `${(level * 100).toFixed(0)}%` : "CI";
  const verdict =
    low > 0 || high < 0 ? "excludes zero" : "straddles zero — not distinguishable from luck";
  return `${confidence} CI [${low.toFixed(2)}, ${high.toFixed(2)}] — ${verdict}`;
}

/**
 * A headline figure with its interval drawn underneath and spelled out below
 * that.
 *
 * The drawing and the sentence say the same thing on purpose. The sentence is
 * the one that has to be right — the picture is small, and a reader who cannot
 * see it, or who has the drawing disabled, loses nothing. What the picture buys
 * is that the reader who *is* skimming cannot skim past the interval, which is
 * the failure this whole page is built to prevent.
 */
function EstimateWithInterval({
  label,
  value,
  point,
  low,
  high,
  level,
  term,
}: {
  label: string;
  value: string;
  point: number | null;
  low: number | null;
  high: number | null;
  level: number | null;
  term?: string;
}) {
  return (
    <Metric
      label={label}
      value={value}
      term={term}
      hint={
        <>
          <IntervalWhisker point={point} low={low} high={high} label={label} />
          {intervalCaption(low, high, level)}
        </>
      }
    />
  );
}

export default function TrackRecord() {
  const { data, error, loading } = useApi(() => api.backtest(20), []);

  if (loading) {
    return (
      <>
        <h1>Backtest / Track Record</h1>
        <LoadingMetrics what="the track record" count={4} />
        <LoadingTable what="the run history" rows={5} columns={[16, 16, 16, 12, 12, 12, 12]} />
      </>
    );
  }
  if (error) return <ErrorBox error={error} />;
  if (!data || data.length === 0) {
    return (
      <>
        <h1>Backtest / Track Record</h1>
        <p className="standfirst">
          No backtest has been stored yet. The refresh job runs the survivorship- and
          cost-aware strategy backtest on its weekly cadence, and this page fills in from
          the run it stores.
        </p>
      </>
    );
  }

  const latest = data[0];

  return (
    <>
      <h1>Backtest / Track Record</h1>
      <p className="standfirst">
        What the ranking would have returned had it been traded, on a survivorship-aware
        universe with transaction costs assumed. Every figure carries the bootstrap interval
        around it, because a Sharpe with an interval spanning zero has not been distinguished
        from luck — and that is a different claim from the number alone.
      </p>
      <p className="muted small">
        Most recent run <strong>{latest.run_date}</strong>, covering {latest.period_start} →{" "}
        {latest.period_end} · {latest.cadence} rebalancing · {latest.n_periods} periods
      </p>

      {/* The subject of the page: four estimates, each with its interval drawn
          against zero. This is the one thing here worth looking at first. */}
      <section className="card lede">
        <h2 className="h-lede">The estimate, and how sure it is</h2>
        <div className="metrics">
          <EstimateWithInterval
            label="Sharpe"
            value={formatScore(latest.sharpe, 2)}
            point={latest.sharpe}
            low={latest.sharpe_ci_low}
            high={latest.sharpe_ci_high}
            level={latest.ci_confidence_level}
            term="Sharpe ratio"
          />
          <EstimateWithInterval
            label="CAGR"
            value={formatPercent(latest.cagr)}
            point={latest.cagr}
            low={latest.cagr_ci_low}
            high={latest.cagr_ci_high}
            level={latest.ci_confidence_level}
            term="CAGR"
          />
          <Metric
            label="Max drawdown"
            value={formatPercent(latest.max_drawdown)}
            term="Max drawdown"
            hint="deliberately not bootstrapped — a path-dependent extremum has no meaningful resampled interval"
          />
          <Metric
            label="Win rate"
            value={formatPercent(latest.win_rate)}
            hint={`average turnover ${formatPercent(latest.avg_turnover)} per rebalance`}
          />
        </div>
        <p className="note">
          The bar under each figure is that bootstrap interval, and the hairline crossing it is
          zero. A bar that overlaps the hairline is a result the data has not separated from
          luck; it is drawn grey rather than in the accent to say so.
        </p>
      </section>

      <div className="callout callout-warn">
        <strong>Read this honestly.</strong> These are backtested, hypothetical results on a
        survivorship-aware universe with assumed costs — not realized returns, and not a
        prediction. Nothing here was traded. Transaction cost assumed:{" "}
        {formatPercent(latest.assumed_txn_cost, 2)} per unit of turnover.
      </div>

      <section className="block">
        <h2>Versus benchmark</h2>
        <div className="metrics">
          <Metric label="Strategy CAGR" value={formatPercent(latest.cagr)} />
          <Metric label="Benchmark CAGR" value={formatPercent(latest.benchmark_cagr)} />
          <Metric label="Strategy Sharpe" value={formatScore(latest.sharpe, 2)} />
          <Metric label="Benchmark Sharpe" value={formatScore(latest.benchmark_sharpe, 2)} />
        </div>
        <p className="note">
          The benchmark is an equal-weight proxy for the market, because no S&amp;P 500 price
          series is ingested anywhere in this project — the same honest stand-in the beta
          calculation uses.
        </p>
      </section>

      {latest.kelly_fraction !== null && latest.payoff_ratio !== null && (
        <section className="block">
          <h2>How much to bet</h2>
          <div className="metrics">
            <Metric
              label="Suggested position"
              value={formatPercent(latest.kelly_fraction)}
              term="Kelly fraction"
            />
            <Metric label="Win rate used" value={formatPercent(latest.win_rate)} />
            <Metric label="Payoff ratio used" value={formatScore(latest.payoff_ratio, 2)} />
          </div>
          {latest.kelly_fraction <= 0 ? (
            <p className="callout callout-warn">
              The Kelly criterion says <strong>do not take this bet at all</strong> — at this
              win rate and payoff ratio the strategy has no positive edge to size, so any
              position is a losing proposition on average.
            </p>
          ) : (
            <p className="note">
              A <strong>quarter-Kelly</strong> size: the growth-optimal bet given this run's
              own {formatPercent(latest.win_rate)} win rate and{" "}
              {formatScore(latest.payoff_ratio, 2)} payoff ratio, then cut to a quarter
              because full Kelly is famously too volatile to live with and is exquisitely
              sensitive to an over-estimated edge. Treat it as an upper bound, not a
              recommendation — it assumes the future resembles this backtest, which is
              exactly the assumption the intervals above tell you to doubt.
            </p>
          )}
        </section>
      )}

      <section className="block">
        <h2>Run history</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">From</th>
                <th scope="col">To</th>
                <th scope="col" className="num">Periods</th>
                <th scope="col" className="num">Sharpe</th>
                <th scope="col" className="num">CAGR</th>
                <th scope="col" className="num">Max DD</th>
              </tr>
            </thead>
            <tbody>
              {data.map((run, i) => (
                <tr key={i}>
                  <td>{run.run_date}</td>
                  <td>{run.period_start ?? "—"}</td>
                  <td>{run.period_end ?? "—"}</td>
                  <td className="num">{run.n_periods}</td>
                  <td className="num">{formatScore(run.sharpe, 2)}</td>
                  <td className="num">{formatPercent(run.cagr)}</td>
                  <td className="num">{formatPercent(run.max_drawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
