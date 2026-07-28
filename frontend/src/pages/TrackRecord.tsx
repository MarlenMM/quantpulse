import { ErrorBox, Loading, Metric } from "../components/Common";
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

export default function TrackRecord() {
  const { data, error, loading } = useApi(() => api.backtest(20), []);

  if (loading) return <Loading what="the track record" />;
  if (error) return <ErrorBox error={error} />;
  if (!data || data.length === 0) {
    return (
      <>
        <h1>Track Record</h1>
        <p className="muted">
          No backtest has been stored yet. The refresh job runs the survivorship- and
          cost-aware strategy backtest on its weekly cadence.
        </p>
      </>
    );
  }

  const latest = data[0];

  return (
    <>
      <h1>Backtest / Track Record</h1>
      <p className="muted small">
        Most recent run <strong>{latest.run_date}</strong> covering {latest.period_start} →{" "}
        {latest.period_end} · {latest.cadence} rebalancing · {latest.n_periods} periods
      </p>

      <div className="metrics">
        <Metric
          label="Sharpe"
          value={formatScore(latest.sharpe, 2)}
          hint={intervalCaption(latest.sharpe_ci_low, latest.sharpe_ci_high, latest.ci_confidence_level)}
        />
        <Metric
          label="CAGR"
          value={formatPercent(latest.cagr)}
          hint={intervalCaption(latest.cagr_ci_low, latest.cagr_ci_high, latest.ci_confidence_level)}
        />
        <Metric
          label="Max drawdown"
          value={formatPercent(latest.max_drawdown)}
          hint="deliberately not bootstrapped — a path-dependent extremum has no meaningful resampled interval"
        />
        <Metric
          label="Win rate"
          value={formatPercent(latest.win_rate)}
          hint={`average turnover ${formatPercent(latest.avg_turnover)} per rebalance`}
        />
      </div>

      <div className="callout callout-warn">
        <strong>Read this honestly.</strong> These are backtested, hypothetical results on a
        survivorship-aware universe with assumed costs — not realized returns, and not a
        prediction. A confidence interval that straddles zero means the result has not been
        distinguished from luck. Transaction cost assumed:{" "}
        {formatPercent(latest.assumed_txn_cost, 2)} per unit of turnover.
      </div>

      <section className="panel">
        <h2>Versus benchmark</h2>
        <div className="metrics">
          <Metric label="Strategy CAGR" value={formatPercent(latest.cagr)} />
          <Metric label="Benchmark CAGR" value={formatPercent(latest.benchmark_cagr)} />
          <Metric label="Strategy Sharpe" value={formatScore(latest.sharpe, 2)} />
          <Metric label="Benchmark Sharpe" value={formatScore(latest.benchmark_sharpe, 2)} />
        </div>
      </section>

      <section className="panel">
        <h2>Run history</h2>
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
      </section>
    </>
  );
}
