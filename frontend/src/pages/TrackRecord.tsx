import { ErrorBox, LoadingMetrics, LoadingTable, Metric } from "../components/Common";
import { IntervalWhisker } from "../components/IntervalWhisker";
import { api } from "../lib/api";
import { formatPercent, formatScore } from "../lib/format";
import type { BacktestRun } from "../lib/types";
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

/** What each stored `signal_name` ranked, in the reader's terms. */
const SIGNAL_LABELS: Record<string, string> = {
  momentum_category: "the momentum category of the composite score",
};

/**
 * Name the signal, and say what it is not.
 *
 * The correction this page most needed. It called itself a "followed the
 * algorithm's ratings" track record while every stored run was ranked by a
 * hand-rolled trailing return, and nothing on screen said otherwise — the
 * project's most load-bearing page quietly testing something other than the
 * thing the project publishes. The wording is kept in step with the Streamlit
 * page deliberately: two front ends describing one limitation differently is
 * how a limitation stops being one.
 */
function WhatWasRanked({ run }: { run: BacktestRun }) {
  const label = run.signal_name === null ? null : SIGNAL_LABELS[run.signal_name];
  if (!label) {
    return (
      <p className="callout callout-warn">
        <strong>This run does not record what it ranked.</strong> It was stored before the
        signal was written down with the result, so treat its numbers as uninterpretable
        rather than as a track record of anything in particular. The next weekly run
        replaces it with one that says.
      </p>
    );
  }
  return (
    <div className="callout">
      <p>
        <strong>What was ranked: {label}</strong> — <code>scoring.score_momentum</code>, the
        same function the nightly composite calls, over survivorship-aware point-in-time
        prices.
      </p>
      <p>
        <strong>This is not the full Buy/Sell rating</strong>, and it cannot honestly be
        yet. Five of the seven categories — fundamental, analyst, sentiment, industry/macro
        and smart money — have only weeks of stored history, so ranking a 2023 rebalance by
        them would mean using 2026 data to pick 2023 stocks. Technical is left out for a
        narrower reason: it reads OHLC, and only the closing price is split-adjusted, so
        over a multi-year window a stock split would read to every indicator as a crash.
      </p>
      <p>
        The rating itself becomes testable once the stored composite history spans the
        window. It holds <strong>{run.composite_history_days} day(s)</strong> so far,
        growing by one per refresh, against the years this backtest covers.
      </p>
    </div>
  );
}

/**
 * The other half of the answer to "you fitted that" (Sections 10, 32).
 *
 * Everything else on this page is a claim about a past that has already
 * happened, ranked by one of the composite's seven categories because the other
 * five hold weeks of stored history rather than years. This section asks the
 * same question forward: a paper account trading the *published* rating, whose
 * every position was chosen before its outcome was known.
 *
 * Written to be honest while it is still short, because it will be short for
 * months. The day count comes before any return, nothing is annualised, and a
 * record that has not started says so rather than drawing a flat line at zero.
 */
function ForwardTestSection() {
  const { data, error, loading } = useApi(() => api.forwardTest(), []);

  if (loading) return <LoadingMetrics what="the forward test" count={4} />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  if (data.n_snapshots === 0) {
    return (
      <p className="callout">
        <strong>Not started.</strong> The forward test paper-trades the published
        Buy/Sell rating on Alpaca&rsquo;s paper endpoint — simulated money, real market
        data, real fills — and records what the account was worth each day. It runs only
        when the two Alpaca credentials are configured; unset, this stays empty rather
        than showing a curve nobody traded. It exists because the backtest above cannot
        rank the full rating until the stored composite history spans its window, and
        forward testing accumulates that history instead of waiting for it.
      </p>
    );
  }

  const start = data.points[0]?.equity ?? null;
  const min = Math.min(...data.points.map((p) => p.equity));
  const max = Math.max(...data.points.map((p) => p.equity));
  const span = max - min || 1;

  return (
    <>
      <div className="metrics">
        <Metric label="Trading days recorded" value={String(data.n_snapshots)} />
        <Metric label="Rebalances" value={String(data.rebalances)} />
        <Metric
          label="Strategy, total return"
          value={data.total_return === null ? "—" : formatPercent(data.total_return)}
        />
        <Metric
          label="S&P 500, same window"
          value={
            data.benchmark_total_return === null
              ? "—"
              : formatPercent(data.benchmark_total_return)
          }
        />
      </div>

      {!data.is_meaningful ? (
        <p className="callout callout-warn">
          <strong>
            {data.n_snapshots} of {data.min_days_for_meaning} trading days.
          </strong>{" "}
          The numbers above are what actually happened, but this is a start rather than a
          track record — treat it as one until the record is longer. Nothing here is
          annualised: a good first week, annualised, is a headline that describes nothing.
        </p>
      ) : null}

      <svg
        viewBox="0 0 600 160"
        role="img"
        aria-label={`Paper account equity from ${data.first_date} to ${data.last_date}`}
        className="equity-curve"
      >
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          points={data.points
            .map((p, i) => {
              const x = (i / Math.max(1, data.points.length - 1)) * 600;
              const y = 150 - ((p.equity - min) / span) * 140;
              return `${x.toFixed(1)},${y.toFixed(1)}`;
            })
            .join(" ")}
        />
      </svg>
      <p className="note">
        Paper account equity, {data.first_date} to {data.last_date}
        {start !== null ? `, starting at ${start.toLocaleString()}` : ""}. Equity is read
        before each run&rsquo;s orders, and orders placed after the close fill at the next
        open — so a rebalance day&rsquo;s point is the book being left, not the one being
        bought.
      </p>
    </>
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

      <WhatWasRanked run={latest} />

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
          The benchmark is the <strong>equal-weight</strong> universe held buy-and-hold, which
          is the comparison that isolates the signal: this strategy equal-weights the names it
          picks, so measuring it against an equal-weight version of the whole universe asks{" "}
          <em>did ranking help</em>, with the weighting scheme held fixed. Against the
          cap-weighted index it would be measuring the ranking and the equal-weight tilt
          together, as one number.
        </p>
      </section>

      {latest.kelly_fraction !== null && latest.excess_payoff_ratio !== null && (
        <section className="block">
          <h2>How much to bet</h2>
          <div className="metrics">
            <Metric
              label="Suggested position"
              value={formatPercent(latest.kelly_fraction)}
              term="Kelly fraction"
            />
            <Metric
              label="Periods beating benchmark"
              value={formatPercent(latest.excess_win_rate)}
            />
            <Metric
              label="Excess payoff ratio"
              value={formatScore(latest.excess_payoff_ratio, 2)}
            />
          </div>
          {latest.kelly_fraction <= 0 ? (
            <p className="callout callout-warn">
              The Kelly criterion says <strong>do not take this bet at all</strong> — over
              this run the strategy had no edge on the benchmark, so tilting away from
              simply holding the benchmark is a losing proposition on average. That is a
              real result, not a missing number.
            </p>
          ) : (
            <p className="note">
              A <strong>quarter-Kelly</strong> size: the growth-optimal tilt away from the
              benchmark, given that this run beat it in{" "}
              {formatPercent(latest.excess_win_rate)} of periods at a{" "}
              {formatScore(latest.excess_payoff_ratio, 2)} excess payoff ratio, then cut to
              a quarter because full Kelly is famously too volatile to live with and is
              exquisitely sensitive to an over-estimated edge. Treat it as an upper bound,
              not a recommendation — it assumes the future resembles this backtest, which
              is exactly the assumption the intervals above tell you to doubt.
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

      <section>
        <h2>Forward test — the same question, asked forward</h2>
        <ForwardTestSection />
      </section>
    </>
  );
}
