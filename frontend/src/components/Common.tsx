/**
 * Small shared building blocks.
 *
 * `Loading`, `ErrorBox` and `EmptyState` exist so every page distinguishes the
 * three states that look identical if you are careless: still fetching, the
 * backend is unreachable, and the pipeline has not been run. Collapsing them
 * into one blank panel is how a working app gets reported as broken.
 *
 * The loading state used to be the single line "Loading data…" on an otherwise
 * empty page. For the second or two it is on screen — longer on the published
 * demo, which fetches a JSON file per section — that is indistinguishable from
 * a broken deploy, and it is the state a first-time visitor sees first. The
 * skeletons below occupy the *shape* of what is arriving instead: a table
 * loading looks like a table, a row of figures looks like a row of figures, and
 * nothing shifts when the data lands. Sighted readers get the shape; screen
 * readers still get the sentence, from the visually-hidden `role="status"`.
 */
import type { ReactNode } from "react";
import { ratingGlyph, ratingText } from "../lib/format";
import { Tip } from "./Tip";

/** The announcement half of every loading state. */
function Announce({ what }: { what: string }) {
  return (
    <span className="sr-only" role="status">
      Loading {what}…
    </span>
  );
}

/**
 * A generic placeholder, for a region whose shape is not worth mimicking
 * exactly. Still a block rather than a line, so the page does not collapse and
 * then jump.
 */
export function Loading({ what = "data" }: { what?: string }) {
  return (
    <div aria-busy="true">
      <Announce what={what} />
      <div className="skeleton sk-line" style={{ width: "42%" }} />
      <div className="skeleton sk-line" style={{ width: "78%" }} />
      <div className="skeleton sk-line" style={{ width: "61%" }} />
    </div>
  );
}

/**
 * A table's shape: a header rule and `rows` bars, with the columns' widths
 * roughly where the real ones will be so the eye does not have to re-find them.
 */
export function LoadingTable({
  what = "data",
  rows = 6,
  columns = [22, 38, 14, 12, 14],
}: {
  what?: string;
  rows?: number;
  columns?: number[];
}) {
  return (
    <div aria-busy="true">
      <Announce what={what} />
      <div className="sk-row">
        {columns.map((width, i) => (
          <div key={i} className="skeleton sk-head" style={{ width: `${width}%` }} />
        ))}
      </div>
      {Array.from({ length: rows }, (_, row) => (
        <div className="sk-row" key={row}>
          {columns.map((width, i) => (
            <div key={i} className="skeleton" style={{ width: `${width}%` }} />
          ))}
        </div>
      ))}
    </div>
  );
}

/** A row of headline figures, in the same grid the real ones will land in. */
export function LoadingMetrics({ what = "data", count = 4 }: { what?: string; count?: number }) {
  return (
    <div className="metrics" aria-busy="true">
      <Announce what={what} />
      {Array.from({ length: count }, (_, i) => (
        <div key={i}>
          <div className="skeleton sk-line" style={{ width: "4.5rem", height: "0.6rem" }} />
          <div className="skeleton sk-figure" />
        </div>
      ))}
    </div>
  );
}

/** A figure-shaped block, for a region that will be filled by a chart. */
export function LoadingChart({ what = "the chart" }: { what?: string }) {
  return (
    <div aria-busy="true">
      <Announce what={what} />
      <div className="skeleton sk-chart" />
    </div>
  );
}

export function ErrorBox({ error }: { error: Error }) {
  return (
    <div className="callout callout-error" role="alert">
      <strong>This section didn't load.</strong> {error.message} Everything else on the page
      is unaffected — the sections fetch independently.
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="callout callout-note">{children}</div>;
}

/**
 * A rating chip.
 *
 * Glyph + word carry the meaning; the colour is a third, redundant channel
 * (Section 12). The colour itself comes from `data-rating` and the stylesheet
 * rather than an inline hex, so a rating is the same green in a chip, a table
 * cell and a chart, and re-tints correctly in dark mode — an inline literal
 * could only ever be right in one theme.
 */
export function RatingChip({ rating }: { rating: string | null }) {
  return (
    <span className="chip" data-rating={rating ?? undefined}>
      <span className="chip-glyph" aria-hidden="true">
        {ratingGlyph(rating)}
      </span>
      {ratingText(rating)}
    </span>
  );
}

export function Metric({
  label,
  value,
  hint,
  term,
  title,
  /** Set for a value that is a word or a chip rather than a number: it keeps the
      UI face instead of being set in the tabular mono meant for figures. */
  text = false,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** Glossary term explaining this metric. Preferred over `title`. */
  term?: string;
  /** A one-off explanation for something the glossary has no entry for. */
  title?: string;
  text?: boolean;
}) {
  return (
    <div className="metric">
      <div className="metric-label">
        {label}
        <Tip term={term} text={title} label={label} />
      </div>
      <div className={text ? "metric-value is-text" : "metric-value"}>{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

export function Disclaimer() {
  return (
    <footer className="disclaimer">
      <strong>Educational and research tool.</strong> Not financial advice, and not a
      registered investment advisor. Every return figure on this site is backtested and
      hypothetical; none of it was traded. Past backtested performance does not guarantee
      future results.
    </footer>
  );
}
