/**
 * Small shared building blocks.
 *
 * `Loading`, `ErrorBox` and `EmptyState` exist so every page distinguishes the
 * three states that look identical if you are careless: still fetching, the
 * backend is unreachable, and the pipeline has not been run. Collapsing them
 * into one blank panel is how a working app gets reported as broken.
 */
import type { ReactNode } from "react";
import { ratingColor, ratingLabel } from "../lib/format";

export function Loading({ what = "data" }: { what?: string }) {
  return <p className="muted" role="status">Loading {what}…</p>;
}

export function ErrorBox({ error }: { error: Error }) {
  return (
    <div className="callout callout-error" role="alert">
      <strong>Couldn't load this.</strong> {error.message}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="callout">{children}</div>;
}

/** A rating chip. Icon + word carry the meaning; color is decoration (Section 12). */
export function RatingChip({ rating }: { rating: string | null }) {
  return (
    <span className="chip" style={{ borderColor: ratingColor(rating) }}>
      {ratingLabel(rating)}
    </span>
  );
}

export function Metric({
  label,
  value,
  hint,
  title,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  title?: string;
}) {
  return (
    <div className="metric" title={title}>
      <div className="metric-label">
        {label}
        {title ? <span className="info" aria-hidden="true"> ⓘ</span> : null}
      </div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

export function Disclaimer() {
  return (
    <footer className="disclaimer">
      Educational/research tool. Not financial advice. Not a registered investment
      advisor. Past backtested performance does not guarantee future results.
    </footer>
  );
}
