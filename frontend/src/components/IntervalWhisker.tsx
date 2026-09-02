/**
 * The interval whisker.
 *
 * Every headline figure on the Track Record page is a bootstrap estimate, and
 * the interval around it is the part that decides whether the figure means
 * anything: a Sharpe of 0.8 whose 90% interval runs from −0.4 to 2.0 has not
 * been distinguished from luck, and the number alone cannot say so. The caption
 * beside it says exactly that, in words, and has since the page was written.
 *
 * This says it in one glance. The bar is the interval, the tick inside it is
 * the point estimate, and the hairline is zero. If the hairline falls inside
 * the bar, the result is not distinguishable from luck — and the bar drops to
 * the muted neutral to say so, redundantly with the words underneath, never
 * instead of them.
 *
 * It is drawn with two absolutely-positioned spans over four custom properties.
 * No chart library, no canvas: a 40-pixel-tall figure that has to sit inline
 * next to a number does not need Plotly, and loading Plotly on a page that is
 * otherwise a table of numbers would cost more than the whole page.
 */

/** Where a value sits in the drawn domain, as a CSS percentage. */
function position(value: number, lo: number, hi: number): string {
  const span = hi - lo;
  const fraction = span === 0 ? 0.5 : (value - lo) / span;
  return `${(Math.min(1, Math.max(0, fraction)) * 100).toFixed(2)}%`;
}

export function IntervalWhisker({
  point,
  low,
  high,
  label,
}: {
  point: number | null;
  low: number | null;
  high: number | null;
  /** What the figure is, for the accessible description. */
  label: string;
}) {
  // Nothing honest to draw. The caller's caption already explains why the
  // interval is missing (too short a run to bootstrap), so this stays out of
  // the way rather than drawing a bar with invented ends.
  if (point === null || low === null || high === null) return null;

  // The domain always contains zero, because zero is the whole point of the
  // picture — an interval drawn without it would be a bar with no reference.
  // 12% padding on each side keeps an endpoint from sitting exactly on the
  // edge, where a rounded cap gets clipped and reads as "continues off-screen".
  const rawLo = Math.min(low, 0);
  const rawHi = Math.max(high, 0);
  const pad = (rawHi - rawLo || Math.abs(point) || 1) * 0.12;
  const lo = rawLo - pad;
  const hi = rawHi + pad;

  const straddlesZero = low <= 0 && high >= 0;

  return (
    <div
      className={straddlesZero ? "whisker is-inconclusive" : "whisker"}
      style={
        {
          "--lo": position(low, lo, hi),
          "--hi": position(high, lo, hi),
          "--pos": position(point, lo, hi),
          "--zero": position(0, lo, hi),
        } as React.CSSProperties
      }
      role="img"
      aria-label={
        `${label}: point estimate ${point.toFixed(2)}, interval ${low.toFixed(2)} to ` +
        `${high.toFixed(2)}, which ${straddlesZero ? "includes" : "excludes"} zero`
      }
    >
      <span className="w-bar" />
      <span className="w-point" />
    </div>
  );
}
