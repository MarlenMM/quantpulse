/**
 * The Market Regime Index as a hand-drawn half-dial (finding 29).
 *
 * This was a Plotly `indicator`, and it was the only chart on the Dashboard --
 * so the landing page, the one a recruiter opens on a phone, downloaded the
 * whole charting engine to draw it: **4.1 MB of 4.7 MB** (1.24 MB gzipped),
 * measured on the built site. The figure is a semicircle, three bands, an arc
 * and a number. The interval whisker already showed that shapes this simple
 * need no library; this is the same decision for the same reason.
 *
 * **The bands come from the data, not from here.** Both gauges used to draw the
 * risk-on zone from a literal 65 while `market_regime` labels risk-on from 60,
 * so every score from 60 to 65 read "Risk On" with its arc ending in the
 * neutral band. `risk_on_at` / `risk_off_at` travel with each regime point.
 *
 * Colours are the page's own custom properties, set through `style` (an SVG
 * presentation attribute cannot resolve `var()`), so both themes are right
 * without reading tokens at runtime. One `role="img"` wrapper with the same
 * label the Plotly version had, and nothing focusable inside it.
 */

const CX = 100;
const CY = 100;
const R = 78;

/** A score's point on the dial: 0 at the left end, 100 at the right. */
function at(score: number, radius = R): [number, number] {
  const clamped = Math.min(100, Math.max(0, score));
  const angle = Math.PI * (1 - clamped / 100);
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)];
}

/** The arc from `from` to `to`, clockwise over the top. Never more than a half turn. */
function arc(from: number, to: number): string {
  const [x1, y1] = at(from);
  const [x2, y2] = at(to);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

const BANDS = [
  { key: "risk_off", colour: "var(--down)" },
  { key: "neutral", colour: "var(--muted)" },
  { key: "risk_on", colour: "var(--up)" },
] as const;

export function RegimeGauge({
  score,
  label,
  riskOffAt,
  riskOnAt,
  ariaLabel,
}: {
  score: number;
  /** Already humanized, e.g. "Risk On". */
  label: string;
  riskOffAt: number;
  riskOnAt: number;
  ariaLabel: string;
}) {
  const edges = [0, riskOffAt, riskOnAt, 100];
  const ticks = [0, riskOffAt, riskOnAt, 100];
  return (
    <div className="regime-gauge" role="img" aria-label={ariaLabel}>
      <svg viewBox="-10 0 220 124" aria-hidden="true" focusable="false">
        {BANDS.map((band, i) => (
          <path
            key={band.key}
            d={arc(edges[i], edges[i + 1])}
            data-band={band.key}
            data-from={edges[i]}
            data-to={edges[i + 1]}
            fill="none"
            style={{ stroke: band.colour, strokeOpacity: 0.2, strokeWidth: 20 }}
          />
        ))}
        {score > 0 ? (
          <path
            d={arc(0, score)}
            data-value={score}
            fill="none"
            style={{ stroke: "var(--accent)", strokeWidth: 9, strokeLinecap: "butt" }}
          />
        ) : null}
        {ticks.map((tick) => {
          const [x, y] = at(tick, R + 17);
          return (
            <text
              key={tick}
              x={x}
              y={y + 3}
              textAnchor="middle"
              style={{ fill: "var(--muted)", fontSize: 8, fontFamily: "var(--font-mono)" }}
            >
              {tick.toFixed(0)}
            </text>
          );
        })}
        <text
          x={CX}
          y={CY - 8}
          textAnchor="middle"
          style={{ fill: "var(--text)", fontSize: 30, fontFamily: "var(--font-mono)" }}
        >
          {score.toFixed(0)}
        </text>
        <text
          x={CX}
          y={CY + 16}
          textAnchor="middle"
          style={{ fill: "var(--text-quiet)", fontSize: 12, fontFamily: "var(--font-ui)" }}
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
