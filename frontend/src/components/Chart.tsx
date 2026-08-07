/**
 * Plotly wrapper (Section 12: "would carry over cleanly to a React migration
 * via `react-plotly.js`" — this is that claim being cashed in).
 *
 * **Plotly is loaded lazily, and that matters here.** The library is ~5 MB
 * unminified and dominates the bundle, but only two of the five pages draw a
 * chart — the Screener, Track Record and Glossary are tables and prose. A
 * static import would make every visitor download a charting engine to read a
 * table, which on the free hosting tier this project targets is the difference
 * between a fast first paint and a slow one. `lazy` + `Suspense` splits it into
 * its own chunk that is fetched only when a chart is actually rendered.
 *
 * Transparent backgrounds so the figure inherits the page theme, matching the
 * Streamlit charts rather than hardcoding a palette that would drift from the
 * other front end.
 */
import { Suspense, lazy } from "react";
import type { ComponentType } from "react";
import type { Data, Layout } from "plotly.js";
import type { PlotParams } from "react-plotly.js";

/**
 * Resolving `react-plotly.js`'s component has broken twice, in two different
 * ways, so this unwraps by *asking what a React element type looks like* rather
 * than by pattern-matching one library version's packaging.
 *
 * A valid element type is a function (function or class component) **or** an
 * object carrying `$$typeof` — which is what `forwardRef` and `memo` produce.
 * Testing only for a function is what broke the second time.
 *
 * The two failures, both runtime-only — `tsc --noEmit` and `vite build` pass
 * either way, and the page dies with "Element type is invalid. Received a
 * promise that resolves to: [object Object]":
 *
 * 1. **Vite 7 → 8.** v7 (Rollup + esbuild) resolved the CJS namespace's
 *    `.default` to the component. v8 bundles with Rolldown, whose interop
 *    yields `{ default: { default: Component } }`, so React got an object.
 * 2. **react-plotly.js 2 → 4.** v4 is dual ESM/CJS and ships the component as a
 *    **`forwardRef` object** (`{ $$typeof, render }`), not a plain function — so
 *    a `typeof === "function"` test fell through to a non-existent `.default`
 *    and handed React `undefined`.
 *
 * With the versions currently pinned a bare `import()` would in fact resolve
 * correctly, because v4's ESM `exports` hand back the forwardRef object
 * directly — failure (1) needed v2's CJS-only packaging and is no longer
 * reachable. This is kept anyway: it costs nothing, it survives either
 * packaging, and it fails with a named error instead of a blank page. Two
 * breakages in two days is enough evidence that this boundary moves.
 *
 * `tests/charts.spec.ts` is what guards this now. Only failure (2) is still
 * reproducible there, for exactly the reason above.
 */
function isElementType(value: unknown): boolean {
  if (typeof value === "function") return true;
  return typeof value === "object" && value !== null && "$$typeof" in value;
}

const Plot = lazy(async () => {
  const mod: unknown = await import("react-plotly.js");
  const first = (mod as { default: unknown }).default;
  const component = isElementType(first) ? first : (first as { default: unknown })?.default;
  if (!isElementType(component)) {
    throw new Error(
      "react-plotly.js did not resolve to a React component — its packaging changed again",
    );
  }
  return { default: component as ComponentType<PlotParams> };
});

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c9d1d9" },
  margin: { l: 48, r: 16, t: 24, b: 40 },
  legend: { orientation: "h", y: 1.08, x: 0 },
  xaxis: { gridcolor: "rgba(255,255,255,0.08)" },
  yaxis: { gridcolor: "rgba(255,255,255,0.08)" },
};

export function Chart({
  data,
  layout,
  height = 340,
  ariaLabel,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  ariaLabel: string;
}) {
  return (
    // Plotly renders to canvas/SVG with no inherent description, so without
    // this the whole figure is invisible to a screen reader.
    <div role="img" aria-label={ariaLabel} style={{ minHeight: height }}>
      <Suspense fallback={<p className="muted" role="status">Loading chart…</p>}>
        <Plot
          data={data}
          layout={{ ...BASE_LAYOUT, ...layout, height, autosize: true }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </Suspense>
    </div>
  );
}
