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
import type { Data, Layout } from "plotly.js";

const Plot = lazy(() => import("react-plotly.js"));

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
