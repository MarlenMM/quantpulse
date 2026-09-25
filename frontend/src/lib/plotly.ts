/**
 * The Plotly the app ships: the core plus the three trace types it draws.
 *
 * `react-plotly.js`'s default entry imports `plotly.js/dist/plotly` -- every
 * trace type Plotly has, maps, 3D and all -- which was 4.1 MB raw (1.24 MB
 * gzipped) on every page with a chart. Stock Detail, the only such page since
 * the regime dial became SVG (finding 29), draws a candlestick, three scatter
 * figures and a radar. So the component is built from the factory around a
 * Plotly that registers exactly those.
 *
 * **Adding a chart of a new type means registering it here.** An unregistered
 * type does not fail the build or the type check; Plotly draws an empty figure.
 * `tests/charts.spec.ts` renders every Stock Detail figure and asserts each has
 * its traces, and `CHART_TRACE_TYPES` is what it checks the page against.
 */
// Must stay the first import: Plotly's source reads Node's `global`.
import "./plotly-global";
import Plotly from "plotly.js/lib/core";
import candlestick from "plotly.js/lib/candlestick";
import scatterpolar from "plotly.js/lib/scatterpolar";

/** The trace types registered below (`scatter` ships with the core). */
export const CHART_TRACE_TYPES = ["scatter", "candlestick", "scatterpolar"] as const;

Plotly.register([candlestick, scatterpolar]);

export default Plotly;
