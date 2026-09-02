/**
 * The QuantPulse mark.
 *
 * It replaces a 📈 that used to sit in the masthead. An emoji in a wordmark is
 * a placeholder that shipped: it renders as a different picture on every
 * platform, it cannot take the page's colours, and it is the single fastest way
 * to tell a reader that nobody drew anything.
 *
 * This is drawn instead — three OHLC candles, hand-plotted on a 26×24 grid, one
 * for each pole of the app's own rating vocabulary: a low candle in the Sell
 * red, a doji in the Hold amber, a high one in the Buy green, stepping upward.
 * The logo is therefore made out of the same five colours the ratings are, and
 * it re-tints itself in dark mode along with everything else because the fills
 * are the CSS custom properties rather than literals.
 *
 * The coordinates below are typed by hand, not generated from a series, and are
 * chosen for how they read at 18px in a masthead rather than for what they
 * would mean as data — which is why the wicks are asymmetric and the doji's
 * body is offset a pixel low. A generated glyph would be tidier and would look
 * like every other generated glyph.
 */

/** x-centre, [wick top, wick bottom], [body top, body bottom], colour token. */
const CANDLES: ReadonlyArray<{
  x: number;
  wick: [number, number];
  body: [number, number];
  token: string;
}> = [
  { x: 4, wick: [5, 21], body: [9, 18], token: "--down" },
  { x: 13, wick: [3, 19], body: [8, 15], token: "--flat" },
  { x: 22, wick: [1, 16], body: [4, 12], token: "--up" },
];

const BODY_WIDTH = 5;

export function Mark({ className = "brand-mark" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 26 24"
      // Decorative: the wordmark beside it already says "QuantPulse", and a
      // second announcement of the same name is noise in a screen reader.
      aria-hidden="true"
      focusable="false"
    >
      {CANDLES.map(({ x, wick, body, token }) => (
        <g key={x} fill={`var(${token})`} stroke={`var(${token})`}>
          <line
            x1={x}
            x2={x}
            y1={wick[0]}
            y2={wick[1]}
            strokeWidth={1.5}
            strokeLinecap="round"
          />
          <rect
            x={x - BODY_WIDTH / 2}
            y={body[0]}
            width={BODY_WIDTH}
            height={body[1] - body[0]}
            rx={1}
            stroke="none"
          />
        </g>
      ))}
    </svg>
  );
}
