/**
 * The ⓘ hint, as an actual tooltip.
 *
 * It used to be a `title` attribute. That is a real tooltip in the sense that
 * the browser eventually shows something, but in practice it is unusable: it
 * takes about a second of holding still to appear, it is unstyled OS chrome,
 * it never appears on a touch device, and it does not appear at all for a
 * keyboard user. The reported symptom was simply "there is no popup" — and
 * looking things up in the Glossary page instead is exactly the friction the
 * hint exists to remove.
 *
 * Two decisions worth keeping:
 *
 * **The text comes from the glossary**, not from a string written here. There
 * is one definition per term (`quantpulse.glossary.TERMS`), the Glossary page
 * shows it, and now so does the tooltip. Hand-written hint strings had already
 * drifted into paraphrases of the real entries.
 *
 * **The bubble is positioned in fixed coordinates**, computed on open, rather
 * than absolutely inside its parent. Most of these markers live inside table
 * cells and panels that scroll or clip their overflow, so a CSS-only bubble
 * gets cut off exactly where the table is widest — which is where the columns
 * most need explaining. Fixed positioning also lets it flip above the trigger
 * near the bottom of the window and clamp to the viewport near either edge.
 */

import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useDefinition } from "../lib/glossary";

/** Distance from the trigger, and the gap kept from the window edge. */
const OFFSET = 10;
const MARGIN = 8;
const WIDTH = 300;

type Placement = { left: number; top: number; above: boolean };

export function Tip({
  term,
  text,
  label,
}: {
  /** Glossary term to explain. Matched case-insensitively. */
  term?: string;
  /** Used when there is no glossary entry — e.g. something specific to one screen. */
  text?: string;
  /** Overrides the accessible name; defaults to the term. */
  label?: string;
}) {
  const definition = useDefinition(term);
  const body = definition ?? text ?? null;
  const [placement, setPlacement] = useState<Placement | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const id = useId();

  const open = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    // Flip above when there is not enough room below. 140px is a generous
    // estimate of the tallest definition; being wrong only picks the less good
    // side, never a clipped one, because the clamp below still applies.
    const above = rect.bottom + 140 > window.innerHeight && rect.top > 140;
    const left = Math.min(
      Math.max(MARGIN, rect.left + rect.width / 2 - WIDTH / 2),
      Math.max(MARGIN, window.innerWidth - WIDTH - MARGIN),
    );
    setPlacement({
      left,
      top: above ? rect.top - OFFSET : rect.bottom + OFFSET,
      above,
    });
  }, []);

  const close = useCallback(() => setPlacement(null), []);

  // Scrolling or resizing while it is open would leave the bubble behind,
  // because its coordinates were computed once against the old layout.
  useLayoutEffect(() => {
    if (!placement) return;
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [placement, close]);

  if (!body) return null;

  return (
    <span className="tip">
      <button
        ref={triggerRef}
        type="button"
        className="tip-trigger"
        aria-label={`What is ${label ?? term ?? "this"}?`}
        aria-describedby={placement ? id : undefined}
        aria-expanded={placement ? true : false}
        onMouseEnter={open}
        onMouseLeave={close}
        onFocus={open}
        onBlur={close}
        // Tap to open on touch, where there is no hover at all.
        onClick={(event) => {
          event.preventDefault();
          placement ? close() : open();
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") close();
        }}
      >
        ⓘ
      </button>
      {placement ? (
        <span
          id={id}
          role="tooltip"
          className={`tip-bubble ${placement.above ? "tip-above" : ""}`}
          style={{ left: placement.left, top: placement.top, width: WIDTH }}
        >
          {term ? <strong className="tip-term">{term}</strong> : null}
          <span className="tip-body">{body as ReactNode}</span>
        </span>
      ) : null}
    </span>
  );
}
