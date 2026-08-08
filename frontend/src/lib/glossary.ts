/**
 * One definition, two surfaces — the same rule the Streamlit app follows.
 *
 * The Glossary page and every ⓘ tooltip read from this one fetch of
 * `/api/glossary`, which is served from `quantpulse.glossary.TERMS`. Before
 * this, the tooltips were hand-written strings at each call site that
 * paraphrased the glossary, so the tooltip and the glossary entry for the same
 * term could say different things — and after a definition changed, they did.
 *
 * Cached at module level rather than in a context provider: it is one small
 * static document, every page wants it, and a promise cache means N components
 * asking simultaneously still produce exactly one request.
 */

import { useEffect, useState } from "react";
import { api } from "./api";
import type { GlossaryTerm } from "./types";

let cache: Promise<Map<string, GlossaryTerm>> | null = null;

function load(): Promise<Map<string, GlossaryTerm>> {
  if (!cache) {
    cache = api
      .glossary()
      .then((terms) => new Map(terms.map((term) => [term.term.toLowerCase(), term])))
      // A failed glossary fetch must never take a page down. Tooltips are
      // helpful, not load-bearing: an empty map means `Tip` falls back to its
      // own `text`, or renders nothing at all.
      .catch(() => new Map<string, GlossaryTerm>());
  }
  return cache;
}

/** Every glossary term, keyed by lower-cased name. Empty until the fetch lands. */
export function useGlossary(): Map<string, GlossaryTerm> {
  const [terms, setTerms] = useState<Map<string, GlossaryTerm>>(new Map());
  useEffect(() => {
    let alive = true;
    load().then((loaded) => {
      if (alive) setTerms(loaded);
    });
    return () => {
      alive = false;
    };
  }, []);
  return terms;
}

/** Definition for one term, or null while loading / if the term is unknown. */
export function useDefinition(term: string | undefined): string | null {
  const terms = useGlossary();
  if (!term) return null;
  return terms.get(term.toLowerCase())?.definition ?? null;
}
