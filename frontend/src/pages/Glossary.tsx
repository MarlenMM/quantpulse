import { useMemo, useState } from "react";
import { ErrorBox, Loading } from "../components/Common";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

/**
 * Served from `/api/glossary`, which reads the same `quantpulse.glossary.TERMS`
 * dict the Streamlit tooltips use — so the two front ends cannot define a term
 * differently.
 */
export default function Glossary() {
  const { data, error, loading } = useApi(() => api.glossary(), []);
  const [query, setQuery] = useState("");

  const grouped = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matching = (data ?? []).filter(
      (t) =>
        !needle ||
        t.term.toLowerCase().includes(needle) ||
        t.definition.toLowerCase().includes(needle),
    );
    const byCategory = new Map<string, typeof matching>();
    for (const term of matching) {
      const list = byCategory.get(term.category) ?? [];
      list.push(term);
      byCategory.set(term.category, list);
    }
    return [...byCategory.entries()];
  }, [data, query]);

  if (loading) {
    return (
      <>
        <h1>Glossary</h1>
        <Loading what="the glossary" />
      </>
    );
  }
  if (error) return <ErrorBox error={error} />;

  return (
    <>
      <h1>Every number, in plain English</h1>
      <p className="standfirst">
        {(data ?? []).length} terms — every metric this app puts on screen, defined once. The
        same definitions back the ⓘ hints beside the figures, so a term cannot mean one thing
        in a tooltip and another here.
      </p>
      <label>
        Search
        <input
          type="search"
          value={query}
          placeholder="e.g. sharpe, beta, drawdown, FIFO…"
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>

      {grouped.length === 0 ? (
        <p className="muted" style={{ marginTop: "var(--s4)" }}>
          Nothing matches “{query}”. The search looks at both the term and its definition, so a
          plain-English word usually finds the jargon for it.
        </p>
      ) : null}

      {grouped.map(([category, terms]) => (
        <section className="block" key={category}>
          <h2>{category}</h2>
          <dl className="glossary">
            {terms
              .slice()
              .sort((a, b) => a.term.localeCompare(b.term))
              .map((term) => (
                <div key={term.term}>
                  <dt>{term.term}</dt>
                  <dd>{term.definition}</dd>
                </div>
              ))}
          </dl>
        </section>
      ))}
    </>
  );
}
