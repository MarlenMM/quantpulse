"""Glossary — every term the app uses, in plain English (Sections 10, 12).

Section 10: "worth doing given a recruiter reviewing your live demo may not be
a finance person, and it signals you're thinking about your users, not just
your algorithms."

The page is a thin renderer over `lib.glossary.TERMS`, which is also what every
inline `help=` tooltip in the app reads from — so a definition can never say one
thing here and another beside the number it explains.
"""

import streamlit as st

from lib.glossary import CATEGORIES, TERMS, search_terms

st.set_page_config(page_title="QuantPulse — Glossary", page_icon="📖", layout="wide")


def main() -> None:
    st.title("📖 Glossary")
    st.caption(
        "Every metric this app shows, explained without jargon. The same definitions "
        "appear as ⓘ tooltips next to the numbers themselves."
    )

    query = st.text_input(
        "Search",
        placeholder="e.g. sharpe, beta, drawdown, FIFO…",
        help="Searches term names and their definitions.",
    )
    matching = set(search_terms(query))
    if query and not matching:
        st.warning(f"No glossary entry matches “{query}”.")
        return
    if query:
        st.caption(f"{len(matching)} matching term{'s' if len(matching) != 1 else ''}.")

    for category in CATEGORIES:
        terms = [
            term
            for term, (term_category, _) in TERMS.items()
            if term_category == category and term in matching
        ]
        if not terms:
            continue
        st.subheader(category)
        for term in sorted(terms):
            with st.expander(term, expanded=bool(query)):
                st.markdown(TERMS[term][1])

    st.divider()
    st.caption(
        "Definitions are descriptive, not advice. Nothing here tells you what to buy, "
        "sell or hold — see the disclaimer on the Settings page."
    )


main()
