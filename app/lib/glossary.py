"""Re-export of the shared glossary (`quantpulse.glossary`).

The term data moved into `src/quantpulse/` when the React frontend arrived —
it is reference content both UIs need, not Streamlit presentation logic. This
shim keeps `from lib.glossary import tip` working across the Streamlit pages,
so promoting it cost the app nothing.
"""

from quantpulse.glossary import CATEGORIES, TERMS, define, search_terms, tip

__all__ = ["TERMS", "CATEGORIES", "tip", "define", "search_terms"]
