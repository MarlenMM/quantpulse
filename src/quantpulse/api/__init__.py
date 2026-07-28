"""REST API over the analysis engine (ADR 4.1's React + FastAPI stretch goal).

Read-only by design — see `main.py` for why. The Streamlit app in `app/` and
this API are peers: both are thin presentation layers over the same
`storage.persistence` readers, and neither is allowed to compute analysis of
its own (Section 14's UI-agnostic engine rule, applied in both directions).
"""
