"""Shared helpers for the Streamlit app (Section 12).

Kept out of `pages/` because Streamlit turns every module in that directory
into a navigable page. Importable as `lib.<module>` from `Home.py` and from any
page, since Streamlit puts the entrypoint's directory on `sys.path`.

The split mirrors Section 14's layering rule in miniature: `data.py` is the only
module that touches the database (and only through `storage.persistence`, never
raw SQL), `charts.py` only arranges already-computed numbers into figures, and
`format.py` is pure strings. Nothing here computes analysis -- that all lives in
`src/quantpulse/`, which never imports from `app/`.
"""
