"""The one visual identity both front ends share.

`assets/mark.svg` is a hand-drawn glyph — three OHLC candles in the app's own
Buy/Hold/Sell colours — and it is what `page_icon` points at on every page.

It replaces a per-page emoji (📈 🔎 🔬 💼 📊 ⚙️ 📖). Emoji are not an icon set:
each one renders as a different picture on every platform, they cannot be tinted
or sized as a family, and a different one per page means the tab strip has no
identity that says "these seven tabs are the same app". A single drawn mark does.

Streamlit reads a local `.svg` path, inlines it as a data URI and serves it as
the favicon, so this costs no extra request and no binary in the repository.
"""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

#: Passed to every page's `st.set_page_config(page_icon=...)`.
PAGE_ICON = str(ASSETS / "mark.svg")

__all__ = ["ASSETS", "PAGE_ICON"]
