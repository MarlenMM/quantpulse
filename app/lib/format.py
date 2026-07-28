"""Display formatting shared across the Streamlit pages (Section 12).

Pure string/number formatting -- no Streamlit import, so every rule here is
unit-testable without a running app. Two of Section 12's rules are encoded as
functions rather than left to each page's discretion:

* **Never encode a rating with color alone.** `rating_label` always returns an
  arrow *and* a word ("▲ Buy"), because red/green alone is illegible to the
  ~1 in 12 men with red-green color blindness. Pages call this instead of
  styling a cell, so a page physically cannot ship a color-only rating.
  (The wider accessibility pass -- tooltips, a glossary, the mobile check --
  is Phase 10's second Section-21 row; this is just the part that would have
  to be un-built later if it weren't done from the start.)
* **Never present a stale or thin number as if it were fresh and complete.**
  `freshness_label` and `confidence_label` turn a date and a coverage score
  into the visible badges Section 12 asks for, including an explicit "never
  run" state so an empty pipeline can't read as an up-to-date one.
"""

from __future__ import annotations

from datetime import date

__all__ = [
    "RATING_ORDER",
    "RATING_DISPLAY",
    "ACTION_DISPLAY",
    "rating_label",
    "rating_color",
    "action_label",
    "format_price",
    "format_money",
    "format_percent",
    "format_signed_percent",
    "format_score",
    "format_ratio",
    "freshness_label",
    "confidence_label",
    "humanize",
]

RATING_ORDER = ("strong_buy", "buy", "hold", "sell", "strong_sell")

# icon + word + color. The icon and word carry the meaning; the color is
# decoration on top of them, never the sole channel (Section 12).
RATING_DISPLAY: dict[str, tuple[str, str, str]] = {
    "strong_buy": ("▲▲", "Strong Buy", "#1a7f37"),
    "buy": ("▲", "Buy", "#2da44e"),
    "hold": ("■", "Hold", "#9a6700"),
    "sell": ("▼", "Sell", "#cf222e"),
    "strong_sell": ("▼▼", "Strong Sell", "#a40e26"),
}

# Section 9's per-holding guidance vocabulary, same icon+word discipline.
ACTION_DISPLAY: dict[str, tuple[str, str, str]] = {
    "add": ("▲", "Add", "#2da44e"),
    "hold": ("■", "Hold", "#9a6700"),
    "trim": ("▼", "Trim", "#cf222e"),
    "sell": ("▼▼", "Sell", "#a40e26"),
}


def humanize(label: str | None) -> str:
    """`strong_buy` -> `Strong Buy`; `None`/empty -> an em dash."""
    if not label:
        return "—"
    return label.replace("_", " ").title()


def rating_label(rating: str | None) -> str:
    """A rating as icon + text, e.g. "▲ Buy" -- never color alone (Section 12)."""
    if not rating:
        return "—"
    icon, text, _ = RATING_DISPLAY.get(rating, ("", humanize(rating)))[:2] + ("",)
    return f"{icon} {text}".strip()


def rating_color(rating: str | None) -> str:
    """The decorative color for a rating; always paired with `rating_label`'s text."""
    if not rating or rating not in RATING_DISPLAY:
        return "#6e7781"
    return RATING_DISPLAY[rating][2]


def action_label(action: str | None) -> str:
    """A portfolio action as icon + text, e.g. "▼ Trim"."""
    if not action:
        return "—"
    entry = ACTION_DISPLAY.get(action)
    if entry is None:
        return humanize(action)
    icon, text, _ = entry
    return f"{icon} {text}"


def format_price(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def format_money(value: float | None) -> str:
    """Whole-dollar money, for totals where cents are noise."""
    return "—" if value is None else f"${value:,.0f}"


def format_percent(value: float | None, *, digits: int = 1) -> str:
    """A fraction (0.153) as a percentage ("15.3%")."""
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def format_pct_already_scaled(value: float | None, *, digits: int = 1) -> str:
    """A value already on a 0-100 scale (62.0) as a percentage ("62.0%").

    Distinct from `format_percent`, which multiplies a 0-1 fraction by 100 --
    using that on an already-0-100 value (e.g. `market_regime.compute_breadth`'s
    "share, 0-100" return) silently inflates it 100x (62.0 -> "6200.0%").
    """
    return "—" if value is None else f"{value:.{digits}f}%"


def format_signed_percent(value: float | None, *, digits: int = 1) -> str:
    """A fraction as a signed percentage ("+15.3%" / "-4.0%") for changes."""
    return "—" if value is None else f"{value * 100:+.{digits}f}%"


def format_score(value: float | None, *, digits: int = 1) -> str:
    """A 0-100 score; an explicit em dash when the category had no data."""
    return "—" if value is None else f"{value:.{digits}f}"


def format_ratio(value: float | None, *, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def freshness_label(as_of: date | None, *, today: date | None = None) -> str:
    """Section 12's data-freshness badge: "today" / "3 days ago" / "never run".

    The "never run" case is distinct on purpose. An empty table and a
    just-refreshed one must not look the same, or the freshness indicator is
    reassuring exactly when it should be alarming.
    """
    if as_of is None:
        return "never run"
    reference = today or date.today()
    days = (reference - as_of).days
    if days < 0:
        return as_of.isoformat()  # a future-dated row is odd; show it literally
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def confidence_label(data_confidence: float | None) -> str:
    """Section 7.5's coverage score as a readable badge, not a bare number.

    A thinly-covered micro-cap and a heavily-covered mega-cap must not be
    presented with the same implied confidence (Section 12, Section 22).
    """
    if data_confidence is None:
        return "coverage unknown"
    if data_confidence >= 80:
        return f"good coverage ({data_confidence:.0f}%)"
    if data_confidence >= 50:
        return f"partial coverage ({data_confidence:.0f}%)"
    return f"thin coverage ({data_confidence:.0f}%)"
