"""Every text colour clears WCAG AA against every background it sits on (finding 31).

axe-core on the light theme found 13-116 `color-contrast` failures per page, all
one token: `--muted` (#78746a) was 4.39:1 on the paper background and 4.06:1 on
sunken panels, under AA's 4.5:1 for body-size text. The dark theme passes, which
is probably why the audit -- run in dark mode -- reported zero. The token is now
#706c63: same hue, 4.93 / 4.56 / 5.23 on page / sunken / card.

Computed here rather than by driving axe in a browser: the rule is arithmetic on
the stylesheet's own tokens, so a palette edit that breaks it fails in
milliseconds with the pair named. The value is also kept by three other files
(the second light block, the chart fallback, Streamlit's theme), and those must
agree or the two front ends drift apart again.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STYLES = (REPO / "frontend" / "src" / "styles.css").read_text()

AA_BODY_TEXT = 4.5
TEXT_TOKENS = ("text", "text-quiet", "muted")
BACKGROUNDS = ("bg", "surface", "sunken")


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def _block(selector_start: str) -> dict[str, str]:
    start = STYLES.index(selector_start)
    return _tokens(STYLES[start : STYLES.index("}", start)])


THEMES = {
    "light": _block(":root {"),
    "dark": _block(':root[data-theme="dark"],'),
    "light (OS preference)": _block(':root:not([data-theme="dark"]) {'),
}


def test_contrast_is_computed_correctly() -> None:
    # WCAG's own reference points, so the arithmetic below is not trusted blind.
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_every_text_token_clears_aa_on_every_background(theme: str) -> None:
    tokens = THEMES[theme]
    assert set(TEXT_TOKENS) | set(BACKGROUNDS) <= set(tokens), f"{theme}: tokens not found"
    failures = [
        f"--{fg} {tokens[fg]} on --{bg} {tokens[bg]}: {contrast(tokens[fg], tokens[bg]):.2f}"
        for fg in TEXT_TOKENS
        for bg in BACKGROUNDS
        if contrast(tokens[fg], tokens[bg]) < AA_BODY_TEXT
    ]
    assert not failures, f"{theme} theme below {AA_BODY_TEXT}:1 -- " + "; ".join(failures)


def _toml_section(header: str) -> str:
    text = (REPO / ".streamlit" / "config.toml").read_text()
    start = text.index(header + "\n")
    end = text.find("\n[", start + len(header))
    return text[start : end if end != -1 else len(text)]


def test_every_copy_of_the_light_muted_colour_agrees() -> None:
    muted = THEMES["light"]["muted"].lower()
    copies = {
        "styles.css (OS-preference light block)": THEMES["light (OS preference)"]["muted"],
        "lib/theme.ts fallback": re.search(
            r'muted:\s*"(#[0-9a-fA-F]{6})"',
            (REPO / "frontend" / "src" / "lib" / "theme.ts").read_text(),
        ).group(1),  # type: ignore[union-attr]
        # The light section only -- the dark theme has its own, passing, grey.
        ".streamlit/config.toml [theme.light] grayColor": re.search(
            r'grayColor\s*=\s*"(#[0-9a-fA-F]{6})"', _toml_section("[theme.light]")
        ).group(1),  # type: ignore[union-attr]
    }
    disagree = {where: value for where, value in copies.items() if value.lower() != muted}
    assert not disagree, f"--muted is {muted} but {disagree}"


# ---------------------------------------------------------------- rating chips

_CHIP_RULE = re.compile(
    r'\.chip\[data-rating="([a-z_]+)"\]\s*\{\s*--chip-ink:\s*var\(--([a-z-]+)(?:,[^)]*)?\);\s*'
    r"--chip-tint:\s*color-mix\(in srgb,\s*var\(--([a-z-]+)\)\s*(\d+)%,\s*transparent\);"
)


def _over(tint: str, percent: int, background: str) -> str:
    """`color-mix(in srgb, tint N%, transparent)` painted over `background`."""
    t = [int(tint[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(background[i : i + 2], 16) for i in (1, 3, 5)]
    p = percent / 100
    return "#" + "".join(f"{round(p * x + (1 - p) * y):02x}" for x, y in zip(t, b, strict=True))


def test_every_rating_chip_is_found() -> None:
    assert {m.group(1) for m in _CHIP_RULE.finditer(STYLES)} == {
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
    }


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_every_rating_chip_clears_aa_on_its_own_tint(theme: str) -> None:
    """The chips failed in the light theme: Buy 3.27-3.72, Hold 3.81-4.32, Sell 4.12-4.71.

    Their text took `--up/--flat/--down`, the tokens the charts use as marks. The
    chip text now has its own darker inks (`--up-ink` etc.), same hue; the tint
    and the charts keep the mark colour.
    """
    tokens = THEMES[theme]
    failures = []
    for match in _CHIP_RULE.finditer(STYLES):
        rating, ink_token, tint_token, percent = match.groups()
        assert ink_token in tokens, f"{theme}: --{ink_token} is not defined"
        for bg in BACKGROUNDS:
            ratio = contrast(tokens[ink_token], _over(tokens[tint_token], int(percent), tokens[bg]))
            if ratio < AA_BODY_TEXT:
                failures.append(
                    f"{rating} (--{ink_token} on --{tint_token} over --{bg}): {ratio:.2f}"
                )
    assert not failures, f"{theme}: " + "; ".join(failures)
