"""Cross-fade the README screenshots into `docs/screenshots/demo.gif`.

The other half of the recipe `docs/screenshots/README.md` describes. It used to
say "the four PNGs cross-faded together with Pillow" and leave the reader to
work out the frame count, the hold time and the palette; this is that sentence
as something you can run.

Deliberately a slow cross-fade rather than a screen recording. A recording of
someone driving the app carries a cursor, scroll jitter and whatever the frame
grabber does to type rendering, and it is heavier for the same information. Four
still pages that dissolve into each other read as "here is what the product
looks like", which is what a README image is for.

    uv run python scripts/build_demo_gif.py

Every input frame must already be the same size; a mismatch is an error rather
than a silent resize, because a resized screenshot is a blurry screenshot and
the whole point of these is to show the type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from PIL.Image import Dither, Quantize, Resampling

REPO = Path(__file__).resolve().parents[1]
SHOTS = REPO / "docs" / "screenshots"

#: In the order a reader would meet them: the market-wide view, the ranking it
#: comes from, one company in full, and the honest track record underneath.
FRAMES = ("dashboard.png", "screener.png", "stock_detail.png", "backtest.png")

#: Milliseconds a page is held still before it starts dissolving. Long enough to
#: read the headline and the first few table rows.
HOLD_MS = 2_600

#: Intermediate frames in each dissolve, and how long each is shown. Eight steps
#: at 60ms is a ~half-second fade: fast enough not to feel like a slideshow,
#: slow enough that it reads as a transition rather than a cut.
FADE_STEPS = 8
FADE_MS = 60

#: Half width, which is the retina screenshot at its intended size. A 2880px-wide
#: GIF is a 12MB file that GitHub scales down anyway.
SCALE = 0.5


def build(source: Path, target: Path) -> Path:
    frames = [Image.open(source / name).convert("RGB") for name in FRAMES]

    sizes = {frame.size for frame in frames}
    if len(sizes) != 1:
        raise SystemExit(
            f"frames differ in size ({sorted(sizes)}); recapture them at one viewport "
            "rather than resizing, which softens the type these images exist to show"
        )

    width, height = frames[0].size
    size = (int(width * SCALE), int(height * SCALE))
    scaled = [frame.resize(size, Resampling.LANCZOS) for frame in frames]

    sequence: list[Image.Image] = []
    durations: list[int] = []
    for index, frame in enumerate(scaled):
        sequence.append(frame)
        durations.append(HOLD_MS)
        nxt = scaled[(index + 1) % len(scaled)]
        for step in range(1, FADE_STEPS + 1):
            sequence.append(Image.blend(frame, nxt, step / (FADE_STEPS + 1)))
            durations.append(FADE_MS)

    # One adaptive palette for the whole animation rather than per frame: a
    # per-frame palette makes the dissolve shimmer, because the colours shift
    # under the blend as well as the blend itself.
    palette = sequence[0].quantize(colors=255, method=Quantize.MEDIANCUT)
    quantized = [
        image.quantize(palette=palette, dither=Dither.FLOYDSTEINBERG) for image in sequence
    ]

    target.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        target,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SHOTS)
    parser.add_argument("--out", type=Path, default=SHOTS / "demo.gif")
    args = parser.parse_args(argv)

    written = build(args.source, args.out)
    print(f"wrote {written} ({written.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
