"""CLI over `quantpulse.demo_data.fetch` -- see that module for the reasoning.

A thin wrapper on purpose: the Streamlit app cannot import anything under
`scripts/`, so the logic lives in the package and this is only the command-line
face of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from quantpulse.demo_data import DEFAULT_TARGET, fetch


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    target = Path(arguments[0]) if arguments else DEFAULT_TARGET
    if target.exists():
        print(f"{target} is already here; leaving it alone.")
        return 0
    print(f"Downloading the demo database (~66 MB) to {target}...")
    fetch(target)
    print(f"Wrote {target} ({target.stat().st_size // 1048576} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
