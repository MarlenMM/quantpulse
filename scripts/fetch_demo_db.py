"""CLI over `quantpulse.demo_data.fetch` -- see that module for the reasoning.

A thin wrapper on purpose: the Streamlit app cannot import anything under
`scripts/`, so the logic lives in the package and this is only the command-line
face of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from quantpulse.demo_data import (
    CI_FIXTURE_SHA256,
    CI_FIXTURE_URL,
    DEFAULT_TARGET,
    RELEASE_URL,
    fetch,
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # `--ci-fixture` asks for the pinned copy the test job reads, verified by
    # digest, rather than the rolling one the demo and `./run.sh` read. See
    # `quantpulse.demo_data` for why those are two different files.
    pinned = "--ci-fixture" in arguments
    positional = [argument for argument in arguments if not argument.startswith("-")]
    target = Path(positional[0]) if positional else DEFAULT_TARGET
    if target.exists():
        print(f"{target} is already here; leaving it alone.")
        return 0
    source = "pinned test" if pinned else "demo"
    print(f"Downloading the {source} database (~70 MB) to {target}...")
    fetch(
        target,
        url=CI_FIXTURE_URL if pinned else RELEASE_URL,
        expected_sha256=CI_FIXTURE_SHA256 if pinned else None,
    )
    print(f"Wrote {target} ({target.stat().st_size // 1048576} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
