#!/usr/bin/env bash
# Thin wrapper so shell callers (run.sh, both workflows) and the Streamlit app
# share one implementation. See scripts/fetch_demo_db.py for why this is Python.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v uv >/dev/null 2>&1; then
  uv run python scripts/fetch_demo_db.py "$@"
else
  python3 scripts/fetch_demo_db.py "$@"
fi
