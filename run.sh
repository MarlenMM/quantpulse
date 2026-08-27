#!/usr/bin/env bash
#
# Run the full QuantPulse app locally, in one command:
#
#     ./run.sh
#
# The published demo (https://marlenmm.github.io/quantpulse/) is read-only --
# GitHub Pages serves files, and the Portfolio Manager needs to write. This
# script is the other half: the whole seven-page Streamlit app, including the
# Portfolio Manager, against the demo database that is committed to this repo.
#
# It needs no API key and no account. Everything it shows comes from
# `quantpulse_demo.db`, which a dispatched GitHub Actions run refreshes.
#
# Deliberately built from `requirements.txt` rather than `uv sync`: that is the
# app's own dependency set (18 packages, ~30 seconds) instead of the whole
# project's, which includes the refresh job's machine-learning stack -- around
# 2.5 GB of wheels the app never imports. It is also exactly what the hosted
# deploy installs, so running this locally exercises the deployed shape.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv-app"
STAMP="$VENV/.requirements-stamp"

if [[ ! -f quantpulse_demo.db ]]; then
  echo "quantpulse_demo.db is missing. It is committed to this repository -- if" >&2
  echo "you are in a shallow or partial clone, run: git checkout quantpulse_demo.db" >&2
  exit 1
fi

# Reinstall only when requirements.txt has actually changed. Without this every
# start pays the install cost, which is the difference between "one command" and
# "one command you avoid running".
want="$(shasum requirements.txt | cut -d' ' -f1)"
have="$(cat "$STAMP" 2>/dev/null || true)"

if [[ "$want" != "$have" ]]; then
  if command -v uv >/dev/null 2>&1; then
    # uv downloads a matching interpreter itself, so this works even on a
    # machine whose only python3 is a version this project does not support.
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV" -r requirements.txt
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
  else
    echo "Python 3.12 is required and was not found." >&2
    echo >&2
    echo "Easiest fix -- install uv, which fetches the right Python for you:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo >&2
    echo "Then run ./run.sh again." >&2
    exit 1
  fi
  echo "$want" >"$STAMP"
fi

echo
echo "Starting QuantPulse at http://localhost:8501"
echo "Press Ctrl-C to stop."
echo

# `session` keeps each visitor's portfolio in their own browser session rather
# than in a shared file (ADR 4.5) -- the same setting the hosted app uses, and
# the reason nothing you enter can ever end up committed to this repository.
# Change it to `sqlite` if you want holdings to persist across restarts.
export DATABASE_URL="sqlite:///./quantpulse_demo.db"
export PORTFOLIO_BACKEND="${PORTFOLIO_BACKEND:-session}"

exec "$VENV/bin/streamlit" run app/Home.py "$@"
