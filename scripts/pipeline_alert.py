"""Tell someone when the pipeline breaks (finding 27).

Two subcommands, each run by a workflow job:

``failure``
    From an ``if: failure()`` job: names the failed jobs and steps of this run
    (read from ``gh run view --json jobs``) and posts them with the run's URL.
    Exits 0 once the notice is sent or, with no webhook configured, logged.

``staleness``
    From the nightly, before the refresh: reads the *published* ``health.json``
    and alerts when its newest price date is more than
    ``pipeline.STALE_AFTER_SESSIONS`` trading sessions behind. Exits 1 when the
    site is stale, whether or not a webhook is configured: the missing secret
    is not an error, the frozen site is -- and a red run is also what makes
    GitHub's own failure email fire.

Either exits 1 when a configured webhook refuses the post.

    uv run python scripts/pipeline_alert.py failure --workflow "Data Refresh" \\
        --run-url "$RUN_URL" --jobs-json jobs.json
    uv run python scripts/pipeline_alert.py staleness \\
        --site-url https://marlenmm.github.io/quantpulse/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quantpulse.alerting import discord, pipeline
from quantpulse.ingestion import http

logger = logging.getLogger("pipeline_alert")


def fetch_health(url: str) -> dict[str, Any]:
    """The published ``health.json``. The only network call in this script."""
    data = http.get_json(url, timeout=30.0)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _now() -> datetime:
    return datetime.now(UTC)


def _failure(args: argparse.Namespace) -> int:
    jobs = json.loads(Path(args.jobs_json).read_text() or "[]")
    if isinstance(jobs, dict):  # `gh run view --json jobs` wraps the array
        jobs = jobs.get("jobs", [])
    text = pipeline.failure_message(args.workflow, args.run_url, pipeline.failed_jobs(jobs))
    pipeline.deliver(text)
    return 0


def _staleness(args: argparse.Namespace) -> int:
    site = args.site_url if args.site_url.endswith("/") else f"{args.site_url}/"
    health_url = f"{site}data/health.json"
    try:
        newest_text = (fetch_health(health_url).get("freshness") or {}).get("prices")
        if not newest_text:
            raise ValueError("it has no freshness.prices date")
        newest = date.fromisoformat(str(newest_text))
    except Exception as error:  # noqa: BLE001 -- any failure to read it is the finding
        text = (
            f"QuantPulse: could not read the published health.json ({type(error).__name__}: "
            f"{error}). The demo may be down or unpublished.\n{site}"
        )
        print(text)
        pipeline.deliver(text)
        return 1

    missing = pipeline.sessions_missing(newest, _now())
    if missing <= pipeline.STALE_AFTER_SESSIONS:
        print(
            f"the published site's newest prices are {newest.isoformat()}: {missing} completed "
            f"session(s) newer, within the {pipeline.STALE_AFTER_SESSIONS} tolerated"
        )
        return 0
    text = pipeline.staleness_message(newest, missing, site_url=site)
    print(f"::error::{text.splitlines()[0]}")
    pipeline.deliver(text)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    failure = sub.add_parser("failure")
    failure.add_argument("--workflow", required=True)
    failure.add_argument("--run-url", required=True)
    failure.add_argument("--jobs-json", required=True)
    staleness = sub.add_parser("staleness")
    staleness.add_argument("--site-url", required=True)
    args = parser.parse_args(argv)

    try:
        return _failure(args) if args.command == "failure" else _staleness(args)
    except discord.WebhookError as error:
        # Already scrubbed of the URL by `discord.send`.
        print(f"::error::{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
