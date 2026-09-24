"""Keep the nightly schedule from switching itself off (finding 26).

GitHub: *"In a public repository, scheduled workflows are automatically
disabled when no repository activity has occurred in 60 days."* Until point 18
the refresh committed the demo database every weeknight, so the repository was
never idle. Since the database moved to a release asset, the only commits are a
person's, and a portfolio project can easily go two months without one -- after
which the demo quietly stops updating, with nothing red anywhere.

GitHub does not say what counts as "repository activity". Reports disagree
about releases and API calls; the one thing every account agrees counts is a
commit on the default branch. So when ``main`` has been idle for
``DEFAULT_MAX_IDLE_DAYS``, this writes a short status file and the workflow
commits it. When anyone else has committed recently it touches nothing -- point
18 existed to stop the nightly bloating the history, and a heartbeat must not
bring that back.

Standard library only: ``.github/workflows/keepalive.yml`` runs it with the
runner's system Python, without installing the project.

    python scripts/keep_schedule_alive.py --last-commit-epoch 1790000000 \\
        --status-file docs/refresh_status.md --github-output "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

#: GitHub's documented limit for public repositories.
GITHUB_INACTIVITY_LIMIT_DAYS = 60

#: Idle days before a heartbeat commit. Half the limit: the refresh runs on
#: weekdays, so the remaining thirty days are about twenty more nightly chances
#: to land the commit if one night fails. In practice this is at most one
#: commit a month, and none in a month with any other commit.
DEFAULT_MAX_IDLE_DAYS = 30


def needs_heartbeat(last_commit: datetime, now: datetime, max_idle_days: int) -> bool:
    """True when ``main`` has been idle for at least ``max_idle_days``.

    ``0`` always answers yes, which is how the commit path is proven by hand
    (``workflow_dispatch`` with ``max_idle_days: 0``) instead of by waiting a month.
    """
    if not 0 <= max_idle_days < GITHUB_INACTIVITY_LIMIT_DAYS:
        raise ValueError(
            f"max_idle_days={max_idle_days}: must be below GitHub's "
            f"{GITHUB_INACTIVITY_LIMIT_DAYS}-day limit, or the schedule is disabled first"
        )
    return now - last_commit >= timedelta(days=max_idle_days)


def render_status(
    *,
    now: datetime,
    published_at: str,
    published_bytes: int,
    refresh_result: str,
    run_url: str,
) -> str:
    """The committed file: the last-known state of the pipeline, and why it exists."""
    size = f"{published_bytes / 1e6:.1f} MB" if published_bytes else "size unknown"
    published = f"{published_at} ({size})" if published_at else "unknown"
    refresh = (
        f"{refresh_result} -- {run_url}"
        if refresh_result
        else f"not part of a refresh run (manual heartbeat) -- {run_url}"
    )
    return (
        "# Refresh status\n"
        "\n"
        "Written by `.github/workflows/keepalive.yml`, and committed only when\n"
        f"`main` has had no commit for {DEFAULT_MAX_IDLE_DAYS} days. GitHub disables\n"
        "a public repository's scheduled workflows after 60 days without activity,\n"
        "and since the demo database moved to a release asset the refresh no\n"
        "longer commits anything itself. This commit is what keeps the nightly\n"
        "schedule on. Deleting the file is harmless; the next heartbeat rewrites it.\n"
        "\n"
        f"- Heartbeat: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"- Refresh run: {refresh}\n"
        f"- Demo database last published: {published}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--last-commit-epoch", type=int, required=True)
    parser.add_argument("--max-idle-days", type=int, default=DEFAULT_MAX_IDLE_DAYS)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--now", default=None, help="ISO timestamp; defaults to the clock")
    parser.add_argument("--published-at", default="")
    parser.add_argument("--published-bytes", type=int, default=0)
    parser.add_argument("--refresh-result", default="")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args(argv)

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(UTC)
    last = datetime.fromtimestamp(args.last_commit_epoch, UTC)
    idle = (now - last).total_seconds() / 86400
    commit = needs_heartbeat(last, now, args.max_idle_days)

    if commit:
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
        args.status_file.write_text(
            render_status(
                now=now,
                published_at=args.published_at,
                published_bytes=args.published_bytes,
                refresh_result=args.refresh_result,
                run_url=args.run_url,
            )
        )
        print(
            f"main has been idle {idle:.1f} days (threshold {args.max_idle_days}); "
            f"writing {args.status_file} for a heartbeat commit"
        )
    else:
        print(
            f"main last committed {idle:.1f} days ago (threshold {args.max_idle_days}); "
            f"nothing to do -- GitHub's {GITHUB_INACTIVITY_LIMIT_DAYS}-day clock was reset by it"
        )
    with args.github_output.open("a") as out:
        out.write(f"commit={'true' if commit else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
