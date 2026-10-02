"""Finding 40: how frontend dependency bumps are grouped and checked (user's policy).

Measured 2026-10-02 on the four open Dependabot PRs, each through both browser
suites: plotly.js 4.1.1 and vite 8.3.1 passed alone; react-dom 19.3 passed alone
(its exact peer pulls react along in the lockfile); **react 19.3 alone never
mounted the app** -- React error #527, react 19.3.0 against react-dom 19.2.8 --
which is why that PR's frontend check was red. React and react-dom must move
together, and Dependabot opens them separately unless told otherwise.

The second gap was in CI. Pull requests ran only the stubbed e2e suite; the
static-site suite -- the one that loads real data and checks every Plotly
figure drew the trace type it asked for, the failure that has blanked the
charts before -- ran only in the publish workflow, after a merge. The owner's
call: group the bumps, and run the static suite on any pull request that
changes the frontend's dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"


def _npm_update() -> dict[str, Any]:
    config = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text())
    (npm,) = [u for u in config["updates"] if u["package-ecosystem"] == "npm"]
    return dict(npm)


def _group_for(package: str, groups: dict[str, Any]) -> str | None:
    """The group Dependabot puts `package` in: the first whose patterns match."""
    from fnmatch import fnmatch

    for name, group in groups.items():
        if any(fnmatch(package, pattern) for pattern in group.get("patterns", [])) and not any(
            fnmatch(package, pattern) for pattern in group.get("exclude-patterns", [])
        ):
            return str(name)
    return None


def test_react_and_react_dom_always_move_together() -> None:
    groups = _npm_update().get("groups", {})
    members = ("react", "react-dom", "@types/react", "@types/react-dom")
    placed = {package: _group_for(package, groups) for package in members}
    assert None not in placed.values(), f"not grouped: {placed}"
    assert len(set(placed.values())) == 1, f"split across groups: {placed}"
    # Every update type, majors included: a major is exactly when they must match.
    group = groups[placed["react"]]
    assert "update-types" not in group, "the React group must not be limited to minor/patch"


def test_other_minor_and_patch_bumps_arrive_as_one_pull_request() -> None:
    groups = _npm_update().get("groups", {})
    for package in ("plotly.js", "vite", "react-plotly.js", "@playwright/test"):
        name = _group_for(package, groups)
        assert name is not None, f"{package} is not grouped"
        assert sorted(groups[name]["update-types"]) == ["minor", "patch"], (
            f"{package}'s group should hold minor and patch bumps only; majors stay separate"
        )


def _static_job() -> tuple[dict[str, Any], str]:
    ci_text = (WORKFLOWS / "ci.yml").read_text()
    jobs = yaml.safe_load(ci_text)["jobs"]
    matching = [
        (name, job)
        for name, job in jobs.items()
        if any("test:static" in str(step.get("run", "")) for step in job.get("steps", []))
    ]
    assert matching, "ci.yml runs no static-site suite"
    (name, job) = matching[0]
    return dict(job), name


def test_a_dependency_pull_request_runs_the_static_suite() -> None:
    job, name = _static_job()
    steps = job["steps"]
    runs = [str(step.get("run", "")) for step in steps]
    joined = "\n".join(runs)
    # On pull requests, gated on the frontend's dependency files changing.
    assert "pull_request" in str(job.get("if", "")), f"{name} must run on pull requests"
    assert "frontend/package-lock.json" in joined, (
        f"{name} must decide from whether the lockfile changed"
    )
    # The pinned database, migrated, then pre-rendered, built in static mode,
    # given its route pages -- the publish workflow's sequence.
    order = [
        "./scripts/fetch_demo_db.sh --ci-fixture",
        "uv run alembic upgrade head",
        "uv run python scripts/build_static_site.py",
        "npm run build",
        "scripts/emit_route_pages.py",
        "npm run test:static",
    ]
    positions = [next(i for i, run in enumerate(runs) if anchor in run) for anchor in order]
    assert positions == sorted(positions), f"{name} runs its steps out of order: {order}"
    build = next(step for step in steps if "npm run build" in str(step.get("run", "")))
    assert build.get("env", {}).get("VITE_STATIC_API") == "1", (
        "the static build needs VITE_STATIC_API"
    )
    # Every step after the change check is conditional on it, so a pull request
    # that does not touch the dependencies pays nothing.
    check = next(
        i for i, step in enumerate(steps) if "id" in step and "changed" in str(step.get("run", ""))
    )
    for step in steps[check + 1 :]:
        assert "steps." in str(step.get("if", "")), f"{name}: unconditional step {step.get('name')}"
