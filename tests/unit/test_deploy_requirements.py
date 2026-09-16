"""`requirements.txt` -- what the deployed app installs -- stays honest.

Two things can quietly break the deploy without breaking anything a developer
runs locally, so both are pinned here:

1. A dependency bump updates `uv.lock` and nobody regenerates `requirements.txt`,
   so the hosted app runs a version the test suite never saw.
2. Someone adds the machine-learning stack to the app's dependency set. It is
   ~2.5 GB of wheels on Linux and does not fit the free tier; the app has never
   needed it (the models belong to the nightly refresh job).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sync_requirements import APP_PACKAGES, locked_versions, main, render

REPO = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO / "requirements.txt"

# The nightly job's models, plus thinc (spaCy's tensor library) so a transitive
# pull-in is caught too.
HEAVY = ("torch", "transformers", "spacy", "thinc")


def test_requirements_txt_is_in_sync_with_the_lockfile() -> None:
    assert REQUIREMENTS.read_text() == render(), (
        "requirements.txt is stale -- run `python scripts/sync_requirements.py`"
    )


def test_check_mode_passes_on_the_committed_file() -> None:
    assert main(["--check"]) == 0


def test_the_machine_learning_stack_is_not_installed_by_the_deployed_app() -> None:
    text = REQUIREMENTS.read_text()
    for package in HEAVY:
        assert f"\n{package}==" not in text, (
            f"{package} would be installed on Streamlit Community Cloud, which has "
            "neither the disk nor the memory for it. The app does not import it -- "
            "the nightly refresh job does."
        )


@pytest.mark.parametrize("package", APP_PACKAGES)
def test_every_pinned_package_is_a_real_locked_version(package: str) -> None:
    """Pins come from `uv.lock`, so a typo in the name list fails here."""
    assert package in locked_versions()


class TestDemoDatabaseIsFetched:
    """The demo database is a release asset now, not a committed file.

    Forty committed revisions of it were 309 MB of a 317 MB repository — 97.5%,
    against 8 MB for all the source, tests and docs together — growing about
    7.7 MB per refresh, five refreshes a week.

    This check deliberately does **not** live in
    `tests/integration/test_ui_pages_real_data.py`, which is the module that
    needs the file. That module carries a `skipif` on the file's existence, so a
    guard placed there is inside the very skip it exists to detect: with the
    database removed it skipped quietly, exactly like the ten tests it was meant
    to protect.
    """

    def test_ci_has_the_database_before_the_tests_that_need_it(self) -> None:
        if not os.environ.get("CI"):
            pytest.skip("not CI; a fresh clone has not fetched the release asset yet")
        assert (REPO / "quantpulse_demo.db").exists(), (
            "quantpulse_demo.db is missing in CI. It is a GitHub Release asset — the "
            "workflow must run scripts/fetch_demo_db.sh before pytest, or the ten "
            "page-render tests skip silently and CI checks nothing."
        )

    def test_every_workflow_that_reads_the_database_fetches_it_first(self) -> None:
        """Ordering, not mere presence: fetching after the step that reads the
        file is the same outage with more YAML.

        **The workflow list is read off the directory, not typed here.** It used
        to say `("ci.yml", "pages.yml")` -- and `refresh_data.yml`, the one
        workflow that *republishes* the database, was therefore the only one
        nothing checked. It had no fetch step at all, so on 2026-09-15 it
        migrated an empty file into existence, filled it with four weeks of
        data, and clobbered three years of history with it. A literal list is a
        guard that stops covering the thing you add next.
        """
        for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            name = path.name
            text = path.read_text()
            reads_or_writes_the_database = (
                "pytest" in text
                or "build_static_site.py" in text
                or "refresh_data.py" in text
                or "gh release upload" in text
            )
            if not reads_or_writes_the_database:
                continue
            assert "fetch_demo_db.sh" in text, f"{name} never fetches the demo database"
            # Matched as the command a step actually runs, not as a bare
            # filename: `build_static_site.py` is named in pages.yml's header
            # comment on line 9, so searching for the filename compared the
            # fetch against a sentence about the file and failed on a workflow
            # that was correctly ordered.
            fetch = text.index("./scripts/fetch_demo_db.sh")
            # After the install, because the fetch imports `quantpulse`. Pages
            # failed in nine seconds on `ModuleNotFoundError: No module named
            # 'quantpulse'` with the step correctly placed before every reader
            # and wrongly placed before `uv sync` -- which the reader-only
            # version of this assertion could not see.
            install = text.index("run: uv sync --locked")
            assert install < fetch, f"{name} fetches the database before installing the package"
            for reader in ("run: uv run pytest", "run: uv run python scripts/build_static_site.py"):
                if reader in text:
                    assert fetch < text.index(reader), (
                        f"{name} runs `{reader}` before fetching the database"
                    )
            # The refresh job is the one that can *destroy* the asset, and it
            # has two orderings of its own. `alembic upgrade head` is what
            # quietly creates an empty database when the real one is absent, so
            # it must come after the fetch -- and so must the refresh script,
            # which would otherwise fill that empty file in and make it look
            # like a successful run.
            #
            # Matched in their `run:` form, for the reason written above the
            # fetch anchor: the step that fixed this bug also *explains* it in a
            # comment, so `text.index("alembic upgrade head")` found the prose
            # at byte 4379 and compared the fetch against a sentence about
            # migrating rather than against the migration. This assertion failed
            # on correct YAML until it matched the command.
            migrate = "run: uv run alembic upgrade head"
            if migrate in text:
                assert fetch < text.index(migrate), (
                    f"{name} migrates before fetching the database, which creates an empty one"
                )
            refresh = "uv run python scripts/refresh_data.py"
            if refresh in text:
                assert fetch < text.index(refresh), (
                    f"{name} runs the refresh before fetching the database"
                )
            # And nothing may clobber the published asset without the census
            # guard having compared what is about to be uploaded against what
            # was fetched.
            if "gh release upload" in text:
                assert "demo_db_guard.py check" in text, (
                    f"{name} uploads over the released database with no census guard in front of it"
                )
                assert text.index("demo_db_guard.py check") < text.index("gh release upload"), (
                    f"{name} runs the census guard after the upload it is supposed to prevent"
                )

    def test_ci_reads_the_pinned_database_and_the_demo_reads_the_rolling_one(self) -> None:
        """The two assets exist to keep data failures out of code pull requests.

        CI used to fetch `demo-data`, which the refresh replaces five nights a
        week. When one of those nights published a gutted database, three
        unrelated frontend pull requests went red on a Streamlit test about
        effective weights. Both halves of the split are asserted here, because
        pinning CI while quietly pinning the demo too would freeze the shop
        window, and pinning neither is where this started.
        """
        workflows = REPO / ".github" / "workflows"
        ci = (workflows / "ci.yml").read_text()
        assert "fetch_demo_db.sh --ci-fixture" in ci, (
            "the test job must read the pinned database, or a bad refresh fails "
            "pull requests that never touched it"
        )
        # A pinned snapshot is a schema snapshot: a migration in the pull
        # request under test has to be applied to it before anything reads it.
        assert ci.index("run: uv run alembic upgrade head") < ci.index("run: uv run pytest"), (
            "ci.yml runs the tests before migrating the pinned database to head"
        )
        for name in ("pages.yml", "refresh_data.yml"):
            text = (workflows / name).read_text()
            assert "--ci-fixture" not in text, (
                f"{name} serves or refreshes the live demo and must read the rolling "
                "asset, not the pinned test copy"
            )

    def test_no_job_level_env_uses_a_step_only_context(self) -> None:
        """Valid YAML is not the same as a valid workflow.

        `runner`, `steps`, `job` and `env` itself are step-level contexts.
        A job-level `env:` that references one -- `${{ runner.temp }}/x.json`
        was this file's version -- is refused by GitHub *as a whole file*: the
        run is created, fails in zero seconds with "workflow file issue", and
        appears in the run list under `.github/workflows/<name>.yml` rather than
        the workflow's name. `yaml.safe_load` reads it without complaint, so
        `test_every_workflow_is_parseable_yaml` passes throughout, and the only
        feedback is a red run after a push.
        """
        yaml = pytest.importorskip("yaml")
        step_only = ("runner.", "steps.", "job.", "env.")
        for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            document = yaml.safe_load(path.read_text())
            for job_name, job in document["jobs"].items():
                for key, value in (job.get("env") or {}).items():
                    assert not any(context in str(value) for context in step_only), (
                        f"{path.name}: job `{job_name}` sets env `{key}` from a step-level "
                        f"context ({value!r}). GitHub refuses to parse the whole file."
                    )

    def test_every_workflow_is_parseable_yaml(self) -> None:
        """A syntax error here is only ever found by pushing it.

        `run: echo "publishing $(...): $(...)"` on one line is invalid YAML --
        the colon inside the string ends the mapping key -- and GitHub answers
        that with a job that fails in nine seconds having run nothing. Nothing
        else in this repository parses these files.
        """
        yaml = pytest.importorskip("yaml")
        for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
            document = yaml.safe_load(path.read_text())
            assert isinstance(document, dict), f"{path.name} is not a mapping"
            assert document.get("jobs"), f"{path.name} defines no jobs"

    def test_the_database_is_not_tracked_in_git(self) -> None:
        """The whole point. A file both ignored and tracked stays tracked, and
        the repository keeps growing while the ignore rule suggests otherwise.
        """
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "quantpulse_demo.db"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode != 0, (
            "quantpulse_demo.db is still tracked by git — `.gitignore` does not apply "
            "to files already in the index; it needs `git rm --cached`."
        )
