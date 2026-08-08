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
