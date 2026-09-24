"""The README's secrets table must name every secret the workflows read (finding 28).

Five features wait on secrets only the owner can add, so the table is the whole
of the instructions for turning them on: the exact name, what it unlocks, and
where to get it. It had drifted -- no source for two of the keys, and nothing
about the pipeline notices the webhook now carries. A secret a workflow reads
and the README does not name is a feature nobody can switch on.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _workflow_secrets() -> set[str]:
    names: set[str] = set()
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        names |= set(re.findall(r"secrets\.([A-Z0-9_]+)", workflow.read_text()))
    return names - {"GITHUB_TOKEN"}


def _table_rows() -> dict[str, list[str]]:
    """Each row of the README's secrets table, keyed by every secret it names."""
    text = (ROOT / "README.md").read_text()
    start = text.index("| Secret | What it unlocks")
    rows: dict[str, list[str]] = {}
    for line in text[start:].splitlines()[2:]:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        for name in re.findall(r"`([A-Z0-9_]+)`", cells[0]):
            rows[name] = cells
    return rows


def test_the_workflows_read_the_secrets_this_test_expects() -> None:
    # Not vacuous: if the regex stopped matching, every check below would pass.
    assert {"FRED_API_KEY", "ALERT_DISCORD_WEBHOOK_URL", "ALPACA_API_KEY_ID"} <= _workflow_secrets()


def test_every_secret_a_workflow_reads_has_a_row() -> None:
    missing = _workflow_secrets() - set(_table_rows())
    assert not missing, f"README's secrets table does not name: {sorted(missing)}"


def test_every_row_says_what_it_unlocks_and_where_to_get_it() -> None:
    for name, cells in _table_rows().items():
        assert len(cells) == 3, f"{name}: expected Secret | unlocks | where, got {cells}"
        assert cells[1], f"{name}: says nothing about what it unlocks"
        assert cells[2], f"{name}: says nothing about where to get it"
