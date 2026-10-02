import pytest
from hypothesis import settings

from quantpulse.ingestion.circuit_breaker import reset_all_breakers

# Section 29's property-based layer (`tests/property/`) runs on every push
# (Section 17's CI), so the example budget is capped well below Hypothesis's
# 100-per-test default -- these tests build DataFrames/dataclasses per example,
# not bare scalars, and an uncapped suite would slow CI for marginal extra
# coverage. `deadline=None` because that per-example cost (not a runaway loop)
# is exactly what would otherwise trip Hypothesis's flaky-looking timing check.
settings.register_profile("quantpulse-ci", max_examples=40, deadline=None)
settings.load_profile("quantpulse-ci")


@pytest.fixture(autouse=True)
def _reset_circuit_breakers() -> None:
    """Keep the module-global circuit-breaker registry from leaking across tests."""
    reset_all_breakers()
    yield
    reset_all_breakers()


def pytest_collection_finish(session: pytest.Session) -> None:
    """Record the collected count when -- and only when -- it is the whole suite.

    `tests/unit/test_documented_counts.py` checks the README's test count against
    it (finding 38). A `-k`, `-m`, `--deselect` or path argument collects a
    fraction, so then nothing is recorded and that check skips.
    """
    config = session.config
    option = config.option
    if (
        config.args_source == pytest.Config.ArgsSource.TESTPATHS
        and not option.keyword
        and not option.markexpr
        and not getattr(option, "deselect", None)
    ):
        config._quantpulse_whole_suite = len(session.items)  # type: ignore[attr-defined]
