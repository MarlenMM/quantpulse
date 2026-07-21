import pytest

from quantpulse.ingestion.circuit_breaker import reset_all_breakers


@pytest.fixture(autouse=True)
def _reset_circuit_breakers() -> None:
    """Keep the module-global circuit-breaker registry from leaking across tests."""
    reset_all_breakers()
    yield
    reset_all_breakers()
