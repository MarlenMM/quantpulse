import threading

import pytest

from quantpulse.ingestion.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    get_breaker,
    reset_all_breakers,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock: _FakeClock, **kw: object) -> CircuitBreaker:
    return CircuitBreaker("test", clock=clock, **kw)  # type: ignore[arg-type]


def test_stays_closed_below_the_failure_threshold() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "closed"
    breaker.before_call()  # does not raise


def test_opens_after_consecutive_failures_and_short_circuits() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    for _ in range(3):
        breaker.record_failure()

    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_a_success_resets_the_failure_count() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "closed"  # count restarted, threshold not reached


def test_transitions_to_half_open_after_the_reset_timeout() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1, reset_timeout_seconds=60)

    breaker.record_failure()
    assert breaker.state == "open"

    clock.now += 61
    assert breaker.state == "half_open"
    breaker.before_call()  # half-open allows a trial call through


def test_half_open_success_closes_and_failure_reopens() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1, reset_timeout_seconds=60)

    breaker.record_failure()
    clock.now += 61
    assert breaker.state == "half_open"
    breaker.record_success()
    assert breaker.state == "closed"

    breaker.record_failure()  # open again
    clock.now += 61
    assert breaker.state == "half_open"
    breaker.record_failure()  # trial failed -> straight back to open
    assert breaker.state == "open"


def test_guard_records_success_on_clean_exit() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1)
    with breaker.guard():
        pass
    assert breaker.state == "closed"


def test_guard_records_failure_and_reraises() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1)

    with pytest.raises(ValueError):
        with breaker.guard():
            raise ValueError("boom")

    assert breaker.state == "open"


def test_registry_returns_the_same_instance_per_name() -> None:
    reset_all_breakers()
    assert get_breaker("finnhub") is get_breaker("finnhub")
    assert get_breaker("finnhub") is not get_breaker("fred")


def test_reset_all_breakers_clears_registry_state() -> None:
    first = get_breaker("finnhub")
    reset_all_breakers()
    assert get_breaker("finnhub") is not first


# --------------------------------------------------------------------------- #
# Concurrency (Section 21: "one instance is shared across the nightly job's
# thread pool" -- these exercise that claim with real threads, not just the
# single-threaded state-machine tests above).
# --------------------------------------------------------------------------- #


def test_concurrent_failures_are_never_lost() -> None:
    # `failure_threshold` set to exactly the thread count: the breaker can
    # only end up "open" if `_failure_count` reached that number exactly, so
    # this is a public, deterministic proxy for "no increment was lost" --
    # without `record_failure`'s lock, concurrent `+= 1`s can race and drop
    # updates, leaving the breaker (wrongly) closed.
    clock = _FakeClock()
    n_threads = 50
    breaker = _breaker(clock, failure_threshold=n_threads)
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()  # release every thread at once to maximize contention
        breaker.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker.state == "open"


def test_concurrent_calls_never_see_a_torn_state() -> None:
    # Half the threads race a success, half a failure, against a
    # freshly-opened breaker. `_state`/`_failure_count`/`_opened_at` are all
    # only ever mutated together under the same lock, so no interleaving of
    # `record_success`/`record_failure` should be able to leave the breaker
    # in a state `before_call` can't classify as cleanly open or closed --
    # this just shouldn't raise anything other than the two expected outcomes.
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1)
    n_threads = 40
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(succeed: bool) -> None:
        barrier.wait()
        try:
            breaker.record_success() if succeed else breaker.record_failure()
            state = breaker.state  # must not raise / must return a valid literal
            assert state in ("open", "closed", "half_open")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert breaker.state in ("open", "closed")
