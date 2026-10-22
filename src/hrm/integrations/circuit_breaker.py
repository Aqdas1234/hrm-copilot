"""Circuit breaker: stop calling a failing service for a while, then try again carefully.

CLOSED  : calls go through; consecutive failures are counted.
OPEN    : after `failure_threshold` failures in a row, calls fail at once for `reset_after_s`
          seconds, so users get a fast "temporarily unavailable" instead of slow timeouts,
          and the struggling service gets time to recover.
HALF_OPEN: after that pause, one trial call is let through. Success closes the circuit;
          failure opens it again.
"""

import enum
import time
from collections.abc import Callable
from typing import TypeVar

from hrm.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class State(enum.StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """The circuit is open: the call was not attempted."""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        reset_after_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._reset_after = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return State.CLOSED
        if self._clock() - self._opened_at >= self._reset_after:
            return State.HALF_OPEN
        return State.OPEN

    def call(
        self, fn: Callable[[], T], *, is_failure: Callable[[Exception], bool] = lambda e: True
    ) -> T:
        """Run fn through the breaker. Exceptions for which is_failure() is False (e.g.
        "not found") pass through without counting against the service."""
        if self.state is State.OPEN:
            raise CircuitOpenError(f"{self.name} is temporarily unavailable.")
        try:
            result = fn()
        except Exception as exc:
            if is_failure(exc):
                self._record_failure()
            raise
        self._record_success()
        return result

    def _record_failure(self) -> None:
        self._failures += 1
        if self.state is State.HALF_OPEN or self._failures >= self._threshold:
            self._opened_at = self._clock()
            log.warning("circuit.opened", name=self.name, failures=self._failures)

    def _record_success(self) -> None:
        if self._opened_at is not None:
            log.info("circuit.closed", name=self.name)
        self._failures = 0
        self._opened_at = None
