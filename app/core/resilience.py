"""Fault-tolerance primitives: retry with jittered backoff, and a circuit breaker.

A single failing source must never take the pipeline down. Two independent
mechanisms cooperate:

* :func:`retry_async` absorbs *transient* failures (timeouts, 5xx, connection
  resets) with exponential backoff plus full jitter to avoid thundering herds.
* :class:`CircuitBreaker` absorbs *persistent* failures: after N consecutive
  errors the circuit opens and calls fail fast for a cool-down period, then a
  single probe decides whether to close it again.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from app.core.errors import CircuitOpenError
from app.core.logging import get_logger
from app.core.metrics import circuit_state

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Declarative retry configuration."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Backoff for a 1-based attempt number, with full jitter."""
        raw = min(self.base_delay * (self.multiplier ** max(0, attempt - 1)), self.max_delay)
        if not self.jitter:
            return raw
        return random.uniform(0, raw)  # noqa: S311 - jitter, not cryptography


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``func`` until it succeeds or the policy is exhausted.

    ``give_up_on`` wins over ``retry_on`` so that non-retryable errors (4xx,
    validation failures) surface immediately.
    """
    policy = policy or RetryPolicy()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func()
        except give_up_on:
            raise
        except retry_on as exc:
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            logger.warning(
                "retrying_after_error",
                extra={
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "delay_seconds": round(delay, 3),
                    "error_type": exc.__class__.__name__,
                },
            )
            await sleep(delay)

    assert last_error is not None
    raise last_error


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_STATE_VALUE = {CircuitState.CLOSED: 0.0, CircuitState.HALF_OPEN: 1.0, CircuitState.OPEN: 2.0}


@dataclass
class CircuitBreaker:
    """Per-source circuit breaker.

    ``failure_threshold`` consecutive failures open the circuit for
    ``recovery_timeout`` seconds. The next call is then admitted as a probe
    (half-open); ``success_threshold`` consecutive probe successes close it.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is CircuitState.OPEN

    def _set_state(self, state: CircuitState) -> None:
        if state is not self._state:
            logger.info(
                "circuit_state_changed",
                extra={"circuit": self.name, "from": str(self._state), "to": str(state)},
            )
        self._state = state
        circuit_state.set(_STATE_VALUE[state], labels={"source": self.name})

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._set_state(CircuitState.HALF_OPEN)
                    self._successes = 0
                else:
                    remaining = self.recovery_timeout - (time.monotonic() - self._opened_at)
                    raise CircuitOpenError(
                        self.name,
                        f"Circuit '{self.name}' is open; retry in {remaining:.0f}s.",
                        details={"retry_after_seconds": round(remaining, 1)},
                    )

    async def record_success(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self.success_threshold:
                    self._set_state(CircuitState.CLOSED)
                    self._failures = 0
            else:
                self._failures = 0
                self._set_state(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                self._set_state(CircuitState.OPEN)

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Run ``func`` under the breaker."""
        await self._before_call()
        try:
            result = await func()
        except Exception:
            await self.record_failure()
            raise
        await self.record_success()
        return result

    async def reset(self) -> None:
        async with self._lock:
            self._failures = 0
            self._successes = 0
            self._set_state(CircuitState.CLOSED)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self._state),
            "consecutive_failures": self._failures,
            "opened_seconds_ago": (
                round(time.monotonic() - self._opened_at, 1) if self._opened_at else None
            ),
        }


class CircuitBreakerRegistry:
    """Keeps one breaker per source name."""

    def __init__(self, *, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def get(self, name: str) -> CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
            self._breakers[name] = breaker
        return breaker

    def status(self) -> list[dict[str, Any]]:
        return [
            breaker.status() for breaker in sorted(self._breakers.values(), key=lambda b: b.name)
        ]

    async def reset_all(self) -> None:
        for breaker in list(self._breakers.values()):
            await breaker.reset()

    def clear(self) -> None:
        self._breakers.clear()


breakers = CircuitBreakerRegistry()

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "RetryPolicy",
    "breakers",
    "retry_async",
]
