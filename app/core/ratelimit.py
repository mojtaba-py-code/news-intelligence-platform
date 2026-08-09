"""Token-bucket rate limiting, shared by the HTTP API and the fetchers.

Two distinct needs:

* **Inbound** - protect the API from abuse (per client identity, per route
  class). Backed by the cache so it works across processes when Redis is
  configured, and degrades to per-process counters otherwise.
* **Outbound** - stay polite towards news sources (per-host request spacing and
  a concurrency ceiling), see :class:`HostRateLimiter`.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass

from app.core.cache import CacheBackend, get_cache, make_key
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of a limiter check."""

    allowed: bool
    limit: int
    remaining: int
    reset_after: int

    def headers(self) -> dict[str, str]:
        """Standard ``X-RateLimit-*`` response headers."""
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.reset_after),
        }
        if not self.allowed:
            out["Retry-After"] = str(self.reset_after)
        return out


class RateLimiter:
    """Fixed-window counter with a cache backend.

    A fixed window is chosen over a sliding log deliberately: it is O(1) in
    memory per identity, survives process restarts when Redis is present, and
    the worst-case burst (2x limit across a window boundary) is acceptable for
    an API guard.
    """

    def __init__(
        self,
        requests: int,
        window_seconds: int,
        *,
        namespace: str = "rl",
        cache: CacheBackend | None = None,
    ) -> None:
        if requests < 1 or window_seconds < 1:
            raise ValueError("requests and window_seconds must be >= 1")
        self.requests = requests
        self.window_seconds = window_seconds
        self.namespace = namespace
        self._cache = cache

    @property
    def cache(self) -> CacheBackend:
        return self._cache if self._cache is not None else get_cache()

    async def check(self, identity: str, *, cost: int = 1) -> RateLimitDecision:
        """Consume ``cost`` units for ``identity`` and report the decision."""
        now = int(time.time())
        window_start = now - (now % self.window_seconds)
        reset_after = window_start + self.window_seconds - now
        key = make_key(self.namespace, identity, window_start)

        used = await self.cache.incr(key, cost, ttl=self.window_seconds + 1)
        remaining = self.requests - used
        allowed = used <= self.requests
        if not allowed:
            logger.info(
                "rate_limit_exceeded",
                extra={"namespace": self.namespace, "limit": self.requests},
            )
        return RateLimitDecision(
            allowed=allowed,
            limit=self.requests,
            remaining=max(0, remaining),
            reset_after=max(1, reset_after),
        )

    async def reset(self, identity: str) -> None:
        """Forget the current window for ``identity`` (used after a successful login)."""
        now = int(time.time())
        window_start = now - (now % self.window_seconds)
        await self.cache.delete(make_key(self.namespace, identity, window_start))


class HostRateLimiter:
    """Per-host politeness for outbound fetching.

    Guarantees at least ``delay`` seconds between two requests to the same host
    and caps total in-flight requests. Instances are per-process, which matches
    how workers are deployed (one fetcher pool per worker process).
    """

    def __init__(self, delay: float = 1.0, max_concurrency: int = 8) -> None:
        self.delay = max(0.0, delay)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._last_call: dict[str, float] = defaultdict(float)
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _lock_for(self, host: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._host_locks.get(host)
            if lock is None:
                lock = asyncio.Lock()
                self._host_locks[host] = lock
            return lock

    async def acquire(self, host: str, *, delay: float | None = None) -> None:
        """Block until it is polite to call ``host`` again."""
        await self._semaphore.acquire()
        try:
            wait_for = self.delay if delay is None else max(0.0, delay)
            lock = await self._lock_for(host)
            async with lock:
                elapsed = time.monotonic() - self._last_call[host]
                if elapsed < wait_for:
                    await asyncio.sleep(wait_for - elapsed)
                self._last_call[host] = time.monotonic()
        except BaseException:
            self._semaphore.release()
            raise

    def release(self) -> None:
        """Release the concurrency slot taken by :meth:`acquire`."""
        self._semaphore.release()

    def slot(self, host: str, *, delay: float | None = None) -> _HostSlot:
        """``async with limiter.slot(host):`` - acquire and always release."""
        return _HostSlot(self, host, delay)


class _HostSlot:
    __slots__ = ("_delay", "_host", "_limiter")

    def __init__(self, limiter: HostRateLimiter, host: str, delay: float | None) -> None:
        self._limiter = limiter
        self._host = host
        self._delay = delay

    async def __aenter__(self) -> None:
        await self._limiter.acquire(self._host, delay=self._delay)

    async def __aexit__(self, *exc_info: object) -> None:
        self._limiter.release()


__all__ = ["HostRateLimiter", "RateLimitDecision", "RateLimiter"]
