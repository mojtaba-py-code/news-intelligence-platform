"""Cache abstraction with a Redis backend and an in-memory fallback.

The platform must run in three situations: a full docker-compose stack (Redis
available), a laptop with nothing installed, and a test suite. A tiny
:class:`CacheBackend` protocol keeps the call sites identical in all three.

Only *derived, non-sensitive* data is cached. Cache keys are namespaced and
hashed so that user-supplied search strings can never break out of the
key-space or leak into logs verbatim.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Final

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

KEY_PREFIX: Final[str] = "nip"
MAX_VALUE_BYTES: Final[int] = 1_048_576  # never cache multi-MiB blobs


def make_key(namespace: str, *parts: Any) -> str:
    """Build a collision-resistant, injection-proof cache key."""
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:32]
    safe_ns = "".join(ch for ch in namespace if ch.isalnum() or ch in "._-")[:48]
    return f"{KEY_PREFIX}:{safe_ns}:{digest}"


class CacheBackend(ABC):
    """Minimal async cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int: ...

    @abstractmethod
    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def ping(self) -> bool:  # pragma: no cover - overridden where meaningful
        return True


class InMemoryCache(CacheBackend):
    """Process-local TTL cache. Correct for a single process, not shared."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at and expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        async with self._lock:
            if len(self._data) >= self._max_entries:
                self._evict_locked()
            expires_at = time.monotonic() + ttl if ttl else 0.0
            self._data[key] = (expires_at, value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def clear_namespace(self, namespace: str) -> int:
        prefix = f"{KEY_PREFIX}:{namespace}"
        async with self._lock:
            keys = [key for key in self._data if key.startswith(prefix)]
            for key in keys:
                del self._data[key]
            return len(keys)

    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        async with self._lock:
            entry = self._data.get(key)
            now = time.monotonic()
            current = 0
            if entry and (not entry[0] or entry[0] >= now):
                current = int(entry[1])
            new_value = current + amount
            expires_at = entry[0] if entry and entry[0] else (now + ttl if ttl else 0.0)
            self._data[key] = (expires_at, new_value)
            return new_value

    async def close(self) -> None:
        async with self._lock:
            self._data.clear()

    def _evict_locked(self) -> None:
        """Drop expired entries first, then the oldest 10% (insertion order)."""
        now = time.monotonic()
        expired = [key for key, (exp, _) in self._data.items() if exp and exp < now]
        for key in expired:
            del self._data[key]
        if len(self._data) >= self._max_entries:
            for key in list(self._data)[: max(1, self._max_entries // 10)]:
                del self._data[key]


class RedisCache(CacheBackend):
    """Redis-backed cache. Values are JSON so they stay language-agnostic."""

    def __init__(self, url: str, *, default_ttl: int = 60) -> None:
        import redis.asyncio as redis  # imported lazily: optional dependency

        self._client = redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            health_check_interval=30,
        )
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.warning("cache_get_failed", extra={"error": str(exc)})
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError):
            logger.warning("cache_set_unserialisable", extra={"key": key})
            return
        if len(payload) > MAX_VALUE_BYTES:
            return
        try:
            await self._client.set(key, payload, ex=ttl or self._default_ttl)
        except Exception as exc:
            logger.warning("cache_set_failed", extra={"error": str(exc)})

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except Exception as exc:
            logger.warning("cache_delete_failed", extra={"error": str(exc)})

    async def clear_namespace(self, namespace: str) -> int:
        pattern = f"{KEY_PREFIX}:{namespace}*"
        removed = 0
        try:
            async for key in self._client.scan_iter(match=pattern, count=500):
                await self._client.delete(key)
                removed += 1
        except Exception as exc:
            logger.warning("cache_clear_failed", extra={"error": str(exc)})
        return removed

    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        try:
            pipe = self._client.pipeline()
            pipe.incrby(key, amount)
            if ttl:
                pipe.expire(key, ttl, nx=True)
            results = await pipe.execute()
            return int(results[0])
        except Exception as exc:
            logger.warning("cache_incr_failed", extra={"error": str(exc)})
            return amount

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:
            # Shutdown must never raise; record it and move on.
            logger.debug("cache_close_failed", extra={"error": str(exc)})


class NullCache(CacheBackend):
    """No-op backend used when caching is disabled."""

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def clear_namespace(self, namespace: str) -> int:
        return 0

    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        return amount

    async def close(self) -> None:
        return None


_cache: CacheBackend | None = None


def build_cache(config: Settings | None = None) -> CacheBackend:
    """Construct the backend implied by configuration."""
    config = config or get_settings()
    if not config.cache_enabled:
        return NullCache()
    if config.redis_url:
        try:
            return RedisCache(config.redis_url, default_ttl=config.cache_ttl_seconds)
        except Exception as exc:
            logger.warning("redis_unavailable_using_memory", extra={"error": str(exc)})
    return InMemoryCache()


def get_cache() -> CacheBackend:
    """Return the process-wide cache backend."""
    global _cache
    if _cache is None:
        _cache = build_cache()
    return _cache


def set_cache(backend: CacheBackend | None) -> None:
    """Override the singleton (used by the app lifespan and by tests)."""
    global _cache
    _cache = backend


async def close_cache() -> None:
    """Release cache resources during shutdown."""
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None


__all__ = [
    "CacheBackend",
    "InMemoryCache",
    "NullCache",
    "RedisCache",
    "build_cache",
    "close_cache",
    "get_cache",
    "make_key",
    "set_cache",
]
