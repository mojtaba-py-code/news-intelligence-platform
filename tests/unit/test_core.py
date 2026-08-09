"""Configuration, logging redaction, cache, rate limiting, resilience, metrics."""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.core.cache import InMemoryCache, NullCache, make_key
from app.core.config import PLACEHOLDER_SECRET, Environment, Settings
from app.core.errors import CircuitOpenError, NotFoundError, PlatformError, RateLimitError
from app.core.logging import (
    REDACTED,
    JSONFormatter,
    RedactionFilter,
    redact,
    redact_text,
)
from app.core.metrics import MetricsRegistry
from app.core.ratelimit import HostRateLimiter, RateLimiter
from app.core.resilience import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    RetryPolicy,
    retry_async,
)
from app.core.utils import (
    chunked,
    clamp,
    content_fingerprint,
    dedupe_preserving_order,
    ensure_utc,
    normalize_for_hash,
    percentage_change,
    safe_float,
    safe_int,
    truncate,
)

pytestmark = pytest.mark.unit


class TestSettings:
    def test_csv_lists_are_parsed(self) -> None:
        config = Settings(cors_origins="https://a.example, https://b.example")  # type: ignore[arg-type]
        assert config.cors_origins == ["https://a.example", "https://b.example"]

    def test_invalid_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            Settings(allowed_url_schemes="ftp")  # type: ignore[arg-type]

    def test_sync_database_driver_rejected(self) -> None:
        with pytest.raises(ValueError, match="async driver"):
            Settings(database_url="postgresql://user:pass@host/db")

    def test_production_refuses_placeholder_secret(self) -> None:
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            Settings(
                jwt_secret_key=PLACEHOLDER_SECRET,
                environment=Environment.PRODUCTION,
                database_url="postgresql+asyncpg://u:p@h/db",
                cors_origins=["https://app.example"],
                trusted_hosts=["app.example"],
                debug=False,
                rate_limit_enabled=True,
            )

    def test_production_refuses_wildcard_cors_and_sqlite(self) -> None:
        with pytest.raises(ValueError) as error:
            Settings(
                environment=Environment.PRODUCTION,
                jwt_secret_key="x" * 64,
                cors_origins=["*"],
                trusted_hosts=["*"],
                debug=True,
            )
        message = str(error.value)
        assert "CORS_ORIGINS" in message
        assert "TRUSTED_HOSTS" in message
        assert "SQLite" in message
        assert "DEBUG" in message

    def test_valid_production_config_is_accepted(self) -> None:
        config = Settings(
            environment=Environment.PRODUCTION,
            jwt_secret_key="s" * 64,
            database_url="postgresql+asyncpg://u:p@h/db",
            cors_origins=["https://app.example"],
            trusted_hosts=["app.example"],
            debug=False,
            rate_limit_enabled=True,
        )
        assert config.environment.is_production
        assert config.is_sqlite is False

    def test_sqlite_path_resolution(self) -> None:
        assert Settings(database_url="sqlite+aiosqlite:///:memory:").sqlite_path() is None
        path = Settings(database_url="sqlite+aiosqlite:///./data/x.db").sqlite_path()
        assert path is not None and path.name == "x.db"


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghij",
            "api_key=super-secret-value",
            "password: hunter2hunter2",
            "https://user:p4ssw0rd@example.com/feed",
            "token=abcdef123456",
        ],
    )
    def test_secrets_are_stripped_from_free_text(self, text: str) -> None:
        result = redact_text(text)
        for leak in ("super-secret-value", "hunter2hunter2", "p4ssw0rd", "abcdef123456"):
            assert leak not in result
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_sensitive_keys_are_redacted_recursively(self) -> None:
        payload = {
            "user": {"name": "jane", "password": "hunter2"},
            "headers": {"Authorization": "Bearer abc.def.ghi"},
            "items": [{"api_key": "k-123"}],
        }
        result = redact(payload)
        assert result["user"]["password"] == REDACTED
        assert result["headers"]["Authorization"] == REDACTED
        assert result["items"][0]["api_key"] == REDACTED
        assert result["user"]["name"] == "jane"

    def test_non_sensitive_values_survive(self) -> None:
        assert redact({"count": 5, "name": "ok"}) == {"count": 5, "name": "ok"}

    def test_deep_nesting_is_truncated(self) -> None:
        payload: dict[str, object] = {"a": "leaf"}
        for _ in range(12):
            payload = {"a": payload}
        assert "TRUNCATED" in str(redact(payload))

    def test_filter_applies_to_log_records(self) -> None:
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "login with password=hunter2hunter2", None, None
        )
        record.api_key = "k-secret"  # type: ignore[attr-defined]
        assert RedactionFilter().filter(record) is True
        assert "hunter2hunter2" not in record.getMessage()
        assert record.api_key == REDACTED  # type: ignore[attr-defined]

    def test_json_formatter_emits_one_object(self) -> None:
        import json

        record = logging.LogRecord("t", logging.INFO, __file__, 1, "message", None, None)
        record.source = "bbc"  # type: ignore[attr-defined]
        payload = json.loads(JSONFormatter().format(record))
        assert payload["message"] == "message"
        assert payload["level"] == "INFO"
        assert payload["source"] == "bbc"


class TestCache:
    async def test_set_get_delete(self) -> None:
        cache = InMemoryCache()
        await cache.set("k", {"a": 1})
        assert await cache.get("k") == {"a": 1}
        await cache.delete("k")
        assert await cache.get("k") is None

    async def test_ttl_expiry(self) -> None:
        cache = InMemoryCache()
        await cache.set("k", "v", ttl=1)
        assert await cache.get("k") == "v"
        # Fast-forward by rewriting the stored deadline rather than sleeping.
        cache._data["k"] = (0.0001, "v")
        assert await cache.get("k") is None

    async def test_incr_counts(self) -> None:
        cache = InMemoryCache()
        assert await cache.incr("c") == 1
        assert await cache.incr("c", 4) == 5

    async def test_namespace_clear(self) -> None:
        cache = InMemoryCache()
        await cache.set(make_key("ns", "a"), 1)
        await cache.set(make_key("other", "b"), 2)
        assert await cache.clear_namespace("ns") == 1
        assert await cache.get(make_key("other", "b")) == 2

    async def test_eviction_keeps_the_cache_bounded(self) -> None:
        cache = InMemoryCache(max_entries=10)
        for index in range(25):
            await cache.set(f"k{index}", index)
        assert len(cache._data) <= 10

    async def test_null_cache_is_inert(self) -> None:
        cache = NullCache()
        await cache.set("k", "v")
        assert await cache.get("k") is None

    def test_keys_are_namespaced_and_hashed(self) -> None:
        key = make_key("search", "'; DROP TABLE articles; --")
        assert key.startswith("nip:search:")
        assert "DROP TABLE" not in key


class TestRateLimiting:
    async def test_requests_are_allowed_until_the_limit(self) -> None:
        limiter = RateLimiter(3, 60, cache=InMemoryCache())
        decisions = [await limiter.check("client") for _ in range(4)]
        assert [decision.allowed for decision in decisions] == [True, True, True, False]
        assert decisions[-1].remaining == 0
        assert "Retry-After" in decisions[-1].headers()

    async def test_identities_are_independent(self) -> None:
        limiter = RateLimiter(1, 60, cache=InMemoryCache())
        assert (await limiter.check("a")).allowed
        assert (await limiter.check("b")).allowed

    async def test_reset_clears_the_window(self) -> None:
        limiter = RateLimiter(1, 60, cache=InMemoryCache())
        await limiter.check("a")
        await limiter.reset("a")
        assert (await limiter.check("a")).allowed

    def test_invalid_configuration_rejected(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(0, 60)

    async def test_host_limiter_spaces_requests(self) -> None:
        limiter = HostRateLimiter(delay=0.05, max_concurrency=2)
        loop = asyncio.get_running_loop()
        started = loop.time()
        async with limiter.slot("example.com"):
            pass
        async with limiter.slot("example.com"):
            pass
        assert loop.time() - started >= 0.04

    async def test_host_limiter_does_not_delay_other_hosts(self) -> None:
        limiter = HostRateLimiter(delay=0.2, max_concurrency=4)
        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.gather(
            *(self._touch(limiter, host) for host in ("a.example", "b.example", "c.example"))
        )
        assert loop.time() - started < 0.2

    @staticmethod
    async def _touch(limiter: HostRateLimiter, host: str) -> None:
        async with limiter.slot(host):
            return


class TestResilience:
    async def test_retry_succeeds_after_transient_failures(self) -> None:
        attempts = 0

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("boom")
            return "ok"

        result = await retry_async(
            flaky, policy=RetryPolicy(max_attempts=5, base_delay=0), sleep=_no_sleep
        )
        assert result == "ok"
        assert attempts == 3

    async def test_retry_gives_up_and_reraises(self) -> None:
        async def always_fails() -> None:
            raise TimeoutError("nope")

        with pytest.raises(TimeoutError):
            await retry_async(
                always_fails, policy=RetryPolicy(max_attempts=2, base_delay=0), sleep=_no_sleep
            )

    async def test_non_retryable_errors_surface_immediately(self) -> None:
        attempts = 0

        async def bad_request() -> None:
            nonlocal attempts
            attempts += 1
            raise NotFoundError("gone")

        with pytest.raises(NotFoundError):
            await retry_async(
                bad_request,
                policy=RetryPolicy(max_attempts=5, base_delay=0),
                give_up_on=(NotFoundError,),
                sleep=_no_sleep,
            )
        assert attempts == 1

    def test_backoff_grows_and_is_capped(self) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=5.0, jitter=False)
        assert policy.delay_for(1) == 1.0
        assert policy.delay_for(2) == 2.0
        assert policy.delay_for(10) == 5.0

    def test_jitter_stays_within_bounds(self) -> None:
        policy = RetryPolicy(base_delay=2.0, multiplier=2.0, jitter=True)
        assert all(0 <= policy.delay_for(2) <= 4.0 for _ in range(20))

    async def test_circuit_opens_after_repeated_failures(self) -> None:
        breaker = CircuitBreaker(name="src", failure_threshold=2, recovery_timeout=60)

        async def failing() -> None:
            raise RuntimeError("upstream down")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing)
        assert breaker.state is CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await breaker.call(failing)

    async def test_circuit_half_opens_then_closes(self) -> None:
        breaker = CircuitBreaker(
            name="src", failure_threshold=1, recovery_timeout=0.01, success_threshold=1
        )

        async def failing() -> None:
            raise RuntimeError("down")

        async def working() -> str:
            return "ok"

        with pytest.raises(RuntimeError):
            await breaker.call(failing)
        assert breaker.is_open

        await asyncio.sleep(0.02)
        assert await breaker.call(working) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_registry_isolates_sources(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout=60)
        await registry.get("a").record_failure()
        assert registry.get("a").is_open
        assert registry.get("b").is_open is False
        assert len(registry.status()) == 2
        await registry.reset_all()
        assert registry.get("a").is_open is False


class TestMetrics:
    def test_counter_gauge_histogram(self) -> None:
        registry = MetricsRegistry()
        counter = registry.counter("requests_total", "Requests")
        gauge = registry.gauge("queue", "Queue depth")
        histogram = registry.histogram("latency", "Latency")

        counter.inc(labels={"route": "/a"})
        counter.inc(2, labels={"route": "/a"})
        gauge.set(7)
        gauge.dec(2)
        histogram.observe(0.03)
        histogram.observe(1.5)

        assert counter.value(labels={"route": "/a"}) == 3
        assert gauge.value() == 5
        assert histogram.count() == 2
        assert histogram.average() == pytest.approx(0.765, abs=0.01)

    def test_counters_cannot_decrease(self) -> None:
        registry = MetricsRegistry()
        with pytest.raises(ValueError):
            registry.counter("c").inc(-1)

    def test_prometheus_rendering(self) -> None:
        registry = MetricsRegistry()
        registry.counter("hits_total", "Hits").inc(2, labels={"source": "bbc"})
        registry.histogram("dur_seconds", "Duration").observe(0.2)
        output = registry.render()
        assert "# TYPE hits_total counter" in output
        assert 'hits_total{source="bbc"} 2.0' in output
        assert "dur_seconds_bucket" in output
        assert "dur_seconds_count" in output

    def test_label_values_are_escaped(self) -> None:
        registry = MetricsRegistry()
        registry.counter("c").inc(labels={"route": 'a"b'})
        assert '\\"' in registry.render()

    def test_reset_clears_samples(self) -> None:
        registry = MetricsRegistry()
        registry.counter("c").inc()
        registry.reset()
        assert registry.counter("c").value() == 0

    def test_duplicate_name_with_other_type_is_rejected(self) -> None:
        registry = MetricsRegistry()
        registry.counter("x")
        with pytest.raises(ValueError):
            registry.gauge("x")


class TestErrors:
    def test_error_envelope(self) -> None:
        error = NotFoundError("Article 5 does not exist.", details={"id": 5})
        payload = error.to_dict("req-1")
        assert payload["error"]["code"] == "not_found"
        assert payload["error"]["details"] == {"id": 5}
        assert payload["error"]["request_id"] == "req-1"

    def test_status_codes(self) -> None:
        assert NotFoundError().status_code == 404
        assert RateLimitError().status_code == 429
        assert PlatformError().status_code == 500

    def test_rate_limit_carries_retry_after(self) -> None:
        assert RateLimitError(retry_after=30).retry_after == 30


class TestUtils:
    def test_normalisation_for_hashing(self) -> None:
        assert normalize_for_hash("Hello,   World!") == "hello world"
        assert content_fingerprint("A B") == content_fingerprint("a  b!")

    def test_chunked(self) -> None:
        assert [list(chunk) for chunk in chunked([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]
        with pytest.raises(ValueError):
            list(chunked([1], 0))

    def test_truncate_on_word_boundary(self) -> None:
        assert truncate("hello world foo", 12).endswith("…")
        assert truncate("short", 20) == "short"
        assert truncate(None, 5) == ""

    def test_numeric_coercions(self) -> None:
        assert safe_float("1.5") == 1.5
        assert safe_float("x", 2.0) == 2.0
        assert safe_float(float("nan")) == 0.0
        assert safe_int("3") == 3
        assert safe_int(None, -1) == -1

    def test_clamp_and_dedupe(self) -> None:
        assert clamp(1.5) == 1.0
        assert clamp(-1.0, -1.0, 1.0) == -1.0
        assert dedupe_preserving_order(["b", "a", "b"]) == ["b", "a"]

    def test_percentage_change(self) -> None:
        assert percentage_change(100, 150) == 50.0
        assert percentage_change(0, 3) == 300.0
        assert percentage_change(0, 0) == 0.0

    def test_ensure_utc(self) -> None:
        from datetime import datetime

        assert ensure_utc(None) is None
        assert ensure_utc(datetime(2026, 1, 1)).tzinfo is not None  # type: ignore[union-attr]


async def _no_sleep(_seconds: float) -> None:
    """Skip real backoff delays in tests."""
    return None
