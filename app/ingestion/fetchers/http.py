"""The secure HTTP fetcher used by every connector.

Security controls, all enforced here rather than in individual connectors:

* **SSRF guard on every hop.** The initial URL *and* each redirect target is
  re-validated against :mod:`app.core.url_safety`; redirects are followed
  manually precisely so that no hop escapes the check.
* **Response size cap.** Bodies are streamed and aborted once the configured
  limit is exceeded, so a malicious endpoint cannot exhaust memory.
* **Content-type allowlist.** A connector that asked for a feed will not be
  handed a 40 MB binary.
* **No credential leakage across hosts.** Authorization headers are dropped
  when a redirect crosses an origin.
* **Timeouts everywhere**, connection pooling, per-host politeness delays,
  retries with jittered exponential backoff, and a circuit breaker per source.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Self
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import (
    FetchError,
    PermanentFetchError,
    RateLimitedUpstreamError,
    UnsafeURLError,
)
from app.core.logging import get_logger
from app.core.metrics import source_errors_total, source_fetch_duration_seconds
from app.core.ratelimit import HostRateLimiter
from app.core.resilience import RetryPolicy, breakers, retry_async
from app.core.url_safety import validate_url
from app.ingestion.fetchers.robots import RobotsCache

logger = get_logger(__name__)

#: Response content types the platform knows how to consume.
ALLOWED_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "text/html",
    "text/plain",
    "text/xml",
    "application/xml",
    "application/rss+xml",
    "application/atom+xml",
    "application/xhtml+xml",
    "application/json",
    "application/feed+json",
    "application/ld+json",
)

#: Status codes worth retrying: transient server and infrastructure failures.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

_CHUNK_SIZE: Final[int] = 64 * 1024


@dataclass(slots=True)
class FetchResponse:
    """A validated, size-bounded HTTP response."""

    url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    content: bytes
    elapsed_ms: float
    from_cache: bool = False

    @property
    def text(self) -> str:
        """Decoded body, honouring the charset with a permissive fallback."""
        encoding = _charset_of(self.headers.get("content-type", ""))
        for candidate in (encoding, "utf-8", "cp1252"):
            if not candidate:
                continue
            try:
                return self.content.decode(candidate)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        import json

        return json.loads(self.text)

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


def _charset_of(content_type: str) -> str | None:
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip("\"'") or None
    return None


@dataclass
class SecureHTTPFetcher:
    """Shared async HTTP client with the platform's safety policy applied."""

    config: Settings = field(default_factory=get_settings)
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _limiter: HostRateLimiter | None = field(default=None, init=False, repr=False)
    _robots: RobotsCache | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = HostRateLimiter(
            delay=self.config.default_request_delay_seconds,
            max_concurrency=self.config.http_max_concurrency,
        )
        self._robots = RobotsCache(user_agent=self.config.http_user_agent)

    # ----------------------------------------------------------- lifecycle
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.config.http_timeout_seconds,
                    connect=min(10.0, self.config.http_timeout_seconds),
                ),
                # Redirects are followed by hand so every hop is re-validated.
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=self.config.http_max_concurrency * 2,
                    max_keepalive_connections=self.config.http_max_concurrency,
                    keepalive_expiry=30.0,
                ),
                headers={
                    "User-Agent": self.config.http_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept-Language": "en;q=0.9,*;q=0.5",
                },
                trust_env=False,  # ignore ambient proxy env vars
            )
        return self._client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    @property
    def robots(self) -> RobotsCache:
        assert self._robots is not None
        return self._robots

    @property
    def limiter(self) -> HostRateLimiter:
        assert self._limiter is not None
        return self._limiter

    # -------------------------------------------------------------- fetching
    async def fetch(
        self,
        url: str,
        *,
        source: str = "unknown",
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        accept: str | None = None,
        respect_robots: bool | None = None,
        delay: float | None = None,
        retries: int | None = None,
    ) -> FetchResponse:
        """Fetch ``url`` under the full safety policy.

        Raises :class:`FetchError` (or a subclass) on failure; the caller is
        expected to isolate that to the one source.
        """
        validation = validate_url(url, config=self.config)
        respect = self.config.respect_robots_txt if respect_robots is None else respect_robots
        if respect and not await self._robots_allow(url, source=source):
            raise FetchError(source, "Blocked by robots.txt", details={"url": url[:200]})

        request_headers = dict(headers or {})
        if accept:
            request_headers["Accept"] = accept

        policy = RetryPolicy(
            max_attempts=(retries if retries is not None else self.config.http_max_retries) + 1,
            base_delay=self.config.http_backoff_base,
        )
        breaker = breakers.get(source)
        crawl_delay = self.robots.crawl_delay(url)
        effective_delay = max(delay or 0.0, crawl_delay or 0.0) or None

        async def attempt() -> FetchResponse:
            async with self.limiter.slot(validation.host, delay=effective_delay):
                return await self._request(
                    url,
                    method=method,
                    params=params,
                    headers=request_headers,
                    source=source,
                )

        async def guarded() -> FetchResponse:
            return await breaker.call(attempt)

        try:
            with source_fetch_duration_seconds.time(labels={"source": source}):
                return await retry_async(
                    guarded,
                    policy=policy,
                    retry_on=(FetchError, httpx.TransportError, asyncio.TimeoutError),
                    give_up_on=(UnsafeURLError, PermanentFetchError),
                )
        except UnsafeURLError:
            source_errors_total.inc(labels={"source": source, "kind": "unsafe_url"})
            raise
        except httpx.TransportError as exc:
            source_errors_total.inc(labels={"source": source, "kind": "transport"})
            raise FetchError(source, f"Transport error: {exc.__class__.__name__}") from exc

    async def _request(
        self,
        url: str,
        *,
        method: str,
        params: dict[str, Any] | None,
        headers: dict[str, str],
        source: str,
    ) -> FetchResponse:
        """Single request plus manual, re-validated redirect following."""
        import time

        started = time.perf_counter()
        current_url = url
        current_headers = dict(headers)
        origin = _origin(url)

        for hop in range(self.config.http_max_redirects + 1):
            validate_url(current_url, config=self.config)

            request = self.client.build_request(
                method,
                current_url,
                params=params if hop == 0 else None,
                headers=current_headers,
            )
            response = await self.client.send(request, stream=True)

            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(source, "Redirect without a Location header")
                    next_url = urljoin(current_url, location)
                    if _origin(next_url) != origin:
                        # Never forward credentials to a different origin.
                        current_headers.pop("Authorization", None)
                        current_headers.pop("authorization", None)
                        current_headers.pop("X-Api-Key", None)
                        origin = _origin(next_url)
                    current_url = next_url
                    continue

                self._check_status(response, source=source)
                self._check_content_type(response, source=source)
                content = await self._read_bounded(response, source=source)
            finally:
                await response.aclose()

            return FetchResponse(
                url=url,
                final_url=str(response.url),
                status_code=response.status_code,
                headers={key.lower(): value for key, value in response.headers.items()},
                content=content,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        raise FetchError(
            source, "Too many redirects", details={"max": self.config.http_max_redirects}
        )

    # ------------------------------------------------------------- guards
    def _check_status(self, response: httpx.Response, *, source: str) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        source_errors_total.inc(labels={"source": source, "kind": f"http_{status // 100}xx"})
        if status == 429:
            retry_after = _retry_after(response.headers.get("retry-after"))
            raise RateLimitedUpstreamError(source, retry_after=retry_after)
        if status in RETRYABLE_STATUS:
            raise FetchError(source, f"Upstream returned {status}", details={"status": status})
        # 4xx other than 429: retrying will not help.
        raise PermanentFetchError(source, f"Upstream returned {status}", details={"status": status})

    @staticmethod
    def _check_content_type(response: httpx.Response, *, source: str) -> None:
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type:
            return  # some feeds omit it entirely; the parser will decide
        if not any(content_type.startswith(allowed) for allowed in ALLOWED_CONTENT_TYPES):
            raise FetchError(
                source,
                f"Unsupported content type '{content_type}'",
                details={"content_type": content_type},
            )

    async def _read_bounded(self, response: httpx.Response, *, source: str) -> bytes:
        """Stream the body, aborting once the configured cap is exceeded."""
        limit = self.config.http_max_response_bytes
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise FetchError(
                source,
                "Response exceeds the maximum allowed size",
                details={"content_length": int(declared), "limit": limit},
            )

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes(_CHUNK_SIZE):
            total += len(chunk)
            if total > limit:
                raise FetchError(
                    source,
                    "Response exceeds the maximum allowed size",
                    details={"limit": limit},
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def _robots_allow(self, url: str, *, source: str) -> bool:
        """Consult (and populate) the robots.txt cache for ``url``'s origin."""
        cached = self.robots.can_fetch(url)
        if cached is not None:
            return cached

        robots_url = self.robots.robots_url(url)
        try:
            validate_url(robots_url, config=self.config)
            response = await self._request(
                robots_url, method="GET", params=None, headers={}, source=source
            )
            self.robots.store(url, response.text)
        except FetchError as exc:
            status = exc.details.get("status")
            # An explicit refusal on robots.txt means "do not crawl"; anything
            # else (404, 5xx, timeout) means "no policy published".
            self.robots.store(url, None, available=status in (401, 403))
            return status not in (401, 403)
        except (UnsafeURLError, httpx.HTTPError):
            self.robots.store(url, None, available=False)
            return True

        result = self.robots.can_fetch(url)
        return True if result is None else result


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _retry_after(value: str | None) -> float:
    if not value:
        return 60.0
    try:
        return max(1.0, min(float(value), 3600.0))
    except ValueError:
        return 60.0


_fetcher: SecureHTTPFetcher | None = None


def get_fetcher(config: Settings | None = None) -> SecureHTTPFetcher:
    """Process-wide fetcher (connection pooling is the whole point)."""
    global _fetcher
    if _fetcher is None:
        _fetcher = SecureHTTPFetcher(config=config or get_settings())
    return _fetcher


def set_fetcher(fetcher: SecureHTTPFetcher | None) -> None:
    """Override the singleton (tests, and the app lifespan)."""
    global _fetcher
    _fetcher = fetcher


async def close_fetcher() -> None:
    """Release pooled connections during shutdown."""
    global _fetcher
    if _fetcher is not None:
        await _fetcher.aclose()
    _fetcher = None


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "RETRYABLE_STATUS",
    "FetchResponse",
    "SecureHTTPFetcher",
    "close_fetcher",
    "get_fetcher",
    "set_fetcher",
]
