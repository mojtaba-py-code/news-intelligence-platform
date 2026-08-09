"""ASGI middleware: correlation ids, security headers, body limits, rate limiting.

Ordering matters. Starlette runs middleware in reverse registration order, so
:func:`app.main.create_app` adds them such that the request id is established
first (everything downstream can log it) and the body limit is checked before
any handler reads the stream.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings, get_settings
from app.core.errors import ErrorCode
from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import (
    http_errors_total,
    http_request_duration_seconds,
    http_requests_total,
    rate_limited_total,
)
from app.core.ratelimit import RateLimiter

logger = get_logger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]

#: Endpoints exempt from the global rate limit (probes and metrics scraping).
RATE_LIMIT_EXEMPT: frozenset[str] = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/circuits",
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/metrics",
    }
)

#: Paths whose *failed* requests deserve a stricter budget.
AUTH_PATH_PREFIXES: tuple[str, ...] = ("/api/v1/auth/login", "/api/v1/auth/register")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, records latency, and emits one structured log line."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        incoming = request.headers.get("x-request-id", "")
        # Never trust a client-supplied id verbatim - it lands in logs.
        request_id = incoming if _is_safe_request_id(incoming) else uuid.uuid4().hex[:16]
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        route = _route_label(request)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            http_errors_total.inc(labels={"route": route, "status": "500"})
            http_request_duration_seconds.observe(duration, labels={"route": route})
            logger.exception(
                "request_failed",
                extra={"method": request.method, "route": route, "duration_ms": duration * 1000},
            )
            raise
        finally:
            request_id_ctx.reset(token)

        duration = time.perf_counter() - started
        http_requests_total.inc(labels={"route": route, "method": request.method})
        http_request_duration_seconds.observe(duration, labels={"route": route})
        if response.status_code >= 400:
            http_errors_total.inc(labels={"route": route, "status": str(response.status_code)})

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "route": route,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "client": _client_ip(request),
            },
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds the standard hardening headers to every response.

    The CSP is strict because the dashboard ships no third-party assets: no
    external origins are allowed, and inline scripts are not used.
    """

    def __init__(self, app: object, *, config: Settings | None = None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.config = config or get_settings()

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        response = await call_next(request)
        if not self.config.secure_headers_enabled:
            return response

        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=(), payment=()"
        )
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
        headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'self'; object-src 'none'",
        )
        if self.config.hsts_enabled:
            headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        # API responses must never be cached by a shared proxy.
        if request.url.path.startswith("/api/"):
            headers.setdefault("Cache-Control", "no-store")
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized request bodies before a handler can buffer them."""

    def __init__(self, app: object, *, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": ErrorCode.PAYLOAD_TOO_LARGE,
                        "message": "Request body is too large.",
                        "details": {"limit_bytes": self.max_bytes},
                    }
                },
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiting keyed by authenticated user or client IP."""

    def __init__(self, app: object, *, config: Settings | None = None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.config = config or get_settings()
        self._general = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window_seconds,
            namespace="rl.general",
        )
        self._auth = RateLimiter(
            self.config.auth_rate_limit_requests,
            self.config.auth_rate_limit_window_seconds,
            namespace="rl.auth",
        )

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        if not self.config.rate_limit_enabled or request.url.path in RATE_LIMIT_EXEMPT:
            return await call_next(request)

        path = request.url.path
        limiter = self._auth if path.startswith(AUTH_PATH_PREFIXES) else self._general
        identity = _identity(request)

        decision = await limiter.check(identity)
        if not decision.allowed:
            rate_limited_total.inc(labels={"route": _route_label(request)})
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": ErrorCode.RATE_LIMITED,
                        "message": "Too many requests. Please slow down.",
                        "details": {"retry_after_seconds": decision.reset_after},
                    }
                },
                headers=decision.headers(),
            )

        response = await call_next(request)
        for key, value in decision.headers().items():
            response.headers.setdefault(key, value)
        return response


def _identity(request: Request) -> str:
    """Rate-limit key: the bearer token's fingerprint, else the client IP.

    The raw token is never used as a key - only a short hash of it, so the
    cache never stores a credential.
    """
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        import hashlib

        token = authorization[7:].strip()
        if token:
            return "u:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return "ip:" + (_client_ip(request) or "unknown")


def _client_ip(request: Request) -> str | None:
    """Client address.

    ``X-Forwarded-For`` is only honoured when the deployment sits behind a
    trusted proxy; otherwise a client could spoof its identity and bypass
    per-IP limits.
    """
    config = get_settings()
    if config.trusted_hosts and "*" not in config.trusted_hosts:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else None


def _route_label(request: Request) -> str:
    """Templated path (``/articles/{id}``) so metrics do not explode per id."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    return request.url.path[:120]


def _is_safe_request_id(value: str) -> bool:
    return bool(value) and len(value) <= 64 and all(ch.isalnum() or ch in "-_" for ch in value)


__all__ = [
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "SecurityHeadersMiddleware",
]
