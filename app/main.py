"""FastAPI application factory.

Wires middleware, routers, exception handlers and the lifespan. Nothing here
contains business logic - it is composition only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.api.routes import api_router
from app.api.routes.health import router as health_router
from app.core.cache import build_cache, close_cache, set_cache
from app.core.config import Settings, get_settings
from app.core.errors import ErrorCode, PlatformError, RateLimitError
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.dashboard.router import STATIC_DIR
from app.dashboard.router import router as dashboard_router
from app.database.session import dispose_engine, init_engine
from app.ingestion.fetchers.http import close_fetcher

logger = get_logger(__name__)

DESCRIPTION = """
Multi-source news intelligence: ingestion, normalisation, deduplication,
NLP enrichment, trend and event detection, served over a versioned REST API.

**Authentication** - obtain a token from `POST /api/v1/auth/login`, then send
`Authorization: Bearer <token>`. Roles: `USER` < `ANALYST` < `ADMIN`.

**Errors** - every non-2xx response uses the envelope
`{"error": {"code", "message", "details", "request_id"}}`.
"""

TAGS_METADATA: list[dict[str, Any]] = [
    {"name": "articles", "description": "Search, filter and read normalised articles."},
    {"name": "sources", "description": "Source registry, health and manual ingestion."},
    {"name": "intelligence", "description": "Topics, entities, events and trends."},
    {"name": "analytics", "description": "Aggregates powering the dashboard."},
    {"name": "authentication", "description": "Registration, login, token rotation."},
    {"name": "users", "description": "Profile, preferences and saved articles."},
    {"name": "alerts", "description": "Structured alert rules and notifications."},
    {"name": "admin", "description": "User administration, jobs and the audit log."},
    {"name": "operations", "description": "Health, readiness and metrics."},
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Start and stop shared resources exactly once per process."""
    config: Settings = get_settings()
    configure_logging(config)
    init_engine(config)
    set_cache(build_cache(config))

    logger.info(
        "application_started",
        extra={
            "version": __version__,
            "environment": str(config.environment),
            "database": config.database_url.split("://", 1)[0],
            "cache": "redis" if config.redis_url else "memory",
        },
    )
    try:
        yield
    finally:
        await close_fetcher()
        await close_cache()
        await dispose_engine()
        logger.info("application_stopped")


def create_app(config: Settings | None = None) -> FastAPI:
    """Build the application. Called by uvicorn and by the test suite."""
    config = config or get_settings()
    configure_logging(config)

    application = FastAPI(
        title=config.app_name,
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        # Interactive docs are disabled in production: they are an information
        # disclosure surface that production clients do not need.
        docs_url=None if config.environment.is_production else "/docs",
        redoc_url=None if config.environment.is_production else "/redoc",
        openapi_url=None if config.environment.is_production else "/openapi.json",
        contact={"name": "News Intelligence Platform"},
        license_info={"name": "MIT"},
    )

    _register_middleware(application, config)
    _register_exception_handlers(application)

    application.include_router(api_router, prefix=config.api_v1_prefix)
    # Probes are also exposed unprefixed: orchestrators (Kubernetes, Compose
    # healthchecks) expect /health, /ready and /metrics at the root.
    application.include_router(health_router, include_in_schema=False)
    application.include_router(dashboard_router)
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return application


def _register_middleware(application: FastAPI, config: Settings) -> None:
    """Install middleware.

    Starlette executes these in reverse registration order, so the request-id
    middleware is added last and therefore runs first.
    """
    application.add_middleware(RateLimitMiddleware, config=config)
    application.add_middleware(BodySizeLimitMiddleware, max_bytes=config.max_request_bytes)
    application.add_middleware(SecurityHeadersMiddleware, config=config)

    if config.trusted_hosts and "*" not in config.trusted_hosts:
        application.add_middleware(TrustedHostMiddleware, allowed_hosts=config.trusted_hosts)

    if config.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "Retry-After"],
            max_age=600,
        )

    application.add_middleware(RequestContextMiddleware)


def _register_exception_handlers(application: FastAPI) -> None:
    """Map exceptions onto the uniform error envelope."""

    @application.exception_handler(PlatformError)
    async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after)
        if exc.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        logger.info(
            "platform_error",
            extra={"code": exc.code, "status": exc.status_code, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(request_id_ctx.get()),
            headers=headers or None,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Report *where* validation failed without echoing the submitted values.

        Echoing input back is how validation errors leak passwords into logs
        and error trackers, so only the field location and message are kept.
        """
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
                "message": str(error.get("msg", ""))[:200],
                "type": str(error.get("type", ""))[:64],
            }
            for error in exc.errors()[:20]
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR,
                    "message": "The submitted data failed validation.",
                    "details": {"errors": details},
                    "request_id": request_id_ctx.get(),
                }
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: ErrorCode.UNAUTHENTICATED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            413: ErrorCode.PAYLOAD_TOO_LARGE,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail) if exc.detail else "Request failed.",
                    "details": {},
                    "request_id": request_id_ctx.get(),
                }
            },
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Last resort: log the detail, return a generic body.

        The exception message may contain a query, a file path or a connection
        string, so the client only ever sees the correlation id.
        """
        request_id = request_id_ctx.get()
        logger.exception(
            "unhandled_exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": exc.__class__.__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": "An internal error occurred.",
                    "details": {},
                    "request_id": request_id,
                }
            },
        )


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
