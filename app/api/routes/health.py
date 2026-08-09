"""Liveness, readiness and metrics endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Response, status

from app import __version__
from app.core.cache import get_cache
from app.core.config import get_settings
from app.core.metrics import registry
from app.core.resilience import breakers
from app.database.session import check_connection
from app.schemas.common import ComponentHealth, HealthResponse, HealthStatus

router = APIRouter(tags=["operations"])

_STARTED_AT = time.monotonic()


def uptime_seconds() -> float:
    return round(time.monotonic() - _STARTED_AT, 2)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Cheap check that the process is alive. Never touches the database."""
    config = get_settings()
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        version=__version__,
        environment=str(config.environment),
        uptime_seconds=uptime_seconds(),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={503: {"description": "A required dependency is unavailable"}},
)
async def ready(response: Response) -> HealthResponse:
    """Verify the dependencies needed to serve traffic.

    The database is required; the cache is optional, so an unreachable Redis
    degrades the status rather than failing readiness.
    """
    config = get_settings()
    components: list[ComponentHealth] = []

    started = time.perf_counter()
    db_ok = await check_connection()
    components.append(
        ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY if db_ok else HealthStatus.UNHEALTHY,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=None if db_ok else "connection failed",
        )
    )

    started = time.perf_counter()
    cache_ok = await get_cache().ping()
    components.append(
        ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY if cache_ok else HealthStatus.DEGRADED,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=None if cache_ok else "using in-process fallback",
        )
    )

    open_circuits = [entry for entry in breakers.status() if entry["state"] == "open"]
    if open_circuits:
        components.append(
            ComponentHealth(
                name="sources",
                status=HealthStatus.DEGRADED,
                detail=f"{len(open_circuits)} source circuit(s) open",
            )
        )

    if not db_ok:
        overall = HealthStatus.UNHEALTHY
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(component.status is HealthStatus.DEGRADED for component in components):
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY

    return HealthResponse(
        status=overall,
        version=__version__,
        environment=str(config.environment),
        uptime_seconds=uptime_seconds(),
        components=components,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
async def metrics() -> Response:
    """Prometheus text exposition format.

    Exposes only counters and latencies - no request bodies, user identifiers
    or credentials ever reach this endpoint.
    """
    return Response(
        content=registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/circuits", summary="Source circuit-breaker state")
async def circuits() -> dict[str, object]:
    """Current breaker state per source - the first thing to check on an outage."""
    return {"circuits": breakers.status()}


__all__ = ["router", "uptime_seconds"]
