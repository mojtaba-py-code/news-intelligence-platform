"""Analytics endpoints powering the dashboard."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import AnalyticsServiceDep
from app.schemas.intelligence import (
    AnalyticsOverview,
    CountBucket,
    SentimentBreakdown,
    TimeSeriesPoint,
)
from app.schemas.source import SourceStats

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverview, summary="Platform overview")
async def overview(service: AnalyticsServiceDep, refresh: bool = False) -> AnalyticsOverview:
    """Headline counters, trending topics and the sentiment split.

    Cached briefly (``CACHE_TTL_SECONDS``); pass ``refresh=true`` to bypass.
    """
    return await service.overview(use_cache=not refresh)


@router.get("/sentiment", response_model=SentimentBreakdown, summary="Sentiment distribution")
async def sentiment(
    service: AnalyticsServiceDep,
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> SentimentBreakdown:
    return await service.sentiment(hours=hours)


@router.get("/topics", response_model=list[CountBucket], summary="Topic distribution")
async def topics(
    service: AnalyticsServiceDep,
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
) -> list[CountBucket]:
    return await service.topics(hours=hours, limit=limit)


@router.get("/timeseries", response_model=list[TimeSeriesPoint], summary="Volume over time")
async def timeseries(
    service: AnalyticsServiceDep,
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> list[TimeSeriesPoint]:
    """Hourly article volume and mean sentiment."""
    return await service.timeseries(hours=hours)


@router.get("/sources", response_model=list[SourceStats], summary="Source comparison")
async def sources(
    service: AnalyticsServiceDep,
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> list[SourceStats]:
    return await service.source_stats(hours=hours)


__all__ = ["router"]
