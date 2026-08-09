"""Topics, entities, events and trends."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.dependencies import (
    EntityRepoDep,
    EventRepoDep,
    EventServiceDep,
    PaginationDep,
    RequireAnalyst,
    TopicRepoDep,
    TrendRepoDep,
    TrendServiceDep,
)
from app.core.errors import ErrorResponse, NotFoundError
from app.database.models.taxonomy import EntityType, TrendSubject
from app.schemas.common import Page
from app.schemas.intelligence import (
    EntityGraph,
    EntityGraphEdge,
    EntityRead,
    EventRead,
    TopicRead,
    TrendRead,
)

router = APIRouter(tags=["intelligence"])


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #
@router.get("/topics", response_model=list[TopicRead], summary="List topics")
async def list_topics(repository: TopicRepoDep) -> list[TopicRead]:
    rows = await repository.active()
    return [TopicRead.model_validate(row) for row in rows]


@router.get(
    "/topics/{slug}",
    response_model=TopicRead,
    summary="Topic detail",
    responses={404: {"model": ErrorResponse, "description": "Topic not found"}},
)
async def get_topic(
    repository: TopicRepoDep, slug: Annotated[str, Path(max_length=64)]
) -> TopicRead:
    topic = await repository.get_by_slug(slug)
    if topic is None:
        raise NotFoundError(f"Topic '{slug}' does not exist.")
    return TopicRead.model_validate(topic)


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
@router.get("/entities", response_model=list[EntityRead], summary="Top entities")
async def list_entities(
    repository: EntityRepoDep,
    entity_type: EntityType | None = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[EntityRead]:
    rows = (
        await repository.search(q, limit=limit)
        if q
        else await repository.top(entity_type=entity_type, limit=limit)
    )
    return [EntityRead.model_validate(row) for row in rows]


@router.get(
    "/entities/{entity_id}/graph",
    response_model=EntityGraph,
    summary="Knowledge-graph neighbourhood",
    responses={404: {"model": ErrorResponse, "description": "Entity not found"}},
)
async def entity_graph(
    repository: EntityRepoDep,
    entity_id: Annotated[int, Path(ge=1)],
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
) -> EntityGraph:
    """Entities that co-occur with this one, weighted by shared articles.

    Co-occurrence is the honest signal available without a relation extractor:
    the edge means "appears together with", not an asserted relationship.
    """
    entity = await repository.get(entity_id)
    if entity is None:
        raise NotFoundError(f"Entity {entity_id} does not exist.")

    neighbours = await repository.co_occurrences(entity_id, limit=limit)
    nodes = [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "mentions": entity.mention_count,
            "root": True,
        }
    ]
    edges: list[EntityGraphEdge] = []
    peak = max((shared for _, shared in neighbours), default=1)

    for neighbour, shared in neighbours:
        nodes.append(
            {
                "id": neighbour.id,
                "name": neighbour.name,
                "type": neighbour.entity_type,
                "mentions": neighbour.mention_count,
                "root": False,
            }
        )
        edges.append(
            EntityGraphEdge(
                subject=entity.name,
                subject_type=entity.entity_type,
                predicate="co_occurs_with",
                object=neighbour.name,
                object_type=neighbour.entity_type,
                weight=round(shared / peak, 4),
            )
        )
    return EntityGraph(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@router.get("/events", response_model=Page[EventRead], summary="Detected events")
async def list_events(
    repository: EventRepoDep,
    pagination: PaginationDep,
    min_sources: Annotated[int, Query(ge=1, le=20)] = 2,
) -> Page[EventRead]:
    rows, total = await repository.recent(
        limit=pagination.limit, offset=pagination.offset, min_sources=min_sources
    )
    return Page.build([EventRead.model_validate(row) for row in rows], total, pagination)


@router.get("/events/breaking", response_model=list[EventRead], summary="Breaking news")
async def breaking_events(
    repository: EventRepoDep,
    hours: Annotated[int, Query(ge=1, le=72)] = 6,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[EventRead]:
    """Clusters that formed recently across three or more sources."""
    rows = await repository.breaking(hours=hours, limit=limit)
    return [EventRead.model_validate(row) for row in rows]


@router.get(
    "/events/{event_id}",
    response_model=EventRead,
    summary="Event detail",
    responses={404: {"model": ErrorResponse, "description": "Event not found"}},
)
async def get_event(repository: EventRepoDep, event_id: Annotated[int, Path(ge=1)]) -> EventRead:
    event = await repository.get_detail(event_id)
    if event is None:
        raise NotFoundError(f"Event {event_id} does not exist.")
    return EventRead.model_validate(event)


@router.post(
    "/events/detect",
    response_model=list[EventRead],
    summary="Run event detection now (analyst)",
)
async def detect_events(
    analyst: RequireAnalyst,
    service: EventServiceDep,
    repository: EventRepoDep,
    hours: Annotated[int, Query(ge=1, le=168)] = 48,
) -> list[EventRead]:
    await service.detect(hours=hours)
    await repository.commit()
    rows, _ = await repository.recent(limit=20, offset=0)
    return [EventRead.model_validate(row) for row in rows]


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #
@router.get("/trends", response_model=list[TrendRead], summary="Current trends")
async def list_trends(
    repository: TrendRepoDep,
    subject_type: TrendSubject | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> list[TrendRead]:
    rows = await repository.latest(subject_type=subject_type, limit=limit, min_score=min_score)
    return [TrendRead.model_validate(row) for row in rows]


@router.get(
    "/trends/{subject_type}/{subject_key}/history",
    response_model=list[TrendRead],
    summary="Trend history for one subject",
)
async def trend_history(
    repository: TrendRepoDep,
    subject_type: TrendSubject,
    subject_key: Annotated[str, Path(max_length=160)],
    limit: Annotated[int, Query(ge=1, le=200)] = 48,
) -> list[TrendRead]:
    rows = await repository.history(subject_type, subject_key, limit=limit)
    return [TrendRead.model_validate(row) for row in rows]


@router.post(
    "/trends/compute",
    response_model=list[TrendRead],
    summary="Recompute trends now (analyst)",
)
async def compute_trends(
    analyst: RequireAnalyst,
    service: TrendServiceDep,
    repository: TrendRepoDep,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TrendRead]:
    await service.compute(hours=hours)
    await repository.commit()
    rows = await repository.latest(limit=limit)
    return [TrendRead.model_validate(row) for row in rows]


__all__ = ["router"]
