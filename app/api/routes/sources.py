"""Source management endpoints (read for everyone, write for admins)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from app.api.dependencies import (
    AuditRepoDep,
    ClientDep,
    PaginationDep,
    RequireAdmin,
    RequireAnalyst,
    SourceRepoDep,
)
from app.core.errors import ConflictError, ErrorResponse, NotFoundError
from app.database.models.job import AuditAction
from app.ingestion.pipeline import ingest_sources
from app.schemas.common import Message, Page
from app.schemas.source import (
    IngestionResult,
    SourceCreate,
    SourceHealthRead,
    SourceRead,
    SourceUpdate,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=Page[SourceRead], summary="List sources")
async def list_sources(
    repository: SourceRepoDep,
    pagination: PaginationDep,
    include_disabled: bool = True,
) -> Page[SourceRead]:
    rows = list(await repository.all_sources(include_disabled=include_disabled))
    window = rows[pagination.offset : pagination.offset + pagination.limit]
    return Page.build([SourceRead.model_validate(row) for row in window], len(rows), pagination)


@router.get(
    "/{slug}",
    response_model=SourceRead,
    summary="Source detail",
    responses={404: {"model": ErrorResponse, "description": "Source not found"}},
)
async def get_source(
    repository: SourceRepoDep, slug: Annotated[str, Path(max_length=64)]
) -> SourceRead:
    source = await repository.get_by_slug(slug)
    if source is None:
        raise NotFoundError(f"Source '{slug}' does not exist.")
    return SourceRead.model_validate(source)


@router.get(
    "/{slug}/health",
    response_model=list[SourceHealthRead],
    summary="Recent fetch history",
)
async def source_health(
    repository: SourceRepoDep,
    slug: Annotated[str, Path(max_length=64)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SourceHealthRead]:
    source = await repository.get_by_slug(slug)
    if source is None:
        raise NotFoundError(f"Source '{slug}' does not exist.")
    history = await repository.health_history(source.id, limit=limit)
    return [SourceHealthRead.model_validate(row) for row in history]


@router.post(
    "",
    response_model=SourceRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a source (admin)",
    responses={409: {"model": ErrorResponse, "description": "Slug already registered"}},
)
async def create_source(
    payload: SourceCreate,
    admin: RequireAdmin,
    repository: SourceRepoDep,
    audit: AuditRepoDep,
    client: ClientDep,
) -> SourceRead:
    """Register a new source.

    The URL is validated against the SSRF policy by the schema, and API keys
    are referenced by environment-variable name - never stored here.
    """
    if await repository.get_by_slug(payload.slug):
        raise ConflictError(f"Source '{payload.slug}' already exists.")

    row = payload.model_dump()
    row["kind"] = str(payload.kind)
    source = await repository.upsert_definition(row)
    await audit.record(
        AuditAction.SOURCE_CREATED,
        user_id=admin.id,
        actor=admin.username,
        client_ip=client.ip,
        resource=f"source:{payload.slug}",
    )
    await repository.commit()
    return SourceRead.model_validate(source)


@router.patch(
    "/{slug}",
    response_model=SourceRead,
    summary="Update a source (admin)",
    responses={404: {"model": ErrorResponse, "description": "Source not found"}},
)
async def update_source(
    payload: SourceUpdate,
    admin: RequireAdmin,
    repository: SourceRepoDep,
    audit: AuditRepoDep,
    client: ClientDep,
    slug: Annotated[str, Path(max_length=64)],
) -> SourceRead:
    source = await repository.get_by_slug(slug)
    if source is None:
        raise NotFoundError(f"Source '{slug}' does not exist.")

    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(source, key, str(value) if key == "status" else value)

    await audit.record(
        AuditAction.SOURCE_UPDATED,
        user_id=admin.id,
        actor=admin.username,
        client_ip=client.ip,
        resource=f"source:{slug}",
        detail={"fields": sorted(updates)},
    )
    await repository.commit()
    return SourceRead.model_validate(source)


@router.delete(
    "/{slug}",
    response_model=Message,
    summary="Delete a source (admin)",
    responses={404: {"model": ErrorResponse, "description": "Source not found"}},
)
async def delete_source(
    admin: RequireAdmin,
    repository: SourceRepoDep,
    audit: AuditRepoDep,
    client: ClientDep,
    slug: Annotated[str, Path(max_length=64)],
) -> Message:
    """Delete a source **and its articles** (cascade)."""
    source = await repository.get_by_slug(slug)
    if source is None:
        raise NotFoundError(f"Source '{slug}' does not exist.")
    await repository.delete(source)
    await audit.record(
        AuditAction.SOURCE_DELETED,
        user_id=admin.id,
        actor=admin.username,
        client_ip=client.ip,
        resource=f"source:{slug}",
    )
    await repository.commit()
    return Message(message=f"Source '{slug}' deleted.")


@router.post(
    "/{slug}/ingest",
    response_model=IngestionResult,
    summary="Trigger ingestion for one source (analyst)",
)
async def trigger_ingestion(
    analyst: RequireAnalyst,
    repository: SourceRepoDep,
    audit: AuditRepoDep,
    client: ClientDep,
    slug: Annotated[str, Path(max_length=64)],
) -> IngestionResult:
    """Run one source's pipeline now.

    Ingestion opens its own sessions, so this endpoint stays short-lived even
    though the work itself is I/O bound.
    """
    source = await repository.get_by_slug(slug)
    if source is None:
        raise NotFoundError(f"Source '{slug}' does not exist.")

    await audit.record(
        AuditAction.INGESTION_RUN,
        user_id=analyst.id,
        actor=analyst.username,
        client_ip=client.ip,
        resource=f"source:{slug}",
    )
    await repository.commit()

    results = await ingest_sources([slug])
    return results[0] if results else IngestionResult(source=slug, success=False, error="no result")


__all__ = ["router"]
