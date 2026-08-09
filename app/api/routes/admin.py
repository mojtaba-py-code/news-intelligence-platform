"""Administrative endpoints: users, jobs, audit log and maintenance."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.dependencies import (
    AuditRepoDep,
    ClientDep,
    JobRepoDep,
    PaginationDep,
    RequireAdmin,
    SourceRepoDep,
    UserRepoDep,
)
from app.core.errors import ConflictError, ErrorResponse, NotFoundError, ValidationError
from app.core.security import Role
from app.database.models.job import AuditAction, JobType
from app.ingestion.pipeline import ingest_sources
from app.schemas.common import Message, Page
from app.schemas.source import IngestionResult
from app.schemas.user import AdminUserUpdate, UserRead

router = APIRouter(prefix="/admin", tags=["admin"])


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@router.get("/users", response_model=Page[UserRead], summary="List users")
async def list_users(
    admin: RequireAdmin, repository: UserRepoDep, pagination: PaginationDep
) -> Page[UserRead]:
    rows, total = await repository.list_users(limit=pagination.limit, offset=pagination.offset)
    return Page.build([UserRead.model_validate(row) for row in rows], total, pagination)


@router.patch(
    "/users/{user_id}",
    response_model=UserRead,
    summary="Update a user's role or status",
    responses={404: {"model": ErrorResponse, "description": "User not found"}},
)
async def update_user(
    payload: AdminUserUpdate,
    admin: RequireAdmin,
    repository: UserRepoDep,
    audit: AuditRepoDep,
    client: ClientDep,
    user_id: Annotated[int, Path(ge=1)],
) -> UserRead:
    """Change a user's role or activation state.

    Two guardrails: an admin cannot demote or deactivate themselves, and the
    last active admin cannot be removed - either would lock everyone out.
    """
    target = await repository.get(user_id)
    if target is None:
        raise NotFoundError(f"User {user_id} does not exist.")

    updates = payload.model_dump(exclude_unset=True)
    if target.id == admin.id and (
        updates.get("role") not in (None, Role.ADMIN) or updates.get("is_active") is False
    ):
        raise ValidationError("You cannot demote or deactivate your own account.")

    demoting_admin = target.role == str(Role.ADMIN) and (
        updates.get("role") not in (None, Role.ADMIN) or updates.get("is_active") is False
    )
    if demoting_admin and await repository.count_admins() <= 1:
        raise ConflictError("The last active administrator cannot be demoted.")

    if "role" in updates and updates["role"] is not None:
        target.role = str(updates["role"])
    if "is_active" in updates and updates["is_active"] is not None:
        target.is_active = bool(updates["is_active"])
        if not target.is_active:
            await repository.revoke_tokens(target)
    if "is_verified" in updates and updates["is_verified"] is not None:
        target.is_verified = bool(updates["is_verified"])
    if "full_name" in updates:
        target.full_name = updates["full_name"]

    await audit.record(
        AuditAction.ROLE_CHANGED if "role" in updates else AuditAction.USER_UPDATED,
        user_id=admin.id,
        actor=admin.username,
        client_ip=client.ip,
        resource=f"user:{target.id}",
        detail={"fields": sorted(updates)},
    )
    await repository.commit()
    return UserRead.model_validate(target)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
@router.post(
    "/ingest",
    response_model=list[IngestionResult],
    summary="Run ingestion for every active source",
)
async def ingest_all(
    admin: RequireAdmin, audit: AuditRepoDep, client: ClientDep, sources: SourceRepoDep
) -> list[IngestionResult]:
    await audit.record(
        AuditAction.INGESTION_RUN,
        user_id=admin.id,
        actor=admin.username,
        client_ip=client.ip,
        resource="sources:all",
    )
    await sources.commit()
    return await ingest_sources()


@router.get("/jobs", response_model=list[dict], summary="Recent background jobs")
async def list_jobs(
    admin: RequireAdmin,
    repository: JobRepoDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    dead_letters_only: bool = False,
) -> list[dict]:
    rows = (
        await repository.dead_letters(limit=limit)
        if dead_letters_only
        else await repository.recent(limit=limit)
    )
    return [
        {
            "id": job.id,
            "type": job.job_type,
            "status": job.status,
            "target": job.target,
            "attempts": job.attempts,
            "scheduled_for": job.scheduled_for,
            "finished_at": job.finished_at,
            "duration_ms": job.duration_ms,
            "error": job.error,
        }
        for job in rows
    ]


@router.post("/jobs/{job_id}/requeue", response_model=Message, summary="Requeue a failed job")
async def requeue_job(
    admin: RequireAdmin, repository: JobRepoDep, job_id: Annotated[int, Path(ge=1)]
) -> Message:
    if not await repository.requeue(job_id):
        raise NotFoundError(f"Job {job_id} does not exist.")
    await repository.commit()
    return Message(message=f"Job {job_id} requeued.")


@router.post("/jobs", response_model=Message, summary="Enqueue a maintenance job")
async def enqueue_job(
    admin: RequireAdmin,
    repository: JobRepoDep,
    job_type: JobType,
    target: Annotated[str | None, Query(max_length=160)] = None,
) -> Message:
    job = await repository.enqueue(job_type, target=target)
    await repository.commit()
    return Message(message=f"Queued job {job.id} ({job_type}).")


@router.get("/audit", response_model=Page[dict], summary="Audit log")
async def audit_log(
    admin: RequireAdmin,
    repository: AuditRepoDep,
    pagination: PaginationDep,
    action: AuditAction | None = None,
    user_id: Annotated[int | None, Query(ge=1)] = None,
) -> Page[dict]:
    """Security event trail. Client IPs are stored anonymised."""
    rows, total = await repository.recent(
        limit=pagination.limit, offset=pagination.offset, action=action, user_id=user_id
    )
    items = [
        {
            "id": entry.id,
            "created_at": entry.created_at,
            "action": entry.action,
            "user_id": entry.user_id,
            "actor": entry.actor,
            "client_ip": entry.client_ip,
            "resource": entry.resource,
            "success": entry.success,
            "detail": entry.detail,
            "request_id": entry.request_id,
        }
        for entry in rows
    ]
    return Page.build(items, total, pagination)


@router.post("/maintenance/cleanup", response_model=Message, summary="Run retention cleanup")
async def cleanup(
    admin: RequireAdmin,
    sources: SourceRepoDep,
    jobs: JobRepoDep,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
) -> Message:
    """Prune old health samples and finished jobs."""
    health_removed = await sources.prune_health(days=min(days, 30))
    jobs_removed = await jobs.prune(days=min(days, 30))
    await sources.commit()
    return Message(
        message="Cleanup complete.",
        detail=f"{health_removed} health rows and {jobs_removed} jobs removed.",
    )


__all__ = ["router"]
