"""User-defined alert rules."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from app.api.dependencies import AlertRepoDep, AlertServiceDep, CurrentUser
from app.core.errors import AuthorizationError, ErrorResponse, NotFoundError
from app.database.models.job import Alert
from app.schemas.common import Message
from app.schemas.intelligence import AlertCreate, AlertRead, AlertTriggerRead

router = APIRouter(prefix="/alerts", tags=["alerts"])

MAX_ALERTS_PER_USER = 25


async def _owned_alert(repository: AlertRepoDep, alert_id: int, user_id: int) -> Alert:
    """Fetch an alert, enforcing ownership.

    A 404 for someone else's alert would still confirm the id exists, so
    ownership is checked before anything is returned.
    """
    alert = await repository.get(alert_id)
    if alert is None:
        raise NotFoundError(f"Alert {alert_id} does not exist.")
    if alert.user_id != user_id:
        raise AuthorizationError("You do not own this alert.")
    return alert


@router.get("", response_model=list[AlertRead], summary="List your alerts")
async def list_alerts(user: CurrentUser, repository: AlertRepoDep) -> list[AlertRead]:
    rows = await repository.for_user(user.id)
    return [AlertRead.model_validate(row) for row in rows]


@router.post(
    "",
    response_model=AlertRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an alert rule",
    responses={409: {"model": ErrorResponse, "description": "Alert quota reached"}},
)
async def create_alert(
    payload: AlertCreate, user: CurrentUser, repository: AlertRepoDep
) -> AlertRead:
    """Create a structured alert rule.

    Conditions are typed fields, never an expression string - there is nothing
    here for an attacker to inject into.
    """
    existing = await repository.for_user(user.id)
    if len(existing) >= MAX_ALERTS_PER_USER:
        from app.core.errors import ConflictError

        raise ConflictError(f"You may define at most {MAX_ALERTS_PER_USER} alerts.")

    alert = Alert(
        user_id=user.id,
        name=payload.name,
        keywords=payload.keywords,
        topics=payload.topics,
        entities=payload.entities,
        sources=payload.sources,
        languages=payload.languages,
        min_articles=payload.min_articles,
        window_minutes=payload.window_minutes,
        min_relevance=payload.min_relevance,
        sentiment_filter=payload.sentiment_filter,
        channel=str(payload.channel),
        destination=payload.destination,
        cooldown_minutes=payload.cooldown_minutes,
    )
    await repository.add(alert)
    await repository.commit()
    return AlertRead.model_validate(alert)


@router.patch("/{alert_id}", response_model=AlertRead, summary="Enable or disable an alert")
async def toggle_alert(
    user: CurrentUser,
    repository: AlertRepoDep,
    alert_id: Annotated[int, Path(ge=1)],
    is_active: bool,
) -> AlertRead:
    alert = await _owned_alert(repository, alert_id, user.id)
    alert.is_active = is_active
    await repository.commit()
    return AlertRead.model_validate(alert)


@router.delete("/{alert_id}", response_model=Message, summary="Delete an alert")
async def delete_alert(
    user: CurrentUser, repository: AlertRepoDep, alert_id: Annotated[int, Path(ge=1)]
) -> Message:
    alert = await _owned_alert(repository, alert_id, user.id)
    await repository.delete(alert)
    await repository.commit()
    return Message(message="Alert deleted.")


@router.get(
    "/triggers",
    response_model=list[AlertTriggerRead],
    summary="Recent alert notifications",
)
async def list_triggers(
    user: CurrentUser,
    repository: AlertRepoDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    unacknowledged_only: bool = False,
) -> list[AlertTriggerRead]:
    rows = await repository.triggers_for_user(
        user.id, limit=limit, unacknowledged_only=unacknowledged_only
    )
    return [AlertTriggerRead.model_validate(row) for row in rows]


@router.post(
    "/triggers/{trigger_id}/acknowledge",
    response_model=Message,
    summary="Acknowledge a notification",
)
async def acknowledge(
    user: CurrentUser, repository: AlertRepoDep, trigger_id: Annotated[int, Path(ge=1)]
) -> Message:
    if not await repository.acknowledge(trigger_id, user.id):
        raise NotFoundError(f"Trigger {trigger_id} does not exist.")
    await repository.commit()
    return Message(message="Acknowledged.")


@router.post("/{alert_id}/test", response_model=Message, summary="Evaluate an alert now")
async def test_alert(
    user: CurrentUser,
    repository: AlertRepoDep,
    service: AlertServiceDep,
    alert_id: Annotated[int, Path(ge=1)],
) -> Message:
    """Evaluate one rule immediately, ignoring its cooldown."""
    alert = await _owned_alert(repository, alert_id, user.id)
    evaluation = await service.evaluate(alert)
    await repository.commit()
    return Message(
        message=f"Matched {evaluation.matched} article(s).",
        detail="triggered" if evaluation.triggered else "threshold not reached",
    )


__all__ = ["router"]
