"""Background-job and alert persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select

from app.core.utils import ensure_utc, utcnow
from app.database.models.job import Alert, AlertTrigger, JobStatus, JobType, ProcessingJob
from app.database.repositories.base import BaseRepository, affected_rows


class JobRepository(BaseRepository[ProcessingJob]):
    """Durable job queue backed by the database.

    A dedicated broker (Celery/RQ) would be the choice at high throughput; the
    database queue is deliberate here because it gives exactly-once semantics
    with the ingest transaction, survives restarts, and needs no extra service
    for a single-node deployment.
    """

    model = ProcessingJob

    async def enqueue(
        self,
        job_type: JobType,
        *,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
        scheduled_for: datetime | None = None,
        max_attempts: int = 3,
    ) -> ProcessingJob:
        job = ProcessingJob(
            job_type=str(job_type),
            target=target,
            payload=payload or {},
            scheduled_for=scheduled_for or utcnow(),
            max_attempts=max_attempts,
        )
        await self.add(job)
        return job

    async def claim_next(self, *, job_types: list[JobType] | None = None) -> ProcessingJob | None:
        """Atomically take the next due job.

        ``with_for_update(skip_locked=True)`` is what makes multiple workers
        safe on PostgreSQL; SQLite serialises writes anyway, so it degrades to
        a plain select there.
        """
        statement = (
            select(ProcessingJob)
            .where(ProcessingJob.status == str(JobStatus.QUEUED))
            .where(ProcessingJob.scheduled_for <= utcnow())
            .order_by(ProcessingJob.scheduled_for.asc(), ProcessingJob.id.asc())
            .limit(1)
        )
        if job_types:
            statement = statement.where(
                ProcessingJob.job_type.in_([str(job_type) for job_type in job_types])
            )
        if self.session.bind is not None and self.session.bind.dialect.name != "sqlite":
            statement = statement.with_for_update(skip_locked=True)

        job = (await self.session.execute(statement)).scalar_one_or_none()
        if job is None:
            return None

        job.status = str(JobStatus.RUNNING)
        job.started_at = utcnow()
        job.attempts += 1
        await self.flush()
        return job

    async def complete(self, job: ProcessingJob, *, result: dict[str, Any] | None = None) -> None:
        job.status = str(JobStatus.SUCCEEDED)
        job.finished_at = utcnow()
        job.result = result or {}
        # SQLite hands back naive datetimes even for timezone-aware columns, so
        # the stored value has to be re-anchored before any arithmetic.
        started = ensure_utc(job.started_at)
        if started is not None:
            job.duration_ms = (job.finished_at - started).total_seconds() * 1000
        await self.flush()

    async def fail(self, job: ProcessingJob, error: str, *, retry_in_seconds: int = 60) -> None:
        """Reschedule with backoff, or move to the dead-letter state."""
        job.error = error[:2000]
        job.finished_at = utcnow()
        if job.attempts >= job.max_attempts:
            job.status = str(JobStatus.DEAD_LETTER)
        else:
            job.status = str(JobStatus.QUEUED)
            job.scheduled_for = utcnow() + timedelta(
                seconds=retry_in_seconds * (2 ** (job.attempts - 1))
            )
        await self.flush()

    async def queue_depth(self) -> int:
        statement = select(func.count(ProcessingJob.id)).where(
            ProcessingJob.status == str(JobStatus.QUEUED)
        )
        return int((await self.session.execute(statement)).scalar_one() or 0)

    async def recent(self, *, limit: int = 50) -> Sequence[ProcessingJob]:
        statement = select(ProcessingJob).order_by(ProcessingJob.id.desc()).limit(min(limit, 200))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def dead_letters(self, *, limit: int = 50) -> Sequence[ProcessingJob]:
        statement = (
            select(ProcessingJob)
            .where(ProcessingJob.status == str(JobStatus.DEAD_LETTER))
            .order_by(ProcessingJob.finished_at.desc())
            .limit(min(limit, 200))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def requeue(self, job_id: int) -> bool:
        job = await self.get(job_id)
        if job is None:
            return False
        job.status = str(JobStatus.QUEUED)
        job.attempts = 0
        job.error = None
        job.scheduled_for = utcnow()
        await self.flush()
        return True

    async def prune(self, *, days: int = 14) -> int:
        cutoff = utcnow() - timedelta(days=days)
        result = await self.session.execute(
            delete(ProcessingJob)
            .where(ProcessingJob.finished_at < cutoff)
            .where(ProcessingJob.status.in_([str(JobStatus.SUCCEEDED), str(JobStatus.CANCELLED)]))
        )
        return affected_rows(result)


class AlertRepository(BaseRepository[Alert]):
    """Alert rules and their trigger history."""

    model = Alert

    async def for_user(self, user_id: int) -> Sequence[Alert]:
        statement = select(Alert).where(Alert.user_id == user_id).order_by(Alert.created_at.desc())
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def active(self) -> Sequence[Alert]:
        statement = select(Alert).where(Alert.is_active.is_(True))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def due(self) -> Sequence[Alert]:
        """Active rules whose cooldown has elapsed."""
        now = utcnow()
        alerts = await self.active()
        return [
            alert
            for alert in alerts
            if alert.last_triggered_at is None
            or alert.last_triggered_at + timedelta(minutes=alert.cooldown_minutes) <= now
        ]

    async def record_trigger(
        self,
        alert: Alert,
        *,
        article_ids: list[int],
        message: str,
        delivered: bool = False,
        delivery_error: str | None = None,
    ) -> AlertTrigger:
        now = utcnow()
        trigger = AlertTrigger(
            alert_id=alert.id,
            triggered_at=now,
            article_count=len(article_ids),
            article_ids=[str(article_id) for article_id in article_ids[:50]],
            message=message[:512],
            delivered=delivered,
            delivery_error=(delivery_error or "")[:512] or None,
        )
        self.session.add(trigger)
        alert.last_triggered_at = now
        alert.trigger_count += 1
        await self.flush()
        return trigger

    async def triggers_for_user(
        self, user_id: int, *, limit: int = 50, unacknowledged_only: bool = False
    ) -> Sequence[AlertTrigger]:
        statement = (
            select(AlertTrigger)
            .join(Alert, Alert.id == AlertTrigger.alert_id)
            .where(Alert.user_id == user_id)
            .order_by(AlertTrigger.triggered_at.desc())
            .limit(min(limit, 200))
        )
        if unacknowledged_only:
            statement = statement.where(AlertTrigger.acknowledged.is_(False))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def acknowledge(self, trigger_id: int, user_id: int) -> bool:
        trigger = await self.session.scalar(
            select(AlertTrigger)
            .join(Alert, Alert.id == AlertTrigger.alert_id)
            .where(AlertTrigger.id == trigger_id, Alert.user_id == user_id)
        )
        if trigger is None:
            return False
        trigger.acknowledged = True
        await self.flush()
        return True


__all__ = ["AlertRepository", "JobRepository"]
