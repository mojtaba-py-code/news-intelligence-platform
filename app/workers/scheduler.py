"""The scheduler: enqueues recurring jobs at configured intervals.

Kept separate from the worker on purpose. The scheduler only *enqueues* - it
never executes - so scheduling stays cheap and idempotent, and the number of
workers can be scaled independently of the schedule.

Duplicate suppression matters: if a worker is slow, the scheduler must not pile
up ten identical ingest jobs. Before enqueueing it checks whether an equivalent
job is already queued or running.
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field

from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.models.job import JobStatus, JobType, ProcessingJob
from app.database.repositories.job import JobRepository
from app.database.session import session_scope

logger = get_logger(__name__)


@dataclass(slots=True)
class ScheduledTask:
    """One recurring job definition."""

    job_type: JobType
    interval_seconds: int
    payload: dict[str, object] = field(default_factory=dict)
    #: Seconds to wait before the first run (staggers startup load).
    initial_delay: float = 0.0
    enabled: bool = True


def default_schedule(config: Settings | None = None) -> list[ScheduledTask]:
    """The schedule from the prompt, with every interval configurable."""
    config = config or get_settings()
    return [
        ScheduledTask(JobType.INGEST, config.schedule_ingest_seconds, initial_delay=5),
        ScheduledTask(JobType.PROCESS, config.schedule_process_seconds, initial_delay=30),
        ScheduledTask(JobType.TRENDS, config.schedule_trends_seconds, initial_delay=60),
        ScheduledTask(JobType.EVENTS, config.schedule_events_seconds, initial_delay=90),
        ScheduledTask(
            JobType.ALERTS,
            max(60, config.schedule_process_seconds // 2),
            initial_delay=45,
            enabled=config.alerts_enabled,
        ),
        ScheduledTask(JobType.CLEANUP, config.schedule_cleanup_seconds, initial_delay=300),
    ]


@dataclass
class Scheduler:
    """Runs one asyncio task per scheduled job."""

    config: Settings = field(default_factory=get_settings)
    tasks: list[ScheduledTask] = field(default_factory=list)
    _stopping: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    def __post_init__(self) -> None:
        if not self.tasks:
            self.tasks = default_schedule(self.config)

    def request_stop(self) -> None:
        self._stopping.set()

    async def run_forever(self) -> None:
        """Start every enabled task and wait for a stop request."""
        active = [task for task in self.tasks if task.enabled]
        logger.info(
            "scheduler_started",
            extra={"tasks": [f"{t.job_type}@{t.interval_seconds}s" for t in active]},
        )
        runners = [asyncio.create_task(self._run_task(task)) for task in active]
        try:
            await self._stopping.wait()
        finally:
            for runner in runners:
                runner.cancel()
            await asyncio.gather(*runners, return_exceptions=True)
            logger.info("scheduler_stopped")

    async def _run_task(self, task: ScheduledTask) -> None:
        if task.initial_delay:
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=task.initial_delay)
                return
            except TimeoutError:
                pass

        while not self._stopping.is_set():
            try:
                await self.enqueue(task)
            except Exception:
                logger.exception("schedule_enqueue_failed", extra={"job_type": str(task.job_type)})
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=task.interval_seconds)
                return
            except TimeoutError:
                continue

    async def enqueue(self, task: ScheduledTask) -> bool:
        """Enqueue the task unless an identical one is already pending."""
        async with session_scope() as session:
            pending = await session.scalar(
                select(func.count(ProcessingJob.id))
                .where(ProcessingJob.job_type == str(task.job_type))
                .where(ProcessingJob.status.in_([str(JobStatus.QUEUED), str(JobStatus.RUNNING)]))
            )
            if pending:
                logger.debug(
                    "schedule_skipped_backlog",
                    extra={"job_type": str(task.job_type), "pending": int(pending)},
                )
                return False

            await JobRepository(session).enqueue(task.job_type, payload=dict(task.payload))
        logger.info("job_scheduled", extra={"job_type": str(task.job_type)})
        return True


async def main() -> None:  # pragma: no cover - process entrypoint
    """Run the scheduler until SIGINT/SIGTERM."""
    from app.core.logging import configure_logging
    from app.database.session import dispose_engine, init_engine

    config = get_settings()
    configure_logging(config)
    init_engine(config)

    scheduler = Scheduler(config=config)
    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue
        try:
            loop.add_signal_handler(signal_number, scheduler.request_stop)
        except NotImplementedError:
            signal.signal(signal_number, lambda *_: scheduler.request_stop())

    try:
        await scheduler.run_forever()
    finally:
        await dispose_engine()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())


__all__ = ["ScheduledTask", "Scheduler", "default_schedule", "main"]
