"""The background worker: claims queued jobs and runs them.

Claiming is atomic (``SELECT … FOR UPDATE SKIP LOCKED`` on PostgreSQL), so
several worker processes can share one queue safely. Failures are retried with
exponential backoff and end in a dead-letter state rather than looping forever.
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import queue_depth
from app.database.models.job import JobType
from app.database.repositories.job import JobRepository
from app.database.session import session_scope
from app.workers.jobs import run_job

logger = get_logger(__name__)

#: How long to sleep when the queue is empty, in seconds.
IDLE_SLEEP = 2.0


@dataclass
class Worker:
    """Polls the job table and executes handlers."""

    config: Settings = field(default_factory=get_settings)
    job_types: list[JobType] | None = None
    poll_interval: float = IDLE_SLEEP
    _stopping: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    def request_stop(self) -> None:
        """Ask the loop to finish the current job and exit."""
        self._stopping.set()

    async def run_forever(self) -> None:
        """Main loop. Returns when :meth:`request_stop` is called."""
        logger.info(
            "worker_started",
            extra={"job_types": [str(t) for t in (self.job_types or [])] or "all"},
        )
        while not self._stopping.is_set():
            try:
                worked = await self.run_once()
            except Exception:
                logger.exception("worker_iteration_failed")
                worked = False
            if not worked:
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    continue
        logger.info("worker_stopped")

    async def run_once(self) -> bool:
        """Claim and execute one job. Returns ``True`` when work was done."""
        async with session_scope() as session:
            repository = JobRepository(session)
            job = await repository.claim_next(job_types=self.job_types)
            queue_depth.set(await repository.queue_depth())
            if job is None:
                return False
            job_id, job_type, payload = job.id, job.job_type, dict(job.payload or {})

        logger.info("job_started", extra={"job_id": job_id, "job_type": job_type})
        try:
            result = await run_job(job_type, payload, config=self.config)
        except Exception as exc:
            logger.exception("job_failed", extra={"job_id": job_id, "job_type": job_type})
            async with session_scope() as session:
                repository = JobRepository(session)
                failed = await repository.get(job_id)
                if failed is not None:
                    await repository.fail(failed, f"{exc.__class__.__name__}: {exc}")
            return True

        async with session_scope() as session:
            repository = JobRepository(session)
            done = await repository.get(job_id)
            if done is not None:
                await repository.complete(done, result=result)
        logger.info(
            "job_completed", extra={"job_id": job_id, "job_type": job_type, **_flat(result)}
        )
        return True


def _flat(result: dict[str, object]) -> dict[str, object]:
    """Keep only scalar fields so the log line stays one level deep."""
    return {key: value for key, value in result.items() if isinstance(value, (int, float, str))}


async def main() -> None:  # pragma: no cover - process entrypoint
    """Run a worker until SIGINT/SIGTERM."""
    from app.core.logging import configure_logging
    from app.database.session import dispose_engine, init_engine

    config = get_settings()
    configure_logging(config)
    init_engine(config)

    worker = Worker(config=config)
    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue
        try:
            loop.add_signal_handler(signal_number, worker.request_stop)
        except NotImplementedError:
            # Windows: signal handlers are not supported on the proactor loop.
            signal.signal(signal_number, lambda *_: worker.request_stop())

    try:
        await worker.run_forever()
    finally:
        await dispose_engine()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())


__all__ = ["Worker", "main"]
