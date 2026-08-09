"""Background jobs, the worker loop and the scheduler."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.models.article import Article
from app.database.models.job import JobStatus, JobType
from app.database.models.source import Source
from app.database.repositories.job import JobRepository
from app.database.repositories.taxonomy import TopicRepository
from app.intelligence.topics import default_topic_seed
from app.workers.jobs import JOB_HANDLERS, run_job
from app.workers.scheduler import ScheduledTask, Scheduler, default_schedule
from app.workers.worker import Worker

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _shared_engine(engine: object) -> None:
    """Workers open their own sessions, so they need the module-level engine."""
    return None


class TestJobHandlers:
    def test_every_job_type_has_a_handler(self) -> None:
        assert {str(job_type) for job_type in JobType} == set(JOB_HANDLERS)

    async def test_unknown_job_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown job type"):
            await run_job("teleport")

    async def test_ingest_job_with_no_sources(self) -> None:
        result = await run_job(str(JobType.INGEST))
        assert result["sources"] == 0
        assert result["stored"] == 0

    async def test_processing_job_refreshes_counters(
        self, session: AsyncSession, source: Source
    ) -> None:
        await TopicRepository(session).seed(default_topic_seed())
        await session.commit()

        result = await run_job(str(JobType.PROCESS))
        assert result["sources_scored"] == 1

    async def test_trend_and_event_jobs_run_on_an_empty_database(self) -> None:
        trends = await run_job(str(JobType.TRENDS))
        events = await run_job(str(JobType.EVENTS))
        assert trends["detected"] == 0
        assert events["clusters"] == 0

    async def test_cleanup_job_reports_what_it_removed(self) -> None:
        result = await run_job(str(JobType.CLEANUP), {"retention_days": 1})
        assert set(result) == {"articles", "health_samples", "jobs", "trend_snapshots"}

    async def test_alert_job_with_no_rules(self) -> None:
        result = await run_job(str(JobType.ALERTS))
        assert result["evaluated"] == 0

    async def test_deduplicate_job_marks_late_duplicates(
        self, session: AsyncSession, source: Source
    ) -> None:
        """A duplicate that arrived outside the ingest-time window is caught here."""
        from tests.conftest import make_article

        body = (
            "The regulator approved the merger after a review lasting more than a year, "
            "clearing the way for the combined company to operate in both markets."
        )
        session.add_all(
            [
                make_article(
                    source,
                    title="Regulator approves merger",
                    url="https://e.com/1",
                    content=body,
                    hours_old=5,
                ),
                # Same story, a sentence longer: different hash, close SimHash.
                make_article(
                    source,
                    title="Regulator approves merger",
                    url="https://e.com/2",
                    content=body + " Analysts welcomed the decision on Friday.",
                    hours_old=4,
                ),
            ]
        )
        await session.commit()

        result = await run_job(str(JobType.DEDUPLICATE))
        assert result["scanned"] == 2
        assert result["marked_duplicate"] == 1


class TestWorker:
    async def test_worker_claims_and_completes_a_job(self, session: AsyncSession) -> None:
        repository = JobRepository(session)
        job = await repository.enqueue(JobType.CLEANUP, payload={"retention_days": 1})
        await session.commit()
        job_id = job.id

        worked = await Worker(config=get_settings()).run_once()
        assert worked is True

        session.expire_all()
        finished = await repository.get(job_id)
        assert finished is not None
        assert finished.status == str(JobStatus.SUCCEEDED)
        assert finished.result
        assert finished.duration_ms is not None

    async def test_worker_returns_false_when_idle(self) -> None:
        assert await Worker(config=get_settings()).run_once() is False

    async def test_failing_job_is_retried_then_dead_lettered(
        self, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.workers.worker as worker_module

        async def explode(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("handler blew up")

        monkeypatch.setattr(worker_module, "run_job", explode)

        repository = JobRepository(session)
        job = await repository.enqueue(JobType.TRENDS, max_attempts=2)
        await session.commit()
        job_id = job.id

        worker = Worker(config=get_settings())
        assert await worker.run_once() is True

        session.expire_all()
        first = await repository.get(job_id)
        assert first is not None
        assert first.status == str(JobStatus.QUEUED)  # scheduled for a retry
        assert "handler blew up" in (first.error or "")

        # Make the retry due immediately, then exhaust the attempt budget.
        first.scheduled_for = first.created_at
        await session.commit()

        assert await worker.run_once() is True
        session.expire_all()
        second = await repository.get(job_id)
        assert second is not None
        assert second.status == str(JobStatus.DEAD_LETTER)

    async def test_worker_loop_stops_on_request(self) -> None:
        worker = Worker(config=get_settings(), poll_interval=0.01)
        task = asyncio.create_task(worker.run_forever())
        await asyncio.sleep(0.05)
        worker.request_stop()
        await asyncio.wait_for(task, timeout=2)
        assert task.done()


class TestScheduler:
    def test_default_schedule_covers_every_recurring_job(self) -> None:
        types = {str(task.job_type) for task in default_schedule()}
        assert types == {
            str(JobType.INGEST),
            str(JobType.PROCESS),
            str(JobType.TRENDS),
            str(JobType.EVENTS),
            str(JobType.ALERTS),
            str(JobType.CLEANUP),
        }
        assert all(task.interval_seconds > 0 for task in default_schedule())

    async def test_enqueue_creates_a_job(self, session: AsyncSession) -> None:
        scheduler = Scheduler(config=get_settings(), tasks=[])
        assert await scheduler.enqueue(ScheduledTask(JobType.TRENDS, 60)) is True
        assert await JobRepository(session).queue_depth() == 1

    async def test_backlog_suppresses_duplicate_scheduling(self, session: AsyncSession) -> None:
        """A slow worker must not accumulate ten identical ingest jobs."""
        scheduler = Scheduler(config=get_settings(), tasks=[])
        task = ScheduledTask(JobType.INGEST, 60)
        assert await scheduler.enqueue(task) is True
        assert await scheduler.enqueue(task) is False
        assert await JobRepository(session).queue_depth() == 1

    async def test_scheduler_loop_stops_on_request(self) -> None:
        scheduler = Scheduler(
            config=get_settings(), tasks=[ScheduledTask(JobType.TRENDS, 3600, initial_delay=5)]
        )
        loop_task = asyncio.create_task(scheduler.run_forever())
        await asyncio.sleep(0.05)
        scheduler.request_stop()
        await asyncio.wait_for(loop_task, timeout=2)
        assert loop_task.done()


class TestQueueEndToEnd:
    async def test_scheduled_job_is_picked_up_by_a_worker(
        self, session: AsyncSession, source: Source
    ) -> None:
        scheduler = Scheduler(config=get_settings(), tasks=[])
        await scheduler.enqueue(ScheduledTask(JobType.PROCESS, 60))

        worker = Worker(config=get_settings())
        assert await worker.run_once() is True
        assert await worker.run_once() is False  # queue drained

        session.expire_all()
        jobs = await JobRepository(session).recent()
        assert len(jobs) == 1
        assert jobs[0].status == str(JobStatus.SUCCEEDED)

    async def test_articles_survive_a_cleanup_within_retention(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        result = await run_job(str(JobType.CLEANUP), {"retention_days": 365})
        assert result["articles"] == 0
