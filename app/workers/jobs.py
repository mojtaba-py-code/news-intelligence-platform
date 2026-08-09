"""Job handlers - the actual work the background worker performs.

Each handler owns its transaction and returns a JSON-serialisable summary that
is stored on the job row, so an operator can see what a run actually did
without reading logs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.models.job import JobType
from app.database.repositories.article import ArticleRepository
from app.database.repositories.job import JobRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import EntityRepository, TopicRepository, TrendRepository
from app.database.session import session_scope
from app.ingestion.pipeline import ingest_sources
from app.services.alerts import AlertService
from app.services.analytics import AnalyticsService
from app.services.events import EventService
from app.services.trends import TrendService

logger = get_logger(__name__)

JobHandler = Callable[[dict[str, Any], Settings], Awaitable[dict[str, Any]]]


async def run_ingestion(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Fetch and process every active source (or the ones named in the payload)."""
    slugs = payload.get("sources") or None
    results = await ingest_sources(slugs, config=config)

    async with session_scope() as session:
        await AnalyticsService(session, config=config).invalidate()

    return {
        "sources": len(results),
        "stored": sum(result.stored for result in results),
        "duplicates": sum(result.duplicates for result in results),
        "rejected": sum(result.rejected for result in results),
        "failed": [result.source for result in results if not result.success],
    }


async def run_processing(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Refresh denormalised counters after a batch of articles landed."""
    async with session_scope() as session:
        topics = TopicRepository(session)
        entities = EntityRepository(session)
        await topics.refresh_counts()
        await entities.refresh_counts()

        sources = SourceRepository(session)
        reliability: dict[str, float] = {}
        for source in await sources.all_sources():
            reliability[source.slug] = await sources.recompute_reliability(source.id)

    return {"sources_scored": len(reliability)}


async def run_trends(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Recompute and persist trend snapshots."""
    hours = int(payload.get("hours") or config.trend_window_hours)
    async with session_scope() as session:
        results = await TrendService(session, config=config).compute(hours=hours)
        await AnalyticsService(session, config=config).invalidate()
    breaking = [result.subject_label for result in results if result.is_breaking][:10]
    return {"detected": len(results), "breaking": breaking}


async def run_events(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Cluster recent articles into events."""
    hours = int(payload.get("hours") or config.event_window_hours)
    async with session_scope() as session:
        clusters = await EventService(session, config=config).detect(hours=hours)
    return {
        "clusters": len(clusters),
        "largest": max((cluster.size for cluster in clusters), default=0),
    }


async def run_alerts(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Evaluate every due alert rule."""
    async with session_scope() as session:
        evaluations = await AlertService(session, config=config).evaluate_all()
    return {
        "evaluated": len(evaluations),
        "triggered": sum(1 for item in evaluations if item.triggered),
        "delivery_failures": sum(
            1 for item in evaluations if item.triggered and not item.delivered
        ),
    }


async def run_cleanup(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Apply retention policies to articles, health samples, jobs and trends."""
    retention_days = int(payload.get("retention_days") or config.retention_days)
    async with session_scope() as session:
        removed_articles = await ArticleRepository(session).delete_older_than(days=retention_days)
        removed_health = await SourceRepository(session).prune_health(days=30)
        removed_jobs = await JobRepository(session).prune(days=14)
        removed_trends = await TrendRepository(session).prune(days=30)
    summary = {
        "articles": removed_articles,
        "health_samples": removed_health,
        "jobs": removed_jobs,
        "trend_snapshots": removed_trends,
    }
    logger.info("cleanup_complete", extra=summary)
    return summary


async def run_deduplicate(payload: dict[str, Any], config: Settings) -> dict[str, Any]:
    """Re-check recent articles for duplicates missed by the ingest-time window."""
    from app.processing.deduplication.engine import DeduplicationEngine

    async with session_scope() as session:
        articles = ArticleRepository(session)
        candidates = list(
            await articles.recent_candidates(hours=config.dedup_lookback_hours, limit=2000)
        )
        engine = DeduplicationEngine.from_settings(config)
        marked = 0
        seen: list[Any] = []
        for article in sorted(candidates, key=lambda row: row.published_at):
            engine.load(seen)
            match = engine.find_duplicate(
                canonical_url=article.canonical_url,
                content_hash=article.content_hash,
                title=article.title,
                title_hash=article.title_hash,
                content=article.content,
                description=article.description,
                simhash=article.simhash,
            )
            if match is not None and match.article_id != article.id:
                await articles.mark_duplicate(
                    article.id,
                    original_id=match.article_id,
                    score=match.score,
                    method=str(match.level),
                )
                marked += 1
            else:
                seen.append(article)
    return {"scanned": len(candidates), "marked_duplicate": marked}


JOB_HANDLERS: dict[str, JobHandler] = {
    str(JobType.INGEST): run_ingestion,
    str(JobType.PROCESS): run_processing,
    str(JobType.TRENDS): run_trends,
    str(JobType.EVENTS): run_events,
    str(JobType.ALERTS): run_alerts,
    str(JobType.CLEANUP): run_cleanup,
    str(JobType.DEDUPLICATE): run_deduplicate,
}


async def run_job(
    job_type: str, payload: dict[str, Any] | None = None, *, config: Settings | None = None
) -> dict[str, Any]:
    """Dispatch one job by type."""
    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        raise ValueError(f"unknown job type '{job_type}'")
    return await handler(payload or {}, config or get_settings())


__all__ = ["JOB_HANDLERS", "run_job"]
