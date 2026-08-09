"""Event service: cluster recent articles into stories and persist them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.utils import utcnow
from app.database.models.event import Event
from app.database.repositories.article import ArticleRepository
from app.database.repositories.event import EventRepository
from app.intelligence.events import EventCluster, cluster_articles
from app.intelligence.summarize import lede

logger = get_logger(__name__)


class EventService:
    """Runs clustering over the recent window and upserts the resulting events."""

    def __init__(self, session: AsyncSession, *, config: Settings | None = None) -> None:
        self.session = session
        self.config = config or get_settings()
        self.articles = ArticleRepository(session)
        self.events = EventRepository(session)

    async def detect(self, *, hours: int | None = None, persist: bool = True) -> list[EventCluster]:
        """Cluster the recent processed articles into events."""
        hours = hours or self.config.event_window_hours
        articles = list(await self.articles.for_clustering(hours=hours))
        if len(articles) < 2:
            logger.info("event_detection_skipped", extra={"articles": len(articles)})
            return []

        clusters = cluster_articles(articles, threshold=self.config.event_similarity_threshold)
        if not persist or not clusters:
            return clusters

        by_id = {article.id: article for article in articles}
        stored = 0
        for cluster in clusters:
            await self._persist(cluster, by_id)
            stored += 1

        logger.info(
            "events_detected",
            extra={"window_hours": hours, "articles": len(articles), "clusters": stored},
        )
        return clusters

    async def _persist(self, cluster: EventCluster, by_id: Mapping[int, Any]) -> Event:
        """Materialise one cluster, aggregating topics/entities from members."""
        topics: list[str] = []
        entities: list[dict[str, object]] = []
        seen_entities: set[str] = set()
        anchor_summary: str | None = None

        for article_id in cluster.article_ids:
            article = by_id.get(article_id)
            if article is None:
                continue
            if anchor_summary is None:
                anchor_summary = getattr(article, "summary", None) or lede(
                    getattr(article, "description", None) or getattr(article, "content", None)
                )
            for link in getattr(article, "topics", []):
                if link.topic is not None and link.topic.slug not in topics:
                    topics.append(link.topic.slug)
            for link in getattr(article, "entities", []):
                entity = link.entity
                if entity is None or entity.normalized_name in seen_entities:
                    continue
                seen_entities.add(entity.normalized_name)
                entities.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "salience": round(float(link.salience or 0.0), 4),
                    }
                )

        similarities = dict(zip(cluster.article_ids, cluster.similarities, strict=False))
        now = utcnow()
        return await self.events.upsert_cluster(
            event_key=cluster.key,
            title=cluster.title,
            summary=anchor_summary,
            first_seen=cluster.first_seen or now,
            last_updated=cluster.last_updated or now,
            article_ids=cluster.article_ids,
            similarities=similarities,
            sources=sorted(cluster.sources),
            topics=topics,
            keywords=cluster.keywords,
            entities=entities[:30],
            avg_sentiment=cluster.avg_sentiment,
            importance=cluster.importance,
            centroid=dict(list(cluster.centroid_vector().items())[:80]),
        )

    async def recent(self, *, limit: int = 20, offset: int = 0) -> tuple[list[Event], int]:
        events, total = await self.events.recent(limit=limit, offset=offset)
        return list(events), total

    async def breaking(self, *, hours: int = 6, limit: int = 10) -> list[Event]:
        return list(await self.events.breaking(hours=hours, limit=limit))


__all__ = ["EventService"]
