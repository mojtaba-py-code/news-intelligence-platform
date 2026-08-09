"""Event-cluster persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from app.core.utils import utcnow
from app.database.models.event import Event, EventArticle
from app.database.repositories.base import BaseRepository


class EventRepository(BaseRepository[Event]):
    """Reads and writes for detected story clusters."""

    model = Event

    async def get_by_key(self, event_key: str) -> Event | None:
        return await self.get_by(event_key=event_key)

    async def get_detail(self, event_id: int) -> Event | None:
        statement = (
            select(Event)
            .where(Event.id == event_id)
            .options(selectinload(Event.articles).selectinload(EventArticle.article))
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def recent(
        self, *, limit: int = 20, offset: int = 0, min_sources: int = 2
    ) -> tuple[Sequence[Event], int]:
        base = select(Event).where(Event.source_count >= min_sources)
        total = await self.count(base)
        statement = (
            base.order_by(Event.importance.desc(), Event.last_updated_at.desc())
            .limit(max(1, min(limit, 100)))
            .offset(max(0, offset))
        )
        result = await self.session.execute(statement)
        return result.scalars().all(), total

    async def breaking(self, *, hours: int = 6, limit: int = 10) -> Sequence[Event]:
        cutoff = utcnow() - timedelta(hours=hours)
        statement = (
            select(Event)
            .where(Event.first_seen_at >= cutoff)
            .where(Event.source_count >= 3)
            .order_by(Event.importance.desc())
            .limit(min(limit, 50))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def upsert_cluster(
        self,
        *,
        event_key: str,
        title: str,
        summary: str | None,
        first_seen: datetime,
        last_updated: datetime,
        article_ids: list[int],
        similarities: dict[int, float],
        sources: list[str],
        topics: list[str],
        keywords: list[str],
        entities: list[dict[str, object]],
        avg_sentiment: float,
        importance: float,
        centroid: dict[str, object],
    ) -> Event:
        """Create or update one cluster and its membership rows.

        Idempotent: re-running clustering over an overlapping window updates the
        existing event instead of creating a near-duplicate.
        """
        event = await self.get_by_key(event_key)
        if event is None:
            event = Event(
                event_key=event_key, first_seen_at=first_seen, last_updated_at=last_updated
            )
            self.session.add(event)

        event.title = title[:512]
        event.summary = summary
        event.first_seen_at = (
            min(event.first_seen_at, first_seen) if event.first_seen_at else first_seen
        )
        event.last_updated_at = (
            max(event.last_updated_at, last_updated) if event.last_updated_at else last_updated
        )
        event.article_count = len(article_ids)
        event.source_count = len(set(sources))
        event.sources = sorted(set(sources))[:50]
        event.topics = topics[:20]
        event.keywords = keywords[:20]
        event.entities = entities[:30]
        event.avg_sentiment = round(avg_sentiment, 4)
        event.importance = round(importance, 4)
        event.centroid = centroid
        await self.flush()

        existing = await self.session.execute(
            select(EventArticle.article_id).where(EventArticle.event_id == event.id)
        )
        known = {int(article_id) for article_id in existing.scalars().all()}
        now = utcnow()
        new_links = [
            EventArticle(
                event_id=event.id,
                article_id=article_id,
                similarity=round(similarities.get(article_id, 0.0), 4),
                added_at=now,
            )
            for article_id in article_ids
            if article_id not in known
        ]
        if new_links:
            self.session.add_all(new_links)
            await self.flush()
        return event

    async def count_recent(self, *, hours: int = 24) -> int:
        cutoff = utcnow() - timedelta(hours=hours)
        statement = select(func.count(Event.id)).where(Event.last_updated_at >= cutoff)
        return int((await self.session.execute(statement)).scalar_one() or 0)

    async def prune(self, *, days: int = 60) -> int:
        cutoff = utcnow() - timedelta(days=days)
        result = await self.session.execute(delete(Event).where(Event.last_updated_at < cutoff))
        return int(result.rowcount or 0)


__all__ = ["EventRepository"]
