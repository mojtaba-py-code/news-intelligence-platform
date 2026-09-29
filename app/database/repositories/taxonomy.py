"""Topic, entity and trend persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select, update

from app.core.utils import utcnow
from app.database.models.article import ArticleEntity, ArticleTopic
from app.database.models.taxonomy import Entity, EntityType, Topic, TrendSnapshot, TrendSubject
from app.database.repositories.base import BaseRepository, affected_rows
from app.intelligence.entities import normalize_entity_name


class TopicRepository(BaseRepository[Topic]):
    """Topic catalogue - the classifier's configuration lives here."""

    model = Topic

    async def get_by_slug(self, slug: str) -> Topic | None:
        return await self.get_by(slug=slug.lower())

    async def active(self) -> Sequence[Topic]:
        statement = select(Topic).where(Topic.is_active.is_(True)).order_by(Topic.name)
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def slug_to_id(self) -> dict[str, int]:
        """Lookup table so the pipeline resolves topics without a query per article."""
        result = await self.session.execute(select(Topic.slug, Topic.id))
        return {str(slug): int(topic_id) for slug, topic_id in result.all()}

    async def seed(self, rows: list[dict[str, Any]]) -> int:
        """Insert any missing default topics. Idempotent."""
        existing = set((await self.slug_to_id()).keys())
        created = 0
        for row in rows:
            if row["slug"] in existing:
                continue
            self.session.add(
                Topic(
                    slug=row["slug"],
                    name=row["name"],
                    keywords=list(row.get("keywords") or []),
                    parent_slug=row.get("parent_slug"),
                )
            )
            created += 1
        if created:
            await self.flush()
        return created

    async def refresh_counts(self) -> None:
        """Recompute denormalised article counts (cheap, runs after processing)."""
        statement = select(ArticleTopic.topic_id, func.count(ArticleTopic.article_id)).group_by(
            ArticleTopic.topic_id
        )
        counts = {
            int(topic_id): int(count)
            for topic_id, count in (await self.session.execute(statement)).all()
        }
        for topic_id, count in counts.items():
            await self.session.execute(
                update(Topic).where(Topic.id == topic_id).values(article_count=count)
            )

    async def top(self, *, limit: int = 15) -> list[tuple[Topic, int]]:
        statement = (
            select(Topic, Topic.article_count)
            .where(Topic.is_active.is_(True))
            .order_by(Topic.article_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(statement)
        return [(topic, int(count)) for topic, count in result.all()]


class EntityRepository(BaseRepository[Entity]):
    """Named-entity storage and the knowledge-graph queries."""

    model = Entity

    async def get_or_create(
        self, name: str, entity_type: EntityType, *, seen_at: datetime | None = None
    ) -> Entity:
        """Fetch an entity by its normalised key, creating it when new."""
        normalized = normalize_entity_name(name)
        if not normalized:
            raise ValueError("entity name normalises to an empty string")

        existing = await self.get_by(normalized_name=normalized, entity_type=str(entity_type))
        now = seen_at or utcnow()
        if existing is not None:
            existing.last_seen_at = now
            return existing

        entity = Entity(
            name=name[:200],
            normalized_name=normalized[:200],
            entity_type=str(entity_type),
            first_seen_at=now,
            last_seen_at=now,
        )
        await self.add(entity)
        return entity

    async def bulk_resolve(
        self, pairs: list[tuple[str, EntityType]], *, seen_at: datetime | None = None
    ) -> dict[str, Entity]:
        """Resolve many entities with **one** SELECT instead of one per entity."""
        if not pairs:
            return {}
        now = seen_at or utcnow()
        wanted = {
            f"{normalize_entity_name(name)}|{entity_type}": (name, entity_type)
            for name, entity_type in pairs
            if normalize_entity_name(name)
        }
        if not wanted:
            return {}

        normalized_names = list({key.split("|")[0] for key in wanted})
        statement = select(Entity).where(Entity.normalized_name.in_(normalized_names))
        found = {
            f"{entity.normalized_name}|{entity.entity_type}": entity
            for entity in (await self.session.execute(statement)).scalars().all()
        }

        resolved: dict[str, Entity] = {}
        for key, (name, entity_type) in wanted.items():
            entity = found.get(key)
            if entity is None:
                entity = Entity(
                    name=name[:200],
                    normalized_name=key.split("|")[0][:200],
                    entity_type=str(entity_type),
                    first_seen_at=now,
                    last_seen_at=now,
                )
                self.session.add(entity)
                found[key] = entity
            else:
                entity.last_seen_at = now
            resolved[key] = entity

        await self.flush()
        return resolved

    async def top(
        self, *, entity_type: EntityType | None = None, limit: int = 20
    ) -> Sequence[Entity]:
        statement = select(Entity).order_by(Entity.mention_count.desc()).limit(min(limit, 200))
        if entity_type is not None:
            statement = statement.where(Entity.entity_type == str(entity_type))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def search(self, term: str, *, limit: int = 20) -> Sequence[Entity]:
        pattern = f"%{term.lower().replace('%', '')}%"
        statement = (
            select(Entity)
            .where(Entity.normalized_name.ilike(pattern))
            .order_by(Entity.mention_count.desc())
            .limit(min(limit, 100))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def refresh_counts(self) -> None:
        statement = select(
            ArticleEntity.entity_id,
            func.count(ArticleEntity.article_id),
            func.sum(ArticleEntity.mentions),
        ).group_by(ArticleEntity.entity_id)
        for entity_id, articles, mentions in (await self.session.execute(statement)).all():
            await self.session.execute(
                update(Entity)
                .where(Entity.id == int(entity_id))
                .values(article_count=int(articles or 0), mention_count=int(mentions or 0))
            )

    async def co_occurrences(self, entity_id: int, *, limit: int = 20) -> list[tuple[Entity, int]]:
        """Entities appearing in the same articles - the knowledge-graph edges."""
        article_ids = select(ArticleEntity.article_id).where(ArticleEntity.entity_id == entity_id)
        statement = (
            select(Entity, func.count(ArticleEntity.article_id).label("shared"))
            .join(ArticleEntity, ArticleEntity.entity_id == Entity.id)
            .where(ArticleEntity.article_id.in_(article_ids))
            .where(Entity.id != entity_id)
            .group_by(Entity.id)
            .order_by(func.count(ArticleEntity.article_id).desc())
            .limit(min(limit, 100))
        )
        result = await self.session.execute(statement)
        return [(entity, int(shared)) for entity, shared in result.all()]


class TrendRepository(BaseRepository[TrendSnapshot]):
    """Trend snapshot storage."""

    model = TrendSnapshot

    async def record(self, snapshots: list[TrendSnapshot]) -> int:
        """Insert snapshots, skipping any that already exist for the window."""
        if not snapshots:
            return 0
        window_start = snapshots[0].window_start
        existing = await self.session.execute(
            select(TrendSnapshot.subject_type, TrendSnapshot.subject_key).where(
                TrendSnapshot.window_start == window_start
            )
        )
        seen = {(str(a), str(b)) for a, b in existing.all()}
        fresh = [
            snapshot
            for snapshot in snapshots
            if (str(snapshot.subject_type), str(snapshot.subject_key)) not in seen
        ]
        if fresh:
            self.session.add_all(fresh)
            await self.flush()
        return len(fresh)

    async def latest(
        self,
        *,
        subject_type: TrendSubject | None = None,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> Sequence[TrendSnapshot]:
        """Most recent window's trends, highest score first."""
        latest_window = await self.session.execute(select(func.max(TrendSnapshot.window_start)))
        window_start = latest_window.scalar_one_or_none()
        if window_start is None:
            return []

        statement = (
            select(TrendSnapshot)
            .where(TrendSnapshot.window_start == window_start)
            .where(TrendSnapshot.trend_score >= min_score)
            .order_by(TrendSnapshot.trend_score.desc())
            .limit(min(limit, 200))
        )
        if subject_type is not None:
            statement = statement.where(TrendSnapshot.subject_type == str(subject_type))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def history(
        self, subject_type: TrendSubject, subject_key: str, *, limit: int = 48
    ) -> Sequence[TrendSnapshot]:
        statement = (
            select(TrendSnapshot)
            .where(TrendSnapshot.subject_type == str(subject_type))
            .where(TrendSnapshot.subject_key == subject_key)
            .order_by(TrendSnapshot.window_start.desc())
            .limit(min(limit, 500))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def prune(self, *, days: int = 30) -> int:
        from datetime import timedelta

        cutoff = utcnow() - timedelta(days=days)
        result = await self.session.execute(
            delete(TrendSnapshot).where(TrendSnapshot.window_start < cutoff)
        )
        return affected_rows(result)


__all__ = ["EntityRepository", "TopicRepository", "TrendRepository"]
