"""Article queries: search, filtering, statistics and duplicate lookup."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Select, and_, case, func, or_, select, update
from sqlalchemy import delete as sql_delete
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement

from app.core.metrics import db_query_duration_seconds
from app.core.utils import hours_ago, utcnow
from app.database.models.article import (
    Article,
    ArticleEntity,
    ArticleTopic,
    ProcessingStatus,
)
from app.database.models.source import Source
from app.database.models.taxonomy import Entity, Topic
from app.database.repositories.base import BaseRepository
from app.schemas.article import ArticleSearchQuery, ArticleSortField, ArticleStats
from app.schemas.common import SortOrder

#: Sort key -> column. An allowlist is the only injection-safe way to expose
#: client-chosen ordering, since ORDER BY cannot be parameterised.
SORT_COLUMNS: dict[str, Any] = {
    ArticleSortField.PUBLISHED_AT: Article.published_at,
    ArticleSortField.RELEVANCE: Article.relevance_score,
    ArticleSortField.SENTIMENT: Article.sentiment_score,
    ArticleSortField.QUALITY: Article.quality_score,
    ArticleSortField.CREATED_AT: Article.created_at,
    ArticleSortField.WORD_COUNT: Article.word_count,
}


class ArticleRepository(BaseRepository[Article]):
    """All article reads and writes."""

    model = Article

    # ------------------------------------------------------------------ reads
    async def get_detail(self, article_id: int) -> Article | None:
        """Load one article with topics/entities eagerly - avoids N+1 in the API."""
        statement = (
            select(Article)
            .where(Article.id == article_id)
            .options(
                selectinload(Article.topics).selectinload(ArticleTopic.topic),
                selectinload(Article.entities).selectinload(ArticleEntity.entity),
                selectinload(Article.source),
            )
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_canonical_url(self, canonical_url: str) -> Article | None:
        return await self.get_by(canonical_url=canonical_url)

    async def get_by_content_hash(self, content_hash: str) -> Article | None:
        return await self.get_by(content_hash=content_hash)

    async def search(
        self, query: ArticleSearchQuery, *, limit: int, offset: int
    ) -> tuple[Sequence[Article], int]:
        """Filtered, sorted, paginated search. Returns ``(rows, total)``."""
        statement = self._apply_filters(select(Article), query)

        total = await self.count(statement)
        if total == 0:
            return [], 0

        column = SORT_COLUMNS.get(query.sort_by, Article.published_at)
        ordering = column.desc() if query.order is SortOrder.DESC else column.asc()

        statement = (
            statement.order_by(ordering, Article.id.desc())
            .limit(max(1, min(limit, 200)))
            .offset(max(0, offset))
            .options(selectinload(Article.source))
        )
        with db_query_duration_seconds.time(labels={"op": "search", "model": "Article"}):
            result = await self.session.execute(statement)
        return result.scalars().unique().all(), total

    def _apply_filters(self, statement: Select[Any], query: ArticleSearchQuery) -> Select[Any]:
        """Translate a validated query object into SQL predicates."""
        conditions: list[ColumnElement[bool]] = []

        if not query.include_duplicates:
            conditions.append(Article.is_duplicate.is_(False))

        if query.q:
            # ILIKE-style matching with escaped wildcards. PostgreSQL
            # deployments should add a GIN index / tsvector column; this keeps
            # the same API working on SQLite for development and tests.
            pattern = f"%{_escape_like(query.q)}%"
            conditions.append(
                or_(
                    Article.title.ilike(pattern, escape="\\"),
                    Article.description.ilike(pattern, escape="\\"),
                    Article.content.ilike(pattern, escape="\\"),
                )
            )
        if query.phrase:
            pattern = f"%{_escape_like(query.phrase)}%"
            conditions.append(
                or_(
                    Article.title.ilike(pattern, escape="\\"),
                    Article.content.ilike(pattern, escape="\\"),
                )
            )
        if query.source_id:
            conditions.append(Article.source_id == query.source_id)
        if query.source:
            statement = statement.join(Source, Source.id == Article.source_id)
            conditions.append(Source.slug == query.source.lower())
        if query.category:
            conditions.append(Article.category == query.category.lower())
        if query.language:
            conditions.append(Article.language == query.language.lower())
        if query.country:
            conditions.append(Article.country == query.country.upper())
        if query.author:
            conditions.append(
                Article.author_name.ilike(f"%{_escape_like(query.author)}%", escape="\\")
            )
        if query.sentiment:
            conditions.append(Article.sentiment_label == str(query.sentiment))
        if query.min_sentiment is not None:
            conditions.append(Article.sentiment_score >= query.min_sentiment)
        if query.max_sentiment is not None:
            conditions.append(Article.sentiment_score <= query.max_sentiment)
        if query.min_relevance is not None:
            conditions.append(Article.relevance_score >= query.min_relevance)
        if query.published_after:
            conditions.append(Article.published_at >= query.published_after)
        if query.published_before:
            conditions.append(Article.published_at <= query.published_before)

        if query.topic:
            topic_filter = (
                select(ArticleTopic.article_id)
                .join(Topic, Topic.id == ArticleTopic.topic_id)
                .where(Topic.slug == query.topic.lower())
            )
            conditions.append(Article.id.in_(topic_filter))
        if query.entity:
            entity_filter = (
                select(ArticleEntity.article_id)
                .join(Entity, Entity.id == ArticleEntity.entity_id)
                .where(
                    Entity.normalized_name.ilike(
                        f"%{_escape_like(query.entity.lower())}%", escape="\\"
                    )
                )
            )
            conditions.append(Article.id.in_(entity_filter))

        return statement.where(and_(*conditions)) if conditions else statement

    async def recent_candidates(
        self, *, hours: int, limit: int = 2000, source_id: int | None = None
    ) -> Sequence[Article]:
        """Candidate window for deduplication - one indexed query per batch."""
        statement = (
            select(Article)
            .where(Article.published_at >= hours_ago(hours))
            .where(Article.is_duplicate.is_(False))
            .order_by(Article.published_at.desc())
            .limit(min(limit, 5000))
        )
        if source_id is not None:
            statement = statement.where(Article.source_id != source_id)
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def pending(self, *, limit: int = 100) -> Sequence[Article]:
        statement = (
            select(Article)
            .where(Article.status == ProcessingStatus.PENDING)
            .order_by(Article.created_at.asc())
            .limit(min(limit, 1000))
            .options(selectinload(Article.source))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def for_clustering(self, *, hours: int, limit: int = 800) -> Sequence[Article]:
        """Articles for event clustering.

        Topics, entities and the source are eager-loaded: the clustering
        service aggregates them per cluster, and a lazy load inside async code
        raises ``MissingGreenlet``.
        """
        statement = (
            select(Article)
            .where(Article.published_at >= hours_ago(hours))
            .where(Article.is_duplicate.is_(False))
            .where(Article.status == ProcessingStatus.PROCESSED)
            .order_by(Article.published_at.desc())
            .limit(min(limit, 3000))
            .options(
                selectinload(Article.topics).selectinload(ArticleTopic.topic),
                selectinload(Article.entities).selectinload(ArticleEntity.entity),
                selectinload(Article.source),
            )
        )
        result = await self.session.execute(statement)
        return result.scalars().unique().all()

    async def in_window(
        self, start: datetime, end: datetime, *, limit: int = 5000
    ) -> Sequence[Article]:
        """Articles published in ``[start, end)`` - the trend-detection input."""
        statement = (
            select(Article)
            .where(Article.published_at >= start, Article.published_at < end)
            .where(Article.is_duplicate.is_(False))
            .order_by(Article.published_at.desc())
            .limit(min(limit, 20_000))
            .options(
                selectinload(Article.topics).selectinload(ArticleTopic.topic),
                selectinload(Article.entities).selectinload(ArticleEntity.entity),
                selectinload(Article.source),
            )
        )
        result = await self.session.execute(statement)
        return result.scalars().unique().all()

    async def similar_to(self, article: Article, *, limit: int = 5) -> Sequence[Article]:
        """Cheap "related articles": shared topics, recent, excluding duplicates.

        The topic ids are queried rather than read off ``article.topics``: the
        caller may have loaded the row without eager-loading relationships, and
        a lazy load inside async code raises ``MissingGreenlet``.
        """
        topic_ids = list(
            (
                await self.session.execute(
                    select(ArticleTopic.topic_id).where(ArticleTopic.article_id == article.id)
                )
            )
            .scalars()
            .all()
        )
        statement = (
            select(Article)
            .where(Article.id != article.id)
            .where(Article.is_duplicate.is_(False))
            .order_by(Article.published_at.desc())
            .limit(min(limit, 50))
        )
        if topic_ids:
            statement = statement.where(
                Article.id.in_(
                    select(ArticleTopic.article_id).where(ArticleTopic.topic_id.in_(topic_ids))
                )
            )
        elif article.category:
            statement = statement.where(Article.category == article.category)
        result = await self.session.execute(statement)
        return result.scalars().all()

    # ------------------------------------------------------------- statistics
    async def stats(self) -> ArticleStats:
        """One aggregate query rather than eight counts."""
        now = utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        statement = select(
            func.count(Article.id),
            func.sum(_case_when(Article.published_at >= day_start)),
            func.sum(_case_when(Article.published_at >= hours_ago(24))),
            func.sum(_case_when(Article.published_at >= hours_ago(24 * 7))),
            func.sum(_case_when(Article.is_duplicate.is_(True))),
            func.sum(_case_when(Article.status == ProcessingStatus.PENDING)),
            func.sum(_case_when(Article.status == ProcessingStatus.FAILED)),
            func.avg(Article.relevance_score),
            func.avg(Article.sentiment_score),
        )
        row = (await self.session.execute(statement)).one()
        return ArticleStats(
            total=int(row[0] or 0),
            today=int(row[1] or 0),
            last_24h=int(row[2] or 0),
            last_7d=int(row[3] or 0),
            duplicates=int(row[4] or 0),
            pending=int(row[5] or 0),
            failed=int(row[6] or 0),
            avg_relevance=round(float(row[7] or 0.0), 4),
            avg_sentiment=round(float(row[8] or 0.0), 4),
        )

    async def sentiment_breakdown(self, *, hours: int = 24) -> dict[str, int]:
        statement = (
            select(Article.sentiment_label, func.count(Article.id))
            .where(Article.published_at >= hours_ago(hours))
            .where(Article.is_duplicate.is_(False))
            .group_by(Article.sentiment_label)
        )
        result = await self.session.execute(statement)
        return {str(label): int(count) for label, count in result.all()}

    async def category_breakdown(
        self, *, hours: int = 24, limit: int = 15
    ) -> list[tuple[str, int]]:
        statement = (
            select(Article.category, func.count(Article.id).label("total"))
            .where(Article.published_at >= hours_ago(hours))
            .where(Article.is_duplicate.is_(False))
            .where(Article.category.is_not(None))
            .group_by(Article.category)
            .order_by(func.count(Article.id).desc())
            .limit(limit)
        )
        result = await self.session.execute(statement)
        return [(str(category), int(count)) for category, count in result.all()]

    async def volume_series(self, *, hours: int = 24) -> list[tuple[datetime, int, float]]:
        """Hourly article volume and mean sentiment for the dashboard chart."""
        statement = (
            select(Article.published_at, Article.sentiment_score)
            .where(Article.published_at >= hours_ago(hours))
            .where(Article.is_duplicate.is_(False))
        )
        result = await self.session.execute(statement)
        buckets: dict[datetime, list[float]] = {}
        for published_at, sentiment in result.all():
            if published_at is None:
                continue
            bucket = published_at.replace(minute=0, second=0, microsecond=0)
            buckets.setdefault(bucket, []).append(float(sentiment or 0.0))
        return [
            (bucket, len(values), round(sum(values) / len(values), 4))
            for bucket, values in sorted(buckets.items())
        ]

    async def source_totals(self, *, hours: int = 24) -> dict[int, int]:
        statement = (
            select(Article.source_id, func.count(Article.id))
            .where(Article.published_at >= hours_ago(hours))
            .group_by(Article.source_id)
        )
        result = await self.session.execute(statement)
        return {int(source_id): int(count) for source_id, count in result.all()}

    # ----------------------------------------------------------------- writes
    async def mark_processed(
        self,
        article_id: int,
        *,
        status: ProcessingStatus = ProcessingStatus.PROCESSED,
        error: str | None = None,
    ) -> None:
        await self.session.execute(
            update(Article)
            .where(Article.id == article_id)
            .values(status=str(status), processed_at=utcnow(), processing_error=error)
            # "fetch" keeps objects already loaded in this session consistent
            # with the row we just wrote; the default heuristic does not.
            .execution_options(synchronize_session="fetch")
        )

    async def mark_duplicate(
        self, article_id: int, *, original_id: int, score: float, method: str
    ) -> None:
        await self.session.execute(
            update(Article)
            .where(Article.id == article_id)
            .values(
                is_duplicate=True,
                duplicate_of_id=original_id,
                duplicate_score=score,
                duplicate_method=method,
                status=str(ProcessingStatus.PROCESSED),
                processed_at=utcnow(),
            )
            .execution_options(synchronize_session="fetch")
        )

    async def delete_older_than(self, *, days: int) -> int:
        """Retention cleanup. Returns the number of rows removed."""
        cutoff = utcnow() - timedelta(days=days)
        statement = select(Article.id).where(Article.published_at < cutoff).limit(5000)
        ids = list((await self.session.execute(statement)).scalars().all())
        if not ids:
            return 0
        await self.session.execute(sql_delete(Article).where(Article.id.in_(ids)))
        return len(ids)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so a search for ``100%`` is not a prefix match."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _case_when(condition: Any) -> Any:
    """``SUM(CASE WHEN cond THEN 1 ELSE 0 END)`` - portable conditional count."""
    return case((condition, 1), else_=0)


__all__ = ["SORT_COLUMNS", "ArticleRepository"]
