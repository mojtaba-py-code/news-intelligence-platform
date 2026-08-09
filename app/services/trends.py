"""Trend service: turn stored articles into scored, persisted trend snapshots."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.utils import utcnow
from app.database.models.article import Article
from app.database.models.taxonomy import TrendSnapshot, TrendSubject
from app.database.repositories.article import ArticleRepository
from app.database.repositories.taxonomy import TrendRepository
from app.intelligence.trends import (
    SubjectObservation,
    TrendResult,
    detect_trends,
    window_bounds,
)

logger = get_logger(__name__)

#: Keywords per article that participate in trend counting.
KEYWORDS_PER_ARTICLE = 5


class TrendService:
    """Computes trends for the current window and stores the snapshots."""

    def __init__(self, session: AsyncSession, *, config: Settings | None = None) -> None:
        self.session = session
        self.config = config or get_settings()
        self.articles = ArticleRepository(session)
        self.trends = TrendRepository(session)

    async def compute(
        self, *, hours: int | None = None, now: datetime | None = None, persist: bool = True
    ) -> list[TrendResult]:
        """Compare the last ``hours`` against the preceding equal window."""
        hours = hours or self.config.trend_window_hours
        reference = now or utcnow()
        (current_start, current_end), (previous_start, previous_end) = window_bounds(
            hours=hours, now=reference
        )

        current_articles = await self.articles.in_window(current_start, current_end)
        previous_articles = await self.articles.in_window(previous_start, previous_end)

        results = detect_trends(
            self._observations(current_articles),
            self._observations(previous_articles),
            min_articles=self.config.trend_min_articles,
        )

        if persist and results:
            snapshots = [
                TrendSnapshot(
                    subject_type=str(result.subject_type),
                    subject_key=result.subject_key[:160],
                    subject_label=result.subject_label[:200],
                    window_start=current_start,
                    window_end=current_end,
                    window_hours=hours,
                    current_count=result.current_count,
                    previous_count=result.previous_count,
                    growth_percent=result.growth_percent,
                    trend_score=result.trend_score,
                    direction=str(result.direction),
                    confidence=result.confidence,
                    avg_sentiment=result.avg_sentiment,
                    sentiment_delta=result.sentiment_delta,
                    source_count=result.source_count,
                    sample_article_ids=[str(i) for i in result.sample_article_ids],
                    created_at=reference,
                )
                for result in results
            ]
            stored = await self.trends.record(snapshots)
            logger.info(
                "trends_computed",
                extra={"window_hours": hours, "detected": len(results), "stored": stored},
            )
        return results

    @staticmethod
    def _observations(articles: list[Article] | object) -> list[SubjectObservation]:
        """Explode articles into one observation per topic/keyword/entity/source."""
        observations: list[SubjectObservation] = []
        for article in articles:  # type: ignore[union-attr]
            source_slug = article.source.slug if article.source else str(article.source_id)
            sentiment = float(article.sentiment_score or 0.0)

            for link in article.topics:
                if link.topic is None:
                    continue
                observations.append(
                    SubjectObservation(
                        subject_type=TrendSubject.TOPIC,
                        key=link.topic.slug,
                        label=link.topic.name,
                        source_slug=source_slug,
                        sentiment=sentiment,
                        article_id=article.id,
                    )
                )

            for keyword in (article.keywords or [])[:KEYWORDS_PER_ARTICLE]:
                term = str(keyword).strip().casefold()
                if len(term) < 3:
                    continue
                observations.append(
                    SubjectObservation(
                        subject_type=TrendSubject.KEYWORD,
                        key=term[:160],
                        label=str(keyword)[:200],
                        source_slug=source_slug,
                        sentiment=sentiment,
                        article_id=article.id,
                    )
                )

            for link in article.entities:
                if link.entity is None:
                    continue
                observations.append(
                    SubjectObservation(
                        subject_type=TrendSubject.ENTITY,
                        key=link.entity.normalized_name[:160],
                        label=link.entity.name[:200],
                        source_slug=source_slug,
                        sentiment=sentiment,
                        article_id=article.id,
                    )
                )

            observations.append(
                SubjectObservation(
                    subject_type=TrendSubject.SOURCE,
                    key=source_slug,
                    label=article.source_name,
                    source_slug=source_slug,
                    sentiment=sentiment,
                    article_id=article.id,
                )
            )
        return observations

    async def latest(
        self, *, subject_type: TrendSubject | None = None, limit: int = 20
    ) -> list[TrendSnapshot]:
        return list(await self.trends.latest(subject_type=subject_type, limit=limit))

    async def breaking(self, *, limit: int = 10) -> list[TrendResult]:
        """Trends that qualify as breaking news right now."""
        results = await self.compute(persist=False, hours=6)
        return [result for result in results if result.is_breaking][:limit]


__all__ = ["TrendService"]
