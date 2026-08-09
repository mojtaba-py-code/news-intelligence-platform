"""Analytics service: dashboard aggregates, cached where it is safe to cache."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import CacheBackend, get_cache, make_key
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import cache_hits_total, cache_misses_total
from app.core.utils import utcnow
from app.database.models.article import SentimentLabel
from app.database.models.taxonomy import TrendSubject
from app.database.repositories.article import ArticleRepository
from app.database.repositories.event import EventRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import TrendRepository
from app.schemas.intelligence import (
    AnalyticsOverview,
    CountBucket,
    SentimentBreakdown,
    TimeSeriesPoint,
)
from app.schemas.source import SourceStats

logger = get_logger(__name__)

CACHE_NAMESPACE = "analytics"


class AnalyticsService:
    """Read-only aggregates for the API and the dashboard.

    Only derived, non-personal data is cached: overview counters and
    distributions. Anything user-specific is computed per request.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        config: Settings | None = None,
        cache: CacheBackend | None = None,
    ) -> None:
        self.session = session
        self.config = config or get_settings()
        self._cache = cache
        self.articles = ArticleRepository(session)
        self.sources = SourceRepository(session)
        self.events = EventRepository(session)
        self.trends = TrendRepository(session)

    @property
    def cache(self) -> CacheBackend:
        return self._cache if self._cache is not None else get_cache()

    async def overview(self, *, use_cache: bool = True) -> AnalyticsOverview:
        """Headline numbers for the dashboard."""
        key = make_key(CACHE_NAMESPACE, "overview")
        if use_cache:
            cached = await self.cache.get(key)
            if cached:
                cache_hits_total.inc(labels={"key": "overview"})
                return AnalyticsOverview.model_validate(cached)
            cache_misses_total.inc(labels={"key": "overview"})

        stats = await self.articles.stats()
        all_sources = await self.sources.all_sources()
        active = [source for source in all_sources if source.is_operational]
        events = await self.events.count_recent(hours=24)
        trending = await self.trends.latest(subject_type=TrendSubject.TOPIC, limit=8)
        sentiment = await self.sentiment(hours=24)

        overview = AnalyticsOverview(
            total_articles=stats.total,
            articles_today=stats.today,
            articles_24h=stats.last_24h,
            active_sources=len(active),
            total_sources=len(all_sources),
            detected_events=events,
            duplicate_rate=round(stats.duplicates / stats.total, 4) if stats.total else 0.0,
            avg_relevance=stats.avg_relevance,
            trending_topics=[
                CountBucket(
                    key=trend.subject_key,
                    label=trend.subject_label,
                    count=trend.current_count,
                    percentage=round(trend.growth_percent, 2),
                )
                for trend in trending
            ],
            sentiment=sentiment,
            generated_at=utcnow(),
        )
        if use_cache:
            await self.cache.set(
                key, overview.model_dump(mode="json"), ttl=self.config.cache_ttl_seconds
            )
        return overview

    async def sentiment(self, *, hours: int = 24) -> SentimentBreakdown:
        """Distribution of sentiment labels over the window."""
        counts = await self.articles.sentiment_breakdown(hours=hours)
        total = sum(counts.values())
        weights = {
            str(SentimentLabel.VERY_NEGATIVE): -1.0,
            str(SentimentLabel.NEGATIVE): -0.5,
            str(SentimentLabel.NEUTRAL): 0.0,
            str(SentimentLabel.POSITIVE): 0.5,
            str(SentimentLabel.VERY_POSITIVE): 1.0,
        }
        average = (
            sum(weights.get(label, 0.0) * count for label, count in counts.items()) / total
            if total
            else 0.0
        )
        return SentimentBreakdown(
            very_negative=counts.get(str(SentimentLabel.VERY_NEGATIVE), 0),
            negative=counts.get(str(SentimentLabel.NEGATIVE), 0),
            neutral=counts.get(str(SentimentLabel.NEUTRAL), 0),
            positive=counts.get(str(SentimentLabel.POSITIVE), 0),
            very_positive=counts.get(str(SentimentLabel.VERY_POSITIVE), 0),
            average_score=round(average, 4),
        )

    async def topics(self, *, hours: int = 24, limit: int = 15) -> list[CountBucket]:
        """Category distribution over the window."""
        rows = await self.articles.category_breakdown(hours=hours, limit=limit)
        total = sum(count for _, count in rows) or 1
        return [
            CountBucket(
                key=category,
                label=category.replace("-", " ").title(),
                count=count,
                percentage=round(100 * count / total, 2),
            )
            for category, count in rows
        ]

    async def timeseries(self, *, hours: int = 24) -> list[TimeSeriesPoint]:
        rows = await self.articles.volume_series(hours=hours)
        return [
            TimeSeriesPoint(timestamp=bucket, count=count, avg_sentiment=sentiment)
            for bucket, count, sentiment in rows
        ]

    async def source_stats(self, *, hours: int = 24) -> list[SourceStats]:
        """Per-source comparison table.

        Three bulk queries - one for sources, one for recent volume, one for
        success rates - instead of a query per source.
        """
        sources = await self.sources.all_sources()
        volumes = await self.articles.source_totals(hours=hours)
        success = await self.sources.success_rates(days=7)

        stats: list[SourceStats] = []
        for source in sources:
            duplicate_rate = 0.0
            stats.append(
                SourceStats(
                    slug=source.slug,
                    name=source.name,
                    total_articles=source.total_articles,
                    articles_24h=volumes.get(source.id, 0),
                    reliability_score=round(source.reliability_score, 4),
                    success_rate=success.get(source.id, 0.0),
                    duplicate_rate=duplicate_rate,
                )
            )
        stats.sort(key=lambda item: item.articles_24h, reverse=True)
        return stats

    async def invalidate(self) -> None:
        """Drop cached aggregates after an ingestion run."""
        await self.cache.clear_namespace(CACHE_NAMESPACE)


__all__ = ["AnalyticsService"]
