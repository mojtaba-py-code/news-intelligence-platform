"""The ingestion pipeline.

::

    FETCH → VALIDATE → NORMALIZE → CLEAN → DEDUPLICATE → ENRICH
          → CLASSIFY → STORE → INDEX → ANALYZE

Two properties are non-negotiable:

**Idempotency.** Running the same job twice must not create duplicates. Three
layers guarantee it: the in-memory dedup engine, an explicit lookup by
canonical URL / content hash, and unique constraints in the database that catch
anything that races past the first two.

**Fault isolation.** A source that times out, returns 429, or emits malformed
XML must not affect the others. Each source runs inside its own transaction and
its own error boundary, and its outcome is recorded either way.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import SourceError
from app.core.logging import get_logger
from app.core.metrics import articles_stored_total, processing_duration_seconds
from app.core.utils import utcnow
from app.database.models.article import (
    Article,
    ArticleEntity,
    ArticleTopic,
    Author,
    ProcessingStatus,
)
from app.database.models.source import Source
from app.database.repositories.article import ArticleRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import EntityRepository, TopicRepository
from app.database.session import session_scope
from app.ingestion.sources.base import FetchOutcome, NewsSource, SourceContext
from app.ingestion.sources.registry import build_source
from app.intelligence.entities import normalize_entity_name
from app.intelligence.pipeline import EnrichmentResult, NLPPipeline
from app.intelligence.ranking import InterestProfile, RelevanceScorer
from app.intelligence.topics import TopicClassifier, TopicDefinition
from app.processing.deduplication.engine import (
    DeduplicationEngine,
    DuplicateLevel,
    DuplicateMatch,
)
from app.processing.normalization.normalizer import ArticleNormalizer, NormalizationError
from app.processing.normalization.text import author_key
from app.processing.validation.quality import QualityReportBuilder, validate_article
from app.schemas.article import NormalizedArticle, RawArticle
from app.schemas.source import IngestionResult

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineStats:
    """Per-source counters accumulated during one run."""

    fetched: int = 0
    valid: int = 0
    rejected: int = 0
    duplicates: int = 0
    stored: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "fetched": self.fetched,
            "valid": self.valid,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "stored": self.stored,
            "failed": self.failed,
        }


@dataclass
class IngestionPipeline:
    """Coordinates one ingestion run.

    Constructed per run (it caches topic ids and a dedup window), but the
    engines it delegates to are stateless and shared.
    """

    session: AsyncSession
    config: Settings = field(default_factory=get_settings)
    normalizer: ArticleNormalizer = field(default_factory=ArticleNormalizer)
    nlp: NLPPipeline = field(default_factory=NLPPipeline)
    scorer: RelevanceScorer = field(default_factory=RelevanceScorer)
    _dedup: DeduplicationEngine | None = field(default=None, init=False)
    _topic_ids: dict[str, int] = field(default_factory=dict, init=False)
    _quality: QualityReportBuilder = field(default_factory=QualityReportBuilder, init=False)

    def __post_init__(self) -> None:
        self.articles = ArticleRepository(self.session)
        self.sources = SourceRepository(self.session)
        self.topics = TopicRepository(self.session)
        self.entities = EntityRepository(self.session)

    # ------------------------------------------------------------------ setup
    async def prepare(self, *, source_id: int | None = None) -> None:
        """Load the dedup candidate window and the topic lookup table."""
        self._topic_ids = await self.topics.slug_to_id()
        definitions = await self._topic_definitions()
        if definitions:
            self.nlp.classifier = TopicClassifier.from_definitions(definitions)

        candidates = await self.articles.recent_candidates(
            hours=self.config.dedup_lookback_hours, limit=3000
        )
        self._dedup = DeduplicationEngine.from_settings(self.config).load(list(candidates))
        logger.debug(
            "pipeline_prepared",
            extra={"candidates": len(candidates), "topics": len(self._topic_ids)},
        )

    async def _topic_definitions(self) -> list[TopicDefinition]:
        """Build classifier definitions from the database's topic rows."""
        rows = await self.topics.active()
        definitions: list[TopicDefinition] = []
        for row in rows:
            keywords = frozenset(str(word).casefold() for word in (row.keywords or []))
            if not keywords:
                continue
            definitions.append(
                TopicDefinition(
                    slug=row.slug, name=row.name, keywords=keywords, parent=row.parent_slug
                )
            )
        return definitions

    @property
    def dedup(self) -> DeduplicationEngine:
        if self._dedup is None:
            self._dedup = DeduplicationEngine.from_settings(self.config)
        return self._dedup

    # -------------------------------------------------------------- ingestion
    async def ingest_source(
        self, source: Source, *, connector: NewsSource | None = None
    ) -> IngestionResult:
        """Run the whole pipeline for one source."""
        started = time.perf_counter()
        stats = PipelineStats()
        context = SourceContext.from_model(source)
        engine = connector or build_source(context, config=self.config)

        outcome: FetchOutcome = await engine.fetch()
        stats.fetched = outcome.fetched

        if not outcome.success:
            await self.sources.record_run(
                source,
                success=False,
                status_code=outcome.status_code,
                latency_ms=outcome.duration_ms,
                fetched=outcome.fetched,
                error_type=outcome.error_type,
                error_message=outcome.error,
            )
            return IngestionResult(
                source=source.slug,
                success=False,
                fetched=outcome.fetched,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error=outcome.error,
            )

        for raw in outcome.articles:
            try:
                stored = await self._process_one(raw, source, stats)
            except Exception:
                stats.failed += 1
                logger.exception("article_processing_failed", extra={"source": source.slug})
                continue
            if stored:
                stats.stored += 1

        await self.sources.record_run(
            source,
            success=True,
            status_code=outcome.status_code,
            latency_ms=outcome.duration_ms,
            fetched=stats.fetched,
            valid=stats.stored,
            duplicates=stats.duplicates,
        )
        articles_stored_total.inc(stats.stored, labels={"source": source.slug})

        duration = round((time.perf_counter() - started) * 1000, 2)
        logger.info("source_ingested", extra={"source": source.slug, **stats.as_dict()})
        return IngestionResult(
            source=source.slug,
            success=True,
            fetched=stats.fetched,
            valid=stats.valid,
            rejected=stats.rejected + outcome.rejected,
            duplicates=stats.duplicates,
            stored=stats.stored,
            duration_ms=duration,
        )

    async def _process_one(self, raw: RawArticle, source: Source, stats: PipelineStats) -> bool:
        """NORMALIZE → CLEAN → VALIDATE → DEDUPLICATE → ENRICH → STORE."""
        with processing_duration_seconds.time(labels={"stage": "normalize"}):
            try:
                normalized = self.normalizer.normalize(raw)
            except NormalizationError as exc:
                stats.rejected += 1
                self._quality.record_malformed()
                logger.debug(
                    "article_normalization_failed",
                    extra={"source": source.slug, "reason": str(exc)},
                )
                return False

        outcome = validate_article(normalized)
        self._quality.record(outcome)
        if not outcome.is_valid:
            stats.rejected += 1
            logger.debug(
                "article_rejected",
                extra={"source": source.slug, "reason": outcome.reason()},
            )
            return False
        stats.valid += 1

        match = await self._find_duplicate(normalized)
        if match is not None:
            stats.duplicates += 1
            self._quality.record_duplicate()
            return False

        with processing_duration_seconds.time(labels={"stage": "enrich"}):
            enrichment = self.nlp.enrich(
                title=normalized.title,
                content=normalized.content,
                description=normalized.description,
                language_hint=normalized.language,
            )

        article = await self._persist(normalized, source, enrichment)
        return article is not None

    async def _find_duplicate(self, normalized: NormalizedArticle) -> DuplicateMatch | None:
        """In-memory check first, then an authoritative database lookup."""
        match = self.dedup.find_duplicate(
            canonical_url=normalized.canonical_url,
            content_hash=normalized.content_hash,
            title=normalized.title,
            title_hash=normalized.title_hash,
            content=normalized.content,
            description=normalized.description,
            simhash=normalized.simhash,
        )
        if match is not None:
            return match

        # The candidate window is time-bounded; an older article with the same
        # URL would otherwise slip through and violate the unique constraint.
        existing = await self.articles.get_by_canonical_url(normalized.canonical_url)
        if existing is None:
            existing = await self.articles.get_by_content_hash(normalized.content_hash)
        if existing is not None:
            return DuplicateMatch(existing.id, DuplicateLevel.URL, 1.0)
        return None

    # ------------------------------------------------------------------ store
    async def _persist(
        self, normalized: NormalizedArticle, source: Source, enrichment: EnrichmentResult
    ) -> Article | None:
        """Write the article and its topic/entity links in one transaction."""
        author = await self._resolve_author(normalized.author_name)
        relevance = self.scorer.score(
            published_at=normalized.published_at,
            source_weight=source.weight,
            source_reliability=source.reliability_score,
            source_slug=source.slug,
            quality_score=normalized.quality_score,
            topics=[topic.slug for topic in enrichment.topics],
            keywords=enrichment.keyword_strings,
            entities=[entity.name for entity in enrichment.entities],
            profile=InterestProfile(),
        )

        article = Article(
            source_id=source.id,
            source_name=source.name,
            external_id=normalized.external_id,
            author_id=author.id if author else None,
            author_name=normalized.author_name,
            title=normalized.title,
            description=normalized.description,
            content=normalized.content,
            summary=enrichment.summary,
            url=normalized.url,
            canonical_url=normalized.canonical_url,
            image_url=normalized.image_url,
            published_at=normalized.published_at,
            source_updated_at=normalized.source_updated_at,
            language=enrichment.language or normalized.language,
            country=normalized.country,
            category=normalized.category or enrichment.primary_topic,
            content_hash=normalized.content_hash,
            title_hash=normalized.title_hash,
            simhash=normalized.simhash,
            sentiment_score=enrichment.sentiment_score,
            sentiment_label=str(enrichment.sentiment_label),
            sentiment_confidence=enrichment.sentiment_confidence,
            relevance_score=relevance.score,
            quality_score=normalized.quality_score,
            readability_score=enrichment.readability,
            keywords=enrichment.keyword_strings,
            enrichment=enrichment.metadata(),
            word_count=enrichment.word_count or normalized.word_count,
            status=str(ProcessingStatus.PROCESSED),
            processed_at=utcnow(),
        )

        self.session.add(article)
        try:
            await self.session.flush()
        except IntegrityError:
            # A concurrent worker inserted the same article first. The unique
            # constraint is the final arbiter, and losing this race is normal.
            await self.session.rollback()
            logger.debug("duplicate_insert_rejected", extra={"source": source.slug})
            return None

        await self._link_topics(article, enrichment)
        await self._link_entities(article, enrichment)

        self.dedup.add(
            article.id,
            canonical_url=article.canonical_url,
            content_hash=article.content_hash,
            title=article.title,
            title_hash=article.title_hash,
            text=f"{article.title}\n{(article.content or article.description or '')[:4000]}",
            simhash=article.simhash,
        )
        return article

    async def _resolve_author(self, name: str | None) -> Author | None:
        if not name:
            return None
        key = author_key(name)
        if not key:
            return None
        author = await self.session.scalar(
            select(Author).where(Author.normalized_name == key).limit(1)
        )
        if author is None:
            author = Author(name=name[:200], normalized_name=key[:200], article_count=0)
            self.session.add(author)
            await self.session.flush()
        author.article_count += 1
        return author

    async def _link_topics(self, article: Article, enrichment: EnrichmentResult) -> None:
        links: list[ArticleTopic] = []
        for index, topic in enumerate(enrichment.topics):
            topic_id = self._topic_ids.get(topic.slug)
            if topic_id is None:
                continue
            links.append(
                ArticleTopic(
                    article_id=article.id,
                    topic_id=topic_id,
                    score=topic.score,
                    is_primary=index == 0,
                )
            )
        if links:
            self.session.add_all(links)
            await self.session.flush()

    async def _link_entities(self, article: Article, enrichment: EnrichmentResult) -> None:
        if not enrichment.entities:
            return
        resolved = await self.entities.bulk_resolve(
            [(entity.name, entity.entity_type) for entity in enrichment.entities]
        )
        links: list[ArticleEntity] = []
        seen: set[int] = set()
        for entity in enrichment.entities:
            key = f"{normalize_entity_name(entity.name)}|{entity.entity_type}"
            row = resolved.get(key)
            if row is None or row.id in seen:
                continue
            seen.add(row.id)
            links.append(
                ArticleEntity(
                    article_id=article.id,
                    entity_id=row.id,
                    mentions=entity.mentions,
                    salience=entity.salience,
                )
            )
        if links:
            self.session.add_all(links)
            await self.session.flush()

    # ----------------------------------------------------------------- report
    def quality_report(self) -> Any:
        """Data-quality counters for this run."""
        return self._quality.build()


async def ingest_sources(
    slugs: Sequence[str] | None = None,
    *,
    config: Settings | None = None,
    concurrency: int | None = None,
) -> list[IngestionResult]:
    """Ingest every active source (or the named ones) with bounded concurrency.

    Each source gets its own session and transaction, so a failure - including
    a database error - is contained.
    """
    config = config or get_settings()
    limit = max(1, concurrency or config.worker_concurrency)

    async with session_scope() as session:
        repository = SourceRepository(session)
        if slugs:
            wanted = [await repository.get_by_slug(slug) for slug in slugs]
            targets = [source for source in wanted if source is not None]
            missing = set(slugs) - {source.slug for source in targets}
            if missing:
                logger.warning("unknown_sources_requested", extra={"slugs": sorted(missing)})
        else:
            targets = list(await repository.active())
        source_ids = [source.id for source in targets]

    if not source_ids:
        logger.info("no_sources_to_ingest")
        return []

    semaphore = asyncio.Semaphore(limit)

    async def run(source_id: int) -> IngestionResult:
        async with semaphore:
            return await _ingest_one(source_id, config)

    results = await asyncio.gather(*(run(source_id) for source_id in source_ids))
    logger.info(
        "ingestion_complete",
        extra={
            "sources": len(results),
            "stored": sum(result.stored for result in results),
            "failed": sum(1 for result in results if not result.success),
        },
    )
    return list(results)


async def _ingest_one(source_id: int, config: Settings) -> IngestionResult:
    """One source, one transaction, one error boundary."""
    try:
        async with session_scope() as session:
            repository = SourceRepository(session)
            source = await repository.get(source_id)
            if source is None:
                return IngestionResult(
                    source=str(source_id), success=False, error="source not found"
                )

            pipeline = IngestionPipeline(session=session, config=config)
            await pipeline.prepare(source_id=source_id)
            return await pipeline.ingest_source(source)
    except SourceError as exc:
        logger.warning("source_ingest_failed", extra={"source_id": source_id, "code": exc.code})
        return IngestionResult(source=str(source_id), success=False, error=exc.message)
    except Exception as exc:
        logger.exception("source_ingest_crashed", extra={"source_id": source_id})
        return IngestionResult(
            source=str(source_id), success=False, error=f"{exc.__class__.__name__}: {exc}"[:300]
        )


__all__ = ["IngestionPipeline", "PipelineStats", "ingest_sources"]
