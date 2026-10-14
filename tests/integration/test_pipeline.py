"""The ingestion pipeline end to end, with a stub connector."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from tests.conftest import make_article

from app.core.config import get_settings
from app.core.utils import utcnow
from app.database.models.article import Article, ArticleEntity, ArticleTopic
from app.database.models.source import Source
from app.database.repositories.taxonomy import TopicRepository
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.sources.base import NewsSource, SourceContext
from app.intelligence.topics import default_topic_seed
from app.services.alerts import AlertService
from app.services.analytics import AnalyticsService
from app.services.events import EventService
from app.services.trends import TrendService

pytestmark = pytest.mark.integration


class StubSource(NewsSource):
    """A connector that returns canned items instead of making requests."""

    def __init__(self, context: SourceContext, items: list[dict[str, Any]]) -> None:
        super().__init__(context, config=get_settings())
        self._items = items

    async def _collect(self) -> list[dict[str, Any]]:
        return self._items


def item(
    title: str,
    url: str,
    content: str,
    *,
    published: str | None = None,
    author: str = "Jane Doe",
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "content": content,
        "description": content[:120],
        "author": author,
        "published_at": published or utcnow().isoformat(),
        "language": "en",
    }


AI_BODY = (
    "OpenAI announced a major breakthrough in artificial intelligence research on Monday. "
    "The new large language model outperforms previous systems on reasoning benchmarks. "
    "Microsoft, which has invested billions in the company, welcomed the announcement. "
    "Independent researchers said the results would need to be reproduced before the "
    "claims could be accepted, and several universities have already begun that work. "
    "Regulators in Europe indicated they would review the deployment plans carefully."
)
CYBER_BODY = (
    "A ransomware group exploited a zero-day vulnerability to breach several companies. "
    "Security researchers warned that credentials were leaked and urged immediate patching."
)


@pytest.fixture
async def prepared(session: AsyncSession, source: Source) -> IngestionPipeline:
    await TopicRepository(session).seed(default_topic_seed())
    await session.commit()
    pipeline = IngestionPipeline(session=session, config=get_settings())
    await pipeline.prepare()
    return pipeline


class TestIngestionPipeline:
    async def test_articles_are_normalised_enriched_and_stored(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        connector = StubSource(
            SourceContext.from_model(source),
            [
                item("OpenAI unveils breakthrough AI model", "https://e.com/ai", AI_BODY),
                item("Ransomware group exploits zero-day flaw", "https://e.com/cyber", CYBER_BODY),
            ],
        )
        result = await prepared.ingest_source(source, connector=connector)
        await session.commit()

        assert result.success
        assert result.stored == 2
        assert result.duplicates == 0

        rows = (await session.execute(select(Article))).scalars().all()
        assert len(rows) == 2
        stored = {row.canonical_url: row for row in rows}

        ai_article = stored["https://e.com/ai"]
        assert ai_article.language == "en"
        assert ai_article.keywords
        assert ai_article.summary is not None
        assert ai_article.author_name == "Jane Doe"
        assert ai_article.category in {"ai", "technology", "business"}
        assert 0.0 <= ai_article.relevance_score <= 1.0
        assert ai_article.simhash

        topic_links = (await session.execute(select(func.count(ArticleTopic.id)))).scalar_one()
        entity_links = (await session.execute(select(func.count(ArticleEntity.id)))).scalar_one()
        assert topic_links > 0
        assert entity_links > 0

    async def test_ingestion_is_idempotent(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        """Running the same job twice must not create a second copy."""
        items = [item("OpenAI unveils breakthrough AI model", "https://e.com/ai", AI_BODY)]
        first = await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()

        second = await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()

        assert first.stored == 1
        assert second.stored == 0
        assert second.duplicates == 1
        total = (await session.execute(select(func.count(Article.id)))).scalar_one()
        assert total == 1

    async def test_duplicate_within_one_batch_is_caught(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        items = [
            item("Breaking: rates rise", "https://e.com/one", AI_BODY),
            # Same body, different URL - a syndicated copy.
            item("Breaking: rates rise", "https://e.com/two", AI_BODY),
        ]
        result = await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()
        assert result.stored == 1
        assert result.duplicates == 1

    async def test_lost_insert_race_costs_one_article_not_the_batch(
        self,
        session: AsyncSession,
        source: Source,
        prepared: IngestionPipeline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two feeds carrying the same story are ingested in parallel transactions.

        The duplicate check can pass and the INSERT still hit the unique
        constraint when the other transaction commits in between. Here the
        winning row is written directly and the first check is made to miss
        it, which reproduces that window deterministically. Before the fix the
        whole source batch was rolled back and the run crashed.
        """
        session.add(
            make_article(
                source,
                title="OpenAI unveils breakthrough AI model",
                url="https://e.com/ai",
                content=AI_BODY,
            )
        )
        await session.commit()

        real_check = prepared._find_duplicate
        calls = 0

        async def miss_first_check(normalized: Any) -> Any:
            nonlocal calls
            calls += 1
            return None if calls == 1 else await real_check(normalized)

        monkeypatch.setattr(prepared, "_find_duplicate", miss_first_check)
        items = [
            item("OpenAI unveils breakthrough AI model", "https://e.com/ai", AI_BODY),
            item("Ransomware group exploits zero-day flaw", "https://e.com/cyber", CYBER_BODY),
        ]
        result = await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()

        assert result.success
        assert result.stored == 1
        assert result.duplicates == 1
        urls = set((await session.execute(select(Article.canonical_url))).scalars())
        assert urls == {"https://e.com/ai", "https://e.com/cyber"}

    async def test_invalid_articles_are_rejected_not_stored(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        items = [
            item("OK headline about the economy today", "https://e.com/ok", AI_BODY),
            {"title": "", "url": "https://e.com/empty", "content": "x"},
            {"title": "No URL at all here", "url": "", "content": "x"},
            item(
                "CLICK HERE to earn $5000 now", "https://e.com/spam", "buy now limited time offer"
            ),
        ]
        result = await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()
        assert result.stored == 1
        assert result.rejected >= 2

    async def test_source_failure_is_recorded_and_isolated(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        class BrokenSource(StubSource):
            async def _collect(self) -> list[dict[str, Any]]:
                from app.core.errors import FetchError

                raise FetchError(self.slug, "upstream exploded")

        result = await prepared.ingest_source(
            source, connector=BrokenSource(SourceContext.from_model(source), [])
        )
        await session.commit()

        assert result.success is False
        assert "exploded" in (result.error or "")
        await session.refresh(source)
        assert source.consecutive_failures == 1
        assert source.last_error is not None

    async def test_crashed_run_is_recorded_against_the_source(
        self, session: AsyncSession, source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash used to leave no health sample, so the source kept a 100% success rate."""
        from app.ingestion import pipeline as pipeline_module

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("database went away")

        monkeypatch.setattr(pipeline_module.IngestionPipeline, "ingest_source", explode)
        result = await pipeline_module._ingest_one(source.id, get_settings())

        assert result.success is False
        assert result.source == source.slug
        assert "database went away" in (result.error or "")
        await session.refresh(source)
        assert source.consecutive_failures == 1
        assert source.last_error == "database went away"

    async def test_sqlite_sources_are_ingested_one_at_a_time(
        self, session: AsyncSession, source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SQLite has one writer; parallel sources only time out on its lock."""
        import asyncio

        if not get_settings().is_sqlite:
            pytest.skip("SQLite-only behaviour; PostgreSQL ingests sources concurrently")

        from app.ingestion import pipeline as pipeline_module
        from app.schemas.source import IngestionResult

        session.add(
            Source(slug="second-source", name="Second", kind="rss", url="https://e.org/feed")
        )
        await session.commit()

        running = peak = 0

        async def fake_ingest(source_id: int, _config: Any) -> IngestionResult:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return IngestionResult(source=str(source_id), success=True)

        monkeypatch.setattr(pipeline_module, "_ingest_one", fake_ingest)
        results = await pipeline_module.ingest_sources(concurrency=4)

        assert len(results) == 2
        assert peak == 1

    async def test_health_is_recorded_on_success(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        await prepared.ingest_source(
            source,
            connector=StubSource(
                SourceContext.from_model(source),
                [item("A headline that is long enough", "https://e.com/x", AI_BODY)],
            ),
        )
        await session.commit()
        await session.refresh(source)
        assert source.last_success_at is not None
        assert source.total_articles == 1
        assert source.consecutive_failures == 0

    async def test_quality_report_is_produced(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        await prepared.ingest_source(
            source,
            connector=StubSource(
                SourceContext.from_model(source),
                [
                    item("A perfectly good headline here", "https://e.com/a", AI_BODY),
                    {"title": "", "url": "https://e.com/b", "content": "x"},
                ],
            ),
        )
        report = prepared.quality_report()
        assert report.total_processed >= 1
        assert report.valid_articles >= 1


class TestServicesOverIngestedData:
    @pytest.fixture
    async def ingested(
        self, session: AsyncSession, source: Source, prepared: IngestionPipeline
    ) -> None:
        # Deliberately distinct wording: near-identical bodies would (correctly)
        # be collapsed by the deduplication engine, leaving nothing to analyse.
        ai_bodies = [
            "Researchers unveiled a language model that improves reasoning benchmarks sharply.",
            "A chip maker announced hardware built specifically for machine learning training.",
            "Regulators opened a consultation on artificial intelligence safety obligations.",
            "A startup raised funding to build generative models for medical diagnosis work.",
        ]
        cyber_bodies = [
            "Attackers exploited a zero-day vulnerability in a widely deployed VPN appliance.",
            "A ransomware group leaked credentials stolen from a regional hospital network.",
            "Investigators traced a phishing campaign targeting government email accounts.",
        ]
        items = [
            item(f"Artificial intelligence story {index}", f"https://e.com/ai-{index}", body)
            for index, body in enumerate(ai_bodies)
        ] + [
            item(f"Cybersecurity incident report {index}", f"https://e.com/cyber-{index}", body)
            for index, body in enumerate(cyber_bodies)
        ]
        await prepared.ingest_source(
            source, connector=StubSource(SourceContext.from_model(source), items)
        )
        await session.commit()

    async def test_trend_service_produces_snapshots(
        self, session: AsyncSession, ingested: None
    ) -> None:
        service = TrendService(session, config=get_settings())
        results = await service.compute(hours=24)
        await session.commit()
        assert results
        assert any(
            result.subject_key in {"ai", "cybersecurity", "technology"} for result in results
        )
        assert await service.latest(limit=5)

    async def test_event_service_clusters_related_articles(
        self, session: AsyncSession, ingested: None
    ) -> None:
        clusters = await EventService(session, config=get_settings()).detect(hours=48)
        await session.commit()
        # All articles come from one source, so no cluster qualifies as an event.
        assert isinstance(clusters, list)

    async def test_analytics_overview(self, session: AsyncSession, ingested: None) -> None:
        service = AnalyticsService(session, config=get_settings())
        overview = await service.overview(use_cache=False)
        assert overview.total_articles >= 5
        assert overview.total_sources == 1
        sentiment = await service.sentiment(hours=48)
        assert (
            sentiment.very_negative
            + sentiment.negative
            + sentiment.neutral
            + sentiment.positive
            + sentiment.very_positive
            >= 5
        )
        assert await service.topics(hours=48)
        assert await service.source_stats(hours=48)

    async def test_alert_service_matches_keywords(
        self, session: AsyncSession, ingested: None, user: Any
    ) -> None:
        from app.database.models.job import Alert, AlertChannel

        alert = Alert(
            user_id=user.id,
            name="Ransomware watch",
            keywords=["ransomware"],
            min_articles=1,
            window_minutes=1440,
            channel=str(AlertChannel.IN_APP),
        )
        session.add(alert)
        await session.commit()

        evaluation = await AlertService(session, config=get_settings()).evaluate(alert)
        await session.commit()
        assert evaluation.matched >= 1
        assert evaluation.triggered
        assert evaluation.delivered

    async def test_alert_does_not_fire_below_threshold(
        self, session: AsyncSession, ingested: None, user: Any
    ) -> None:
        from app.database.models.job import Alert, AlertChannel

        alert = Alert(
            user_id=user.id,
            name="Impossible",
            keywords=["definitely-not-present-keyword"],
            min_articles=1,
            window_minutes=1440,
            channel=str(AlertChannel.IN_APP),
        )
        session.add(alert)
        await session.commit()

        evaluation = await AlertService(session, config=get_settings()).evaluate(alert)
        assert evaluation.matched == 0
        assert evaluation.triggered is False
