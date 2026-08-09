"""Repository behaviour against a real (in-memory) database."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from tests.conftest import make_article

from app.core.security import Role
from app.core.utils import utcnow
from app.database.models.article import Article, ProcessingStatus
from app.database.models.job import AuditAction, JobType
from app.database.models.source import Source, SourceStatus
from app.database.models.taxonomy import EntityType
from app.database.repositories.article import ArticleRepository
from app.database.repositories.audit import AuditRepository, anonymize_ip
from app.database.repositories.job import JobRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import EntityRepository, TopicRepository
from app.database.repositories.user import UserRepository
from app.intelligence.topics import default_topic_seed
from app.schemas.article import ArticleSearchQuery, ArticleSortField
from app.schemas.common import SortOrder

pytestmark = pytest.mark.integration


class TestArticleRepository:
    async def test_search_filters_and_paginates(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = ArticleRepository(session)
        rows, total = await repository.search(ArticleSearchQuery(), limit=5, offset=0)
        assert total == 10
        assert len(rows) == 5
        # Default ordering is most recent first.
        assert rows[0].published_at >= rows[-1].published_at

    async def test_full_text_filter(self, session: AsyncSession, articles: list[Article]) -> None:
        repository = ArticleRepository(session)
        rows, total = await repository.search(
            ArticleSearchQuery(q="machine learning"), limit=20, offset=0
        )
        assert total == 10
        assert rows

    async def test_like_wildcards_are_escaped(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        """``%`` in a query must be a literal, not a match-everything wildcard."""
        repository = ArticleRepository(session)
        _, total = await repository.search(ArticleSearchQuery(q="%"), limit=20, offset=0)
        assert total == 0

    async def test_sql_injection_attempt_is_treated_as_text(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = ArticleRepository(session)
        _, total = await repository.search(
            ArticleSearchQuery(q="'; DROP TABLE articles; --"), limit=20, offset=0
        )
        assert total == 0
        # The table is still there.
        _, still_there = await repository.search(ArticleSearchQuery(), limit=1, offset=0)
        assert still_there == 10

    async def test_sorting_uses_the_allowlist(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = ArticleRepository(session)
        rows, _ = await repository.search(
            ArticleSearchQuery(sort_by=ArticleSortField.RELEVANCE, order=SortOrder.DESC),
            limit=10,
            offset=0,
        )
        scores = [row.relevance_score for row in rows]
        assert scores == sorted(scores, reverse=True)

    async def test_sentiment_and_relevance_ranges(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = ArticleRepository(session)
        _, positive = await repository.search(
            ArticleSearchQuery(min_sentiment=0.1), limit=20, offset=0
        )
        _, relevant = await repository.search(
            ArticleSearchQuery(min_relevance=0.8), limit=20, offset=0
        )
        assert positive == 5
        assert 0 < relevant < 10

    async def test_duplicates_excluded_by_default(
        self, session: AsyncSession, source: Source
    ) -> None:
        session.add_all(
            [
                make_article(source, title="Unique one story here", url="https://e.com/u1"),
                make_article(
                    source,
                    title="Duplicate story here",
                    url="https://e.com/d1",
                    is_duplicate=True,
                ),
            ]
        )
        await session.commit()

        repository = ArticleRepository(session)
        _, without = await repository.search(ArticleSearchQuery(), limit=10, offset=0)
        _, with_duplicates = await repository.search(
            ArticleSearchQuery(include_duplicates=True), limit=10, offset=0
        )
        assert without == 1
        assert with_duplicates == 2

    async def test_unique_constraints_enforce_idempotency(
        self, session: AsyncSession, source: Source
    ) -> None:
        """The database is the final guard against a duplicate insert race."""
        first = make_article(source, url="https://example.com/same")
        session.add(first)
        await session.commit()

        session.add(make_article(source, url="https://example.com/same"))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

    async def test_stats_aggregate(self, session: AsyncSession, articles: list[Article]) -> None:
        stats = await ArticleRepository(session).stats()
        assert stats.total == 10
        assert stats.last_24h >= 9
        assert 0.0 <= stats.avg_relevance <= 1.0

    async def test_breakdowns(self, session: AsyncSession, articles: list[Article]) -> None:
        repository = ArticleRepository(session)
        sentiment = await repository.sentiment_breakdown(hours=48)
        categories = await repository.category_breakdown(hours=48)
        series = await repository.volume_series(hours=48)
        assert sum(sentiment.values()) == 10
        assert categories[0][0] == "technology"
        assert series

    async def test_mark_duplicate_and_processed(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = ArticleRepository(session)
        original_id, duplicate_id, failed_id = articles[0].id, articles[1].id, articles[2].id
        await repository.mark_duplicate(
            duplicate_id, original_id=original_id, score=0.97, method="simhash"
        )
        await repository.mark_processed(failed_id, status=ProcessingStatus.FAILED, error="x")
        await session.commit()
        # Read the persisted rows, not the identity map: this asserts what the
        # next worker process would actually see.
        session.expire_all()

        duplicate = await repository.get(duplicate_id)
        failed = await repository.get(failed_id)
        assert duplicate is not None and duplicate.is_duplicate
        assert duplicate.duplicate_of_id == original_id
        assert failed is not None and failed.status == str(ProcessingStatus.FAILED)

    async def test_retention_cleanup(self, session: AsyncSession, source: Source) -> None:
        session.add_all(
            [
                make_article(
                    source,
                    title="Old story from the archive",
                    url="https://e.com/old",
                    hours_old=24 * 400,
                ),
                make_article(
                    source, title="Fresh story from today", url="https://e.com/new", hours_old=1
                ),
            ]
        )
        await session.commit()

        repository = ArticleRepository(session)
        removed = await repository.delete_older_than(days=90)
        await session.commit()
        assert removed == 1
        _, remaining = await repository.search(ArticleSearchQuery(), limit=10, offset=0)
        assert remaining == 1


class TestSourceRepository:
    async def test_upsert_preserves_operator_state(self, session: AsyncSession) -> None:
        repository = SourceRepository(session)
        created = await repository.upsert_definition(
            {"slug": "bbc", "name": "BBC", "kind": "rss", "url": "https://bbc.example/feed"}
        )
        created.status = str(SourceStatus.PAUSED)
        created.reliability_score = 0.9
        await session.commit()

        await repository.upsert_definition(
            {"slug": "bbc", "name": "BBC News", "kind": "rss", "url": "https://bbc.example/feed2"}
        )
        await session.commit()

        updated = await repository.get_by_slug("bbc")
        assert updated is not None
        assert updated.name == "BBC News"
        assert updated.status == str(SourceStatus.PAUSED)  # not reset by a config reload
        assert updated.reliability_score == 0.9

    async def test_health_recording_and_auto_pause(
        self, session: AsyncSession, source: Source
    ) -> None:
        repository = SourceRepository(session)
        await repository.record_run(source, success=True, fetched=10, valid=8, duplicates=2)
        assert source.status == str(SourceStatus.ACTIVE)
        assert source.total_articles == 8

        for _ in range(10):
            await repository.record_run(source, success=False, error_type="Timeout")
        await session.commit()
        assert source.status == str(SourceStatus.PAUSED)
        assert source.consecutive_failures >= 10

    async def test_reliability_blends_success_and_quality(
        self, session: AsyncSession, source: Source
    ) -> None:
        repository = SourceRepository(session)
        for _ in range(8):
            await repository.record_run(source, success=True, fetched=10, valid=9, duplicates=1)
        await session.commit()
        score = await repository.recompute_reliability(source.id)
        assert 0.7 < score <= 1.0

        other = await repository.upsert_definition(
            {"slug": "flaky", "name": "Flaky", "kind": "rss", "url": "https://f.example/feed"}
        )
        for _ in range(8):
            await repository.record_run(other, success=False, error_type="Timeout")
        await session.commit()
        assert await repository.recompute_reliability(other.id) < 0.5

    async def test_active_excludes_disabled(self, session: AsyncSession, source: Source) -> None:
        source.enabled = False
        await session.commit()
        assert await SourceRepository(session).active() == []


class TestTaxonomyRepositories:
    async def test_topic_seeding_is_idempotent(self, session: AsyncSession) -> None:
        repository = TopicRepository(session)
        first = await repository.seed(default_topic_seed())
        await session.commit()
        second = await repository.seed(default_topic_seed())
        assert first > 0
        assert second == 0
        assert len(await repository.slug_to_id()) == first

    async def test_entity_resolution_is_bulk_and_deduplicating(self, session: AsyncSession) -> None:
        repository = EntityRepository(session)
        resolved = await repository.bulk_resolve(
            [
                ("Apple", EntityType.ORGANIZATION),
                ("Apple Inc.", EntityType.ORGANIZATION),
                ("London", EntityType.LOCATION),
            ]
        )
        await session.commit()
        # "Apple" and "Apple Inc." normalise to the same key.
        assert len({entity.id for entity in resolved.values()}) == 2

    async def test_entity_get_or_create(self, session: AsyncSession) -> None:
        repository = EntityRepository(session)
        first = await repository.get_or_create("OpenAI", EntityType.ORGANIZATION)
        await session.commit()
        second = await repository.get_or_create("openai", EntityType.ORGANIZATION)
        assert first.id == second.id


class TestUserRepository:
    async def test_lookup_by_email_or_username(self, session: AsyncSession) -> None:
        repository = UserRepository(session)
        created = await repository.create(
            email="a@example.com", username="alice", password_hash="hash"
        )
        await session.commit()
        assert (await repository.get_by_identifier("a@example.com")).id == created.id  # type: ignore[union-attr]
        assert (await repository.get_by_identifier("ALICE")).id == created.id  # type: ignore[union-attr]

    async def test_lockout_after_repeated_failures(self, session: AsyncSession) -> None:
        repository = UserRepository(session)
        user = await repository.create(email="b@example.com", username="bob", password_hash="h")
        await session.commit()

        locked = False
        for _ in range(5):
            locked = await repository.register_failed_login(user)
        assert locked
        assert user.is_locked(utcnow())

        await repository.register_successful_login(user)
        assert not user.is_locked(utcnow())

    async def test_token_revocation_bumps_version(self, session: AsyncSession) -> None:
        repository = UserRepository(session)
        user = await repository.create(email="c@example.com", username="carol", password_hash="h")
        before = user.token_version
        await repository.revoke_tokens(user)
        assert user.token_version == before + 1

    async def test_admin_count(self, session: AsyncSession) -> None:
        repository = UserRepository(session)
        await repository.create(
            email="d@example.com", username="dave", password_hash="h", role=Role.ADMIN
        )
        await session.commit()
        assert await repository.count_admins() == 1

    async def test_saved_articles_roundtrip(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        repository = UserRepository(session)
        user = await repository.create(email="e@example.com", username="erin", password_hash="h")
        await session.commit()

        await repository.save_article(user.id, articles[0].id, "note")
        await session.commit()
        rows, total = await repository.saved_articles(user.id)
        assert total == 1 and rows[0].id == articles[0].id

        assert await repository.unsave_article(user.id, articles[0].id) is True
        await session.commit()
        _, after = await repository.saved_articles(user.id)
        assert after == 0


class TestJobRepository:
    async def test_claim_is_exclusive(self, session: AsyncSession) -> None:
        repository = JobRepository(session)
        await repository.enqueue(JobType.INGEST, target="bbc")
        await session.commit()

        first = await repository.claim_next()
        assert first is not None
        second = await repository.claim_next()
        assert second is None  # already claimed

    async def test_failure_retries_then_dead_letters(self, session: AsyncSession) -> None:
        repository = JobRepository(session)
        job = await repository.enqueue(JobType.TRENDS, max_attempts=2)
        await session.commit()

        claimed = await repository.claim_next()
        assert claimed is not None
        await repository.fail(claimed, "boom", retry_in_seconds=1)
        assert claimed.status == "queued"

        claimed.attempts = 2
        await repository.fail(claimed, "boom again")
        assert claimed.status == "dead_letter"
        assert len(await repository.dead_letters()) == 1

        assert await repository.requeue(job.id) is True
        assert job.status == "queued"

    async def test_queue_depth(self, session: AsyncSession) -> None:
        repository = JobRepository(session)
        await repository.enqueue(JobType.INGEST)
        await repository.enqueue(JobType.EVENTS)
        await session.commit()
        assert await repository.queue_depth() == 2


class TestAuditRepository:
    async def test_entries_are_written_with_anonymised_ip(self, session: AsyncSession) -> None:
        repository = AuditRepository(session)
        await repository.record(
            AuditAction.LOGIN_SUCCESS,
            actor="alice",
            client_ip="203.0.113.42",
            detail={"reason": "ok", "password": "should-not-persist"},
        )
        await session.commit()

        rows, total = await repository.recent()
        assert total == 1
        assert rows[0].client_ip == "203.0.113.0"
        assert "password" not in rows[0].detail

    def test_ip_anonymisation(self) -> None:
        assert anonymize_ip("198.51.100.7") == "198.51.100.0"
        assert anonymize_ip("2001:db8::1234") == "2001:db8::"
        assert anonymize_ip("not-an-ip") is None
        assert anonymize_ip(None) is None

    async def test_filtering_by_action(self, session: AsyncSession) -> None:
        repository = AuditRepository(session)
        await repository.record(AuditAction.LOGIN_SUCCESS, actor="a")
        await repository.record(AuditAction.LOGIN_FAILURE, actor="b", success=False)
        await session.commit()

        _, failures = await repository.recent(action=AuditAction.LOGIN_FAILURE)
        assert failures == 1
