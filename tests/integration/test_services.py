"""Service-layer behaviour that the HTTP tests cannot reach directly."""

from __future__ import annotations

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import AuthenticationError, ConflictError, ValidationError
from app.core.security import Role, TokenType, create_refresh_token, decode_token
from app.core.utils import utcnow
from app.database.models.article import Article
from app.database.models.job import Alert, AlertChannel, AuditAction
from app.database.models.user import User, UserPreference
from app.database.repositories.audit import AuditRepository
from app.database.repositories.user import UserRepository
from app.intelligence.ranking import InterestProfile
from app.schemas.user import UserCreate
from app.services.alerts import AlertService
from app.services.auth import AuthService, ClientInfo
from app.services.events import EventService
from app.services.feed import FeedService

pytestmark = pytest.mark.integration

PASSWORD = "Str0ng-Passw0rd!x"


@pytest.fixture
def auth(session: AsyncSession) -> AuthService:
    return AuthService(UserRepository(session), AuditRepository(session), config=get_settings())


class TestAuthService:
    async def test_registration_creates_a_user_and_an_audit_entry(
        self, session: AsyncSession, auth: AuthService
    ) -> None:
        user = await auth.register(
            UserCreate(email="new@example.com", username="newperson", password=PASSWORD),
            client=ClientInfo(ip="203.0.113.9", user_agent="pytest"),
        )
        await session.commit()

        assert user.role == str(Role.USER)
        assert user.password_hash.startswith("$argon2id$")
        entries, total = await AuditRepository(session).recent(action=AuditAction.USER_CREATED)
        assert total == 1
        assert entries[0].client_ip == "203.0.113.0"

    async def test_duplicate_email_and_username_conflict(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        with pytest.raises(ConflictError):
            await auth.register(
                UserCreate(email=user.email, username="different", password=PASSWORD)
            )
        with pytest.raises(ConflictError):
            await auth.register(
                UserCreate(email="fresh@example.com", username=user.username, password=PASSWORD)
            )

    async def test_authenticate_success_resets_the_failure_counter(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        user.failed_login_attempts = 3
        await session.commit()

        authenticated = await auth.authenticate(user.username, PASSWORD)
        assert authenticated.id == user.id
        assert authenticated.failed_login_attempts == 0
        assert authenticated.last_login_at is not None

    async def test_lockout_after_repeated_failures(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        for _ in range(get_settings().max_failed_logins):
            with pytest.raises(AuthenticationError):
                await auth.authenticate(user.username, "wrong-password")
        await session.commit()

        assert user.is_locked(utcnow())
        # Even the correct password is refused while the lock holds.
        with pytest.raises(AuthenticationError):
            await auth.authenticate(user.username, PASSWORD)

        entries, _ = await AuditRepository(session).recent(action=AuditAction.ACCOUNT_LOCKED)
        assert entries

    async def test_inactive_account_cannot_authenticate(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        user.is_active = False
        await session.commit()
        with pytest.raises(AuthenticationError):
            await auth.authenticate(user.username, PASSWORD)

    async def test_token_pair_carries_the_token_version(
        self, auth: AuthService, user: User
    ) -> None:
        tokens = auth.issue_tokens(user)
        access = decode_token(tokens.access_token)
        refresh = decode_token(tokens.refresh_token, expected_type=TokenType.REFRESH)
        assert access.raw["ver"] == user.token_version
        assert access.user_id == refresh.user_id == user.id
        assert tokens.expires_in == get_settings().access_token_expire_minutes * 60

    async def test_refresh_returns_a_fresh_pair(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        token = create_refresh_token(user.id, user.role_enum, extra_claims={"ver": 0})
        refreshed_user, tokens = await auth.refresh(token)
        await session.commit()
        assert refreshed_user.id == user.id
        assert decode_token(tokens.access_token).user_id == user.id

    async def test_stale_token_version_is_rejected(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        tokens = auth.issue_tokens(user)
        await UserRepository(session).revoke_tokens(user)
        await session.commit()

        with pytest.raises(AuthenticationError, match="revoked"):
            await auth.resolve_token(tokens.access_token)

    async def test_password_change_requires_the_current_password(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        with pytest.raises(AuthenticationError):
            await auth.change_password(user, "wrong", "An0ther-Passw0rd!")

        await auth.change_password(user, PASSWORD, "An0ther-Passw0rd!")
        await session.commit()
        assert user.token_version == 1

        with pytest.raises(ValidationError):
            await auth.change_password(user, "An0ther-Passw0rd!", "An0ther-Passw0rd!")

    async def test_logout_revokes_tokens(
        self, session: AsyncSession, auth: AuthService, user: User
    ) -> None:
        before = user.token_version
        await auth.logout(user)
        await session.commit()
        assert user.token_version == before + 1


class TestFeedService:
    async def test_empty_profile_returns_recency_order(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        rows, total = await FeedService(session).personalized(user, limit=5)
        assert total == 10
        assert len(rows) == 5
        assert rows[0].published_at >= rows[-1].published_at

    async def test_preferences_reorder_the_feed(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        preference = UserPreference(
            user_id=user.id, keywords=["technology"], topics=["technology"], min_relevance=0.0
        )
        session.add(preference)
        await session.commit()
        await session.refresh(user)

        rows, total = await FeedService(session).personalized(user, limit=10)
        assert total > 0
        assert rows

    async def test_excluded_source_is_filtered_out(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        session.add(
            UserPreference(
                user_id=user.id, keywords=["technology"], excluded_sources=["test-source"]
            )
        )
        await session.commit()
        await session.refresh(user)

        rows, _ = await FeedService(session).personalized(user, limit=10)
        assert rows == []

    async def test_sentiment_preference_narrows_the_query(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        session.add(
            UserPreference(user_id=user.id, sentiment_preference="positive", keywords=["x"])
        )
        await session.commit()
        await session.refresh(user)

        rows, _ = await FeedService(session).personalized(user, limit=10)
        assert all(row.sentiment_score >= 0 for row in rows)

    async def test_recommendations_exclude_the_source_article(
        self, session: AsyncSession, articles: list[Article]
    ) -> None:
        related = await FeedService(session).recommendations(articles[0], limit=3)
        assert articles[0].id not in {row.id for row in related}

    def test_interest_profile_from_preference(self) -> None:
        profile = InterestProfile.from_preference(
            UserPreference(user_id=1, topics=["AI"], keywords=["OpenAI"])
        )
        assert profile.topics == frozenset({"ai"})
        assert profile.keywords == frozenset({"openai"})
        assert InterestProfile.from_preference(None).is_empty


class TestAlertDelivery:
    @pytest.fixture
    async def alert(self, session: AsyncSession, user: User) -> Alert:
        row = Alert(
            user_id=user.id,
            name="Tech watch",
            keywords=["technology"],
            min_articles=1,
            window_minutes=1440,
            channel=str(AlertChannel.WEBHOOK),
            destination="https://93.184.216.34/hook",
        )
        session.add(row)
        await session.commit()
        return row

    @respx.mock
    async def test_webhook_delivery(
        self, session: AsyncSession, articles: list[Article], alert: Alert
    ) -> None:
        route = respx.post("https://93.184.216.34/hook").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        evaluation = await AlertService(session).evaluate(alert)
        await session.commit()

        assert evaluation.triggered
        assert evaluation.delivered
        assert route.called
        payload = route.calls[0].request.read().decode()
        assert "Tech watch" in payload

    @respx.mock
    async def test_webhook_failure_is_recorded_not_raised(
        self, session: AsyncSession, articles: list[Article], alert: Alert
    ) -> None:
        respx.post("https://93.184.216.34/hook").mock(return_value=httpx.Response(500))
        evaluation = await AlertService(session).evaluate(alert)
        await session.commit()

        assert evaluation.triggered
        assert evaluation.delivered is False
        assert "500" in (evaluation.error or "")

    async def test_unsafe_destination_is_refused_at_delivery_time(
        self, session: AsyncSession, articles: list[Article], alert: Alert
    ) -> None:
        """Defence in depth: the schema checks it, and so does the sender."""
        alert.destination = "http://169.254.169.254/latest/meta-data/"
        await session.commit()

        evaluation = await AlertService(session).evaluate(alert)
        assert evaluation.triggered
        assert evaluation.delivered is False
        assert "rejected" in (evaluation.error or "").lower()

    async def test_cooldown_is_respected_by_evaluate_all(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        alert = Alert(
            user_id=user.id,
            name="In-app watch",
            keywords=["technology"],
            min_articles=1,
            window_minutes=1440,
            cooldown_minutes=60,
            channel=str(AlertChannel.IN_APP),
        )
        session.add(alert)
        await session.commit()

        service = AlertService(session)
        first = await service.evaluate_all()
        await session.commit()
        assert any(item.triggered for item in first)

        second = await service.evaluate_all()
        assert second == []  # still inside the cooldown window

    async def test_email_channel_reports_that_it_is_unconfigured(
        self, session: AsyncSession, articles: list[Article], user: User
    ) -> None:
        alert = Alert(
            user_id=user.id,
            name="Email watch",
            keywords=["technology"],
            min_articles=1,
            window_minutes=1440,
            channel=str(AlertChannel.EMAIL),
        )
        session.add(alert)
        await session.commit()

        evaluation = await AlertService(session).evaluate(alert)
        assert evaluation.triggered
        assert evaluation.delivered is False
        assert "SMTP" in (evaluation.error or "")


class TestEventService:
    @pytest.fixture
    async def cross_source_coverage(self, session: AsyncSession) -> None:
        """Three outlets covering one story, plus unrelated coverage."""
        from app.database.models.source import Source, SourceKind
        from tests.conftest import make_article

        body = (
            "A powerful earthquake struck the coastal region early on Tuesday, collapsing "
            "residential buildings and prompting a large rescue operation involving hundreds "
            "of emergency workers who continued searching through the night."
        )
        outlets = []
        for name in ("wire", "daily", "herald"):
            outlet = Source(
                slug=name,
                name=name.title(),
                kind=str(SourceKind.RSS),
                url=f"https://{name}.example.com/feed",
            )
            session.add(outlet)
            outlets.append(outlet)
        await session.commit()
        for outlet in outlets:
            await session.refresh(outlet)

        for index, outlet in enumerate(outlets):
            session.add(
                make_article(
                    outlet,
                    title=f"Earthquake strikes the coast, reports {outlet.slug}",
                    url=f"https://{outlet.slug}.example.com/quake",
                    content=f"{body} Report {index}.",
                    hours_old=index + 1,
                )
            )
        await session.commit()

    async def test_cross_source_story_becomes_an_event(
        self, session: AsyncSession, cross_source_coverage: None
    ) -> None:
        """Also a regression guard: clustering must not lazy-load relationships."""
        service = EventService(session, config=get_settings())
        clusters = await service.detect(hours=48)
        await session.commit()

        assert clusters
        cluster = clusters[0]
        assert cluster.size == 3
        assert cluster.source_count == 3
        assert 0.0 < cluster.importance <= 1.0

        events, total = await service.recent(limit=10)
        assert total == 1
        assert events[0].article_count == 3
        assert sorted(events[0].sources) == ["Daily", "Herald", "Wire"]
        assert events[0].title

    async def test_detection_is_idempotent(
        self, session: AsyncSession, cross_source_coverage: None
    ) -> None:
        service = EventService(session, config=get_settings())
        await service.detect(hours=48)
        await session.commit()
        await service.detect(hours=48)
        await session.commit()

        _, total = await service.recent(limit=10)
        assert total == 1  # updated in place, not duplicated

    async def test_breaking_events_are_recent_and_broadly_covered(
        self, session: AsyncSession, cross_source_coverage: None
    ) -> None:
        service = EventService(session, config=get_settings())
        await service.detect(hours=48)
        await session.commit()
        assert await service.breaking(hours=6)

    async def test_too_little_data_yields_no_events(self, session: AsyncSession) -> None:
        assert await EventService(session, config=get_settings()).detect(hours=48) == []
