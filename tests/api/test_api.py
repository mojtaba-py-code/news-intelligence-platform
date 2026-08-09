"""HTTP surface: authentication, authorisation, validation, headers, errors."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import Role, create_access_token, create_refresh_token
from app.database.models.article import Article
from app.database.models.user import User

pytestmark = pytest.mark.integration

STRONG_PASSWORD = "Str0ng-Passw0rd!x"


class TestOperations:
    async def test_health_is_public(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"]

    async def test_readiness_reports_components(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/ready")
        assert response.status_code in (200, 503)
        names = {component["name"] for component in response.json()["components"]}
        assert {"database", "cache"} <= names

    async def test_metrics_are_prometheus_text(self, client: AsyncClient) -> None:
        await client.get("/api/v1/health")
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "http_requests_total" in response.text

    async def test_probes_are_also_unprefixed(self, client: AsyncClient) -> None:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200

    async def test_openapi_is_served_outside_production(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        assert "/api/v1/articles" in response.json()["paths"]


class TestSecurityHeaders:
    async def test_hardening_headers_present(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        headers = response.headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in headers
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert headers["Cache-Control"] == "no-store"

    async def test_request_id_is_returned(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.headers["X-Request-ID"]

    async def test_client_supplied_request_id_is_sanitised(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/health", headers={"X-Request-ID": "bad id with spaces & <script>"}
        )
        returned = response.headers["X-Request-ID"]
        assert "<" not in returned and " " not in returned

    async def test_clean_request_id_is_preserved(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
        assert response.headers["X-Request-ID"] == "abc-123"


class TestAuthentication:
    async def test_registration_and_login(self, client: AsyncClient) -> None:
        registration = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": STRONG_PASSWORD,
                "full_name": "New User",
            },
        )
        assert registration.status_code == 201
        assert registration.json()["role"] == "USER"
        assert "password" not in registration.text

        login = await client.post(
            "/api/v1/auth/login",
            json={"username": "newuser", "password": STRONG_PASSWORD},
        )
        assert login.status_code == 200
        tokens = login.json()
        assert tokens["token_type"] == "bearer"

        me = await client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        assert me.json()["username"] == "newuser"

    async def test_registration_cannot_set_a_role(self, client: AsyncClient) -> None:
        """Mass assignment: extra fields are forbidden, not silently accepted."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "sneaky@example.com",
                "username": "sneaky",
                "password": STRONG_PASSWORD,
                "role": "ADMIN",
            },
        )
        assert response.status_code == 422

    async def test_weak_password_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "weak@example.com", "username": "weakuser", "password": "password1234"},
        )
        assert response.status_code == 422

    async def test_duplicate_registration_conflicts(self, client: AsyncClient, user: User) -> None:
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": user.email,
                "username": "someoneelse",
                "password": STRONG_PASSWORD,
            },
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "conflict"

    async def test_wrong_password_and_unknown_user_are_indistinguishable(
        self, client: AsyncClient, user: User
    ) -> None:
        wrong = await client.post(
            "/api/v1/auth/login", json={"username": user.username, "password": "not-the-password"}
        )
        unknown = await client.post(
            "/api/v1/auth/login", json={"username": "ghost", "password": "not-the-password"}
        )
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]

    async def test_protected_endpoint_requires_a_token(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    async def test_garbage_token_rejected(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
        )
        assert response.status_code == 401

    async def test_refresh_token_cannot_be_used_as_access_token(
        self, client: AsyncClient, user: User
    ) -> None:
        refresh = create_refresh_token(user.id, user.role_enum, extra_claims={"ver": 0})
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {refresh}"}
        )
        assert response.status_code == 401

    async def test_refresh_rotates_tokens(self, client: AsyncClient, user: User) -> None:
        refresh = create_refresh_token(user.id, user.role_enum, extra_claims={"ver": 0})
        response = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert response.status_code == 200
        assert response.json()["access_token"]

    async def test_password_change_revokes_existing_tokens(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        headers = auth_headers(user)
        change = await client.post(
            "/api/v1/auth/change-password",
            json={"current_password": STRONG_PASSWORD, "new_password": "An0ther-Passw0rd!"},
            headers=headers,
        )
        assert change.status_code == 200
        # The old token embedded the previous token_version.
        assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 401

    async def test_token_for_a_deactivated_account_is_refused(
        self,
        client: AsyncClient,
        session: AsyncSession,
        user: User,
        auth_headers: Callable[[User], dict[str, str]],
    ) -> None:
        headers = auth_headers(user)
        user.is_active = False
        await session.commit()
        assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 401


class TestAuthorization:
    async def test_admin_endpoint_denies_regular_users(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.get("/api/v1/admin/users", headers=auth_headers(user))
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"

    async def test_admin_endpoint_allows_admins(
        self, client: AsyncClient, admin: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.get("/api/v1/admin/users", headers=auth_headers(admin))
        assert response.status_code == 200
        assert response.json()["meta"]["total"] >= 1

    async def test_role_claim_cannot_be_forged(self, client: AsyncClient, user: User) -> None:
        """A USER-signed token claiming ADMIN must not grant admin access.

        The role is read from the *database* record, never trusted from the
        token payload alone.
        """
        forged = create_access_token(user.id, Role.ADMIN, extra_claims={"ver": user.token_version})
        response = await client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {forged}"}
        )
        assert response.status_code == 403

    async def test_analyst_role_hierarchy(
        self,
        client: AsyncClient,
        session: AsyncSession,
        auth_headers: Callable[[User], dict[str, str]],
    ) -> None:
        from app.core.security import hash_password

        analyst = User(
            email="an@example.com",
            username="analyst2",
            password_hash=hash_password(STRONG_PASSWORD),
            role=str(Role.ANALYST),
        )
        session.add(analyst)
        await session.commit()
        await session.refresh(analyst)

        # ANALYST may trigger ingestion but not administer users.
        assert (
            await client.get("/api/v1/admin/users", headers=auth_headers(analyst))
        ).status_code == 403
        assert (
            await client.post("/api/v1/sources/nonexistent/ingest", headers=auth_headers(analyst))
        ).status_code == 404


class TestArticlesAPI:
    async def test_listing_is_public_and_paginated(
        self, client: AsyncClient, articles: list[Article]
    ) -> None:
        response = await client.get("/api/v1/articles?page=1&page_size=3")
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 3
        assert body["meta"]["total"] == 10
        assert body["meta"]["pages"] == 4
        assert body["meta"]["has_next"] is True

    async def test_page_size_is_capped(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/articles?page_size=5000")
        assert response.status_code == 422

    async def test_search_filters(self, client: AsyncClient, articles: list[Article]) -> None:
        response = await client.get("/api/v1/articles/search?q=artificial+intelligence")
        assert response.status_code == 200
        assert response.json()["meta"]["total"] > 0

    async def test_unknown_sort_field_is_rejected(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/articles?sort_by=id;DROP+TABLE+articles")
        assert response.status_code == 422

    async def test_detail_and_missing_article(
        self, client: AsyncClient, articles: list[Article]
    ) -> None:
        found = await client.get(f"/api/v1/articles/{articles[0].id}")
        assert found.status_code == 200
        assert found.json()["content"]

        missing = await client.get("/api/v1/articles/999999")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"

    async def test_invalid_path_parameter(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/articles/-1")).status_code == 422
        assert (await client.get("/api/v1/articles/abc")).status_code == 422

    async def test_stats_endpoint(self, client: AsyncClient, articles: list[Article]) -> None:
        response = await client.get("/api/v1/articles/stats")
        assert response.json()["total"] == 10

    async def test_personalised_feed_requires_authentication(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/articles/feed")).status_code == 401

    async def test_personalised_feed(
        self,
        client: AsyncClient,
        articles: list[Article],
        user: User,
        auth_headers: Callable[[User], dict[str, str]],
    ) -> None:
        response = await client.get("/api/v1/articles/feed", headers=auth_headers(user))
        assert response.status_code == 200
        assert response.json()["items"]

    async def test_similar_articles(self, client: AsyncClient, articles: list[Article]) -> None:
        response = await client.get(f"/api/v1/articles/{articles[0].id}/similar?limit=3")
        assert response.status_code == 200
        assert len(response.json()) <= 3


class TestSourcesAPI:
    async def test_listing_is_public(self, client: AsyncClient, source: object) -> None:
        response = await client.get("/api/v1/sources")
        assert response.status_code == 200
        assert response.json()["meta"]["total"] == 1

    async def test_creation_requires_admin(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        payload = {
            "slug": "new-source",
            "name": "New Source",
            "kind": "rss",
            "url": "https://93.184.216.34/feed.xml",
        }
        assert (await client.post("/api/v1/sources", json=payload)).status_code == 401
        assert (
            await client.post("/api/v1/sources", json=payload, headers=auth_headers(user))
        ).status_code == 403

    async def test_admin_can_create_a_source(
        self, client: AsyncClient, admin: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.post(
            "/api/v1/sources",
            json={
                "slug": "new-source",
                "name": "New Source",
                "kind": "rss",
                "url": "https://93.184.216.34/feed.xml",
            },
            headers=auth_headers(admin),
        )
        assert response.status_code == 201
        assert response.json()["slug"] == "new-source"

    async def test_ssrf_unsafe_source_url_is_rejected(
        self, client: AsyncClient, admin: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.post(
            "/api/v1/sources",
            json={
                "slug": "internal",
                "name": "Internal",
                "kind": "rss",
                "url": "http://169.254.169.254/latest/meta-data/",
            },
            headers=auth_headers(admin),
        )
        assert response.status_code == 422

    async def test_inline_credentials_are_rejected(
        self, client: AsyncClient, admin: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.post(
            "/api/v1/sources",
            json={
                "slug": "leaky",
                "name": "Leaky",
                "kind": "api",
                "url": "https://93.184.216.34/api",
                "config": {"api_key": "secret-value"},
            },
            headers=auth_headers(admin),
        )
        assert response.status_code == 422

    async def test_source_response_never_includes_credentials(
        self, client: AsyncClient, source: object
    ) -> None:
        response = await client.get("/api/v1/sources/test-source")
        assert response.status_code == 200
        assert "api_key_env" not in response.json()
        assert "config" not in response.json()


class TestUsersAPI:
    async def test_preferences_roundtrip(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        headers = auth_headers(user)
        update = await client.put(
            "/api/v1/users/me/preferences",
            json={
                "topics": ["ai", "cybersecurity"],
                "keywords": ["openai"],
                "sentiment_preference": "positive",
                "min_relevance": 0.4,
            },
            headers=headers,
        )
        assert update.status_code == 200
        assert update.json()["topics"] == ["ai", "cybersecurity"]

        read = await client.get("/api/v1/users/me/preferences", headers=headers)
        assert read.json()["min_relevance"] == 0.4

    async def test_preference_limits_enforced(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.put(
            "/api/v1/users/me/preferences",
            json={"keywords": [f"term-{index}" for index in range(200)]},
            headers=auth_headers(user),
        )
        assert response.status_code == 422

    async def test_profile_update_cannot_escalate(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.patch(
            "/api/v1/users/me", json={"role": "ADMIN"}, headers=auth_headers(user)
        )
        assert response.status_code == 422

    async def test_saved_articles(
        self,
        client: AsyncClient,
        articles: list[Article],
        user: User,
        auth_headers: Callable[[User], dict[str, str]],
    ) -> None:
        headers = auth_headers(user)
        saved = await client.post(
            "/api/v1/users/me/saved",
            json={"article_id": articles[0].id, "note": "read later"},
            headers=headers,
        )
        assert saved.status_code == 201

        listing = await client.get("/api/v1/users/me/saved", headers=headers)
        assert listing.json()["meta"]["total"] == 1

        removed = await client.delete(f"/api/v1/users/me/saved/{articles[0].id}", headers=headers)
        assert removed.status_code == 200

    async def test_saving_a_missing_article_404s(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.post(
            "/api/v1/users/me/saved", json={"article_id": 987654}, headers=auth_headers(user)
        )
        assert response.status_code == 404


class TestAlertsAPI:
    async def test_alert_lifecycle(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        headers = auth_headers(user)
        created = await client.post(
            "/api/v1/alerts",
            json={
                "name": "OpenAI watch",
                "keywords": ["openai"],
                "min_articles": 1,
                "window_minutes": 60,
            },
            headers=headers,
        )
        assert created.status_code == 201
        alert_id = created.json()["id"]

        assert (await client.get("/api/v1/alerts", headers=headers)).json()[0]["id"] == alert_id
        toggled = await client.patch(f"/api/v1/alerts/{alert_id}?is_active=false", headers=headers)
        assert toggled.json()["is_active"] is False
        assert (
            await client.delete(f"/api/v1/alerts/{alert_id}", headers=headers)
        ).status_code == 200

    async def test_webhook_destination_is_ssrf_checked(
        self, client: AsyncClient, user: User, auth_headers: Callable[[User], dict[str, str]]
    ) -> None:
        response = await client.post(
            "/api/v1/alerts",
            json={
                "name": "Exfiltrate",
                "keywords": ["x"],
                "channel": "webhook",
                "destination": "http://169.254.169.254/latest/meta-data/",
            },
            headers=auth_headers(user),
        )
        assert response.status_code == 422

    async def test_alerts_are_scoped_to_their_owner(
        self,
        client: AsyncClient,
        user: User,
        admin: User,
        auth_headers: Callable[[User], dict[str, str]],
    ) -> None:
        created = await client.post(
            "/api/v1/alerts",
            json={"name": "Mine", "keywords": ["x"]},
            headers=auth_headers(user),
        )
        alert_id = created.json()["id"]
        # Even an admin does not own another user's alert.
        response = await client.delete(f"/api/v1/alerts/{alert_id}", headers=auth_headers(admin))
        assert response.status_code == 403


class TestAnalyticsAPI:
    async def test_overview(self, client: AsyncClient, articles: list[Article]) -> None:
        response = await client.get("/api/v1/analytics/overview?refresh=true")
        assert response.status_code == 200
        assert response.json()["total_articles"] == 10

    async def test_sentiment_and_topics(self, client: AsyncClient, articles: list[Article]) -> None:
        assert (await client.get("/api/v1/analytics/sentiment?hours=48")).status_code == 200
        assert (await client.get("/api/v1/analytics/topics?hours=48")).status_code == 200
        assert (await client.get("/api/v1/analytics/sources?hours=48")).status_code == 200
        assert (await client.get("/api/v1/analytics/timeseries?hours=48")).status_code == 200

    async def test_hours_parameter_is_bounded(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/analytics/sentiment?hours=99999")).status_code == 422


class TestErrorHandling:
    async def test_unknown_route_returns_the_error_envelope(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_validation_errors_do_not_echo_submitted_values(
        self, client: AsyncClient
    ) -> None:
        """A 422 must not reflect the password back into logs or responses."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "username": "u", "password": "sup3rSecret!Value"},
        )
        assert response.status_code == 422
        assert "sup3rSecret!Value" not in response.text
        assert response.json()["error"]["details"]["errors"]

    async def test_oversized_body_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/auth/login",
            content=b"x" * 2_000_000,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"


class TestDashboard:
    async def test_dashboard_renders(self, client: AsyncClient, articles: list[Article]) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        assert "News Intelligence Platform" in response.text
        assert "Content-Security-Policy" in response.headers

    async def test_article_titles_are_escaped(
        self, client: AsyncClient, session: AsyncSession, source: object
    ) -> None:
        """Feed content is untrusted; Jinja autoescaping must neutralise it."""
        from tests.conftest import make_article

        session.add(
            make_article(
                source,  # type: ignore[arg-type]
                title="<script>alert('xss')</script> Breaking news story",
                url="https://e.com/xss",
            )
        )
        await session.commit()

        response = await client.get("/")
        assert "<script>alert('xss')</script>" not in response.text
        assert "&lt;script&gt;" in response.text
