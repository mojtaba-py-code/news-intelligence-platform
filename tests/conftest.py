"""Shared pytest fixtures.

The environment is configured **before** ``app`` is imported: settings are a
cached singleton, so the ordering here decides what the whole suite runs
against (in-memory SQLite, no Redis, deterministic secrets).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Environment - must come before any `app` import.
# --------------------------------------------------------------------------- #
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-used-in-production-0123456789abcdef")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("RESPECT_ROBOTS_TXT", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("PASSWORD_MIN_LENGTH", "12")
os.environ.setdefault("SOURCES_CONFIG_PATH", "configs/sources.yaml")

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.cache import InMemoryCache, set_cache
from app.core.config import Settings, get_settings
from app.core.metrics import registry
from app.core.resilience import breakers
from app.core.security import Role, hash_password
from app.core.utils import (
    content_fingerprint,
    utcnow,
)
from app.database.base import Base
from app.database.models.article import Article, ProcessingStatus, SentimentLabel
from app.database.models.source import Source, SourceKind
from app.database.models.user import User
from app.database.session import dispose_engine, init_engine
from app.main import create_app
from app.processing.deduplication.hashing import simhash64


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[object]:
    """Fresh in-memory schema per test."""
    instance = init_engine(settings, force=True)
    async with instance.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield instance
    finally:
        await dispose_engine()


@pytest.fixture
async def session(engine: object) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)  # type: ignore[arg-type]
    async with factory() as db_session:
        yield db_session


@pytest.fixture(autouse=True)
def reset_process_state() -> AsyncIterator[None]:
    """Metrics, cache and breakers are process-wide; isolate them per test."""
    registry.reset()
    set_cache(InMemoryCache())
    breakers.clear()
    yield
    registry.reset()
    breakers.clear()


@pytest.fixture
async def client(engine: object) -> AsyncIterator[AsyncClient]:
    """HTTP client bound to the ASGI app (no network, no server process)."""
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
@pytest.fixture
async def source(session: AsyncSession) -> Source:
    row = Source(
        slug="test-source",
        name="Test Source",
        kind=str(SourceKind.RSS),
        url="https://example.com/feed.xml",
        language="en",
        country="US",
        category="technology",
        weight=1.2,
        reliability_score=0.8,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


def make_article(
    source_row: Source,
    *,
    title: str = "A significant technology announcement today",
    url: str = "https://example.com/articles/1",
    content: str | None = "Detailed body text about the announcement and its impact.",
    hours_old: float = 1.0,
    sentiment: float = 0.0,
    label: SentimentLabel = SentimentLabel.NEUTRAL,
    relevance: float = 0.5,
    category: str | None = "technology",
    is_duplicate: bool = False,
) -> Article:
    """Build an ``Article`` with consistent hashes (constraints are real)."""
    body = content or title
    return Article(
        source_id=source_row.id,
        source_name=source_row.name,
        title=title,
        description=body[:200],
        content=content,
        url=url,
        canonical_url=url,
        published_at=utcnow() - timedelta(hours=hours_old),
        language="en",
        country="US",
        category=category,
        content_hash=content_fingerprint(f"{title}\n{body}"),
        title_hash=content_fingerprint(title),
        simhash=str(simhash64(f"{title} {body}")),
        sentiment_score=sentiment,
        sentiment_label=str(label),
        sentiment_confidence=0.5,
        relevance_score=relevance,
        quality_score=0.7,
        keywords=["technology", "announcement"],
        word_count=len(body.split()),
        status=str(ProcessingStatus.PROCESSED),
        processed_at=utcnow(),
        is_duplicate=is_duplicate,
    )


@pytest.fixture
async def articles(session: AsyncSession, source: Source) -> list[Article]:
    """Ten varied articles - enough for search, stats and trend assertions."""
    rows = [
        make_article(
            source,
            title=f"Technology story number {index} about artificial intelligence",
            url=f"https://example.com/articles/{index}",
            content=(
                f"Body {index}. Researchers announced a breakthrough in machine learning "
                "that improves accuracy across benchmarks and reduces costs."
            ),
            hours_old=index,
            sentiment=0.4 if index % 2 == 0 else -0.4,
            label=SentimentLabel.POSITIVE if index % 2 == 0 else SentimentLabel.NEGATIVE,
            relevance=0.9 - index * 0.05,
        )
        for index in range(1, 11)
    ]
    session.add_all(rows)
    await session.commit()
    return rows


@pytest.fixture
async def user(session: AsyncSession) -> User:
    row = User(
        email="analyst@example.com",
        username="analyst",
        password_hash=hash_password("Str0ng-Passw0rd!x"),
        role=str(Role.USER),
        is_active=True,
        is_verified=True,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@pytest.fixture
async def admin(session: AsyncSession) -> User:
    row = User(
        email="admin@example.com",
        username="rootadmin",
        password_hash=hash_password("Str0ng-Passw0rd!x"),
        role=str(Role.ADMIN),
        is_active=True,
        is_verified=True,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@pytest.fixture
def auth_headers() -> object:
    """Callable building an ``Authorization`` header for a user."""
    from app.core.security import create_access_token

    def build(target: User) -> dict[str, str]:
        token = create_access_token(
            target.id, target.role_enum, extra_claims={"ver": target.token_version}
        )
        return {"Authorization": f"Bearer {token}"}

    return build
