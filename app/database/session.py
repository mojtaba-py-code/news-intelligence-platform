"""Async engine/session management.

One engine per process, created lazily so that importing the package never
opens a connection (important for CLI commands and tests). SQLite gets WAL mode
and foreign-key enforcement, which are *not* on by default and silently break
cascade rules otherwise.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(config: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": config.db_echo,
        "future": True,
        "pool_pre_ping": True,
    }
    if config.is_sqlite:
        # SQLite has no server-side pool to size; an in-memory database must
        # share a single connection or every session sees an empty schema.
        if ":memory:" in config.database_url:
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_timeout=config.db_pool_timeout,
            pool_recycle=1800,
        )
    return kwargs


def init_engine(config: Settings | None = None, *, force: bool = False) -> AsyncEngine:
    """Create (or return) the process-wide async engine."""
    global _engine, _session_factory
    config = config or get_settings()

    if _engine is not None and not force:
        return _engine

    sqlite_path = config.sqlite_path()
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(config.database_url, **_engine_kwargs(config))

    if config.is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    _engine = engine
    _session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    logger.debug("database_engine_initialised", extra={"dialect": engine.dialect.name})
    return engine


def get_engine() -> AsyncEngine:
    """Return the engine, creating it on first use."""
    return _engine if _engine is not None else init_engine()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory, creating the engine if needed."""
    if _session_factory is None:
        init_engine()
    assert _session_factory is not None
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session with commit/rollback handling."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for workers and the CLI.

    Commits on success, rolls back on any exception.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all(config: Settings | None = None) -> None:
    """Create the schema directly (development/tests; use Alembic in production)."""
    from app.database import models  # noqa: F401 - ensure models are imported
    from app.database.base import Base

    engine = init_engine(config)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database_schema_created")


async def drop_all(config: Settings | None = None) -> None:
    """Drop every table. Guarded against production use."""
    from app.database.base import Base

    config = config or get_settings()
    if config.environment.is_production:
        raise RuntimeError("drop_all() is not permitted in production")
    engine = init_engine(config)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


async def check_connection() -> bool:
    """Cheap liveness probe used by ``/health`` and ``/ready``."""
    try:
        engine = get_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("database_unreachable", extra={"error": str(exc)})
        return False


async def dispose_engine() -> None:
    """Close pooled connections during shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def database_file(config: Settings | None = None) -> Path | None:
    """Path of the SQLite file backing the current configuration, if any."""
    return (config or get_settings()).sqlite_path()


__all__ = [
    "check_connection",
    "create_all",
    "database_file",
    "dispose_engine",
    "drop_all",
    "get_engine",
    "get_session",
    "get_session_factory",
    "init_engine",
    "session_scope",
]
