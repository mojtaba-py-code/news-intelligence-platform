"""Source registry and per-source health tracking."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.article import Article


class SourceKind(StrEnum):
    """Connector implementation backing a source."""

    API = "api"
    RSS = "rss"
    SCRAPER = "scraper"
    JSON_FEED = "json_feed"
    CUSTOM = "custom"


class SourceStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    FAILING = "failing"
    DISABLED = "disabled"


class Source(Base, TimestampMixin):
    """A configured news source.

    ``config`` holds connector-specific settings (endpoint, selectors, query
    parameters). Credentials are *not* stored here: a source references an
    environment variable name via ``api_key_env`` and the connector resolves it
    at runtime, so the database never contains a secret.
    """

    __tablename__ = "sources"
    __table_args__ = (
        Index("ix_sources_status_kind", "status", "kind"),
        {"comment": "Registry of configured news sources"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default=SourceKind.RSS)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SourceStatus.ACTIVE, index=True
    )

    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    homepage: Mapped[str | None] = mapped_column(String(2048))
    description: Mapped[str | None] = mapped_column(Text)

    language: Mapped[str | None] = mapped_column(String(8), index=True)
    country: Mapped[str | None] = mapped_column(String(8), index=True)
    category: Mapped[str | None] = mapped_column(String(48), index=True)

    #: Name of the environment variable holding this source's API key.
    api_key_env: Mapped[str | None] = mapped_column(String(64))
    #: Connector-specific configuration (selectors, params, field mapping).
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    #: Multiplier applied by the relevance engine (0..2, trusted sources > 1).
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    #: Rolling reliability score in [0, 1] derived from fetch/quality history.
    reliability_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    request_delay_seconds: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    max_articles_per_run: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    respect_robots: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_articles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    articles: Mapped[list[Article]] = relationship(
        back_populates="source", cascade="all, delete-orphan", passive_deletes=True
    )
    health_records: Mapped[list[SourceHealth]] = relationship(
        back_populates="source", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_operational(self) -> bool:
        return self.enabled and self.status in (SourceStatus.ACTIVE, SourceStatus.FAILING)


class SourceHealth(Base):
    """One row per fetch attempt - the raw material for reliability scoring."""

    __tablename__ = "source_health"
    __table_args__ = (
        Index("ix_source_health_source_time", "source_id", "checked_at"),
        {"comment": "Per-run health samples for each source"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    articles_fetched: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    articles_valid: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicates: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(512))

    source: Mapped[Source] = relationship(back_populates="health_records")


__all__ = ["Source", "SourceHealth", "SourceKind", "SourceStatus"]
