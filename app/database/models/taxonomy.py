"""Topics, named entities and time-series trend snapshots."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.article import ArticleEntity, ArticleTopic


class EntityType(StrEnum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    COUNTRY = "COUNTRY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    OTHER = "OTHER"


class TrendDirection(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    NEW = "new"


class TrendSubject(StrEnum):
    TOPIC = "topic"
    KEYWORD = "keyword"
    ENTITY = "entity"
    SOURCE = "source"


class Topic(Base, TimestampMixin):
    """A configurable classification category."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    #: Seed terms used by the classifier; editable without a code change.
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    parent_slug: Mapped[str | None] = mapped_column(String(64), index=True)
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    articles: Mapped[list[ArticleTopic]] = relationship(
        back_populates="topic", cascade="all, delete-orphan", passive_deletes=True
    )


class Entity(Base, TimestampMixin):
    """A named entity mentioned across articles."""

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entities_name_type"),
        Index("ix_entities_type_mentions", "entity_type", "mention_count"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    #: Knowledge-graph edges: ``[{"predicate": "works_for", "object": "…"}, …]``
    relations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    articles: Mapped[list[ArticleEntity]] = relationship(
        back_populates="entity", cascade="all, delete-orphan", passive_deletes=True
    )


class TrendSnapshot(Base):
    """One measurement of one subject in one time window.

    Trends are stored rather than computed on the fly so that the dashboard is
    cheap to render and history survives retention cleanup of raw articles.
    """

    __tablename__ = "trend_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "subject_type", "subject_key", "window_start", name="uq_trend_subject_window"
        ),
        Index("ix_trend_window_score", "window_start", "trend_score"),
        Index("ix_trend_type_score", "subject_type", "trend_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    subject_key: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    subject_label: Mapped[str] = mapped_column(String(200), nullable=False)

    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_hours: Mapped[int] = mapped_column(Integer, default=24, nullable=False)

    current_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    previous_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    growth_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    direction: Mapped[str] = mapped_column(
        String(12), default=TrendDirection.STABLE, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    avg_sentiment: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment_delta: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sample_article_ids: Mapped[list[str]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


__all__ = ["Entity", "EntityType", "Topic", "TrendDirection", "TrendSnapshot", "TrendSubject"]
