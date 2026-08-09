"""The canonical article model plus its association tables."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
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
    from app.database.models.event import EventArticle
    from app.database.models.source import Source
    from app.database.models.taxonomy import Entity, Topic


class SentimentLabel(StrEnum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    REJECTED = "rejected"


class Author(Base, TimestampMixin):
    """Normalised author, shared across articles."""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True, index=True
    )
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    articles: Mapped[list[Article]] = relationship(back_populates="author")


class Article(Base, TimestampMixin):
    """A single normalised, de-duplicated, enriched news article.

    Indexing strategy
    -----------------
    * ``canonical_url`` and ``content_hash`` are **unique** - the database is the
      last line of defence for idempotent ingestion, even under concurrency.
    * Composite indexes back the two hot query shapes: recency-ordered feeds
      filtered by source/category/language, and relevance-ordered search.
    """

    __tablename__ = "articles"
    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_articles_canonical_url"),
        UniqueConstraint("content_hash", name="uq_articles_content_hash"),
        Index("ix_articles_published_desc", "published_at"),
        Index("ix_articles_source_published", "source_id", "published_at"),
        Index("ix_articles_category_published", "category", "published_at"),
        Index("ix_articles_language_published", "language", "published_at"),
        Index("ix_articles_status_published", "status", "published_at"),
        Index("ix_articles_relevance", "relevance_score"),
        Index("ix_articles_sentiment", "sentiment_score"),
        Index("ix_articles_title_hash", "title_hash"),
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="relevance_range"),
        CheckConstraint("sentiment_score >= -1 AND sentiment_score <= 1", name="sentiment_range"),
        {"comment": "Canonical, de-duplicated news articles"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ------------------------------------------------------------- provenance
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_name: Mapped[str] = mapped_column(String(160), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), index=True)
    author_id: Mapped[int | None] = mapped_column(
        ForeignKey("authors.id", ondelete="SET NULL"), index=True
    )
    author_name: Mapped[str | None] = mapped_column(String(200))

    # ---------------------------------------------------------------- content
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)

    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    canonical_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    image_url: Mapped[str | None] = mapped_column(String(2048))

    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    language: Mapped[str | None] = mapped_column(String(8), index=True)
    country: Mapped[str | None] = mapped_column(String(8), index=True)
    category: Mapped[str | None] = mapped_column(String(48), index=True)

    # ----------------------------------------------------------- fingerprints
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    title_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: SimHash of the content, stored as text so SQLite can hold 64-bit values.
    simhash: Mapped[str | None] = mapped_column(String(20), index=True)

    # ------------------------------------------------------------ enrichment
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment_label: Mapped[str] = mapped_column(
        String(20), default=SentimentLabel.NEUTRAL, nullable=False, index=True
    )
    sentiment_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    readability_score: Mapped[float | None] = mapped_column(Float)

    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    #: Denormalised NLP metadata (word count, detection confidence, model ids…).
    enrichment: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # -------------------------------------------------------------- lifecycle
    status: Mapped[str] = mapped_column(
        String(20), default=ProcessingStatus.PENDING, nullable=False, index=True
    )
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    duplicate_of_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), index=True
    )
    duplicate_score: Mapped[float | None] = mapped_column(Float)
    duplicate_method: Mapped[str | None] = mapped_column(String(32))

    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_error: Mapped[str | None] = mapped_column(String(512))

    # ---------------------------------------------------------- relationships
    source: Mapped[Source] = relationship(back_populates="articles", lazy="selectin")
    author: Mapped[Author | None] = relationship(back_populates="articles", lazy="selectin")
    topics: Mapped[list[ArticleTopic]] = relationship(
        back_populates="article", cascade="all, delete-orphan", passive_deletes=True
    )
    entities: Mapped[list[ArticleEntity]] = relationship(
        back_populates="article", cascade="all, delete-orphan", passive_deletes=True
    )
    events: Mapped[list[EventArticle]] = relationship(
        back_populates="article", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_processed(self) -> bool:
        return self.status == ProcessingStatus.PROCESSED


class ArticleTopic(Base):
    """Article ↔ topic link carrying the classifier's confidence."""

    __tablename__ = "article_topics"
    __table_args__ = (
        UniqueConstraint("article_id", "topic_id", name="uq_article_topics_pair"),
        Index("ix_article_topics_topic_score", "topic_id", "score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    article: Mapped[Article] = relationship(back_populates="topics")
    topic: Mapped[Topic] = relationship(back_populates="articles", lazy="selectin")


class ArticleEntity(Base):
    """Article ↔ entity link with mention count and salience."""

    __tablename__ = "article_entities"
    __table_args__ = (
        UniqueConstraint("article_id", "entity_id", name="uq_article_entities_pair"),
        Index("ix_article_entities_entity_salience", "entity_id", "salience"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mentions: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    salience: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    article: Mapped[Article] = relationship(back_populates="entities")
    entity: Mapped[Entity] = relationship(back_populates="articles", lazy="selectin")


__all__ = [
    "Article",
    "ArticleEntity",
    "ArticleTopic",
    "Author",
    "ProcessingStatus",
    "SentimentLabel",
]
