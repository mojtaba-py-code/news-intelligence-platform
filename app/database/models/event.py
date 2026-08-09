"""Event clusters - groups of articles covering the same real-world story."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
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
    from app.database.models.article import Article


class Event(Base, TimestampMixin):
    """A cluster of near-duplicate-but-distinct articles about one story.

    Corroboration matters: an event covered by eight independent sources is
    materially more important than one covered by a single outlet, so
    ``source_count`` feeds directly into ``importance``.
    """

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_importance_last_updated", "importance", "last_updated_at"),
        Index("ix_events_first_seen", "first_seen_at"),
        {"comment": "Story clusters detected across sources"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Denormalised for cheap listing: ``["reuters", "bbc", …]``
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    entities: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    avg_sentiment: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    #: Centroid of the cluster's TF-IDF vector, kept sparse: ``{term: weight}``.
    centroid: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    articles: Mapped[list[EventArticle]] = relationship(
        back_populates="event", cascade="all, delete-orphan", passive_deletes=True
    )


class EventArticle(Base):
    """Membership of an article in an event cluster."""

    __tablename__ = "event_articles"
    __table_args__ = (
        UniqueConstraint("event_id", "article_id", name="uq_event_articles_pair"),
        Index("ix_event_articles_event_similarity", "event_id", "similarity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    event: Mapped[Event] = relationship(back_populates="articles")
    article: Mapped[Article] = relationship(back_populates="events", lazy="selectin")


__all__ = ["Event", "EventArticle"]
