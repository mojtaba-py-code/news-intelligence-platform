"""Users, their preferences and saved articles."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
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

from app.core.security import Role
from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.article import Article


class User(Base, TimestampMixin):
    """An authenticated platform user.

    Only the Argon2 hash is stored. ``failed_login_attempts``/``locked_until``
    implement account lockout, and ``token_version`` invalidates every issued
    JWT for this user when bumped (password change, forced logout).
    """

    __tablename__ = "users"
    __table_args__ = ({"comment": "Platform users"},)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(160))

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default=Role.USER, nullable=False, index=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Bumped to revoke all outstanding tokens for this user.
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    preference: Mapped[UserPreference | None] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
        lazy="selectin",
    )
    saved_articles: Mapped[list[SavedArticle]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def role_enum(self) -> Role:
        return Role(self.role)

    def is_locked(self, now: datetime) -> bool:
        return self.locked_until is not None and self.locked_until > now


class UserPreference(Base, TimestampMixin):
    """Personalisation settings driving the intelligence feed."""

    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )

    topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    entities: Mapped[list[str]] = mapped_column(JSON, default=list)
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    excluded_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    languages: Mapped[list[str]] = mapped_column(JSON, default=list)
    countries: Mapped[list[str]] = mapped_column(JSON, default=list)

    #: ``any`` | ``positive`` | ``neutral`` | ``negative``
    sentiment_preference: Mapped[str] = mapped_column(String(16), default="any", nullable=False)
    min_relevance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: Free-form UI settings; validated by the API schema before it lands here.
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="preference")


class SavedArticle(Base):
    """Bookmarked article with an optional private note."""

    __tablename__ = "saved_articles"
    __table_args__ = (
        UniqueConstraint("user_id", "article_id", name="uq_saved_articles_pair"),
        Index("ix_saved_articles_user_time", "user_id", "saved_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="saved_articles")
    article: Mapped[Article] = relationship(lazy="selectin")


__all__ = ["SavedArticle", "User", "UserPreference"]
