"""Declarative base, shared column types and mixins."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit naming conventions keep Alembic autogenerate deterministic and make
# constraint names stable across PostgreSQL and SQLite.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # JSON columns declare their type explicitly at the ``mapped_column`` call
    # site (``from __future__ import annotations`` turns generics into strings,
    # which the annotation map cannot resolve reliably).
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy API
        datetime: DateTime(timezone=True),
    }

    def to_dict(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """Shallow dict of column values (never includes relationships)."""
        exclude = exclude or set()
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in exclude
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = getattr(self, "id", None)
        return f"<{self.__class__.__name__} id={pk}>"


class TimestampMixin:
    """``created_at``/``updated_at`` maintained by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


__all__ = ["Base", "TimestampMixin"]
