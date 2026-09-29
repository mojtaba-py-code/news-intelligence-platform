"""Shared repository behaviour: generic CRUD, counting and pagination."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar, cast

from sqlalchemy import Select, delete, func, select
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.metrics import db_query_duration_seconds
from app.database.base import Base

ModelT = TypeVar("ModelT", bound=Base)


def affected_rows(result: Result[Any]) -> int:
    """Number of rows an UPDATE or DELETE matched.

    ``AsyncSession.execute`` is typed to return ``Result``, but for DML the
    object is a ``CursorResult``, which is the class that carries ``rowcount``.
    SQLAlchemy 2.1 no longer types ``rowcount`` on ``Result``, so it is read
    through the concrete class here rather than at every call site.
    """
    return int(cast("CursorResult[Any]", result).rowcount or 0)


class BaseRepository(Generic[ModelT]):
    """CRUD helpers shared by every repository.

    All filtering goes through SQLAlchemy expressions - user input is bound as
    a parameter, never string-formatted into SQL.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------ reads
    async def get(self, entity_id: int) -> ModelT | None:
        with db_query_duration_seconds.time(labels={"op": "get", "model": self.model.__name__}):
            return await self.session.get(self.model, entity_id)

    async def get_by(self, **filters: Any) -> ModelT | None:
        statement = select(self.model).filter_by(**filters).limit(1)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list(self, *, limit: int = 100, offset: int = 0) -> Sequence[ModelT]:
        statement = select(self.model).limit(min(limit, 1000)).offset(max(0, offset))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def count(self, statement: Select[Any] | None = None) -> int:
        """Count rows, reusing an existing filtered statement when given.

        ``maintain_column_froms=True`` is essential: without it SQLAlchemy drops
        the FROM that was inferred from the selected entity, and a statement
        with no WHERE clause degenerates to ``SELECT count(*)`` - which returns
        1 on every dialect.
        """
        if statement is None:
            statement = select(self.model)
        counted = statement.with_only_columns(func.count(), maintain_column_froms=True).order_by(
            None
        )
        result = await self.session.execute(counted)
        return int(result.scalar_one() or 0)

    async def exists(self, **filters: Any) -> bool:
        statement = select(func.count()).select_from(self.model).filter_by(**filters)
        result = await self.session.execute(statement)
        return bool(result.scalar_one())

    # ----------------------------------------------------------------- writes
    async def add(self, entity: ModelT, *, flush: bool = True) -> ModelT:
        self.session.add(entity)
        if flush:
            await self.session.flush()
        return entity

    async def add_all(self, entities: Sequence[ModelT], *, flush: bool = True) -> Sequence[ModelT]:
        """Bulk insert - one round trip instead of N."""
        if not entities:
            return entities
        self.session.add_all(list(entities))
        if flush:
            await self.session.flush()
        return entities

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)

    async def delete_by_id(self, entity_id: int) -> bool:
        statement = delete(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        result = await self.session.execute(statement)
        return affected_rows(result) > 0

    async def commit(self) -> None:
        await self.session.commit()

    async def flush(self) -> None:
        await self.session.flush()

    async def refresh(self, entity: ModelT) -> ModelT:
        await self.session.refresh(entity)
        return entity


def paginate(statement: Select[Any], *, limit: int, offset: int) -> Select[Any]:
    """Apply bounded LIMIT/OFFSET to a statement."""
    return statement.limit(max(1, min(limit, 200))).offset(max(0, offset))


__all__ = ["BaseRepository", "paginate"]
