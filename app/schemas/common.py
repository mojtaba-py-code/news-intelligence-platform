"""Shared schema building blocks: pagination, sorting and generic envelopes."""

from __future__ import annotations

from enum import StrEnum
from math import ceil
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

T = TypeVar("T")

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"


class StrictModel(BaseModel):
    """Base model that refuses unknown fields.

    Rejecting extras is a validation control, not a nicety: it stops mass
    assignment (a client sending ``{"role": "ADMIN"}`` to a profile update)
    from silently succeeding.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ORMModel(BaseModel):
    """Read model populated from SQLAlchemy objects."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class PaginationParams(BaseModel):
    """``?page=&page_size=`` with a hard server-side ceiling."""

    model_config = ConfigDict(extra="forbid")

    page: Annotated[int, Field(ge=1, le=10_000, description="1-based page number")] = 1
    page_size: Annotated[int, Field(ge=1, le=MAX_PAGE_SIZE, description="Items per page")] = (
        DEFAULT_PAGE_SIZE
    )

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


class PageMeta(BaseModel):
    """Pagination metadata attached to every list response."""

    page: int
    page_size: int
    total: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pages(self) -> int:
        return ceil(self.total / self.page_size) if self.page_size else 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_previous(self) -> bool:
        return self.page > 1


class Page(BaseModel, Generic[T]):
    """Uniform paginated envelope: ``{"items": [...], "meta": {...}}``."""

    items: list[T]
    meta: PageMeta

    @classmethod
    def build(cls, items: list[T], total: int, params: PaginationParams) -> Page[T]:
        return cls(
            items=items,
            meta=PageMeta(page=params.page, page_size=params.page_size, total=total),
        )


class Message(BaseModel):
    """Simple acknowledgement payload."""

    message: str
    detail: str | None = None


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    environment: str
    uptime_seconds: float
    components: list[ComponentHealth] = Field(default_factory=list)


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "ComponentHealth",
    "HealthResponse",
    "HealthStatus",
    "Message",
    "ORMModel",
    "Page",
    "PageMeta",
    "PaginationParams",
    "SortOrder",
    "StrictModel",
]
