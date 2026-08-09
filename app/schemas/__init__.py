"""Pydantic schemas: the validation boundary between the outside world and the core."""

from app.schemas.article import (
    ArticleDetail,
    ArticleRead,
    ArticleSearchQuery,
    NormalizedArticle,
    RawArticle,
)
from app.schemas.common import Page, PageMeta, PaginationParams, SortOrder
from app.schemas.source import SourceCreate, SourceRead, SourceUpdate
from app.schemas.user import (
    LoginRequest,
    PreferenceRead,
    PreferenceUpdate,
    TokenResponse,
    UserCreate,
    UserRead,
)

__all__ = [
    "ArticleDetail",
    "ArticleRead",
    "ArticleSearchQuery",
    "LoginRequest",
    "NormalizedArticle",
    "Page",
    "PageMeta",
    "PaginationParams",
    "PreferenceRead",
    "PreferenceUpdate",
    "RawArticle",
    "SortOrder",
    "SourceCreate",
    "SourceRead",
    "SourceUpdate",
    "TokenResponse",
    "UserCreate",
    "UserRead",
]
