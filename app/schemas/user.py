"""User, authentication and personalisation schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import EmailStr, Field, field_validator, model_validator

from app.core.security import Role, validate_password_strength
from app.schemas.common import ORMModel, StrictModel

USERNAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,62}[a-zA-Z0-9]$"


def _check_password(value: str) -> str:
    result = validate_password_strength(value)
    if not result.ok:
        raise ValueError("Password " + "; ".join(result.problems))
    return value


class UserCreate(StrictModel):
    """Registration payload.

    ``role`` is deliberately absent: privilege is assigned by an administrator
    through a separate endpoint, so registration cannot escalate.
    """

    email: EmailStr
    username: Annotated[str, Field(min_length=4, max_length=64, pattern=USERNAME_PATTERN)]
    password: Annotated[str, Field(min_length=12, max_length=1024)]
    full_name: Annotated[str | None, Field(max_length=160)] = None

    @field_validator("password")
    @classmethod
    def _strength(cls, value: str) -> str:
        return _check_password(value)

    @field_validator("username")
    @classmethod
    def _reserved(cls, value: str) -> str:
        if value.lower() in {"admin", "root", "system", "administrator", "support", "api"}:
            raise ValueError("username is reserved")
        return value


class UserUpdate(StrictModel):
    """Self-service profile update. Cannot touch role or activation flags."""

    full_name: Annotated[str | None, Field(max_length=160)] = None
    email: EmailStr | None = None


class AdminUserUpdate(StrictModel):
    """Administrative update; separated from :class:`UserUpdate` on purpose."""

    role: Role | None = None
    is_active: bool | None = None
    is_verified: bool | None = None
    full_name: Annotated[str | None, Field(max_length=160)] = None


class PasswordChange(StrictModel):
    current_password: Annotated[str, Field(min_length=1, max_length=1024)]
    new_password: Annotated[str, Field(min_length=12, max_length=1024)]

    @field_validator("new_password")
    @classmethod
    def _strength(cls, value: str) -> str:
        return _check_password(value)

    @model_validator(mode="after")
    def _must_differ(self) -> PasswordChange:
        if self.current_password == self.new_password:
            raise ValueError("new password must differ from the current one")
        return self


class LoginRequest(StrictModel):
    username: Annotated[str, Field(min_length=1, max_length=255)]
    password: Annotated[str, Field(min_length=1, max_length=1024)]


class TokenResponse(StrictModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 scheme name
    expires_in: int


class RefreshRequest(StrictModel):
    refresh_token: Annotated[str, Field(min_length=10, max_length=4096)]


class UserRead(ORMModel):
    id: int
    email: str
    username: str
    full_name: str | None = None
    role: Role
    is_active: bool
    is_verified: bool
    last_login_at: datetime | None = None
    created_at: datetime


class PreferenceUpdate(StrictModel):
    """Personalisation settings. Every list is length-capped."""

    topics: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    keywords: Annotated[list[str], Field(max_length=50)] = Field(default_factory=list)
    entities: Annotated[list[str], Field(max_length=50)] = Field(default_factory=list)
    sources: Annotated[list[str], Field(max_length=50)] = Field(default_factory=list)
    excluded_sources: Annotated[list[str], Field(max_length=50)] = Field(default_factory=list)
    languages: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    countries: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    sentiment_preference: Annotated[str, Field(pattern=r"^(any|positive|neutral|negative)$")] = (
        "any"
    )
    min_relevance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "topics", "keywords", "entities", "sources", "excluded_sources", "languages", "countries"
    )
    @classmethod
    def _clean_terms(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            term = str(item).strip()
            if not term:
                continue
            if len(term) > 100:
                raise ValueError("each entry must be at most 100 characters")
            if term not in cleaned:
                cleaned.append(term)
        return cleaned

    @field_validator("settings")
    @classmethod
    def _bound_settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 30 or len(str(value)) > 5_000:
            raise ValueError("settings object is too large")
        return value


class PreferenceRead(ORMModel):
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    sentiment_preference: str = "any"
    min_relevance: float = 0.0
    settings: dict[str, Any] = Field(default_factory=dict)


class SavedArticleCreate(StrictModel):
    article_id: Annotated[int, Field(ge=1)]
    note: Annotated[str | None, Field(max_length=2000)] = None


class SavedArticleRead(ORMModel):
    id: int
    article_id: int
    saved_at: datetime
    note: str | None = None


__all__ = [
    "AdminUserUpdate",
    "LoginRequest",
    "PasswordChange",
    "PreferenceRead",
    "PreferenceUpdate",
    "RefreshRequest",
    "SavedArticleCreate",
    "SavedArticleRead",
    "TokenResponse",
    "UserCreate",
    "UserRead",
    "UserUpdate",
]
