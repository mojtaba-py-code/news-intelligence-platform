"""User profile, preferences and saved articles."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, status

from app.api.dependencies import (
    ArticleRepoDep,
    CurrentUser,
    PaginationDep,
    UserRepoDep,
)
from app.core.errors import ErrorResponse, NotFoundError
from app.schemas.article import ArticleRead
from app.schemas.common import Message, Page
from app.schemas.user import (
    PreferenceRead,
    PreferenceUpdate,
    SavedArticleCreate,
    SavedArticleRead,
    UserRead,
    UserUpdate,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserRead, summary="Current profile")
async def read_profile(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)


@router.patch("/me", response_model=UserRead, summary="Update profile")
async def update_profile(
    payload: UserUpdate, user: CurrentUser, repository: UserRepoDep
) -> UserRead:
    """Update the caller's own profile.

    :class:`UserUpdate` deliberately excludes ``role`` and ``is_active``, so
    this endpoint cannot be used to escalate privilege.
    """
    updates = payload.model_dump(exclude_unset=True)
    if updates.get("email"):
        email = str(updates["email"]).lower()
        existing = await repository.get_by_email(email)
        if existing is not None and existing.id != user.id:
            from app.core.errors import ConflictError

            raise ConflictError("That e-mail is already in use.")
        user.email = email
        user.is_verified = False
    if "full_name" in updates:
        user.full_name = updates["full_name"]

    await repository.commit()
    return UserRead.model_validate(user)


@router.get("/me/preferences", response_model=PreferenceRead, summary="Read preferences")
async def read_preferences(user: CurrentUser, repository: UserRepoDep) -> PreferenceRead:
    preference = user.preference or await repository.get_preference(user.id)
    if preference is None:
        return PreferenceRead()
    return PreferenceRead.model_validate(preference)


@router.put("/me/preferences", response_model=PreferenceRead, summary="Replace preferences")
async def update_preferences(
    payload: PreferenceUpdate, user: CurrentUser, repository: UserRepoDep
) -> PreferenceRead:
    """Replace the caller's personalisation settings."""
    preference = await repository.upsert_preference(user.id, payload.model_dump())
    await repository.commit()
    return PreferenceRead.model_validate(preference)


@router.get("/me/saved", response_model=Page[ArticleRead], summary="Saved articles")
async def list_saved(
    user: CurrentUser, repository: UserRepoDep, pagination: PaginationDep
) -> Page[ArticleRead]:
    rows, total = await repository.saved_articles(
        user.id, limit=pagination.limit, offset=pagination.offset
    )
    return Page.build([ArticleRead.model_validate(row) for row in rows], total, pagination)


@router.post(
    "/me/saved",
    response_model=SavedArticleRead,
    status_code=status.HTTP_201_CREATED,
    summary="Save an article",
    responses={404: {"model": ErrorResponse, "description": "Article not found"}},
)
async def save_article(
    payload: SavedArticleCreate,
    user: CurrentUser,
    repository: UserRepoDep,
    articles: ArticleRepoDep,
) -> SavedArticleRead:
    if await articles.get(payload.article_id) is None:
        raise NotFoundError(f"Article {payload.article_id} does not exist.")
    saved = await repository.save_article(user.id, payload.article_id, payload.note)
    await repository.commit()
    return SavedArticleRead.model_validate(saved)


@router.delete(
    "/me/saved/{article_id}",
    response_model=Message,
    summary="Remove a saved article",
    responses={404: {"model": ErrorResponse, "description": "Not saved"}},
)
async def unsave_article(
    user: CurrentUser, repository: UserRepoDep, article_id: Annotated[int, Path(ge=1)]
) -> Message:
    removed = await repository.unsave_article(user.id, article_id)
    if not removed:
        raise NotFoundError("That article is not in your saved list.")
    await repository.commit()
    return Message(message="Removed from saved articles.")


__all__ = ["router"]
