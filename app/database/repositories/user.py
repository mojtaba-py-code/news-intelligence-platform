"""User, preference and saved-article persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.core.security import Role
from app.core.utils import utcnow
from app.database.models.article import Article
from app.database.models.user import SavedArticle, User, UserPreference
from app.database.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """All user reads/writes, including lockout accounting."""

    model = User

    async def get_by_email(self, email: str) -> User | None:
        return await self.get_by(email=email.strip().lower())

    async def get_by_username(self, username: str) -> User | None:
        return await self.get_by(username=username.strip().lower())

    async def get_by_identifier(self, identifier: str) -> User | None:
        """Look up by e-mail *or* username, as the login form allows both."""
        value = identifier.strip().lower()
        statement = (
            select(User)
            .where(or_(User.email == value, User.username == value))
            .limit(1)
            .options(selectinload(User.preference))
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        email: str,
        username: str,
        password_hash: str,
        full_name: str | None = None,
        role: Role = Role.USER,
        is_verified: bool = False,
    ) -> User:
        user = User(
            email=email.strip().lower(),
            username=username.strip().lower(),
            password_hash=password_hash,
            full_name=full_name,
            role=str(role),
            is_verified=is_verified,
        )
        await self.add(user)
        user.preference = UserPreference(user_id=user.id)
        await self.flush()
        return user

    async def list_users(self, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[User], int]:
        base = select(User)
        total = await self.count(base)
        statement = (
            base.order_by(User.created_at.desc()).limit(min(limit, 200)).offset(max(0, offset))
        )
        result = await self.session.execute(statement)
        return result.scalars().all(), total

    async def count_admins(self) -> int:
        statement = select(func.count(User.id)).where(
            User.role == str(Role.ADMIN), User.is_active.is_(True)
        )
        return int((await self.session.execute(statement)).scalar_one() or 0)

    # ----------------------------------------------------------- auth support
    async def register_failed_login(self, user: User, *, config: Settings | None = None) -> bool:
        """Increment the failure counter and lock the account when it trips.

        Returns ``True`` when the account is now locked.
        """
        config = config or get_settings()
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= config.max_failed_logins:
            user.locked_until = utcnow() + timedelta(minutes=config.lockout_minutes)
            user.failed_login_attempts = 0
            await self.flush()
            return True
        await self.flush()
        return False

    async def register_successful_login(self, user: User) -> None:
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = utcnow()
        await self.flush()

    async def revoke_tokens(self, user: User) -> None:
        """Invalidate every outstanding JWT for this user."""
        user.token_version += 1
        await self.flush()

    # ------------------------------------------------------------ preferences
    async def get_preference(self, user_id: int) -> UserPreference | None:
        return await self.session.scalar(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )

    async def upsert_preference(self, user_id: int, values: dict[str, object]) -> UserPreference:
        preference = await self.get_preference(user_id)
        if preference is None:
            preference = UserPreference(user_id=user_id)
            self.session.add(preference)
        for key, value in values.items():
            if hasattr(preference, key):
                setattr(preference, key, value)
        await self.flush()
        return preference

    # --------------------------------------------------------- saved articles
    async def save_article(self, user_id: int, article_id: int, note: str | None) -> SavedArticle:
        existing = await self.session.scalar(
            select(SavedArticle).where(
                SavedArticle.user_id == user_id, SavedArticle.article_id == article_id
            )
        )
        if existing is not None:
            existing.note = note
            await self.flush()
            return existing

        saved = SavedArticle(user_id=user_id, article_id=article_id, saved_at=utcnow(), note=note)
        self.session.add(saved)
        await self.flush()
        return saved

    async def unsave_article(self, user_id: int, article_id: int) -> bool:
        saved = await self.session.scalar(
            select(SavedArticle).where(
                SavedArticle.user_id == user_id, SavedArticle.article_id == article_id
            )
        )
        if saved is None:
            return False
        await self.session.delete(saved)
        return True

    async def saved_articles(
        self, user_id: int, *, limit: int = 20, offset: int = 0
    ) -> tuple[Sequence[Article], int]:
        base = (
            select(Article)
            .join(SavedArticle, SavedArticle.article_id == Article.id)
            .where(SavedArticle.user_id == user_id)
        )
        total = await self.count(base)
        statement = (
            base.order_by(SavedArticle.saved_at.desc())
            .limit(min(limit, 100))
            .offset(max(0, offset))
        )
        result = await self.session.execute(statement)
        return result.scalars().all(), total


__all__ = ["UserRepository"]
