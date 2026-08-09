"""FastAPI dependencies: sessions, repositories, services and access control."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, AuthorizationError
from app.core.security import Role
from app.database.models.job import AuditAction
from app.database.models.user import User
from app.database.repositories.article import ArticleRepository
from app.database.repositories.audit import AuditRepository
from app.database.repositories.event import EventRepository
from app.database.repositories.job import AlertRepository, JobRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import EntityRepository, TopicRepository, TrendRepository
from app.database.repositories.user import UserRepository
from app.database.session import get_session
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PaginationParams
from app.services.alerts import AlertService
from app.services.analytics import AnalyticsService
from app.services.auth import AuthService, ClientInfo
from app.services.events import EventService
from app.services.feed import FeedService
from app.services.trends import TrendService

#: ``auto_error=False`` so a missing header produces our own 401 envelope
#: rather than FastAPI's default shape.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
async def db_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped database session."""
    async for session in get_session():
        yield session


SessionDep = Annotated[AsyncSession, Depends(db_session)]


def settings_dependency() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dependency)]


def client_info(request: Request) -> ClientInfo:
    """Client metadata for the audit trail."""
    from app.api.middleware import _client_ip

    return ClientInfo(
        ip=_client_ip(request),
        user_agent=(request.headers.get("user-agent") or "")[:256] or None,
    )


ClientDep = Annotated[ClientInfo, Depends(client_info)]


# --------------------------------------------------------------------------- #
# Repositories
# --------------------------------------------------------------------------- #
def article_repository(session: SessionDep) -> ArticleRepository:
    return ArticleRepository(session)


def source_repository(session: SessionDep) -> SourceRepository:
    return SourceRepository(session)


def user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


def audit_repository(session: SessionDep) -> AuditRepository:
    return AuditRepository(session)


def topic_repository(session: SessionDep) -> TopicRepository:
    return TopicRepository(session)


def entity_repository(session: SessionDep) -> EntityRepository:
    return EntityRepository(session)


def trend_repository(session: SessionDep) -> TrendRepository:
    return TrendRepository(session)


def event_repository(session: SessionDep) -> EventRepository:
    return EventRepository(session)


def alert_repository(session: SessionDep) -> AlertRepository:
    return AlertRepository(session)


def job_repository(session: SessionDep) -> JobRepository:
    return JobRepository(session)


ArticleRepoDep = Annotated[ArticleRepository, Depends(article_repository)]
SourceRepoDep = Annotated[SourceRepository, Depends(source_repository)]
UserRepoDep = Annotated[UserRepository, Depends(user_repository)]
AuditRepoDep = Annotated[AuditRepository, Depends(audit_repository)]
TopicRepoDep = Annotated[TopicRepository, Depends(topic_repository)]
EntityRepoDep = Annotated[EntityRepository, Depends(entity_repository)]
TrendRepoDep = Annotated[TrendRepository, Depends(trend_repository)]
EventRepoDep = Annotated[EventRepository, Depends(event_repository)]
AlertRepoDep = Annotated[AlertRepository, Depends(alert_repository)]
JobRepoDep = Annotated[JobRepository, Depends(job_repository)]


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
def auth_service(users: UserRepoDep, audit: AuditRepoDep, config: SettingsDep) -> AuthService:
    return AuthService(users, audit, config=config)


def analytics_service(session: SessionDep, config: SettingsDep) -> AnalyticsService:
    return AnalyticsService(session, config=config)


def trend_service(session: SessionDep, config: SettingsDep) -> TrendService:
    return TrendService(session, config=config)


def event_service(session: SessionDep, config: SettingsDep) -> EventService:
    return EventService(session, config=config)


def feed_service(session: SessionDep, config: SettingsDep) -> FeedService:
    return FeedService(session, config=config)


def alert_service(session: SessionDep, config: SettingsDep) -> AlertService:
    return AlertService(session, config=config)


AuthServiceDep = Annotated[AuthService, Depends(auth_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(analytics_service)]
TrendServiceDep = Annotated[TrendService, Depends(trend_service)]
EventServiceDep = Annotated[EventService, Depends(event_service)]
FeedServiceDep = Annotated[FeedService, Depends(feed_service)]
AlertServiceDep = Annotated[AlertService, Depends(alert_service)]


# --------------------------------------------------------------------------- #
# Authentication & authorisation
# --------------------------------------------------------------------------- #
async def current_user(
    service: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """Resolve the authenticated user, or raise 401."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication required.")
    if credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Unsupported authorization scheme.")
    return await service.resolve_token(credentials.credentials)


CurrentUser = Annotated[User, Depends(current_user)]


async def optional_user(
    service: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User | None:
    """Resolve the user when a token is present; never raises for anonymous access."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return await service.resolve_token(credentials.credentials)
    except AuthenticationError:
        return None


OptionalUser = Annotated[User | None, Depends(optional_user)]


def require_role(minimum: Role) -> Callable[..., object]:
    """Dependency factory enforcing the role hierarchy.

    Denials are audited: repeated 403s are a meaningful security signal.
    """

    async def guard(
        user: CurrentUser, audit: AuditRepoDep, request: Request, client: ClientDep
    ) -> User:
        if not user.role_enum.can_act_as(minimum):
            await audit.record(
                AuditAction.UNAUTHORIZED_ACCESS,
                user_id=user.id,
                actor=user.username,
                client_ip=client.ip,
                resource=request.url.path[:160],
                success=False,
                detail={"required_role": str(minimum), "actual_role": user.role},
            )
            raise AuthorizationError(f"This action requires the {minimum} role.")
        return user

    return guard


RequireAnalyst = Annotated[User, Depends(require_role(Role.ANALYST))]
RequireAdmin = Annotated[User, Depends(require_role(Role.ADMIN))]


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #
def pagination(
    page: Annotated[int, Query(ge=1, le=10_000, description="1-based page number")] = 1,
    page_size: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")
    ] = DEFAULT_PAGE_SIZE,
) -> PaginationParams:
    """Validated pagination with a hard server-side ceiling."""
    return PaginationParams(page=page, page_size=page_size)


PaginationDep = Annotated[PaginationParams, Depends(pagination)]


__all__ = [
    "AlertServiceDep",
    "AnalyticsServiceDep",
    "ArticleRepoDep",
    "AuditRepoDep",
    "AuthServiceDep",
    "ClientDep",
    "CurrentUser",
    "EntityRepoDep",
    "EventRepoDep",
    "EventServiceDep",
    "FeedServiceDep",
    "JobRepoDep",
    "OptionalUser",
    "PaginationDep",
    "RequireAdmin",
    "RequireAnalyst",
    "SessionDep",
    "SettingsDep",
    "SourceRepoDep",
    "TopicRepoDep",
    "TrendRepoDep",
    "TrendServiceDep",
    "UserRepoDep",
    "require_role",
]
