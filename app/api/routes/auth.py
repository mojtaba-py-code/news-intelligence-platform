"""Authentication endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.dependencies import AuthServiceDep, ClientDep, CurrentUser
from app.core.errors import ErrorResponse
from app.schemas.common import Message
from app.schemas.user import (
    LoginRequest,
    PasswordChange,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserRead,
)

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing credentials"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
    },
)


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    responses={409: {"model": ErrorResponse, "description": "E-mail or username already taken"}},
)
async def register(payload: UserCreate, service: AuthServiceDep, client: ClientDep) -> UserRead:
    """Register a new user.

    The role is always ``USER``; privilege is granted separately by an admin.
    """
    user = await service.register(payload, client=client)
    await service.users.commit()
    return UserRead.model_validate(user)


@router.post("/login", response_model=TokenResponse, summary="Obtain an access token")
async def login(payload: LoginRequest, service: AuthServiceDep, client: ClientDep) -> TokenResponse:
    """Exchange credentials for an access/refresh token pair."""
    user = await service.authenticate(payload.username, payload.password, client=client)
    tokens = service.issue_tokens(user)
    await service.users.commit()
    return tokens


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 password flow (Swagger 'Authorize' button)",
    include_in_schema=True,
)
async def login_oauth2(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    service: AuthServiceDep,
    client: ClientDep,
) -> TokenResponse:
    """Standard OAuth2 form login, so the interactive docs can authenticate."""
    user = await service.authenticate(form.username, form.password, client=client)
    tokens = service.issue_tokens(user)
    await service.users.commit()
    return tokens


@router.post("/refresh", response_model=TokenResponse, summary="Rotate tokens")
async def refresh(
    payload: RefreshRequest, service: AuthServiceDep, client: ClientDep
) -> TokenResponse:
    """Exchange a refresh token for a fresh pair."""
    _, tokens = await service.refresh(payload.refresh_token, client=client)
    await service.users.commit()
    return tokens


@router.get("/me", response_model=UserRead, summary="Current user")
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)


@router.post("/change-password", response_model=Message, summary="Change password")
async def change_password(
    payload: PasswordChange, user: CurrentUser, service: AuthServiceDep, client: ClientDep
) -> Message:
    """Change the password. Every existing token is revoked."""
    await service.change_password(
        user, payload.current_password, payload.new_password, client=client
    )
    await service.users.commit()
    return Message(message="Password changed. Please sign in again.")


@router.post("/logout", response_model=Message, summary="Revoke all tokens")
async def logout(user: CurrentUser, service: AuthServiceDep, client: ClientDep) -> Message:
    await service.logout(user, client=client)
    await service.users.commit()
    return Message(message="Signed out.")


__all__ = ["router"]
