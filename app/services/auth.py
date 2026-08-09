"""Authentication service: registration, login, refresh, password changes.

Security properties enforced here rather than in the route:

* **Uniform failure.** Unknown user, wrong password and inactive account all
  produce the same error and the same amount of work (a dummy hash is verified
  for unknown users), so the endpoint cannot be used to enumerate accounts.
* **Lockout.** Consecutive failures lock the account for a configured window.
* **Token invalidation.** Every token embeds the user's ``token_version``; a
  password change bumps it and instantly invalidates outstanding sessions.
* **Audit.** Every outcome - success or failure - is written to the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, ConflictError, ValidationError
from app.core.logging import get_logger
from app.core.metrics import auth_failures_total
from app.core.security import (
    Role,
    TokenPayload,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from app.core.utils import utcnow
from app.database.models.job import AuditAction
from app.database.models.user import User
from app.database.repositories.audit import AuditRepository
from app.database.repositories.user import UserRepository
from app.schemas.user import TokenResponse, UserCreate

logger = get_logger(__name__)

#: Verified when the account does not exist, so timing does not reveal that.
_DUMMY_HASH = hash_password("dummy-password-for-constant-time-comparison")


@dataclass(frozen=True, slots=True)
class ClientInfo:
    """Request metadata recorded in the audit trail."""

    ip: str | None = None
    user_agent: str | None = None


class AuthService:
    """Coordinates the user repository, password hashing and token issuing."""

    def __init__(
        self,
        users: UserRepository,
        audit: AuditRepository,
        *,
        config: Settings | None = None,
    ) -> None:
        self.users = users
        self.audit = audit
        self.config = config or get_settings()

    # ---------------------------------------------------------- registration
    async def register(
        self, payload: UserCreate, *, client: ClientInfo | None = None, role: Role = Role.USER
    ) -> User:
        """Create a user. Raises :class:`ConflictError` on duplicates."""
        client = client or ClientInfo()
        if await self.users.get_by_email(payload.email):
            raise ConflictError("An account with this e-mail already exists.")
        if await self.users.get_by_username(payload.username):
            raise ConflictError("This username is taken.")

        user = await self.users.create(
            email=payload.email,
            username=payload.username,
            password_hash=hash_password(payload.password),
            full_name=payload.full_name,
            role=role,
        )
        await self.audit.record(
            AuditAction.USER_CREATED,
            user_id=user.id,
            actor=user.username,
            client_ip=client.ip,
            user_agent=client.user_agent,
            resource=f"user:{user.id}",
        )
        logger.info("user_registered", extra={"user_id": user.id})
        return user

    # ----------------------------------------------------------------- login
    async def authenticate(
        self, identifier: str, password: str, *, client: ClientInfo | None = None
    ) -> User:
        """Verify credentials, or raise a deliberately vague error."""
        client = client or ClientInfo()
        user = await self.users.get_by_identifier(identifier)

        if user is None:
            # Same work as a real verification so timing does not leak existence.
            verify_password(password, _DUMMY_HASH)
            await self._deny(None, identifier, client, "unknown_user")

        assert user is not None
        now = utcnow()
        if user.is_locked(now):
            await self._deny(user, identifier, client, "locked")
        if not user.is_active:
            await self._deny(user, identifier, client, "inactive")

        if not verify_password(password, user.password_hash):
            locked = await self.users.register_failed_login(user, config=self.config)
            if locked:
                await self.audit.record(
                    AuditAction.ACCOUNT_LOCKED,
                    user_id=user.id,
                    actor=user.username,
                    client_ip=client.ip,
                    success=False,
                    detail={"reason": "too_many_failed_logins"},
                )
            await self._deny(user, identifier, client, "bad_password")

        # Transparently upgrade the stored hash when parameters have changed.
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)

        await self.users.register_successful_login(user)
        await self.audit.record(
            AuditAction.LOGIN_SUCCESS,
            user_id=user.id,
            actor=user.username,
            client_ip=client.ip,
            user_agent=client.user_agent,
        )
        return user

    async def _deny(
        self, user: User | None, identifier: str, client: ClientInfo, reason: str
    ) -> None:
        auth_failures_total.inc(labels={"reason": reason})
        await self.audit.record(
            AuditAction.LOGIN_FAILURE,
            user_id=user.id if user else None,
            actor=identifier[:160],
            client_ip=client.ip,
            user_agent=client.user_agent,
            success=False,
            detail={"reason": reason},
        )
        # One message for every failure mode - no account enumeration.
        raise AuthenticationError("Invalid credentials.")

    # ---------------------------------------------------------------- tokens
    def issue_tokens(self, user: User) -> TokenResponse:
        """Mint an access/refresh pair bound to the user's token version."""
        claims = {"ver": user.token_version, "username": user.username}
        return TokenResponse(
            access_token=create_access_token(
                user.id, user.role_enum, extra_claims=claims, config=self.config
            ),
            refresh_token=create_refresh_token(
                user.id, user.role_enum, extra_claims=claims, config=self.config
            ),
            expires_in=self.config.access_token_expire_minutes * 60,
        )

    async def refresh(
        self, refresh_token: str, *, client: ClientInfo | None = None
    ) -> tuple[User, TokenResponse]:
        """Exchange a valid refresh token for a new pair."""
        client = client or ClientInfo()
        payload = decode_token(refresh_token, expected_type=TokenType.REFRESH, config=self.config)
        user = await self._user_for(payload)
        await self.audit.record(
            AuditAction.TOKEN_REFRESH,
            user_id=user.id,
            actor=user.username,
            client_ip=client.ip,
        )
        return user, self.issue_tokens(user)

    async def resolve_token(self, token: str) -> User:
        """Validate an access token and return the user it belongs to."""
        payload = decode_token(token, expected_type=TokenType.ACCESS, config=self.config)
        return await self._user_for(payload)

    async def _user_for(self, payload: TokenPayload) -> User:
        user_id = payload.user_id
        if user_id is None:
            raise AuthenticationError("Token is invalid.")
        user = await self.users.get(user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("Account is unavailable.")
        if user.is_locked(utcnow()):
            raise AuthenticationError("Account is locked.")
        # A token issued before the last credential change is no longer valid.
        if int(payload.raw.get("ver", 0)) != user.token_version:
            raise AuthenticationError("Token has been revoked.")
        return user

    # -------------------------------------------------------------- password
    async def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
        *,
        client: ClientInfo | None = None,
    ) -> None:
        """Change a password and revoke every outstanding token."""
        client = client or ClientInfo()
        if not verify_password(current_password, user.password_hash):
            auth_failures_total.inc(labels={"reason": "password_change"})
            await self.audit.record(
                AuditAction.PASSWORD_CHANGED,
                user_id=user.id,
                actor=user.username,
                client_ip=client.ip,
                success=False,
            )
            raise AuthenticationError("Current password is incorrect.")
        if current_password == new_password:
            raise ValidationError("The new password must differ from the current one.")

        user.password_hash = hash_password(new_password)
        await self.users.revoke_tokens(user)
        await self.audit.record(
            AuditAction.PASSWORD_CHANGED,
            user_id=user.id,
            actor=user.username,
            client_ip=client.ip,
        )
        logger.info("password_changed", extra={"user_id": user.id})

    async def logout(self, user: User, *, client: ClientInfo | None = None) -> None:
        """Revoke this user's tokens."""
        client = client or ClientInfo()
        await self.users.revoke_tokens(user)
        await self.audit.record(
            AuditAction.LOGOUT, user_id=user.id, actor=user.username, client_ip=client.ip
        )


__all__ = ["AuthService", "ClientInfo"]
