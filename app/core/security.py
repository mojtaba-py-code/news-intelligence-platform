"""Authentication primitives: password hashing, JWT issuing/verification, RBAC.

Design notes
------------
* **Argon2id** is the password KDF (memory-hard, OWASP's first recommendation).
  Hashes are self-describing, so parameters can be tuned later and old hashes
  are transparently re-hashed on the next successful login.
* **JWTs are pinned to a single algorithm.** ``jwt.decode`` is called with an
  explicit one-element ``algorithms`` list, which defeats both ``alg: none``
  and HS/RS confusion attacks.
* Access and refresh tokens carry a ``typ`` claim and are validated against the
  expected type, so a refresh token can never be replayed as an access token.
* Every token carries a ``jti`` so individual tokens can be revoked.
"""

from __future__ import annotations

import hmac
import re
import secrets
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import HashingError, InvalidHashError, VerifyMismatchError
from argon2.low_level import Type as Argon2Type

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, ValidationError

# --------------------------------------------------------------------------- #
# Roles & permissions
# --------------------------------------------------------------------------- #


class Role(StrEnum):
    """Platform roles, ordered from least to most privileged."""

    USER = "USER"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"

    @property
    def level(self) -> int:
        return _ROLE_LEVELS[self]

    def can_act_as(self, required: Role) -> bool:
        """Role hierarchy check: ADMIN satisfies ANALYST satisfies USER."""
        return self.level >= required.level


_ROLE_LEVELS: Final[dict[Role, int]] = {Role.USER: 0, Role.ANALYST: 10, Role.ADMIN: 20}


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #

# OWASP-aligned Argon2id parameters: 64 MiB, 3 passes, 4 lanes.
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65_536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Argon2Type.ID,
)

#: Longest password we will hash. Unbounded input is a trivial DoS against a
#: memory-hard KDF, so cap it well above any realistic passphrase.
MAX_PASSWORD_LENGTH: Final[int] = 1024

_COMMON_PASSWORDS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "password1",
        "passw0rd",
        "123456789",
        "1234567890",
        "qwertyuiop",
        "administrator",
        "changeme",
        "letmein123",
        "welcome123",
        "iloveyou",
        "adminadmin",
        "newsplatform",
    }
)


def hash_password(password: str) -> str:
    """Hash a plaintext password with Argon2id."""
    if not isinstance(password, str) or not password:
        raise ValidationError("Password must be a non-empty string.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValidationError(f"Password exceeds {MAX_PASSWORD_LENGTH} characters.")
    try:
        return _hasher.hash(_normalise(password))
    except HashingError as exc:  # pragma: no cover - argon2 internal failure
        raise ValidationError("Password could not be hashed.") from exc


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time-ish verification. Never raises on a wrong password."""
    if not password or not password_hash or len(password) > MAX_PASSWORD_LENGTH:
        return False
    try:
        return _hasher.verify(password_hash, _normalise(password))
    except (VerifyMismatchError, InvalidHashError, ValueError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash uses outdated Argon2 parameters."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except (InvalidHashError, ValueError):
        return True


def _normalise(password: str) -> str:
    """NFKC-normalise so visually identical passwords hash identically."""
    return unicodedata.normalize("NFKC", password)


@dataclass(frozen=True, slots=True)
class PasswordPolicyResult:
    ok: bool
    problems: tuple[str, ...] = ()


def validate_password_strength(
    password: str, *, config: Settings | None = None
) -> PasswordPolicyResult:
    """Enforce length, character-class variety and a common-password blocklist."""
    config = config or get_settings()
    problems: list[str] = []

    if len(password) < config.password_min_length:
        problems.append(f"must be at least {config.password_min_length} characters long")
    if len(password) > MAX_PASSWORD_LENGTH:
        problems.append(f"must be at most {MAX_PASSWORD_LENGTH} characters long")
    if not re.search(r"[a-z]", password):
        problems.append("must contain a lowercase letter")
    if not re.search(r"[A-Z]", password):
        problems.append("must contain an uppercase letter")
    if not re.search(r"\d", password):
        problems.append("must contain a digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("must contain a symbol")
    if password.lower() in _COMMON_PASSWORDS:
        problems.append("is a commonly used password")
    if re.search(r"(.)\1{3,}", password):
        problems.append("must not repeat the same character four or more times")

    return PasswordPolicyResult(ok=not problems, problems=tuple(problems))


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TokenPayload:
    """Validated JWT claims."""

    subject: str
    role: Role
    token_type: TokenType
    jti: str
    issued_at: datetime
    expires_at: datetime
    raw: dict[str, Any]

    @property
    def user_id(self) -> int | None:
        try:
            return int(self.subject)
        except (TypeError, ValueError):
            return None


ISSUER: Final[str] = "news-intelligence-platform"
AUDIENCE: Final[str] = "news-intelligence-api"


def create_token(
    subject: str | int,
    role: Role,
    token_type: TokenType = TokenType.ACCESS,
    *,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
    config: Settings | None = None,
) -> str:
    """Sign a JWT for ``subject``."""
    config = config or get_settings()
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = (
            timedelta(minutes=config.access_token_expire_minutes)
            if token_type is TokenType.ACCESS
            else timedelta(days=config.refresh_token_expire_days)
        )

    claims: dict[str, Any] = {
        "sub": str(subject),
        "role": str(role),
        "typ": str(token_type),
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": ISSUER,
        "aud": AUDIENCE,
    }
    if extra_claims:
        # Reserved claims are not overridable from the outside.
        reserved = set(claims)
        claims.update({k: v for k, v in extra_claims.items() if k not in reserved})

    return jwt.encode(
        claims, config.jwt_secret_key.get_secret_value(), algorithm=config.jwt_algorithm
    )


def create_access_token(subject: str | int, role: Role, **kwargs: Any) -> str:
    return create_token(subject, role, TokenType.ACCESS, **kwargs)


def create_refresh_token(subject: str | int, role: Role, **kwargs: Any) -> str:
    return create_token(subject, role, TokenType.REFRESH, **kwargs)


def decode_token(
    token: str,
    *,
    expected_type: TokenType | None = TokenType.ACCESS,
    config: Settings | None = None,
) -> TokenPayload:
    """Verify signature, standard claims and token type.

    Raises :class:`AuthenticationError` for every failure mode - the caller
    must not be able to distinguish "expired" from "forged" beyond the message
    we deliberately expose.
    """
    config = config or get_settings()
    if not token or not isinstance(token, str):
        raise AuthenticationError("Missing authentication token.")

    try:
        claims = jwt.decode(
            token,
            config.jwt_secret_key.get_secret_value(),
            algorithms=[config.jwt_algorithm],  # pinned - blocks alg confusion & "none"
            audience=AUDIENCE,
            issuer=ISSUER,
            options={
                "require": ["exp", "iat", "nbf", "sub", "jti", "iss", "aud"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Token is invalid.") from exc

    try:
        token_type = TokenType(claims["typ"])
        role = Role(claims["role"])
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Token is invalid.") from exc

    if expected_type is not None and token_type is not expected_type:
        raise AuthenticationError(f"Expected a {expected_type} token.")

    return TokenPayload(
        subject=str(claims["sub"]),
        role=role,
        token_type=token_type,
        jti=str(claims["jti"]),
        issued_at=datetime.fromtimestamp(claims["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
        raw=claims,
    )


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def generate_api_key(prefix: str = "nip") -> tuple[str, str]:
    """Return ``(plaintext_key, storable_hash)`` for a machine credential.

    The plaintext is shown to the user once; only the hash is persisted.
    """
    raw = f"{prefix}_{secrets.token_urlsafe(32)}"
    return raw, hash_password(raw)


def constant_time_compare(left: str, right: str) -> bool:
    """Timing-safe string comparison."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


__all__ = [
    "AUDIENCE",
    "ISSUER",
    "MAX_PASSWORD_LENGTH",
    "PasswordPolicyResult",
    "Role",
    "TokenPayload",
    "TokenType",
    "constant_time_compare",
    "create_access_token",
    "create_refresh_token",
    "create_token",
    "decode_token",
    "generate_api_key",
    "hash_password",
    "needs_rehash",
    "validate_password_strength",
    "verify_password",
]
