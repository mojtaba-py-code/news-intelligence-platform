"""Password hashing, JWT handling and the role hierarchy."""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest

from app.core.config import get_settings
from app.core.errors import AuthenticationError, ValidationError
from app.core.security import (
    AUDIENCE,
    ISSUER,
    MAX_PASSWORD_LENGTH,
    Role,
    TokenType,
    constant_time_compare,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    validate_password_strength,
    verify_password,
)

pytestmark = pytest.mark.unit


class TestPasswordHashing:
    def test_hash_is_argon2id_and_verifies(self) -> None:
        digest = hash_password("Str0ng-Passw0rd!x")
        assert digest.startswith("$argon2id$")
        assert verify_password("Str0ng-Passw0rd!x", digest)

    def test_hash_is_salted(self) -> None:
        assert hash_password("same-password") != hash_password("same-password")

    def test_wrong_password_returns_false(self) -> None:
        digest = hash_password("correct-horse-battery")
        assert verify_password("wrong", digest) is False

    def test_unicode_is_normalised(self) -> None:
        # NFKC: the composed and decomposed forms must hash identically.
        digest = hash_password("café-Passw0rd!")
        assert verify_password("café-Passw0rd!", digest)

    def test_verify_never_raises_on_garbage(self) -> None:
        assert verify_password("x", "not-a-hash") is False
        assert verify_password("", "") is False

    def test_oversized_password_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            hash_password("a" * (MAX_PASSWORD_LENGTH + 1))

    def test_oversized_password_fails_verification_without_hashing(self) -> None:
        digest = hash_password("Str0ng-Passw0rd!x")
        assert verify_password("a" * (MAX_PASSWORD_LENGTH + 1), digest) is False


class TestPasswordPolicy:
    def test_strong_password_passes(self) -> None:
        assert validate_password_strength("Str0ng-Passw0rd!x").ok

    @pytest.mark.parametrize(
        "password",
        ["short1!A", "alllowercase1!", "ALLUPPERCASE1!", "NoDigits!!!!", "NoSymbols1234"],
    )
    def test_weak_passwords_rejected(self, password: str) -> None:
        assert validate_password_strength(password).ok is False

    def test_common_password_rejected(self) -> None:
        result = validate_password_strength("password")
        assert result.ok is False
        assert any("common" in problem for problem in result.problems)

    def test_repeated_characters_rejected(self) -> None:
        result = validate_password_strength("Aaaaa1!bcdefg")
        assert any("repeat" in problem for problem in result.problems)


class TestJWT:
    def test_roundtrip_preserves_claims(self, user_id: int = 42) -> None:
        token = create_access_token(user_id, Role.ANALYST, extra_claims={"ver": 3})
        payload = decode_token(token)
        assert payload.user_id == user_id
        assert payload.role is Role.ANALYST
        assert payload.token_type is TokenType.ACCESS
        assert payload.raw["ver"] == 3
        assert payload.raw["iss"] == ISSUER
        assert payload.raw["aud"] == AUDIENCE

    def test_refresh_token_rejected_as_access_token(self) -> None:
        token = create_refresh_token(1, Role.USER)
        with pytest.raises(AuthenticationError):
            decode_token(token, expected_type=TokenType.ACCESS)
        assert decode_token(token, expected_type=TokenType.REFRESH).user_id == 1

    def test_expired_token_rejected(self) -> None:
        token = create_access_token(1, Role.USER, expires_delta=timedelta(seconds=-10))
        with pytest.raises(AuthenticationError, match="expired"):
            decode_token(token)

    def test_tampered_signature_rejected(self) -> None:
        token = create_access_token(1, Role.USER)
        header, payload, signature = token.split(".")
        forged = f"{header}.{payload}.{signature[:-4]}abcd"
        with pytest.raises(AuthenticationError):
            decode_token(forged)

    def test_alg_none_rejected(self) -> None:
        """The classic JWT bypass: an unsigned token claiming ``alg: none``."""
        forged = jwt.encode({"sub": "1", "role": "ADMIN"}, key="", algorithm="none")
        with pytest.raises(AuthenticationError):
            decode_token(forged)

    def test_token_signed_with_other_secret_rejected(self) -> None:
        forged = jwt.encode(
            {
                "sub": "1",
                "role": "ADMIN",
                "typ": "access",
                "jti": "x",
                "iat": 0,
                "nbf": 0,
                "exp": 9_999_999_999,
                "iss": ISSUER,
                "aud": AUDIENCE,
            },
            "attacker-secret",
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError):
            decode_token(forged)

    def test_wrong_audience_rejected(self) -> None:
        config = get_settings()
        forged = jwt.encode(
            {
                "sub": "1",
                "role": "USER",
                "typ": "access",
                "jti": "x",
                "iat": 0,
                "nbf": 0,
                "exp": 9_999_999_999,
                "iss": ISSUER,
                "aud": "some-other-api",
            },
            config.jwt_secret_key.get_secret_value(),
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError):
            decode_token(forged)

    def test_reserved_claims_cannot_be_overridden(self) -> None:
        token = create_access_token(
            5, Role.USER, extra_claims={"sub": "1", "role": "ADMIN", "custom": "ok"}
        )
        payload = decode_token(token)
        assert payload.user_id == 5
        assert payload.role is Role.USER
        assert payload.raw["custom"] == "ok"

    def test_empty_token_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            decode_token("")

    def test_each_token_has_a_unique_jti(self) -> None:
        first = decode_token(create_access_token(1, Role.USER))
        second = decode_token(create_access_token(1, Role.USER))
        assert first.jti != second.jti


class TestRoles:
    def test_hierarchy(self) -> None:
        assert Role.ADMIN.can_act_as(Role.ANALYST)
        assert Role.ADMIN.can_act_as(Role.USER)
        assert Role.ANALYST.can_act_as(Role.USER)
        assert not Role.USER.can_act_as(Role.ANALYST)
        assert not Role.ANALYST.can_act_as(Role.ADMIN)

    def test_role_satisfies_itself(self) -> None:
        for role in Role:
            assert role.can_act_as(role)


class TestMisc:
    def test_api_key_is_hashed_not_stored(self) -> None:
        raw, digest = generate_api_key("nip")
        assert raw.startswith("nip_")
        assert raw not in digest
        assert verify_password(raw, digest)

    def test_constant_time_compare(self) -> None:
        assert constant_time_compare("abc", "abc")
        assert not constant_time_compare("abc", "abd")
