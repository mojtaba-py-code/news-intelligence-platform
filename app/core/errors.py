"""Custom exception hierarchy and structured error responses.

Two rules drive this module:

1. **Every** failure the platform raises deliberately is a :class:`PlatformError`
   subclass carrying a machine-readable ``code`` plus a *safe* message.
2. Unexpected exceptions never leak internals to the client - they are logged
   with a correlation id and answered with a generic 500 body.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorCode:
    """Stable, machine-readable error codes returned to API clients."""

    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UPSTREAM_ERROR = "upstream_error"
    UNSAFE_URL = "unsafe_url"
    SOURCE_ERROR = "source_error"
    CIRCUIT_OPEN = "circuit_open"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    INTERNAL_ERROR = "internal_error"


class PlatformError(Exception):
    """Base class for all deliberate platform failures."""

    status_code: int = 500
    code: str = ErrorCode.INTERNAL_ERROR
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {"code": self.code, "message": self.message, "details": self.details}
        }
        if request_id:
            payload["error"]["request_id"] = request_id
        return payload


# --------------------------------------------------------------------------- #
# Client-facing (4xx)
# --------------------------------------------------------------------------- #
class ValidationError(PlatformError):
    status_code = 422
    code = ErrorCode.VALIDATION_ERROR
    message = "The submitted data failed validation."


class NotFoundError(PlatformError):
    status_code = 404
    code = ErrorCode.NOT_FOUND
    message = "The requested resource was not found."


class ConflictError(PlatformError):
    status_code = 409
    code = ErrorCode.CONFLICT
    message = "The resource already exists or conflicts with current state."


class AuthenticationError(PlatformError):
    status_code = 401
    code = ErrorCode.UNAUTHENTICATED
    message = "Authentication is required or the credentials are invalid."


class AuthorizationError(PlatformError):
    status_code = 403
    code = ErrorCode.FORBIDDEN
    message = "You do not have permission to perform this action."


class RateLimitError(PlatformError):
    status_code = 429
    code = ErrorCode.RATE_LIMITED
    message = "Too many requests. Please slow down."

    def __init__(self, retry_after: int = 60, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.retry_after = retry_after


class PayloadTooLargeError(PlatformError):
    status_code = 413
    code = ErrorCode.PAYLOAD_TOO_LARGE
    message = "Request body exceeds the configured maximum size."


# --------------------------------------------------------------------------- #
# Ingestion / processing (5xx & internal)
# --------------------------------------------------------------------------- #
class SourceError(PlatformError):
    """A named source failed; the pipeline isolates and continues."""

    status_code = 502
    code = ErrorCode.SOURCE_ERROR
    message = "The news source could not be processed."

    def __init__(self, source: str, message: str | None = None, **kwargs: Any) -> None:
        self.source = source
        super().__init__(message or f"Source '{source}' failed.", **kwargs)
        self.details.setdefault("source", source)


class FetchError(SourceError):
    status_code = 502
    code = ErrorCode.UPSTREAM_ERROR
    message = "Upstream fetch failed."


class PermanentFetchError(FetchError):
    """An upstream failure that retrying cannot fix (404, 401, 400…)."""

    message = "Upstream returned a permanent error."


class ParseError(SourceError):
    status_code = 422
    code = ErrorCode.PROCESSING_ERROR
    message = "The source response could not be parsed."


class RateLimitedUpstreamError(FetchError):
    """Upstream asked us to slow down. Retryable, after backoff."""

    status_code = 429
    code = ErrorCode.RATE_LIMITED
    message = "The upstream source rate-limited our request."

    def __init__(self, source: str, retry_after: float = 60.0, **kwargs: Any) -> None:
        super().__init__(source, **kwargs)
        self.retry_after = retry_after


class CircuitOpenError(SourceError):
    status_code = 503
    code = ErrorCode.CIRCUIT_OPEN
    message = "Circuit breaker is open for this source."


class UnsafeURLError(PlatformError):
    """Raised by the SSRF guard - a URL points somewhere we refuse to fetch."""

    status_code = 400
    code = ErrorCode.UNSAFE_URL
    message = "The URL was rejected by the URL safety policy."


class ProcessingError(PlatformError):
    status_code = 500
    code = ErrorCode.PROCESSING_ERROR
    message = "Article processing failed."


class ConfigurationError(PlatformError):
    status_code = 500
    code = ErrorCode.CONFIGURATION_ERROR
    message = "The platform is misconfigured."


# --------------------------------------------------------------------------- #
# Response schema (documented in OpenAPI)
# --------------------------------------------------------------------------- #
class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["not_found"])
    message: str = Field(..., examples=["The requested resource was not found."])
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx API response."""

    error: ErrorDetail


__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "CircuitOpenError",
    "ConfigurationError",
    "ConflictError",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "FetchError",
    "NotFoundError",
    "ParseError",
    "PayloadTooLargeError",
    "PermanentFetchError",
    "PlatformError",
    "ProcessingError",
    "RateLimitError",
    "RateLimitedUpstreamError",
    "SourceError",
    "UnsafeURLError",
    "ValidationError",
]
