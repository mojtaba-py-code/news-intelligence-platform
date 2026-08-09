"""Structured logging with automatic secret redaction.

The platform handles API keys, JWTs, passwords and authorization headers. A log
line is the easiest place to leak them, so redaction happens inside a logging
*filter*: it applies to every record regardless of which module emitted it,
including third-party libraries.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings, get_settings

#: Correlation id for the in-flight request, set by the middleware.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

REDACTED = "***REDACTED***"

#: Keys whose values must never be written to a log, whatever the nesting.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "new_password",
        "current_password",
        "password_hash",
        "hashed_password",
        "secret",
        "secret_key",
        "jwt_secret_key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "x-api-key",
        "authorization",
        "auth",
        "cookie",
        "set-cookie",
        "session",
        "private_key",
        "client_secret",
        "smtp_password",
        "credentials",
    }
)

#: Patterns that catch secrets embedded in free-form strings.
#: The replacement is either a template string or a callable (``re.sub`` accepts both).
_PATTERNS: tuple[tuple[re.Pattern[str], Any], ...] = (
    # Authorization: Bearer <jwt|opaque>
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 " + REDACTED),
    # key=value / key: value in query strings, URLs and free text
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|token|password|secret"
            r"|client[_-]?secret|authorization)\b(\s*[=:]\s*|=)([\"']?)([^\s,&\"';]{3,})\3"
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}",
    ),
    # Bare JWTs anywhere in the message
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), REDACTED),
    # userinfo in URLs: scheme://user:pass@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@"), r"\1\2:" + REDACTED + "@"),
)

_MAX_STR = 4096


def redact_text(text: str) -> str:
    """Strip credential-looking substrings from a free-form string."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive values in mappings, sequences and strings."""
    if _depth > 8:
        return "***TRUNCATED***"
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and key.lower() in SENSITIVE_KEYS
                else redact(item, _depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact(item, _depth + 1) for item in value)  # type: ignore[call-arg]
    if isinstance(value, str):
        return redact_text(value[:_MAX_STR])
    return value


class RedactionFilter(logging.Filter):
    """Applies :func:`redact` to the message, args and structured extras."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        elif isinstance(record.msg, dict):
            record.msg = redact(record.msg)
        if record.args:
            record.args = redact(record.args)  # type: ignore[assignment]
        for key, value in list(record.__dict__.items()):
            if key in _RESERVED:
                continue
            if key.lower() in SENSITIVE_KEYS:
                record.__dict__[key] = REDACTED
            else:
                record.__dict__[key] = redact(value)
        return True


class RequestContextFilter(logging.Filter):
    """Attaches the current request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()  # type: ignore[attr-defined]
        return True


_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName"}
)


class JSONFormatter(logging.Formatter):
    """One JSON object per line - ready for Loki/ELK/CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "request_id":
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly single-line output for local development."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"

    def __init__(self) -> None:
        super().__init__(self._FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and key != "request_id" and not key.startswith("_")
        }
        rid = getattr(record, "request_id", None)
        if rid:
            base = f"{base} [req={rid}]"
        if extras:
            rendered = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
            base = f"{base} | {rendered}"
        return base


def configure_logging(config: Settings | None = None) -> None:
    """Install handlers/filters on the root logger. Safe to call repeatedly."""
    config = config or get_settings()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter() if config.log_format == "json" else ConsoleFormatter())
    handler.addFilter(RedactionFilter())
    handler.addFilter(RequestContextFilter())

    root.addHandler(handler)
    root.setLevel(config.log_level)

    # Third-party loggers are noisy; keep them at WARNING unless we are debugging.
    noisy = ("httpx", "httpcore", "urllib3", "asyncio", "aiosqlite", "multipart")
    for name in noisy:
        logging.getLogger(name).setLevel(
            logging.DEBUG if config.log_level == "DEBUG" else logging.WARNING
        )
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if config.db_echo else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (``get_logger(__name__)``)."""
    return logging.getLogger(name)


__all__ = [
    "REDACTED",
    "SENSITIVE_KEYS",
    "ConsoleFormatter",
    "JSONFormatter",
    "RedactionFilter",
    "configure_logging",
    "get_logger",
    "redact",
    "redact_text",
    "request_id_ctx",
]
