"""Date parsing for the many formats news sources emit.

RSS uses RFC 822, JSON feeds use ISO 8601, scraped pages use whatever the CMS
felt like. Everything is coerced to timezone-aware UTC; anything unparseable
returns ``None`` so the caller decides the fallback rather than silently
inventing a timestamp.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final

from app.core.logging import get_logger
from app.core.utils import ensure_utc, utcnow

logger = get_logger(__name__)

#: Explicit patterns tried before falling back to fuzzy parsing.
_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d %B %Y %H:%M",
    "%d %B %Y",
    "%B %d, %Y %H:%M",
    "%B %d, %Y",
    "%b %d, %Y",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
)

_RELATIVE_RE = re.compile(
    r"(?i)\b(\d{1,3})\s*(second|sec|minute|min|hour|hr|day|week|month)s?\s+ago\b"
)
_RELATIVE_UNITS: Final[dict[str, float]] = {
    "second": 1 / 3600,
    "sec": 1 / 3600,
    "minute": 1 / 60,
    "min": 1 / 60,
    "hour": 1.0,
    "hr": 1.0,
    "day": 24.0,
    "week": 168.0,
    "month": 730.0,
}

#: Bounds outside which a timestamp is treated as corrupt.
_MIN_PLAUSIBLE = datetime(1995, 1, 1, tzinfo=UTC)
_MAX_FUTURE_SKEW = timedelta(hours=6)


def parse_datetime(value: Any, *, default: datetime | None = None) -> datetime | None:
    """Best-effort conversion of ``value`` to an aware UTC datetime."""
    if value is None or value == "":
        return default
    if isinstance(value, datetime):
        return _bounded(ensure_utc(value))
    if isinstance(value, (int, float)):
        return _from_timestamp(float(value)) or default
    if not isinstance(value, str):
        return default

    text = value.strip()
    if not text:
        return default

    parsed = (
        _try_iso(text)
        or _try_formats(text)
        or _try_rfc822(text)
        or _try_relative(text)
        or _try_numeric_string(text)
    )
    if parsed is None:
        logger.debug("unparseable_date", extra={"sample": text[:64]})
        return default
    return _bounded(parsed)


def _try_iso(text: str) -> datetime | None:
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return ensure_utc(datetime.fromisoformat(candidate))
    except ValueError:
        return None


def _try_formats(text: str) -> datetime | None:
    for fmt in _FORMATS:
        try:
            return ensure_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _try_rfc822(text: str) -> datetime | None:
    try:
        return ensure_utc(parsedate_to_datetime(text))
    except (TypeError, ValueError, IndexError):
        return None


def _try_relative(text: str) -> datetime | None:
    match = _RELATIVE_RE.search(text)
    if not match:
        return None
    amount = int(match.group(1))
    hours = _RELATIVE_UNITS.get(match.group(2).lower())
    if hours is None:
        return None
    return utcnow() - timedelta(hours=amount * hours)


def _try_numeric_string(text: str) -> datetime | None:
    if text.isdigit():
        return _from_timestamp(float(text))
    return None


def _from_timestamp(value: float) -> datetime | None:
    """Accept both seconds and milliseconds since the epoch."""
    if value > 1e12:  # milliseconds
        value /= 1000.0
    if value <= 0 or value > 4.1e9:  # beyond ~2100
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _bounded(value: datetime | None) -> datetime | None:
    """Reject timestamps outside the plausible range."""
    if value is None:
        return None
    if value < _MIN_PLAUSIBLE:
        return None
    if value > utcnow() + _MAX_FUTURE_SKEW:
        return None
    return value


def is_recent(value: datetime | None, hours: float) -> bool:
    """True when ``value`` falls within the last ``hours``."""
    if value is None:
        return False
    aware = ensure_utc(value)
    assert aware is not None
    return aware >= utcnow() - timedelta(hours=hours)


__all__ = ["is_recent", "parse_datetime"]
