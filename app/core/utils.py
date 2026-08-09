"""Small shared helpers: time, hashing, chunking, safe coercions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

T = TypeVar("T")

_WHITESPACE_RE = re.compile(r"\s+")


def utcnow() -> datetime:
    """Timezone-aware current UTC time (single indirection point for tests)."""
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to naive datetimes and convert aware ones."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def hours_ago(hours: float) -> datetime:
    return utcnow() - timedelta(hours=hours)


def days_ago(days: float) -> datetime:
    return utcnow() - timedelta(days=days)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a string (used for content/title fingerprints)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_for_hash(text: str) -> str:
    """Case-folded, whitespace-collapsed, punctuation-free form for hashing.

    Two articles that differ only by smart quotes or extra spacing must produce
    the same fingerprint, otherwise deduplication misses obvious duplicates.
    """
    if not text:
        return ""
    lowered = text.casefold()
    stripped = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def content_fingerprint(text: str) -> str:
    """Stable hash of normalised content."""
    return sha256_text(normalize_for_hash(text))


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield ``size``-sized slices; the last chunk may be shorter."""
    if size < 1:
        raise ValueError("size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def truncate(text: str | None, limit: int, suffix: str = "…") -> str:
    """Cut ``text`` to ``limit`` characters on a word boundary when possible."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(suffix))]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + suffix


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result  # NaN check


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def dedupe_preserving_order(items: Iterable[T]) -> list[T]:
    """Remove duplicates while keeping first-seen order."""
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def percentage_change(previous: float, current: float) -> float:
    """Growth in percent; a jump from zero is reported as +100% per new unit."""
    if previous == 0:
        return 100.0 * current if current else 0.0
    return ((current - previous) / abs(previous)) * 100.0


__all__ = [
    "chunked",
    "clamp",
    "content_fingerprint",
    "days_ago",
    "dedupe_preserving_order",
    "ensure_utc",
    "hours_ago",
    "normalize_for_hash",
    "percentage_change",
    "safe_float",
    "safe_int",
    "sha256_text",
    "truncate",
    "utcnow",
]
