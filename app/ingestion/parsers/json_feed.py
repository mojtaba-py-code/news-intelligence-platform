"""Declarative mapping from an arbitrary JSON API response to :class:`RawArticle`.

News APIs all return "a list of articles" but never agree on the shape. Rather
than writing a bespoke parser per vendor, a source declares a
:class:`JSONFieldMapping` in its config - dotted paths from the response root to
each canonical field - and this module does the rest. Adding a new API becomes
configuration, not code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from app.core.errors import ParseError
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_ITEMS: Final[int] = 500
MAX_DEPTH: Final[int] = 8


@dataclass(slots=True)
class JSONFieldMapping:
    """Dotted paths from an item object to the canonical article fields.

    ``items_path`` locates the array inside the response envelope
    (``"articles"``, ``"data.results"``, ``""`` when the root *is* the array).
    Each field accepts a list of candidate paths, tried in order - APIs
    frequently populate one of several alternatives.
    """

    items_path: str = ""
    title: tuple[str, ...] = ("title", "headline")
    url: tuple[str, ...] = ("url", "link", "webUrl")
    description: tuple[str, ...] = ("description", "summary", "abstract", "excerpt")
    content: tuple[str, ...] = ("content", "body", "fullText", "articleBody")
    author: tuple[str, ...] = ("author", "byline", "creator", "author.name")
    published_at: tuple[str, ...] = (
        "publishedAt",
        "published_at",
        "pubDate",
        "date",
        "webPublicationDate",
    )
    updated_at: tuple[str, ...] = ("updatedAt", "updated_at", "lastModified")
    image_url: tuple[str, ...] = ("urlToImage", "image", "imageUrl", "thumbnail", "image.url")
    external_id: tuple[str, ...] = ("id", "uuid", "guid", "articleId")
    language: tuple[str, ...] = ("language", "lang")
    country: tuple[str, ...] = ("country", "countryCode")
    category: tuple[str, ...] = ("category", "section", "sectionName", "topic")
    source_name: tuple[str, ...] = ("source.name", "source", "publisher")
    #: Extra keys copied verbatim into ``RawArticle.raw`` for debugging.
    keep_raw: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> JSONFieldMapping:
        """Build a mapping from a source's ``config["mapping"]`` block."""
        if not config:
            return cls()
        values: dict[str, Any] = {}
        for key, value in config.items():
            if key == "items_path":
                values[key] = str(value)
            elif key in cls.__slots__:
                if isinstance(value, str):
                    values[key] = (value,)
                elif isinstance(value, (list, tuple)):
                    values[key] = tuple(str(item) for item in value)
        return cls(**values)


def get_path(payload: Any, path: str, *, depth: int = 0) -> Any:
    """Resolve a dotted path such as ``source.name`` or ``data.0.title``."""
    if not path:
        return payload
    if depth > MAX_DEPTH:
        return None

    current = payload
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            if not part.isdigit():
                return None
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
    return current


def first_value(payload: Any, paths: tuple[str, ...]) -> Any:
    """First non-empty value among ``paths``."""
    for path in paths:
        value = get_path(payload, path)
        if value not in (None, "", [], {}):
            return value
    return None


def coerce_text(value: Any, *, limit: int = 100_000) -> str | None:
    """Flatten whatever the API returned into a string."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned[:limit] or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("name", "title", "value", "text", "rendered", "url"):
            if key in value:
                return coerce_text(value[key], limit=limit)
        return None
    if isinstance(value, list):
        parts = [coerce_text(item, limit=limit) for item in value[:10]]
        joined = ", ".join(part for part in parts if part)
        return joined[:limit] or None
    return None


def extract_items(payload: Any, mapping: JSONFieldMapping, *, source: str = "api") -> list[Any]:
    """Locate the array of article objects inside a response envelope."""
    container = get_path(payload, mapping.items_path) if mapping.items_path else payload
    if container is None:
        raise ParseError(source, f"No items found at path '{mapping.items_path}'")
    if isinstance(container, dict):
        # Some APIs wrap the array one level deeper than documented.
        for key in ("items", "articles", "results", "data", "docs", "stories", "entries"):
            if isinstance(container.get(key), list):
                container = container[key]
                break
    if not isinstance(container, list):
        raise ParseError(source, "Expected a list of articles in the response")
    return container[:MAX_ITEMS]


def parse_json_items(
    payload: Any,
    mapping: JSONFieldMapping,
    *,
    source_slug: str,
    source_name: str,
) -> list[dict[str, Any]]:
    """Map a JSON response onto ``RawArticle``-shaped dictionaries.

    Returns dictionaries rather than models so the caller decides how to handle
    per-item validation failures (usually: count and skip).
    """
    items = extract_items(payload, mapping, source=source_slug)
    results: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, (dict, list)):
            continue
        title = coerce_text(first_value(item, mapping.title), limit=2000)
        url = coerce_text(first_value(item, mapping.url), limit=2048)
        if not title or not url:
            continue

        raw_extra = (
            {key: get_path(item, key) for key in mapping.keep_raw} if mapping.keep_raw else {}
        )

        results.append(
            {
                "source_slug": source_slug,
                "source_name": coerce_text(first_value(item, mapping.source_name)) or source_name,
                "title": title,
                "url": url,
                "external_id": coerce_text(first_value(item, mapping.external_id), limit=255),
                "description": coerce_text(first_value(item, mapping.description), limit=20_000),
                "content": coerce_text(first_value(item, mapping.content), limit=200_000),
                "author": coerce_text(first_value(item, mapping.author), limit=400),
                "image_url": coerce_text(first_value(item, mapping.image_url), limit=2048),
                "published_at": first_value(item, mapping.published_at),
                "updated_at": first_value(item, mapping.updated_at),
                "language": coerce_text(first_value(item, mapping.language), limit=16),
                "country": coerce_text(first_value(item, mapping.country), limit=16),
                "category": coerce_text(first_value(item, mapping.category), limit=64),
                "raw": raw_extra,
            }
        )

    return results


__all__ = [
    "JSONFieldMapping",
    "coerce_text",
    "extract_items",
    "first_value",
    "get_path",
    "parse_json_items",
]
