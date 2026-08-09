"""RSS 2.0 / Atom 1.0 / RDF feed parsing.

``defusedxml`` is used rather than the stdlib ``ElementTree`` directly: feeds
are attacker-controlled XML, and the stdlib parser will happily process entity
declarations. That opens the door to billion-laughs denial of service and, with
external entities, local-file disclosure and SSRF. ``defusedxml`` refuses both.

A hand-written parser is preferred over ``feedparser`` here for three reasons:
it is ~200 lines, it keeps the XXE-safe parser choice explicit, and it produces
the platform's own :class:`FeedItem` shape directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import ParseError as DefusedParseError
from defusedxml.ElementTree import fromstring as safe_fromstring

from app.core.errors import ParseError
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_FEED_BYTES: Final[int] = 10_000_000
MAX_ITEMS: Final[int] = 500

_NS: Final[dict[str, str]] = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "media": "http://search.yahoo.com/mrss/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rss": "http://purl.org/rss/1.0/",
}


@dataclass(slots=True)
class FeedItem:
    """One entry from a feed, still in feed vocabulary."""

    title: str = ""
    link: str = ""
    description: str = ""
    content: str = ""
    author: str = ""
    published: str = ""
    updated: str = ""
    guid: str = ""
    categories: list[str] = field(default_factory=list)
    image: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return bool(self.title and self.link)


@dataclass(slots=True)
class ParsedFeed:
    """A parsed feed: channel metadata plus its items."""

    title: str = ""
    link: str = ""
    description: str = ""
    language: str = ""
    items: list[FeedItem] = field(default_factory=list)
    format: str = "unknown"


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(element: Element | None) -> str:
    if element is None:
        return ""
    parts = [element.text or ""]
    for child in element:
        parts.append(_text(child))
        parts.append(child.tail or "")
    return "".join(parts).strip()


def _find_child(parent: Element, *names: str) -> Element | None:
    """First direct child whose local name matches any of ``names``."""
    wanted = {name.lower() for name in names}
    for child in parent:
        if _localname(child.tag).lower() in wanted:
            return child
    return None


def _find_children(parent: Element, *names: str) -> list[Element]:
    wanted = {name.lower() for name in names}
    return [child for child in parent if _localname(child.tag).lower() in wanted]


def parse_feed(payload: str | bytes, *, source: str = "feed") -> ParsedFeed:
    """Parse an RSS/Atom/RDF document.

    Raises :class:`ParseError` when the document is not well-formed XML or does
    not look like a feed at all.
    """
    if isinstance(payload, bytes):
        if len(payload) > MAX_FEED_BYTES:
            raise ParseError(source, "Feed exceeds the maximum allowed size")
        text = payload.decode("utf-8", errors="replace")
    else:
        text = payload
    if len(text) > MAX_FEED_BYTES:
        raise ParseError(source, "Feed exceeds the maximum allowed size")

    stripped = text.lstrip("﻿ \t\r\n")
    if not stripped:
        raise ParseError(source, "Empty feed document")

    try:
        # forbid_dtd is not enabled: many legitimate feeds declare a doctype.
        # Entity *expansion* and external entities are blocked, which is what
        # actually matters for billion-laughs and XXE.
        root = safe_fromstring(stripped)
    except (DefusedParseError, ValueError) as exc:
        raise ParseError(source, f"Malformed XML: {exc}") from exc
    except Exception as exc:  # defusedxml raises its own security exceptions
        raise ParseError(source, f"Rejected XML document: {exc.__class__.__name__}") from exc

    if root is None:
        raise ParseError(source, "Empty XML document")

    root_name = _localname(root.tag).lower()
    if root_name == "rss":
        return _parse_rss(root)
    if root_name == "feed":
        return _parse_atom(root)
    if root_name == "rdf":
        return _parse_rdf(root)
    raise ParseError(source, f"Unrecognised feed root element '<{root_name}>'")


# --------------------------------------------------------------------------- #
# RSS 2.0
# --------------------------------------------------------------------------- #
def _parse_rss(root: Element) -> ParsedFeed:
    channel = _find_child(root, "channel")
    if channel is None:
        return ParsedFeed(format="rss")

    feed = ParsedFeed(
        title=_text(_find_child(channel, "title")),
        link=_text(_find_child(channel, "link")),
        description=_text(_find_child(channel, "description")),
        language=_text(_find_child(channel, "language")),
        format="rss",
    )
    for element in _find_children(channel, "item")[:MAX_ITEMS]:
        item = _rss_item(element)
        if item.is_usable:
            feed.items.append(item)
    return feed


def _rss_item(element: Element) -> FeedItem:
    guid_element = _find_child(element, "guid")
    link = _text(_find_child(element, "link"))
    guid = _text(guid_element)
    if not link and guid.startswith(("http://", "https://")):
        link = guid

    return FeedItem(
        title=_text(_find_child(element, "title")),
        link=link,
        description=_text(_find_child(element, "description", "summary")),
        content=_text(_find_child(element, "encoded")) or _text(_find_child(element, "content")),
        author=_text(_find_child(element, "creator", "author")),
        published=_text(_find_child(element, "pubDate", "date", "published")),
        updated=_text(_find_child(element, "updated", "modified")),
        guid=guid,
        categories=[_text(child) for child in _find_children(element, "category") if _text(child)][
            :10
        ],
        image=_extract_image(element),
        raw={},
    )


def _extract_image(element: Element) -> str:
    """Find an image URL across the several conventions feeds use."""
    for child in element:
        name = _localname(child.tag).lower()
        if name == "thumbnail" and child.get("url"):
            return str(child.get("url", ""))
        if name == "content" and child.get("url") and "image" in str(child.get("type", "")):
            return str(child.get("url", ""))
        if name == "enclosure":
            mime = str(child.get("type", ""))
            if mime.startswith("image/") and child.get("url"):
                return str(child.get("url", ""))
        if name == "image":
            url = _text(_find_child(child, "url")) or str(child.get("href", ""))
            if url:
                return url
    return ""


# --------------------------------------------------------------------------- #
# Atom 1.0
# --------------------------------------------------------------------------- #
def _parse_atom(root: Element) -> ParsedFeed:
    feed = ParsedFeed(
        title=_text(_find_child(root, "title")),
        link=_atom_link(root),
        description=_text(_find_child(root, "subtitle")),
        language=root.get("{http://www.w3.org/XML/1998/namespace}lang", ""),
        format="atom",
    )
    for element in _find_children(root, "entry")[:MAX_ITEMS]:
        author_element = _find_child(element, "author")
        item = FeedItem(
            title=_text(_find_child(element, "title")),
            link=_atom_link(element),
            description=_text(_find_child(element, "summary")),
            content=_text(_find_child(element, "content")),
            author=_text(_find_child(author_element, "name")) if author_element is not None else "",
            published=_text(_find_child(element, "published")),
            updated=_text(_find_child(element, "updated")),
            guid=_text(_find_child(element, "id")),
            categories=[
                str(child.get("term", "")) for child in _find_children(element, "category")
            ][:10],
            image=_extract_image(element),
        )
        if item.is_usable:
            feed.items.append(item)
    return feed


def _atom_link(element: Element) -> str:
    """Prefer ``rel="alternate"``; fall back to the first usable ``<link>``."""
    fallback = ""
    for child in _find_children(element, "link"):
        rel = str(child.get("rel", "alternate")).lower()
        href = str(child.get("href", "")).strip()
        if not href:
            continue
        if rel == "alternate":
            return href
        if not fallback and rel not in ("self", "edit", "hub"):
            fallback = href
    return fallback


# --------------------------------------------------------------------------- #
# RDF (RSS 1.0)
# --------------------------------------------------------------------------- #
def _parse_rdf(root: Element) -> ParsedFeed:
    channel = _find_child(root, "channel")
    feed = ParsedFeed(
        title=_text(_find_child(channel, "title")) if channel is not None else "",
        link=_text(_find_child(channel, "link")) if channel is not None else "",
        description=_text(_find_child(channel, "description")) if channel is not None else "",
        format="rdf",
    )
    for element in _find_children(root, "item")[:MAX_ITEMS]:
        item = _rss_item(element)
        if not item.link:
            item.link = str(element.get(f"{{{_NS['rdf']}}}about", ""))
        if item.is_usable:
            feed.items.append(item)
    return feed


__all__ = ["MAX_ITEMS", "FeedItem", "ParsedFeed", "parse_feed"]
