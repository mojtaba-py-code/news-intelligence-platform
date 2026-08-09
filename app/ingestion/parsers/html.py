"""Configurable HTML article extraction.

Website structures differ, so extraction is driven by :class:`ExtractionRules`
- CSS selectors declared in the source's config. Every field falls back through
a chain: explicit selector, then JSON-LD ``NewsArticle`` metadata, then
OpenGraph/meta tags, then structural heuristics. That ordering means a source
works out of the box and can be sharpened with selectors when needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.core.logging import get_logger
from app.processing.cleaning.html_clean import (
    MAX_HTML_BYTES,
    extract_main_text,
    extract_meta,
    normalize_whitespace,
    text_from_node,
)

logger = get_logger(__name__)

MAX_LINKS: Final[int] = 200


@dataclass(slots=True)
class ExtractionRules:
    """CSS selectors for one site. Every field is optional."""

    title: str | None = None
    description: str | None = None
    content: str | None = None
    author: str | None = None
    published_at: str | None = None
    image: str | None = None
    category: str | None = None
    #: Selector for links on an index page (used by the crawl-then-fetch flow).
    article_links: str | None = None
    #: Only follow links whose URL contains one of these fragments.
    link_must_contain: tuple[str, ...] = ()
    #: Attribute holding the date when it is not the element's text.
    date_attribute: str = "datetime"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> ExtractionRules:
        if not config:
            return cls()
        selectors = config.get("selectors", config)
        values: dict[str, Any] = {}
        for key in cls.__slots__:
            value = selectors.get(key)
            if value is None:
                continue
            if key == "link_must_contain":
                values[key] = (
                    tuple(str(item) for item in value) if isinstance(value, list) else (str(value),)
                )
            else:
                values[key] = str(value)
        return cls(**values)


class HTMLArticleParser:
    """Extracts article fields from a rendered HTML page."""

    def __init__(self, rules: ExtractionRules | None = None) -> None:
        self.rules = rules or ExtractionRules()

    def parse(self, markup: str, *, url: str) -> dict[str, Any]:
        """Return an ``RawArticle``-shaped dictionary (minus source identity)."""
        soup = BeautifulSoup(markup[:MAX_HTML_BYTES], "html.parser")
        meta = extract_meta(markup)
        json_ld = self._json_ld(soup)

        title = (
            self._select_text(soup, self.rules.title)
            or json_ld.get("headline")
            or meta.get("title")
            or text_from_node(soup.find("h1"))
        )
        description = (
            self._select_text(soup, self.rules.description)
            or json_ld.get("description")
            or meta.get("description")
        )
        content = self._select_text(soup, self.rules.content, block=True) or extract_main_text(
            markup
        )
        author = (
            self._select_text(soup, self.rules.author)
            or _person_name(json_ld.get("author"))
            or meta.get("author")
        )
        published = (
            self._select_date(soup, self.rules.published_at)
            or json_ld.get("datePublished")
            or meta.get("published_at")
        )
        updated = json_ld.get("dateModified") or meta.get("updated_at")
        image = (
            self._select_attr(soup, self.rules.image, "src")
            or _image_url(json_ld.get("image"))
            or meta.get("image_url")
        )
        category = self._select_text(soup, self.rules.category) or json_ld.get("articleSection")

        return {
            "title": normalize_whitespace(title or "")[:2000] or None,
            "description": normalize_whitespace(description or "")[:20_000] or None,
            "content": content or None,
            "author": normalize_whitespace(author or "")[:400] or None,
            "published_at": published or None,
            "updated_at": updated or None,
            "image_url": _absolute(image, url),
            "category": normalize_whitespace(category or "")[:64] or None,
            "language": meta.get("language"),
            "canonical_url": _absolute(meta.get("canonical_url"), url) or url,
        }

    def extract_links(self, markup: str, *, base_url: str) -> list[str]:
        """Article URLs from an index page, filtered by the source's rules."""
        soup = BeautifulSoup(markup[:MAX_HTML_BYTES], "html.parser")
        anchors: list[Tag] = []
        if self.rules.article_links:
            try:
                anchors = [
                    node for node in soup.select(self.rules.article_links) if isinstance(node, Tag)
                ]
            except Exception:
                logger.warning("bad_link_selector", extra={"selector": self.rules.article_links})
        if not anchors:
            anchors = [node for node in soup.find_all("a", href=True) if isinstance(node, Tag)]

        seen: set[str] = set()
        links: list[str] = []
        for anchor in anchors:
            href = str(anchor.get("href", "")).strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
                continue
            absolute = urljoin(base_url, href)
            if not absolute.startswith(("http://", "https://")):
                continue
            if self.rules.link_must_contain and not any(
                fragment in absolute for fragment in self.rules.link_must_contain
            ):
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append(absolute)
            if len(links) >= MAX_LINKS:
                break
        return links

    # ------------------------------------------------------------------ inner
    @staticmethod
    def _select(soup: BeautifulSoup, selector: str | None) -> Tag | None:
        if not selector:
            return None
        try:
            node = soup.select_one(selector)
        except Exception:
            logger.warning("bad_selector", extra={"selector": selector[:80]})
            return None
        return node if isinstance(node, Tag) else None

    def _select_text(
        self, soup: BeautifulSoup, selector: str | None, *, block: bool = False
    ) -> str | None:
        node = self._select(soup, selector)
        if node is None:
            return None
        if block:
            return extract_main_text(str(node))
        return text_from_node(node) or None

    def _select_attr(self, soup: BeautifulSoup, selector: str | None, attribute: str) -> str | None:
        node = self._select(soup, selector)
        if node is None:
            return None
        for candidate in (attribute, "content", "data-src", "href"):
            value = node.get(candidate)
            if value:
                return str(value)
        return None

    def _select_date(self, soup: BeautifulSoup, selector: str | None) -> str | None:
        node = self._select(soup, selector)
        if node is None:
            return None
        attribute = node.get(self.rules.date_attribute) or node.get("content")
        return str(attribute) if attribute else (text_from_node(node) or None)

    @staticmethod
    def _json_ld(soup: BeautifulSoup) -> dict[str, Any]:
        """Merge any ``NewsArticle``/``Article`` JSON-LD blocks on the page."""
        merged: dict[str, Any] = {}
        for script in soup.find_all("script", attrs={"type": "application/ld+json"})[:10]:
            raw = script.string or script.get_text() or ""
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            for block in _iter_ld_blocks(data):
                block_type = str(block.get("@type", "")).lower()
                if block_type in ("newsarticle", "article", "reportagenewsarticle", "blogposting"):
                    for key, value in block.items():
                        merged.setdefault(key, value)
        return merged


def _iter_ld_blocks(data: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Flatten JSON-LD, which may be an object, a list, or use ``@graph``."""
    if depth > 4:
        return []
    if isinstance(data, list):
        blocks: list[dict[str, Any]] = []
        for item in data[:20]:
            blocks.extend(_iter_ld_blocks(item, depth + 1))
        return blocks
    if isinstance(data, dict):
        if "@graph" in data:
            return _iter_ld_blocks(data["@graph"], depth + 1)
        return [data]
    return []


def _person_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else None
    if isinstance(value, list) and value:
        return _person_name(value[0])
    return None


def _image_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        url = value.get("url") or value.get("contentUrl")
        return str(url) if url else None
    if isinstance(value, list) and value:
        return _image_url(value[0])
    return None


def _absolute(url: str | None, base: str) -> str | None:
    if not url:
        return None
    candidate = urljoin(base, str(url).strip())
    return candidate if candidate.startswith(("http://", "https://")) else None


__all__ = ["ExtractionRules", "HTMLArticleParser"]
