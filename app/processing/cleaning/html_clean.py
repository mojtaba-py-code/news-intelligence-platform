"""HTML cleaning, text extraction and sanitisation.

Every string that reaches this module is hostile until proven otherwise: it
comes from a third-party feed or a scraped page. Three distinct jobs, kept
separate because conflating them is how XSS happens:

``strip_html``
    Throw away all markup, keep the text. Used for titles/descriptions that we
    store as plain text.
``extract_main_text``
    Locate the article body and drop navigation/boilerplate.
``sanitize_fragment``
    Keep a *small allowlist* of formatting tags for the rare case where markup
    must survive. Anything not on the list - scripts, iframes, event handlers,
    ``javascript:`` URLs - is removed.

The parser is ``html.parser`` (stdlib): it never fetches external entities, so
there is no XXE/SSRF surface, unlike some lxml configurations.
"""

from __future__ import annotations

import html
import re
from typing import Final

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Hard ceiling on the HTML we are willing to parse (memory-exhaustion guard).
MAX_HTML_BYTES: Final[int] = 4_000_000

#: Elements removed with their entire subtree.
DROP_TAGS: Final[frozenset[str]] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "iframe",
        "frame",
        "frameset",
        "object",
        "embed",
        "applet",
        "canvas",
        "svg",
        "math",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "nav",
        "aside",
        "footer",
        "header",
        "menu",
        "dialog",
        "template",
        "link",
        "meta",
        "base",
    }
)

#: Tags allowed to survive :func:`sanitize_fragment`.
ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {"p", "br", "strong", "b", "em", "i", "u", "ul", "ol", "li", "blockquote", "h2", "h3", "h4"}
)

#: Class/id substrings that mark chrome rather than content.
BOILERPLATE_HINTS: Final[tuple[str, ...]] = (
    "nav",
    "menu",
    "sidebar",
    "footer",
    "header",
    "banner",
    "advert",
    "ads",
    "promo",
    "share",
    "social",
    "subscribe",
    "newsletter",
    "cookie",
    "consent",
    "related",
    "recommend",
    "comment",
    "breadcrumb",
    "pagination",
    "widget",
    "popup",
    "modal",
    "paywall",
)

#: Selectors, most specific first, that usually wrap the article body.
CONTENT_SELECTORS: Final[tuple[str, ...]] = (
    "article",
    "main",
    '[itemprop="articleBody"]',
    ".article-body",
    ".article__body",
    ".story-body",
    ".post-content",
    ".entry-content",
    ".content__article-body",
    "#article-body",
    ".article-content",
)

_WS_RE = re.compile(r"[ \t   ]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(share (this|on)|read more|advertisement|sign up|subscribe|follow us"
    r"|related articles?|photo:|image:|getty images|copyright \d{4})",
    re.IGNORECASE,
)


def _soup(markup: str) -> BeautifulSoup:
    """Parse with the stdlib parser (no external entity resolution)."""
    return BeautifulSoup(markup[:MAX_HTML_BYTES], "html.parser")


def looks_like_html(text: str) -> bool:
    """Cheap check so plain-text fields skip the parser entirely."""
    return bool(text) and ("<" in text and ">" in text)


def strip_html(text: str | None, *, unescape: bool = True) -> str:
    """Return the plain-text content of ``text``, markup removed."""
    if not text:
        return ""
    if not looks_like_html(text):
        return normalize_whitespace(html.unescape(text) if unescape else text)

    soup = _soup(text)
    for element in soup.find_all(list(DROP_TAGS)):
        element.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    return normalize_whitespace(soup.get_text(separator=" "))


def clean_html(text: str | None) -> str:
    """Alias of :func:`strip_html` kept for pipeline readability."""
    return strip_html(text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, normalise newlines, trim."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("​", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NEWLINES_RE.sub("\n\n", text).strip()


def _is_boilerplate(tag: Tag) -> bool:
    """Heuristic: does this element's class/id look like page chrome?"""
    # ``decompose()`` on an ancestor leaves descendants with ``attrs is None``,
    # and this runs while the tree is being pruned - hence the guard.
    attributes = getattr(tag, "attrs", None)
    if not attributes:
        return False
    classes = attributes.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    haystack = " ".join(
        [
            " ".join(str(item) for item in classes),
            str(attributes.get("id") or ""),
            str(attributes.get("role") or ""),
            str(attributes.get("data-testid") or ""),
        ]
    ).lower()
    return any(hint in haystack for hint in BOILERPLATE_HINTS)


def extract_main_text(markup: str | None, *, min_paragraph_chars: int = 40) -> str:
    """Extract the article body from a full HTML page.

    Strategy: drop known-bad elements, prefer a semantic content container, then
    keep paragraphs long enough to be prose. Short ``<p>`` elements are almost
    always captions, bylines or share prompts.
    """
    if not markup:
        return ""
    if not looks_like_html(markup):
        return normalize_whitespace(markup)

    soup = _soup(markup)
    for element in soup.find_all(list(DROP_TAGS)):
        element.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for element in list(soup.find_all(True)):
        if isinstance(element, Tag) and _is_boilerplate(element):
            element.decompose()

    container: Tag | None = None
    for selector in CONTENT_SELECTORS:
        try:
            found = soup.select_one(selector)
        except Exception as exc:
            logger.debug("selector_failed", extra={"selector": selector, "error": str(exc)})
            continue
        if found is not None and len(found.get_text(strip=True)) > 200:
            container = found
            break
    root: Tag | BeautifulSoup = container if container is not None else soup

    paragraphs: list[str] = []
    for node in root.find_all(["p", "h2", "h3", "li", "blockquote"]):
        text = normalize_whitespace(node.get_text(separator=" "))
        if not text or _BOILERPLATE_LINE_RE.match(text):
            continue
        if node.name in {"p", "blockquote"} and len(text) < min_paragraph_chars:
            continue
        paragraphs.append(text)

    if not paragraphs:
        # Fall back to the container's raw text rather than returning nothing.
        return normalize_whitespace(root.get_text(separator=" "))

    return dedupe_repeated_lines("\n\n".join(paragraphs))


def dedupe_repeated_lines(text: str) -> str:
    """Remove repeated paragraphs (feeds often duplicate the lede)."""
    seen: set[str] = set()
    kept: list[str] = []
    for block in text.split("\n\n"):
        fingerprint = block.strip().casefold()
        if len(fingerprint) > 25 and fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(block)
    return "\n\n".join(kept)


_UNSAFE_URL_SCHEME_RE = re.compile(r"^\s*(javascript|data|vbscript|file)\s*:", re.IGNORECASE)


def sanitize_fragment(markup: str | None) -> str:
    """Return ``markup`` reduced to an allowlist of formatting tags.

    Removes every attribute except ``href`` on ``<a>`` (and only for http/https
    or relative targets), which eliminates ``onerror=``-style handlers and
    ``javascript:`` URLs.
    """
    if not markup:
        return ""
    soup = _soup(markup)

    for element in soup.find_all(list(DROP_TAGS)):
        element.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for element in list(soup.find_all(True)):
        if not isinstance(element, Tag):
            continue
        name = element.name.lower()
        if name == "a":
            href = str(element.get("href", "")).strip()
            element.attrs = {}
            if href and not _UNSAFE_URL_SCHEME_RE.match(href):
                element["href"] = href
                element["rel"] = "noopener noreferrer nofollow"
                element["target"] = "_blank"
            else:
                element.unwrap()
                continue
        elif name in ALLOWED_TAGS:
            element.attrs = {}
        else:
            element.unwrap()

    return str(soup).strip()


def extract_meta(markup: str | None) -> dict[str, str]:
    """Pull the OpenGraph/standard metadata a scraper cares about."""
    if not markup:
        return {}
    soup = _soup(markup)
    meta: dict[str, str] = {}

    for tag in soup.find_all("meta"):
        if not isinstance(tag, Tag):
            continue
        key = str(tag.get("property") or tag.get("name") or "").strip().lower()
        value = str(tag.get("content") or "").strip()
        if key and value and key in _WANTED_META:
            meta.setdefault(_WANTED_META[key], value[:2048])

    canonical = soup.find("link", rel=lambda v: bool(v) and "canonical" in v)
    if isinstance(canonical, Tag) and canonical.get("href"):
        meta.setdefault("canonical_url", str(canonical["href"])[:2048])

    if "title" not in meta:
        title_tag = soup.find("title")
        if isinstance(title_tag, Tag):
            meta["title"] = normalize_whitespace(title_tag.get_text())[:512]

    html_tag = soup.find("html")
    if isinstance(html_tag, Tag) and html_tag.get("lang"):
        meta.setdefault("language", str(html_tag["lang"])[:16])

    return meta


_WANTED_META: Final[dict[str, str]] = {
    "og:title": "title",
    "twitter:title": "title",
    "og:description": "description",
    "twitter:description": "description",
    "description": "description",
    "og:image": "image_url",
    "twitter:image": "image_url",
    "og:url": "canonical_url",
    "article:published_time": "published_at",
    "article:modified_time": "updated_at",
    "og:site_name": "site_name",
    "author": "author",
    "article:author": "author",
    "og:locale": "language",
    "content-language": "language",
}


def text_from_node(node: Tag | NavigableString | None) -> str:
    """Normalised text of a BeautifulSoup node (``""`` when missing)."""
    if node is None:
        return ""
    if isinstance(node, NavigableString):
        return normalize_whitespace(str(node))
    return normalize_whitespace(node.get_text(separator=" "))


__all__ = [
    "ALLOWED_TAGS",
    "DROP_TAGS",
    "MAX_HTML_BYTES",
    "clean_html",
    "dedupe_repeated_lines",
    "extract_main_text",
    "extract_meta",
    "looks_like_html",
    "normalize_whitespace",
    "sanitize_fragment",
    "strip_html",
    "text_from_node",
]
