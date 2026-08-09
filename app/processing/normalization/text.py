"""Field-level text normalisation: titles, authors, languages, countries, categories."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from app.processing.cleaning.html_clean import normalize_whitespace, strip_html

MAX_TITLE_LENGTH: Final[int] = 512
MAX_AUTHOR_LENGTH: Final[int] = 200

#: Trailing site branding that feeds append to every headline.
# The character class deliberately lists the typographic separators publishers
# use before their brand name (en/em dash, bullet, middle dot).
_TITLE_SUFFIX_RE = re.compile(
    r"\s*[|\-–—•·]\s*(?:[A-Z][\w.& ]{2,40})\s*$",
)
# Smart quotes are exactly what this table exists to normalise.
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‹": "'", "›": "'"})

_AUTHOR_PREFIX_RE = re.compile(r"(?i)^\s*(by|von|par|por|written by|reporting by)[:\s]+")
_AUTHOR_SPLIT_RE = re.compile(r"(?i)\s*(?:,|;|/|\band\b|\&)\s*")
_AUTHOR_NOISE_RE = re.compile(
    r"(?i)\b(staff|correspondent|reporter|editor|contributor|agencies|newsroom|"
    r"associated press|reuters staff)\b"
)

#: Canonical categories the platform recognises (configurable via topics table).
CATEGORY_ALIASES: Final[dict[str, str]] = {
    "tech": "technology",
    "technology": "technology",
    "sci-tech": "technology",
    "gadgets": "technology",
    "ai": "ai",
    "artificial-intelligence": "ai",
    "machine-learning": "ai",
    "business": "business",
    "economy": "business",
    "markets": "finance",
    "finance": "finance",
    "money": "finance",
    "crypto": "cryptocurrency",
    "cryptocurrency": "cryptocurrency",
    "blockchain": "cryptocurrency",
    "politics": "politics",
    "policy": "politics",
    "world": "world",
    "international": "world",
    "science": "science",
    "health": "health",
    "healthcare": "health",
    "medicine": "health",
    "sport": "sports",
    "sports": "sports",
    "security": "cybersecurity",
    "cyber": "cybersecurity",
    "cybersecurity": "cybersecurity",
    "infosec": "cybersecurity",
    "energy": "energy",
    "climate": "energy",
    "environment": "energy",
    "entertainment": "entertainment",
    "culture": "entertainment",
    "general": "general",
}

_LANG_ALIASES: Final[dict[str, str]] = {
    "english": "en",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
    "en_us": "en",
    "french": "fr",
    "fra": "fr",
    "german": "de",
    "deu": "de",
    "ger": "de",
    "spanish": "es",
    "spa": "es",
    "portuguese": "pt",
    "por": "pt",
    "italian": "it",
    "ita": "it",
    "russian": "ru",
    "rus": "ru",
    "arabic": "ar",
    "ara": "ar",
    "persian": "fa",
    "farsi": "fa",
    "fas": "fa",
    "per": "fa",
    "chinese": "zh",
    "zho": "zh",
    "japanese": "ja",
    "jpn": "ja",
    "korean": "ko",
    "kor": "ko",
    "turkish": "tr",
    "tur": "tr",
    "dutch": "nl",
    "nld": "nl",
    "hindi": "hi",
    "hin": "hi",
}


def normalize_title(raw: str | None) -> str:
    """Clean a headline: strip markup, unify quotes, drop site branding."""
    if not raw:
        return ""
    title = strip_html(raw).translate(_QUOTES)
    title = unicodedata.normalize("NFKC", title)
    title = normalize_whitespace(title)

    # Only strip a trailing brand when the headline stays substantial.
    stripped = _TITLE_SUFFIX_RE.sub("", title).strip()
    if len(stripped) >= 25:
        title = stripped

    return title[:MAX_TITLE_LENGTH].strip()


def normalize_text_block(raw: str | None, limit: int) -> str | None:
    """Clean a description/content block, or ``None`` when it is empty."""
    if not raw:
        return None
    text = normalize_whitespace(strip_html(raw).translate(_QUOTES))
    if not text:
        return None
    return text[:limit]


def normalize_author(raw: str | None) -> str | None:
    """Extract a clean primary author name, or ``None``.

    Feeds put anything in this field: ``"By Jane Doe and John Roe, CNN"``,
    ``"newsroom@example.com"``, or an entire copyright notice.
    """
    if not raw:
        return None
    text = strip_html(raw)
    text = _AUTHOR_PREFIX_RE.sub("", text)
    text = re.sub(r"[<>@]\S+", " ", text)  # drop e-mail addresses
    text = normalize_whitespace(text.strip(" ,;|-"))
    if not text:
        return None

    parts = [part.strip() for part in _AUTHOR_SPLIT_RE.split(text) if part.strip()]
    for part in parts:
        candidate = normalize_whitespace(_AUTHOR_NOISE_RE.sub("", part)).strip(" ,.-")
        if 2 <= len(candidate) <= MAX_AUTHOR_LENGTH and any(ch.isalpha() for ch in candidate):
            return candidate[:MAX_AUTHOR_LENGTH]
    return None


def author_key(name: str) -> str:
    """Case/accent-insensitive key used to deduplicate author rows."""
    folded = unicodedata.normalize("NFKD", name.casefold())
    ascii_only = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()


def normalize_language(raw: str | None) -> str | None:
    """Map anything language-shaped to a two-letter ISO 639-1 code."""
    if not raw:
        return None
    value = str(raw).strip().lower().replace("_", "-")
    if not value:
        return None
    if value in _LANG_ALIASES:
        return _LANG_ALIASES[value]
    base = value.split("-", 1)[0]
    if base in _LANG_ALIASES:
        return _LANG_ALIASES[base]
    if len(base) == 2 and base.isalpha():
        return base
    return None


def normalize_country(raw: str | None) -> str | None:
    """Return an upper-case two-letter country code, or ``None``."""
    if not raw:
        return None
    value = str(raw).strip().upper()
    if len(value) == 2 and value.isalpha():
        return value
    if "-" in value:
        tail = value.rsplit("-", 1)[-1]
        if len(tail) == 2 and tail.isalpha():
            return tail
    return None


def normalize_category(raw: str | None) -> str | None:
    """Map a source category onto the platform's canonical vocabulary."""
    if not raw:
        return None
    value = re.sub(r"[^a-z0-9]+", "-", str(raw).strip().lower()).strip("-")
    if not value:
        return None
    if value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]
    for token in value.split("-"):
        if token in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[token]
    return value[:48]


def word_count(text: str | None) -> int:
    """Number of word-like tokens (works for non-Latin scripts too)."""
    if not text:
        return 0
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


__all__ = [
    "CATEGORY_ALIASES",
    "author_key",
    "normalize_author",
    "normalize_category",
    "normalize_country",
    "normalize_language",
    "normalize_text_block",
    "normalize_title",
    "word_count",
]
