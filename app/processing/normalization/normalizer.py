"""Turn a :class:`RawArticle` from any connector into a :class:`NormalizedArticle`.

This is the boundary where source-specific shapes stop existing. Everything
downstream - deduplication, NLP, scoring, storage - only ever sees the
normalised form.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.url_safety import canonicalize_url
from app.core.utils import content_fingerprint, sha256_text, utcnow
from app.processing.cleaning.html_clean import extract_main_text, normalize_whitespace, strip_html
from app.processing.deduplication.hashing import simhash64
from app.processing.normalization.dates import parse_datetime
from app.processing.normalization.text import (
    normalize_author,
    normalize_category,
    normalize_country,
    normalize_language,
    normalize_text_block,
    normalize_title,
    word_count,
)
from app.processing.validation.quality import validate_article
from app.schemas.article import (
    MAX_CONTENT,
    MAX_DESCRIPTION,
    MAX_TITLE,
    NormalizedArticle,
    RawArticle,
)

logger = get_logger(__name__)


class NormalizationError(Exception):
    """Raised when a raw item cannot be normalised at all."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ArticleNormalizer:
    """Stateless normaliser. Safe to share across tasks."""

    def __init__(self, *, detect_language: bool = True) -> None:
        self._detect_language = detect_language

    def normalize(self, raw: RawArticle) -> NormalizedArticle:
        """Map, clean and fingerprint ``raw``.

        Raises :class:`NormalizationError` when a mandatory field cannot be
        recovered - the caller records it as a rejected article and moves on.
        """
        title = normalize_title(raw.title)
        if not title:
            raise NormalizationError("empty title after cleaning")

        canonical = canonicalize_url(raw.url)
        if not canonical:
            raise NormalizationError("URL could not be canonicalised")

        description = normalize_text_block(raw.description, MAX_DESCRIPTION)
        content = self._clean_content(raw.content)

        # A description that merely repeats the truncated content adds nothing.
        if description and content and content.startswith(description[:120]):
            description = description if len(description) < len(content) else None

        published_at = parse_datetime(raw.published_at) or parse_datetime(
            raw.raw.get("pubDate") or raw.raw.get("date")
        )
        if published_at is None:
            # Missing dates are common in scraped pages; ingestion time is the
            # honest fallback and is flagged by the validator as a quality hit.
            published_at = utcnow()

        language = normalize_language(raw.language)
        if language is None and self._detect_language:
            from app.intelligence.language import detect_language

            detection = detect_language(f"{title}\n{description or ''}\n{(content or '')[:1500]}")
            language = detection.language if detection.confidence >= 0.35 else None

        body_for_hash = content or description or title
        normalized = NormalizedArticle(
            source_slug=raw.source_slug,
            source_name=raw.source_name[:160],
            external_id=raw.external_id,
            title=title[:MAX_TITLE],
            description=description,
            content=content,
            author_name=normalize_author(raw.author),
            url=raw.url[:2048],
            canonical_url=canonical,
            image_url=self._clean_image_url(raw.image_url),
            published_at=published_at,
            source_updated_at=parse_datetime(raw.updated_at),
            language=language,
            country=normalize_country(raw.country),
            category=normalize_category(raw.category),
            content_hash=content_fingerprint(f"{title}\n{body_for_hash}"),
            title_hash=content_fingerprint(title),
            simhash=str(simhash64(f"{title} {body_for_hash}")),
            word_count=word_count(content or description or ""),
        )

        outcome = validate_article(normalized)
        return normalized.model_copy(update={"quality_score": outcome.score})

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _clean_content(raw_content: str | None) -> str | None:
        if not raw_content:
            return None
        text = extract_main_text(raw_content) if "<" in raw_content else raw_content
        text = normalize_whitespace(strip_html(text))
        return text[:MAX_CONTENT] if text else None

    @staticmethod
    def _clean_image_url(raw_url: str | None) -> str | None:
        """Keep only absolute http(s) image URLs; relative ones are useless."""
        if not raw_url:
            return None
        candidate = raw_url.strip()
        if not candidate.startswith(("http://", "https://")):
            return None
        return candidate[:2048]

    @staticmethod
    def fingerprint(title: str, body: str) -> tuple[str, str]:
        """``(content_hash, title_hash)`` for an already-clean pair."""
        return content_fingerprint(f"{title}\n{body}"), content_fingerprint(title)

    @staticmethod
    def stable_external_id(source_slug: str, url: str) -> str:
        """Deterministic id for sources that do not provide one."""
        return sha256_text(f"{source_slug}|{canonicalize_url(url)}")[:32]


__all__ = ["ArticleNormalizer", "NormalizationError"]
