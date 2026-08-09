"""The NLP enrichment pipeline.

Runs the individual engines in dependency order and returns one immutable
result object. Every stage is defensive: a failure in, say, entity extraction
must not cost the article its sentiment score, so each stage is isolated and
degrades to an empty result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.metrics import processing_duration_seconds
from app.core.utils import clamp
from app.database.models.article import SentimentLabel
from app.intelligence.entities import ExtractedEntity, extract_entities
from app.intelligence.keywords import Keyword, extract_keywords
from app.intelligence.language import LanguageDetection, detect_language
from app.intelligence.sentiment import SentimentResult, analyze_sentiment
from app.intelligence.summarize import Summary, summarize
from app.intelligence.topics import TopicClassifier, TopicMatch, classify_topics
from app.processing.normalization.text import word_count

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """Everything the NLP layer derived from one article."""

    language: str | None = None
    language_confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment_confidence: float = 0.0
    keywords: tuple[Keyword, ...] = ()
    entities: tuple[ExtractedEntity, ...] = ()
    topics: tuple[TopicMatch, ...] = ()
    summary: str | None = None
    readability: float | None = None
    word_count: int = 0
    duration_ms: float = 0.0
    errors: tuple[str, ...] = ()

    @property
    def keyword_strings(self) -> list[str]:
        return [keyword.text for keyword in self.keywords]

    @property
    def primary_topic(self) -> str | None:
        return self.topics[0].slug if self.topics else None

    def metadata(self) -> dict[str, Any]:
        """Compact record stored in ``Article.enrichment`` for auditability."""
        return {
            "language_confidence": round(self.language_confidence, 3),
            "sentiment": {
                "score": self.sentiment_score,
                "label": str(self.sentiment_label),
                "confidence": self.sentiment_confidence,
            },
            "topics": [{"slug": topic.slug, "score": topic.score} for topic in self.topics],
            "entity_count": len(self.entities),
            "keyword_count": len(self.keywords),
            "readability": self.readability,
            "duration_ms": round(self.duration_ms, 2),
            "errors": list(self.errors),
        }


@dataclass
class NLPPipeline:
    """Coordinates the enrichment engines.

    Stateless apart from its configuration, so a single instance can be shared
    by every worker task.
    """

    classifier: TopicClassifier | None = None
    max_keywords: int = 12
    max_entities: int = 25
    max_topics: int = 3
    summarize_enabled: bool = True
    _errors: list[str] = field(default_factory=list, init=False, repr=False)

    def enrich(
        self,
        *,
        title: str,
        content: str | None = None,
        description: str | None = None,
        language_hint: str | None = None,
    ) -> EnrichmentResult:
        """Run every enrichment stage over one article."""
        started = time.perf_counter()
        errors: list[str] = []
        body = content or description or ""
        full_text = f"{title}\n\n{body}".strip()

        with processing_duration_seconds.time(labels={"stage": "nlp"}):
            detection = self._safe(
                "language",
                errors,
                lambda: detect_language(full_text),
                LanguageDetection(language=None, confidence=0.0),
            )
            language = language_hint or detection.language

            sentiment = self._safe(
                "sentiment",
                errors,
                lambda: analyze_sentiment(full_text),
                SentimentResult(0.0, SentimentLabel.NEUTRAL, 0.0),
            )
            keywords = self._safe(
                "keywords",
                errors,
                lambda: extract_keywords(body, title=title, limit=self.max_keywords),
                [],
            )
            entities = self._safe(
                "entities",
                errors,
                lambda: extract_entities(body, title=title, limit=self.max_entities),
                [],
            )
            topics = self._safe(
                "topics",
                errors,
                lambda: classify_topics(
                    body, title=title, limit=self.max_topics, classifier=self.classifier
                ),
                [],
            )
            summary: Summary | None = None
            if self.summarize_enabled and len(body) > 400:
                summary = self._safe("summary", errors, lambda: summarize(body, title=title), None)

        words = word_count(body)
        return EnrichmentResult(
            language=language,
            language_confidence=detection.confidence,
            sentiment_score=sentiment.score,
            sentiment_label=sentiment.label,
            sentiment_confidence=sentiment.confidence,
            keywords=tuple(keywords),
            entities=tuple(entities),
            topics=tuple(topics),
            summary=summary.text if summary and summary.text else None,
            readability=readability_score(body) if words else None,
            word_count=words,
            duration_ms=(time.perf_counter() - started) * 1000,
            errors=tuple(errors),
        )

    @staticmethod
    def _safe(stage: str, errors: list[str], call: Any, fallback: Any) -> Any:
        """Run one stage; on failure record it and continue with ``fallback``."""
        try:
            return call()
        except Exception as exc:
            logger.warning(
                "nlp_stage_failed", extra={"stage": stage, "error_type": exc.__class__.__name__}
            )
            errors.append(stage)
            return fallback


def readability_score(text: str) -> float | None:
    """Approximate Flesch reading ease, rescaled to ``[0, 1]``.

    A rough syllable heuristic (vowel groups) keeps this dependency-free; the
    value is only ever used comparatively, never reported as a linguistic fact.
    """
    if not text or len(text) < 100:
        return None
    sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    words = text.split()
    if len(words) < 20:
        return None

    syllables = 0
    for word in words:
        cleaned = "".join(ch for ch in word.casefold() if ch.isalpha())
        if not cleaned:
            continue
        groups = 0
        previous_vowel = False
        for char in cleaned:
            is_vowel = char in "aeiouy"
            if is_vowel and not previous_vowel:
                groups += 1
            previous_vowel = is_vowel
        if cleaned.endswith("e") and groups > 1:
            groups -= 1
        syllables += max(1, groups)

    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return round(clamp(flesch / 100.0), 4)


__all__ = ["EnrichmentResult", "NLPPipeline", "readability_score"]
