"""Article validation and quality scoring.

Two distinct outputs:

* a **hard verdict** - is this article fit to store at all? (missing title,
  unusable URL, no date, empty content, spam-looking payload);
* a **soft score** in ``[0, 1]`` describing completeness, which later feeds the
  relevance engine and the source-reliability calculation.

Rejections are counted per reason so the platform can report *why* a source is
producing garbage rather than merely that it is.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from app.core.url_safety import canonicalize_url, extract_domain
from app.core.utils import clamp, utcnow
from app.processing.normalization.text import word_count
from app.schemas.article import NormalizedArticle, RawArticle
from app.schemas.intelligence import DataQualityReport

MIN_TITLE_LENGTH: Final[int] = 8
MIN_CONTENT_WORDS: Final[int] = 20
MAX_TITLE_UPPER_RATIO: Final[float] = 0.7
MAX_FUTURE_HOURS: Final[int] = 6

_SPAM_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(click here|buy now|limited time offer|100% free|earn \$\d+)\b"),
    re.compile(r"(?i)\b(viagra|casino|porn|xxx|escort)\b"),
    re.compile(r"(?i)(https?://\S+){6,}"),  # link farm
)


class QualityIssue(StrEnum):
    """Machine-readable reasons an article is rejected or downgraded."""

    MISSING_TITLE = "missing_title"
    TITLE_TOO_SHORT = "title_too_short"
    SHOUTING_TITLE = "shouting_title"
    INVALID_URL = "invalid_url"
    MISSING_DATE = "missing_date"
    FUTURE_DATE = "future_date"
    EMPTY_CONTENT = "empty_content"
    THIN_CONTENT = "thin_content"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    SPAM_LIKE = "spam_like"
    MISSING_DESCRIPTION = "missing_description"
    MISSING_AUTHOR = "missing_author"
    MISSING_IMAGE = "missing_image"
    MALFORMED = "malformed"


#: Issues that make an article unusable rather than merely incomplete.
FATAL_ISSUES: Final[frozenset[QualityIssue]] = frozenset(
    {
        QualityIssue.MISSING_TITLE,
        QualityIssue.TITLE_TOO_SHORT,
        QualityIssue.INVALID_URL,
        QualityIssue.MISSING_DATE,
        QualityIssue.SPAM_LIKE,
        QualityIssue.MALFORMED,
        QualityIssue.UNSUPPORTED_LANGUAGE,
    }
)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Verdict plus the score and the issues that produced it."""

    is_valid: bool
    score: float
    issues: tuple[QualityIssue, ...] = ()

    @property
    def fatal_issues(self) -> tuple[QualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue in FATAL_ISSUES)

    def reason(self) -> str:
        fatal = self.fatal_issues
        return ", ".join(str(issue) for issue in (fatal or self.issues)) or "ok"


def validate_article(
    article: NormalizedArticle | RawArticle,
    *,
    allowed_languages: frozenset[str] | None = None,
    require_content: bool = False,
) -> ValidationOutcome:
    """Check ``article`` and return a verdict plus a completeness score.

    ``allowed_languages`` is optional: when a deployment only cares about a
    subset of languages, everything else is rejected early instead of paying
    for NLP on articles nobody will read.
    """
    issues: list[QualityIssue] = []
    score = 1.0

    title = (getattr(article, "title", "") or "").strip()
    if not title:
        issues.append(QualityIssue.MISSING_TITLE)
        score -= 0.5
    elif len(title) < MIN_TITLE_LENGTH:
        issues.append(QualityIssue.TITLE_TOO_SHORT)
        score -= 0.3
    else:
        letters = [ch for ch in title if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) > MAX_TITLE_UPPER_RATIO:
            issues.append(QualityIssue.SHOUTING_TITLE)
            score -= 0.05

    url = (getattr(article, "url", "") or "").strip()
    canonical = getattr(article, "canonical_url", None) or canonicalize_url(url)
    if not canonical or not extract_domain(canonical):
        issues.append(QualityIssue.INVALID_URL)
        score -= 0.5

    published_at = getattr(article, "published_at", None)
    if published_at is None:
        issues.append(QualityIssue.MISSING_DATE)
        score -= 0.3
    else:
        delta_hours = (published_at - utcnow()).total_seconds() / 3600
        if delta_hours > MAX_FUTURE_HOURS:
            issues.append(QualityIssue.FUTURE_DATE)
            score -= 0.1

    description = getattr(article, "description", None)
    content = getattr(article, "content", None)
    body_words = word_count(content or description or "")
    if not content and not description:
        issues.append(QualityIssue.EMPTY_CONTENT)
        score -= 0.3 if require_content else 0.15
    elif body_words < MIN_CONTENT_WORDS:
        issues.append(QualityIssue.THIN_CONTENT)
        score -= 0.1

    if not description:
        issues.append(QualityIssue.MISSING_DESCRIPTION)
        score -= 0.05
    if not getattr(article, "author_name", None) and not getattr(article, "author", None):
        issues.append(QualityIssue.MISSING_AUTHOR)
        score -= 0.05
    if not getattr(article, "image_url", None):
        issues.append(QualityIssue.MISSING_IMAGE)
        score -= 0.02

    language = getattr(article, "language", None)
    if allowed_languages and language and language not in allowed_languages:
        issues.append(QualityIssue.UNSUPPORTED_LANGUAGE)
        score -= 0.5

    haystack = f"{title}\n{description or ''}\n{(content or '')[:2000]}"
    if any(pattern.search(haystack) for pattern in _SPAM_PATTERNS):
        issues.append(QualityIssue.SPAM_LIKE)
        score -= 0.6

    if require_content and QualityIssue.EMPTY_CONTENT in issues:
        issues.append(QualityIssue.MALFORMED)

    fatal = any(issue in FATAL_ISSUES for issue in issues)
    return ValidationOutcome(is_valid=not fatal, score=round(clamp(score), 4), issues=tuple(issues))


@dataclass
class QualityReportBuilder:
    """Accumulates per-run data-quality counters."""

    total_processed: int = 0
    valid_articles: int = 0
    invalid_articles: int = 0
    duplicate_articles: int = 0
    issue_counts: Counter[str] = field(default_factory=Counter)

    def record(self, outcome: ValidationOutcome) -> None:
        self.total_processed += 1
        if outcome.is_valid:
            self.valid_articles += 1
        else:
            self.invalid_articles += 1
        for issue in outcome.issues:
            self.issue_counts[str(issue)] += 1

    def record_duplicate(self) -> None:
        self.duplicate_articles += 1

    def record_malformed(self) -> None:
        self.total_processed += 1
        self.invalid_articles += 1
        self.issue_counts[str(QualityIssue.MALFORMED)] += 1

    def build(self) -> DataQualityReport:
        return DataQualityReport(
            total_processed=self.total_processed,
            valid_articles=self.valid_articles,
            invalid_articles=self.invalid_articles,
            duplicate_articles=self.duplicate_articles,
            missing_title=self.issue_counts[str(QualityIssue.MISSING_TITLE)],
            invalid_url=self.issue_counts[str(QualityIssue.INVALID_URL)],
            invalid_date=self.issue_counts[str(QualityIssue.MISSING_DATE)]
            + self.issue_counts[str(QualityIssue.FUTURE_DATE)],
            empty_content=self.issue_counts[str(QualityIssue.EMPTY_CONTENT)],
            unsupported_language=self.issue_counts[str(QualityIssue.UNSUPPORTED_LANGUAGE)],
            malformed_response=self.issue_counts[str(QualityIssue.MALFORMED)],
        )


__all__ = [
    "FATAL_ISSUES",
    "MIN_CONTENT_WORDS",
    "MIN_TITLE_LENGTH",
    "QualityIssue",
    "QualityReportBuilder",
    "ValidationOutcome",
    "validate_article",
]
