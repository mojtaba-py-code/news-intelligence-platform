"""The four-level deduplication engine.

Levels are ordered by cost, cheapest first, and the search short-circuits as
soon as a confident match is found:

1. **canonical URL** - O(1) dictionary hit, certain.
2. **content hash** - O(1), certain (exact text match after normalisation).
3. **SimHash** - integer Hamming distance, catches near-identical rewrites.
4. **title + TF-IDF cosine** - the expensive path, only for what survives.

The engine is deliberately storage-agnostic: it operates on a *candidate
window* supplied by the caller (typically "articles from the last N hours"),
which the repository fetches with one indexed query. That keeps the algorithm
unit-testable without a database and keeps the query count at O(1) per batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import duplicates_detected_total
from app.core.utils import clamp
from app.processing.deduplication.hashing import (
    DEFAULT_HAMMING_THRESHOLD,
    NEAR_CANDIDATE_DISTANCE,
    hamming_distance,
    parse_simhash,
    simhash64,
    similarity_from_distance,
)
from app.processing.deduplication.similarity import TfidfIndex, cosine_similarity, token_set_ratio

logger = get_logger(__name__)

#: Minimum body similarity required before an identical headline counts as a
#: duplicate. Below this the two articles are treated as separate stories.
TITLE_MATCH_MIN_CONTENT: float = 0.55


class DuplicateLevel(StrEnum):
    """Which stage produced the verdict."""

    URL = "url"
    CONTENT_HASH = "content_hash"
    SIMHASH = "simhash"
    TITLE = "title"
    CONTENT_SIMILARITY = "content_similarity"


class CandidateArticle(Protocol):
    """Structural type of the rows the engine compares against.

    A ``Protocol`` rather than the ORM class: unit tests can pass simple
    dataclasses, and the processing layer stays independent of persistence.
    """

    id: int
    title: str
    canonical_url: str
    content_hash: str
    title_hash: str
    simhash: str | None
    content: str | None
    description: str | None


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """A detected duplicate relationship."""

    article_id: int
    level: DuplicateLevel
    score: float

    @property
    def is_exact(self) -> bool:
        return self.level in (DuplicateLevel.URL, DuplicateLevel.CONTENT_HASH)


@dataclass(slots=True)
class _Candidate:
    """Internal, pre-processed view of an existing article."""

    id: int
    title: str
    canonical_url: str
    content_hash: str
    title_hash: str
    simhash: int | None
    text: str


@dataclass
class DeduplicationEngine:
    """Stateful over a candidate window; cheap to build, reusable per batch."""

    title_threshold: float = 0.85
    content_threshold: float = 0.80
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD

    _by_url: dict[str, int] = field(default_factory=dict, init=False)
    _by_content_hash: dict[str, int] = field(default_factory=dict, init=False)
    _by_title_hash: dict[str, int] = field(default_factory=dict, init=False)
    _candidates: list[_Candidate] = field(default_factory=list, init=False)
    _index: TfidfIndex | None = field(default=None, init=False)
    _vectors: dict[int, dict[str, float]] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_settings(cls, config: Settings | None = None) -> DeduplicationEngine:
        config = config or get_settings()
        return cls(
            title_threshold=config.dedup_title_threshold,
            content_threshold=config.dedup_content_threshold,
        )

    def load(self, candidates: list[CandidateArticle]) -> DeduplicationEngine:
        """Index a candidate window. Later entries never overwrite earlier ones,
        so the *oldest* article always wins as the canonical original."""
        self._by_url.clear()
        self._by_content_hash.clear()
        self._by_title_hash.clear()
        self._candidates.clear()
        self._vectors.clear()
        self._index = None

        for candidate in candidates:
            text = self._text_of(candidate.title, candidate.content, candidate.description)
            entry = _Candidate(
                id=candidate.id,
                title=candidate.title or "",
                canonical_url=candidate.canonical_url or "",
                content_hash=candidate.content_hash or "",
                title_hash=candidate.title_hash or "",
                simhash=parse_simhash(candidate.simhash),
                text=text,
            )
            self._candidates.append(entry)
            if entry.canonical_url:
                self._by_url.setdefault(entry.canonical_url, entry.id)
            if entry.content_hash:
                self._by_content_hash.setdefault(entry.content_hash, entry.id)
            if entry.title_hash:
                self._by_title_hash.setdefault(entry.title_hash, entry.id)
        return self

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    # -------------------------------------------------------------- detection
    def find_duplicate(
        self,
        *,
        canonical_url: str,
        content_hash: str,
        title: str,
        title_hash: str = "",
        content: str | None = None,
        description: str | None = None,
        simhash: str | int | None = None,
    ) -> DuplicateMatch | None:
        """Return the best duplicate match for the given article, if any."""
        # Level 1 - canonical URL.
        existing = self._by_url.get(canonical_url)
        if existing is not None:
            return self._hit(existing, DuplicateLevel.URL, 1.0)

        # Level 2 - exact normalised content hash.
        existing = self._by_content_hash.get(content_hash)
        if existing is not None:
            return self._hit(existing, DuplicateLevel.CONTENT_HASH, 1.0)

        # Level 2b - identical normalised title is a strong signal on its own,
        # but only when the bodies also agree; handled in the fuzzy pass below.
        text = self._text_of(title, content, description)
        fingerprint = parse_simhash(simhash) if simhash is not None else simhash64(text)

        best: DuplicateMatch | None = None
        near_candidates: list[_Candidate] = []

        # Level 3 - SimHash. A very small distance is a verdict on its own; a
        # merely *close* fingerprint only earns a place in the expensive stage.
        if fingerprint:
            for candidate in self._candidates:
                if candidate.simhash is None:
                    continue
                distance = hamming_distance(fingerprint, candidate.simhash)
                if distance <= self.hamming_threshold:
                    score = similarity_from_distance(distance)
                    if best is None or score > best.score:
                        best = DuplicateMatch(candidate.id, DuplicateLevel.SIMHASH, score)
                elif distance <= NEAR_CANDIDATE_DISTANCE:
                    near_candidates.append(candidate)
            if best is not None:
                return self._hit(best.article_id, best.level, best.score)

        # Level 4 - title similarity, then TF-IDF cosine on the survivors.
        title_matches: list[tuple[_Candidate, float]] = []
        seen_ids: set[int] = set()
        for candidate in self._candidates:
            if title_hash and candidate.title_hash == title_hash:
                title_matches.append((candidate, 1.0))
                seen_ids.add(candidate.id)
                continue
            ratio = token_set_ratio(title, candidate.title)
            if ratio >= self.title_threshold:
                title_matches.append((candidate, ratio))
                seen_ids.add(candidate.id)

        # Syndicated copy is routinely re-headlined, so a near-identical body
        # must still be checked even when the headlines diverge.
        for candidate in near_candidates:
            if candidate.id not in seen_ids:
                title_matches.append((candidate, token_set_ratio(title, candidate.title)))
                seen_ids.add(candidate.id)

        if not title_matches:
            return None

        vector = self._vector_for_query(text)
        for candidate, title_score in title_matches:
            content_score = cosine_similarity(vector, self._vector_for(candidate))
            # A near-identical headline with an unrelated body is a different
            # story (e.g. a recurring column), so both signals must agree.
            combined = clamp(0.5 * title_score + 0.5 * content_score)
            # An identical *headline* is never sufficient on its own: recurring
            # columns, series instalments and numbered updates ("Market wrap 3")
            # share a headline while reporting different facts. The body has to
            # agree too, just less strongly than for a pure content match.
            if content_score >= self.content_threshold or (
                title_score >= 0.95 and content_score >= TITLE_MATCH_MIN_CONTENT
            ):
                level = (
                    DuplicateLevel.CONTENT_SIMILARITY
                    if content_score >= self.content_threshold
                    else DuplicateLevel.TITLE
                )
                if best is None or combined > best.score:
                    best = DuplicateMatch(candidate.id, level, round(combined, 4))

        return self._hit(best.article_id, best.level, best.score) if best else None

    def add(
        self,
        article_id: int,
        *,
        canonical_url: str,
        content_hash: str,
        title: str,
        title_hash: str,
        text: str = "",
        simhash: str | int | None = None,
    ) -> None:
        """Register a freshly stored article so the *same batch* self-deduplicates."""
        entry = _Candidate(
            id=article_id,
            title=title,
            canonical_url=canonical_url,
            content_hash=content_hash,
            title_hash=title_hash,
            simhash=parse_simhash(simhash) if simhash is not None else simhash64(text or title),
            text=text or title,
        )
        self._candidates.append(entry)
        self._by_url.setdefault(canonical_url, article_id)
        self._by_content_hash.setdefault(content_hash, article_id)
        self._by_title_hash.setdefault(title_hash, article_id)
        self._index = None  # vocabulary changed; rebuild lazily
        self._vectors.clear()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _hit(article_id: int, level: DuplicateLevel, score: float) -> DuplicateMatch:
        duplicates_detected_total.inc(labels={"level": str(level)})
        return DuplicateMatch(article_id=article_id, level=level, score=round(score, 4))

    @staticmethod
    def _text_of(title: str, content: str | None, description: str | None) -> str:
        body = content or description or ""
        return f"{title}\n{body[:4000]}"

    def _ensure_index(self) -> TfidfIndex:
        if self._index is None:
            self._index = TfidfIndex().fit([candidate.text for candidate in self._candidates])
        return self._index

    def _vector_for(self, candidate: _Candidate) -> dict[str, float]:
        vector = self._vectors.get(candidate.id)
        if vector is None:
            vector = self._ensure_index().transform(candidate.text)
            self._vectors[candidate.id] = vector
        return vector

    def _vector_for_query(self, text: str) -> dict[str, float]:
        return self._ensure_index().transform(text)


__all__ = ["CandidateArticle", "DeduplicationEngine", "DuplicateLevel", "DuplicateMatch"]
