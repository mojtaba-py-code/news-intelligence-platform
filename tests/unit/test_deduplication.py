"""SimHash, similarity measures and the four-level deduplication engine."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.utils import content_fingerprint
from app.processing.deduplication.engine import (
    DeduplicationEngine,
    DuplicateLevel,
)
from app.processing.deduplication.hashing import (
    hamming_distance,
    is_near_duplicate,
    parse_simhash,
    shingles,
    simhash64,
    similarity_from_distance,
    tokenize,
)
from app.processing.deduplication.similarity import (
    TfidfIndex,
    centroid,
    cosine_similarity,
    jaccard_similarity,
    token_set_ratio,
)

pytestmark = pytest.mark.unit

#: Regression guard for the feature-hashing implementation.
_STABLE_FINGERPRINT = 4778332398863515652


@dataclass
class FakeArticle:
    """Minimal stand-in for the ORM row the engine consumes."""

    id: int
    title: str
    canonical_url: str
    content_hash: str
    title_hash: str
    simhash: str | None
    content: str | None
    description: str | None = None


def build(article_id: int, title: str, url: str, content: str) -> FakeArticle:
    return FakeArticle(
        id=article_id,
        title=title,
        canonical_url=url,
        content_hash=content_fingerprint(f"{title}\n{content}"),
        title_hash=content_fingerprint(title),
        simhash=str(simhash64(f"{title} {content}")),
        content=content,
    )


class TestSimHash:
    def test_identical_text_has_identical_fingerprint(self) -> None:
        text = "Central bank raises interest rates by fifty basis points"
        assert simhash64(text) == simhash64(text)

    def test_fingerprint_is_process_stable(self) -> None:
        """blake2b, not the randomised built-in ``hash()``.

        The literal is recorded so a change in the feature hashing shows up as
        a test failure rather than as silently missed duplicates in production.
        """
        assert simhash64("stable input") == _STABLE_FINGERPRINT

    def test_near_duplicates_are_close(self) -> None:
        original = (
            "The central bank raised interest rates by fifty basis points on Thursday, "
            "citing persistent inflation across the services sector."
        )
        rewrite = (
            "The central bank raised interest rates by fifty basis points on Thursday, "
            "citing persistent inflation across the services sector of the economy."
        )
        distance = hamming_distance(simhash64(original), simhash64(rewrite))
        assert distance <= 6

    def test_unrelated_texts_are_far_apart(self) -> None:
        left = simhash64("Football club signs striker ahead of the summer transfer window")
        right = simhash64("Quantum computing researchers demonstrate error correction milestone")
        assert hamming_distance(left, right) > 10

    def test_empty_text_is_zero(self) -> None:
        assert simhash64("") == 0
        assert is_near_duplicate(0, 12345) is False

    def test_similarity_from_distance(self) -> None:
        assert similarity_from_distance(0) == 1.0
        assert similarity_from_distance(64) == 0.0

    def test_parse_simhash_handles_bad_input(self) -> None:
        assert parse_simhash(None) is None
        assert parse_simhash("") is None
        assert parse_simhash("abc") is None
        assert parse_simhash("42") == 42

    def test_tokenize_and_shingles(self) -> None:
        tokens = tokenize("Hello, World! 123")
        assert tokens == ["hello", "world", "123"]
        assert shingles(["a", "b", "c", "d"], 2) == ["a b", "b c", "c d"]
        assert shingles(["a"], 3) == ["a"]


class TestSimilarity:
    def test_jaccard(self) -> None:
        assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0
        assert jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0
        assert jaccard_similarity([], ["a"]) == 0.0

    def test_token_set_ratio_is_order_insensitive(self) -> None:
        left = "Apple unveils new processor for laptops"
        right = "New processor for laptops unveiled by Apple"
        reordered = "For laptops Apple unveils new processor"
        # Reordering alone leaves the score at 1.0; a genuine rewrite scores
        # lower, which is why the dedup title threshold sits at 0.85.
        assert token_set_ratio(left, reordered) == 1.0
        assert 0.5 < token_set_ratio(left, right) < 0.85

    def test_cosine_similarity(self) -> None:
        assert cosine_similarity({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
        assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0
        assert cosine_similarity({}, {"a": 1.0}) == 0.0

    def test_tfidf_ranks_shared_rare_terms_highest(self) -> None:
        documents = [
            "the central bank raised interest rates today",
            "the football team won the championship final",
            "the central bank kept interest rates unchanged",
        ]
        index = TfidfIndex().fit(documents)
        related = index.similarity(documents[0], documents[2])
        unrelated = index.similarity(documents[0], documents[1])
        assert related > unrelated
        assert index.document_count == 3
        assert index.vocabulary_size > 0

    def test_tfidf_handles_empty_documents(self) -> None:
        index = TfidfIndex().fit([""])
        assert index.transform("") == {}

    def test_centroid_averages_vectors(self) -> None:
        result = centroid([{"a": 1.0}, {"a": 3.0}])
        assert result["a"] == pytest.approx(2.0)
        assert centroid([]) == {}


class TestDeduplicationEngine:
    def test_url_match_is_level_one(self) -> None:
        existing = build(1, "Original headline here", "https://example.com/a", "Body text one")
        engine = DeduplicationEngine().load([existing])

        match = engine.find_duplicate(
            canonical_url="https://example.com/a",
            content_hash="different" * 8,
            title="Completely different headline",
        )
        assert match is not None
        assert match.level is DuplicateLevel.URL
        assert match.article_id == 1
        assert match.is_exact

    def test_content_hash_match(self) -> None:
        existing = build(2, "Title", "https://example.com/b", "Shared body")
        engine = DeduplicationEngine().load([existing])

        match = engine.find_duplicate(
            canonical_url="https://other.example.com/mirror",
            content_hash=existing.content_hash,
            title="Different title entirely",
        )
        assert match is not None
        assert match.level is DuplicateLevel.CONTENT_HASH

    def test_near_duplicate_detected_by_simhash(self) -> None:
        body = (
            "Regulators approved the merger after a year-long review, clearing the way "
            "for the combined company to operate across both markets from next quarter."
        )
        existing = build(3, "Regulators approve landmark merger", "https://a.example/1", body)
        engine = DeduplicationEngine().load([existing])

        match = engine.find_duplicate(
            canonical_url="https://b.example/2",
            content_hash=content_fingerprint("slightly different"),
            title="Regulators approve landmark merger deal",
            content=body + " Analysts welcomed the decision.",
        )
        assert match is not None
        assert match.article_id == 3

    def test_unrelated_article_is_unique(self) -> None:
        existing = build(
            4,
            "Football club signs striker",
            "https://a.example/football",
            "The club completed the signing of a striker from a rival team this week.",
        )
        engine = DeduplicationEngine().load([existing])

        match = engine.find_duplicate(
            canonical_url="https://b.example/quantum",
            content_hash=content_fingerprint("quantum"),
            title="Quantum computing milestone reached by researchers",
            content="Physicists demonstrated a new error-correction scheme in a lab experiment.",
        )
        assert match is None

    def test_oldest_article_wins_as_canonical(self) -> None:
        first = build(10, "Same headline", "https://a.example/x", "Same body")
        second = build(20, "Same headline", "https://b.example/y", "Same body")
        engine = DeduplicationEngine().load([first, second])
        assert engine._by_title_hash[first.title_hash] == 10

    def test_batch_self_deduplication(self) -> None:
        """An article added mid-batch must be visible to later comparisons."""
        engine = DeduplicationEngine().load([])
        engine.add(
            1,
            canonical_url="https://example.com/a",
            content_hash="hash-a",
            title="Breaking story",
            title_hash="title-a",
            text="Breaking story body",
        )
        match = engine.find_duplicate(
            canonical_url="https://example.com/a", content_hash="hash-b", title="Breaking story"
        )
        assert match is not None and match.article_id == 1

    def test_empty_window_finds_nothing(self) -> None:
        engine = DeduplicationEngine().load([])
        assert engine.candidate_count == 0
        assert (
            engine.find_duplicate(
                canonical_url="https://example.com/x",
                content_hash="h",
                title="Anything at all",
            )
            is None
        )

    def test_thresholds_come_from_settings(self) -> None:
        from app.core.config import get_settings

        engine = DeduplicationEngine.from_settings(get_settings())
        assert engine.title_threshold == get_settings().dedup_title_threshold
        assert engine.content_threshold == get_settings().dedup_content_threshold
