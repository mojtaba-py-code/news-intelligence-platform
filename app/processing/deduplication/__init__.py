"""Multi-level duplicate detection."""

from app.processing.deduplication.engine import (
    DeduplicationEngine,
    DuplicateLevel,
    DuplicateMatch,
)
from app.processing.deduplication.hashing import hamming_distance, simhash64
from app.processing.deduplication.similarity import (
    TfidfIndex,
    cosine_similarity,
    jaccard_similarity,
    token_set_ratio,
)

__all__ = [
    "DeduplicationEngine",
    "DuplicateLevel",
    "DuplicateMatch",
    "TfidfIndex",
    "cosine_similarity",
    "hamming_distance",
    "jaccard_similarity",
    "simhash64",
    "token_set_ratio",
]
