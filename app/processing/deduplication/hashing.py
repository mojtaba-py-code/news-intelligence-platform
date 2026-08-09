"""Content fingerprints: exact SHA-256 hashes and 64-bit SimHash.

SimHash gives a *locality-sensitive* fingerprint: near-identical documents land
within a small Hamming distance of each other, so wire-service copies that
differ by a paragraph are detectable with an integer comparison instead of an
O(n²) similarity sweep.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Final

HASH_BITS: Final[int] = 64
_MASK: Final[int] = (1 << HASH_BITS) - 1

#: Distance under which two documents are treated as near-identical.
#: 3/64 bits is the value the original Google paper reports for web-page scale.
DEFAULT_HAMMING_THRESHOLD: Final[int] = 3

#: Wider band used as a *candidate filter* rather than a verdict: anything this
#: close is worth the cost of a TF-IDF comparison, even when the headlines
#: differ (syndicated copy is routinely re-headlined by each outlet).
NEAR_CANDIDATE_DISTANCE: Final[int] = 18

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (Unicode-aware, so non-Latin scripts work)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.casefold())


def shingles(tokens: list[str], size: int = 3) -> list[str]:
    """Overlapping n-grams; word order matters for near-duplicate detection."""
    if size < 1:
        raise ValueError("size must be >= 1")
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _feature_hash(feature: str) -> int:
    """Stable 64-bit hash. ``hash()`` is randomised per process, so use blake2b."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simhash64(text: str, *, shingle_size: int = 1) -> int:
    """Compute the 64-bit SimHash of ``text``.

    ``shingle_size=1`` (weighted unigrams) is the default because news copy is
    routinely edited by *inserting* clauses. Word n-grams shift every shingle
    that straddles an insertion, so a one-sentence addition can move a 3-gram
    fingerprint by ~10 bits while the unigram fingerprint moves by ~3.
    Larger shingles are still available where word order must matter.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0

    features = shingles(tokens, shingle_size) if len(tokens) >= shingle_size else tokens
    weights = Counter(features)

    vector = [0] * HASH_BITS
    for feature, weight in weights.items():
        feature_hash = _feature_hash(feature)
        for bit in range(HASH_BITS):
            if feature_hash >> bit & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight

    value = 0
    for bit in range(HASH_BITS):
        if vector[bit] > 0:
            value |= 1 << bit
    return value & _MASK


def hamming_distance(left: int, right: int) -> int:
    """Number of differing bits between two fingerprints."""
    return ((left ^ right) & _MASK).bit_count()


def is_near_duplicate(left: int, right: int, threshold: int = DEFAULT_HAMMING_THRESHOLD) -> bool:
    """True when the two fingerprints are within ``threshold`` bits."""
    if not left or not right:
        return False
    return hamming_distance(left, right) <= threshold


def similarity_from_distance(distance: int) -> float:
    """Map a Hamming distance onto a ``[0, 1]`` similarity score."""
    return max(0.0, 1.0 - distance / HASH_BITS)


def parse_simhash(value: str | int | None) -> int | None:
    """Read a SimHash stored as text (SQLite cannot hold unsigned 64-bit ints)."""
    if value is None or value == "":
        return None
    try:
        return int(value) & _MASK
    except (TypeError, ValueError):
        return None


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes (used for response-body fingerprints)."""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "DEFAULT_HAMMING_THRESHOLD",
    "HASH_BITS",
    "NEAR_CANDIDATE_DISTANCE",
    "hamming_distance",
    "is_near_duplicate",
    "parse_simhash",
    "sha256_bytes",
    "shingles",
    "simhash64",
    "similarity_from_distance",
    "tokenize",
]
