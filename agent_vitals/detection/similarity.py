"""Content-based output similarity detection for Agent Vitals.

Computes similarity between agent outputs across iterations using
fingerprinting (exact match) and word-level Jaccard similarity
(near-duplicate detection). Zero external dependencies — uses only
Python stdlib.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Result of comparing current output against previous outputs."""

    max_similarity: float
    """Highest pairwise similarity to any output in the comparison window."""

    mean_similarity: float
    """Average pairwise similarity across all outputs in the window."""

    consecutive_similar: int
    """Count of most-recent consecutive outputs above the threshold."""

    is_exact_repeat: bool
    """Whether any output in the window has the same fingerprint."""


def compute_output_fingerprint(text: str) -> str:
    """Compute a stable SHA-256 hex digest of normalized text.

    Normalization: lowercase, collapse whitespace, strip.

    Args:
        text: Raw output text.

    Returns:
        64-character hex digest string.
    """
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_pairwise_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts.

    Both inputs are normalized (lowercased, whitespace collapsed)
    before tokenization.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score in [0.0, 1.0]. Returns 1.0 when both texts
        are empty (trivially identical).
    """
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def compute_similarity_scores(
    current_output: str,
    previous_outputs: Sequence[str],
    *,
    threshold: float = 0.8,
    window: Optional[int] = None,
) -> SimilarityResult:
    """Compare current output against previous outputs.

    Args:
        current_output: The agent's latest output text.
        previous_outputs: Prior outputs (oldest → newest).
        threshold: Similarity threshold for counting as "similar".
        window: Maximum number of recent outputs to compare against.
            None means use all provided outputs.

    Returns:
        SimilarityResult with max/mean similarity, consecutive count,
        and exact-repeat flag.
    """
    if not previous_outputs:
        return SimilarityResult(
            max_similarity=0.0,
            mean_similarity=0.0,
            consecutive_similar=0,
            is_exact_repeat=False,
        )

    # Trim to window (most recent)
    candidates = list(previous_outputs)
    if window is not None and window > 0:
        candidates = candidates[-window:]

    current_fp = compute_output_fingerprint(current_output)
    is_exact = False
    similarities: list[float] = []

    for prev in candidates:
        sim = compute_pairwise_similarity(current_output, prev)
        similarities.append(sim)
        if not is_exact:
            prev_fp = compute_output_fingerprint(prev)
            if prev_fp == current_fp:
                is_exact = True

    max_sim = max(similarities) if similarities else 0.0
    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Count consecutive similar outputs from the most recent backward
    consecutive = 0
    for sim in reversed(similarities):
        if sim >= threshold:
            consecutive += 1
        else:
            break

    return SimilarityResult(
        max_similarity=max_sim,
        mean_similarity=mean_sim,
        consecutive_similar=consecutive,
        is_exact_repeat=is_exact,
    )


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Normalize and split text into word tokens."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


__all__ = [
    "SimilarityResult",
    "compute_output_fingerprint",
    "compute_pairwise_similarity",
    "compute_similarity_scores",
]
