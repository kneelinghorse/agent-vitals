"""Unit tests for content-based output similarity detection.

Tests cover: fingerprinting, pairwise similarity, similarity scoring,
edge cases (empty text, first iteration, identical outputs, dissimilar
outputs), and window behavior.
"""

from __future__ import annotations

from agent_vitals.detection.similarity import (
    compute_output_fingerprint,
    compute_pairwise_similarity,
    compute_similarity_scores,
)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


class TestOutputFingerprint:
    """Tests for compute_output_fingerprint."""

    def test_deterministic(self) -> None:
        """Same text always produces the same fingerprint."""
        text = "The agent found 3 new research papers on transformer architectures."
        fp1 = compute_output_fingerprint(text)
        fp2 = compute_output_fingerprint(text)
        assert fp1 == fp2

    def test_different_texts_differ(self) -> None:
        """Different texts produce different fingerprints."""
        fp1 = compute_output_fingerprint("Alpha result")
        fp2 = compute_output_fingerprint("Beta result")
        assert fp1 != fp2

    def test_normalizes_whitespace(self) -> None:
        """Whitespace variations produce the same fingerprint."""
        fp1 = compute_output_fingerprint("hello   world")
        fp2 = compute_output_fingerprint("hello world")
        assert fp1 == fp2

    def test_normalizes_case(self) -> None:
        """Case variations produce the same fingerprint."""
        fp1 = compute_output_fingerprint("Hello World")
        fp2 = compute_output_fingerprint("hello world")
        assert fp1 == fp2

    def test_empty_string(self) -> None:
        """Empty string produces a valid fingerprint."""
        fp = compute_output_fingerprint("")
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex digest

    def test_hex_digest_format(self) -> None:
        """Fingerprint is a 64-char hex string."""
        fp = compute_output_fingerprint("test")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------


class TestPairwiseSimilarity:
    """Tests for compute_pairwise_similarity."""

    def test_identical_texts(self) -> None:
        """Identical texts have similarity 1.0."""
        text = "The agent completed the research task successfully."
        assert compute_pairwise_similarity(text, text) == 1.0

    def test_completely_different(self) -> None:
        """Texts with no shared words have similarity 0.0."""
        assert compute_pairwise_similarity("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap(self) -> None:
        """Texts with some shared words have intermediate similarity."""
        sim = compute_pairwise_similarity(
            "the agent found three papers",
            "the agent discovered five papers",
        )
        assert 0.0 < sim < 1.0

    def test_both_empty(self) -> None:
        """Two empty strings are trivially identical (1.0)."""
        assert compute_pairwise_similarity("", "") == 1.0

    def test_one_empty_one_not(self) -> None:
        """One empty, one non-empty has similarity 0.0."""
        assert compute_pairwise_similarity("", "hello world") == 0.0
        assert compute_pairwise_similarity("hello world", "") == 0.0

    def test_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        assert compute_pairwise_similarity("Hello World", "hello world") == 1.0

    def test_whitespace_normalization(self) -> None:
        """Extra whitespace is normalized before comparison."""
        assert compute_pairwise_similarity("hello   world", "hello world") == 1.0

    def test_symmetric(self) -> None:
        """Similarity is symmetric: sim(a,b) == sim(b,a)."""
        text_a = "the quick brown fox"
        text_b = "the slow brown dog"
        assert compute_pairwise_similarity(text_a, text_b) == compute_pairwise_similarity(text_b, text_a)

    def test_known_jaccard(self) -> None:
        """Verify Jaccard computation for a known case."""
        # tokens_a = {"a", "b", "c"}, tokens_b = {"b", "c", "d"}
        # intersection = {"b", "c"} = 2, union = {"a", "b", "c", "d"} = 4
        # Jaccard = 2/4 = 0.5
        assert compute_pairwise_similarity("a b c", "b c d") == 0.5


# ---------------------------------------------------------------------------
# Similarity scores (window-based)
# ---------------------------------------------------------------------------


class TestSimilarityScores:
    """Tests for compute_similarity_scores."""

    def test_no_previous_outputs(self) -> None:
        """First iteration (no history) returns zero similarity."""
        result = compute_similarity_scores("hello world", [])
        assert result.max_similarity == 0.0
        assert result.mean_similarity == 0.0
        assert result.consecutive_similar == 0
        assert result.is_exact_repeat is False

    def test_exact_repeat(self) -> None:
        """Identical previous output is detected as exact repeat."""
        text = "Found 5 papers on reinforcement learning."
        result = compute_similarity_scores(text, [text])
        assert result.max_similarity == 1.0
        assert result.is_exact_repeat is True
        assert result.consecutive_similar == 1

    def test_near_duplicate(self) -> None:
        """Near-duplicate outputs have high similarity but not exact repeat."""
        current = "Found 5 papers on reinforcement learning in 2024."
        previous = ["Found 5 papers on reinforcement learning in 2023."]
        result = compute_similarity_scores(current, previous)
        assert result.max_similarity > 0.7
        assert result.is_exact_repeat is False

    def test_dissimilar_outputs(self) -> None:
        """Dissimilar outputs have low similarity."""
        current = "Analyzing market trends in semiconductor industry."
        previous = ["Completed database migration with zero downtime."]
        result = compute_similarity_scores(current, previous)
        assert result.max_similarity < 0.3

    def test_consecutive_similar_count(self) -> None:
        """Counts consecutive similar outputs from most recent."""
        current = "The agent found papers on AI safety."
        previous = [
            "Totally different output about cooking recipes.",
            "The agent found papers on AI safety and alignment.",
            "The agent found papers on AI safety research.",
        ]
        result = compute_similarity_scores(current, previous, threshold=0.5)
        assert result.consecutive_similar >= 2

    def test_consecutive_resets_on_dissimilar(self) -> None:
        """Consecutive count resets when a dissimilar output is encountered."""
        current = "Research paper analysis complete."
        previous = [
            "Research paper analysis complete.",  # similar
            "Totally unrelated: cooking recipe for pasta.",  # breaks streak
            "Research paper analysis complete.",  # similar again
        ]
        result = compute_similarity_scores(current, previous, threshold=0.8)
        # Only the most recent one is counted before the break
        assert result.consecutive_similar == 1

    def test_window_limits_comparison(self) -> None:
        """Window parameter limits how far back we compare."""
        current = "output X"
        previous = [
            "output X",  # oldest — outside window=2
            "different stuff entirely",
            "output X",  # within window
        ]
        result_windowed = compute_similarity_scores(current, previous, window=2)
        result_full = compute_similarity_scores(current, previous)
        # Full comparison finds exact repeat from oldest; windowed may miss it
        assert result_full.is_exact_repeat is True
        # Windowed still finds the most recent exact repeat
        assert result_windowed.max_similarity == 1.0

    def test_mean_similarity(self) -> None:
        """Mean similarity is the average across all compared outputs."""
        current = "a b c"
        previous = ["a b c", "d e f"]  # one identical, one fully different
        result = compute_similarity_scores(current, previous)
        # Jaccard("a b c", "a b c") = 1.0, Jaccard("a b c", "d e f") = 0.0
        assert result.mean_similarity == 0.5

    def test_empty_current_output(self) -> None:
        """Empty current output with non-empty history."""
        result = compute_similarity_scores("", ["hello world"])
        assert result.max_similarity == 0.0
        assert result.is_exact_repeat is False

    def test_threshold_boundary(self) -> None:
        """Outputs exactly at threshold are counted as similar."""
        # Build texts with exactly 0.5 Jaccard: {"a","b","c"} vs {"b","c","d"}
        current = "a b c"
        previous = ["b c d"]
        result = compute_similarity_scores(current, previous, threshold=0.5)
        assert result.consecutive_similar == 1
        result_strict = compute_similarity_scores(current, previous, threshold=0.51)
        assert result_strict.consecutive_similar == 0
