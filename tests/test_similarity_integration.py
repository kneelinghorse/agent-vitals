"""Integration tests for content-based similarity in the detection pipeline.

Tests cover: loop detector with content_similarity signal, stuck detector
with low-similarity boost, monitor pipeline with output_text parameter,
and backward compatibility (no output_text provided).
"""

from __future__ import annotations

import copy

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.monitor import AgentVitals
from agent_vitals.schema import VitalsSnapshot


def _make_snapshot(
    base: dict,
    *,
    loop_index: int,
    findings_count: int,
    coverage_score: float,
    total_tokens: int,
    query_count: int,
    unique_domains: int,
    dm_coverage: float,
    cv_coverage: float,
    output_similarity: float | None = None,
    output_fingerprint: str | None = None,
    objectives_covered: int | None = None,
) -> VitalsSnapshot:
    payload = copy.deepcopy(base)
    payload["loop_index"] = loop_index
    payload["signals"]["findings_count"] = findings_count
    payload["signals"]["coverage_score"] = coverage_score
    payload["signals"]["total_tokens"] = total_tokens
    payload["signals"]["query_count"] = query_count
    payload["signals"]["unique_domains"] = unique_domains
    if objectives_covered is not None:
        payload["signals"]["objectives_covered"] = objectives_covered
    payload["metrics"]["dm_coverage"] = dm_coverage
    payload["metrics"]["cv_coverage"] = cv_coverage
    payload["loop_detected"] = False
    payload["loop_confidence"] = 0.0
    payload["loop_trigger"] = None
    payload["stuck_detected"] = False
    payload["stuck_confidence"] = 0.0
    payload["stuck_trigger"] = None
    if output_similarity is not None:
        payload["output_similarity"] = output_similarity
    if output_fingerprint is not None:
        payload["output_fingerprint"] = output_fingerprint
    return VitalsSnapshot.model_validate(payload)


# ---------------------------------------------------------------------------
# Loop detector: content_similarity signal
# ---------------------------------------------------------------------------


class TestLoopDetectorContentSimilarity:
    """Tests for content_similarity loop detection signal."""

    def test_high_similarity_triggers_loop(self, vitals_snapshot_healthy: dict) -> None:
        """High output similarity should trigger content_similarity loop signal."""
        config = VitalsConfig(loop_similarity_threshold=0.8)
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=i,
                findings_count=3 + i,
                coverage_score=0.5,
                total_tokens=1000 * (i + 1),
                query_count=i + 1,
                unique_domains=i + 1,
                dm_coverage=0.3,
                cv_coverage=0.2,
            )
            for i in range(3)
        ]
        # Current snapshot with very high similarity (simulating repeated output)
        current = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=6,
            coverage_score=0.6,
            total_tokens=4000,
            query_count=4,
            unique_domains=4,
            dm_coverage=0.3,
            cv_coverage=0.2,
            output_similarity=0.95,
        )

        result = detect_loop(current, history, config=config)
        assert result.loop_detected is True
        assert result.loop_trigger == "content_similarity"
        assert result.loop_confidence >= 0.80

    def test_low_similarity_no_loop(self, vitals_snapshot_healthy: dict) -> None:
        """Low output similarity should not trigger content_similarity."""
        config = VitalsConfig(loop_similarity_threshold=0.8)
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=i,
                findings_count=3 + i,
                coverage_score=0.5,
                total_tokens=1000 * (i + 1),
                query_count=i + 1,
                unique_domains=i + 1,
                dm_coverage=0.3,
                cv_coverage=0.2,
            )
            for i in range(3)
        ]
        current = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=6,
            coverage_score=0.6,
            total_tokens=4000,
            query_count=4,
            unique_domains=4,
            dm_coverage=0.3,
            cv_coverage=0.2,
            output_similarity=0.3,
        )

        result = detect_loop(current, history, config=config)
        # Should not trigger content_similarity (other loop signals may/may not fire)
        if result.loop_detected:
            assert result.loop_trigger != "content_similarity"

    def test_no_similarity_field_no_effect(self, vitals_snapshot_healthy: dict) -> None:
        """When output_similarity is None, content_similarity doesn't fire."""
        config = VitalsConfig()
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=0,
                findings_count=1,
                coverage_score=0.2,
                total_tokens=100,
                query_count=1,
                unique_domains=1,
                dm_coverage=0.4,
                cv_coverage=0.2,
            )
        ]
        current = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=2,
            coverage_score=0.4,
            total_tokens=200,
            query_count=2,
            unique_domains=2,
            dm_coverage=0.4,
            cv_coverage=0.2,
            # output_similarity not set (None)
        )

        result = detect_loop(current, history, config=config)
        assert result.loop_detected is False

    def test_similarity_suppressed_by_errors(self, vitals_snapshot_healthy: dict) -> None:
        """Content similarity loop signal should be suppressed when errors present."""
        config = VitalsConfig(loop_similarity_threshold=0.8)
        base = copy.deepcopy(vitals_snapshot_healthy)
        base["signals"]["error_count"] = 2  # Errors present

        history = [
            _make_snapshot(
                base,
                loop_index=0,
                findings_count=3,
                coverage_score=0.5,
                total_tokens=1000,
                query_count=1,
                unique_domains=1,
                dm_coverage=0.3,
                cv_coverage=0.2,
            )
        ]
        current = _make_snapshot(
            base,
            loop_index=1,
            findings_count=3,
            coverage_score=0.5,
            total_tokens=2000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.3,
            cv_coverage=0.2,
            output_similarity=0.95,
        )

        result = detect_loop(current, history, config=config)
        # Error suppression clears all loop candidates including content_similarity
        assert result.loop_detected is False

    def test_similarity_at_threshold_boundary(self, vitals_snapshot_healthy: dict) -> None:
        """Output similarity exactly at threshold should trigger."""
        config = VitalsConfig(loop_similarity_threshold=0.8)
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=0,
                findings_count=3,
                coverage_score=0.5,
                total_tokens=1000,
                query_count=1,
                unique_domains=1,
                dm_coverage=0.3,
                cv_coverage=0.2,
            )
        ]
        current = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=4,
            coverage_score=0.55,
            total_tokens=2000,
            query_count=2,
            unique_domains=2,
            dm_coverage=0.3,
            cv_coverage=0.2,
            output_similarity=0.8,  # exactly at threshold
        )

        result = detect_loop(current, history, config=config)
        assert result.loop_detected is True
        assert result.loop_trigger == "content_similarity"


# ---------------------------------------------------------------------------
# Stuck detector: low-similarity boost
# ---------------------------------------------------------------------------


class TestStuckDetectorLowSimilarity:
    """Tests for low-similarity boosting stuck confidence."""

    def test_low_similarity_boosts_stuck_confidence(self, vitals_snapshot_healthy: dict) -> None:
        """Low output similarity should boost stuck confidence when stuck is detected."""
        config = VitalsConfig()
        # Build a trace that triggers stuck detection via coverage_stagnation
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=i,
                findings_count=i + 1,
                coverage_score=0.5,
                total_tokens=100 * (i + 1),
                query_count=i + 1,
                unique_domains=i + 1,
                dm_coverage=0.0,
                cv_coverage=0.6,
            )
            for i in range(4)
        ]
        # Without similarity — baseline stuck
        current_no_sim = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=500,
            query_count=5,
            unique_domains=5,
            dm_coverage=0.0,
            cv_coverage=0.6,
        )
        result_no_sim = detect_loop(current_no_sim, history, config=config)

        # With low similarity — should boost
        current_low_sim = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=500,
            query_count=5,
            unique_domains=5,
            dm_coverage=0.0,
            cv_coverage=0.6,
            output_similarity=0.1,
        )
        result_low_sim = detect_loop(current_low_sim, history, config=config)

        # Both should detect stuck
        assert result_no_sim.stuck_detected is True
        assert result_low_sim.stuck_detected is True
        # Low similarity should boost confidence
        assert result_low_sim.stuck_confidence >= result_no_sim.stuck_confidence

    def test_high_similarity_triggers_loop_suppresses_stuck(self, vitals_snapshot_healthy: dict) -> None:
        """High similarity triggers loop via content_similarity, which cross-suppresses stuck."""
        config = VitalsConfig()
        history = [
            _make_snapshot(
                vitals_snapshot_healthy,
                loop_index=i,
                findings_count=i + 1,
                coverage_score=0.5,
                total_tokens=100 * (i + 1),
                query_count=i + 1,
                unique_domains=i + 1,
                dm_coverage=0.0,
                cv_coverage=0.6,
            )
            for i in range(4)
        ]
        # Without similarity — stuck fires
        current_no_sim = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=500,
            query_count=5,
            unique_domains=5,
            dm_coverage=0.0,
            cv_coverage=0.6,
        )
        result_no_sim = detect_loop(current_no_sim, history, config=config)
        assert result_no_sim.stuck_detected is True

        # With high similarity — content_similarity fires as loop,
        # which cross-detector suppresses the stuck signal
        current_high_sim = _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=500,
            query_count=5,
            unique_domains=5,
            dm_coverage=0.0,
            cv_coverage=0.6,
            output_similarity=0.9,
        )
        result_high_sim = detect_loop(current_high_sim, history, config=config)

        # Loop fires from content_similarity
        assert result_high_sim.loop_detected is True
        assert result_high_sim.loop_trigger == "content_similarity"
        # Stuck is suppressed by cross-detector suppression (loop detected)
        assert result_high_sim.stuck_detected is False


# ---------------------------------------------------------------------------
# Monitor pipeline: output_text integration
# ---------------------------------------------------------------------------


class TestMonitorOutputText:
    """Tests for AgentVitals.step() with output_text parameter."""

    def test_step_without_output_text(self) -> None:
        """Step without output_text should work identically to before."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())
        snapshot = monitor.step(
            findings_count=5,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
        )
        assert snapshot.output_similarity is None
        assert snapshot.output_fingerprint is None

    def test_step_with_output_text_first_step(self) -> None:
        """First step with output_text has fingerprint but no similarity."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())
        snapshot = monitor.step(
            findings_count=5,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
            output_text="Found 3 papers on AI safety.",
        )
        assert snapshot.output_fingerprint is not None
        assert snapshot.output_similarity is None  # no prior to compare to

    def test_step_with_repeated_output(self) -> None:
        """Repeated output_text should produce high similarity."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())
        text = "The agent analyzed transformer architectures and found key patterns."

        monitor.step(
            findings_count=2,
            coverage_score=0.3,
            total_tokens=500,
            error_count=0,
            output_text=text,
        )
        snapshot = monitor.step(
            findings_count=2,
            coverage_score=0.3,
            total_tokens=1000,
            error_count=0,
            output_text=text,
        )
        assert snapshot.output_similarity is not None
        assert snapshot.output_similarity == 1.0

    def test_step_with_different_output(self) -> None:
        """Different output_text should produce low similarity."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())

        monitor.step(
            findings_count=2,
            coverage_score=0.3,
            total_tokens=500,
            error_count=0,
            output_text="Analyzing semiconductor market trends in Asia.",
        )
        snapshot = monitor.step(
            findings_count=4,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
            output_text="Completed database schema migration with zero downtime.",
        )
        assert snapshot.output_similarity is not None
        assert snapshot.output_similarity < 0.3

    def test_repeated_output_triggers_loop(self) -> None:
        """Repeatedly identical output_text should trigger content_similarity loop."""
        config = VitalsConfig(loop_similarity_threshold=0.8, loop_consecutive_count=3)
        monitor = AgentVitals(mission_id="test", config=config)
        text = "I found the same 5 papers again. No new information."

        # Feed several steps with the exact same output and flat metrics
        for i in range(4):
            snapshot = monitor.step(
                findings_count=5,
                coverage_score=0.5,
                total_tokens=1000 * (i + 1),
                error_count=0,
                query_count=i + 1,
                unique_domains=3,
                output_text=text,
            )

        # By the last step, content_similarity should fire
        assert snapshot.loop_detected is True
        assert snapshot.loop_trigger == "content_similarity"

    def test_mixed_output_text_and_no_text(self) -> None:
        """Mixing steps with and without output_text should be safe."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())

        s1 = monitor.step(
            findings_count=2, coverage_score=0.3, total_tokens=500, error_count=0,
            output_text="First output.",
        )
        s2 = monitor.step(
            findings_count=3, coverage_score=0.4, total_tokens=1000, error_count=0,
            # No output_text
        )
        s3 = monitor.step(
            findings_count=4, coverage_score=0.5, total_tokens=1500, error_count=0,
            output_text="Third output.",
        )

        assert s1.output_fingerprint is not None
        assert s2.output_fingerprint is None
        assert s2.output_similarity is None
        assert s3.output_fingerprint is not None
        # s3 compares against "First output." only (s2 had no text)
        assert s3.output_similarity is not None

    def test_reset_clears_output_history(self) -> None:
        """Reset should clear the output text buffer."""
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())
        text = "Repeated output."

        monitor.step(
            findings_count=2, coverage_score=0.3, total_tokens=500, error_count=0,
            output_text=text,
        )
        monitor.reset()

        # After reset, first step should have no similarity
        snapshot = monitor.step(
            findings_count=2, coverage_score=0.3, total_tokens=500, error_count=0,
            output_text=text,
        )
        assert snapshot.output_similarity is None  # no prior to compare to

    def test_step_from_signals_with_output_text(self) -> None:
        """step_from_signals should accept output_text."""
        from agent_vitals.schema import RawSignals
        monitor = AgentVitals(mission_id="test", config=VitalsConfig())
        signals = RawSignals(
            findings_count=5, coverage_score=0.5, total_tokens=1000, error_count=0,
        )
        snapshot = monitor.step_from_signals(signals, output_text="Test output.")
        assert snapshot.output_fingerprint is not None


# ---------------------------------------------------------------------------
# Schema: output fields on VitalsSnapshot
# ---------------------------------------------------------------------------


class TestSnapshotOutputFields:
    """Tests for output_similarity and output_fingerprint on VitalsSnapshot."""

    def test_defaults_to_none(self, vitals_snapshot_healthy: dict) -> None:
        """New fields default to None when not provided."""
        snapshot = VitalsSnapshot.model_validate(vitals_snapshot_healthy)
        assert snapshot.output_similarity is None
        assert snapshot.output_fingerprint is None

    def test_serialization_roundtrip(self, vitals_snapshot_healthy: dict) -> None:
        """Fields survive JSON serialization roundtrip."""
        data = copy.deepcopy(vitals_snapshot_healthy)
        data["output_similarity"] = 0.85
        data["output_fingerprint"] = "abc123"
        snapshot = VitalsSnapshot.model_validate(data)
        assert snapshot.output_similarity == 0.85
        assert snapshot.output_fingerprint == "abc123"

        json_str = snapshot.model_dump_json()
        restored = VitalsSnapshot.model_validate_json(json_str)
        assert restored.output_similarity == 0.85
        assert restored.output_fingerprint == "abc123"

    def test_similarity_validation_bounds(self, vitals_snapshot_healthy: dict) -> None:
        """output_similarity must be in [0.0, 1.0]."""
        import pytest
        data = copy.deepcopy(vitals_snapshot_healthy)

        data["output_similarity"] = 1.5
        with pytest.raises(Exception):
            VitalsSnapshot.model_validate(data)

        data["output_similarity"] = -0.1
        with pytest.raises(Exception):
            VitalsSnapshot.model_validate(data)
