"""Tests for agent_vitals.monitor (AgentVitals class)."""

import pytest

from agent_vitals import AgentVitals, VitalsConfig, VitalsSnapshot, RawSignals
from agent_vitals.adapters import TelemetryAdapter
from agent_vitals.exceptions import AdapterError


class TestAgentVitals:
    """Tests for the AgentVitals monitor class."""

    def test_basic_step(self) -> None:
        monitor = AgentVitals(mission_id="test")
        snap = monitor.step(
            findings_count=3,
            coverage_score=0.5,
            total_tokens=5000,
            error_count=0,
        )
        assert isinstance(snap, VitalsSnapshot)
        assert snap.mission_id == "test"
        assert snap.loop_index == 0
        assert snap.signals.findings_count == 3

    def test_ratio_fields_and_declining_trajectory(self) -> None:
        """Snapshots should include ratio fields and track declining steps."""
        monitor = AgentVitals(mission_id="ratio-trend")

        snap1 = monitor.step(
            findings_count=2,
            sources_count=2,
            coverage_score=0.2,
            total_tokens=500,
            error_count=0,
        )
        assert snap1.source_finding_ratio == pytest.approx(1.0)
        assert snap1.ratio_trend == "insufficient_data"
        assert snap1.ratio_declining_steps == 0

        snap2 = monitor.step(
            findings_count=4,
            sources_count=2,
            coverage_score=0.3,
            total_tokens=1000,
            error_count=0,
        )
        assert snap2.source_finding_ratio == pytest.approx(0.5)
        assert snap2.ratio_trend == "declining"
        assert snap2.ratio_declining_steps == 1

        snap3 = monitor.step(
            findings_count=8,
            sources_count=2,
            coverage_score=0.4,
            total_tokens=1500,
            error_count=0,
        )
        assert snap3.source_finding_ratio == pytest.approx(0.25)
        assert snap3.ratio_trend == "declining"
        assert snap3.ratio_declining_steps == 2

    def test_loop_index_increments(self) -> None:
        monitor = AgentVitals(mission_id="test")
        for i in range(5):
            snap = monitor.step(
                findings_count=i,
                coverage_score=i * 0.1,
                total_tokens=i * 1000,
                error_count=0,
            )
            assert snap.loop_index == i
        assert monitor.loop_index == 5

    def test_history_grows(self) -> None:
        monitor = AgentVitals(mission_id="test")
        for i in range(3):
            monitor.step(
                findings_count=i,
                coverage_score=0.3,
                total_tokens=i * 1000,
                error_count=0,
            )
        assert len(monitor.history) == 3

    def test_reset_clears_state(self) -> None:
        monitor = AgentVitals(mission_id="test")
        monitor.step(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
        )
        old_run_id = monitor._run_id
        monitor.reset()
        assert len(monitor.history) == 0
        assert monitor.loop_index == 0
        assert monitor.health_state == "healthy"
        assert monitor._run_id != old_run_id

    def test_summary(self) -> None:
        monitor = AgentVitals(mission_id="test-summary")
        monitor.step(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
        )
        summary = monitor.summary()
        assert summary["mission_id"] == "test-summary"
        assert summary["total_steps"] == 1
        assert summary["health_state"] == "healthy"
        assert summary["any_loop_detected"] is False

    def test_step_from_signals(self) -> None:
        monitor = AgentVitals(mission_id="test")
        signals = RawSignals(
            findings_count=5,
            coverage_score=0.7,
            total_tokens=8000,
            error_count=0,
        )
        snap = monitor.step_from_signals(signals)
        assert snap.signals.findings_count == 5
        assert snap.signals.coverage_score == 0.7

    def test_step_from_state_with_adapter(self) -> None:
        adapter = TelemetryAdapter()
        monitor = AgentVitals(mission_id="test", adapter=adapter)
        state = {
            "cumulative_outputs": 3,
            "coverage_score": 0.4,
            "cumulative_tokens": 5000,
            "cumulative_errors": 0,
        }
        snap = monitor.step_from_state(state)
        assert snap.signals.findings_count == 3
        assert snap.signals.coverage_score == 0.4

    def test_step_from_state_without_adapter_raises(self) -> None:
        monitor = AgentVitals(mission_id="test")
        with pytest.raises(AdapterError, match="No adapter configured"):
            monitor.step_from_state({"foo": "bar"})

    def test_from_yaml(self, tmp_path: object) -> None:
        """Test creating AgentVitals from a YAML config file."""
        import yaml
        from pathlib import Path

        yaml_path = Path(str(tmp_path)) / "test_thresholds.yaml"
        yaml_path.write_text(yaml.dump({
            "loop_consecutive_count": 4,
            "stuck_dm_threshold": 0.2,
        }))
        monitor = AgentVitals.from_yaml(yaml_path, mission_id="yaml-test")
        assert monitor.config.loop_consecutive_count == 4
        assert monitor.config.stuck_dm_threshold == 0.2

    def test_config_property(self) -> None:
        cfg = VitalsConfig(loop_consecutive_count=3)
        monitor = AgentVitals(config=cfg, mission_id="test")
        assert monitor.config.loop_consecutive_count == 3


class TestDetectionIntegration:
    """Integration tests for detection through the monitor."""

    def test_failure_detection_on_flat_coverage(self) -> None:
        """Flat coverage should trigger a failure detector."""
        config = VitalsConfig(workflow_stuck_enabled="all")
        monitor = AgentVitals(
            config=config,
            mission_id="test-stuck",
            workflow_type="research",
        )
        # Run 8 steps with flat coverage
        any_failure = False
        for i in range(8):
            snap = monitor.step(
                findings_count=2,
                coverage_score=0.3,
                total_tokens=(i + 1) * 2000,
                error_count=0,
            )
            if snap.loop_detected or snap.stuck_detected:
                any_failure = True
                assert snap.detector_priority in {"loop", "stuck"}
        assert any_failure, "Expected detector firing on flat coverage"

    def test_healthy_run_no_false_positives(self) -> None:
        """Increasing findings and coverage should not trigger detections."""
        config = VitalsConfig(workflow_stuck_enabled="all")
        monitor = AgentVitals(
            config=config,
            mission_id="test-healthy",
            workflow_type="research",
        )
        for i in range(5):
            snap = monitor.step(
                findings_count=(i + 1) * 3,
                coverage_score=min(1.0, (i + 1) * 0.2),
                total_tokens=(i + 1) * 3000,
                error_count=0,
            )
            assert not snap.loop_detected, f"False loop at step {i}"
