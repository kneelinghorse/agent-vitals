"""Integration tests for agent-vitals: end-to-end monitor lifecycle.

Tests cover full monitor sessions, summary generation, reset behavior,
adapter integration, and exporter wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from agent_vitals import AgentVitals, JSONLExporter, VitalsSnapshot
from agent_vitals.schema import RawSignals


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


def test_monitor_lifecycle_healthy_run() -> None:
    """Full lifecycle: init → step×6 → summary → reset."""
    monitor = AgentVitals(mission_id="lifecycle-test", run_id="run-1")

    for i in range(6):
        snapshot = monitor.step(
            findings_count=i + 1,
            coverage_score=min(1.0, 0.15 * (i + 1)),
            total_tokens=500 * (i + 1),
            error_count=0,
        )
        assert snapshot.loop_index == i
        assert snapshot.mission_id == "lifecycle-test"

    assert monitor.loop_index == 6
    assert len(monitor.history) == 6
    assert monitor.health_state == "healthy"

    summary = monitor.summary()
    assert summary["total_steps"] == 6
    assert summary["any_loop_detected"] is False
    assert summary["any_stuck_detected"] is False

    monitor.reset()
    assert monitor.loop_index == 0
    assert len(monitor.history) == 0


def test_monitor_detects_failure_mid_run() -> None:
    """Monitor should detect failure state when coverage stagnates."""
    monitor = AgentVitals(mission_id="stuck-test")

    detected_at = None
    for i in range(8):
        snapshot = monitor.step(
            findings_count=3,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            error_count=0,
        )
        if snapshot.any_failure and detected_at is None:
            detected_at = i

    assert detected_at is not None, "Failure should be detected during stagnant run"
    summary = monitor.summary()
    assert summary["any_loop_detected"] is True or summary["any_stuck_detected"] is True
    assert summary["first_detection_at"] is not None


def test_monitor_context_manager() -> None:
    """Context manager should work for scoped monitoring."""
    with AgentVitals(mission_id="ctx-test") as monitor:
        monitor.step(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=500,
            error_count=0,
        )
        assert monitor.loop_index == 1


def test_monitor_step_from_signals() -> None:
    """step_from_signals should accept a pre-built RawSignals object."""
    monitor = AgentVitals(mission_id="signals-test")
    signals = RawSignals(
        findings_count=5,
        coverage_score=0.7,
        total_tokens=3000,
        error_count=0,
    )
    snapshot = monitor.step_from_signals(signals)
    assert snapshot.signals.findings_count == 5
    assert snapshot.loop_index == 0


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


class _DummyAdapter:
    """Minimal adapter for testing step_from_state."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        return RawSignals(
            findings_count=int(state.get("outputs", 0)),
            coverage_score=float(state.get("progress", 0.0)),
            total_tokens=int(state.get("tokens", 0)),
            error_count=int(state.get("errors", 0)),
        )


def test_monitor_step_from_state_with_adapter() -> None:
    """step_from_state should use the configured adapter."""
    monitor = AgentVitals(mission_id="adapter-test", adapter=_DummyAdapter())
    snapshot = monitor.step_from_state({
        "outputs": 3,
        "progress": 0.6,
        "tokens": 2000,
        "errors": 0,
    })
    assert snapshot.signals.findings_count == 3
    assert snapshot.signals.coverage_score == pytest.approx(0.6)


def test_monitor_step_from_state_without_adapter_raises() -> None:
    """step_from_state should raise AdapterError without adapter configured."""
    from agent_vitals.exceptions import AdapterError

    monitor = AgentVitals(mission_id="no-adapter")
    with pytest.raises(AdapterError):
        monitor.step_from_state({"outputs": 1})


# ---------------------------------------------------------------------------
# Exporter integration
# ---------------------------------------------------------------------------


def test_monitor_with_jsonl_exporter_e2e(tmp_path: Path) -> None:
    """End-to-end: monitor with JSONL exporter produces valid output."""
    exporter = JSONLExporter(directory=tmp_path, layout="per_run")

    with AgentVitals(
        mission_id="e2e-export",
        run_id="run-e2e",
        exporters=[exporter],
    ) as monitor:
        for i in range(5):
            monitor.step(
                findings_count=i + 1,
                coverage_score=min(1.0, 0.2 * (i + 1)),
                total_tokens=500 * (i + 1),
                error_count=0,
            )

    # Verify JSONL output
    path = tmp_path / "e2e-export" / "run-e2e.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 5

    for i, line in enumerate(lines):
        data = json.loads(line)
        snap = VitalsSnapshot.model_validate(data)
        assert snap.loop_index == i
        assert snap.signals.findings_count == i + 1


def test_monitor_multiple_exporters(tmp_path: Path) -> None:
    """Multiple exporters should all receive snapshots."""
    dir1 = tmp_path / "exp1"
    dir2 = tmp_path / "exp2"
    exp1 = JSONLExporter(directory=dir1, layout="per_run")
    exp2 = JSONLExporter(directory=dir2, layout="append")

    monitor = AgentVitals(
        mission_id="multi-exp",
        run_id="run-m",
        exporters=[exp1, exp2],
    )
    monitor.step(findings_count=1, coverage_score=0.5, total_tokens=500, error_count=0)
    monitor.step(findings_count=2, coverage_score=0.7, total_tokens=1000, error_count=0)

    assert (dir1 / "multi-exp" / "run-m.jsonl").exists()
    assert (dir2 / "multi-exp.jsonl").exists()

    lines1 = (dir1 / "multi-exp" / "run-m.jsonl").read_text().strip().split("\n")
    lines2 = (dir2 / "multi-exp.jsonl").read_text().strip().split("\n")
    assert len(lines1) == 2
    assert len(lines2) == 2


# ---------------------------------------------------------------------------
# Run ID uniqueness
# ---------------------------------------------------------------------------


def test_reset_generates_new_run_id() -> None:
    """Each reset should generate a new run_id."""
    monitor = AgentVitals(mission_id="runid-test")
    first_run_id = monitor._run_id
    monitor.step(findings_count=1, coverage_score=0.5, total_tokens=500, error_count=0)
    monitor.reset()
    second_run_id = monitor._run_id
    assert first_run_id != second_run_id
