"""Integration tests for source-to-finding ratio confabulation signaling."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.schema import VitalsSnapshot


def _load_trace(path: Path) -> list[VitalsSnapshot]:
    return [
        VitalsSnapshot.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_f02_trace_ratio_signal_fires_high_confidence() -> None:
    """AV26.F02 should produce high-confidence ratio-based confabulation signal."""
    repo_root = Path(__file__).resolve().parents[2]
    trace_path = repo_root / "checkpoints" / "vitals_corpus" / "av26_real" / "traces" / "AV26.F02.jsonl"
    if not trace_path.exists():
        pytest.skip(f"Corpus trace missing: {trace_path}")

    snapshots = _load_trace(trace_path)
    config = VitalsConfig()
    history: list[VitalsSnapshot] = []
    first_ratio_detection_step: int | None = None
    strongest_ratio_confidence = 0.0

    for idx, snap in enumerate(snapshots):
        result = detect_loop(snap, history, config=config, workflow_type="real")
        trigger = result.confabulation_trigger or ""
        if "source_finding_ratio" in trigger and result.confabulation_detected:
            strongest_ratio_confidence = max(
                strongest_ratio_confidence,
                result.confabulation_confidence,
            )
            if first_ratio_detection_step is None:
                first_ratio_detection_step = idx
            assert result.detector_priority == "confabulation"
            assert result.loop_detected is False
            assert result.stuck_detected is False
        history.append(snap)

    assert first_ratio_detection_step is not None, "Expected ratio-based confabulation signal on F02"
    # Step index is zero-based; <=3 means detected by the 4th snapshot.
    assert first_ratio_detection_step <= 3
    assert strongest_ratio_confidence >= 0.8


def test_healthy_r01_to_r08_have_no_ratio_confab_false_positives() -> None:
    """Healthy AV26 real traces R01-R08 should not produce ratio confabulation signals."""
    repo_root = Path(__file__).resolve().parents[2]
    traces_dir = repo_root / "checkpoints" / "vitals_corpus" / "av26_real" / "traces"
    if not traces_dir.exists():
        pytest.skip(f"Corpus not found: {traces_dir}")

    config = VitalsConfig()
    for trace_id in [f"AV26.R0{i}" for i in range(1, 9)]:
        path = traces_dir / f"{trace_id}.jsonl"
        if not path.exists():
            pytest.skip(f"Missing healthy trace: {path}")

        history: list[VitalsSnapshot] = []
        for snap in _load_trace(path):
            result = detect_loop(snap, history, config=config, workflow_type="real")
            trigger = result.confabulation_trigger or ""
            assert "source_finding_ratio" not in trigger, f"Unexpected ratio confab signal for {trace_id}"
            assert result.confabulation_detected is False
            history.append(snap)


def test_confabulation_priority_wins_when_multiple_signals_present() -> None:
    """When confab and loop signals co-occur, detector priority should be confabulation."""
    snapshots = [
        VitalsSnapshot.model_validate(
            {
                "mission_id": "confab-priority",
                "loop_index": 1,
                "signals": {
                    "findings_count": 3,
                    "sources_count": 3,
                    "coverage_score": 0.4,
                    "total_tokens": 1000,
                    "error_count": 0,
                    "query_count": 1,
                    "unique_domains": 1,
                },
                "metrics": {
                    "cv_coverage": 0.2,
                    "cv_findings_rate": 0.2,
                    "dm_coverage": 0.0,
                    "dm_findings": 0.0,
                    "qpf_tokens": 0.5,
                    "cs_effort": 0.5,
                },
                "health_state": "healthy",
                "output_similarity": 0.9,
            }
        ),
        VitalsSnapshot.model_validate(
            {
                "mission_id": "confab-priority",
                "loop_index": 2,
                "signals": {
                    "findings_count": 6,
                    "sources_count": 3,
                    "coverage_score": 0.4,
                    "total_tokens": 2000,
                    "error_count": 0,
                    "query_count": 2,
                    "unique_domains": 1,
                },
                "metrics": {
                    "cv_coverage": 0.2,
                    "cv_findings_rate": 0.2,
                    "dm_coverage": 0.0,
                    "dm_findings": 0.0,
                    "qpf_tokens": 0.5,
                    "cs_effort": 0.5,
                },
                "health_state": "healthy",
                "output_similarity": 0.9,
            }
        ),
        VitalsSnapshot.model_validate(
            {
                "mission_id": "confab-priority",
                "loop_index": 3,
                "signals": {
                    "findings_count": 13,
                    "sources_count": 3,
                    "coverage_score": 0.4,
                    "total_tokens": 3000,
                    "error_count": 0,
                    "query_count": 3,
                    "unique_domains": 1,
                },
                "metrics": {
                    "cv_coverage": 0.2,
                    "cv_findings_rate": 0.2,
                    "dm_coverage": 0.0,
                    "dm_findings": 0.0,
                    "qpf_tokens": 0.5,
                    "cs_effort": 0.5,
                },
                "health_state": "healthy",
                "output_similarity": 0.95,
            }
        ),
    ]

    result = detect_loop(snapshots[-1], snapshots[:-1], config=VitalsConfig())
    assert result.confabulation_detected is True
    assert result.detector_priority == "confabulation"
    assert result.loop_detected is False
    assert result.stuck_detected is False


def test_loop_and_stuck_classifications_remain_intact() -> None:
    """Non-confab traces should remain classified by their native detector."""
    repo_root = Path(__file__).resolve().parents[2]
    synth_dir = repo_root / "checkpoints" / "vitals_corpus" / "av05_synth" / "traces"

    loop_trace = synth_dir / "AV04.SYNTH.s0.loop_plateau.0000.jsonl"
    stuck_trace = synth_dir / "AV04.SYNTH.s0.stuck_coverage.0000.jsonl"
    if not loop_trace.exists() or not stuck_trace.exists():
        pytest.skip("Synthetic corpus traces not found")

    config = VitalsConfig()

    # Loop trace
    loop_history: list[VitalsSnapshot] = []
    loop_priority_seen = False
    for snap in _load_trace(loop_trace):
        result = detect_loop(snap, loop_history, config=config, workflow_type="synthetic")
        if result.detector_priority == "loop":
            loop_priority_seen = True
        assert result.confabulation_detected is False
        loop_history.append(snap)
    assert loop_priority_seen is True

    # Stuck trace
    stuck_history: list[VitalsSnapshot] = []
    stuck_priority_seen = False
    for snap in _load_trace(stuck_trace):
        result = detect_loop(snap, stuck_history, config=config, workflow_type="synthetic")
        if result.detector_priority == "stuck":
            stuck_priority_seen = True
        assert result.confabulation_detected is False
        stuck_history.append(snap)
    assert stuck_priority_seen is True
