"""Integration tests for CUSUM alarm wiring in AgentVitals."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_vitals import AgentVitals
from agent_vitals.schema import VitalsSnapshot


def test_snapshot_includes_cusum_fields() -> None:
    """Each snapshot should include CUSUM alarm metadata fields."""
    monitor = AgentVitals(mission_id="cusum-fields", workflow_type="research")
    snap = monitor.step(
        findings_count=1,
        coverage_score=0.2,
        total_tokens=1000,
        error_count=0,
    )
    assert snap.cusum_alarm is False
    assert isinstance(snap.cusum_alarm_metrics, list)
    assert isinstance(snap.cusum_scores, dict)
    assert "token_usage_delta" in snap.cusum_scores
    assert "findings_count_delta" in snap.cusum_scores


def test_f02_like_pattern_triggers_cusum_by_step_three() -> None:
    """Findings inflation should trigger CUSUM early (step 3 or earlier)."""
    monitor = AgentVitals(mission_id="cusum-f02", workflow_type="research")
    alarm_step: int | None = None

    trajectory = [
        (1, 1000),
        (3, 2200),
        (7, 3600),
        (12, 5200),
    ]
    for index, (findings, tokens) in enumerate(trajectory):
        snap = monitor.step(
            findings_count=findings,
            sources_count=3,
            coverage_score=0.5,
            total_tokens=tokens,
            query_count=index + 1,
            unique_domains=1,
            error_count=0,
        )
        if alarm_step is None and snap.cusum_alarm:
            alarm_step = index

    assert alarm_step is not None, "CUSUM alarm was expected on F02-like inflation pattern"
    assert alarm_step <= 3


def test_real_healthy_r01_to_r08_have_no_cusum_alarm() -> None:
    """Healthy AV26 real traces should not raise CUSUM alarms."""
    repo_root = Path(__file__).resolve().parents[2]
    traces_dir = repo_root / "checkpoints" / "vitals_corpus" / "av26_real" / "traces"
    if not traces_dir.exists():
        pytest.skip(f"Corpus not found at {traces_dir}")

    healthy_ids = [f"AV26.R0{i}" for i in range(1, 9)]
    for trace_id in healthy_ids:
        path = traces_dir / f"{trace_id}.jsonl"
        if not path.exists():
            pytest.skip(f"Missing expected healthy trace: {path}")

        snapshots = [
            VitalsSnapshot.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        monitor = AgentVitals(mission_id=f"cusum-{trace_id}", workflow_type="real")
        for snap in snapshots:
            current = monitor.step_from_signals(snap.signals)
            assert current.cusum_alarm is False, f"Unexpected CUSUM alarm for {trace_id}"
