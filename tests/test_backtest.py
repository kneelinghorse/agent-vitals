"""Tests for the backtest harness — ported from DeepSearch test suite.

Covers: ConfusionCounts, DetectorResult, BacktestReport, load_labels,
load_dataset, run_backtest, and _replay_trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_vitals.backtest import (
    ConfusionCounts,
    DetectorResult,
    Labels,
    _replay_trace,
    load_dataset,
    load_labels,
    resolve_workflow_type,
    run_backtest,
)
from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import LoopDetectionResult
from agent_vitals.detection.stop_rule import StopRuleSignals
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    mission_id: str = "t1",
    loop_index: int = 0,
    findings: int = 1,
    coverage: float = 0.5,
    tokens: int = 1000,
    errors: int = 0,
) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id=mission_id,
        loop_index=loop_index,
        signals=RawSignals(
            findings_count=findings,
            coverage_score=coverage,
            total_tokens=tokens,
            error_count=errors,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.1,
            cv_findings_rate=0.1,
            dm_coverage=0.3,
            dm_findings=0.3,
            qpf_tokens=0.5,
            cs_effort=0.5,
        ),
        health_state="healthy",
    )


def _write_jsonl(path: Path, snapshots: list[VitalsSnapshot]) -> None:
    with open(path, "w") as f:
        for s in snapshots:
            f.write(s.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# ConfusionCounts
# ---------------------------------------------------------------------------


class TestConfusionCounts:
    def test_perfect_classifier(self) -> None:
        cc = ConfusionCounts()
        cc.record(predicted=True, expected=True)
        cc.record(predicted=False, expected=False)
        assert cc.tp == 1 and cc.tn == 1
        assert cc.fp == 0 and cc.fn == 0
        assert cc.precision == 1.0
        assert cc.recall == 1.0
        assert cc.f1 == 1.0

    def test_all_false_positives(self) -> None:
        cc = ConfusionCounts()
        cc.record(predicted=True, expected=False)
        cc.record(predicted=True, expected=False)
        assert cc.precision == 0.0
        assert cc.recall == 0.0
        assert cc.f1 == 0.0

    def test_all_false_negatives(self) -> None:
        cc = ConfusionCounts()
        cc.record(predicted=False, expected=True)
        assert cc.precision == 0.0
        assert cc.recall == 0.0

    def test_empty_counts(self) -> None:
        cc = ConfusionCounts()
        assert cc.precision == 0.0
        assert cc.recall == 0.0
        assert cc.f1 == 0.0

    def test_as_dict(self) -> None:
        cc = ConfusionCounts(tp=3, fp=1, fn=2, tn=4)
        d = cc.as_dict()
        assert d["tp"] == 3
        assert d["precision"] == pytest.approx(0.75)
        assert d["recall"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# DetectorResult
# ---------------------------------------------------------------------------


class TestDetectorResult:
    def test_delegates_to_confusion(self) -> None:
        cc = ConfusionCounts(tp=5, fp=0, fn=0, tn=5)
        dr = DetectorResult(name="test", confusion=cc)
        assert dr.precision == 1.0
        assert dr.recall == 1.0
        assert dr.f1 == 1.0
        assert dr.as_dict()["name"] == "test"


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_onset_format(self, tmp_path: Path) -> None:
        labels_data = {
            "trace-1": {"loop_at": [3, 5], "stuck_at": [], "thrash_at": [], "runaway_cost_at": []},
            "trace-2": {"loop_at": [], "stuck_at": [7], "thrash_at": [], "runaway_cost_at": []},
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(labels_data))
        labels = load_labels(path)
        assert labels["trace-1"]["loop_at"] == {3, 5}
        assert labels["trace-2"]["stuck_at"] == {7}

    def test_cross_agent_format(self, tmp_path: Path) -> None:
        labels_data = {
            "t1": {"label": "loop", "agent_framework": "openai"},
            "t2": {"label": "healthy"},
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(labels_data))
        labels = load_labels(path)
        assert labels["t1"]["loop_at"] == {0}
        assert labels["t2"]["loop_at"] == set()

    def test_corpus_format(self, tmp_path: Path) -> None:
        labels_data = {
            "t1": {"labels": ["thrash", "stuck", "confabulation"]},
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(labels_data))
        labels = load_labels(path)
        assert labels["t1"]["thrash_at"] == {0}
        assert labels["t1"]["stuck_at"] == {0}
        assert labels["t1"]["confabulation_at"] == {0}

    def test_rejects_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            load_labels(path)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_loads_jsonl_files(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_jsonl(traces_dir / "t1.jsonl", [_snapshot("t1", i) for i in range(3)])
        _write_jsonl(traces_dir / "t2.jsonl", [_snapshot("t2", i) for i in range(2)])

        ds = load_dataset(traces_dir)
        assert ds.trace_count == 2
        assert ds.snapshot_count == 5
        assert len(ds.traces["t1"]) == 3

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent")

    def test_raises_on_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            load_dataset(f)

    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            load_dataset(empty)

    def test_skips_invalid_lines(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        path = traces_dir / "t1.jsonl"
        snap = _snapshot("t1", 0)
        path.write_text(snap.model_dump_json() + "\n" + "INVALID JSON\n")

        ds = load_dataset(traces_dir)
        assert ds.trace_count == 1
        assert ds.invalid_lines.get("t1") == 1


class TestResolveWorkflowType:
    def test_prefers_trace_markers(self) -> None:
        assert resolve_workflow_type("av31-model.bc.trace", "real") == "build"
        assert resolve_workflow_type("av31-model.rc.trace", "real") == "research"

    def test_falls_back_to_default(self) -> None:
        assert resolve_workflow_type("plain-trace-id", "synthetic") == "synthetic"


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_healthy_traces_no_detections(self, tmp_path: Path) -> None:
        """All-healthy traces with no labels should produce all TN."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        # Progressive healthy trace
        snaps = [
            _snapshot("h1", i, findings=i + 1, coverage=0.1 * (i + 1), tokens=500 * (i + 1))
            for i in range(5)
        ]
        _write_jsonl(traces_dir / "h1.jsonl", snaps)

        ds = load_dataset(traces_dir)
        labels: Labels = {
            "h1": {
                "loop_at": set(),
                "confabulation_at": set(),
                "stuck_at": set(),
                "thrash_at": set(),
                "runaway_cost_at": set(),
            },
        }

        report = run_backtest(ds, labels)
        assert report.composite_any.confusion.tn >= 1
        assert report.composite_any.confusion.fp == 0
        assert report.dataset_trace_count == 1

    def test_report_structure(self, tmp_path: Path) -> None:
        """BacktestReport.as_dict() should have expected keys."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_jsonl(traces_dir / "t1.jsonl", [_snapshot("t1", 0)])

        ds = load_dataset(traces_dir)
        labels: Labels = {
            "t1": {
                "loop_at": set(),
                "confabulation_at": set(),
                "stuck_at": set(),
                "thrash_at": set(),
                "runaway_cost_at": set(),
            },
        }

        report = run_backtest(ds, labels)
        d = report.as_dict()
        assert "dataset" in d
        assert "config" in d
        assert "detectors" in d
        assert "composite_any" in d
        assert "loop" in d["detectors"]
        assert "confabulation" in d["detectors"]
        assert d["dataset"]["trace_count"] == 1

    def test_thrash_detection_via_errors(self, tmp_path: Path) -> None:
        """Traces with high error_count should trigger thrash via stop_rule."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        snaps = [
            _snapshot("err1", i, findings=i + 1, coverage=0.1 * (i + 1), tokens=500 * (i + 1), errors=i * 2)
            for i in range(5)
        ]
        _write_jsonl(traces_dir / "err1.jsonl", snaps)

        ds = load_dataset(traces_dir)
        labels: Labels = {
            "err1": {
                "loop_at": set(),
                "confabulation_at": set(),
                "stuck_at": set(),
                "thrash_at": {0},
                "runaway_cost_at": set(),
            },
        }

        report = run_backtest(ds, labels)
        assert report.detectors["thrash"].confusion.tp >= 1

    def test_confabulation_detection_scored_first_class(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Confabulation labels should produce confabulation TP accounting."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_jsonl(traces_dir / "c1.jsonl", [_snapshot("c1", 0)])

        ds = load_dataset(traces_dir)
        labels: Labels = {
            "c1": {
                "loop_at": set(),
                "confabulation_at": {0},
                "stuck_at": set(),
                "thrash_at": set(),
                "runaway_cost_at": set(),
            }
        }

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: LoopDetectionResult(
                confabulation_detected=True,
                confabulation_confidence=0.85,
                confabulation_trigger="source_finding_ratio_low",
            ),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: StopRuleSignals(False, False, False, False),
        )

        report = run_backtest(ds, labels, config=VitalsConfig())
        confab = report.detectors["confabulation"]
        assert confab.confusion.tp == 1
        assert confab.precision == pytest.approx(1.0)
        assert confab.recall == pytest.approx(1.0)


class TestReplayTraceOverlap:
    def test_keeps_stuck_for_statistical_loop_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Statistical loop overlap (findings_plateau) keeps stuck (av32-m02)."""
        snapshots = [_snapshot("t1", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(stuck_detected=True, stuck_trigger="findings_plateau"),
                LoopDetectionResult(loop_detected=True, loop_trigger="findings_plateau"),
                LoopDetectionResult(),
            ]
        )
        stop_signals = iter(
            [
                StopRuleSignals(False, True, False, False),
                StopRuleSignals(True, False, False, False),
                StopRuleSignals(False, False, False, False),
            ]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="synthetic",
        )
        assert fired["loop"] is True
        assert fired["stuck"] is True  # Stuck preserved: loop is statistical, not content-based

    def test_suppresses_stuck_for_content_similarity_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Content similarity loop should suppress stuck at trace level (av32-m02)."""
        snapshots = [_snapshot("t1", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(stuck_detected=True, stuck_trigger="coverage_stagnation"),
                LoopDetectionResult(loop_detected=True, loop_trigger="content_similarity"),
                LoopDetectionResult(),
            ]
        )
        stop_signals = iter(
            [
                StopRuleSignals(False, True, False, False),
                StopRuleSignals(True, False, False, False),
                StopRuleSignals(False, False, False, False),
            ]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="synthetic",
        )
        assert fired["loop"] is True
        assert fired["stuck"] is False  # Content similarity is strong loop evidence

    def test_keeps_stuck_when_overlap_priority_is_stuck(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detector-priority stuck overlaps should survive trace-level arbitration."""
        snapshots = [_snapshot("t1", i) for i in range(4)]
        detections = iter(
            [
                LoopDetectionResult(),
                LoopDetectionResult(),
                LoopDetectionResult(),
                LoopDetectionResult(
                    loop_detected=True,
                    loop_confidence=0.95,
                    loop_trigger="content_similarity",
                    stuck_detected=True,
                    stuck_confidence=0.92,
                    stuck_trigger="short_run_zero_coverage",
                    detector_priority="stuck",
                ),
            ]
        )
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(4)]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="research",
        )
        assert fired["loop"] is True
        assert fired["stuck"] is True

    def test_confabulation_priority_does_not_count_secondary_stuck(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secondary stuck signals under confab priority should not count as stuck."""
        snapshots = [_snapshot("t1", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(),
                LoopDetectionResult(),
                LoopDetectionResult(
                    confabulation_detected=True,
                    confabulation_confidence=0.85,
                    confabulation_trigger="source_finding_ratio_low",
                    stuck_detected=True,
                    stuck_confidence=0.7,
                    stuck_trigger="late_onset_stagnation",
                    detector_priority="confabulation",
                ),
            ]
        )
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(3)]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="unknown",
        )
        assert fired["confabulation"] is True
        assert fired["stuck"] is False

    def test_keeps_stuck_for_mixed_overlap_with_thrash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed traces should keep stuck when thrash also appears."""
        snapshots = [_snapshot("t2", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(stuck_detected=True, stuck_trigger="coverage_stagnation"),
                LoopDetectionResult(loop_detected=True, loop_trigger="findings_plateau"),
                LoopDetectionResult(),
            ]
        )
        stop_signals = iter(
            [
                StopRuleSignals(False, True, False, False),
                StopRuleSignals(True, False, False, False),
                StopRuleSignals(False, False, True, False),
            ]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="synthetic",
        )
        assert fired["loop"] is True
        assert fired["thrash"] is True
        assert fired["stuck"] is True

    def test_burn_rate_anomaly_trigger_does_not_fire_stuck(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard for v1.14.1 burn-rate/stuck interaction.

        ``_collect_stuck_candidates`` in loop.py appends ``burn_rate_anomaly``
        as a stuck candidate with confidence 1.0 whenever the burn-rate
        gate passes. The replay layer filters out stuck whose winning
        trigger is ``burn_rate_anomaly`` because burn_rate_anomaly is
        really a runaway_cost signal. That filter doubles as an IMPLICIT
        stuck-suppression side-channel on short runaway-positive traces:
        because confidence 1.0 beats every other stuck candidate in
        arbitration, burn_rate_anomaly wins, then gets masked here, so
        stuck stays False.

        This test pins the filter. If it's ever weakened or removed, the
        crewai cross-framework gate regresses (pre-v1.14.1 crewai stuck
        FP=35, P_lb=0.7022, NO-GO). See thresholds.yaml crewai profile
        comment and backtest.py stuck_trigger filter for the full chain.
        """
        snapshots = [_snapshot("t1", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(),
                LoopDetectionResult(),
                LoopDetectionResult(
                    stuck_detected=True,
                    stuck_confidence=1.0,
                    stuck_trigger="burn_rate_anomaly",
                ),
            ]
        )
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(3)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )
        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="research",
        )
        # stuck must NOT fire when its winning trigger is burn_rate_anomaly.
        assert fired["stuck"] is False

    def test_final_step_adjudication_overrides_transient_confab(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Final-step verdict overrides streaming any-step confab (av-s06-m01).

        Simulates a Path 3 transient: an early window is weak (confab fires),
        later windows recover (final-step verdict says no confab). The
        trace-level label must reflect the final-step verdict.
        """
        snapshots = [_snapshot("t1", i) for i in range(4)]
        detections = iter(
            [
                LoopDetectionResult(),
                LoopDetectionResult(
                    confabulation_detected=True,
                    confabulation_confidence=0.75,
                    confabulation_trigger="verified_source_decoupling",
                ),
                LoopDetectionResult(),
                # Final iteration: bench-equivalent one-shot verdict says no.
                LoopDetectionResult(),
            ]
        )
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(4)]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="research",
        )
        assert fired["confabulation"] is False
        assert fired["any"] is False

    def test_final_step_adjudication_preserves_persistent_confab(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Final-step adjudication must NOT suppress confab when the
        whole-trace verdict still fires at the tail (av-s06-m01)."""
        snapshots = [_snapshot("t1", i) for i in range(3)]
        detections = iter(
            [
                LoopDetectionResult(),
                LoopDetectionResult(
                    confabulation_detected=True,
                    confabulation_confidence=0.7,
                    confabulation_trigger="verified_source_decoupling",
                ),
                LoopDetectionResult(
                    confabulation_detected=True,
                    confabulation_confidence=0.8,
                    confabulation_trigger="verified_source_decoupling",
                ),
            ]
        )
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(3)]
        )

        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        fired = _replay_trace(
            snapshots,
            config=VitalsConfig(),
            workflow_type="research",
        )
        assert fired["confabulation"] is True
