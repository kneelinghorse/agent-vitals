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
    load_dataset,
    load_labels,
    run_backtest,
)
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
            "t1": {"labels": ["thrash", "stuck"]},
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(labels_data))
        labels = load_labels(path)
        assert labels["t1"]["thrash_at"] == {0}
        assert labels["t1"]["stuck_at"] == {0}

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
            "h1": {"loop_at": set(), "stuck_at": set(), "thrash_at": set(), "runaway_cost_at": set()},
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
            "t1": {"loop_at": set(), "stuck_at": set(), "thrash_at": set(), "runaway_cost_at": set()},
        }

        report = run_backtest(ds, labels)
        d = report.as_dict()
        assert "dataset" in d
        assert "config" in d
        assert "detectors" in d
        assert "composite_any" in d
        assert "loop" in d["detectors"]
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
            "err1": {"loop_at": set(), "stuck_at": set(), "thrash_at": {0}, "runaway_cost_at": set()},
        }

        report = run_backtest(ds, labels)
        assert report.detectors["thrash"].confusion.tp >= 1
