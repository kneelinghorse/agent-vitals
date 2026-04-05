"""Unit tests for stratified backtest evaluation (av32-m03)."""

from __future__ import annotations

import pytest

from agent_vitals.backtest import ConfusionCounts
from agent_vitals.stratified import (
    label_distribution,
    macro_average,
    micro_average,
    sample_size_warnings,
    stratified_report,
)


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


def test_label_distribution_counts_per_detector() -> None:
    labels = {
        "t1": {"loop_at": {3}, "stuck_at": set(), "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()},
        "t2": {"loop_at": {1}, "stuck_at": {5}, "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()},
        "t3": {"loop_at": set(), "stuck_at": set(), "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()},
    }
    dist = label_distribution(labels)
    assert dist["total_traces"] == 3
    assert dist["healthy"] == 1
    assert dist["per_detector"]["loop"] == 2
    assert dist["per_detector"]["stuck"] == 1
    assert dist["per_detector"]["thrash"] == 0


def test_label_distribution_imbalance_ratio() -> None:
    labels = {
        f"t{i}": {"loop_at": {0}, "stuck_at": set(), "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()}
        for i in range(10)
    }
    labels["s1"] = {"loop_at": set(), "stuck_at": {0}, "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()}
    dist = label_distribution(labels)
    assert dist["per_detector"]["loop"] == 10
    assert dist["per_detector"]["stuck"] == 1
    assert dist["imbalance_ratio"] == 10.0


def test_label_distribution_all_healthy() -> None:
    labels = {
        "t1": {"loop_at": set(), "stuck_at": set(), "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()},
    }
    dist = label_distribution(labels)
    assert dist["healthy"] == 1
    assert dist["imbalance_ratio"] == float("inf")


# ---------------------------------------------------------------------------
# Macro average
# ---------------------------------------------------------------------------


def test_macro_average_equal_weight() -> None:
    metrics = {
        "loop": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "positive_count": 10},
        "stuck": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "positive_count": 5},
    }
    result = macro_average(metrics)
    assert result["precision"] == pytest.approx(0.75, abs=0.01)
    assert result["recall"] == pytest.approx(0.75, abs=0.01)
    assert result["detector_count"] == 2


def test_macro_average_skips_zero_positive_detectors() -> None:
    metrics = {
        "loop": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "positive_count": 10},
        "thrash": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_count": 0},
    }
    result = macro_average(metrics)
    assert result["precision"] == pytest.approx(1.0)
    assert result["detector_count"] == 1


def test_macro_average_empty() -> None:
    result = macro_average({})
    assert result["detector_count"] == 0
    assert result["precision"] == 0.0


# ---------------------------------------------------------------------------
# Micro average
# ---------------------------------------------------------------------------


def test_micro_average_sums_counts() -> None:
    counts = {
        "loop": ConfusionCounts(tp=10, fp=2, fn=1, tn=50),
        "stuck": ConfusionCounts(tp=5, fp=3, fn=4, tn=40),
    }
    result = micro_average(counts)
    assert result["tp"] == 15
    assert result["fp"] == 5
    assert result["fn"] == 5
    assert result["precision"] == pytest.approx(15 / 20, abs=0.01)
    assert result["recall"] == pytest.approx(15 / 20, abs=0.01)


def test_micro_average_empty_counts() -> None:
    counts = {"loop": ConfusionCounts()}
    result = micro_average(counts)
    assert result["tp"] == 0
    assert result["precision"] == 0.0


# ---------------------------------------------------------------------------
# Sample size warnings
# ---------------------------------------------------------------------------


def test_sample_warnings_flags_insufficient() -> None:
    metrics = {
        "loop": {"positive_count": 50},
        "thrash": {"positive_count": 3},
        "confabulation": {"positive_count": 7},
    }
    warnings = sample_size_warnings(metrics, min_positives=8)
    assert "thrash" in warnings
    assert "confabulation" in warnings
    assert "loop" not in warnings
    assert warnings["thrash"]["positive_count"] == 3


def test_sample_warnings_none_when_all_sufficient() -> None:
    metrics = {
        "loop": {"positive_count": 100},
        "stuck": {"positive_count": 20},
    }
    warnings = sample_size_warnings(metrics, min_positives=8)
    assert warnings == {}


# ---------------------------------------------------------------------------
# Stratified report
# ---------------------------------------------------------------------------


def test_stratified_report_structure() -> None:
    detector_metrics = {
        "loop": {"precision": 0.9, "recall": 0.95, "f1": 0.92, "positive_count": 50, "tp": 48, "fp": 5, "fn": 2, "tn": 100},
        "stuck": {"precision": 0.6, "recall": 0.3, "f1": 0.4, "positive_count": 20, "tp": 6, "fp": 4, "fn": 14, "tn": 100},
    }
    detector_counts = {
        "loop": ConfusionCounts(tp=48, fp=5, fn=2, tn=100),
        "stuck": ConfusionCounts(tp=6, fp=4, fn=14, tn=100),
    }
    labels = {
        "t1": {"loop_at": {0}, "stuck_at": set(), "thrash_at": set(), "confabulation_at": set(), "runaway_cost_at": set()},
    }
    report = stratified_report(detector_metrics, detector_counts, labels, min_positives=8)
    assert "per_class" in report
    assert "macro_average" in report
    assert "micro_average" in report
    assert "label_distribution" in report
    assert "sample_warnings" in report
    assert report["macro_average"]["detector_count"] == 2
