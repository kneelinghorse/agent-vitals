"""Tests for per-detector CI gate promotion utilities."""

from __future__ import annotations

import pytest

from agent_vitals.backtest import ConfusionCounts
from agent_vitals.ci_gate import (
    evaluate_hard_gate,
    evaluate_promotion,
    metrics_with_ci,
    wilson_interval,
)


def test_wilson_interval_known_case() -> None:
    """Wilson 95% interval should match known 9/10 bounds."""
    lower, upper = wilson_interval(9, 10)
    assert lower == pytest.approx(0.5958499732, rel=1e-6)
    assert upper == pytest.approx(0.9821237869, rel=1e-6)


def test_metrics_with_ci_contains_positive_count() -> None:
    """CI payload should include positive_count and bounded CI values."""
    cc = ConfusionCounts(tp=12, fp=3, fn=2, tn=9)
    metric = metrics_with_ci(cc)

    assert metric["positive_count"] == 14
    assert metric["precision"] == pytest.approx(12 / 15)
    assert metric["recall"] == pytest.approx(12 / 14)
    assert 0.0 <= metric["precision_ci95"]["lower"] <= metric["precision_ci95"]["upper"] <= 1.0
    assert 0.0 <= metric["recall_ci95"]["lower"] <= metric["recall_ci95"]["upper"] <= 1.0


def test_evaluate_promotion_qualifies_when_all_thresholds_met() -> None:
    """Detector should promote when positive count and CI lower bounds pass."""
    cc = ConfusionCounts(tp=40, fp=2, fn=2, tn=10)
    metric = metrics_with_ci(cc)
    decision = evaluate_promotion(
        detector_name="loop",
        metric=metric,
        min_positives=8,
        min_precision_lb=0.80,
        min_recall_lb=0.75,
    )

    assert decision["qualifies_for_hard_gate"] is True
    assert decision["status"] == "hard"
    assert decision["reasons"] == []


def test_evaluate_promotion_stays_soft_when_lb_or_support_fails() -> None:
    """Detector should remain soft when CI lower bounds or support are insufficient."""
    cc = ConfusionCounts(tp=5, fp=5, fn=3, tn=12)
    metric = metrics_with_ci(cc)
    decision = evaluate_promotion(
        detector_name="stuck",
        metric=metric,
        min_positives=8,
        min_precision_lb=0.80,
        min_recall_lb=0.75,
    )

    assert decision["qualifies_for_hard_gate"] is False
    assert decision["status"] == "soft"
    assert decision["reasons"]


def test_evaluate_hard_gate() -> None:
    """Hard-gate evaluator should return explicit failure reasons."""
    cc = ConfusionCounts(tp=10, fp=4, fn=5, tn=20)
    metric = metrics_with_ci(cc)
    passed, reasons = evaluate_hard_gate(
        metric=metric,
        min_precision_lb=0.80,
        min_recall_lb=0.75,
    )
    assert passed is False
    assert reasons
