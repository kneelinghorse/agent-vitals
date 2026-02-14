"""Per-detector CI gate utilities for backtest promotion decisions."""

from __future__ import annotations

import math
from typing import Any

from .backtest import ConfusionCounts

_WILSON_Z_95 = 1.959963984540054


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


def wilson_interval(successes: int, trials: int, *, z: float = _WILSON_Z_95) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a Bernoulli proportion."""
    if trials <= 0:
        return (0.0, 1.0)

    p = successes / trials
    z2 = z * z
    denom = 1.0 + (z2 / trials)
    center = (p + (z2 / (2.0 * trials))) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def metrics_with_ci(confusion: ConfusionCounts) -> dict[str, Any]:
    """Return confusion metrics enriched with 95% CI metadata."""
    precision_den = confusion.tp + confusion.fp
    recall_den = confusion.tp + confusion.fn
    p_lo, p_hi = wilson_interval(confusion.tp, precision_den)
    r_lo, r_hi = wilson_interval(confusion.tp, recall_den)
    f1_lo = f1_from_precision_recall(p_lo, r_lo)
    f1_hi = f1_from_precision_recall(p_hi, r_hi)

    return {
        "precision": confusion.precision,
        "recall": confusion.recall,
        "f1": confusion.f1,
        "tp": confusion.tp,
        "fp": confusion.fp,
        "fn": confusion.fn,
        "tn": confusion.tn,
        "precision_ci95": {"lower": p_lo, "upper": p_hi},
        "recall_ci95": {"lower": r_lo, "upper": r_hi},
        "f1_ci95": {"lower": f1_lo, "upper": f1_hi},
        "positive_count": confusion.tp + confusion.fn,
    }


def evaluate_promotion(
    *,
    detector_name: str,
    metric: dict[str, Any],
    min_positives: int,
    min_precision_lb: float,
    min_recall_lb: float,
) -> dict[str, Any]:
    """Evaluate whether a detector should be promoted from soft to hard gate."""
    positive_count = int(metric.get("positive_count", 0))
    precision_lb = float(metric["precision_ci95"]["lower"])
    recall_lb = float(metric["recall_ci95"]["lower"])

    reasons: list[str] = []
    if positive_count < min_positives:
        reasons.append(
            f"positive_count={positive_count} < min_positives={min_positives}"
        )
    if precision_lb < min_precision_lb:
        reasons.append(
            f"precision_lb={precision_lb:.3f} < min_precision_lb={min_precision_lb:.3f}"
        )
    if recall_lb < min_recall_lb:
        reasons.append(
            f"recall_lb={recall_lb:.3f} < min_recall_lb={min_recall_lb:.3f}"
        )

    qualifies = not reasons
    return {
        "detector": detector_name,
        "qualifies_for_hard_gate": qualifies,
        "status": "hard" if qualifies else "soft",
        "positive_count": positive_count,
        "precision_lb": precision_lb,
        "recall_lb": recall_lb,
        "reasons": reasons,
    }


def evaluate_hard_gate(
    *,
    metric: dict[str, Any],
    min_precision_lb: float,
    min_recall_lb: float,
) -> tuple[bool, list[str]]:
    """Evaluate hard-gate pass/fail using lower confidence bounds."""
    precision_lb = float(metric["precision_ci95"]["lower"])
    recall_lb = float(metric["recall_ci95"]["lower"])

    failures: list[str] = []
    if precision_lb < min_precision_lb:
        failures.append(
            f"precision_lb={precision_lb:.3f} < min_precision_lb={min_precision_lb:.3f}"
        )
    if recall_lb < min_recall_lb:
        failures.append(
            f"recall_lb={recall_lb:.3f} < min_recall_lb={min_recall_lb:.3f}"
        )
    return (not failures, failures)
