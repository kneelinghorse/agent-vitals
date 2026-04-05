"""Stratified backtest evaluation for label-imbalanced corpora.

Provides per-class metrics with Wilson CI, macro/micro averages, label
distribution analysis, and sample-size warnings for gate promotion.
"""

from __future__ import annotations

from typing import Any

from .backtest import ConfusionCounts, Labels
from .ci_gate import f1_from_precision_recall

DETECTOR_NAMES = ("loop", "confabulation", "stuck", "thrash", "runaway_cost")
_LABEL_KEYS = {
    "loop": "loop_at",
    "confabulation": "confabulation_at",
    "stuck": "stuck_at",
    "thrash": "thrash_at",
    "runaway_cost": "runaway_cost_at",
}


def label_distribution(labels: Labels) -> dict[str, Any]:
    """Compute label distribution and imbalance metrics.

    Returns:
        Dict with per-detector positive counts, healthy count,
        total traces, and imbalance ratio.
    """
    per_detector: dict[str, int] = {name: 0 for name in DETECTOR_NAMES}
    healthy = 0
    total = len(labels)

    for _trace_id, onsets in labels.items():
        has_any = False
        for detector, key in _LABEL_KEYS.items():
            if onsets.get(key):
                per_detector[detector] += 1
                has_any = True
        if not has_any:
            healthy += 1

    counts = [c for c in per_detector.values() if c > 0]
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = (
        float(max_count) / float(min_count) if min_count > 0 else float("inf")
    )

    return {
        "total_traces": total,
        "healthy": healthy,
        "per_detector": per_detector,
        "max_count": max_count,
        "min_count": min_count,
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


def macro_average(detector_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Compute macro-averaged P/R/F1 (equal weight per detector).

    Only includes detectors with at least one positive sample.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for _name, m in detector_metrics.items():
        positive_count = int(m.get("positive_count", m.get("tp", 0) + m.get("fn", 0)))
        if positive_count > 0:
            precisions.append(float(m["precision"]))
            recalls.append(float(m["recall"]))
            f1s.append(float(m["f1"]))

    n = len(precisions)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "detector_count": 0}

    return {
        "precision": round(sum(precisions) / n, 4),
        "recall": round(sum(recalls) / n, 4),
        "f1": round(sum(f1s) / n, 4),
        "detector_count": n,
    }


def micro_average(detector_counts: dict[str, ConfusionCounts]) -> dict[str, Any]:
    """Compute micro-averaged P/R/F1 (sum all TP/FP/FN across detectors)."""
    total_tp = sum(cc.tp for cc in detector_counts.values())
    total_fp = sum(cc.fp for cc in detector_counts.values())
    total_fn = sum(cc.fn for cc in detector_counts.values())
    total_tn = sum(cc.tn for cc in detector_counts.values())

    precision_den = total_tp + total_fp
    recall_den = total_tp + total_fn
    precision = float(total_tp / precision_den) if precision_den > 0 else 0.0
    recall = float(total_tp / recall_den) if recall_den > 0 else 0.0
    f1 = f1_from_precision_recall(precision, recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def sample_size_warnings(
    detector_metrics: dict[str, dict[str, Any]],
    *,
    min_positives: int = 8,
) -> dict[str, dict[str, Any]]:
    """Flag detectors with insufficient positive samples for gate promotion.

    Returns:
        Dict mapping detector name to warning info.  Only detectors with
        insufficient samples are included.
    """
    warnings: dict[str, dict[str, Any]] = {}
    for name, m in detector_metrics.items():
        positive_count = int(m.get("positive_count", m.get("tp", 0) + m.get("fn", 0)))
        if positive_count < min_positives:
            warnings[name] = {
                "positive_count": positive_count,
                "min_required": min_positives,
                "message": (
                    f"{name}: {positive_count} positives < {min_positives} minimum — "
                    f"insufficient for gate promotion"
                ),
            }
    return warnings


def stratified_report(
    detector_metrics: dict[str, dict[str, Any]],
    detector_counts: dict[str, ConfusionCounts],
    labels: Labels,
    *,
    min_positives: int = 8,
) -> dict[str, Any]:
    """Build complete stratified evaluation report.

    Combines per-class metrics, macro/micro averages, label distribution,
    and sample-size warnings into a single report dict.
    """
    return {
        "per_class": {
            name: detector_metrics[name]
            for name in DETECTOR_NAMES
            if name in detector_metrics
        },
        "macro_average": macro_average(detector_metrics),
        "micro_average": micro_average(detector_counts),
        "label_distribution": label_distribution(labels),
        "sample_warnings": sample_size_warnings(
            detector_metrics, min_positives=min_positives
        ),
    }


__all__ = [
    "DETECTOR_NAMES",
    "label_distribution",
    "macro_average",
    "micro_average",
    "sample_size_warnings",
    "stratified_report",
]
