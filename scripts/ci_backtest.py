#!/usr/bin/env python3
"""Run lightweight backtest gates for CI.

This script replays the bundled AV corpora and enforces a blocking gate on
composite ``vitals.any`` precision/recall. Per-detector metrics are reported
as informational CI annotations and serialized to JSON for artifact upload.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_vitals.backtest import ConfusionCounts, Dataset, _replay_trace, load_dataset
from agent_vitals.ci_gate import evaluate_hard_gate, evaluate_promotion, metrics_with_ci
from agent_vitals.config import VitalsConfig

ROOT = Path(__file__).resolve().parents[1]

Labels = dict[str, dict[str, set[int]]]


def _empty_onsets() -> dict[str, set[int]]:
    return {
        "loop_at": set(),
        "confabulation_at": set(),
        "stuck_at": set(),
        "thrash_at": set(),
        "runaway_cost_at": set(),
    }


def _convert_synth_labels(raw_labels: dict[str, Any], trace_keys: set[str]) -> Labels:
    """Convert av05_synth label schema to backtest onset labels."""
    labels: Labels = {}
    by_mission: dict[str, list[dict[str, Any]]] = {}

    for key, entry_any in raw_labels.items():
        entry = entry_any if isinstance(entry_any, dict) else {}
        mission_id = str(entry.get("mission_id", key.rsplit(":", 1)[0]))
        by_mission.setdefault(mission_id, []).append(entry)

    for mission_id, entries in by_mission.items():
        if mission_id not in trace_keys:
            continue

        onset = _empty_onsets()
        for entry in entries:
            labels_list = entry.get("labels")
            if not isinstance(labels_list, list):
                continue
            active = {str(item).strip().lower() for item in labels_list}
            for label_name, detector_key in [
                ("loop", "loop_at"),
                ("confabulation", "confabulation_at"),
                ("stuck", "stuck_at"),
                ("thrash", "thrash_at"),
                ("runaway_cost", "runaway_cost_at"),
            ]:
                if label_name not in active:
                    continue
                onset_val = entry.get(f"{label_name}_from_loop")
                if onset_val is not None:
                    onset[detector_key].add(int(onset_val))
                else:
                    onset[detector_key].add(0)

        labels[mission_id] = onset

    return labels


def _convert_real_labels(raw_labels: dict[str, Any], trace_keys: set[str]) -> Labels:
    """Convert av26_real label schema to backtest onset labels."""
    labels: Labels = {}

    for key, entry_any in raw_labels.items():
        entry = entry_any if isinstance(entry_any, dict) else {}
        mission_id = str(entry.get("mission_id", key))
        if mission_id not in trace_keys:
            continue

        onset = _empty_onsets()
        labels_list = entry.get("labels")
        active = {str(item).strip().lower() for item in labels_list} if isinstance(labels_list, list) else set()

        for label_name, detector_key in [
            ("loop", "loop_at"),
            ("confabulation", "confabulation_at"),
            ("stuck", "stuck_at"),
            ("thrash", "thrash_at"),
            ("runaway_cost", "runaway_cost_at"),
        ]:
            if label_name not in active:
                continue
            onset_val = entry.get(f"{label_name}_from_loop")
            if onset_val is not None:
                onset[detector_key].add(int(onset_val))
            else:
                onset[detector_key].add(0)

        labels[mission_id] = onset

    return labels


def _init_counts() -> dict[str, ConfusionCounts]:
    return {
        "loop": ConfusionCounts(),
        "confabulation": ConfusionCounts(),
        "stuck": ConfusionCounts(),
        "thrash": ConfusionCounts(),
        "runaway_cost": ConfusionCounts(),
        "vitals.any": ConfusionCounts(),
    }


def _evaluate(
    dataset: Dataset,
    labels: Labels,
    *,
    workflow_type: str,
    config: VitalsConfig,
    counts: dict[str, ConfusionCounts],
) -> None:
    for trace_id, snapshots in dataset.traces.items():
        expected = labels.get(trace_id) or _empty_onsets()

        loop_expected = bool(expected.get("loop_at"))
        confab_expected = bool(expected.get("confabulation_at"))
        stuck_expected = bool(expected.get("stuck_at"))
        thrash_expected = bool(expected.get("thrash_at"))
        runaway_expected = bool(expected.get("runaway_cost_at"))
        any_expected = loop_expected or confab_expected or stuck_expected or thrash_expected or runaway_expected

        fired = _replay_trace(snapshots, config=config, workflow_type=workflow_type)

        counts["loop"].record(predicted=fired["loop"], expected=loop_expected)
        counts["confabulation"].record(predicted=fired["confabulation"], expected=confab_expected)
        counts["stuck"].record(predicted=fired["stuck"], expected=stuck_expected)
        counts["thrash"].record(predicted=fired["thrash"], expected=thrash_expected)
        counts["runaway_cost"].record(predicted=fired["runaway_cost"], expected=runaway_expected)
        counts["vitals.any"].record(predicted=fired["any"], expected=any_expected)


def _annotate(level: str, title: str, message: str) -> None:
    """Emit GitHub Actions annotation syntax."""
    safe_message = message.replace("\n", " ")
    print(f"::{level} title={title}::{safe_message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CI backtest gate for agent-vitals")
    parser.add_argument(
        "--synth-corpus",
        type=Path,
        default=ROOT / "checkpoints" / "vitals_corpus" / "av05_synth",
        help="Path to av05_synth corpus root (contains labels.json and traces/)",
    )
    parser.add_argument(
        "--real-corpus",
        type=Path,
        default=ROOT / "checkpoints" / "vitals_corpus" / "av26_real",
        help="Path to av26_real corpus root (contains labels.json and traces/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "backtest-results.json",
        help="Output JSON results path",
    )
    parser.add_argument("--composite-min-precision", type=float, default=0.90)
    parser.add_argument("--composite-min-recall", type=float, default=0.85)
    parser.add_argument("--detector-min-positives", type=int, default=8)
    parser.add_argument("--detector-min-precision-lb", type=float, default=0.80)
    parser.add_argument("--detector-min-recall-lb", type=float, default=0.75)
    args = parser.parse_args()

    t0 = time.perf_counter()

    synth_ds = load_dataset(args.synth_corpus / "traces")
    real_ds = load_dataset(args.real_corpus / "traces")

    synth_raw = json.loads((args.synth_corpus / "labels.json").read_text(encoding="utf-8"))
    real_raw = json.loads((args.real_corpus / "labels.json").read_text(encoding="utf-8"))

    synth_labels = _convert_synth_labels(synth_raw, set(synth_ds.traces.keys()))
    real_labels = _convert_real_labels(real_raw, set(real_ds.traces.keys()))

    synth_unlabeled = sorted(set(synth_ds.traces.keys()) - set(synth_labels.keys()))
    real_unlabeled = sorted(set(real_ds.traces.keys()) - set(real_labels.keys()))

    cfg = VitalsConfig.from_yaml(allow_env_override=False)
    counts = _init_counts()

    _evaluate(synth_ds, synth_labels, workflow_type="synthetic", config=cfg, counts=counts)
    _evaluate(real_ds, real_labels, workflow_type="real", config=cfg, counts=counts)

    runtime_s = time.perf_counter() - t0

    detectors = {
        name: metrics_with_ci(cc)
        for name, cc in counts.items()
        if name != "vitals.any"
    }
    composite = metrics_with_ci(counts["vitals.any"])

    composite_gate_pass = (
        composite["precision"] >= args.composite_min_precision
        and composite["recall"] >= args.composite_min_recall
    )

    promotion_decisions: dict[str, dict[str, Any]] = {}
    for detector_name, metric in detectors.items():
        promotion_decisions[detector_name] = evaluate_promotion(
            detector_name=detector_name,
            metric=metric,
            min_positives=args.detector_min_positives,
            min_precision_lb=args.detector_min_precision_lb,
            min_recall_lb=args.detector_min_recall_lb,
        )

    hard_detectors = sorted(
        name
        for name, decision in promotion_decisions.items()
        if bool(decision["qualifies_for_hard_gate"])
    )
    soft_detectors = sorted(set(detectors.keys()) - set(hard_detectors))

    hard_gate_failures: dict[str, list[str]] = {}
    for detector_name in hard_detectors:
        passed, reasons = evaluate_hard_gate(
            metric=detectors[detector_name],
            min_precision_lb=args.detector_min_precision_lb,
            min_recall_lb=args.detector_min_recall_lb,
        )
        if not passed:
            hard_gate_failures[detector_name] = reasons

    gate_pass = composite_gate_pass and not hard_gate_failures
    if not composite_gate_pass:
        gate_reason = "composite_threshold_failed"
    elif hard_gate_failures:
        gate_reason = "hard_detector_threshold_failed"
    else:
        gate_reason = "composite_and_hard_detectors_passed"

    result_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_s,
        "dataset": {
            "synthetic_trace_count": synth_ds.trace_count,
            "real_trace_count": real_ds.trace_count,
            "trace_count": synth_ds.trace_count + real_ds.trace_count,
            "snapshot_count": synth_ds.snapshot_count + real_ds.snapshot_count,
            "unlabeled": {
                "synthetic": synth_unlabeled,
                "real": real_unlabeled,
            },
        },
        "thresholds": {
            "composite_min_precision": args.composite_min_precision,
            "composite_min_recall": args.composite_min_recall,
            "detector_min_positives": args.detector_min_positives,
            "detector_min_precision_lb": args.detector_min_precision_lb,
            "detector_min_recall_lb": args.detector_min_recall_lb,
        },
        "composite_any": composite,
        "detectors": detectors,
        "detector_gate_policy": {
            "hard_detectors": hard_detectors,
            "soft_detectors": soft_detectors,
            "decisions": promotion_decisions,
        },
        "gate": {
            "passed": gate_pass,
            "reason": gate_reason,
            "composite_passed": composite_gate_pass,
            "hard_detector_failures": hard_gate_failures,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "Backtest summary: "
        f"traces={result_payload['dataset']['trace_count']} "
        f"snapshots={result_payload['dataset']['snapshot_count']} "
        f"runtime={runtime_s:.3f}s"
    )
    print(
        "Composite vitals.any: "
        f"P={composite['precision']:.3f} R={composite['recall']:.3f} "
        f"F1={composite['f1']:.3f} TP={composite['tp']} FP={composite['fp']} "
        f"FN={composite['fn']} TN={composite['tn']}"
    )
    print(
        "Detector gate policy: "
        f"hard={len(hard_detectors)} soft={len(soft_detectors)} "
        f"criteria=(positives>={args.detector_min_positives}, "
        f"P_lb>={args.detector_min_precision_lb:.2f}, "
        f"R_lb>={args.detector_min_recall_lb:.2f})"
    )

    # Informational per-detector annotations only.
    for detector_name, metric in detectors.items():
        decision = promotion_decisions[detector_name]
        _annotate(
            "notice",
            f"backtest:{detector_name}",
            (
                f"P={metric['precision']:.3f} R={metric['recall']:.3f} F1={metric['f1']:.3f} "
                f"TP={metric['tp']} FP={metric['fp']} FN={metric['fn']} TN={metric['tn']}"
            ),
        )
        _annotate(
            "notice" if decision["qualifies_for_hard_gate"] else "warning",
            f"backtest:gate_policy:{detector_name}",
            (
                f"status={decision['status']} positives={decision['positive_count']} "
                f"P_lb={decision['precision_lb']:.3f} R_lb={decision['recall_lb']:.3f} "
                f"reasons={'; '.join(decision['reasons']) if decision['reasons'] else 'meets_promotion_criteria'}"
            ),
        )

    if synth_unlabeled or real_unlabeled:
        _annotate(
            "warning",
            "backtest:label_coverage",
            f"Unlabeled traces detected synth={len(synth_unlabeled)} real={len(real_unlabeled)}",
        )

    if runtime_s > 60.0:
        _annotate(
            "warning",
            "backtest:runtime",
            f"Backtest runtime exceeded 60s ({runtime_s:.3f}s).",
        )

    if hard_gate_failures:
        for detector_name, reasons in hard_gate_failures.items():
            _annotate(
                "error",
                f"backtest:hard_gate:{detector_name}",
                "; ".join(reasons),
            )

    if gate_pass:
        _annotate(
            "notice",
            "backtest:gate",
            (
                f"PASS CI gate policy composite(P={composite['precision']:.3f}, "
                f"R={composite['recall']:.3f}); hard_detectors={hard_detectors}"
            ),
        )
        return 0

    _annotate(
        "error",
        "backtest:gate",
        (
            f"FAIL CI gate policy composite(P={composite['precision']:.3f}, "
            f"R={composite['recall']:.3f}); hard_failures={hard_gate_failures}"
        ),
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
