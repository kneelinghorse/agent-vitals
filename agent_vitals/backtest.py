"""Backtest harness for Agent Vitals threshold validation.

Runs all evaluation trajectories through the detection engine and reports
precision, recall, and F1 per detector and for the ``vitals.any`` composite.

Evaluation is **per-trace**: for each trace, the harness checks whether
the detector fired at any point during replay, and compares that against
whether the trace is labeled as having that failure mode (non-empty onset
set = positive label).

The harness is designed to be:
- Callable programmatically via ``run_backtest()``
- Callable via pytest
- Reproducible: deterministic given the same traces + labels + config
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .config import VitalsConfig
from .detection.loop import detect_loop
from .schema import VitalsSnapshot
from .detection.stop_rule import derive_stop_signals


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConfusionCounts:
    """Confusion matrix counts for a binary classifier."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def record(self, *, predicted: bool, expected: bool) -> None:
        if predicted and expected:
            self.tp += 1
        elif predicted and not expected:
            self.fp += 1
        elif (not predicted) and expected:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return float(2.0 * self.precision * self.recall / denom) if denom > 0 else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Per-detector backtest result."""

    name: str
    confusion: ConfusionCounts

    @property
    def precision(self) -> float:
        return self.confusion.precision

    @property
    def recall(self) -> float:
        return self.confusion.recall

    @property
    def f1(self) -> float:
        return self.confusion.f1

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.confusion.as_dict()}


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Complete backtest report across all detectors."""

    dataset_trace_count: int
    dataset_snapshot_count: int
    dataset_invalid_lines: dict[str, int]
    config_summary: dict[str, Any]
    detectors: dict[str, DetectorResult]
    composite_any: DetectorResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "trace_count": self.dataset_trace_count,
                "snapshot_count": self.dataset_snapshot_count,
                "invalid_lines": dict(self.dataset_invalid_lines),
            },
            "config": self.config_summary,
            "detectors": {
                name: result.as_dict() for name, result in self.detectors.items()
            },
            "composite_any": self.composite_any.as_dict(),
        }


Labels = dict[str, dict[str, set[int]]]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> Labels:
    """Load ground-truth labels from JSON.

    Supports three label formats:

    1. **Onset format** (original)::

        {"MISSION_ID": {"loop_at": [3, 5], "stuck_at": [7], ...}}

    2. **Cross-agent format** (label string)::

        {"MISSION_ID": {"label": "loop", "scenario": "loop", ...}}

    3. **Corpus format** (labels list)::

        {"MISSION_ID": {"labels": ["thrash", "stuck"], ...}}

    For formats 2 and 3, a synthetic onset set ``{0}`` is created for
    each detected failure mode so the per-trace evaluation works correctly.

    Args:
        path: Path to a JSON labels file.

    Returns:
        Mapping from mission_id to per-detector onset sets.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("labels JSON must be an object mapping mission_id -> labels")

    labels: Labels = {}
    for mission_id, entry in payload.items():
        if not isinstance(mission_id, str) or not mission_id.strip():
            continue
        if not isinstance(entry, Mapping):
            entry = {}

        # Format 1: onset sets present
        if any(key in entry for key in ("loop_at", "stuck_at", "thrash_at", "runaway_cost_at")):
            labels[mission_id.strip()] = {
                "loop_at": _coerce_int_set(entry.get("loop_at")),
                "stuck_at": _coerce_int_set(entry.get("stuck_at")),
                "thrash_at": _coerce_int_set(entry.get("thrash_at")),
                "runaway_cost_at": _coerce_int_set(entry.get("runaway_cost_at")),
            }
            continue

        # Format 2 & 3: derive onset sets from label(s)
        active_modes: set[str] = set()

        # Format 3: "labels" list
        label_list = entry.get("labels")
        if isinstance(label_list, list):
            for item in label_list:
                active_modes.add(str(item).strip().lower())

        # Format 2: "label" string (may also have "scenario")
        label_str = entry.get("label") or entry.get("scenario")
        if isinstance(label_str, str):
            active_modes.add(label_str.strip().lower())

        # Map mode names to detector keys (synthetic onset at step 0)
        result: dict[str, set[int]] = {
            "loop_at": set(),
            "stuck_at": set(),
            "thrash_at": set(),
            "runaway_cost_at": set(),
        }
        mode_to_key = {
            "loop": "loop_at",
            "stuck": "stuck_at",
            "thrash": "thrash_at",
            "runaway_cost": "runaway_cost_at",
            "runaway": "runaway_cost_at",
            "burn_rate": "runaway_cost_at",
        }
        for mode in active_modes:
            key = mode_to_key.get(mode)
            if key:
                result[key] = {0}

        labels[mission_id.strip()] = result
    return labels


@dataclass(frozen=True, slots=True)
class Dataset:
    """Loaded vitals traces for backtesting."""

    traces: dict[str, list[VitalsSnapshot]]
    invalid_lines: dict[str, int]

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    @property
    def snapshot_count(self) -> int:
        return sum(len(items) for items in self.traces.values())


def load_dataset(traces_dir: Path) -> Dataset:
    """Load ``*.jsonl`` vitals traces from a directory (non-recursive).

    Args:
        traces_dir: Directory containing one ``.jsonl`` file per mission.

    Returns:
        Dataset with parsed VitalsSnapshot sequences.

    Raises:
        FileNotFoundError: If traces_dir does not exist.
        NotADirectoryError: If traces_dir is not a directory.
        ValueError: If no valid traces are found.
    """

    if not traces_dir.exists():
        raise FileNotFoundError(traces_dir)
    if not traces_dir.is_dir():
        raise NotADirectoryError(traces_dir)

    traces: dict[str, list[VitalsSnapshot]] = {}
    invalid_lines: dict[str, int] = {}
    for path in sorted(traces_dir.glob("*.jsonl")):
        mission_id = path.stem
        items: list[VitalsSnapshot] = []
        bad = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(VitalsSnapshot.model_validate_json(stripped))
            except Exception:
                bad += 1
                continue

        if not items:
            continue
        traces[mission_id] = items
        if bad:
            invalid_lines[mission_id] = bad

    if not traces:
        raise ValueError(f"No valid *.jsonl traces found in {traces_dir}")

    return Dataset(traces=traces, invalid_lines=invalid_lines)


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _replay_trace(
    snapshots: Sequence[VitalsSnapshot],
    *,
    config: VitalsConfig,
    workflow_type: str,
) -> dict[str, bool]:
    """Replay a trace and return per-detector fired flags.

    Returns:
        Dict with keys ``loop``, ``stuck``, ``thrash``, ``runaway_cost``,
        ``any`` — each True if the detector fired at any snapshot.
    """

    loop_fired = False
    stuck_fired = False
    thrash_fired = False
    runaway_fired = False

    history: list[VitalsSnapshot] = []
    for snapshot in snapshots:
        detection = detect_loop(
            snapshot,
            history,
            config=config,
            workflow_type=workflow_type,
        )

        stop_signals = derive_stop_signals(
            {
                "loop_detected": detection.loop_detected,
                "stuck_detected": detection.stuck_detected,
                "stuck_trigger": detection.stuck_trigger,
                "error_count": int(snapshot.signals.error_count),
            },
        )

        if detection.loop_detected:
            loop_fired = True
        if detection.stuck_detected:
            stuck_fired = True
        if stop_signals.thrash_detected:
            thrash_fired = True
        if stop_signals.runaway_cost_detected:
            runaway_fired = True

        history.append(snapshot)

    any_fired = loop_fired or stuck_fired or thrash_fired or runaway_fired
    return {
        "loop": loop_fired,
        "stuck": stuck_fired,
        "thrash": thrash_fired,
        "runaway_cost": runaway_fired,
        "any": any_fired,
    }


def run_backtest(
    dataset: Dataset,
    labels: Labels,
    *,
    config: Optional[VitalsConfig] = None,
    workflow_type: str = "unknown",
) -> BacktestReport:
    """Run the backtest harness over a dataset with ground-truth labels.

    Evaluation is **per-trace**: for each trace, replays all snapshots
    through ``detect_loop()`` and ``derive_stop_signals()``, then
    checks whether the detector fired at any point. A trace is a positive
    example for a detector if its label set for that detector is non-empty.

    Args:
        dataset: Loaded vitals traces.
        labels: Ground-truth labels mapping mission_id -> detector onsets.
        config: VitalsConfig to use. Defaults to ``VitalsConfig.from_yaml()``.
        workflow_type: Workflow type hint for detection.

    Returns:
        BacktestReport with per-detector and composite metrics.
    """

    cfg = config or VitalsConfig.from_yaml(allow_env_override=False)

    loop_counts = ConfusionCounts()
    stuck_counts = ConfusionCounts()
    thrash_counts = ConfusionCounts()
    runaway_counts = ConfusionCounts()
    any_counts = ConfusionCounts()

    for mission_id, snapshots in dataset.traces.items():
        expected = labels.get(mission_id) or {
            "loop_at": set(),
            "stuck_at": set(),
            "thrash_at": set(),
            "runaway_cost_at": set(),
        }
        loop_expected = bool(expected.get("loop_at"))
        stuck_expected = bool(expected.get("stuck_at"))
        thrash_expected = bool(expected.get("thrash_at"))
        runaway_expected = bool(expected.get("runaway_cost_at"))
        any_expected = loop_expected or stuck_expected or thrash_expected or runaway_expected

        fired = _replay_trace(snapshots, config=cfg, workflow_type=workflow_type)

        loop_counts.record(predicted=fired["loop"], expected=loop_expected)
        stuck_counts.record(predicted=fired["stuck"], expected=stuck_expected)
        thrash_counts.record(predicted=fired["thrash"], expected=thrash_expected)
        runaway_counts.record(predicted=fired["runaway_cost"], expected=runaway_expected)
        any_counts.record(predicted=fired["any"], expected=any_expected)

    detectors = {
        "loop": DetectorResult(name="loop", confusion=loop_counts),
        "stuck": DetectorResult(name="stuck", confusion=stuck_counts),
        "thrash": DetectorResult(name="thrash", confusion=thrash_counts),
        "runaway_cost": DetectorResult(name="runaway_cost", confusion=runaway_counts),
    }
    composite = DetectorResult(name="vitals.any", confusion=any_counts)

    config_summary = {
        "loop_consecutive_count": cfg.loop_consecutive_count,
        "loop_similarity_threshold": cfg.loop_similarity_threshold,
        "stuck_dm_threshold": cfg.stuck_dm_threshold,
        "stuck_cv_threshold": cfg.stuck_cv_threshold,
        "burn_rate_multiplier": cfg.burn_rate_multiplier,
        "workflow_stuck_enabled": cfg.workflow_stuck_enabled,
    }

    return BacktestReport(
        dataset_trace_count=dataset.trace_count,
        dataset_snapshot_count=dataset.snapshot_count,
        dataset_invalid_lines=dict(dataset.invalid_lines),
        config_summary=config_summary,
        detectors=detectors,
        composite_any=composite,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_int_set(value: Any) -> set[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set()
    output: set[int] = set()
    for item in value:
        try:
            output.add(int(item))
        except (TypeError, ValueError):
            continue
    return output


__all__ = [
    "BacktestReport",
    "ConfusionCounts",
    "Dataset",
    "DetectorResult",
    "Labels",
    "load_dataset",
    "load_labels",
    "run_backtest",
]
