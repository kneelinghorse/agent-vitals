"""Optional Hopfield early-detection layer.

This module is the third layer of the agent-vitals detection stack
(see ``specs/`` and CMOS reference memory
``reference_hopfield_inference_contract.md``). It loads ONNX exports of
the bench Hopfield prefix-models and provides per-detector predictions
from short trace prefixes — its purpose is to fire **earlier** than the
handcrafted screen + TDA adjudicator, on traces that are still only 3
to ~20 steps long.

Two prefix variants ship for every detector:

- ``p3`` — trained at prefix length 3, used when ``len(snapshots) <= 4``
- ``p5`` — trained at prefix length 5, used when ``len(snapshots) >= 5``

The runtime selector is intentionally trivial: bench's empirical A/B
showed p5-eval-at-cutoff-3 collapses to macro-F1 0.556 vs p3-trained
macro-F1 0.901 (classic out-of-distribution failure on shorter inputs),
so we ship and route through both. p7 is intentionally not bundled —
see CMOS memory ``project_hopfield_p7_retrain_backlog.md`` for the
deferral rationale.

Runtime dependencies (``onnxruntime``, ``numpy``) are **optional**: the
module imports lazily and reports availability via
``is_hopfield_available()``. The base ``agent-vitals`` install pulls in
neither, so the package keeps the lightweight footprint required by the
direct-integration positioning. Install ``agent-vitals[hopfield]`` to
enable this layer.

This mission (AV-S09-M01) only ships the adapter scaffolding. Wiring it
into ``_resolve_detections`` as a 3rd-layer override is AV-S09-M02.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import VitalsSnapshot

# Default location of the bundled Hopfield ONNX artifacts and JSON sidecars.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hopfield"

# Detectors with bundled prefix-model artifacts. Order matches the bench
# DETECTORS tuple in prototypes/hopfield_detector.py.
HOPFIELD_DETECTORS: tuple[str, ...] = (
    "loop",
    "stuck",
    "confabulation",
    "thrash",
    "runaway_cost",
)

# Feature ordering MUST match the bench training pipeline. See
# ``reference_hopfield_inference_contract.md`` for the canonical spec —
# reordering breaks inference because the .onnx and the JSON sidecar
# both assume this exact 17-feature layout.
_SIGNAL_KEYS: tuple[str, ...] = (
    "findings_count",
    "sources_count",
    "objectives_covered",
    "coverage_score",
    "total_tokens",
    "error_count",
    "confidence_score",
    "convergence_delta",
    "prompt_tokens",
    "completion_tokens",
    "refinement_count",
)
_METRIC_KEYS: tuple[str, ...] = (
    "dm_coverage",
    "dm_findings",
    "cv_coverage",
    "cv_findings_rate",
    "qpf_tokens",
    "cs_effort",
)
N_FEATURES: int = len(_SIGNAL_KEYS) + len(_METRIC_KEYS)


class MissingHopfieldDependencyError(RuntimeError):
    """Raised when optional Hopfield dependencies are not installed."""


@dataclass(frozen=True)
class HopfieldConfig:
    """Configuration for the Hopfield early-detection layer.

    The defaults match the bundled artifacts. Changing ``decision_threshold``
    rebalances precision/recall on the bench corpus; see the per-prefix F1
    grid in ``reference_hopfield_inference_contract.md``.
    """

    decision_threshold: float = 0.5
    # Selector cutoffs: traces with length <= p3_max_len use p3,
    # <= p5_max_len use p5, longer traces use p7.
    p3_max_len: int = 4
    p5_max_len: int = 6
    model_dir: Path = field(default_factory=lambda: _DEFAULT_MODEL_DIR)


@dataclass(frozen=True)
class HopfieldPrediction:
    """Result of a single Hopfield prefix-model invocation."""

    detector: str
    detected: bool
    probability: float
    prefix_variant: str  # "p3" or "p5"


def is_hopfield_available() -> bool:
    """Return ``True`` when the optional Hopfield backend can be imported.

    The check is cached so the cost is paid only once per process. A
    successful return implies that :func:`predict` can be called
    without raising :class:`MissingHopfieldDependencyError`.
    """

    try:
        _load_hopfield_backend()
    except MissingHopfieldDependencyError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_hopfield_backend() -> Mapping[str, Any]:
    """Lazily import the Hopfield runtime backend.

    Optional dependencies (``numpy``, ``onnxruntime``) are imported
    inside this function so the base ``agent_vitals`` install never pays
    a cost for them and stays importable when they are not present.
    """

    try:
        import numpy as np  # type: ignore[import-not-found,unused-ignore]
        import onnxruntime as ort  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as exc:
        raise MissingHopfieldDependencyError(
            "Hopfield dependencies are missing. Install `agent-vitals[hopfield]` "
            "to enable the Hopfield early-detection layer."
        ) from exc

    return {"np": np, "ort": ort}


def _snapshot_row(snapshot: VitalsSnapshot) -> list[float]:
    """Extract the 17-dim feature row used by the bench Hopfield models."""

    signals = snapshot.signals.model_dump()
    metrics = snapshot.metrics.model_dump()
    row = [float(signals.get(key, 0.0) or 0.0) for key in _SIGNAL_KEYS]
    row.extend(float(metrics.get(key, 0.0) or 0.0) for key in _METRIC_KEYS)
    return row


def select_prefix_variant(
    snapshots: Sequence[VitalsSnapshot],
    *,
    p3_max_len: int = 4,
    p5_max_len: int = 6,
) -> str:
    """Pick the Hopfield prefix variant for a trace of length ``len(snapshots)``.

    Returns ``"p3"`` for short traces (≤ *p3_max_len*), ``"p5"`` for medium
    (≤ *p5_max_len*), and ``"p7"`` for longer traces.  Boundaries are fixed
    by bench's empirical A/B — see the prefix-selection table in
    ``reference_hopfield_inference_contract.md``.
    """

    n = len(snapshots)
    if n <= p3_max_len:
        return "p3"
    if n <= p5_max_len:
        return "p5"
    return "p7"


@dataclass(frozen=True)
class _LoadedArtifact:
    session: Any  # onnxruntime.InferenceSession
    mean: Any  # np.ndarray, shape (n_features,)
    std: Any  # np.ndarray, shape (n_features,)
    max_steps: int
    prefix_len: int
    feature_order: tuple[str, ...]


@lru_cache(maxsize=32)
def _load_artifact(
    detector: str,
    prefix_variant: str,
    model_dir: Path,
) -> _LoadedArtifact:
    if detector not in HOPFIELD_DETECTORS:
        raise ValueError(f"unknown Hopfield detector: {detector!r}")
    if prefix_variant not in ("p3", "p5", "p7"):
        raise ValueError(f"unknown Hopfield prefix variant: {prefix_variant!r}")

    backend = _load_hopfield_backend()
    np = backend["np"]
    ort = backend["ort"]

    onnx_path = model_dir / f"{detector}_{prefix_variant}.onnx"
    sidecar_path = model_dir / f"{detector}_{prefix_variant}.json"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Hopfield ONNX artifact not found: {onnx_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Hopfield sidecar not found: {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    feature_order = tuple(sidecar["feature_order"])
    expected_order = tuple(_SIGNAL_KEYS) + tuple(_METRIC_KEYS)
    if feature_order != expected_order:
        raise RuntimeError(
            f"Hopfield sidecar feature order mismatch for {detector}_{prefix_variant}: "
            f"sidecar={feature_order} expected={expected_order}"
        )

    mean = np.array(sidecar["mean"], dtype=np.float32)
    std = np.array(sidecar["std"], dtype=np.float32)
    if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
        raise RuntimeError(
            f"Hopfield sidecar shape mismatch for {detector}_{prefix_variant}: "
            f"mean={mean.shape} std={std.shape}"
        )

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )

    return _LoadedArtifact(
        session=session,
        mean=mean,
        std=std,
        max_steps=int(sidecar["max_steps"]),
        prefix_len=int(sidecar["prefix_len"]),
        feature_order=feature_order,
    )


def _build_input_tensor(
    snapshots: Sequence[VitalsSnapshot],
    artifact: _LoadedArtifact,
) -> tuple[Any, int]:
    """Build the ``(x, length)`` pair fed to the ONNX session.

    Mirrors ``_trace_to_tensor`` from the bench prototype: build the
    feature matrix in canonical order, normalize per-feature with the
    bundled mean/std, then zero-pad to ``max_steps`` rows. The effective
    length is ``min(actual_trace_len, prefix_len)``.
    """

    backend = _load_hopfield_backend()
    np = backend["np"]

    rows = [_snapshot_row(s) for s in snapshots]
    matrix = np.array(rows, dtype=np.float32) if rows else np.zeros((0, N_FEATURES), dtype=np.float32)
    if matrix.shape[1] != N_FEATURES:
        raise RuntimeError(
            f"feature matrix has {matrix.shape[1]} cols, expected {N_FEATURES}"
        )

    actual_len = matrix.shape[0]
    use_len = min(artifact.prefix_len, actual_len) if actual_len > 0 else 0

    padded = np.zeros((artifact.max_steps, N_FEATURES), dtype=np.float32)
    if use_len > 0:
        padded[:use_len] = (matrix[:use_len] - artifact.mean) / artifact.std

    return padded[np.newaxis, :, :], use_len


def predict(
    snapshots: Sequence[VitalsSnapshot],
    detector: str,
    *,
    config: HopfieldConfig | None = None,
) -> HopfieldPrediction | None:
    """Run the Hopfield prefix-model for ``detector`` on ``snapshots``.

    Returns ``None`` when the trace is empty (no input to score). Raises
    :class:`MissingHopfieldDependencyError` if the optional backend is
    missing or :class:`FileNotFoundError` if a bundled artifact is
    absent.
    """

    if not snapshots:
        return None

    cfg = config or HopfieldConfig()
    backend = _load_hopfield_backend()
    np = backend["np"]

    prefix_variant = select_prefix_variant(
        snapshots, p3_max_len=cfg.p3_max_len, p5_max_len=cfg.p5_max_len
    )
    artifact = _load_artifact(detector, prefix_variant, cfg.model_dir)

    x, length = _build_input_tensor(snapshots, artifact)
    if length == 0:
        return None

    lengths = np.array([length], dtype=np.int64)
    outputs = artifact.session.run(None, {"x": x, "lengths": lengths})
    logit = float(outputs[0].reshape(-1)[0])
    probability = 1.0 / (1.0 + float(np.exp(-logit)))
    return HopfieldPrediction(
        detector=detector,
        detected=probability >= cfg.decision_threshold,
        probability=probability,
        prefix_variant=prefix_variant,
    )


# Per-detector override thresholds for hopfield_override_fires (AV-S09-M02).
# Calibrated against the bench v1 corpus (1494 traces) per AV-S09-M03
# acceptance (intel response 0f771492). The bench acceptance message
# carries the full PR curves at cutoffs 3 and 5 for thresholds
# {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}; the values below are the data-driven
# optimum that maximizes early-window recall without dropping marker
# precision below 0.91 on any detector at any cutoff.
#
# Per-detector rationale (from bench's PR curves):
#
# - loop @ 0.80: lowering from a uniform 0.90 buys
#   R 0.340 → 0.892 at cutoff=3 (P 1.000 → 0.927). Cutoff=5 unchanged
#   (P=1.000, R=0.972). Net: meaningful early-detection recall gain.
# - stuck @ 0.90: precision cliff between 0.70 and 0.80 — lowering buys
#   nothing usable. Accept the 0.261 recall at cutoff=3 and rely on the
#   cutoff=5 path (R=0.804, P=0.974).
# - confabulation @ 0.80: at cutoff=3 buys R 0.338 → 0.555 for
#   P 0.972 → 0.914. At cutoff=5: P 1.000 → 0.991, R 0.708 → 0.747.
# - thrash @ 0.70: signal is exceptionally clean — at every threshold
#   ≥ 0.70, precision is 1.000 and recall is 0.793 at cutoff=3 / 1.000
#   at cutoff=5. The 0.70 floor is interpretable as "any non-trivial
#   confidence" without buying more headroom than we already have.
# - runaway_cost @ 0.90: same precision-cliff dynamic as stuck — usable
#   only at 0.90 (P 1.000, R 0.236 at cutoff=3) or below 0.70
#   (P drops to 0.718). Cutoff=5 is perfect at every threshold.
#
# Tunable via the ``thresholds`` kwarg on hopfield_override_fires() if
# downstream consumers want to trade off precision for early-window
# recall in a specific deployment.
DEFAULT_OVERRIDE_THRESHOLDS: Mapping[str, float] = {
    "loop": 0.80,
    "stuck": 0.90,
    "confabulation": 0.80,
    "thrash": 0.70,
    "runaway_cost": 0.90,
}

# Early-detection window. Hopfield is consulted only when the current trace
# length falls inside [_EARLY_WINDOW_MIN, _EARLY_WINDOW_MAX] inclusive. At
# length <3 the input is too short for either prefix model. At length >=7
# the existing handcrafted+TDA stack regains dominance per bench's
# Five-Paradigm Comparative Report (intel_alert b7416ceb).
_EARLY_WINDOW_MIN: int = 3
_EARLY_WINDOW_MAX: int = 6


def hopfield_override_fires(
    snapshots: Sequence[VitalsSnapshot],
    *,
    config: HopfieldConfig | None = None,
    model_dir: Path | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    """Return ``True`` when the Hopfield override layer should fire.

    The check evaluates every bundled detector against the trace prefix
    and returns ``True`` as soon as any per-detector probability crosses
    its entry in ``thresholds`` (defaulting to
    :data:`DEFAULT_OVERRIDE_THRESHOLDS`). The result is intended to be
    written to ``LoopDetectionResult.hopfield_override_active`` and
    propagated to ``VitalsSnapshot.hopfield_override_active`` so
    early-warning consumers can read it as provenance for early
    intervention.

    Returns ``False`` (gracefully, never raising) when:

    - ``snapshots`` is empty or its length is outside
      ``[_EARLY_WINDOW_MIN, _EARLY_WINDOW_MAX]``
    - the optional Hopfield backend (``onnxruntime``) is not installed
    - the bundled artifacts cannot be located on disk

    The graceful degradation contract matches the TDA pattern from
    ``agent_vitals/detection/tda.py``: callers should be able to enable
    the Hopfield layer in a base install without crashing.
    """

    if not (_EARLY_WINDOW_MIN <= len(snapshots) <= _EARLY_WINDOW_MAX):
        return False

    if not is_hopfield_available():
        return False

    base_config = config or HopfieldConfig()
    resolved_dir = model_dir or base_config.model_dir
    resolved_thresholds: dict[str, float] = dict(DEFAULT_OVERRIDE_THRESHOLDS)
    if thresholds:
        resolved_thresholds.update(thresholds)

    eval_config = HopfieldConfig(
        decision_threshold=base_config.decision_threshold,
        p3_max_len=base_config.p3_max_len,
        p5_max_len=base_config.p5_max_len,
        model_dir=resolved_dir,
    )

    for detector in HOPFIELD_DETECTORS:
        threshold = resolved_thresholds.get(detector, 1.0)
        try:
            prediction = predict(snapshots, detector, config=eval_config)
        except FileNotFoundError:
            return False
        except MissingHopfieldDependencyError:
            return False
        if prediction is None:
            continue
        if prediction.probability >= threshold:
            return True

    return False


class HopfieldEarlyDetector:
    """Convenience facade that caches loaded artifacts across calls.

    Wraps :func:`predict` so callers that score many traces in a row only
    pay the artifact-load cost once per ``(detector, prefix_variant)``.
    Equivalent in behavior to calling :func:`predict` directly because
    the underlying loader is also memoized via ``lru_cache``.
    """

    def __init__(self, *, config: HopfieldConfig | None = None) -> None:
        self._config = config or HopfieldConfig()

    @property
    def config(self) -> HopfieldConfig:
        return self._config

    def predict_one(
        self,
        snapshots: Sequence[VitalsSnapshot],
        detector: str,
    ) -> HopfieldPrediction | None:
        return predict(snapshots, detector, config=self._config)
