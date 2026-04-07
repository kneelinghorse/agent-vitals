"""Optional TDA-based adjudication layer for runaway cost detection.

This module is the second layer of the hybrid runaway cost detector
(see ``specs/hybrid-runaway-cost-detector.md``). The handcrafted
burn-rate detector in ``loop.py`` runs first and screens every trace
with perfect recall on the v1 corpus. When it fires, this module is
called to confirm or override the verdict using a topological data
analysis pipeline trained on the bench corpus.

TDA dependencies (giotto-tda, scikit-learn, joblib, numpy) are
**optional**. The module imports lazily and reports availability via
``is_tda_available()``; callers must check this flag before invoking
``predict_runaway_cost``. When unavailable the handcrafted verdict
stands unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from ..schema import VitalsSnapshot

if TYPE_CHECKING:  # pragma: no cover
    pass


# Default location of the bundled runaway cost classifier.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "runaway_cost.joblib"

# Signal/metric field order — must match the order used to train the
# bench classifier in ``prototypes/tda_detector.py``. Reordering breaks
# inference because the trained pipeline expects features in the same
# layout it was fit on.
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


class MissingTDADependencyError(RuntimeError):
    """Raised when optional TDA dependencies are not installed."""


@dataclass(frozen=True)
class TDAConfig:
    """Configuration for the TDA adjudication layer.

    Defaults match the bench training configuration; changing them
    requires retraining the classifier.
    """

    window_sizes: tuple[int, ...] = (3, 4, 5)
    min_steps: int = 5
    homology_dimensions: tuple[int, ...] = (0, 1)
    decision_threshold: float = 0.5
    model_path: Path = field(default_factory=lambda: _DEFAULT_MODEL_PATH)


@dataclass(frozen=True)
class TDARunawayPrediction:
    """Result of the TDA adjudication layer."""

    detected: bool
    probability: float


def is_tda_available() -> bool:
    """Return ``True`` when the optional TDA backend can be imported.

    The check is cached so the cost is paid only once per process. A
    successful return implies that ``predict_runaway_cost`` can be
    called without raising ``MissingTDADependencyError``.
    """

    try:
        _load_tda_backend()
    except MissingTDADependencyError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_tda_backend() -> Mapping[str, Any]:
    """Lazily import the TDA backend.

    All optional dependencies are imported inside this function so the
    base ``agent_vitals`` install never pays a cost for them and stays
    importable when they are not present.
    """

    try:
        import joblib  # type: ignore[import-not-found,unused-ignore]
        import numpy as np  # type: ignore[import-not-found,unused-ignore]
        from gtda.diagrams import (  # type: ignore[import-not-found,unused-ignore]
            Amplitude,
            BettiCurve,
            NumberOfPoints,
            PersistenceEntropy,
            PersistenceLandscape,
            Silhouette,
        )
        from gtda.homology import (  # type: ignore[import-not-found,unused-ignore]
            VietorisRipsPersistence,
        )
    except ImportError as exc:
        raise MissingTDADependencyError(
            "TDA dependencies are missing. Install `agent-vitals[tda]` to enable "
            "the TDA adjudication layer (requires Python <= 3.12)."
        ) from exc

    return {
        "joblib": joblib,
        "np": np,
        "Amplitude": Amplitude,
        "BettiCurve": BettiCurve,
        "NumberOfPoints": NumberOfPoints,
        "PersistenceEntropy": PersistenceEntropy,
        "PersistenceLandscape": PersistenceLandscape,
        "Silhouette": Silhouette,
        "VietorisRipsPersistence": VietorisRipsPersistence,
    }


def _snapshot_row(snapshot: VitalsSnapshot) -> list[float]:
    """Extract the 17-dim feature row used by the bench TDA pipeline."""

    signals = snapshot.signals.model_dump()
    metrics = snapshot.metrics.model_dump()
    row = [float(signals.get(key, 0.0) or 0.0) for key in _SIGNAL_KEYS]
    row.extend(float(metrics.get(key, 0.0) or 0.0) for key in _METRIC_KEYS)
    return row


def _trace_matrix(
    snapshots: Sequence[VitalsSnapshot],
    config: TDAConfig,
    backend: Mapping[str, Any],
) -> Any | None:
    if len(snapshots) < config.min_steps:
        return None
    np = backend["np"]
    return np.array([_snapshot_row(s) for s in snapshots], dtype=np.float64)


def _normalize_trace(matrix: Any, np_module: Any) -> Any:
    mins = matrix.min(axis=0)
    ranges = matrix.max(axis=0) - mins
    ranges[ranges == 0] = 1.0
    return (matrix - mins) / ranges


def _build_point_cloud(trace_norm: Any, window_size: int, np_module: Any) -> Any:
    n_steps = int(trace_norm.shape[0])
    width = min(window_size, n_steps - 1)
    if width < 2:
        width = 2
    points = [trace_norm[idx : idx + width].flatten() for idx in range(n_steps - width + 1)]
    return np_module.array(points)


def _diagram_features(diagrams: Any, backend: Mapping[str, Any]) -> dict[str, Any]:
    np = backend["np"]
    features: dict[str, Any] = {}
    features["entropy"] = backend["PersistenceEntropy"]().fit_transform(diagrams).flatten()
    features["amp_wasserstein"] = (
        backend["Amplitude"](metric="wasserstein", order=2).fit_transform(diagrams).flatten()
    )
    features["amp_landscape"] = (
        backend["Amplitude"](metric="landscape").fit_transform(diagrams).flatten()
    )
    features["amp_bottleneck"] = (
        backend["Amplitude"](metric="bottleneck").fit_transform(diagrams).flatten()
    )
    features["n_points"] = backend["NumberOfPoints"]().fit_transform(diagrams).flatten()
    features["landscape"] = (
        backend["PersistenceLandscape"](n_layers=3, n_bins=20).fit_transform(diagrams).flatten()
    )
    features["betti_curve"] = backend["BettiCurve"](n_bins=20).fit_transform(diagrams).flatten()
    features["silhouette"] = backend["Silhouette"](n_bins=20).fit_transform(diagrams).flatten()

    diag = diagrams[0]
    for dim in (0, 1):
        mask = diag[:, 2] == dim
        lifetimes = np.array([], dtype=float)
        if mask.sum() > 0:
            lifetimes = diag[mask, 1] - diag[mask, 0]
            lifetimes = lifetimes[lifetimes > 0]
        prefix = f"h{dim}_lt"
        if len(lifetimes) > 0:
            features[f"{prefix}_mean"] = np.array([lifetimes.mean()])
            features[f"{prefix}_std"] = np.array([lifetimes.std()])
            features[f"{prefix}_max"] = np.array([lifetimes.max()])
            features[f"{prefix}_med"] = np.array([np.median(lifetimes)])
            features[f"{prefix}_skew"] = np.array(
                [
                    0.0
                    if lifetimes.std() == 0
                    else ((lifetimes - lifetimes.mean()) ** 3).mean() / (lifetimes.std() ** 3)
                ]
            )
            features[f"{prefix}_n_long"] = np.array(
                [float((lifetimes > np.percentile(lifetimes, 75)).sum())]
            )
        else:
            for suffix in ("mean", "std", "max", "med", "skew", "n_long"):
                features[f"{prefix}_{suffix}"] = np.array([0.0])

    return features


def extract_tda_features(
    snapshots: Sequence[VitalsSnapshot],
    *,
    config: TDAConfig | None = None,
) -> list[float] | None:
    """Compute the 663-dimensional TDA feature vector for a trace.

    Returns ``None`` when the trace is too short or the persistence
    pipeline cannot produce a usable point cloud (e.g. fewer than 3
    windows). Raises ``MissingTDADependencyError`` when the optional
    TDA backend is not installed.
    """

    cfg = config or TDAConfig()
    backend = _load_tda_backend()
    matrix = _trace_matrix(snapshots, cfg, backend)
    if matrix is None:
        return None

    np = backend["np"]
    trace_norm = _normalize_trace(matrix, np)

    all_features: dict[str, Any] = {}
    for window_size in cfg.window_sizes:
        point_cloud = _build_point_cloud(trace_norm, window_size, np)
        if len(point_cloud) < 3:
            return None
        vrp = backend["VietorisRipsPersistence"](
            homology_dimensions=cfg.homology_dimensions,
            max_edge_length=np.inf,
            n_jobs=1,
        )
        try:
            diagrams = vrp.fit_transform(point_cloud[np.newaxis, :, :])
        except Exception:
            return None
        diagram_features = _diagram_features(diagrams, backend)
        for name, values in diagram_features.items():
            all_features[f"w{window_size}_{name}"] = values

    vector = np.concatenate([all_features[name] for name in sorted(all_features.keys())])
    return [float(value) for value in vector]


@lru_cache(maxsize=4)
def _load_runaway_artifact(model_path: Path) -> Mapping[str, Any]:
    backend = _load_tda_backend()
    if not model_path.exists():
        raise FileNotFoundError(f"TDA runaway model not found: {model_path}")
    return cast(Mapping[str, Any], backend["joblib"].load(model_path))


def predict_runaway_cost(
    snapshots: Sequence[VitalsSnapshot],
    *,
    config: TDAConfig | None = None,
) -> TDARunawayPrediction | None:
    """Run the TDA runaway-cost classifier on a trace.

    Returns ``None`` when the trace is too short for TDA processing,
    so the caller can fall back to the handcrafted verdict. Raises
    ``MissingTDADependencyError`` if the optional backend is missing,
    or ``FileNotFoundError`` if the bundled model artifact is absent.
    """

    cfg = config or TDAConfig()
    features = extract_tda_features(snapshots, config=cfg)
    if features is None:
        return None
    backend = _load_tda_backend()
    np = backend["np"]
    artifact = _load_runaway_artifact(cfg.model_path)
    pipeline = artifact["pipeline"]
    X = np.array([features])
    probability = float(pipeline.predict_proba(X)[0][1])
    return TDARunawayPrediction(
        detected=probability >= cfg.decision_threshold,
        probability=probability,
    )
