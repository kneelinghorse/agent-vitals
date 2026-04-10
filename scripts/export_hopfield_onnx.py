"""One-shot ONNX exporter for the bench Hopfield prefix-models.

Reads each `.pt` artifact in
``agent-vitals-bench/prototypes/hopfield_models/`` for the supported
``(detector, prefix_len)`` combinations, runs ``torch.onnx.export``
following the canonical snippet from the bench inference contract
(reference memory ``reference_hopfield_inference_contract.md``), verifies
that the ONNX runtime output matches the PyTorch reference on a random
fixture batch, and writes a JSON sidecar containing the per-detector
``mean``/``std``/``feature_order``/``max_steps``/``prefix_len``.

Output goes to ``agent_vitals/models/hopfield/`` so the wheel can bundle
the artifacts via the package-data glob.

This script is a developer tool, not part of the runtime; it imports
``torch``, ``hflayers``, ``onnx``, and ``onnxruntime``. Re-run it
whenever bench publishes new artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT.parent / "agent-vitals-bench"
BENCH_PROTOTYPES = BENCH_ROOT / "prototypes"
SOURCE_MODEL_DIR = BENCH_PROTOTYPES / "hopfield_models"
TARGET_DIR = REPO_ROOT / "agent_vitals" / "models" / "hopfield"

DETECTORS = ("loop", "stuck", "confabulation", "thrash", "runaway_cost")
PREFIX_LENS = (3, 5, 7)

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

# Imports below intentionally happen after sys.path manipulation so the
# bench prototype module can be found.
from prototypes.hopfield_detector import (  # noqa: E402
    METRIC_KEYS,
    N_FEATURES,
    SIGNAL_KEYS,
    _build_classifier_class,
    load_hopfield_artifact,
)

FEATURE_ORDER = list(SIGNAL_KEYS) + list(METRIC_KEYS)
assert len(FEATURE_ORDER) == N_FEATURES == 17


def export_one(detector: str, prefix_len: int) -> None:
    artifact = load_hopfield_artifact(
        detector,
        prefix_len=prefix_len,
        model_dir=SOURCE_MODEL_DIR,
        map_location="cpu",
    )
    cfg = artifact["config"]

    Classifier = _build_classifier_class()
    model = Classifier(
        n_features=int(artifact["n_features"]),
        d_model=int(cfg["d_model"]),
        n_stored=int(cfg["n_stored"]),
        n_heads=int(cfg["n_heads"]),
        scaling=float(cfg["scaling"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    max_steps = int(cfg["max_steps"])

    onnx_path = TARGET_DIR / f"{detector}_p{prefix_len}.onnx"
    sidecar_path = TARGET_DIR / f"{detector}_p{prefix_len}.json"

    dummy_x = torch.zeros(1, max_steps, N_FEATURES, dtype=torch.float32)
    dummy_lengths = torch.tensor([prefix_len], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_x, dummy_lengths),
        onnx_path.as_posix(),
        input_names=["x", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "lengths": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Parity check on a random fixture (deterministic seed per artifact).
    rng = np.random.default_rng(seed=hash((detector, prefix_len)) & 0xFFFFFFFF)
    x_np = rng.standard_normal((2, max_steps, N_FEATURES)).astype(np.float32)
    lengths_np = np.array([prefix_len, max(prefix_len - 1, 1)], dtype=np.int64)

    with torch.no_grad():
        torch_out = (
            model(
                torch.from_numpy(x_np),
                torch.from_numpy(lengths_np.astype(np.int64)),
            )
            .cpu()
            .numpy()
        )

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    ort_out = session.run(None, {"x": x_np, "lengths": lengths_np})[0]

    diff = float(np.max(np.abs(torch_out - ort_out)))
    if diff > 1e-4:
        raise RuntimeError(
            f"ONNX/PyTorch divergence for {detector}_p{prefix_len}: max|delta|={diff:.6f}"
        )

    sidecar = {
        "detector": detector,
        "prefix_len": prefix_len,
        "max_steps": max_steps,
        "n_features": N_FEATURES,
        "feature_order": FEATURE_ORDER,
        "mean": [float(v) for v in artifact["mean"]],
        "std": [float(v) for v in artifact["std"]],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    onnx_size = onnx_path.stat().st_size
    print(
        f"  exported {detector}_p{prefix_len}: "
        f"onnx={onnx_size / 1024:.1f}KB max|delta|={diff:.2e}"
    )


def main() -> int:
    if not SOURCE_MODEL_DIR.is_dir():
        raise SystemExit(f"bench hopfield_models not found at {SOURCE_MODEL_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting Hopfield ONNX artifacts to {TARGET_DIR}")
    for detector in DETECTORS:
        for prefix_len in PREFIX_LENS:
            export_one(detector, prefix_len)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
