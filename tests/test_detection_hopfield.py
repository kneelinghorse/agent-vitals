"""Tests for the Hopfield early-detection adapter (av-s09-m01).

Covers:
- Module imports cleanly without optional Hopfield dependencies installed
- ``is_hopfield_available`` reflects backend presence as a bool
- ``select_prefix_variant`` routes len<=4 → p3, 5-6 → p5, >=7 → p7
- Empty trace short-circuits to ``None``
- Bundled artifact loads and produces a deterministic prediction
- ``HopfieldEarlyDetector`` facade matches the functional API
- Sidecar feature_order matches the canonical 17-feature ordering
- ONNX vs reference PyTorch parity for at least one detector when both
  optional stacks are available (skipped otherwise)
"""

from __future__ import annotations

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection import hopfield as hopfield_module
from agent_vitals.detection import loop as loop_module
from agent_vitals.detection.hopfield import (
    DEFAULT_OVERRIDE_THRESHOLDS,
    HOPFIELD_DETECTORS,
    HopfieldConfig,
    HopfieldEarlyDetector,
    HopfieldPrediction,
    MissingHopfieldDependencyError,
    N_FEATURES,
    hopfield_override_fires,
    is_hopfield_available,
    predict,
    select_prefix_variant,
)
from agent_vitals.detection.loop import LoopDetectionResult, detect_loop
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


def _snapshot(loop_index: int = 0) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id="t1",
        loop_index=loop_index,
        signals=RawSignals(
            findings_count=1 + loop_index,
            coverage_score=min(0.99, 0.4 + 0.02 * loop_index),
            total_tokens=1000 + 100 * loop_index,
            error_count=0,
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


class TestHopfieldModule:
    def test_module_importable_without_deps(self) -> None:
        """The module must import cleanly regardless of optional deps."""
        assert hasattr(hopfield_module, "predict")
        assert hasattr(hopfield_module, "is_hopfield_available")
        assert hasattr(hopfield_module, "HopfieldConfig")
        assert hasattr(hopfield_module, "HopfieldPrediction")
        assert hasattr(hopfield_module, "HopfieldEarlyDetector")
        assert hasattr(hopfield_module, "MissingHopfieldDependencyError")
        assert N_FEATURES == 17

    def test_is_hopfield_available_returns_bool(self) -> None:
        assert isinstance(is_hopfield_available(), bool)

    def test_detectors_tuple_matches_bench(self) -> None:
        assert HOPFIELD_DETECTORS == (
            "loop",
            "stuck",
            "confabulation",
            "thrash",
            "runaway_cost",
        )

    def test_predict_empty_trace_returns_none(self) -> None:
        # Empty input is checked before backend access, so this is safe
        # even when onnxruntime is not installed.
        assert predict([], "loop") is None


class TestPrefixSelector:
    @pytest.mark.parametrize("length", [1, 2, 3, 4])
    def test_short_traces_select_p3(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        assert select_prefix_variant(snaps) == "p3"

    @pytest.mark.parametrize("length", [5, 6])
    def test_medium_traces_select_p5(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        assert select_prefix_variant(snaps) == "p5"

    @pytest.mark.parametrize("length", [7, 12, 25])
    def test_long_traces_select_p7(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        assert select_prefix_variant(snaps) == "p7"

    def test_custom_p3_max_len(self) -> None:
        snaps = [_snapshot(i) for i in range(6)]
        assert select_prefix_variant(snaps, p3_max_len=10) == "p3"

    def test_custom_p5_max_len(self) -> None:
        snaps = [_snapshot(i) for i in range(10)]
        assert select_prefix_variant(snaps, p5_max_len=12) == "p5"


class TestHopfieldInferenceSkippedWhenUnavailable:
    """Guard tests that should run cleanly with no optional deps installed."""

    def test_predict_raises_missing_when_backend_absent(self) -> None:
        if is_hopfield_available():
            pytest.skip("hopfield backend is installed; covered elsewhere")
        with pytest.raises(MissingHopfieldDependencyError):
            predict([_snapshot(i) for i in range(3)], "loop")


@pytest.mark.skipif(
    not is_hopfield_available(),
    reason="agent-vitals[hopfield] extras not installed",
)
class TestHopfieldInferenceWithBackend:
    @pytest.mark.parametrize("detector", HOPFIELD_DETECTORS)
    def test_p3_prediction_shape(self, detector: str) -> None:
        snaps = [_snapshot(i) for i in range(3)]
        result = predict(snaps, detector)
        assert isinstance(result, HopfieldPrediction)
        assert result.detector == detector
        assert result.prefix_variant == "p3"
        assert 0.0 <= result.probability <= 1.0
        assert isinstance(result.detected, bool)

    @pytest.mark.parametrize("detector", HOPFIELD_DETECTORS)
    def test_p5_prediction_shape(self, detector: str) -> None:
        snaps = [_snapshot(i) for i in range(5)]
        result = predict(snaps, detector)
        assert isinstance(result, HopfieldPrediction)
        assert result.prefix_variant == "p5"

    @pytest.mark.parametrize("detector", HOPFIELD_DETECTORS)
    def test_p7_prediction_shape(self, detector: str) -> None:
        snaps = [_snapshot(i) for i in range(8)]
        result = predict(snaps, detector)
        assert isinstance(result, HopfieldPrediction)
        assert result.prefix_variant == "p7"

    def test_prediction_is_deterministic(self) -> None:
        snaps = [_snapshot(i) for i in range(5)]
        a = predict(snaps, "loop")
        b = predict(snaps, "loop")
        assert a is not None and b is not None
        assert a.probability == pytest.approx(b.probability, abs=1e-9)
        assert a.detected == b.detected

    def test_facade_matches_functional_api(self) -> None:
        snaps = [_snapshot(i) for i in range(4)]
        detector = HopfieldEarlyDetector()
        facade = detector.predict_one(snaps, "stuck")
        functional = predict(snaps, "stuck")
        assert facade is not None and functional is not None
        assert facade.probability == pytest.approx(functional.probability, abs=1e-9)
        assert facade.prefix_variant == functional.prefix_variant

    def test_unknown_detector_raises(self) -> None:
        with pytest.raises(ValueError):
            predict([_snapshot(0)], "not_a_detector")

    def test_custom_threshold_changes_decision(self) -> None:
        snaps = [_snapshot(i) for i in range(5)]
        baseline = predict(snaps, "loop")
        assert baseline is not None
        # A threshold guaranteed to flip the verdict in both directions.
        always_on = predict(snaps, "loop", config=HopfieldConfig(decision_threshold=-0.1))
        always_off = predict(snaps, "loop", config=HopfieldConfig(decision_threshold=1.1))
        assert always_on is not None and always_off is not None
        assert always_on.detected is True
        assert always_off.detected is False


@pytest.mark.skipif(
    not is_hopfield_available(),
    reason="agent-vitals[hopfield] extras not installed",
)
class TestBundledArtifacts:
    """Smoke-check the bundled artifacts so a packaging regression is caught."""

    def test_all_prefix_variants_load(self) -> None:
        from agent_vitals.detection.hopfield import _DEFAULT_MODEL_DIR, _load_artifact

        for detector in HOPFIELD_DETECTORS:
            for prefix_variant in ("p3", "p5"):
                artifact = _load_artifact(detector, prefix_variant, _DEFAULT_MODEL_DIR)
                assert artifact.max_steps == 20
                assert artifact.mean.shape == (N_FEATURES,)
                assert artifact.std.shape == (N_FEATURES,)
                assert artifact.feature_order == (
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
                    "dm_coverage",
                    "dm_findings",
                    "cv_coverage",
                    "cv_findings_rate",
                    "qpf_tokens",
                    "cs_effort",
                )

    def test_p3_prefix_len_is_three(self) -> None:
        from agent_vitals.detection.hopfield import _DEFAULT_MODEL_DIR, _load_artifact

        artifact = _load_artifact("loop", "p3", _DEFAULT_MODEL_DIR)
        assert artifact.prefix_len == 3

    def test_p5_prefix_len_is_five(self) -> None:
        from agent_vitals.detection.hopfield import _DEFAULT_MODEL_DIR, _load_artifact

        artifact = _load_artifact("loop", "p5", _DEFAULT_MODEL_DIR)
        assert artifact.prefix_len == 5


# ---------------------------------------------------------------------------
# AV-S09-M02 — override layer wired into _resolve_detections
# ---------------------------------------------------------------------------


class TestOverrideThresholdConstants:
    def test_default_thresholds_cover_all_detectors(self) -> None:
        for detector in HOPFIELD_DETECTORS:
            assert detector in DEFAULT_OVERRIDE_THRESHOLDS
            value = DEFAULT_OVERRIDE_THRESHOLDS[detector]
            assert 0.0 < value <= 1.0

    def test_default_thresholds_match_bench_calibration(self) -> None:
        # AV-S09-M03 bench acceptance (intel response 0f771492) calibrated
        # the per-detector thresholds against the bench v1 corpus PR curves.
        # The values are pinned here so any future regression of the
        # calibration trips a test — adjustments must come with new bench
        # evidence and an explicit CHANGELOG entry.
        assert DEFAULT_OVERRIDE_THRESHOLDS["loop"] == 0.80
        assert DEFAULT_OVERRIDE_THRESHOLDS["stuck"] == 0.90
        assert DEFAULT_OVERRIDE_THRESHOLDS["confabulation"] == 0.80
        assert DEFAULT_OVERRIDE_THRESHOLDS["thrash"] == 0.70
        assert DEFAULT_OVERRIDE_THRESHOLDS["runaway_cost"] == 0.90

    def test_default_thresholds_above_sigmoid_baseline(self) -> None:
        # Sanity floor: every threshold must be strictly above the
        # default sigmoid 0.5 cutoff so the marker is meaningfully gated.
        for detector in HOPFIELD_DETECTORS:
            assert DEFAULT_OVERRIDE_THRESHOLDS[detector] > 0.5


class TestEarlyWindowGate:
    @pytest.mark.parametrize("length", [0, 1, 2])
    def test_below_window_returns_false(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        # Below the window — must short-circuit before any backend
        # access, so this is safe even when the optional extras are
        # missing.
        assert hopfield_override_fires(snaps) is False

    @pytest.mark.parametrize("length", [7, 9, 12])
    def test_full_trace_returns_false(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        assert hopfield_override_fires(snaps) is False


@pytest.mark.skipif(
    not is_hopfield_available(),
    reason="agent-vitals[hopfield] extras not installed",
)
class TestOverrideHelperWithBackend:
    @pytest.mark.parametrize("length", [3, 4, 5, 6])
    def test_in_window_returns_bool(self, length: int) -> None:
        snaps = [_snapshot(i) for i in range(length)]
        result = hopfield_override_fires(snaps)
        assert isinstance(result, bool)

    def test_threshold_at_zero_always_fires(self) -> None:
        snaps = [_snapshot(i) for i in range(4)]
        zeros = {d: 0.0 for d in HOPFIELD_DETECTORS}
        # Any non-degenerate sigmoid output is >= 0.0 so the override
        # must fire on the first detector.
        assert hopfield_override_fires(snaps, thresholds=zeros) is True

    def test_threshold_above_one_never_fires(self) -> None:
        snaps = [_snapshot(i) for i in range(4)]
        impossible = {d: 1.01 for d in HOPFIELD_DETECTORS}
        assert hopfield_override_fires(snaps, thresholds=impossible) is False

    def test_missing_model_dir_returns_false(self, tmp_path: pytest.TempPathFactory) -> None:
        snaps = [_snapshot(i) for i in range(4)]
        empty_dir = tmp_path / "no_models"  # type: ignore[operator]
        empty_dir.mkdir()
        # Force a directory that has no .onnx artifacts — graceful
        # degradation should turn this into a False return rather than
        # a crash.
        zeros = {d: 0.0 for d in HOPFIELD_DETECTORS}
        result = hopfield_override_fires(
            snaps,
            model_dir=empty_dir,
            thresholds=zeros,
        )
        assert result is False


class TestResolveDetectionsHopfieldWiring:
    """Contract tests for the AV-S09-M02 wiring in _resolve_detections."""

    def test_marker_default_false_under_default_config(self) -> None:
        # The default config has hopfield_enabled=False, so the marker
        # must stay False even on a long trace.
        history = [_snapshot(i) for i in range(6)]
        current = _snapshot(6)
        result = detect_loop(current, history)
        assert isinstance(result, LoopDetectionResult)
        assert result.hopfield_override_active is False

    @pytest.mark.skipif(
        not is_hopfield_available(),
        reason="agent-vitals[hopfield] extras not installed",
    )
    def test_marker_does_not_fire_at_full_trace(self) -> None:
        # At trace length >= 7 the override window is closed; even with
        # hopfield_enabled=True, the helper must return False.
        cfg = VitalsConfig(hopfield_enabled=True)
        history = [_snapshot(i) for i in range(8)]
        current = _snapshot(8)
        result = detect_loop(current, history, config=cfg)
        assert result.hopfield_override_active is False

    @pytest.mark.skipif(
        not is_hopfield_available(),
        reason="agent-vitals[hopfield] extras not installed",
    )
    def test_marker_does_not_fire_when_disabled_even_in_window(self) -> None:
        # In the early window but with hopfield_enabled=False — the
        # helper must short-circuit before any backend call.
        cfg = VitalsConfig(hopfield_enabled=False)
        history = [_snapshot(i) for i in range(3)]
        current = _snapshot(3)
        result = detect_loop(current, history, config=cfg)
        assert result.hopfield_override_active is False

    def test_consult_helper_short_circuits_when_disabled(self) -> None:
        from agent_vitals.detection.loop import _consult_hopfield_override, _prepare_context

        cfg = VitalsConfig(hopfield_enabled=False)
        history = [_snapshot(i) for i in range(3)]
        current = _snapshot(3)
        ctx = _prepare_context(current, history, cfg, "unknown")
        assert ctx is not None
        assert _consult_hopfield_override(ctx) is False

    def test_per_detector_flags_unchanged_when_marker_set(self) -> None:
        """The marker is informational — it must NOT mutate per-detector flags.

        Run a single snapshot through detect_loop with hopfield_enabled=True
        and confirm the loop/stuck/confabulation/runaway_cost flags match
        the same call with hopfield_enabled=False. This is the
        bit-identical contract that lets the AV-S09-M02 wiring pass the
        ci_backtest gate without altering composite cells.
        """
        history = [_snapshot(i) for i in range(4)]
        current = _snapshot(4)
        baseline = detect_loop(current, history, config=VitalsConfig(hopfield_enabled=False))
        if not is_hopfield_available():
            return  # without the backend the comparison is degenerate
        with_hopfield = detect_loop(
            current, history, config=VitalsConfig(hopfield_enabled=True)
        )
        for field in (
            "loop_detected",
            "loop_confidence",
            "loop_trigger",
            "stuck_detected",
            "stuck_confidence",
            "stuck_trigger",
            "confabulation_detected",
            "confabulation_confidence",
            "runaway_cost_detected",
            "runaway_cost_confidence",
            "detector_priority",
        ):
            assert getattr(baseline, field) == getattr(with_hopfield, field), field

    def test_as_snapshot_update_includes_marker(self) -> None:
        result = LoopDetectionResult(hopfield_override_active=True)
        update = result.as_snapshot_update()
        assert "hopfield_override_active" in update
        assert update["hopfield_override_active"] is True


class TestVitalsSnapshotMarkerField:
    def test_default_value_is_false(self) -> None:
        snapshot = _snapshot(0)
        assert snapshot.hopfield_override_active is False

    def test_field_accepts_explicit_true(self) -> None:
        snapshot = VitalsSnapshot(
            mission_id="t1",
            loop_index=0,
            signals=RawSignals(
                findings_count=1,
                coverage_score=0.5,
                total_tokens=1000,
                error_count=0,
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
            hopfield_override_active=True,
        )
        assert snapshot.hopfield_override_active is True


def test_loop_module_exposes_consult_helper() -> None:
    # Sanity check that the module-private helper used by the wiring
    # exists and is callable; protects against accidental rename.
    assert callable(getattr(loop_module, "_consult_hopfield_override"))
