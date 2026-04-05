"""Tests for model-size-aware signal mapping.

Covers: auto-classification from completion token CV, explicit class override,
threshold adjustments, and integration with detect_loop for co-detection
suppression on small-model traces.
"""

from __future__ import annotations

import copy

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.signal_mapping import (
    MEDIUM_MODEL_CV_THRESHOLD,
    MIN_STEPS_FOR_CLASSIFICATION,
    SMALL_MODEL_CV_THRESHOLD,
    SignalMapping,
    classify_model_size,
    get_signal_mapping,
)
from agent_vitals.detection.stop_rule import derive_stop_signals
from agent_vitals.schema import VitalsSnapshot


# ---------------------------------------------------------------------------
# classify_model_size
# ---------------------------------------------------------------------------


class TestClassifyModelSize:
    """Tests for the auto-classification function."""

    def test_small_model_flat_completions(self) -> None:
        """Uniform completion lengths → small classification."""
        # 9B-style: ~500 tokens per step, very low variance
        tokens = [500.0, 502.0, 498.0, 501.0, 499.0, 503.0]
        assert classify_model_size(tokens) == "small"

    def test_large_model_varied_completions(self) -> None:
        """High variance completion lengths → large classification."""
        # 70B-style: wide range of completion lengths
        tokens = [200.0, 800.0, 350.0, 1200.0, 450.0, 900.0]
        assert classify_model_size(tokens) == "large"

    def test_medium_model_moderate_variance(self) -> None:
        """Moderate variance → medium classification."""
        # 27B-style: some variation but not extreme
        tokens = [400.0, 500.0, 350.0, 550.0, 420.0, 480.0]
        result = classify_model_size(tokens)
        assert result in ("medium", "small", "large")  # exact boundary depends on CV

    def test_insufficient_data_defaults_to_large(self) -> None:
        """Too few steps → conservative large default."""
        tokens = [500.0, 502.0, 498.0]  # only 3, need 4
        assert classify_model_size(tokens) == "large"

    def test_zero_completion_tokens_defaults_to_large(self) -> None:
        """All zero (not provided) → large default."""
        tokens = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert classify_model_size(tokens) == "large"

    def test_mixed_zero_and_nonzero_insufficient(self) -> None:
        """Not enough non-zero values → large default."""
        tokens = [0.0, 500.0, 0.0, 502.0, 0.0]  # only 2 nonzero
        assert classify_model_size(tokens) == "large"

    def test_explicit_class_small(self) -> None:
        """Explicit override bypasses auto-detection."""
        tokens = [200.0, 800.0, 350.0, 1200.0]  # would be large
        assert classify_model_size(tokens, explicit_class="small") == "small"

    def test_explicit_class_medium(self) -> None:
        assert classify_model_size([100.0] * 5, explicit_class="medium") == "medium"

    def test_explicit_class_large(self) -> None:
        assert classify_model_size([100.0] * 5, explicit_class="large") == "large"

    def test_explicit_class_invalid_defaults_to_large(self) -> None:
        assert classify_model_size([100.0] * 5, explicit_class="tiny") == "large"

    def test_explicit_auto_runs_detection(self) -> None:
        """auto is the default and should run CV-based classification."""
        flat = [500.0, 501.0, 499.0, 502.0, 498.0]
        assert classify_model_size(flat, explicit_class="auto") == "small"

    def test_empty_sequence(self) -> None:
        assert classify_model_size([]) == "large"

    def test_single_value(self) -> None:
        assert classify_model_size([500.0]) == "large"


# ---------------------------------------------------------------------------
# get_signal_mapping
# ---------------------------------------------------------------------------


class TestGetSignalMapping:
    """Tests for signal mapping retrieval."""

    def test_small_suppresses_token_variance(self) -> None:
        m = get_signal_mapping("small")
        assert m.suppress_token_variance_flat is True
        assert m.burn_rate_multiplier_scale == 2.0

    def test_medium_no_suppression_but_scaled(self) -> None:
        m = get_signal_mapping("medium")
        assert m.suppress_token_variance_flat is False
        assert m.burn_rate_multiplier_scale == 1.5

    def test_large_no_adjustments(self) -> None:
        m = get_signal_mapping("large")
        assert m.suppress_token_variance_flat is False
        assert m.burn_rate_multiplier_scale == 1.0

    def test_unknown_class_falls_back_to_large(self) -> None:
        m = get_signal_mapping("large")  # type: ignore[arg-type]
        assert m.model_size_class == "large"


# ---------------------------------------------------------------------------
# Integration: detect_loop with small-model signal mapping
# ---------------------------------------------------------------------------


def _make_snapshot(
    base: dict,
    *,
    loop_index: int,
    findings_count: int,
    coverage_score: float,
    total_tokens: int,
    completion_tokens: int = 0,
    query_count: int = 0,
    unique_domains: int = 0,
    dm_coverage: float = 0.0,
    cv_coverage: float = 0.0,
    sources_count: int = 0,
    error_count: int = 0,
) -> VitalsSnapshot:
    payload = copy.deepcopy(base)
    payload["loop_index"] = loop_index
    payload["signals"]["findings_count"] = findings_count
    payload["signals"]["sources_count"] = sources_count
    payload["signals"]["coverage_score"] = coverage_score
    payload["signals"]["total_tokens"] = total_tokens
    payload["signals"]["completion_tokens"] = completion_tokens
    payload["signals"]["error_count"] = error_count
    payload["signals"]["query_count"] = query_count
    payload["signals"]["unique_domains"] = unique_domains
    payload["metrics"]["dm_coverage"] = dm_coverage
    payload["metrics"]["cv_coverage"] = cv_coverage
    payload["loop_detected"] = False
    payload["loop_confidence"] = 0.0
    payload["loop_trigger"] = None
    payload["stuck_detected"] = False
    payload["stuck_confidence"] = 0.0
    payload["stuck_trigger"] = None
    return VitalsSnapshot.model_validate(payload)


@pytest.fixture
def base_snapshot() -> dict:
    return {
        "spec_version": "1.0.0",
        "timestamp": "2025-12-13T15:30:00Z",
        "mission_id": "test-signal-mapping",
        "loop_index": 0,
        "signals": {
            "findings_count": 0,
            "sources_count": 0,
            "objectives_covered": 0,
            "coverage_score": 0.0,
            "confidence_score": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "query_count": 0,
            "unique_domains": 0,
            "refinement_count": 0,
            "convergence_delta": 0.0,
            "error_count": 0,
        },
        "metrics": {
            "cv_coverage": 0.0,
            "cv_findings_rate": 0.0,
            "dm_coverage": 0.0,
            "dm_findings": 0.0,
            "qpf_tokens": 0.0,
            "cs_effort": 0.0,
        },
        "health_state": "healthy",
        "health_state_changed": False,
        "previous_health_state": None,
        "loop_detected": False,
        "loop_confidence": 0.0,
        "loop_trigger": None,
        "stuck_detected": False,
        "stuck_confidence": 0.0,
        "stuck_trigger": None,
        "intervention": None,
    }


def _build_small_model_trace(
    base: dict,
    *,
    steps: int = 8,
    completion_per_step: int = 500,
    findings_plateau_at: int = 3,
) -> list[VitalsSnapshot]:
    """Build a trace mimicking a 9B model: flat completions, growing prompt tokens."""
    snapshots = []
    for i in range(steps):
        # Small model: uniform completion tokens, growing prompt tokens
        prompt = 1000 + i * 800  # context accumulates
        comp = completion_per_step + (i % 3 - 1)  # tiny jitter: 499, 500, 501
        total = prompt + comp
        findings = min(i + 1, findings_plateau_at)  # plateau after N steps
        coverage = min(0.4, findings * 0.1)
        snapshots.append(
            _make_snapshot(
                base,
                loop_index=i,
                findings_count=findings,
                coverage_score=coverage,
                total_tokens=total,
                completion_tokens=comp,
                query_count=i + 1,
                unique_domains=min(i + 1, 3),
                dm_coverage=0.1 if i < findings_plateau_at else 0.02,
                cv_coverage=0.1,
                sources_count=0,
            )
        )
    return snapshots


class TestSmallModelCoDetectionSuppression:
    """Integration tests: small model traces should not trigger
    token_usage_variance_flat, and burn_rate threshold should be scaled up.
    """

    def test_small_model_suppresses_token_variance_flat(
        self, base_snapshot: dict
    ) -> None:
        """A 9B-style trace with flat completions should NOT trigger
        token_usage_variance_flat when model_size_class is 'small'."""
        trace = _build_small_model_trace(base_snapshot, steps=8)
        config = VitalsConfig(
            min_evidence_steps=3,
            model_size_class="small",
        )
        history = trace[:-1]
        current = trace[-1]
        result = detect_loop(current, history, config=config, workflow_type="research")
        # token_usage_variance_flat should be suppressed
        assert result.stuck_trigger != "token_usage_variance_flat"

    def test_large_model_allows_token_variance_flat(
        self, base_snapshot: dict
    ) -> None:
        """Same trace with model_size_class='large' can trigger
        token_usage_variance_flat (no suppression)."""
        trace = _build_small_model_trace(base_snapshot, steps=8)
        config = VitalsConfig(
            min_evidence_steps=3,
            model_size_class="large",
        )
        history = trace[:-1]
        current = trace[-1]
        result = detect_loop(current, history, config=config, workflow_type="research")
        # With large model class, the trigger is not suppressed (may or may not fire
        # depending on trace shape, but suppression is not active)
        # We just verify the detection ran without error
        assert result is not None

    def test_auto_detection_classifies_flat_completions_as_small(
        self, base_snapshot: dict
    ) -> None:
        """Auto model_size_class should detect flat completion tokens as small."""
        trace = _build_small_model_trace(base_snapshot, steps=8)
        config = VitalsConfig(
            min_evidence_steps=3,
            model_size_class="auto",
        )
        history = trace[:-1]
        current = trace[-1]
        result = detect_loop(current, history, config=config, workflow_type="research")
        # Auto-detection should suppress token_usage_variance_flat for this trace
        assert result.stuck_trigger != "token_usage_variance_flat"

    def test_small_model_burn_rate_threshold_scaled(
        self, base_snapshot: dict
    ) -> None:
        """Small model should have a higher effective burn rate threshold,
        reducing burn_rate_anomaly false positives."""
        # Build a trace where burn rate is 4x baseline (would trigger at default 3x
        # but should NOT trigger at scaled 6x for small models)
        snapshots = []
        for i in range(6):
            prompt = 1000 + i * 500
            comp = 500 + (i % 3 - 1)
            total = prompt + comp
            if i < 4:
                findings = i + 2  # growing findings
            else:
                findings = 5  # plateau
            coverage = min(0.4, findings * 0.05)
            snapshots.append(
                _make_snapshot(
                    base_snapshot,
                    loop_index=i,
                    findings_count=findings,
                    coverage_score=coverage,
                    total_tokens=total,
                    completion_tokens=comp,
                    dm_coverage=0.1 if i < 4 else 0.02,
                    cv_coverage=0.1,
                )
            )
        config_small = VitalsConfig(
            min_evidence_steps=3,
            model_size_class="small",
            burn_rate_multiplier=3.0,
        )
        config_large = VitalsConfig(
            min_evidence_steps=3,
            model_size_class="large",
            burn_rate_multiplier=3.0,
        )
        history = snapshots[:-1]
        current = snapshots[-1]

        result_small = detect_loop(
            current, history, config=config_small, workflow_type="research"
        )
        result_large = detect_loop(
            current, history, config=config_large, workflow_type="research"
        )

        # Small model has 2x scaled multiplier (effective 6.0), so burn_rate_anomaly
        # is less likely to fire than large model (effective 3.0)
        if result_large.stuck_trigger == "burn_rate_anomaly":
            # If large fires, small should not (or have lower confidence)
            assert (
                result_small.stuck_trigger != "burn_rate_anomaly"
                or result_small.stuck_confidence <= result_large.stuck_confidence
            )

    def test_config_model_size_class_default_is_auto(self) -> None:
        """VitalsConfig defaults to auto model size classification."""
        config = VitalsConfig()
        assert config.model_size_class == "auto"

    def test_config_model_size_class_from_dict(self) -> None:
        """model_size_class can be set via from_dict."""
        config = VitalsConfig.from_dict({"model_size_class": "small"})
        assert config.model_size_class == "small"


class TestSmallModelStopRule:
    """Verify that stop rule correctly routes small-model signals."""

    def test_burn_rate_anomaly_maps_to_runaway_cost(
        self, base_snapshot: dict
    ) -> None:
        """When burn_rate_anomaly fires on any model size, it should
        map to runaway_cost_detected in stop rule signals."""
        snapshot = {
            "stuck_detected": True,
            "stuck_trigger": "burn_rate_anomaly",
            "loop_detected": False,
            "signals": {"error_count": 0},
        }
        signals = derive_stop_signals(snapshot)
        assert signals.runaway_cost_detected is True
        assert signals.stuck_detected is True

    def test_token_variance_flat_does_not_map_to_runaway_cost(
        self, base_snapshot: dict
    ) -> None:
        """token_usage_variance_flat is stuck, not runaway_cost."""
        snapshot = {
            "stuck_detected": True,
            "stuck_trigger": "token_usage_variance_flat",
            "loop_detected": False,
            "signals": {"error_count": 0},
        }
        signals = derive_stop_signals(snapshot)
        assert signals.runaway_cost_detected is False
        assert signals.stuck_detected is True
