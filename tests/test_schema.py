"""Tests for agent_vitals.schema module."""


import pytest
from pydantic import ValidationError

from agent_vitals.schema import (
    RawSignals,
    TemporalMetricsResult,
    VitalsSnapshot,
)


class TestRawSignals:
    """Tests for RawSignals model."""

    def test_minimum_viable_fields(self) -> None:
        """4-field minimum: findings_count, coverage_score, total_tokens, error_count."""
        signals = RawSignals(
            findings_count=5,
            coverage_score=0.6,
            total_tokens=12000,
            error_count=0,
        )
        assert signals.findings_count == 5
        assert signals.coverage_score == 0.6
        assert signals.total_tokens == 12000
        assert signals.error_count == 0
        # Optional fields default to 0
        assert signals.sources_count == 0
        assert signals.objectives_covered == 0
        assert signals.prompt_tokens == 0
        assert signals.completion_tokens == 0
        assert signals.api_calls == 0
        assert signals.query_count == 0
        assert signals.unique_domains == 0
        assert signals.refinement_count == 0
        assert signals.convergence_delta == 0.0
        assert signals.confidence_score == 0.0

    def test_all_fields(self) -> None:
        signals = RawSignals(
            findings_count=10,
            sources_count=5,
            objectives_covered=3,
            coverage_score=0.8,
            confidence_score=0.9,
            prompt_tokens=5000,
            completion_tokens=7000,
            total_tokens=12000,
            api_calls=15,
            query_count=8,
            unique_domains=4,
            refinement_count=2,
            convergence_delta=0.05,
            error_count=1,
        )
        assert signals.findings_count == 10
        assert signals.sources_count == 5

    def test_negative_findings_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawSignals(
                findings_count=-1,
                coverage_score=0.5,
                total_tokens=100,
                error_count=0,
            )

    def test_coverage_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawSignals(
                findings_count=1,
                coverage_score=1.5,
                total_tokens=100,
                error_count=0,
            )


class TestVitalsSnapshot:
    """Tests for VitalsSnapshot model."""

    @staticmethod
    def _make_snapshot(**overrides: object) -> VitalsSnapshot:
        defaults = dict(
            mission_id="test-mission",
            loop_index=0,
            signals=RawSignals(
                findings_count=5,
                coverage_score=0.6,
                total_tokens=12000,
                error_count=0,
            ),
            metrics=TemporalMetricsResult(
                cv_coverage=0.1,
                cv_findings_rate=0.2,
                dm_coverage=0.3,
                dm_findings=0.4,
                qpf_tokens=0.5,
                cs_effort=0.6,
            ),
            health_state="healthy",
        )
        defaults.update(overrides)
        return VitalsSnapshot(**defaults)  # type: ignore[arg-type]

    def test_basic_creation(self) -> None:
        snap = self._make_snapshot()
        assert snap.mission_id == "test-mission"
        assert snap.loop_index == 0
        assert snap.health_state == "healthy"
        assert snap.loop_detected is False
        assert snap.confabulation_detected is False
        assert snap.stuck_detected is False
        assert snap.source_finding_ratio is None
        assert snap.ratio_trend == "insufficient_data"
        assert snap.ratio_declining_steps == 0

    def test_any_failure_false_when_healthy(self) -> None:
        snap = self._make_snapshot()
        assert snap.any_failure is False

    def test_any_failure_true_on_loop(self) -> None:
        snap = self._make_snapshot(loop_detected=True, loop_confidence=0.85)
        assert snap.any_failure is True

    def test_any_failure_true_on_stuck(self) -> None:
        snap = self._make_snapshot(stuck_detected=True, stuck_confidence=0.7)
        assert snap.any_failure is True

    def test_any_failure_true_on_confabulation(self) -> None:
        snap = self._make_snapshot(
            confabulation_detected=True,
            confabulation_confidence=0.8,
            confabulation_trigger="source_finding_ratio_low",
        )
        assert snap.any_failure is True

    def test_health_state_change_requires_previous(self) -> None:
        with pytest.raises(ValidationError):
            self._make_snapshot(
                health_state="warning",
                health_state_changed=True,
                previous_health_state=None,
            )

    def test_health_state_change_must_differ(self) -> None:
        with pytest.raises(ValidationError):
            self._make_snapshot(
                health_state="warning",
                health_state_changed=True,
                previous_health_state="warning",
            )

    def test_valid_health_state_change(self) -> None:
        snap = self._make_snapshot(
            health_state="warning",
            health_state_changed=True,
            previous_health_state="healthy",
        )
        assert snap.health_state == "warning"
        assert snap.previous_health_state == "healthy"

    def test_spec_version_validation(self) -> None:
        with pytest.raises(ValidationError):
            self._make_snapshot(spec_version="bad")

    def test_json_round_trip(self) -> None:
        snap = self._make_snapshot()
        json_str = snap.model_dump_json()
        restored = VitalsSnapshot.model_validate_json(json_str)
        assert restored.mission_id == snap.mission_id
        assert restored.loop_index == snap.loop_index

    def test_ratio_trend_validation(self) -> None:
        with pytest.raises(ValidationError):
            self._make_snapshot(ratio_trend="invalid-trend")
