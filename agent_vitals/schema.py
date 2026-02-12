"""Pydantic schemas for Agent Vitals snapshots.

These models define the contract for capturing point-in-time agent health
signals and derived metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

HealthState = Literal["healthy", "warning", "critical"]
InterventionType = Literal["shadow", "enforced"]
InterventionAction = Literal[
    "none",
    "break_loop",
    "inject_diversity",
    "force_finalize",
    "reduce_scope",
]


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RawSignals(BaseModel):
    """Raw signals extracted from the agent state.

    Minimum viable fields (4): findings_count, coverage_score,
    total_tokens, error_count.  All other fields enhance detection
    confidence but default to 0.
    """

    model_config = ConfigDict(extra="forbid")

    findings_count: int = Field(..., ge=0)
    sources_count: int = Field(default=0, ge=0)
    objectives_covered: int = Field(default=0, ge=0)
    coverage_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(..., ge=0)
    api_calls: int = Field(default=0, ge=0)

    query_count: int = Field(default=0, ge=0)
    unique_domains: int = Field(default=0, ge=0)
    refinement_count: int = Field(default=0, ge=0)

    convergence_delta: float = Field(default=0.0)
    error_count: int = Field(..., ge=0)


class TemporalMetricsResult(BaseModel):
    """Computed temporal metrics for a vitals snapshot."""

    model_config = ConfigDict(extra="forbid")

    cv_coverage: float = Field(ge=0.0, description="CV of coverage score")
    cv_findings_rate: float = Field(ge=0.0, description="CV of findings per loop")

    dm_coverage: float = Field(ge=-1.0, le=1.0, description="Directional momentum of coverage")
    dm_findings: float = Field(ge=-1.0, le=1.0, description="Directional momentum of findings")

    qpf_tokens: float = Field(ge=0.0, le=1.0, description="Token distribution fairness")
    cs_effort: float = Field(ge=0.0, le=1.0, description="Crescendo symmetry of effort curve")


class InterventionRecord(BaseModel):
    """Record of an intervention decision (shadow or enforced)."""

    model_config = ConfigDict(extra="forbid")

    intervention_type: InterventionType
    action: InterventionAction
    trigger: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)


class VitalsSnapshot(BaseModel):
    """Point-in-time capture of agent health metrics."""

    model_config = ConfigDict(extra="forbid")

    spec_version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mission_id: str = Field(..., min_length=1)
    run_id: Optional[str] = Field(default=None, min_length=1)
    thread_id: Optional[str] = Field(default=None, min_length=1)
    loop_index: int = Field(..., ge=0)

    signals: RawSignals
    metrics: TemporalMetricsResult

    health_state: HealthState
    health_state_changed: bool = Field(default=False)
    previous_health_state: Optional[HealthState] = Field(default=None)

    loop_detected: bool = Field(default=False)
    loop_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    loop_trigger: Optional[str] = Field(default=None)

    stuck_detected: bool = Field(default=False)
    stuck_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stuck_trigger: Optional[str] = Field(default=None)

    output_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    output_fingerprint: Optional[str] = Field(default=None)

    intervention: Optional[InterventionRecord] = Field(default=None)

    @property
    def any_failure(self) -> bool:
        """Return True when any failure-mode signal is active."""
        return bool(self.loop_detected or self.stuck_detected)

    @field_validator("spec_version", mode="before")
    @classmethod
    def _validate_spec_version(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("spec_version cannot be empty")
        parts = text.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError("spec_version must be semver (e.g., 1.0.0)")
        return text

    @field_validator("timestamp", mode="before")
    @classmethod
    def _default_timestamp(cls, value: Any) -> Any:
        if value is None or value == "":
            return datetime.now(timezone.utc)
        return value

    @field_serializer("timestamp", when_used="json")
    def _serialise_timestamp(self, value: datetime) -> str:
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        else:
            normalized = normalized.astimezone(timezone.utc)
        normalized = normalized.replace(microsecond=0)
        return normalized.isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def _enforce_health_state_change_consistency(self) -> "VitalsSnapshot":
        if self.health_state_changed:
            if self.previous_health_state is None:
                raise ValueError("previous_health_state is required when health_state_changed=true")
            if self.previous_health_state == self.health_state:
                raise ValueError("previous_health_state must differ from health_state when changed")
        return self


@dataclass(frozen=True, slots=True)
class HysteresisConfig:
    """Threshold configuration for temporal hysteresis.

    Values are expected to be in the same scale as the input signal
    (typically 0-1 for coverage and confidence scores).
    """

    enter_warning: float = 0.4
    exit_warning: float = 0.6
    enter_critical: float = 0.2
    exit_critical: float = 0.35

    def __post_init__(self) -> None:
        for name, value in (
            ("enter_warning", self.enter_warning),
            ("exit_warning", self.exit_warning),
            ("enter_critical", self.enter_critical),
            ("exit_critical", self.exit_critical),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


__all__ = [
    "HealthState",
    "HysteresisConfig",
    "InterventionAction",
    "InterventionRecord",
    "InterventionType",
    "RawSignals",
    "TemporalMetricsResult",
    "VitalsSnapshot",
]
