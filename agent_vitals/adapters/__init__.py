"""Signal Adapter Protocol for Agent Vitals.

Defines the interface for mapping arbitrary agent state to RawSignals.
Concrete adapters implement `extract()` for their specific framework.

This is the primary extension point for integrating Agent Vitals with
any agent framework. Users can either:
1. Implement SignalAdapter for their framework
2. Construct RawSignals directly (manual mode)
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..schema import RawSignals


@runtime_checkable
class SignalAdapter(Protocol):
    """Protocol for extracting RawSignals from arbitrary agent state.

    Implementors map framework-specific telemetry/state to the generic
    RawSignals schema that the detection engine operates on.

    Minimum viable implementation requires only 4 fields:
        - findings_count (or equivalent output count)
        - coverage_score (or equivalent progress metric, 0.0-1.0)
        - total_tokens (cumulative token usage)
        - error_count (cumulative errors)

    All other fields enhance detection confidence but are not required.
    """

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        """Extract RawSignals from framework-specific agent state.

        Args:
            state: Framework-specific state mapping. The shape depends
                on the agent framework (e.g., LangGraph AgentState,
                LangChain callback data, raw telemetry dict).

        Returns:
            RawSignals instance populated from the agent state.
            Fields that cannot be mapped should use their defaults (0).
        """
        ...


class BaseAdapter:
    """Base class for signal adapters with common helpers.

    Subclass this and override `extract()` for your framework.
    Provides utility methods for safe field extraction.
    """

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_len(value: Any, default: int = 0) -> int:
        try:
            return len(value) if value else default
        except TypeError:
            return default

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        raise NotImplementedError("Subclasses must implement extract()")


class TelemetryAdapter(BaseAdapter):
    """Adapter for generic per-step telemetry dicts.

    Maps the StepTelemetry format from cross-agent harnesses to RawSignals.
    This is the adapter used for cross-agent validation — it works with
    any agent that emits per-step JSON telemetry.

    Expected state keys (from StepTelemetry.to_dict()):
        - outputs_produced: int (maps to findings_count)
        - total_tokens: int
        - errors: int (maps to error_count)
        - tool_calls: int (maps to query_count)
        - tool_results: int (maps to sources_count)

    Cumulative state keys (aggregated across steps):
        - cumulative_outputs: int (total outputs so far)
        - cumulative_tokens: int (total tokens so far)
        - cumulative_errors: int (total errors so far)
        - cumulative_queries: int (total tool calls so far)
        - cumulative_sources: int (total tool results so far)
        - coverage_score: float (progress estimate, 0.0-1.0)
    """

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        return RawSignals(
            findings_count=self._safe_int(state.get("cumulative_outputs", 0)),
            sources_count=self._safe_int(state.get("cumulative_sources", 0)),
            objectives_covered=0,
            coverage_score=self._safe_float(state.get("coverage_score", 0.0)),
            confidence_score=0.0,
            prompt_tokens=self._safe_int(state.get("prompt_tokens", 0)),
            completion_tokens=self._safe_int(state.get("completion_tokens", 0)),
            total_tokens=self._safe_int(state.get("cumulative_tokens", 0)),
            api_calls=self._safe_int(state.get("cumulative_queries", 0)),
            query_count=self._safe_int(state.get("cumulative_queries", 0)),
            unique_domains=0,
            refinement_count=0,
            convergence_delta=0.0,
            error_count=self._safe_int(state.get("cumulative_errors", 0)),
        )


__all__ = [
    "BaseAdapter",
    "SignalAdapter",
    "TelemetryAdapter",
]
