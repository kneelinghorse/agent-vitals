"""Adapter interfaces and built-in integrations for Agent Vitals."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import RawSignals
from .autogen import AutoGenAdapter
from .base import BaseAdapter, SignalAdapter
from .crewai import CrewAIAdapter
from .dspy import DSPyAdapter
from .haystack import HaystackAdapter
from .langchain import LangChainAdapter
from .langgraph import LangGraphAdapter


class TelemetryAdapter(BaseAdapter):
    """Adapter for generic telemetry dicts used in cross-agent validation."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)
        return self.validate(
            RawSignals(
                findings_count=self._safe_int(normalized.get("cumulative_outputs", 0)),
                sources_count=self._safe_int(normalized.get("cumulative_sources", 0)),
                objectives_covered=0,
                coverage_score=self._clip01(
                    self._safe_float(normalized.get("coverage_score", 0.0))
                ),
                confidence_score=0.0,
                prompt_tokens=self._safe_int(normalized.get("prompt_tokens", 0)),
                completion_tokens=self._safe_int(normalized.get("completion_tokens", 0)),
                total_tokens=self._safe_int(normalized.get("cumulative_tokens", 0)),
                api_calls=self._safe_int(normalized.get("cumulative_queries", 0)),
                query_count=self._safe_int(normalized.get("cumulative_queries", 0)),
                unique_domains=0,
                refinement_count=0,
                convergence_delta=0.0,
                error_count=self._safe_int(normalized.get("cumulative_errors", 0)),
            )
        )


__all__ = [
    "AutoGenAdapter",
    "BaseAdapter",
    "CrewAIAdapter",
    "DSPyAdapter",
    "HaystackAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "SignalAdapter",
    "TelemetryAdapter",
]
