"""LangChain callback integration for Agent Vitals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..export import JSONLExporter
from ..monitor import AgentVitals
from ..schema import VitalsSnapshot

try:
    from langchain_core.callbacks import BaseCallbackHandler

    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover - exercised via unit tests with monkeypatch
    _HAS_LANGCHAIN = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback shim when langchain_core is unavailable."""

        pass


logger = logging.getLogger(__name__)

FailureMode = Literal["log", "raise", "callback"]
FailureHook = Callable[[VitalsSnapshot], None]


@dataclass(slots=True)
class _Counters:
    findings_count: int = 0
    coverage_score: float = 0.0
    sources_count: int = 0
    query_count: int = 0
    unique_domains: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    refinement_count: int = 0
    convergence_delta: float = 0.0
    error_count: int = 0


class LangChainVitalsCallback(BaseCallbackHandler):
    """Drop-in LangChain callback that feeds signals into AgentVitals."""

    def __init__(
        self,
        *,
        mission_id: str = "langchain",
        workflow_type: str = "langchain",
        on_failure: FailureMode = "log",
        failure_callback: FailureHook | None = None,
        export_jsonl_dir: str | Path | None = None,
        export_otlp: bool = False,
        monitor: AgentVitals | None = None,
    ) -> None:
        super().__init__()

        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is not installed. Install with: pip install agent-vitals[langchain]"
            )
        if on_failure not in {"log", "raise", "callback"}:
            raise ValueError("on_failure must be one of: log, raise, callback")
        if on_failure == "callback" and failure_callback is None:
            raise ValueError("failure_callback is required when on_failure='callback'")

        exporters = []
        if export_jsonl_dir is not None:
            exporters.append(JSONLExporter(directory=Path(export_jsonl_dir), layout="append"))

        self._monitor = monitor or AgentVitals(
            mission_id=mission_id,
            workflow_type=workflow_type,
            exporters=exporters,
        )
        self._on_failure = on_failure
        self._failure_callback = failure_callback
        self._export_otlp = export_otlp
        self._counters = _Counters()
        self._last_snapshot: VitalsSnapshot | None = None

    @property
    def monitor(self) -> AgentVitals:
        """Expose the underlying AgentVitals monitor."""
        return self._monitor

    @property
    def last_snapshot(self) -> VitalsSnapshot | None:
        """Return the most recent snapshot emitted by on_chain_end."""
        return self._last_snapshot

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ARG002
        prompt_tokens, completion_tokens, total_tokens = self._extract_tokens(response)
        self._counters.prompt_tokens += prompt_tokens
        self._counters.completion_tokens += completion_tokens
        self._counters.total_tokens += total_tokens

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:  # noqa: ARG002
        self._counters.error_count += 1

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._counters.query_count += 1
        self._counters.api_calls += 1
        self._counters.sources_count += self._estimate_sources(output)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:  # noqa: ARG002
        self._counters.query_count += 1
        self._counters.api_calls += 1
        self._counters.error_count += 1

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:  # noqa: ARG002
        findings_increment, coverage = self._extract_chain_signals(outputs)
        self._counters.findings_count += findings_increment
        if coverage is not None:
            self._counters.coverage_score = coverage

        snapshot = self._monitor.step(
            findings_count=self._counters.findings_count,
            coverage_score=self._counters.coverage_score,
            total_tokens=self._counters.total_tokens,
            error_count=self._counters.error_count,
            sources_count=self._counters.sources_count,
            objectives_covered=0,
            query_count=self._counters.query_count,
            unique_domains=self._counters.unique_domains,
            prompt_tokens=self._counters.prompt_tokens,
            completion_tokens=self._counters.completion_tokens,
            api_calls=self._counters.api_calls,
            refinement_count=self._counters.refinement_count,
            convergence_delta=self._counters.convergence_delta,
            confidence_score=0.0,
        )
        self._last_snapshot = snapshot
        self._handle_failure(snapshot)

    def _handle_failure(self, snapshot: VitalsSnapshot) -> None:
        if not snapshot.any_failure:
            return

        trigger = snapshot.stuck_trigger or snapshot.loop_trigger or "unknown"
        message = f"AgentVitals failure detected at loop={snapshot.loop_index}: {trigger}"

        if self._on_failure == "log":
            logger.warning(message)
            return

        if self._on_failure == "raise":
            raise RuntimeError(message)

        if self._failure_callback is not None:
            self._failure_callback(snapshot)

    def _extract_tokens(self, response: Any) -> tuple[int, int, int]:
        usage = self._extract_usage_mapping(response)
        prompt_tokens = self._safe_int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0))
        )
        completion_tokens = self._safe_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        total_tokens = self._safe_int(
            usage.get("total_tokens", prompt_tokens + completion_tokens)
        )
        return prompt_tokens, completion_tokens, total_tokens

    def _extract_usage_mapping(self, response: Any) -> Mapping[str, Any]:
        if isinstance(response, Mapping):
            for key in ("token_usage", "usage_metadata", "llm_output"):
                usage = self._coerce_usage(response.get(key))
                if usage:
                    return usage
            return {}

        for attr in ("token_usage", "usage_metadata", "llm_output"):
            usage = self._coerce_usage(getattr(response, attr, None))
            if usage:
                return usage
        return {}

    def _coerce_usage(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        nested = value.get("token_usage")
        if isinstance(nested, Mapping):
            return nested
        return value

    def _extract_chain_signals(self, outputs: Any) -> tuple[int, float | None]:
        if isinstance(outputs, Mapping):
            findings = self._safe_int(outputs.get("findings_count", 0))
            if findings == 0:
                findings = self._safe_len(outputs.get("findings"))
            if findings == 0 and outputs.get("output") not in (None, "", [], {}):
                findings = 1

            coverage = outputs.get("coverage_score", outputs.get("coverage"))
            if coverage is None and isinstance(outputs.get("mission_objectives"), list):
                total = len(outputs["mission_objectives"])
                covered = self._safe_int(
                    outputs.get(
                        "objectives_covered",
                        self._safe_len(outputs.get("covered_objectives")),
                    )
                )
                if total > 0:
                    coverage = covered / total

            parsed_coverage = None
            if coverage is not None:
                parsed_coverage = self._clip01(self._safe_float(coverage, 0.0))
            return findings, parsed_coverage

        if isinstance(outputs, (list, tuple, set)):
            return len(outputs), None
        if outputs in (None, ""):
            return 0, None
        return 1, None

    def _estimate_sources(self, output: Any) -> int:
        if output is None:
            return 0
        if isinstance(output, Mapping):
            for key in ("source_documents", "sources", "documents", "results"):
                if key in output:
                    return self._safe_len(output.get(key))
            return 1
        if isinstance(output, (list, tuple, set)):
            return len(output)
        return 1

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
            return len(value) if value is not None else default
        except TypeError:
            return default

    @staticmethod
    def _clip01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
