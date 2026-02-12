"""AgentVitals — Primary public API for agent health monitoring.

This module provides the stateful monitor that tracks agent health
across steps, running the detection engine and maintaining history.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .export import VitalsExporter

from .adapters import SignalAdapter
from .config import ADAPTER_FRAMEWORK_MAP, VitalsConfig, get_vitals_config
from .detection.loop import LoopDetectionResult, detect_loop
from .detection.metrics import TemporalMetrics
from .detection.similarity import compute_output_fingerprint, compute_similarity_scores
from .exceptions import AdapterError
from .schema import (
    HealthState,
    RawSignals,
    TemporalMetricsResult,
    VitalsSnapshot,
)


class AgentVitals:
    """Stateful monitor that tracks agent health across steps.

    Usage::

        monitor = AgentVitals(mission_id="my-task")
        for step in range(max_steps):
            snapshot = monitor.step(
                findings_count=5,
                coverage_score=0.6,
                total_tokens=12000,
                error_count=0,
            )
            if snapshot.any_failure:
                break
    """

    def __init__(
        self,
        *,
        config: Optional[VitalsConfig] = None,
        workflow_type: str = "unknown",
        mission_id: str = "default",
        run_id: Optional[str] = None,
        adapter: Optional[SignalAdapter] = None,
        exporters: Optional[Sequence["VitalsExporter"]] = None,
        framework: Optional[str] = None,
    ) -> None:
        base_config = config or get_vitals_config()

        # Resolve framework: explicit > auto-detect from adapter
        resolved_framework = framework
        if resolved_framework is None and adapter is not None:
            resolved_framework = ADAPTER_FRAMEWORK_MAP.get(type(adapter).__name__)

        # Apply framework-specific threshold profile overrides
        if resolved_framework:
            self._config = base_config.for_framework(resolved_framework)
        else:
            self._config = base_config

        self._framework = resolved_framework
        self._workflow_type = workflow_type
        self._mission_id = mission_id
        self._run_id = run_id or str(uuid.uuid4())
        self._adapter = adapter
        self._exporters: list["VitalsExporter"] = list(exporters or [])
        self._history: list[VitalsSnapshot] = []
        self._loop_index = 0
        self._health_state: HealthState = "healthy"
        self._metrics_engine = TemporalMetrics()
        self._recent_output_texts: list[str] = []

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        *,
        workflow_type: str = "unknown",
        mission_id: str = "default",
        run_id: Optional[str] = None,
        adapter: Optional[SignalAdapter] = None,
        exporters: Optional[Sequence["VitalsExporter"]] = None,
        framework: Optional[str] = None,
    ) -> "AgentVitals":
        """Create an AgentVitals instance from a YAML config file."""
        config = VitalsConfig.from_yaml(Path(yaml_path))
        return cls(
            config=config,
            workflow_type=workflow_type,
            mission_id=mission_id,
            run_id=run_id,
            adapter=adapter,
            exporters=exporters,
            framework=framework,
        )

    def step(
        self,
        *,
        # Minimum viable (4 fields)
        findings_count: int,
        coverage_score: float,
        total_tokens: int,
        error_count: int,
        # Optional enhancing fields
        sources_count: int = 0,
        objectives_covered: int = 0,
        query_count: int = 0,
        unique_domains: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        api_calls: int = 0,
        refinement_count: int = 0,
        convergence_delta: float = 0.0,
        confidence_score: float = 0.0,
        # Content-based similarity detection
        output_text: Optional[str] = None,
    ) -> VitalsSnapshot:
        """Record a step and return the health snapshot.

        Args:
            findings_count: Number of unique findings/outputs so far.
            coverage_score: Progress metric (0.0-1.0).
            total_tokens: Cumulative token usage.
            error_count: Cumulative error count.
            sources_count: Number of sources found.
            objectives_covered: Number of objectives addressed.
            query_count: Number of queries/tool calls.
            unique_domains: Number of unique domains queried.
            prompt_tokens: Prompt token count.
            completion_tokens: Completion token count.
            api_calls: Number of API calls.
            refinement_count: Number of refinement iterations.
            convergence_delta: Change in convergence metric.
            confidence_score: Confidence level (0.0-1.0).
            output_text: Optional agent output text for content-based
                similarity detection. When provided, outputs are compared
                across iterations to detect repetitive looping.

        Returns:
            VitalsSnapshot with detection results and health state.
        """
        signals = RawSignals(
            findings_count=findings_count,
            sources_count=sources_count,
            objectives_covered=objectives_covered,
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            api_calls=api_calls,
            query_count=query_count,
            unique_domains=unique_domains,
            refinement_count=refinement_count,
            convergence_delta=convergence_delta,
            error_count=error_count,
        )
        return self._process_signals(signals, output_text=output_text)

    def step_from_state(
        self,
        state: Mapping[str, Any],
        *,
        output_text: Optional[str] = None,
    ) -> VitalsSnapshot:
        """Use the configured adapter to extract signals from framework state.

        Args:
            state: Framework-specific state mapping.
            output_text: Optional agent output text for similarity detection.

        Returns:
            VitalsSnapshot with detection results.

        Raises:
            AdapterError: If no adapter is configured or extraction fails.
        """
        if self._adapter is None:
            raise AdapterError("No adapter configured. Pass adapter= to AgentVitals().")
        try:
            signals = self._adapter.extract(state)
        except Exception as exc:
            raise AdapterError(f"Adapter extraction failed: {exc}") from exc
        return self._process_signals(signals, output_text=output_text)

    def step_from_signals(
        self,
        signals: RawSignals,
        *,
        output_text: Optional[str] = None,
    ) -> VitalsSnapshot:
        """Step with a pre-built RawSignals object.

        Args:
            signals: Pre-constructed RawSignals instance.
            output_text: Optional agent output text for similarity detection.

        Returns:
            VitalsSnapshot with detection results.
        """
        return self._process_signals(signals, output_text=output_text)

    @property
    def history(self) -> list[VitalsSnapshot]:
        """Return the list of all snapshots recorded so far."""
        return list(self._history)

    @property
    def loop_index(self) -> int:
        """Return the current loop index (0-based)."""
        return self._loop_index

    @property
    def health_state(self) -> HealthState:
        """Return the current health state."""
        return self._health_state

    @property
    def config(self) -> VitalsConfig:
        """Return the active configuration."""
        return self._config

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the monitoring session."""
        any_loop = any(s.loop_detected for s in self._history)
        any_stuck = any(s.stuck_detected for s in self._history)
        first_detection: Optional[int] = None
        for s in self._history:
            if s.loop_detected or s.stuck_detected:
                first_detection = s.loop_index
                break

        return {
            "mission_id": self._mission_id,
            "run_id": self._run_id,
            "total_steps": len(self._history),
            "health_state": self._health_state,
            "any_loop_detected": any_loop,
            "any_stuck_detected": any_stuck,
            "first_detection_at": first_detection,
        }

    def reset(self) -> None:
        """Clear history and reset state for a new run.

        Flushes and closes all configured exporters before clearing.
        """
        self._flush_and_close_exporters()
        self._history.clear()
        self._recent_output_texts.clear()
        self._loop_index = 0
        self._health_state = "healthy"
        self._run_id = str(uuid.uuid4())

    def __enter__(self) -> "AgentVitals":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._flush_and_close_exporters()

    def _flush_and_close_exporters(self) -> None:
        """Flush and close all exporters, logging any errors."""
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        for exp in self._exporters:
            try:
                exp.flush()
            except Exception as exc:
                _logger.warning("Exporter flush failed: %s", exc)
            try:
                exp.close()
            except Exception as exc:
                _logger.warning("Exporter close failed: %s", exc)

    def _process_signals(
        self,
        signals: RawSignals,
        *,
        output_text: Optional[str] = None,
    ) -> VitalsSnapshot:
        """Core processing: compute metrics, run detection, build snapshot."""
        # Compute temporal metrics from history
        coverage_series = [float(s.signals.coverage_score) for s in self._history] + [
            float(signals.coverage_score)
        ]
        findings_series = [float(s.signals.findings_count) for s in self._history] + [
            float(signals.findings_count)
        ]
        token_series = [float(s.signals.total_tokens) for s in self._history] + [
            float(signals.total_tokens)
        ]

        cv_coverage = self._metrics_engine.coefficient_of_variation(coverage_series)
        cv_findings = self._metrics_engine.coefficient_of_variation(findings_series)
        dm_coverage = self._metrics_engine.directional_momentum(coverage_series)
        dm_findings = self._metrics_engine.directional_momentum(findings_series)

        # QPF tokens: ratio of prompt to total tokens (fairness proxy)
        total = float(signals.total_tokens)
        qpf_tokens = float(signals.prompt_tokens) / total if total > 0 else 0.0
        qpf_tokens = max(0.0, min(1.0, qpf_tokens))

        # Crescendo symmetry: are tokens increasing over time? (simplified)
        cs_effort = 0.0
        if len(token_series) >= 3:
            deltas = [token_series[i] - token_series[i - 1] for i in range(1, len(token_series))]
            positive = sum(1 for d in deltas if d > 0)
            cs_effort = max(0.0, min(1.0, positive / len(deltas)))

        metrics = TemporalMetricsResult(
            cv_coverage=cv_coverage,
            cv_findings_rate=cv_findings,
            dm_coverage=dm_coverage,
            dm_findings=dm_findings,
            qpf_tokens=qpf_tokens,
            cs_effort=cs_effort,
        )

        # Run hysteresis for health state
        hysteresis_cfg = self._config.hysteresis_config()
        new_health, health_changed = self._metrics_engine.temporal_hysteresis(
            float(signals.coverage_score),
            self._health_state,
            hysteresis_cfg,
        )
        previous_health = self._health_state if health_changed else None

        # Compute output similarity if text provided
        output_fingerprint: Optional[str] = None
        output_similarity: Optional[float] = None
        if output_text is not None:
            output_fingerprint = compute_output_fingerprint(output_text)
            if self._recent_output_texts:
                sim_result = compute_similarity_scores(
                    output_text,
                    self._recent_output_texts,
                    threshold=float(self._config.loop_similarity_threshold),
                )
                output_similarity = sim_result.max_similarity

        # Build snapshot (without detection results yet, but WITH similarity)
        snapshot = VitalsSnapshot(
            mission_id=self._mission_id,
            run_id=self._run_id,
            loop_index=self._loop_index,
            signals=signals,
            metrics=metrics,
            health_state=new_health,
            health_state_changed=health_changed,
            previous_health_state=previous_health,
            output_similarity=output_similarity,
            output_fingerprint=output_fingerprint,
        )

        # Run loop/stuck detection
        detection: LoopDetectionResult = detect_loop(
            snapshot,
            self._history,
            config=self._config,
            workflow_type=self._workflow_type,
        )

        # Rebuild snapshot with detection results
        snapshot = VitalsSnapshot(
            mission_id=self._mission_id,
            run_id=self._run_id,
            loop_index=self._loop_index,
            signals=signals,
            metrics=metrics,
            health_state=new_health,
            health_state_changed=health_changed,
            previous_health_state=previous_health,
            output_similarity=output_similarity,
            output_fingerprint=output_fingerprint,
            loop_detected=detection.loop_detected,
            loop_confidence=detection.loop_confidence,
            loop_trigger=detection.loop_trigger,
            stuck_detected=detection.stuck_detected,
            stuck_confidence=detection.stuck_confidence,
            stuck_trigger=detection.stuck_trigger,
        )

        # Update state
        self._health_state = new_health
        self._history.append(snapshot)
        self._loop_index += 1

        # Update output text sliding window (keep bounded)
        if output_text is not None:
            self._recent_output_texts.append(output_text)
            max_window = max(1, self._config.history_size)
            if len(self._recent_output_texts) > max_window:
                self._recent_output_texts = self._recent_output_texts[-max_window:]

        # Export snapshot to all configured exporters
        for exporter in self._exporters:
            try:
                exporter.export(snapshot)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Exporter %s.export() failed: %s", type(exporter).__name__, exc
                )

        return snapshot


__all__ = ["AgentVitals"]
