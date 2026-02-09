"""OTLP exporter for Agent Vitals snapshots.

This exporter sends Agent Vitals metrics via OpenTelemetry OTLP HTTP.
OpenTelemetry dependencies are optional and only imported at runtime.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..schema import VitalsSnapshot

logger = logging.getLogger(__name__)

_MISSING_OTEL_MESSAGE = (
    "OpenTelemetry OTLP dependencies are not installed. "
    "Install with: pip install 'agent-vitals[otlp]'"
)


@dataclass(frozen=True, slots=True)
class _OTelComponents:
    """Container for dynamically imported OpenTelemetry components."""

    meter_provider_cls: type[Any]
    periodic_reader_cls: type[Any]
    otlp_metric_exporter_cls: type[Any]
    resource_cls: type[Any]
    aggregation_temporality: Any
    counter_cls: type[Any] | None
    up_down_counter_cls: type[Any] | None
    histogram_cls: type[Any] | None
    observable_counter_cls: type[Any] | None
    observable_up_down_counter_cls: type[Any] | None
    gauge_cls: type[Any] | None
    observable_gauge_cls: type[Any] | None


def _load_otel_components() -> _OTelComponents:
    """Load optional OpenTelemetry modules and resolve required classes."""
    try:
        exporter_module = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.metric_exporter"
        )
        sdk_metrics_module = importlib.import_module("opentelemetry.sdk.metrics")
        sdk_export_module = importlib.import_module("opentelemetry.sdk.metrics.export")
        sdk_resources_module = importlib.import_module("opentelemetry.sdk.resources")
        metrics_api_module = importlib.import_module("opentelemetry.metrics")
    except Exception as exc:  # pragma: no cover - tested via monkeypatch
        raise ImportError(_MISSING_OTEL_MESSAGE) from exc

    try:
        return _OTelComponents(
            meter_provider_cls=getattr(sdk_metrics_module, "MeterProvider"),
            periodic_reader_cls=getattr(sdk_export_module, "PeriodicExportingMetricReader"),
            otlp_metric_exporter_cls=getattr(exporter_module, "OTLPMetricExporter"),
            resource_cls=getattr(sdk_resources_module, "Resource"),
            aggregation_temporality=getattr(sdk_export_module, "AggregationTemporality"),
            counter_cls=getattr(metrics_api_module, "Counter", None),
            up_down_counter_cls=getattr(metrics_api_module, "UpDownCounter", None),
            histogram_cls=getattr(metrics_api_module, "Histogram", None),
            observable_counter_cls=getattr(metrics_api_module, "ObservableCounter", None),
            observable_up_down_counter_cls=getattr(
                metrics_api_module, "ObservableUpDownCounter", None
            ),
            gauge_cls=getattr(metrics_api_module, "Gauge", None),
            observable_gauge_cls=getattr(metrics_api_module, "ObservableGauge", None),
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(_MISSING_OTEL_MESSAGE) from exc


class OTLPExporter:
    """Export vitals snapshots as OpenTelemetry metrics over OTLP HTTP.

    Args:
        endpoint: OTLP HTTP metrics endpoint (usually ``.../v1/metrics``).
        headers: Additional request headers (API keys, auth, tenant IDs).
        service_name: Value for ``service.name`` resource attribute.
        mission_id: Value for ``mission_id`` resource attribute.
        run_id: Value for ``run_id`` resource attribute.
        workflow_type: Value for ``workflow_type`` resource attribute.
        export_interval_ms: Export interval for periodic metric reader.
        delta_temporality: Enable delta temporality for counter-like instruments.
    """

    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:4318/v1/metrics",
        headers: Mapping[str, str] | None = None,
        service_name: str = "agent-vitals",
        mission_id: str = "unknown",
        run_id: str = "unknown",
        workflow_type: str = "unknown",
        export_interval_ms: int = 5000,
        delta_temporality: bool = False,
    ) -> None:
        components = _load_otel_components()

        self._workflow_type = workflow_type or "unknown"
        self._closed = False
        self._counter_state: dict[str, dict[str, int]] = {}

        resource_attributes = {
            "service.name": service_name or "agent-vitals",
            "mission_id": mission_id or "unknown",
            "run_id": run_id or "unknown",
            "workflow_type": self._workflow_type,
        }
        resource = components.resource_cls.create(resource_attributes)

        exporter_kwargs: dict[str, Any] = {"endpoint": endpoint}
        if headers:
            exporter_kwargs["headers"] = dict(headers)

        preferred_temporality = self._build_temporality_map(
            components,
            delta_temporality=delta_temporality,
        )
        if preferred_temporality:
            exporter_kwargs["preferred_temporality"] = preferred_temporality

        otlp_exporter = components.otlp_metric_exporter_cls(**exporter_kwargs)
        self._metric_reader = components.periodic_reader_cls(
            exporter=otlp_exporter,
            export_interval_millis=max(1, int(export_interval_ms)),
        )
        self._meter_provider = components.meter_provider_cls(
            metric_readers=[self._metric_reader],
            resource=resource,
        )
        self._meter = self._meter_provider.get_meter("agent_vitals.otlp", "1.0.0")

        self._gauges = {
            "coverage_score": self._meter.create_gauge(
                "agent_vitals.coverage_score",
                unit="1",
                description="Mission coverage score [0,1]",
            ),
            "confidence_score": self._meter.create_gauge(
                "agent_vitals.confidence_score",
                unit="1",
                description="Model confidence score [0,1]",
            ),
            "loop_detected": self._meter.create_gauge(
                "agent_vitals.loop_detected",
                unit="1",
                description="Loop detection flag (0 or 1)",
            ),
            "stuck_detected": self._meter.create_gauge(
                "agent_vitals.stuck_detected",
                unit="1",
                description="Stuck detection flag (0 or 1)",
            ),
            "loop_confidence": self._meter.create_gauge(
                "agent_vitals.loop_confidence",
                unit="1",
                description="Loop detector confidence [0,1]",
            ),
            "stuck_confidence": self._meter.create_gauge(
                "agent_vitals.stuck_confidence",
                unit="1",
                description="Stuck detector confidence [0,1]",
            ),
            "cv_coverage": self._meter.create_gauge(
                "agent_vitals.cv_coverage",
                unit="1",
                description="Coefficient of variation for coverage",
            ),
            "cv_findings_rate": self._meter.create_gauge(
                "agent_vitals.cv_findings_rate",
                unit="1",
                description="Coefficient of variation for findings rate",
            ),
            "dm_coverage": self._meter.create_gauge(
                "agent_vitals.dm_coverage",
                unit="1",
                description="Directional momentum for coverage",
            ),
            "dm_findings": self._meter.create_gauge(
                "agent_vitals.dm_findings",
                unit="1",
                description="Directional momentum for findings",
            ),
            "qpf_tokens": self._meter.create_gauge(
                "agent_vitals.qpf_tokens",
                unit="1",
                description="Token distribution fairness metric",
            ),
            "cs_effort": self._meter.create_gauge(
                "agent_vitals.cs_effort",
                unit="1",
                description="Crescendo symmetry effort metric",
            ),
        }

        self._counters = {
            "total_tokens": self._meter.create_counter(
                "agent_vitals.total_tokens",
                unit="tokens",
                description="Total tokens consumed",
            ),
            "error_count": self._meter.create_counter(
                "agent_vitals.error_count",
                unit="1",
                description="Total errors observed",
            ),
            "loop_index": self._meter.create_counter(
                "agent_vitals.loop_index",
                unit="1",
                description="Loop progression counter",
            ),
            "api_calls": self._meter.create_counter(
                "agent_vitals.api_calls",
                unit="1",
                description="Total API/tool calls",
            ),
        }

    def export(self, snapshot: VitalsSnapshot) -> None:
        """Emit one snapshot as metric points."""
        if self._closed:
            logger.warning("OTLPExporter.export() called after close(); ignoring.")
            return

        attrs = self._measurement_attributes(snapshot)
        self._record_gauges(snapshot, attrs)
        self._record_counters(snapshot, attrs)

    def flush(self) -> None:
        """Flush pending metric data via the meter provider."""
        if self._closed:
            return
        try:
            self._meter_provider.force_flush()
        except Exception as exc:
            logger.warning("OTLP force_flush failed: %s", exc)

    def close(self) -> None:
        """Flush and shutdown the OpenTelemetry meter provider."""
        if self._closed:
            return
        self.flush()
        try:
            self._meter_provider.shutdown()
        except Exception as exc:
            logger.warning("OTLP shutdown failed: %s", exc)
        self._closed = True

    def _measurement_attributes(self, snapshot: VitalsSnapshot) -> Mapping[str, str]:
        return {
            "mission_id": snapshot.mission_id,
            "run_id": snapshot.run_id or "unknown",
            "workflow_type": self._workflow_type,
            "health_state": snapshot.health_state,
        }

    def _record_gauges(self, snapshot: VitalsSnapshot, attrs: Mapping[str, str]) -> None:
        self._set_gauge(self._gauges["coverage_score"], snapshot.signals.coverage_score, attrs)
        self._set_gauge(self._gauges["confidence_score"], snapshot.signals.confidence_score, attrs)
        self._set_gauge(self._gauges["loop_detected"], 1.0 if snapshot.loop_detected else 0.0, attrs)
        self._set_gauge(
            self._gauges["stuck_detected"],
            1.0 if snapshot.stuck_detected else 0.0,
            attrs,
        )
        self._set_gauge(self._gauges["loop_confidence"], snapshot.loop_confidence, attrs)
        self._set_gauge(self._gauges["stuck_confidence"], snapshot.stuck_confidence, attrs)

        self._set_gauge(self._gauges["cv_coverage"], snapshot.metrics.cv_coverage, attrs)
        self._set_gauge(self._gauges["cv_findings_rate"], snapshot.metrics.cv_findings_rate, attrs)
        self._set_gauge(self._gauges["dm_coverage"], snapshot.metrics.dm_coverage, attrs)
        self._set_gauge(self._gauges["dm_findings"], snapshot.metrics.dm_findings, attrs)
        self._set_gauge(self._gauges["qpf_tokens"], snapshot.metrics.qpf_tokens, attrs)
        self._set_gauge(self._gauges["cs_effort"], snapshot.metrics.cs_effort, attrs)

    def _record_counters(self, snapshot: VitalsSnapshot, attrs: Mapping[str, str]) -> None:
        run_key = snapshot.run_id or "__default__"
        current = {
            "total_tokens": int(snapshot.signals.total_tokens),
            "error_count": int(snapshot.signals.error_count),
            "loop_index": int(snapshot.loop_index) + 1,
            "api_calls": int(snapshot.signals.api_calls),
        }

        previous = self._counter_state.get(run_key)
        self._counter_state[run_key] = current

        for key, value in current.items():
            delta = value if previous is None else value - previous.get(key, 0)
            if delta < 0:
                delta = value
            if delta > 0:
                self._counters[key].add(delta, attributes=attrs)

    @staticmethod
    def _set_gauge(instrument: Any, value: float, attrs: Mapping[str, str]) -> None:
        if hasattr(instrument, "set"):
            instrument.set(float(value), attributes=attrs)
            return
        if hasattr(instrument, "record"):
            instrument.record(float(value), attributes=attrs)
            return
        logger.warning("Gauge instrument %s has no set()/record()", type(instrument).__name__)

    def _build_temporality_map(
        self,
        components: _OTelComponents,
        *,
        delta_temporality: bool,
    ) -> Mapping[type[Any], Any] | None:
        if not delta_temporality:
            return None

        delta = getattr(components.aggregation_temporality, "DELTA", None)
        cumulative = getattr(components.aggregation_temporality, "CUMULATIVE", None)
        if delta is None:
            return None
        if cumulative is None:
            cumulative = delta

        mapping: dict[type[Any], Any] = {}
        for cls in (
            components.counter_cls,
            components.observable_counter_cls,
            components.histogram_cls,
        ):
            if cls is not None:
                mapping[cls] = delta

        for cls in (
            components.up_down_counter_cls,
            components.observable_up_down_counter_cls,
            components.gauge_cls,
            components.observable_gauge_cls,
        ):
            if cls is not None:
                mapping[cls] = cumulative

        return mapping if mapping else None


__all__ = ["OTLPExporter"]
