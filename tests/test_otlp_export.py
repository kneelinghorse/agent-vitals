"""Unit tests for OTLP exporter with mocked OpenTelemetry runtime."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from agent_vitals.export import OTLPExporter, VitalsExporter
from agent_vitals.export import otlp as otlp_module
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


def _make_snapshot(
    *,
    mission_id: str = "mission-1",
    run_id: str = "run-1",
    loop_index: int = 0,
    coverage_score: float = 0.5,
    confidence_score: float = 0.3,
    total_tokens: int = 1000,
    error_count: int = 0,
    api_calls: int = 2,
    loop_detected: bool = False,
    stuck_detected: bool = False,
) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id=mission_id,
        run_id=run_id,
        loop_index=loop_index,
        signals=RawSignals(
            findings_count=3,
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            total_tokens=total_tokens,
            error_count=error_count,
            api_calls=api_calls,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.12,
            cv_findings_rate=0.23,
            dm_coverage=0.34,
            dm_findings=0.45,
            qpf_tokens=0.56,
            cs_effort=0.67,
        ),
        health_state="healthy",
        loop_detected=loop_detected,
        loop_confidence=0.7 if loop_detected else 0.0,
        stuck_detected=stuck_detected,
        stuck_confidence=0.8 if stuck_detected else 0.0,
    )


def _install_fake_otel(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    class CounterType: ...

    class UpDownCounterType: ...

    class HistogramType: ...

    class ObservableCounterType: ...

    class ObservableUpDownCounterType: ...

    class GaugeType: ...

    class ObservableGaugeType: ...

    class FakeAggregationTemporality:
        DELTA = "delta"
        CUMULATIVE = "cumulative"

    class FakeGauge:
        def __init__(self, name: str) -> None:
            self.name = name
            self.set_calls: list[tuple[float, dict[str, str]]] = []

        def set(
            self,
            value: float,
            *,
            attributes: Mapping[str, str] | None = None,
        ) -> None:
            self.set_calls.append((float(value), dict(attributes or {})))

    class FakeCounter:
        def __init__(self, name: str) -> None:
            self.name = name
            self.add_calls: list[tuple[int, dict[str, str]]] = []

        def add(
            self,
            value: int,
            *,
            attributes: Mapping[str, str] | None = None,
        ) -> None:
            self.add_calls.append((int(value), dict(attributes or {})))

    class FakeMeter:
        def __init__(self) -> None:
            self.gauges: dict[str, FakeGauge] = {}
            self.counters: dict[str, FakeCounter] = {}

        def create_gauge(
            self,
            name: str,
            *,
            unit: str | None = None,
            description: str | None = None,
        ) -> FakeGauge:
            _ = (unit, description)
            gauge = FakeGauge(name)
            self.gauges[name] = gauge
            return gauge

        def create_counter(
            self,
            name: str,
            *,
            unit: str | None = None,
            description: str | None = None,
        ) -> FakeCounter:
            _ = (unit, description)
            counter = FakeCounter(name)
            self.counters[name] = counter
            return counter

    class FakeOTLPMetricExporter:
        instances: list["FakeOTLPMetricExporter"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            type(self).instances.append(self)

    class FakePeriodicExportingMetricReader:
        def __init__(
            self,
            *,
            exporter: Any,
            export_interval_millis: int,
        ) -> None:
            self.exporter = exporter
            self.export_interval_millis = export_interval_millis

    class FakeResource:
        def __init__(self, attributes: Mapping[str, str]) -> None:
            self.attributes = dict(attributes)

        @classmethod
        def create(cls, attributes: Mapping[str, str]) -> "FakeResource":
            return cls(attributes)

    class FakeMeterProvider:
        instances: list["FakeMeterProvider"] = []

        def __init__(self, *, metric_readers: list[Any], resource: FakeResource) -> None:
            self.metric_readers = metric_readers
            self.resource = resource
            self.meter = FakeMeter()
            self.force_flush_calls = 0
            self.shutdown_calls = 0
            type(self).instances.append(self)

        def get_meter(self, name: str, version: str) -> FakeMeter:
            _ = (name, version)
            return self.meter

        def force_flush(self) -> None:
            self.force_flush_calls += 1

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    components = otlp_module._OTelComponents(
        meter_provider_cls=FakeMeterProvider,
        periodic_reader_cls=FakePeriodicExportingMetricReader,
        otlp_metric_exporter_cls=FakeOTLPMetricExporter,
        resource_cls=FakeResource,
        aggregation_temporality=FakeAggregationTemporality,
        counter_cls=CounterType,
        up_down_counter_cls=UpDownCounterType,
        histogram_cls=HistogramType,
        observable_counter_cls=ObservableCounterType,
        observable_up_down_counter_cls=ObservableUpDownCounterType,
        gauge_cls=GaugeType,
        observable_gauge_cls=ObservableGaugeType,
    )
    monkeypatch.setattr(otlp_module, "_load_otel_components", lambda: components)

    return {
        "provider": FakeMeterProvider,
        "otlp_exporter": FakeOTLPMetricExporter,
        "counter_type": CounterType,
    }


def test_otlp_exporter_satisfies_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_otel(monkeypatch)
    exporter = OTLPExporter()
    assert isinstance(exporter, VitalsExporter)


def test_otlp_exporter_configures_otlp_reader_and_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_otel(monkeypatch)

    OTLPExporter(
        endpoint="https://otlp.example.com/v1/metrics",
        headers={"Authorization": "Bearer token"},
        service_name="agent-vitals-tests",
        mission_id="mission-42",
        run_id="run-99",
        workflow_type="build",
        export_interval_ms=2345,
        delta_temporality=True,
    )

    provider = fake["provider"].instances[0]
    reader = provider.metric_readers[0]
    otlp_exporter = fake["otlp_exporter"].instances[0]

    assert reader.export_interval_millis == 2345
    assert otlp_exporter.kwargs["endpoint"] == "https://otlp.example.com/v1/metrics"
    assert otlp_exporter.kwargs["headers"] == {"Authorization": "Bearer token"}

    preferred_temporality = otlp_exporter.kwargs["preferred_temporality"]
    assert preferred_temporality[fake["counter_type"]] == "delta"

    assert provider.resource.attributes["service.name"] == "agent-vitals-tests"
    assert provider.resource.attributes["mission_id"] == "mission-42"
    assert provider.resource.attributes["run_id"] == "run-99"
    assert provider.resource.attributes["workflow_type"] == "build"


def test_otlp_exporter_records_expected_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_otel(monkeypatch)
    exporter = OTLPExporter(workflow_type="research")

    provider = fake["provider"].instances[0]
    meter = provider.meter

    first = _make_snapshot(
        loop_index=0,
        coverage_score=0.4,
        total_tokens=1000,
        error_count=1,
        api_calls=3,
        loop_detected=True,
    )
    second = _make_snapshot(
        loop_index=2,
        coverage_score=0.9,
        total_tokens=1500,
        error_count=2,
        api_calls=5,
        stuck_detected=True,
    )

    exporter.export(first)
    exporter.export(second)

    coverage_calls = meter.gauges["agent_vitals.coverage_score"].set_calls
    assert [value for value, _ in coverage_calls] == [0.4, 0.9]
    assert coverage_calls[-1][1]["workflow_type"] == "research"
    assert coverage_calls[-1][1]["mission_id"] == "mission-1"

    loop_flag_calls = meter.gauges["agent_vitals.loop_detected"].set_calls
    stuck_flag_calls = meter.gauges["agent_vitals.stuck_detected"].set_calls
    assert [value for value, _ in loop_flag_calls] == [1.0, 0.0]
    assert [value for value, _ in stuck_flag_calls] == [0.0, 1.0]

    total_tokens_calls = meter.counters["agent_vitals.total_tokens"].add_calls
    error_count_calls = meter.counters["agent_vitals.error_count"].add_calls
    loop_index_calls = meter.counters["agent_vitals.loop_index"].add_calls
    api_calls_calls = meter.counters["agent_vitals.api_calls"].add_calls

    assert [value for value, _ in total_tokens_calls] == [1000, 500]
    assert [value for value, _ in error_count_calls] == [1, 1]
    assert [value for value, _ in loop_index_calls] == [1, 2]
    assert [value for value, _ in api_calls_calls] == [3, 2]


def test_otlp_exporter_flush_close_and_post_close_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_otel(monkeypatch)
    exporter = OTLPExporter()
    provider = fake["provider"].instances[0]

    exporter.flush()
    exporter.close()
    exporter.close()
    exporter.export(_make_snapshot())

    assert provider.force_flush_calls == 2
    assert provider.shutdown_calls == 1
    assert provider.meter.counters["agent_vitals.total_tokens"].add_calls == []


def test_otlp_exporter_missing_otel_dependency_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_import(name: str) -> Any:  # noqa: ARG001
        raise ModuleNotFoundError("opentelemetry not installed")

    monkeypatch.setattr(otlp_module.importlib, "import_module", _missing_import)

    with pytest.raises(ImportError, match=r"agent-vitals\[otlp\]"):
        OTLPExporter()
