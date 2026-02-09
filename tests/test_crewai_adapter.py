"""Unit tests for CrewAI adapter signal extraction."""

from __future__ import annotations

import importlib
from typing import Any

from agent_vitals.adapters import CrewAIAdapter, SignalAdapter


class _Crew:
    def __init__(self, usage_metrics: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        self.usage_metrics = usage_metrics
        self.tasks = tasks


class _CrewOutput:
    def __init__(self, tasks_output: list[dict[str, Any]]) -> None:
        self.tasks_output = tasks_output


def test_crewai_adapter_conforms_signal_protocol() -> None:
    assert isinstance(CrewAIAdapter(), SignalAdapter)


def test_crewai_adapter_extracts_tokens_findings_errors_and_coverage_from_objects() -> None:
    adapter = CrewAIAdapter()
    state = {
        "crew": _Crew(
            usage_metrics={
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
            tasks=[
                {"status": "completed"},
                {"status": "failed"},
                {"status": "done"},
            ],
        ),
        "crew_output": _CrewOutput(
            tasks_output=[
                {"result": "finding-a"},
                {"result": "finding-b"},
            ]
        ),
    }

    signals = adapter.extract(state)
    assert signals.prompt_tokens == 120
    assert signals.completion_tokens == 30
    assert signals.total_tokens == 150
    assert signals.findings_count == 2
    assert signals.error_count == 1
    assert signals.coverage_score == 2 / 3


def test_crewai_adapter_extracts_from_mapping_and_step_callback() -> None:
    adapter = CrewAIAdapter()
    state = {
        "usage_metrics": {"input_tokens": 50, "output_tokens": 20},
        "tasks": [
            {"state": "completed"},
            {"state": "completed"},
            {"state": "error"},
        ],
        "task_outputs": [{"result": "a"}, {"result": "b"}],
        "step_callback": [
            {"status": "error", "error": "timeout"},
            {"status": "completed"},
        ],
    }

    signals = adapter.extract(state)
    assert signals.prompt_tokens == 50
    assert signals.completion_tokens == 20
    assert signals.total_tokens == 70
    assert signals.findings_count == 2
    assert signals.error_count == 2
    assert signals.coverage_score == 2 / 3
    assert signals.api_calls == 2
    assert signals.query_count == 2


def test_crewai_adapter_honors_explicit_overrides() -> None:
    adapter = CrewAIAdapter()
    state = {
        "usage_metrics": {"total_tokens": 99},
        "findings_count": 5,
        "coverage_score": 0.91,
        "error_count": 7,
        "api_calls": 12,
    }

    signals = adapter.extract(state)
    assert signals.findings_count == 5
    assert signals.coverage_score == 0.91
    assert signals.error_count == 7
    assert signals.api_calls == 12
    assert signals.total_tokens == 99


def test_crewai_adapter_module_imports_without_crewai_dependency() -> None:
    module = importlib.import_module("agent_vitals.adapters.crewai")
    assert hasattr(module, "CrewAIAdapter")
