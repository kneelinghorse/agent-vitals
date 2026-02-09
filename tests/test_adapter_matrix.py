"""Scenario matrix tests across built-in signal adapters."""

from __future__ import annotations

from typing import Any

import pytest

from agent_vitals.adapters import (
    AutoGenAdapter,
    CrewAIAdapter,
    DSPyAdapter,
    HaystackAdapter,
    LangChainAdapter,
    LangGraphAdapter,
)


class _Crew:
    def __init__(self, usage_metrics: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        self.usage_metrics = usage_metrics
        self.tasks = tasks


class _CrewOutput:
    def __init__(self, tasks_output: list[dict[str, Any]]) -> None:
        self.tasks_output = tasks_output


class _Agent:
    def __init__(self, usage: dict[str, Any], chat_messages: list[dict[str, Any]]) -> None:
        self._usage = usage
        self.chat_messages = chat_messages

    def get_total_usage(self) -> dict[str, Any]:
        return dict(self._usage)


@pytest.mark.parametrize(
    ("adapter", "scenario", "state"),
    [
        (
            LangChainAdapter(),
            "healthy",
            {
                "coverage_score": 0.8,
                "outputs": ["f1", "f2", "f3"],
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 70, "total_tokens": 170},
                "errors": [],
                "intermediate_steps": [("search", "ok"), ("analyze", "ok")],
            },
        ),
        (
            LangChainAdapter(),
            "loop",
            {
                "coverage_score": 0.2,
                "outputs": ["same", "same"],
                "token_usage": {"total_tokens": 120},
            },
        ),
        (
            LangChainAdapter(),
            "stuck",
            {
                "coverage_score": 0.1,
                "outputs": [],
                "token_usage": {"total_tokens": 90},
            },
        ),
        (
            LangChainAdapter(),
            "thrash",
            {
                "coverage_score": 0.3,
                "outputs": ["partial"],
                "token_usage": {"total_tokens": 150},
                "errors": ["timeout", "retry failed"],
            },
        ),
        (
            LangGraphAdapter(),
            "healthy",
            {
                "findings": ["f1", "f2", "f3", "f4"],
                "mission_objectives": ["o1", "o2", "o3", "o4"],
                "covered_objectives": ["o1", "o2", "o3"],
                "token_usage": {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
            },
        ),
        (
            LangGraphAdapter(),
            "loop",
            {
                "findings": ["f1", "f1"],
                "mission_objectives": ["o1", "o2", "o3", "o4"],
                "covered_objectives": ["o1"],
                "total_tokens": 100,
            },
        ),
        (
            LangGraphAdapter(),
            "stuck",
            {
                "findings": [],
                "mission_objectives": ["o1", "o2", "o3", "o4"],
                "covered_objectives": [],
                "total_tokens": 100,
            },
        ),
        (
            LangGraphAdapter(),
            "thrash",
            {
                "findings": ["partial"],
                "coverage_score": 0.25,
                "errors": ["tool error"],
                "total_tokens": 130,
            },
        ),
        (
            CrewAIAdapter(),
            "healthy",
            {
                "crew": _Crew(
                    usage_metrics={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
                    tasks=[{"status": "completed"}, {"status": "completed"}, {"status": "done"}],
                ),
                "crew_output": _CrewOutput(tasks_output=[{"r": 1}, {"r": 2}, {"r": 3}]),
            },
        ),
        (
            CrewAIAdapter(),
            "loop",
            {
                "crew": _Crew(
                    usage_metrics={"total_tokens": 120},
                    tasks=[{"status": "completed"}, {"status": "pending"}, {"status": "pending"}],
                ),
                "task_outputs": [{"r": "same"}, {"r": "same"}],
            },
        ),
        (
            CrewAIAdapter(),
            "stuck",
            {
                "crew": _Crew(
                    usage_metrics={"total_tokens": 80},
                    tasks=[{"status": "pending"}, {"status": "pending"}],
                ),
                "task_outputs": [],
            },
        ),
        (
            CrewAIAdapter(),
            "thrash",
            {
                "crew": _Crew(
                    usage_metrics={"total_tokens": 140},
                    tasks=[{"status": "failed"}, {"status": "error"}, {"status": "completed"}],
                ),
                "task_outputs": [{"error": "timeout"}],
            },
        ),
        (
            AutoGenAdapter(),
            "healthy",
            {
                "agent": _Agent(
                    usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
                    chat_messages=[{"role": "user"}, {"role": "assistant"}, {"role": "assistant"}],
                ),
                "total_turns": 4,
            },
        ),
        (
            AutoGenAdapter(),
            "loop",
            {
                "usage_summary": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
                "chat_messages": [{"content": "same"}, {"content": "same"}],
                "total_turns": 8,
            },
        ),
        (
            AutoGenAdapter(),
            "stuck",
            {
                "usage_summary": {"total_tokens": 50},
                "chat_messages": [],
                "coverage_score": 0.05,
            },
        ),
        (
            AutoGenAdapter(),
            "thrash",
            {
                "usage_summary": {"total_tokens": 90},
                "chat_messages": [{"failed": True}, {"error": "network"}],
                "outputs": [{"status": "error"}],
                "coverage_score": 0.2,
            },
        ),
        # --- DSPy scenarios ---
        (
            DSPyAdapter(),
            "healthy",
            {
                "lm_usage": {
                    "openai/gpt-4o-mini": {
                        "prompt_tokens": 800,
                        "completion_tokens": 300,
                        "total_tokens": 1100,
                    },
                },
                "predictions": ["summary A", "analysis B", "synthesis C"],
                "modules_completed": 3,
                "modules_total": 4,
                "errors": [],
            },
        ),
        (
            DSPyAdapter(),
            "loop",
            {
                "lm_usage": {
                    "openai/gpt-4o-mini": {"total_tokens": 600},
                },
                "predictions": ["same output", "same output"],
                "modules_completed": 1,
                "modules_total": 5,
                "coverage_score": 0.2,
            },
        ),
        (
            DSPyAdapter(),
            "stuck",
            {
                "lm_usage": {
                    "openai/gpt-4o-mini": {"total_tokens": 400},
                },
                "predictions": [],
                "modules_completed": 0,
                "modules_total": 4,
            },
        ),
        (
            DSPyAdapter(),
            "thrash",
            {
                "lm_usage": {
                    "openai/gpt-4o-mini": {"total_tokens": 700},
                },
                "predictions": ["partial result"],
                "modules_completed": 1,
                "modules_total": 4,
                "coverage_score": 0.25,
                "errors": ["rate_limit", "timeout", "parse_error"],
            },
        ),
        # --- Haystack scenarios ---
        (
            HaystackAdapter(),
            "healthy",
            {
                "messages": [
                    {"role": "user", "content": "Research topic"},
                    {
                        "role": "assistant",
                        "content": "Finding A about the topic.",
                        "_meta": {"usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}},
                    },
                    {
                        "role": "assistant",
                        "content": "Finding B with more detail.",
                        "_meta": {"usage": {"prompt_tokens": 250, "completion_tokens": 100, "total_tokens": 350}},
                    },
                ],
                "state": {"coverage_score": 0.75},
                "sources": [
                    {"url": "https://example.com/page1"},
                    {"url": "https://other.org/doc"},
                ],
                "errors": [],
            },
        ),
        (
            HaystackAdapter(),
            "loop",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Repeated answer",
                        "_meta": {"usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}},
                    },
                    {
                        "role": "assistant",
                        "content": "Repeated answer",
                        "_meta": {"usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}},
                    },
                ],
                "state": {"coverage_score": 0.15},
            },
        ),
        (
            HaystackAdapter(),
            "stuck",
            {
                "messages": [
                    {"role": "user", "content": "Do research"},
                ],
                "state": {"coverage_score": 0.05},
                "total_tokens": 80,
            },
        ),
        (
            HaystackAdapter(),
            "thrash",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Partial result before error",
                        "_meta": {"usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}},
                    },
                ],
                "state": {"coverage_score": 0.2},
                "errors": ["connection_timeout", "api_error"],
            },
        ),
    ],
)
def test_adapter_scenario_matrix(adapter: Any, scenario: str, state: dict[str, Any]) -> None:
    signals = adapter.extract(state)

    assert signals.total_tokens >= 0
    assert 0.0 <= signals.coverage_score <= 1.0
    assert signals.findings_count >= 0
    assert signals.error_count >= 0

    if scenario == "healthy":
        assert signals.coverage_score >= 0.5
        assert signals.error_count == 0
    elif scenario == "loop":
        assert signals.coverage_score <= 0.4
    elif scenario == "stuck":
        assert signals.coverage_score <= 0.2
    elif scenario == "thrash":
        assert signals.error_count >= 1
