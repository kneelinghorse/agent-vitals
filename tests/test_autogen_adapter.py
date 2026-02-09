"""Unit tests for AutoGen/AG2 adapter signal extraction."""

from __future__ import annotations

import importlib
from typing import Any

from agent_vitals.adapters import AutoGenAdapter, SignalAdapter


class _Agent:
    def __init__(
        self,
        *,
        total_usage: dict[str, Any] | None = None,
        actual_usage: dict[str, Any] | None = None,
        chat_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._total_usage = total_usage or {}
        self._actual_usage = actual_usage or {}
        self.chat_messages = chat_messages or []

    def get_total_usage(self) -> dict[str, Any]:
        return dict(self._total_usage)

    def get_actual_usage(self) -> dict[str, Any]:
        return dict(self._actual_usage)


def test_autogen_adapter_conforms_signal_protocol() -> None:
    assert isinstance(AutoGenAdapter(), SignalAdapter)


def test_autogen_adapter_extracts_single_agent_usage_and_messages() -> None:
    adapter = AutoGenAdapter()
    agent = _Agent(
        total_usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
        chat_messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "assistant", "status": "error", "error": "tool timeout"},
        ],
    )
    state = {
        "agent": agent,
        "max_turns": 4,
    }

    signals = adapter.extract(state)
    assert signals.prompt_tokens == 120
    assert signals.completion_tokens == 80
    assert signals.total_tokens == 200
    assert signals.findings_count == 3
    assert signals.error_count == 1
    assert signals.coverage_score == 0.75
    assert signals.api_calls == 3
    assert signals.query_count == 3


def test_autogen_adapter_extracts_multi_agent_usage_summary() -> None:
    adapter = AutoGenAdapter()
    state = {
        "usage_summary": {
            "agent_a": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "agent_b": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        },
        "chat_messages": [
            {"role": "user"},
            {"role": "assistant"},
            {"role": "user"},
            {"role": "assistant"},
            {"role": "assistant"},
        ],
        "outputs": [{"text": "f1"}, {"text": "f2"}],
        "total_turns": 10,
    }

    signals = adapter.extract(state)
    assert signals.prompt_tokens == 16
    assert signals.completion_tokens == 9
    assert signals.total_tokens == 25
    assert signals.findings_count == 2
    assert signals.coverage_score == 0.5


def test_autogen_adapter_detects_failed_message_indicators() -> None:
    adapter = AutoGenAdapter()
    state = {
        "chat_messages": [
            {"status": "completed"},
            {"failed": True},
            {"error": "network"},
            {"state": "timeout"},
        ],
        "usage_summary": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    signals = adapter.extract(state)
    assert signals.error_count == 3


def test_autogen_adapter_honors_explicit_overrides() -> None:
    adapter = AutoGenAdapter()
    state = {
        "usage_summary": {"total_tokens": 50},
        "findings_count": 7,
        "coverage_score": 0.88,
        "error_count": 2,
        "api_calls": 11,
    }

    signals = adapter.extract(state)
    assert signals.findings_count == 7
    assert signals.coverage_score == 0.88
    assert signals.error_count == 2
    assert signals.api_calls == 11
    assert signals.total_tokens == 50


def test_autogen_adapter_module_imports_without_autogen_dependency() -> None:
    module = importlib.import_module("agent_vitals.adapters.autogen")
    assert hasattr(module, "AutoGenAdapter")
