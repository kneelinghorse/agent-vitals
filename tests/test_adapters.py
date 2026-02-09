"""Unit tests for built-in adapter implementations."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from agent_vitals.adapters import BaseAdapter, LangChainAdapter, LangGraphAdapter, SignalAdapter
from agent_vitals.schema import RawSignals


class _NoopAdapter(BaseAdapter):
    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        _ = self.normalize(state)
        return self.validate(
            RawSignals(
                findings_count=1,
                coverage_score=0.5,
                total_tokens=100,
                error_count=0,
            )
        )


def test_base_adapter_normalize_and_validate() -> None:
    adapter = _NoopAdapter()
    assert adapter.normalize(None) == {}
    signals = adapter.extract({})
    assert signals.findings_count == 1
    assert signals.coverage_score == 0.5


def test_built_in_adapters_conform_signal_protocol() -> None:
    assert isinstance(LangChainAdapter(), SignalAdapter)
    assert isinstance(LangGraphAdapter(), SignalAdapter)


def test_langchain_adapter_extracts_agentexecutor_like_state() -> None:
    adapter = LangChainAdapter()
    state = {
        "cumulative_outputs": 3,
        "coverage_score": 0.7,
        "llm_output": {
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 180,
                "total_tokens": 300,
            }
        },
        "cumulative_errors": 1,
        "intermediate_steps": [("search", "result"), ("read", "result")],
        "source_documents": [
            {"source": "https://docs.example.com/page-1"},
            {"metadata": {"source": "https://api.example.com/reference"}},
        ],
    }

    signals = adapter.extract(state)
    assert signals.findings_count == 3
    assert signals.coverage_score == 0.7
    assert signals.total_tokens == 300
    assert signals.prompt_tokens == 120
    assert signals.completion_tokens == 180
    assert signals.query_count == 2
    assert signals.api_calls == 2
    assert signals.sources_count == 2
    assert signals.unique_domains == 2
    assert signals.error_count == 1


def test_langchain_adapter_fallbacks_and_clamps_coverage() -> None:
    adapter = LangChainAdapter()
    state = {
        "output": "final answer",
        "coverage": 1.5,
        "token_usage": {"total_tokens": 42},
        "errors": ["tool failure", "retry failed"],
    }

    signals = adapter.extract(state)
    assert signals.findings_count == 1
    assert signals.coverage_score == 1.0
    assert signals.total_tokens == 42
    assert signals.error_count == 2


def test_langgraph_adapter_extracts_typed_dict_state() -> None:
    adapter = LangGraphAdapter()
    state = {
        "findings": ["a", "b", "c", "d"],
        "sources_found": [
            {"url": "https://alpha.example.com/r1"},
            {"metadata": {"source": "https://beta.example.com/r2"}},
            {"source": "https://alpha.example.com/r3"},
        ],
        "mission_objectives": ["o1", "o2", "o3", "o4", "o5"],
        "covered_objectives": ["o1", "o2", "o3"],
        "token_usage": {"prompt_tokens": 60, "completion_tokens": 90, "total_tokens": 150},
        "queries": ["q1", "q2", "q3", "q4"],
        "errors": ["timeout"],
        "research_loop_count": 7,
        "delta_coverage": 0.2,
    }

    signals = adapter.extract(state)
    assert signals.findings_count == 4
    assert signals.sources_count == 3
    assert signals.objectives_covered == 3
    assert signals.coverage_score == 0.6
    assert signals.total_tokens == 150
    assert signals.query_count == 4
    assert signals.api_calls == 4
    assert signals.unique_domains == 2
    assert signals.refinement_count == 7
    assert signals.convergence_delta == 0.2
    assert signals.error_count == 1


def test_langgraph_adapter_prefers_explicit_coverage_score() -> None:
    adapter = LangGraphAdapter()
    state = {
        "coverage_score": 0.9,
        "mission_objectives": ["o1", "o2"],
        "covered_objectives": ["o1"],
        "total_tokens": 100,
        "findings_count": 2,
    }

    signals = adapter.extract(state)
    assert signals.coverage_score == 0.9
    assert signals.objectives_covered == 1


def test_adapter_modules_import_without_optional_framework_deps() -> None:
    module_langchain = importlib.import_module("agent_vitals.adapters.langchain")
    module_langgraph = importlib.import_module("agent_vitals.adapters.langgraph")
    assert hasattr(module_langchain, "LangChainAdapter")
    assert hasattr(module_langgraph, "LangGraphAdapter")
