"""Unit tests for LangSmithAdapter — LangSmith observability platform integration."""

from __future__ import annotations

from typing import Any

import pytest

from agent_vitals.adapters import SignalAdapter
from agent_vitals.adapters.langsmith import LangSmithAdapter
from agent_vitals.schema import RawSignals


@pytest.fixture
def adapter() -> LangSmithAdapter:
    return LangSmithAdapter()


# ------------------------------------------------------------------
# Protocol compliance
# ------------------------------------------------------------------


def test_langsmith_adapter_implements_signal_protocol() -> None:
    assert isinstance(LangSmithAdapter(), SignalAdapter)


def test_extract_returns_raw_signals(adapter: LangSmithAdapter) -> None:
    result = adapter.extract({})
    assert isinstance(result, RawSignals)


# ------------------------------------------------------------------
# Token extraction — usage_metadata
# ------------------------------------------------------------------


def test_tokens_from_usage_metadata(adapter: LangSmithAdapter) -> None:
    state = {
        "usage_metadata": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 200
    assert signals.completion_tokens == 80
    assert signals.total_tokens == 280


def test_tokens_from_usage_metadata_prompt_completion_keys(adapter: LangSmithAdapter) -> None:
    state = {
        "usage_metadata": {"prompt_tokens": 150, "completion_tokens": 60, "total_tokens": 210},
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 150
    assert signals.completion_tokens == 60
    assert signals.total_tokens == 210


def test_tokens_computed_when_total_missing(adapter: LangSmithAdapter) -> None:
    state = {
        "usage_metadata": {"input_tokens": 100, "output_tokens": 50},
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 150


# ------------------------------------------------------------------
# Token extraction — child_runs
# ------------------------------------------------------------------


def test_tokens_from_child_runs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {
                "run_type": "llm",
                "usage_metadata": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
            },
            {
                "run_type": "llm",
                "usage_metadata": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
            },
            {
                "run_type": "tool",
                "usage_metadata": {"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 300
    assert signals.completion_tokens == 120
    assert signals.total_tokens == 420


def test_tokens_skip_non_llm_child_runs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {
                "run_type": "tool",
                "usage_metadata": {"total_tokens": 500},
            },
            {
                "run_type": "retriever",
                "usage_metadata": {"total_tokens": 300},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 0


# ------------------------------------------------------------------
# Token extraction — extra.metadata fallback
# ------------------------------------------------------------------


def test_tokens_from_extra_metadata(adapter: LangSmithAdapter) -> None:
    state = {
        "extra": {
            "metadata": {
                "usage_metadata": {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
            },
        },
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 110


# ------------------------------------------------------------------
# Token extraction — explicit overrides
# ------------------------------------------------------------------


def test_explicit_total_tokens_override(adapter: LangSmithAdapter) -> None:
    state = {
        "total_tokens": 500,
        "prompt_tokens": 300,
        "completion_tokens": 200,
        "usage_metadata": {"total_tokens": 999},
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 500
    assert signals.prompt_tokens == 300


def test_explicit_total_zero_computed_from_parts(adapter: LangSmithAdapter) -> None:
    state = {"total_tokens": 0, "prompt_tokens": 120, "completion_tokens": 80}
    signals = adapter.extract(state)
    assert signals.total_tokens == 200


# ------------------------------------------------------------------
# Findings extraction
# ------------------------------------------------------------------


def test_findings_from_outputs_list(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {
            "outputs": ["Finding A", "Finding B", "Finding C"],
        },
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 3


def test_findings_deduplicates_identical_outputs(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {
            "outputs": ["Same result", "Same result", "Different"],
        },
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 2


def test_findings_from_generations(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {
            "generations": [["gen1"], ["gen2"]],
        },
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 2


def test_findings_from_single_output(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {"output": "Final analysis of the topic"},
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 1


def test_findings_from_text_field(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {"text": "Generated summary"},
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 1


def test_findings_from_child_run_outputs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {"run_type": "llm", "outputs": {"output": "Analysis A"}},
            {"run_type": "llm", "outputs": {"output": "Analysis B"}},
            {"run_type": "llm", "outputs": {"output": "Analysis A"}},  # duplicate
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 2


def test_findings_skips_empty_outputs(adapter: LangSmithAdapter) -> None:
    state = {
        "outputs": {"output": ""},
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 0


def test_findings_explicit_override(adapter: LangSmithAdapter) -> None:
    state = {
        "findings_count": 7,
        "outputs": {"output": "Should be ignored"},
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 7


# ------------------------------------------------------------------
# Coverage extraction
# ------------------------------------------------------------------


def test_coverage_explicit_override(adapter: LangSmithAdapter) -> None:
    state = {"coverage_score": 0.85}
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.85)


def test_coverage_from_feedback_stats_mean(adapter: LangSmithAdapter) -> None:
    state = {
        "feedback_stats": {
            "coverage": {"mean": 0.72, "count": 3},
        },
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.72)


def test_coverage_from_feedback_stats_scalar(adapter: LangSmithAdapter) -> None:
    state = {
        "feedback_stats": {
            "progress": 0.6,
        },
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.6)


def test_coverage_from_extra_metadata(adapter: LangSmithAdapter) -> None:
    state = {
        "extra": {"metadata": {"coverage_score": 0.55}},
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.55)


def test_coverage_clamps_to_01(adapter: LangSmithAdapter) -> None:
    state = {"coverage_score": 1.5}
    signals = adapter.extract(state)
    assert signals.coverage_score == 1.0

    signals2 = adapter.extract({"coverage_score": -0.3})
    assert signals2.coverage_score == 0.0


def test_coverage_defaults_to_zero(adapter: LangSmithAdapter) -> None:
    signals = adapter.extract({})
    assert signals.coverage_score == 0.0


# ------------------------------------------------------------------
# Error extraction
# ------------------------------------------------------------------


def test_errors_from_run_error_field(adapter: LangSmithAdapter) -> None:
    state = {
        "error": "Traceback: ValueError in parsing step",
        "status": "error",
    }
    signals = adapter.extract(state)
    assert signals.error_count == 1


def test_errors_from_status_only(adapter: LangSmithAdapter) -> None:
    state = {"status": "error"}
    signals = adapter.extract(state)
    assert signals.error_count == 1


def test_errors_from_child_runs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {"run_type": "llm", "error": "Model timeout", "status": "error"},
            {"run_type": "llm", "status": "success"},
            {"run_type": "tool", "error": "Connection refused"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 2


def test_errors_from_explicit_errors_list(adapter: LangSmithAdapter) -> None:
    state = {
        "errors": ["timeout", "rate_limit"],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 2


def test_errors_explicit_override(adapter: LangSmithAdapter) -> None:
    state = {
        "error_count": 5,
        "error": "Should not count this",
    }
    signals = adapter.extract(state)
    assert signals.error_count == 5


def test_zero_errors_on_success(adapter: LangSmithAdapter) -> None:
    state = {"status": "success"}
    signals = adapter.extract(state)
    assert signals.error_count == 0


def test_run_error_and_child_errors_combined(adapter: LangSmithAdapter) -> None:
    state = {
        "error": "Chain failed",
        "child_runs": [
            {"run_type": "llm", "error": "LLM timeout"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 2


# ------------------------------------------------------------------
# Query count (LLM run count)
# ------------------------------------------------------------------


def test_query_count_from_llm_child_runs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {"run_type": "llm"},
            {"run_type": "llm"},
            {"run_type": "tool"},
            {"run_type": "retriever"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.query_count == 2


def test_query_count_single_llm_run(adapter: LangSmithAdapter) -> None:
    state = {"run_type": "llm"}
    signals = adapter.extract(state)
    assert signals.query_count == 1


def test_query_count_from_usage_metadata_fallback(adapter: LangSmithAdapter) -> None:
    state = {
        "run_type": "chain",
        "usage_metadata": {"total_tokens": 500},
    }
    signals = adapter.extract(state)
    assert signals.query_count == 1


def test_query_count_explicit_override(adapter: LangSmithAdapter) -> None:
    state = {
        "query_count": 10,
        "child_runs": [{"run_type": "llm"}],
    }
    signals = adapter.extract(state)
    assert signals.query_count == 10


# ------------------------------------------------------------------
# Sources / domains extraction
# ------------------------------------------------------------------


def test_sources_from_explicit_list(adapter: LangSmithAdapter) -> None:
    state = {
        "sources": [
            {"url": "https://example.com/page1"},
            {"url": "https://other.org/doc"},
            {"url": "https://example.com/page2"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 3
    assert signals.unique_domains == 2


def test_sources_from_retriever_child_runs(adapter: LangSmithAdapter) -> None:
    state = {
        "child_runs": [
            {
                "run_type": "retriever",
                "outputs": {
                    "documents": [
                        {"metadata": {"source": "https://docs.python.org/3/"}},
                        {"metadata": {"source": "https://fastapi.tiangolo.com/"}},
                    ],
                },
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 2
    assert signals.unique_domains == 2


def test_sources_from_extra_metadata(adapter: LangSmithAdapter) -> None:
    state = {
        "extra": {
            "metadata": {
                "sources": [{"url": "https://langsmith.dev/docs"}],
            },
        },
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 1
    assert signals.unique_domains == 1


# ------------------------------------------------------------------
# Empty / None state handling
# ------------------------------------------------------------------


def test_empty_state_returns_zeroed_signals(adapter: LangSmithAdapter) -> None:
    signals = adapter.extract({})
    assert signals.findings_count == 0
    assert signals.coverage_score == 0.0
    assert signals.total_tokens == 0
    assert signals.error_count == 0
    assert signals.query_count == 0


def test_none_state_returns_zeroed_signals(adapter: LangSmithAdapter) -> None:
    signals = adapter.extract(None)  # type: ignore[arg-type]
    assert signals.findings_count == 0
    assert signals.total_tokens == 0


# ------------------------------------------------------------------
# Complex / realistic LangSmith state
# ------------------------------------------------------------------


def test_full_langsmith_chain_run(adapter: LangSmithAdapter) -> None:
    """Realistic LangSmith chain run with LLM and retriever children."""
    state = {
        "run_type": "chain",
        "status": "success",
        "outputs": {
            "output": "Comprehensive analysis of WASM adoption trends.",
        },
        "child_runs": [
            {
                "run_type": "retriever",
                "outputs": {
                    "documents": [
                        {"metadata": {"source": "https://fermyon.com/spin"}},
                        {"metadata": {"source": "https://developers.cloudflare.com/workers/"}},
                    ],
                },
            },
            {
                "run_type": "llm",
                "usage_metadata": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
                "outputs": {"output": "WASM is gaining traction."},
            },
            {
                "run_type": "llm",
                "usage_metadata": {"input_tokens": 600, "output_tokens": 250, "total_tokens": 850},
                "outputs": {"output": "Fermyon and Cloudflare lead."},
            },
            {
                "run_type": "tool",
                "outputs": {"result": "search completed"},
            },
        ],
        "feedback_stats": {
            "coverage": {"mean": 0.65, "count": 2},
        },
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 1550
    assert signals.prompt_tokens == 1100
    assert signals.completion_tokens == 450
    assert signals.findings_count == 1  # from top-level output
    assert signals.coverage_score == pytest.approx(0.65)
    assert signals.error_count == 0
    assert signals.query_count == 2  # 2 LLM child runs
    assert signals.sources_count == 2
    assert signals.unique_domains == 2


def test_single_llm_run(adapter: LangSmithAdapter) -> None:
    """LangSmith single LLM run without children."""
    state = {
        "run_type": "llm",
        "usage_metadata": {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
        "outputs": {"output": "Direct LLM response to user query."},
        "status": "success",
        "coverage_score": 0.5,
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 450
    assert signals.findings_count == 1
    assert signals.coverage_score == pytest.approx(0.5)
    assert signals.query_count == 1
    assert signals.error_count == 0
