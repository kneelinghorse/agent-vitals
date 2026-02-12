"""Unit tests for LangfuseAdapter — Langfuse observability platform integration."""

from __future__ import annotations

import pytest

from agent_vitals.adapters import SignalAdapter
from agent_vitals.adapters.langfuse import LangfuseAdapter
from agent_vitals.schema import RawSignals


@pytest.fixture
def adapter() -> LangfuseAdapter:
    return LangfuseAdapter()


# ------------------------------------------------------------------
# Protocol compliance
# ------------------------------------------------------------------


def test_langfuse_adapter_implements_signal_protocol() -> None:
    assert isinstance(LangfuseAdapter(), SignalAdapter)


def test_extract_returns_raw_signals(adapter: LangfuseAdapter) -> None:
    result = adapter.extract({})
    assert isinstance(result, RawSignals)


# ------------------------------------------------------------------
# Token extraction — observations
# ------------------------------------------------------------------


def test_tokens_from_generation_observations(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {
                "type": "GENERATION",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            {
                "type": "GENERATION",
                "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 300
    assert signals.completion_tokens == 130
    assert signals.total_tokens == 430


def test_tokens_skip_non_generation_observations(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {
                "type": "GENERATION",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            {
                "type": "SPAN",
                "usage": {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 150


def test_tokens_from_usage_details_format(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {
                "type": "GENERATION",
                "usage_details": {"input": 80, "output": 40, "total": 120},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 80
    assert signals.completion_tokens == 40
    assert signals.total_tokens == 120


def test_tokens_from_input_output_keys(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {
                "type": "GENERATION",
                "usage": {"input_tokens": 60, "output_tokens": 30},
            },
        ],
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 60
    assert signals.completion_tokens == 30
    assert signals.total_tokens == 90


# ------------------------------------------------------------------
# Token extraction — flat generations
# ------------------------------------------------------------------


def test_tokens_from_flat_generations(adapter: LangfuseAdapter) -> None:
    state = {
        "generations": [
            {"usage": {"prompt_tokens": 150, "completion_tokens": 70, "total_tokens": 220}},
            {"usage": {"prompt_tokens": 180, "completion_tokens": 90, "total_tokens": 270}},
        ],
    }
    signals = adapter.extract(state)
    assert signals.prompt_tokens == 330
    assert signals.completion_tokens == 160
    assert signals.total_tokens == 490


def test_tokens_from_generation_top_level_keys(adapter: LangfuseAdapter) -> None:
    state = {
        "generations": [
            {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        ],
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 75


# ------------------------------------------------------------------
# Token extraction — explicit overrides
# ------------------------------------------------------------------


def test_explicit_total_tokens_override(adapter: LangfuseAdapter) -> None:
    state = {
        "total_tokens": 500,
        "prompt_tokens": 300,
        "completion_tokens": 200,
        "observations": [
            {"type": "GENERATION", "usage": {"total_tokens": 999}},
        ],
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 500
    assert signals.prompt_tokens == 300
    assert signals.completion_tokens == 200


def test_explicit_total_tokens_computed_from_parts(adapter: LangfuseAdapter) -> None:
    state = {"total_tokens": 0, "prompt_tokens": 120, "completion_tokens": 80}
    signals = adapter.extract(state)
    assert signals.total_tokens == 200


# ------------------------------------------------------------------
# Findings extraction
# ------------------------------------------------------------------


def test_findings_from_generation_outputs(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "output": "Finding A about the topic."},
            {"type": "GENERATION", "output": "Finding B with more detail."},
            {"type": "SPAN", "output": "Should not count"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 2


def test_findings_deduplicates_identical_outputs(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "output": "Same result"},
            {"type": "GENERATION", "output": "Same result"},
            {"type": "GENERATION", "output": "Different result"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 2


def test_findings_from_flat_generations(adapter: LangfuseAdapter) -> None:
    state = {
        "generations": [
            {"output": "Result A"},
            {"output": "Result B"},
            {"output": "Result C"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 3


def test_findings_skips_empty_outputs(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "output": "Valid output"},
            {"type": "GENERATION", "output": ""},
            {"type": "GENERATION", "output": None},
            {"type": "GENERATION", "output": {}},
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 1


def test_findings_from_trace_output(adapter: LangfuseAdapter) -> None:
    state = {
        "trace": {"output": "Final summary of findings"},
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 1


def test_findings_explicit_override(adapter: LangfuseAdapter) -> None:
    state = {
        "findings_count": 7,
        "observations": [
            {"type": "GENERATION", "output": "Should be ignored"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.findings_count == 7


# ------------------------------------------------------------------
# Coverage extraction
# ------------------------------------------------------------------


def test_coverage_explicit_override(adapter: LangfuseAdapter) -> None:
    state = {"coverage_score": 0.85}
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.85)


def test_coverage_from_scores_list(adapter: LangfuseAdapter) -> None:
    state = {
        "scores": [
            {"name": "accuracy", "value": 0.9},
            {"name": "coverage", "value": 0.72},
        ],
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.72)


def test_coverage_from_scores_progress_name(adapter: LangfuseAdapter) -> None:
    state = {
        "scores": [
            {"name": "progress", "value": 0.6},
        ],
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.6)


def test_coverage_from_trace_metadata(adapter: LangfuseAdapter) -> None:
    state = {
        "trace": {
            "metadata": {"coverage_score": 0.55},
        },
    }
    signals = adapter.extract(state)
    assert signals.coverage_score == pytest.approx(0.55)


def test_coverage_clamps_to_01(adapter: LangfuseAdapter) -> None:
    state = {"coverage_score": 1.5}
    signals = adapter.extract(state)
    assert signals.coverage_score == 1.0

    state2 = {"coverage_score": -0.3}
    signals2 = adapter.extract(state2)
    assert signals2.coverage_score == 0.0


def test_coverage_defaults_to_zero(adapter: LangfuseAdapter) -> None:
    signals = adapter.extract({})
    assert signals.coverage_score == 0.0


# ------------------------------------------------------------------
# Error extraction
# ------------------------------------------------------------------


def test_errors_from_observation_level(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "level": "ERROR", "status_message": "Model timeout"},
            {"type": "GENERATION", "level": "DEFAULT"},
            {"type": "SPAN", "level": "ERROR", "status_message": "Connection failed"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 2


def test_errors_from_status_message_without_error_level(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "level": "DEFAULT", "status_message": "Rate limit exceeded"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 1


def test_errors_from_explicit_errors_list(adapter: LangfuseAdapter) -> None:
    state = {
        "errors": ["timeout", "rate_limit", "parse_error"],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 3


def test_errors_from_generation_errors(adapter: LangfuseAdapter) -> None:
    state = {
        "generations": [
            {"level": "ERROR", "status_message": "Model crashed"},
            {"level": "DEFAULT"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 1


def test_errors_explicit_override(adapter: LangfuseAdapter) -> None:
    state = {
        "error_count": 5,
        "observations": [
            {"type": "GENERATION", "level": "ERROR"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 5


def test_zero_errors_on_clean_state(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "level": "DEFAULT"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.error_count == 0


# ------------------------------------------------------------------
# Query count (generation count as LLM call proxy)
# ------------------------------------------------------------------


def test_query_count_from_generation_observations(adapter: LangfuseAdapter) -> None:
    state = {
        "observations": [
            {"type": "GENERATION", "model": "gpt-4o"},
            {"type": "GENERATION", "model": "gpt-4o"},
            {"type": "SPAN", "name": "retrieval"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.query_count == 2


def test_query_count_from_flat_generations(adapter: LangfuseAdapter) -> None:
    state = {
        "generations": [{"model": "gpt-4o"}, {"model": "gpt-4o"}, {"model": "gpt-4o"}],
    }
    signals = adapter.extract(state)
    assert signals.query_count == 3


def test_query_count_explicit_override(adapter: LangfuseAdapter) -> None:
    state = {
        "query_count": 10,
        "observations": [
            {"type": "GENERATION"},
        ],
    }
    signals = adapter.extract(state)
    assert signals.query_count == 10


# ------------------------------------------------------------------
# Sources / domains extraction
# ------------------------------------------------------------------


def test_sources_from_url_dicts(adapter: LangfuseAdapter) -> None:
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


def test_sources_from_string_urls(adapter: LangfuseAdapter) -> None:
    state = {
        "sources": [
            "https://docs.python.org/3/",
            "https://fastapi.tiangolo.com/tutorial/",
        ],
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 2
    assert signals.unique_domains == 2


def test_sources_from_trace_metadata(adapter: LangfuseAdapter) -> None:
    state = {
        "trace": {
            "metadata": {
                "sources": [
                    {"url": "https://langfuse.com/docs"},
                ],
            },
        },
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 1
    assert signals.unique_domains == 1


def test_sources_with_nested_meta_url(adapter: LangfuseAdapter) -> None:
    state = {
        "sources": [
            {"metadata": {"url": "https://nested.example.com/path"}},
        ],
    }
    signals = adapter.extract(state)
    assert signals.sources_count == 1
    assert signals.unique_domains == 1


# ------------------------------------------------------------------
# Empty / None state handling
# ------------------------------------------------------------------


def test_empty_state_returns_zeroed_signals(adapter: LangfuseAdapter) -> None:
    signals = adapter.extract({})
    assert signals.findings_count == 0
    assert signals.coverage_score == 0.0
    assert signals.total_tokens == 0
    assert signals.error_count == 0
    assert signals.query_count == 0


def test_none_state_returns_zeroed_signals(adapter: LangfuseAdapter) -> None:
    signals = adapter.extract(None)  # type: ignore[arg-type]
    assert signals.findings_count == 0
    assert signals.total_tokens == 0


# ------------------------------------------------------------------
# Complex / realistic Langfuse state
# ------------------------------------------------------------------


def test_full_langfuse_trace_state(adapter: LangfuseAdapter) -> None:
    """Realistic Langfuse trace with multiple generations, errors, and scores."""
    state = {
        "trace": {
            "input": "Research WebAssembly server-side adoption",
            "output": "Comprehensive analysis of WASM adoption trends",
            "metadata": {"sources": [
                {"url": "https://fermyon.com/spin"},
                {"url": "https://developers.cloudflare.com/workers/"},
            ]},
        },
        "observations": [
            {
                "type": "GENERATION",
                "model": "gpt-4o",
                "output": "WASM is gaining traction in server-side computing.",
                "usage": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                "level": "DEFAULT",
            },
            {
                "type": "SPAN",
                "name": "web_search",
                "output": {"results": ["result1", "result2"]},
            },
            {
                "type": "GENERATION",
                "model": "gpt-4o",
                "output": "Fermyon Spin and Cloudflare Workers lead adoption.",
                "usage": {"prompt_tokens": 600, "completion_tokens": 250, "total_tokens": 850},
                "level": "DEFAULT",
            },
            {
                "type": "GENERATION",
                "model": "gpt-4o",
                "output": "",
                "usage": {"prompt_tokens": 400, "completion_tokens": 10, "total_tokens": 410},
                "level": "ERROR",
                "status_message": "Model returned empty response",
            },
        ],
        "scores": [
            {"name": "coverage", "value": 0.65},
            {"name": "quality", "value": 0.8},
        ],
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 1960
    assert signals.prompt_tokens == 1500
    assert signals.completion_tokens == 460
    assert signals.findings_count == 2  # 2 unique non-empty generation outputs
    assert signals.coverage_score == pytest.approx(0.65)
    assert signals.error_count == 1  # 1 ERROR-level observation
    assert signals.query_count == 3  # 3 GENERATION observations
    assert signals.sources_count == 2
    assert signals.unique_domains == 2


def test_generation_only_state(adapter: LangfuseAdapter) -> None:
    """State with only flat generations, no trace wrapper."""
    state = {
        "generations": [
            {
                "model": "claude-3-opus",
                "output": "Analysis of market trends",
                "usage": {"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450},
            },
            {
                "model": "claude-3-opus",
                "output": "Competitive landscape summary",
                "usage": {"prompt_tokens": 350, "completion_tokens": 180, "total_tokens": 530},
            },
        ],
        "coverage_score": 0.5,
    }
    signals = adapter.extract(state)
    assert signals.total_tokens == 980
    assert signals.findings_count == 2
    assert signals.coverage_score == pytest.approx(0.5)
    assert signals.query_count == 2
