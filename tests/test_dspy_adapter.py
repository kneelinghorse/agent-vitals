"""Tests for DSPyAdapter — no actual dspy dependency required."""

from __future__ import annotations

import pytest

from agent_vitals.adapters import DSPyAdapter, SignalAdapter
from agent_vitals.schema import RawSignals


@pytest.fixture
def adapter() -> DSPyAdapter:
    return DSPyAdapter()


class TestProtocolConformance:
    def test_implements_signal_adapter(self, adapter: DSPyAdapter) -> None:
        assert isinstance(adapter, SignalAdapter)

    def test_extract_returns_raw_signals(self, adapter: DSPyAdapter) -> None:
        result = adapter.extract({})
        assert isinstance(result, RawSignals)


class TestTokenExtraction:
    def test_lm_usage_single_model(self, adapter: DSPyAdapter) -> None:
        state = {
            "lm_usage": {
                "openai/gpt-4o-mini": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            }
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 100
        assert signals.completion_tokens == 50
        assert signals.total_tokens == 150

    def test_lm_usage_multi_model(self, adapter: DSPyAdapter) -> None:
        state = {
            "lm_usage": {
                "openai/gpt-4o-mini": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "openai/gpt-4o": {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "total_tokens": 280,
                },
            }
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 300
        assert signals.completion_tokens == 130
        assert signals.total_tokens == 430

    def test_lm_usage_derives_total(self, adapter: DSPyAdapter) -> None:
        state = {
            "lm_usage": {
                "model-a": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
            }
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 150

    def test_history_usage_fallback(self, adapter: DSPyAdapter) -> None:
        state = {
            "history": [
                {"usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}},
                {"usage": {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}},
            ]
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 110
        assert signals.completion_tokens == 50
        assert signals.total_tokens == 160

    def test_history_with_input_output_keys(self, adapter: DSPyAdapter) -> None:
        state = {
            "history": [
                {"usage": {"input_tokens": 40, "output_tokens": 15}},
            ]
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 40
        assert signals.completion_tokens == 15
        assert signals.total_tokens == 55

    def test_explicit_total_tokens_override(self, adapter: DSPyAdapter) -> None:
        state = {
            "total_tokens": 999,
            "prompt_tokens": 500,
            "completion_tokens": 499,
            "lm_usage": {"model": {"total_tokens": 111}},
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 999
        assert signals.prompt_tokens == 500

    def test_empty_history(self, adapter: DSPyAdapter) -> None:
        state = {"history": []}
        signals = adapter.extract(state)
        assert signals.total_tokens == 0

    def test_malformed_history_entries(self, adapter: DSPyAdapter) -> None:
        state = {"history": ["not-a-dict", None, 42, {"no_usage": True}]}
        signals = adapter.extract(state)
        assert signals.total_tokens == 0

    def test_malformed_lm_usage(self, adapter: DSPyAdapter) -> None:
        state = {"lm_usage": {"model": "not-a-dict"}}
        signals = adapter.extract(state)
        assert signals.total_tokens == 0


class TestFindingsExtraction:
    def test_explicit_findings_count(self, adapter: DSPyAdapter) -> None:
        state = {"findings_count": 5}
        signals = adapter.extract(state)
        assert signals.findings_count == 5

    def test_predictions_list(self, adapter: DSPyAdapter) -> None:
        state = {"predictions": ["result1", "result2", "result3"]}
        signals = adapter.extract(state)
        assert signals.findings_count == 3

    def test_predictions_filters_empty(self, adapter: DSPyAdapter) -> None:
        state = {"predictions": ["result1", None, "", {}, "result2"]}
        signals = adapter.extract(state)
        assert signals.findings_count == 2

    def test_history_outputs_deduplication(self, adapter: DSPyAdapter) -> None:
        state = {
            "history": [
                {"outputs": ["answer A", "answer B"]},
                {"outputs": ["answer A", "answer C"]},
            ]
        }
        signals = adapter.extract(state)
        assert signals.findings_count == 3  # A, B, C (deduplicated)

    def test_no_findings_sources(self, adapter: DSPyAdapter) -> None:
        state = {}
        signals = adapter.extract(state)
        assert signals.findings_count == 0


class TestCoverageExtraction:
    def test_explicit_coverage(self, adapter: DSPyAdapter) -> None:
        state = {"coverage_score": 0.75}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.75

    def test_module_completion(self, adapter: DSPyAdapter) -> None:
        state = {"modules_completed": 3, "modules_total": 4}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.75

    def test_module_completion_zero_total(self, adapter: DSPyAdapter) -> None:
        state = {"modules_completed": 0, "modules_total": 0}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.0

    def test_coverage_clipped(self, adapter: DSPyAdapter) -> None:
        state = {"coverage_score": 1.5}
        signals = adapter.extract(state)
        assert signals.coverage_score == 1.0


class TestErrorExtraction:
    def test_explicit_error_count(self, adapter: DSPyAdapter) -> None:
        state = {"error_count": 3}
        signals = adapter.extract(state)
        assert signals.error_count == 3

    def test_errors_list(self, adapter: DSPyAdapter) -> None:
        state = {"errors": ["timeout", "rate_limit", "parse_error"]}
        signals = adapter.extract(state)
        assert signals.error_count == 3


class TestQueryCount:
    def test_explicit_query_count(self, adapter: DSPyAdapter) -> None:
        state = {"query_count": 10}
        signals = adapter.extract(state)
        assert signals.query_count == 10

    def test_query_from_history_length(self, adapter: DSPyAdapter) -> None:
        state = {
            "history": [
                {"usage": {"total_tokens": 10}},
                {"usage": {"total_tokens": 20}},
                {"usage": {"total_tokens": 30}},
            ]
        }
        signals = adapter.extract(state)
        assert signals.query_count == 3


class TestNullSafety:
    def test_none_state(self, adapter: DSPyAdapter) -> None:
        signals = adapter.extract(None)  # type: ignore[arg-type]
        assert isinstance(signals, RawSignals)
        assert signals.total_tokens == 0

    def test_empty_state(self, adapter: DSPyAdapter) -> None:
        signals = adapter.extract({})
        assert isinstance(signals, RawSignals)
        assert signals.findings_count == 0

    def test_all_none_values(self, adapter: DSPyAdapter) -> None:
        state = {
            "lm_usage": None,
            "history": None,
            "predictions": None,
            "errors": None,
        }
        signals = adapter.extract(state)
        assert isinstance(signals, RawSignals)
        assert signals.total_tokens == 0


class TestRealisticScenario:
    def test_full_dspy_state(self, adapter: DSPyAdapter) -> None:
        """Simulate a realistic DSPy program state with multiple modules."""
        state = {
            "lm_usage": {
                "openai/gpt-4o-mini": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 400,
                    "total_tokens": 1600,
                },
            },
            "history": [
                {"outputs": ["Generated summary of topic A"], "usage": {"total_tokens": 500}},
                {"outputs": ["Generated summary of topic A"], "usage": {"total_tokens": 550}},
                {"outputs": ["Generated analysis of topic B"], "usage": {"total_tokens": 550}},
            ],
            "predictions": [
                {"answer": "Summary A"},
                {"answer": "Analysis B"},
            ],
            "modules_completed": 2,
            "modules_total": 3,
            "errors": [],
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 1600  # from lm_usage (preferred)
        assert signals.findings_count == 2  # from predictions
        assert signals.coverage_score == pytest.approx(2 / 3, abs=0.01)
        assert signals.error_count == 0
        assert signals.query_count == 3  # from history length
