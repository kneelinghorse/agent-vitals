"""Tests for HaystackAdapter — no actual haystack-ai dependency required."""

from __future__ import annotations

import pytest

from agent_vitals.adapters import HaystackAdapter, SignalAdapter
from agent_vitals.schema import RawSignals


@pytest.fixture
def adapter() -> HaystackAdapter:
    return HaystackAdapter()


class TestProtocolConformance:
    def test_implements_signal_adapter(self, adapter: HaystackAdapter) -> None:
        assert isinstance(adapter, SignalAdapter)

    def test_extract_returns_raw_signals(self, adapter: HaystackAdapter) -> None:
        result = adapter.extract({})
        assert isinstance(result, RawSignals)


class TestTokenExtraction:
    def test_messages_meta_usage(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "_meta": {},
                },
                {
                    "role": "assistant",
                    "content": "response",
                    "_meta": {
                        "usage": {
                            "prompt_tokens": 15,
                            "completion_tokens": 59,
                            "total_tokens": 74,
                        }
                    },
                },
            ]
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 15
        assert signals.completion_tokens == 59
        assert signals.total_tokens == 74

    def test_messages_multi_assistant(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "first",
                    "_meta": {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
                },
                {
                    "role": "assistant",
                    "content": "second",
                    "_meta": {"usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}},
                },
            ]
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 25
        assert signals.completion_tokens == 45
        assert signals.total_tokens == 70

    def test_replies_fallback(self, adapter: HaystackAdapter) -> None:
        state = {
            "replies": [
                {
                    "content": "answer",
                    "_meta": {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
                }
            ]
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 150

    def test_component_outputs_fallback(self, adapter: HaystackAdapter) -> None:
        state = {
            "component_outputs": {
                "chat_generator": {
                    "replies": [
                        {"_meta": {"usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}}},
                    ]
                },
                "summarizer": {
                    "replies": [
                        {"_meta": {"usage": {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}}},
                    ]
                },
            }
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 140
        assert signals.completion_tokens == 70
        assert signals.total_tokens == 210

    def test_meta_key_variant(self, adapter: HaystackAdapter) -> None:
        """Some serializations use 'meta' instead of '_meta'."""
        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "reply",
                    "meta": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                }
            ]
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 15

    def test_derives_total_from_parts(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "reply",
                    "_meta": {"usage": {"prompt_tokens": 30, "completion_tokens": 20}},
                }
            ]
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 50

    def test_explicit_total_override(self, adapter: HaystackAdapter) -> None:
        state = {
            "total_tokens": 999,
            "prompt_tokens": 500,
            "completion_tokens": 499,
            "messages": [
                {"role": "assistant", "content": "x", "_meta": {"usage": {"total_tokens": 50}}},
            ],
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 999

    def test_empty_messages(self, adapter: HaystackAdapter) -> None:
        state = {"messages": []}
        signals = adapter.extract(state)
        assert signals.total_tokens == 0


class TestFindingsExtraction:
    def test_explicit_findings_count(self, adapter: HaystackAdapter) -> None:
        state = {"findings_count": 7}
        signals = adapter.extract(state)
        assert signals.findings_count == 7

    def test_unique_assistant_messages(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": [
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "answer A"},
                {"role": "user", "content": "question 2"},
                {"role": "assistant", "content": "answer B"},
                {"role": "assistant", "content": "answer A"},  # duplicate
            ]
        }
        signals = adapter.extract(state)
        assert signals.findings_count == 2  # A and B (deduplicated)

    def test_component_output_replies(self, adapter: HaystackAdapter) -> None:
        state = {
            "component_outputs": {
                "generator": {
                    "replies": [
                        {"content": "reply1", "_meta": {}},
                        {"content": "reply2", "_meta": {}},
                    ]
                }
            }
        }
        signals = adapter.extract(state)
        assert signals.findings_count == 2

    def test_no_findings(self, adapter: HaystackAdapter) -> None:
        state = {"messages": [{"role": "user", "content": "hello"}]}
        signals = adapter.extract(state)
        assert signals.findings_count == 0


class TestCoverageExtraction:
    def test_explicit_coverage(self, adapter: HaystackAdapter) -> None:
        state = {"coverage_score": 0.8}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.8

    def test_agent_state_coverage(self, adapter: HaystackAdapter) -> None:
        state = {"state": {"coverage_score": 0.6}}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.6

    def test_agent_state_progress(self, adapter: HaystackAdapter) -> None:
        state = {"state": {"progress": 0.45}}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.45

    def test_pipeline_component_progress(self, adapter: HaystackAdapter) -> None:
        state = {
            "component_outputs": {"a": {}, "b": {}},
            "components_total": 4,
        }
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.5

    def test_coverage_clipped(self, adapter: HaystackAdapter) -> None:
        state = {"coverage_score": 2.0}
        signals = adapter.extract(state)
        assert signals.coverage_score == 1.0

    def test_no_coverage_info(self, adapter: HaystackAdapter) -> None:
        state = {}
        signals = adapter.extract(state)
        assert signals.coverage_score == 0.0


class TestErrorExtraction:
    def test_explicit_error_count(self, adapter: HaystackAdapter) -> None:
        state = {"error_count": 3}
        signals = adapter.extract(state)
        assert signals.error_count == 3

    def test_errors_list(self, adapter: HaystackAdapter) -> None:
        state = {"errors": ["timeout", "api_error"]}
        signals = adapter.extract(state)
        assert signals.error_count == 2


class TestQueryCount:
    def test_explicit_query_count(self, adapter: HaystackAdapter) -> None:
        state = {"query_count": 5}
        signals = adapter.extract(state)
        assert signals.query_count == 5

    def test_query_from_assistant_messages(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "assistant", "content": "a3"},
            ]
        }
        signals = adapter.extract(state)
        assert signals.query_count == 3  # 3 assistant messages


class TestSourcesAndDomains:
    def test_sources_with_urls(self, adapter: HaystackAdapter) -> None:
        state = {
            "sources": [
                {"url": "https://example.com/page1", "content": "..."},
                {"url": "https://example.com/page2", "content": "..."},
                {"url": "https://other.org/doc", "content": "..."},
            ]
        }
        signals = adapter.extract(state)
        assert signals.sources_count == 3
        assert signals.unique_domains == 2  # example.com, other.org

    def test_sources_with_meta_url(self, adapter: HaystackAdapter) -> None:
        state = {
            "sources": [
                {"meta": {"url": "https://docs.haystack.deepset.ai/page"}},
            ]
        }
        signals = adapter.extract(state)
        assert signals.sources_count == 1
        assert signals.unique_domains == 1

    def test_sources_with_meta_source(self, adapter: HaystackAdapter) -> None:
        state = {
            "sources": [
                {"meta": {"source": "https://arxiv.org/paper"}},
            ]
        }
        signals = adapter.extract(state)
        assert signals.unique_domains == 1

    def test_explicit_overrides_take_precedence(self, adapter: HaystackAdapter) -> None:
        state = {
            "sources_count": 10,
            "unique_domains": 5,
            "sources": [{"url": "https://example.com"}],
        }
        signals = adapter.extract(state)
        assert signals.sources_count == 10
        assert signals.unique_domains == 5


class TestNullSafety:
    def test_none_state(self, adapter: HaystackAdapter) -> None:
        signals = adapter.extract(None)  # type: ignore[arg-type]
        assert isinstance(signals, RawSignals)
        assert signals.total_tokens == 0

    def test_empty_state(self, adapter: HaystackAdapter) -> None:
        signals = adapter.extract({})
        assert isinstance(signals, RawSignals)

    def test_all_none_values(self, adapter: HaystackAdapter) -> None:
        state = {
            "messages": None,
            "replies": None,
            "component_outputs": None,
            "sources": None,
            "errors": None,
        }
        signals = adapter.extract(state)
        assert isinstance(signals, RawSignals)
        assert signals.total_tokens == 0

    def test_malformed_messages(self, adapter: HaystackAdapter) -> None:
        state = {"messages": ["not-a-dict", 42, None]}
        signals = adapter.extract(state)
        assert signals.total_tokens == 0


class TestRealisticScenario:
    def test_full_agent_state(self, adapter: HaystackAdapter) -> None:
        """Simulate a realistic Haystack Agent state after a research session."""
        state = {
            "messages": [
                {"role": "user", "content": "Research quantum computing advances"},
                {
                    "role": "assistant",
                    "content": "I found information about quantum error correction.",
                    "_meta": {"usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}},
                },
                {"role": "user", "content": "Tell me more about applications"},
                {
                    "role": "assistant",
                    "content": "Quantum computing has applications in drug discovery.",
                    "_meta": {"usage": {"prompt_tokens": 250, "completion_tokens": 100, "total_tokens": 350}},
                },
            ],
            "state": {"coverage_score": 0.6},
            "sources": [
                {"url": "https://arxiv.org/paper1"},
                {"url": "https://nature.com/article1"},
                {"url": "https://arxiv.org/paper2"},
            ],
            "errors": [],
        }
        signals = adapter.extract(state)
        assert signals.prompt_tokens == 450
        assert signals.completion_tokens == 180
        assert signals.total_tokens == 630
        assert signals.findings_count == 2  # 2 unique assistant messages
        assert signals.coverage_score == 0.6
        assert signals.sources_count == 3
        assert signals.unique_domains == 2  # arxiv.org, nature.com
        assert signals.query_count == 2  # 2 assistant messages
        assert signals.error_count == 0

    def test_full_pipeline_state(self, adapter: HaystackAdapter) -> None:
        """Simulate a Haystack Pipeline.run() output."""
        state = {
            "component_outputs": {
                "retriever": {
                    "documents": [{"content": "doc1"}, {"content": "doc2"}],
                },
                "chat_generator": {
                    "replies": [
                        {
                            "content": "Generated answer",
                            "_meta": {"usage": {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420}},
                        }
                    ],
                },
            },
            "components_total": 3,
        }
        signals = adapter.extract(state)
        assert signals.total_tokens == 420
        assert signals.findings_count == 1  # 1 reply
        assert signals.coverage_score == pytest.approx(2 / 3, abs=0.01)
