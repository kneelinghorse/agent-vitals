"""Tests for LangChain callback integration."""

from __future__ import annotations

from typing import Any

import pytest

from agent_vitals.callbacks import langchain as lc_callbacks
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


class _FakeMonitor:
    def __init__(self, snapshot: VitalsSnapshot) -> None:
        self._snapshot = snapshot

    def step(self, **kwargs: Any) -> VitalsSnapshot:  # noqa: ARG002
        return self._snapshot


class _FakeResponse:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }


def _failure_snapshot(*, loop_trigger: str | None, stuck_trigger: str | None) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id="test-mission",
        run_id="run-1",
        loop_index=3,
        signals=RawSignals(
            findings_count=1,
            coverage_score=0.2,
            total_tokens=1000,
            error_count=0,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.0,
            cv_findings_rate=0.0,
            dm_coverage=0.0,
            dm_findings=0.0,
            qpf_tokens=0.5,
            cs_effort=0.0,
        ),
        health_state="critical",
        loop_detected=loop_trigger is not None,
        loop_confidence=1.0 if loop_trigger is not None else 0.0,
        loop_trigger=loop_trigger,
        stuck_detected=stuck_trigger is not None,
        stuck_confidence=1.0 if stuck_trigger is not None else 0.0,
        stuck_trigger=stuck_trigger,
    )


def test_langchain_callback_requires_langchain_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", False)
    with pytest.raises(ImportError, match="langchain-core"):
        lc_callbacks.LangChainVitalsCallback()


def test_langchain_callback_tracks_tokens_errors_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", True)
    callback = lc_callbacks.LangChainVitalsCallback(export_jsonl_dir=tmp_path)

    callback.on_llm_end(_FakeResponse(prompt_tokens=12, completion_tokens=18, total_tokens=30))
    callback.on_tool_end({"source_documents": ["doc-1", "doc-2"]})
    callback.on_tool_error(RuntimeError("tool failed"))
    callback.on_chain_end({"output": "final answer", "coverage_score": 0.4})

    snapshot = callback.last_snapshot
    assert snapshot is not None
    assert snapshot.signals.prompt_tokens == 12
    assert snapshot.signals.completion_tokens == 18
    assert snapshot.signals.total_tokens == 30
    assert snapshot.signals.findings_count == 1
    assert snapshot.signals.coverage_score == 0.4
    assert snapshot.signals.query_count == 2
    assert snapshot.signals.api_calls == 2
    assert snapshot.signals.sources_count == 2
    assert snapshot.signals.error_count == 1

    assert (tmp_path / "langchain.jsonl").exists()


def test_langchain_callback_on_failure_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", True)
    callback = lc_callbacks.LangChainVitalsCallback(on_failure="raise")
    callback._monitor = _FakeMonitor(  # type: ignore[assignment]
        _failure_snapshot(loop_trigger="findings_plateau", stuck_trigger=None)
    )

    with pytest.raises(RuntimeError, match="AgentVitals failure detected"):
        callback.on_chain_end({"output": "x"})


def test_langchain_callback_on_failure_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", True)
    observed: list[int] = []

    def _handler(snapshot: VitalsSnapshot) -> None:
        observed.append(snapshot.loop_index)

    callback = lc_callbacks.LangChainVitalsCallback(
        on_failure="callback",
        failure_callback=_handler,
    )
    callback._monitor = _FakeMonitor(  # type: ignore[assignment]
        _failure_snapshot(loop_trigger=None, stuck_trigger="coverage_stagnation")
    )

    callback.on_chain_end({"output": "x"})
    assert observed == [3]


def test_langchain_callback_callback_mode_requires_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", True)
    with pytest.raises(ValueError, match="failure_callback"):
        lc_callbacks.LangChainVitalsCallback(on_failure="callback")


def test_langchain_callback_extends_base_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc_callbacks, "_HAS_LANGCHAIN", True)
    callback = lc_callbacks.LangChainVitalsCallback()
    assert isinstance(callback, lc_callbacks.BaseCallbackHandler)
