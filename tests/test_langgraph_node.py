"""Tests for the LangGraph vitals node integration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from agent_vitals.callbacks.langgraph import LangGraphVitalsNode
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


class _MockStateGraph:
    """Tiny test double for a StateGraph invoke contract."""

    def __init__(self, node: LangGraphVitalsNode) -> None:
        self._node = node

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        updates = self._node(state)
        merged = dict(state)
        merged.update(updates)
        return merged


class _FakeMonitor:
    def __init__(self, snapshot: VitalsSnapshot) -> None:
        self._snapshot = snapshot

    def step_from_signals(self, signals: RawSignals) -> VitalsSnapshot:  # noqa: ARG002
        return self._snapshot


def _failure_snapshot(*, loop_trigger: str | None, stuck_trigger: str | None) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id="graph",
        run_id="run-1",
        loop_index=2,
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


def test_langgraph_vitals_node_returns_state_update() -> None:
    node = LangGraphVitalsNode()
    state = {
        "findings": ["a", "b"],
        "coverage_score": 0.4,
        "total_tokens": 200,
        "errors": [],
    }

    update = node(state)
    assert "agent_vitals" in update
    assert update["agent_vitals"]["signals"]["findings_count"] == 2
    assert update["agent_vitals"]["any_failure"] is False


def test_langgraph_vitals_node_force_finalize_mode() -> None:
    node = LangGraphVitalsNode(
        monitor=_FakeMonitor(_failure_snapshot(loop_trigger="findings_plateau", stuck_trigger=None)),  # type: ignore[arg-type]
        on_failure="force_finalize",
    )

    update = node({"findings_count": 1, "coverage_score": 0.1, "total_tokens": 100})
    assert update["force_finalize"] is True
    assert "agent_vitals_failure" in update


def test_langgraph_vitals_node_raise_mode() -> None:
    node = LangGraphVitalsNode(
        monitor=_FakeMonitor(_failure_snapshot(loop_trigger=None, stuck_trigger="coverage_stagnation")),  # type: ignore[arg-type]
        on_failure="raise",
    )

    with pytest.raises(RuntimeError, match="AgentVitals failure detected"):
        node({"findings_count": 1, "coverage_score": 0.1, "total_tokens": 100})


def test_langgraph_vitals_node_integrates_with_mock_stategraph() -> None:
    graph = _MockStateGraph(LangGraphVitalsNode())

    result = graph.invoke(
        {
            "findings": ["alpha"],
            "coverage_score": 0.3,
            "total_tokens": 150,
            "errors": [],
        }
    )

    assert "agent_vitals" in result
    assert result["agent_vitals"]["signals"]["findings_count"] == 1


def test_langgraph_node_import_safe_without_langgraph() -> None:
    node = LangGraphVitalsNode()
    update = node({"findings_count": 1, "coverage_score": 0.2, "total_tokens": 50})
    assert "agent_vitals" in update
