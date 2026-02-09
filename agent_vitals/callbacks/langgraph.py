"""LangGraph node integration for Agent Vitals."""

from __future__ import annotations

import logging
from typing import Any, Literal, Mapping

from ..adapters import LangGraphAdapter, SignalAdapter
from ..monitor import AgentVitals
from ..schema import VitalsSnapshot

logger = logging.getLogger(__name__)

NodeFailureMode = Literal["log", "force_finalize", "raise"]


class LangGraphVitalsNode:
    """Callable node for LangGraph StateGraph pipelines."""

    def __init__(
        self,
        *,
        monitor: AgentVitals | None = None,
        adapter: SignalAdapter | None = None,
        mission_id: str = "langgraph",
        workflow_type: str = "langgraph",
        on_failure: NodeFailureMode = "log",
        vitals_key: str = "agent_vitals",
        finalize_key: str = "force_finalize",
    ) -> None:
        if on_failure not in {"log", "force_finalize", "raise"}:
            raise ValueError("on_failure must be one of: log, force_finalize, raise")
        self._monitor = monitor or AgentVitals(mission_id=mission_id, workflow_type=workflow_type)
        self._adapter = adapter or LangGraphAdapter()
        self._on_failure = on_failure
        self._vitals_key = vitals_key
        self._finalize_key = finalize_key

    @property
    def monitor(self) -> AgentVitals:
        """Expose the underlying AgentVitals monitor."""
        return self._monitor

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Process current state and return LangGraph-compatible state updates."""
        signals = self._adapter.extract(state)
        snapshot = self._monitor.step_from_signals(signals)

        update: dict[str, Any] = {
            self._vitals_key: self._snapshot_payload(snapshot),
        }

        if snapshot.any_failure:
            self._handle_failure(snapshot, update)

        return update

    def _handle_failure(self, snapshot: VitalsSnapshot, update: dict[str, Any]) -> None:
        trigger = snapshot.stuck_trigger or snapshot.loop_trigger or "unknown"
        message = f"AgentVitals failure detected at loop={snapshot.loop_index}: {trigger}"

        if self._on_failure == "log":
            logger.warning(message)
            return

        if self._on_failure == "force_finalize":
            update[self._finalize_key] = True
            update["agent_vitals_failure"] = message
            return

        raise RuntimeError(message)

    def _snapshot_payload(self, snapshot: VitalsSnapshot) -> dict[str, Any]:
        payload = snapshot.model_dump()
        payload["any_failure"] = snapshot.any_failure
        return payload
