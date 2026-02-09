"""Base adapter abstractions for framework integrations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..schema import RawSignals


@runtime_checkable
class SignalAdapter(Protocol):
    """Protocol for extracting RawSignals from framework-specific state."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        """Extract Agent Vitals raw signals from a framework state mapping."""
        ...


class BaseAdapter:
    """Base adapter that provides shared normalization and validation helpers."""

    def normalize(self, state: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Normalize missing state payloads to an empty mapping."""
        return state if state is not None else {}

    def validate(self, signals: RawSignals) -> RawSignals:
        """Validate and normalize a RawSignals payload through the schema."""
        return RawSignals.model_validate(signals.model_dump())

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_len(value: Any, default: int = 0) -> int:
        try:
            return len(value) if value is not None else default
        except TypeError:
            return default

    @staticmethod
    def _clip01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        raise NotImplementedError("Subclasses must implement extract().")
