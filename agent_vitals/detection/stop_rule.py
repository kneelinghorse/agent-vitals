"""Stop rule helpers for Agent Vitals.

This module centralizes the unified stop-rule logic used by evaluation and
future enforcement work. It accepts dict-like payloads (JSONL) or model
instances without assuming a specific schema implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_THRASH_ERROR_THRESHOLD = 1
DEFAULT_RUNAWAY_TRIGGER = "burn_rate_anomaly"


@dataclass(frozen=True, slots=True)
class StopRuleSignals:
    """Computed stop-rule signals for a snapshot."""

    loop_detected: bool
    stuck_detected: bool
    thrash_detected: bool
    runaway_cost_detected: bool

    @property
    def any_failure(self) -> bool:
        """Return True when any failure-mode signal is active."""

        return bool(
            self.loop_detected
            or self.stuck_detected
            or self.thrash_detected
            or self.runaway_cost_detected
        )

    def triggers(self) -> tuple[str, ...]:
        """Return the ordered list of active failure-mode triggers."""

        triggers: list[str] = []
        if self.loop_detected:
            triggers.append("loop")
        if self.stuck_detected:
            triggers.append("stuck")
        if self.thrash_detected:
            triggers.append("thrash")
        if self.runaway_cost_detected:
            triggers.append("runaway_cost")
        return tuple(triggers)


def derive_stop_signals(
    snapshot: Mapping[str, Any] | object,
    *,
    thrash_error_threshold: int = DEFAULT_THRASH_ERROR_THRESHOLD,
    runaway_trigger: str = DEFAULT_RUNAWAY_TRIGGER,
) -> StopRuleSignals:
    """Compute stop-rule signals from a snapshot payload.

    Args:
        snapshot: Dict-like payload or model instance with vitals fields.
        thrash_error_threshold: Error-count threshold to flag thrash.
        runaway_trigger: Stuck trigger string that maps to runaway cost.

    Returns:
        StopRuleSignals with per-mode detection flags.
    """

    loop_detected = _read_flag(snapshot, "loop_detected")
    stuck_detected = _read_flag(snapshot, "stuck_detected")

    explicit_thrash = _read_optional_flag(snapshot, "thrash_detected")
    explicit_runaway = _read_optional_flag(snapshot, "runaway_cost_detected")

    thrash_detected = False
    if explicit_thrash is not None:
        thrash_detected = bool(explicit_thrash)
    elif thrash_error_threshold > 0:
        error_count = _read_error_count(snapshot)
        thrash_detected = bool(error_count is not None and error_count >= thrash_error_threshold)

    runaway_cost_detected = False
    if explicit_runaway is not None:
        runaway_cost_detected = bool(explicit_runaway)
    else:
        stuck_trigger = _read_optional_text(snapshot, "stuck_trigger")
        runaway_cost_detected = bool(stuck_trigger and stuck_trigger == runaway_trigger)

    return StopRuleSignals(
        loop_detected=loop_detected,
        stuck_detected=stuck_detected,
        thrash_detected=thrash_detected,
        runaway_cost_detected=runaway_cost_detected,
    )


def _read_optional_flag(snapshot: Mapping[str, Any] | object, key: str) -> Optional[bool]:
    if isinstance(snapshot, Mapping):
        if key in snapshot:
            value = snapshot.get(key)
            if value is None:
                return None
            return bool(value)
        return None
    if hasattr(snapshot, key):
        value = getattr(snapshot, key)
        if value is None:
            return None
        return bool(value)
    return None


def _read_flag(snapshot: Mapping[str, Any] | object, key: str) -> bool:
    value = _read_optional_flag(snapshot, key)
    return bool(value) if value is not None else False


def _read_optional_text(snapshot: Mapping[str, Any] | object, key: str) -> Optional[str]:
    if isinstance(snapshot, Mapping):
        if key not in snapshot:
            return None
        value = snapshot.get(key)
    elif hasattr(snapshot, key):
        value = getattr(snapshot, key)
    else:
        return None

    text = str(value or "").strip()
    return text or None


def _read_error_count(snapshot: Mapping[str, Any] | object) -> Optional[int]:
    if isinstance(snapshot, Mapping):
        if "error_count" in snapshot:
            return _coerce_int(snapshot.get("error_count"))
        signals = snapshot.get("signals")
        if isinstance(signals, Mapping) and "error_count" in signals:
            return _coerce_int(signals.get("error_count"))
        return None

    if hasattr(snapshot, "error_count"):
        return _coerce_int(getattr(snapshot, "error_count"))

    signals = getattr(snapshot, "signals", None)
    if signals is None:
        return None
    if isinstance(signals, Mapping):
        return _coerce_int(signals.get("error_count"))
    if hasattr(signals, "error_count"):
        return _coerce_int(getattr(signals, "error_count"))
    return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_RUNAWAY_TRIGGER",
    "DEFAULT_THRASH_ERROR_THRESHOLD",
    "StopRuleSignals",
    "derive_stop_signals",
]
