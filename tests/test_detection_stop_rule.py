"""Unit tests for stop-rule helpers — ported from DeepSearch test suite."""

from __future__ import annotations

from agent_vitals.detection.stop_rule import derive_stop_signals


def test_detects_thrash_from_error_threshold() -> None:
    """Thrash should be flagged when error_count meets the threshold."""
    snapshot = {
        "loop_detected": False,
        "stuck_detected": False,
        "signals": {"error_count": 2},
    }
    signals = derive_stop_signals(snapshot, thrash_error_threshold=2)
    assert signals.thrash_detected is True
    assert signals.any_failure is True


def test_respects_explicit_flags() -> None:
    """Explicit snapshot flags should override derived defaults."""
    snapshot = {
        "thrash_detected": True,
        "runaway_cost_detected": True,
        "signals": {"error_count": 0},
    }
    signals = derive_stop_signals(snapshot, thrash_error_threshold=99)
    assert signals.thrash_detected is True
    assert signals.runaway_cost_detected is True


def test_maps_runaway_cost_from_burn_rate_trigger() -> None:
    """Burn-rate anomaly should map to runaway_cost even when stuck is suppressed."""
    snapshot = {
        "stuck_detected": False,
        "stuck_trigger": "burn_rate_anomaly",
        "signals": {"error_count": 0},
    }
    signals = derive_stop_signals(snapshot)
    assert signals.runaway_cost_detected is True
    assert signals.stuck_detected is False


def test_no_false_flags_on_clean_snapshot() -> None:
    """Clean snapshot should not trigger any stop signals."""
    snapshot = {
        "loop_detected": False,
        "stuck_detected": False,
        "signals": {"error_count": 0},
    }
    signals = derive_stop_signals(snapshot)
    assert signals.any_failure is False
    assert signals.thrash_detected is False
    assert signals.runaway_cost_detected is False
