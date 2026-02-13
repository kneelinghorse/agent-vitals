"""Unit tests for AdaptiveThreshold SPC tracker."""

from __future__ import annotations

import pytest

from agent_vitals.detection.adaptive_threshold import AdaptiveThreshold


def test_increase_direction_spike_triggers_alarm_after_warmup() -> None:
    tracker = AdaptiveThreshold(direction="increase", warmup_steps=2, window_size=5, k_sigma=3.0)

    assert tracker.update(0.2, fallback_threshold=0.8).alarm is False
    assert tracker.update(0.25, fallback_threshold=0.8).alarm is False
    # Warmup complete: baseline near 0.2, spike should breach adaptive control limit.
    update = tracker.update(0.95, fallback_threshold=0.8)
    assert update.warmup_complete is True
    assert update.alarm is True


def test_decrease_direction_drop_triggers_alarm_after_warmup() -> None:
    tracker = AdaptiveThreshold(direction="decrease", warmup_steps=2, window_size=5, k_sigma=3.0)

    assert tracker.update(1.0, fallback_threshold=0.3).alarm is False
    assert tracker.update(0.9, fallback_threshold=0.3).alarm is False
    update = tracker.update(0.2, fallback_threshold=0.3)
    assert update.warmup_complete is True
    assert update.alarm is True


def test_warmup_uses_fallback_threshold() -> None:
    tracker = AdaptiveThreshold(direction="increase", warmup_steps=2, window_size=5)

    step1 = tracker.update(0.75, fallback_threshold=0.8)
    step2 = tracker.update(0.85, fallback_threshold=0.8)

    assert step1.warmup_complete is False
    assert step1.alarm is False
    assert step2.warmup_complete is False
    assert step2.alarm is True


def test_cooldown_suppresses_consecutive_alarm() -> None:
    tracker = AdaptiveThreshold(
        direction="increase",
        warmup_steps=1,
        window_size=2,
        cooldown_steps=1,
        k_sigma=3.0,
    )

    tracker.update(0.10, fallback_threshold=0.8)
    first = tracker.update(0.90, fallback_threshold=0.8)
    second = tracker.update(0.95, fallback_threshold=0.8)

    assert first.alarm is True
    assert second.suppressed_by_cooldown is True
    assert second.alarm is False


def test_invalid_direction_rejected() -> None:
    with pytest.raises(ValueError):
        AdaptiveThreshold(direction="invalid")  # type: ignore[arg-type]
