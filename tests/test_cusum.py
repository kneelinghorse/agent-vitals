"""Unit tests for CUSUM change-point tracking."""

from __future__ import annotations

from agent_vitals.detection.cusum import CUSUMTracker


def test_cusum_tracker_accumulates_and_triggers_alarm() -> None:
    """CUSUM should alarm after persistent above-target drift."""
    tracker = CUSUMTracker(target=0.0, k=0.5, h=2.0, warmup_steps=1, direction="increase")

    updates = [tracker.update(value) for value in (0.0, 1.0, 2.0, 3.0)]

    assert updates[0].warmup_complete is True
    assert updates[0].alarm is False
    assert updates[1].alarm is False
    assert updates[2].alarm is False
    assert updates[3].alarm is True
    assert updates[3].score > updates[2].score


def test_cusum_tracker_stays_below_threshold_without_persistent_drift() -> None:
    """Small oscillations near target should not trigger alarm."""
    tracker = CUSUMTracker(target=1.0, k=0.25, h=1.5, warmup_steps=1, direction="increase")
    for value in (1.0, 1.1, 0.9, 1.0, 1.05):
        update = tracker.update(value)
    assert update.alarm is False
    assert update.score < 1.5


def test_cusum_tracker_supports_decrease_direction() -> None:
    """Decrease-direction CUSUM should alarm on sustained drops."""
    tracker = CUSUMTracker(target=10.0, k=0.2, h=2.0, warmup_steps=1, direction="decrease")

    tracker.update(10.0)  # warmup
    update_1 = tracker.update(9.5)
    update_2 = tracker.update(8.0)
    update_3 = tracker.update(7.0)

    assert update_1.alarm is False
    assert update_2.score > update_1.score
    assert update_3.alarm is True


def test_cusum_tracker_reset_clears_runtime_state() -> None:
    """Reset should clear score and warmup progress."""
    tracker = CUSUMTracker(target=0.0, k=0.1, h=0.5, warmup_steps=1)
    tracker.update(0.0)
    tracker.update(1.0)
    assert tracker.score > 0.0

    tracker.reset()
    assert tracker.score == 0.0
    assert tracker.warmup_complete is False
