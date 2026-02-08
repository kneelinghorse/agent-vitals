"""Unit tests for temporal metrics computation — ported from DeepSearch test suite."""

from __future__ import annotations

import pytest

from agent_vitals.detection.metrics import TemporalMetrics
from agent_vitals.schema import HysteresisConfig


# ---------------------------------------------------------------------------
# Coefficient of Variation
# ---------------------------------------------------------------------------


def test_cv_stable_signal() -> None:
    """CV should be low for stable signals."""
    metrics = TemporalMetrics()
    cv = metrics.coefficient_of_variation([0.5, 0.51, 0.49, 0.5, 0.52])
    assert cv < 0.05


def test_cv_erratic_signal() -> None:
    """CV should be high for erratic signals."""
    metrics = TemporalMetrics()
    cv = metrics.coefficient_of_variation([0.2, 0.8, 0.3, 0.9, 0.1])
    assert cv > 0.5


def test_cv_edge_cases() -> None:
    """CV returns 0 for insufficient or zero-mean data."""
    metrics = TemporalMetrics()
    assert metrics.coefficient_of_variation([0.5]) == 0.0
    assert metrics.coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0


def test_cv_rejects_invalid_window() -> None:
    metrics = TemporalMetrics()
    with pytest.raises(ValueError):
        metrics.coefficient_of_variation([1.0, 2.0], window=0)


def test_cv_handles_non_numeric() -> None:
    """Non-numeric inputs should be ignored rather than raising."""
    metrics = TemporalMetrics()
    cv = metrics.coefficient_of_variation([0.5, "bad", None, 0.5])  # type: ignore[list-item]
    assert cv == 0.0


# ---------------------------------------------------------------------------
# Directional Momentum
# ---------------------------------------------------------------------------


def test_dm_improving() -> None:
    """DM should be positive for improving trend."""
    metrics = TemporalMetrics()
    dm = metrics.directional_momentum([0.3, 0.4, 0.5, 0.6, 0.7])
    assert dm > 0.3


def test_dm_degrading() -> None:
    """DM should be negative for degrading trend."""
    metrics = TemporalMetrics()
    dm = metrics.directional_momentum([0.7, 0.6, 0.5, 0.4, 0.3])
    assert dm < -0.3


def test_dm_stagnant_and_insufficient() -> None:
    """DM should be ~0 for stagnant signal and 0 for insufficient history."""
    metrics = TemporalMetrics()
    assert abs(metrics.directional_momentum([0.5, 0.5, 0.5, 0.5])) < 1e-6
    assert metrics.directional_momentum([0.5, 0.6], lookback=3) == 0.0


def test_dm_clamped() -> None:
    """DM must always remain in [-1, 1]."""
    metrics = TemporalMetrics()
    dm = metrics.directional_momentum([1.0, 10.0, 50.0, 100.0], alpha=1.0, lookback=2)
    assert -1.0 <= dm <= 1.0


def test_dm_returns_zero_when_scale_is_zero() -> None:
    metrics = TemporalMetrics()
    assert metrics.directional_momentum([0.0, 0.0, 0.0, 0.0], lookback=2) == 0.0


def test_dm_rejects_invalid_params() -> None:
    metrics = TemporalMetrics()
    with pytest.raises(ValueError):
        metrics.directional_momentum([0.1, 0.2, 0.3], lookback=0)
    with pytest.raises(ValueError):
        metrics.directional_momentum([0.1, 0.2, 0.3], alpha=0.0)
    with pytest.raises(ValueError):
        metrics.directional_momentum([0.1, 0.2, 0.3], alpha=2.0)


# ---------------------------------------------------------------------------
# Temporal Hysteresis
# ---------------------------------------------------------------------------


def test_hysteresis_state_transitions() -> None:
    """TH should respect enter/exit thresholds."""
    metrics = TemporalMetrics()
    config = HysteresisConfig(enter_warning=0.4, exit_warning=0.6, enter_critical=0.2, exit_critical=0.35)

    state, changed = metrics.temporal_hysteresis(0.5, "healthy", config)
    assert state == "healthy" and not changed

    state, changed = metrics.temporal_hysteresis(0.35, "healthy", config)
    assert state == "warning" and changed

    state, changed = metrics.temporal_hysteresis(0.65, "warning", config)
    assert state == "healthy" and changed

    state, changed = metrics.temporal_hysteresis(0.15, "warning", config)
    assert state == "critical" and changed

    state, changed = metrics.temporal_hysteresis(0.4, "critical", config)
    assert state == "warning" and changed


def test_hysteresis_no_change_between_thresholds() -> None:
    """TH should keep warning/critical when between thresholds."""
    metrics = TemporalMetrics()
    config = HysteresisConfig(enter_warning=0.4, exit_warning=0.6, enter_critical=0.2, exit_critical=0.35)

    state, changed = metrics.temporal_hysteresis(0.5, "warning", config)
    assert state == "warning" and not changed

    state, changed = metrics.temporal_hysteresis(0.1, "critical", config)
    assert state == "critical" and not changed


def test_hysteresis_handles_non_finite() -> None:
    """Non-finite input should preserve state without change."""
    metrics = TemporalMetrics()
    state, changed = metrics.temporal_hysteresis(float("nan"), "warning", HysteresisConfig())
    assert state == "warning" and not changed


def test_hysteresis_unknown_state_defaults_to_healthy() -> None:
    """Unknown states should reset to healthy."""
    metrics = TemporalMetrics()
    state, changed = metrics.temporal_hysteresis(0.5, "unknown", HysteresisConfig())  # type: ignore[arg-type]
    assert state == "healthy" and changed


def test_hysteresis_config_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        HysteresisConfig(enter_warning=float("nan"))
