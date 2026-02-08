"""Unit tests for loop/stuck detection — ported from DeepSearch test suite.

Tests cover: healthy progression, findings plateau, coverage stagnation,
build workflow skip, early FP avoidance, short_run_objective_gap disabled,
burn rate anomaly, grace period, and high-coverage suppression.
"""

from __future__ import annotations

import copy

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.schema import VitalsSnapshot


def _make_snapshot(
    base: dict,
    *,
    loop_index: int,
    findings_count: int,
    coverage_score: float,
    total_tokens: int,
    query_count: int,
    unique_domains: int,
    dm_coverage: float,
    cv_coverage: float,
    objectives_covered: int | None = None,
    cs_effort: float | None = None,
) -> VitalsSnapshot:
    payload = copy.deepcopy(base)
    payload["loop_index"] = loop_index
    payload["signals"]["findings_count"] = findings_count
    payload["signals"]["coverage_score"] = coverage_score
    payload["signals"]["total_tokens"] = total_tokens
    payload["signals"]["query_count"] = query_count
    payload["signals"]["unique_domains"] = unique_domains
    if objectives_covered is not None:
        payload["signals"]["objectives_covered"] = objectives_covered
    payload["metrics"]["dm_coverage"] = dm_coverage
    payload["metrics"]["cv_coverage"] = cv_coverage
    if cs_effort is not None:
        payload["metrics"]["cs_effort"] = cs_effort
    payload["loop_detected"] = False
    payload["loop_confidence"] = 0.0
    payload["loop_trigger"] = None
    payload["stuck_detected"] = False
    payload["stuck_confidence"] = 0.0
    payload["stuck_trigger"] = None
    return VitalsSnapshot.model_validate(payload)


# ---------------------------------------------------------------------------
# Core detection tests
# ---------------------------------------------------------------------------


def test_healthy_progression(vitals_snapshot_healthy: dict) -> None:
    """Healthy progression should not be flagged as loop/stuck."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=0, findings_count=1, coverage_score=0.2, total_tokens=100, query_count=1, unique_domains=1, dm_coverage=0.4, cv_coverage=0.2),
        _make_snapshot(vitals_snapshot_healthy, loop_index=1, findings_count=2, coverage_score=0.35, total_tokens=220, query_count=2, unique_domains=2, dm_coverage=0.4, cv_coverage=0.2),
        _make_snapshot(vitals_snapshot_healthy, loop_index=2, findings_count=4, coverage_score=0.5, total_tokens=350, query_count=3, unique_domains=3, dm_coverage=0.4, cv_coverage=0.2),
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=6, coverage_score=0.65, total_tokens=500, query_count=4, unique_domains=4, dm_coverage=0.45, cv_coverage=0.12)

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False
    assert result.stuck_detected is False


def test_detects_findings_plateau(vitals_snapshot_healthy: dict) -> None:
    """Repeated no-progress with repeated queries should trigger loop proxy."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=10, coverage_score=0.6, total_tokens=700 + 300 * i, query_count=i, unique_domains=5, dm_coverage=0.3, cv_coverage=0.2)
        for i in range(7)
    ]
    current = history.pop()

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is True
    assert result.loop_confidence > 0.0
    assert result.loop_trigger is not None


def test_detects_coverage_stagnation(vitals_snapshot_healthy: dict) -> None:
    """Low DM + low CV on coverage should trigger stuck detection."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.5, total_tokens=100 * (i + 1), query_count=i + 1, unique_domains=i + 1, dm_coverage=0.2, cv_coverage=0.2)
        for i in range(3)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=4, coverage_score=0.5, total_tokens=400, query_count=4, unique_domains=4, dm_coverage=-0.01, cv_coverage=0.04)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is True
    assert result.stuck_confidence > 0.0
    assert result.stuck_trigger is not None


def test_skips_stuck_for_build_workflow(vitals_snapshot_healthy: dict) -> None:
    """Build workflows should skip stuck detection."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.5, total_tokens=100 * (i + 1), query_count=i + 1, unique_domains=i + 1, dm_coverage=0.2, cv_coverage=0.2)
        for i in range(3)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=4, coverage_score=0.5, total_tokens=400, query_count=4, unique_domains=4, dm_coverage=-0.01, cv_coverage=0.04)

    result = detect_loop(current, history, config=config, workflow_type="build")
    assert result.stuck_detected is False


def test_avoids_early_false_positive(vitals_snapshot_healthy: dict) -> None:
    """Short histories should not trigger stagnation-based stuck detection."""
    config = VitalsConfig()
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=0, findings_count=0, coverage_score=0.0, total_tokens=0, query_count=0, unique_domains=0, dm_coverage=0.0, cv_coverage=0.0)

    result = detect_loop(current, history=[], config=config)
    assert result.stuck_detected is False


def test_short_run_objective_gap_disabled(vitals_snapshot_healthy: dict) -> None:
    """AV-20: short_run_objective_gap is disabled — should NOT trigger stuck."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=0, findings_count=3, coverage_score=0.0, total_tokens=2000, query_count=1, unique_domains=1, dm_coverage=0.0, cv_coverage=0.6, objectives_covered=1, cs_effort=0.0),
        _make_snapshot(vitals_snapshot_healthy, loop_index=1, findings_count=6, coverage_score=0.5, total_tokens=4000, query_count=2, unique_domains=1, dm_coverage=0.0, cv_coverage=0.6, objectives_covered=2, cs_effort=0.0),
        _make_snapshot(vitals_snapshot_healthy, loop_index=2, findings_count=9, coverage_score=0.5, total_tokens=6000, query_count=3, unique_domains=1, dm_coverage=0.0, cv_coverage=0.6, objectives_covered=2, cs_effort=0.0),
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=12, coverage_score=0.5, total_tokens=8000, query_count=4, unique_domains=1, dm_coverage=0.0, cv_coverage=0.6, objectives_covered=2, cs_effort=0.0)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is False
    assert result.stuck_trigger != "short_run_objective_gap"


# ---------------------------------------------------------------------------
# Burn rate anomaly
# ---------------------------------------------------------------------------


def test_burn_rate_no_trigger_without_token_spike(vitals_snapshot_healthy: dict) -> None:
    """Burn-rate anomaly should not trigger without a token spike."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.1 * (i + 2), total_tokens=1000 * (i + 1), query_count=i + 1, unique_domains=i + 1, dm_coverage=0.4, cv_coverage=0.2)
        for i in range(3)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=3, coverage_score=0.5, total_tokens=4000, query_count=4, unique_domains=4, dm_coverage=0.4, cv_coverage=0.2)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is False


def test_burn_rate_triggers_with_token_spike(vitals_snapshot_healthy: dict) -> None:
    """Burn-rate anomaly should trigger when tokens spike with zero findings delta."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.1 * (i + 2), total_tokens=1000 * (i + 1), query_count=i + 1, unique_domains=i + 1, dm_coverage=0.4, cv_coverage=0.2)
        for i in range(3)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=3, coverage_score=0.5, total_tokens=10000, query_count=4, unique_domains=4, dm_coverage=0.4, cv_coverage=0.2)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is True
    assert result.stuck_trigger == "burn_rate_anomaly"


# ---------------------------------------------------------------------------
# Grace period (dm_coverage)
# ---------------------------------------------------------------------------


def test_grace_period_blocks_before_four_steps(vitals_snapshot_healthy: dict) -> None:
    """Three consecutive dm=0.0 steps should not trigger stuck detection."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=0, findings_count=1, coverage_score=0.2, total_tokens=100, query_count=1, unique_domains=1, dm_coverage=0.0, cv_coverage=0.6),
        _make_snapshot(vitals_snapshot_healthy, loop_index=1, findings_count=2, coverage_score=0.4, total_tokens=250, query_count=2, unique_domains=2, dm_coverage=0.0, cv_coverage=0.6),
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=2, findings_count=3, coverage_score=0.6, total_tokens=400, query_count=3, unique_domains=3, dm_coverage=0.0, cv_coverage=0.6)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is False


def test_grace_period_triggers_at_fifth_step(vitals_snapshot_healthy: dict) -> None:
    """Five consecutive dm=0.0 steps should trigger stuck detection."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.2 * min(i + 1, 4), total_tokens=100 + 150 * i, query_count=i + 1, unique_domains=i + 1, dm_coverage=0.0, cv_coverage=0.6)
        for i in range(4)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=4, findings_count=5, coverage_score=0.8, total_tokens=700, query_count=5, unique_domains=5, dm_coverage=0.0, cv_coverage=0.6)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is True
    assert result.stuck_trigger == "coverage_stagnation"


def test_grace_period_resets_on_recovery(vitals_snapshot_healthy: dict) -> None:
    """Recovery at step four should reset the dm=0.0 streak."""
    config = VitalsConfig()
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=i, findings_count=i + 1, coverage_score=0.2 * (i + 1), total_tokens=100 + 150 * i, query_count=i + 1, unique_domains=i + 1, dm_coverage=0.0, cv_coverage=0.6)
        for i in range(3)
    ]
    current = _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=4, coverage_score=0.8, total_tokens=550, query_count=4, unique_domains=4, dm_coverage=0.2, cv_coverage=0.6)

    result = detect_loop(current, history, config=config)
    assert result.stuck_detected is False


# ---------------------------------------------------------------------------
# High-coverage suppression (token normalization)
# ---------------------------------------------------------------------------


def _build_burn_rate_trace(
    base: dict,
    *,
    token_multiplier: float = 1.0,
    coverage: float = 0.5,
) -> list[VitalsSnapshot]:
    """Build a trace where burn rate anomaly should fire."""
    return [
        _make_snapshot(base, loop_index=0, findings_count=0, total_tokens=int(500 * token_multiplier), coverage_score=0.2, dm_coverage=0.0, cv_coverage=0.0, query_count=1, unique_domains=1),
        _make_snapshot(base, loop_index=1, findings_count=2, total_tokens=int(1500 * token_multiplier), coverage_score=0.4, dm_coverage=0.3, cv_coverage=0.3, query_count=1, unique_domains=1),
        _make_snapshot(base, loop_index=2, findings_count=4, total_tokens=int(2500 * token_multiplier), coverage_score=coverage, dm_coverage=0.5, cv_coverage=0.5, query_count=1, unique_domains=1),
        _make_snapshot(base, loop_index=3, findings_count=4, total_tokens=int(5000 * token_multiplier), coverage_score=coverage, dm_coverage=0.3, cv_coverage=0.3, query_count=1, unique_domains=1),
        _make_snapshot(base, loop_index=4, findings_count=4, total_tokens=int(10000 * token_multiplier), coverage_score=coverage, dm_coverage=0.2, cv_coverage=0.2, query_count=1, unique_domains=1),
    ]


def test_burn_rate_fires_at_low_coverage(vitals_snapshot_healthy: dict) -> None:
    """Burn rate anomaly should fire when coverage is below 0.95."""
    trace = _build_burn_rate_trace(vitals_snapshot_healthy, coverage=0.5)
    config = VitalsConfig()
    result = detect_loop(trace[-1], trace[:-1], config=config)
    assert result.stuck_detected is True
    assert result.stuck_trigger == "burn_rate_anomaly"


def test_burn_rate_suppressed_at_high_coverage(vitals_snapshot_healthy: dict) -> None:
    """Burn rate anomaly should NOT fire when coverage >= 0.95."""
    trace = _build_burn_rate_trace(vitals_snapshot_healthy, coverage=1.0)
    config = VitalsConfig()
    result = detect_loop(trace[-1], trace[:-1], config=config)
    if result.stuck_detected:
        assert result.stuck_trigger != "burn_rate_anomaly"
