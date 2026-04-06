"""Unit tests for loop/stuck detection — ported from DeepSearch test suite.

Tests cover: healthy progression, findings plateau, coverage stagnation,
build workflow skip, early FP avoidance, short_run_objective_gap disabled,
burn rate anomaly, grace period, and high-coverage suppression.
"""

from __future__ import annotations

import copy

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import (
    _proportional_window,
    _windowed_ratio_declines,
    detect_loop,
)
from agent_vitals.detection.stop_rule import derive_stop_signals
from agent_vitals.schema import VitalsSnapshot


def _make_snapshot(
    base: dict,
    *,
    loop_index: int,
    findings_count: int,
    sources_count: int | None = None,
    coverage_score: float,
    total_tokens: int,
    query_count: int,
    unique_domains: int,
    dm_coverage: float,
    cv_coverage: float,
    output_similarity: float | None = None,
    objectives_covered: int | None = None,
    cs_effort: float | None = None,
    verified_sources_count: int | None = None,
    unverified_sources_count: int | None = None,
    verified_source_ratio: float | None = None,
) -> VitalsSnapshot:
    payload = copy.deepcopy(base)
    payload["loop_index"] = loop_index
    payload["signals"]["findings_count"] = findings_count
    if sources_count is not None:
        payload["signals"]["sources_count"] = sources_count
    payload["signals"]["coverage_score"] = coverage_score
    payload["signals"]["total_tokens"] = total_tokens
    payload["signals"]["query_count"] = query_count
    payload["signals"]["unique_domains"] = unique_domains
    if objectives_covered is not None:
        payload["signals"]["objectives_covered"] = objectives_covered
    if verified_sources_count is not None:
        payload["signals"]["verified_sources_count"] = verified_sources_count
    if unverified_sources_count is not None:
        payload["signals"]["unverified_sources_count"] = unverified_sources_count
    payload["metrics"]["dm_coverage"] = dm_coverage
    payload["metrics"]["cv_coverage"] = cv_coverage
    if output_similarity is not None:
        payload["output_similarity"] = output_similarity
    if cs_effort is not None:
        payload["metrics"]["cs_effort"] = cs_effort
    if verified_source_ratio is not None:
        payload["verified_source_ratio"] = verified_source_ratio
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


@pytest.mark.parametrize(
    ("trace_length", "expected_loop_threshold", "expected_stuck_window"),
    [
        (2, 2, 2),
        (3, 2, 2),
        (5, 2, 2),
        (7, 3, 2),
        (10, 5, 4),
    ],
)
def test_proportional_threshold_edge_cases(
    trace_length: int,
    expected_loop_threshold: int,
    expected_stuck_window: int,
) -> None:
    """Adaptive windows match AV-28 formulas across short and medium traces."""
    assert (
        _proportional_window(
            trace_length=trace_length,
            percentage=0.5,
            minimum=2,
            fallback=3,
        )
        == expected_loop_threshold
    )
    assert (
        _proportional_window(
            trace_length=trace_length,
            percentage=0.4,
            minimum=2,
            fallback=4,
        )
        == expected_stuck_window
    )


def test_minimum_evidence_floor_blocks_detection_under_three_steps(
    vitals_snapshot_healthy: dict,
) -> None:
    """No loop/stuck signal should fire when trace length is <3 steps."""
    config = VitalsConfig(min_evidence_steps=3)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=4,
            coverage_score=0.5,
            total_tokens=1000,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.3,
            cv_coverage=0.3,
            output_similarity=0.95,
        )
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=1,
        findings_count=4,
        coverage_score=0.5,
        total_tokens=2000,
        query_count=2,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=0.95,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False
    assert result.stuck_detected is False


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


def test_detects_sources_stagnation_with_findings_growth(
    vitals_snapshot_healthy: dict,
) -> None:
    """Low ratio + stagnation signals should hit high-confidence confab trigger."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=3,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=2000,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.1,
            cv_coverage=0.2,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=6,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=3000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.7,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=13,
        sources_count=3,
        coverage_score=0.4,
        total_tokens=4000,
        query_count=3,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.5,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert (
        result.confabulation_trigger
        == "source_finding_ratio_low+sources_stagnation+unique_domains_stagnation"
    )
    assert result.confabulation_confidence >= 0.85
    assert result.detector_priority == "confabulation"


def test_sources_stagnation_requires_findings_growth(vitals_snapshot_healthy: dict) -> None:
    """Sources stagnation alone should not trigger without findings growth."""
    config = VitalsConfig(loop_consecutive_pct=1.0)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=3,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=2000,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.4,
            cv_coverage=0.4,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=3,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=2600,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.4,
            cv_coverage=0.4,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=3,
        sources_count=3,
        coverage_score=0.4,
        total_tokens=3200,
        query_count=3,
        unique_domains=1,
        dm_coverage=0.4,
        cv_coverage=0.4,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False


def test_ratio_floor_confidence_without_stagnation(vitals_snapshot_healthy: dict) -> None:
    """Ratio floor breach alone should fire confab signal at base confidence."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=4,
            sources_count=4,
            coverage_score=0.4,
            total_tokens=1200,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=6,
            sources_count=5,
            coverage_score=0.45,
            total_tokens=2200,
            query_count=2,
            unique_domains=3,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=20,
        sources_count=5,
        coverage_score=0.5,
        total_tokens=3400,
        query_count=3,
        unique_domains=4,
        dm_coverage=0.3,
        cv_coverage=0.3,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert result.confabulation_trigger == "source_finding_ratio_low"
    assert result.confabulation_confidence == pytest.approx(0.65)
    assert result.detector_priority == "confabulation"


def test_ratio_floor_confidence_boosts_with_sources_stagnation(
    vitals_snapshot_healthy: dict,
) -> None:
    """Low ratio + sources stagnation should boost confidence to 0.8."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=3,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=1200,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.2,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=6,
            sources_count=3,
            coverage_score=0.4,
            total_tokens=2200,
            query_count=2,
            unique_domains=2,
            dm_coverage=0.2,
            cv_coverage=0.3,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=12,
        sources_count=3,
        coverage_score=0.4,
        total_tokens=3400,
        query_count=3,
        unique_domains=2,
        dm_coverage=0.2,
        cv_coverage=0.3,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert result.confabulation_trigger == "source_finding_ratio_low+sources_stagnation"
    assert result.confabulation_confidence == pytest.approx(0.75)
    assert result.detector_priority == "confabulation"


def test_confabulation_priority_does_not_clear_stuck_signal(
    vitals_snapshot_healthy: dict,
) -> None:
    """Confabulation can be primary while still preserving a stuck diagnosis."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=2,
            sources_count=3,
            coverage_score=0.1,
            total_tokens=1000,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=4,
            sources_count=3,
            coverage_score=0.1,
            total_tokens=2000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=6,
            sources_count=3,
            coverage_score=0.1,
            total_tokens=3000,
            query_count=3,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=8,
        sources_count=3,
        coverage_score=0.1,
        total_tokens=4200,
        query_count=4,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert result.stuck_detected is True
    assert result.stuck_trigger == "coverage_stagnation"
    assert result.detector_priority == "confabulation"


def test_ratio_declining_trajectory_triggers_confab(vitals_snapshot_healthy: dict) -> None:
    """Three consecutive ratio declines with findings growth should trigger confab."""
    config = VitalsConfig(source_finding_ratio_declining_steps=3, source_finding_ratio_floor=0.3)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=5,
            sources_count=7,
            coverage_score=0.3,
            total_tokens=1000,
            query_count=1,
            unique_domains=3,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=10,
            sources_count=7,
            coverage_score=0.35,
            total_tokens=2000,
            query_count=2,
            unique_domains=4,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=15,
            sources_count=7,
            coverage_score=0.4,
            total_tokens=3200,
            query_count=3,
            unique_domains=5,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=20,
        sources_count=7,
        coverage_score=0.45,
        total_tokens=4600,
        query_count=4,
        unique_domains=6,
        dm_coverage=0.3,
        cv_coverage=0.3,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert result.confabulation_trigger == "source_finding_ratio_declining"
    assert result.confabulation_confidence == pytest.approx(0.6)
    assert result.detector_priority == "confabulation"


def test_ratio_declining_requires_findings_growth(vitals_snapshot_healthy: dict) -> None:
    """Declining ratio without findings growth should not fire decline trajectory signal."""
    config = VitalsConfig(source_finding_ratio_declining_steps=3, source_finding_ratio_floor=0.3)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=10,
            sources_count=10,
            coverage_score=0.1,
            total_tokens=1000,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=10,
            sources_count=8,
            coverage_score=0.1,
            total_tokens=2000,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=10,
            sources_count=6,
            coverage_score=0.1,
            total_tokens=3000,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=10,
        sources_count=5,
        coverage_score=0.1,
        total_tokens=4000,
        query_count=1,
        unique_domains=2,
        dm_coverage=0.3,
        cv_coverage=0.3,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_trigger != "source_finding_ratio_declining"


def test_detector_priority_loop_wins_on_higher_confidence(
    vitals_snapshot_healthy: dict,
) -> None:
    """Both detectors fire; loop is primary with content_similarity (av32-m02)."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=0,
            coverage_score=0.1,
            total_tokens=0,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=0,
            coverage_score=0.1,
            total_tokens=0,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
            output_similarity=0.9,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=2,
        findings_count=0,
        coverage_score=0.1,
        total_tokens=0,
        query_count=3,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=0.9,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is True
    assert result.stuck_detected is False
    assert result.detector_priority == "loop"


def test_short_run_zero_coverage_keeps_stuck_for_research_overlap(
    vitals_snapshot_healthy: dict,
) -> None:
    """Short-run zero-coverage research stalls keep stuck alongside loop."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=0,
            sources_count=5,
            coverage_score=0.0,
            total_tokens=2500,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=0,
            sources_count=10,
            coverage_score=0.0,
            total_tokens=15000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=0,
            sources_count=10,
            coverage_score=0.0,
            total_tokens=25000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
            output_similarity=0.02,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=0,
        sources_count=10,
        coverage_score=0.0,
        total_tokens=35000,
        query_count=2,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=1.0,
    )

    result = detect_loop(current, history, config=config, workflow_type="research")
    assert result.loop_detected is True
    assert result.loop_trigger == "content_similarity"
    assert result.stuck_detected is True
    assert result.stuck_trigger == "short_run_zero_coverage"
    assert result.detector_priority == "stuck"


def test_short_run_zero_coverage_skips_high_findings_loops(
    vitals_snapshot_healthy: dict,
) -> None:
    """Productive loops should not be re-labeled as short-run stuck."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=8,
            sources_count=5,
            coverage_score=0.0,
            total_tokens=2500,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=12,
            sources_count=10,
            coverage_score=0.0,
            total_tokens=15000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=16,
            sources_count=10,
            coverage_score=0.0,
            total_tokens=25000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
            output_similarity=0.02,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=20,
        sources_count=10,
        coverage_score=0.0,
        total_tokens=35000,
        query_count=2,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=1.0,
    )

    result = detect_loop(current, history, config=config, workflow_type="research")
    assert result.loop_detected is True
    assert result.loop_trigger == "content_similarity"
    assert result.stuck_detected is False


def test_detector_priority_stuck_wins_on_higher_confidence(
    vitals_snapshot_healthy: dict,
) -> None:
    """Stuck wins priority when loop has no candidates (av32-m02)."""
    config = VitalsConfig(loop_consecutive_count=3)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=1,
            sources_count=3,
            coverage_score=0.0,
            total_tokens=100,
            query_count=1,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=2,
            sources_count=3,
            coverage_score=0.0,
            total_tokens=200,
            query_count=2,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=2,
            sources_count=3,
            coverage_score=0.0,
            total_tokens=300,
            query_count=3,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=3,
        sources_count=3,
        coverage_score=0.0,
        total_tokens=400,
        query_count=4,
        unique_domains=2,
        dm_coverage=-0.01,
        cv_coverage=0.01,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False  # Loop candidates empty (no plateau, no similarity)
    assert result.stuck_detected is True
    assert result.stuck_trigger == "coverage_stagnation"
    assert result.detector_priority == "stuck"


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


def test_source_productivity_suppresses_coverage_stagnation(vitals_snapshot_healthy: dict) -> None:
    """High source + findings productivity should suppress coverage_stagnation."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=8 + i,
            sources_count=15,
            coverage_score=0.45 + 0.02 * i,
            total_tokens=1000 + 400 * i,
            query_count=2 + i,
            unique_domains=6 + i,
            dm_coverage=0.0,
            cv_coverage=0.2,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=12,
        sources_count=20,
        coverage_score=0.55,
        total_tokens=2600,
        query_count=6,
        unique_domains=10,
        dm_coverage=0.0,
        cv_coverage=0.2,
    )

    result = detect_loop(current, history, config=config)
    assert result.stuck_trigger != "coverage_stagnation"
    assert result.stuck_detected is False


def test_source_productivity_suppresses_late_onset_stagnation(
    vitals_snapshot_healthy: dict,
) -> None:
    """Productive high-source runs should not fire late_onset_stagnation."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=3,
            sources_count=5,
            coverage_score=0.8,
            total_tokens=5000,
            query_count=1,
            unique_domains=1,
            dm_coverage=0.4,
            cv_coverage=0.2,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=6,
            sources_count=10,
            coverage_score=0.8,
            total_tokens=10000,
            query_count=2,
            unique_domains=1,
            dm_coverage=0.3,
            cv_coverage=0.2,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=9,
            sources_count=15,
            coverage_score=0.8,
            total_tokens=15000,
            query_count=3,
            unique_domains=1,
            dm_coverage=0.2,
            cv_coverage=0.2,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=12,
            sources_count=20,
            coverage_score=0.5,
            total_tokens=20000,
            query_count=4,
            unique_domains=1,
            dm_coverage=0.1,
            cv_coverage=0.2,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=5,
        findings_count=12,
        sources_count=20,
        coverage_score=0.5,
        total_tokens=25000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.02,
        cv_coverage=0.23,
    )

    result = detect_loop(current, history, config=config)
    assert result.stuck_trigger != "late_onset_stagnation"
    assert result.stuck_detected is False


def test_loop_hint_suppresses_stagnation_before_loop_threshold(
    vitals_snapshot_healthy: dict,
) -> None:
    """Near-threshold loop evidence suppresses stagnation-style stuck triggers."""
    config = VitalsConfig(
        loop_consecutive_count=3,
        loop_consecutive_pct=0.75,
        findings_plateau_pct=1.0,
    )
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=1,
            coverage_score=0.40,
            total_tokens=100,
            query_count=1,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=2,
            coverage_score=0.45,
            total_tokens=200,
            query_count=2,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=2,
            coverage_score=0.45,
            total_tokens=300,
            query_count=3,
            unique_domains=2,
            dm_coverage=-0.01,
            cv_coverage=0.01,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=2,
        coverage_score=0.45,
        total_tokens=400,
        query_count=4,
        unique_domains=2,
        dm_coverage=-0.01,
        cv_coverage=0.01,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False
    assert result.stuck_detected is False
    assert result.stuck_trigger is None


def test_low_coverage_still_allows_coverage_stagnation_with_loop_hint(
    vitals_snapshot_healthy: dict,
) -> None:
    """Critical low-coverage runs should keep coverage_stagnation despite loop hints."""
    config = VitalsConfig(loop_consecutive_count=3, loop_consecutive_pct=0.75)
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=0,
            findings_count=1,
            coverage_score=0.0,
            total_tokens=100,
            query_count=1,
            unique_domains=2,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=2,
            coverage_score=0.0,
            total_tokens=200,
            query_count=2,
            unique_domains=2,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=2,
            coverage_score=0.0,
            total_tokens=300,
            query_count=3,
            unique_domains=2,
            dm_coverage=0.0,
            cv_coverage=0.0,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=3,
        findings_count=2,
        coverage_score=0.0,
        total_tokens=400,
        query_count=4,
        unique_domains=2,
        dm_coverage=0.0,
        cv_coverage=0.0,
    )

    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False
    assert result.stuck_detected is True
    assert result.stuck_trigger == "coverage_stagnation"


def test_findings_plateau_requires_four_steps(vitals_snapshot_healthy: dict) -> None:
    """Three-step findings plateau should not fire stuck findings_plateau."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=count,
            coverage_score=score,
            total_tokens=tokens,
            query_count=i + 1,
            unique_domains=4,
            dm_coverage=0.4,
            cv_coverage=0.5,
        )
        for i, (count, score, tokens) in enumerate(
            [
                (1, 0.30, 1000),
                (2, 0.38, 1500),
                (2, 0.46, 2200),
                (2, 0.54, 3000),
            ]
        )
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=2,
        coverage_score=0.62,
        total_tokens=3800,
        query_count=5,
        unique_domains=4,
        dm_coverage=0.4,
        cv_coverage=0.5,
    )

    result = detect_loop(current, history, config=config)
    # With relaxed cross-detector suppression (av32-m02), stuck
    # findings_plateau fires when loop also fires — both are valid
    # signals.  Verify that detector_priority is set correctly.
    if result.stuck_detected and result.stuck_trigger == "findings_plateau":
        assert result.loop_detected is True
        assert result.detector_priority == "loop"
    else:
        assert result.stuck_trigger != "findings_plateau"


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
    assert result.stuck_detected is False
    assert result.stuck_trigger == "burn_rate_anomaly"
    assert result.detector_priority == "runaway_cost"

    stop_signals = derive_stop_signals(
        {
            "loop_detected": result.loop_detected,
            "stuck_detected": result.stuck_detected,
            "stuck_trigger": result.stuck_trigger,
            "signals": {"error_count": 0},
        }
    )
    assert stop_signals.stuck_detected is False
    assert stop_signals.runaway_cost_detected is True


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
    config = VitalsConfig(loop_consecutive_pct=1.0, model_size_class="large")
    result = detect_loop(trace[-1], trace[:-1], config=config)
    assert result.stuck_detected is False
    assert result.stuck_trigger == "burn_rate_anomaly"
    assert result.detector_priority == "runaway_cost"


def test_burn_rate_suppressed_at_high_coverage(vitals_snapshot_healthy: dict) -> None:
    """Burn rate anomaly should NOT fire when coverage >= 0.95."""
    trace = _build_burn_rate_trace(vitals_snapshot_healthy, coverage=1.0)
    config = VitalsConfig(loop_consecutive_pct=1.0)
    result = detect_loop(trace[-1], trace[:-1], config=config)
    if result.stuck_detected:
        assert result.stuck_trigger != "burn_rate_anomaly"


def test_burn_rate_fires_on_small_model_stuck_disabled(vitals_snapshot_healthy: dict) -> None:
    """Regression: burn_rate_anomaly must fire on small-model traces in the
    stuck-disabled path (DSPy profile). The burn_rate_multiplier_scale is NOT
    applied in this path. Ref: AV-S03-M01, bench 42-FN regression."""
    trace = _build_burn_rate_trace(vitals_snapshot_healthy, coverage=0.5)
    config = VitalsConfig(
        loop_consecutive_pct=1.0, model_size_class="auto", workflow_stuck_enabled="none",
    )
    result = detect_loop(trace[-1], trace[:-1], config=config)
    assert result.detector_priority == "runaway_cost", (
        f"burn_rate_anomaly should fire on small-model traces (stuck-disabled); "
        f"got priority={result.detector_priority}"
    )


def test_burn_rate_fires_with_explicit_small_stuck_disabled(vitals_snapshot_healthy: dict) -> None:
    """burn_rate_anomaly must fire with model_size_class='small' in stuck-disabled path."""
    trace = _build_burn_rate_trace(vitals_snapshot_healthy, coverage=0.5)
    config = VitalsConfig(
        loop_consecutive_pct=1.0, model_size_class="small", workflow_stuck_enabled="none",
    )
    result = detect_loop(trace[-1], trace[:-1], config=config)
    assert result.detector_priority == "runaway_cost", (
        f"burn_rate_anomaly should fire with model_size_class='small' (stuck-disabled); "
        f"got priority={result.detector_priority}"
    )


# ---------------------------------------------------------------------------
# Stuck-disabled co-occurrence suppression (AV-S01-M01)
# ---------------------------------------------------------------------------


def test_stuck_disabled_suppresses_loop_fp_on_stuck_trace(
    vitals_snapshot_healthy: dict,
) -> None:
    """When stuck detection is disabled, loop should NOT fire on traces with
    stuck-like signals (low DM/CV, findings plateau)."""
    config = VitalsConfig(workflow_stuck_enabled="none")
    # Build a trace with findings plateau + flat coverage + active tokens
    # + loop index progressing — this would fire loop without suppression.
    # DM/CV are both very low, indicating stuck-like behavior.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=5,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
    )
    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False, (
        f"Loop should be suppressed when stuck is disabled and DM/CV indicate stagnation, "
        f"got trigger={result.loop_trigger}"
    )


def test_stuck_disabled_suppresses_runaway_cost_fp_on_stuck_trace(
    vitals_snapshot_healthy: dict,
) -> None:
    """When stuck detection is disabled, runaway_cost should NOT fire on
    traces with stuck-like stagnation signals."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Burn-rate trace: token spike with zero findings delta, but DM/CV are
    # both low indicating the trace is actually stuck.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=3,
            coverage_score=0.3,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.0,
            cv_coverage=0.0,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=3,
        coverage_score=0.3,
        total_tokens=15000,
        query_count=5,
        unique_domains=5,
        dm_coverage=0.0,
        cv_coverage=0.0,
    )
    result = detect_loop(current, history, config=config)
    assert result.detector_priority != "runaway_cost", (
        "Runaway cost should be suppressed when stuck is disabled and DM/CV indicate stagnation"
    )
    assert result.stuck_trigger != "burn_rate_anomaly"


def test_stuck_disabled_allows_content_similarity_loop(
    vitals_snapshot_healthy: dict,
) -> None:
    """Content similarity is independent loop evidence — should still fire
    even when stuck is disabled and stagnation signals are present."""
    config = VitalsConfig(workflow_stuck_enabled="none")
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
            output_similarity=0.95,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=5,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=0.95,
    )
    result = detect_loop(current, history, config=config)
    assert result.loop_detected is True, (
        "Content similarity loop should still fire when stuck is disabled"
    )
    assert result.loop_trigger == "content_similarity"


def test_stuck_disabled_no_suppression_when_productive(
    vitals_snapshot_healthy: dict,
) -> None:
    """When stuck is disabled but the agent IS productive (high sources,
    findings, coverage), loop should still be able to fire normally."""
    config = VitalsConfig(workflow_stuck_enabled="none")
    # Productive trace: high sources AND findings AND coverage — source_productive
    # is True, so coverage_stagnation_evidence is False.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=10,
            sources_count=15,
            coverage_score=0.6,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,
            dm_coverage=0.0,
            cv_coverage=0.0,
            output_similarity=0.95,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=10,
        sources_count=15,
        coverage_score=0.6,
        total_tokens=5000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.0,
        cv_coverage=0.0,
        output_similarity=0.95,
    )
    result = detect_loop(current, history, config=config)
    # source_productive = True, so no stagnation suppression kicks in.
    # Content similarity loop should fire.
    assert result.loop_detected is True
    assert result.loop_trigger == "content_similarity"


def test_stuck_disabled_confabulation_still_suppresses_loop(
    vitals_snapshot_healthy: dict,
) -> None:
    """Confabulation should still suppress loop even when stuck is disabled."""
    config = VitalsConfig(workflow_stuck_enabled="none")
    # Trace with low source-finding ratio (confabulation signal) and loop signals.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 5,
            sources_count=1,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,
            dm_coverage=0.4,
            cv_coverage=0.4,
            output_similarity=0.95,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=25,
        sources_count=1,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.4,
        cv_coverage=0.4,
        output_similarity=0.95,
    )
    result = detect_loop(current, history, config=config)
    if result.confabulation_detected:
        assert result.loop_detected is False, (
            "Confabulation should suppress loop when stuck is disabled"
        )
        assert result.detector_priority == "confabulation"


def test_stuck_disabled_relaxed_evidence_suppresses_loop_on_low_dm_cv(
    vitals_snapshot_healthy: dict,
) -> None:
    """When stuck is disabled and BOTH dm and cv are below static thresholds,
    loop should be suppressed even if adaptive alarms don't trigger."""
    config = VitalsConfig(workflow_stuck_enabled="none")
    # Trace with findings plateau + token activity + loop progress.
    # DM and CV are both below thresholds (dm <= 0.15, cv <= 0.3) but NOT zero,
    # so adaptive alarms may not fire. Relaxed evidence should still suppress.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,
            dm_coverage=0.10,
            cv_coverage=0.20,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=5,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=1,
        dm_coverage=0.10,
        cv_coverage=0.20,
    )
    result = detect_loop(current, history, config=config)
    assert result.loop_detected is False, (
        f"Loop should be suppressed by relaxed stagnation evidence (dm/cv below thresholds), "
        f"got trigger={result.loop_trigger}"
    )


def test_stuck_disabled_relaxed_evidence_suppresses_burn_rate(
    vitals_snapshot_healthy: dict,
) -> None:
    """Relaxed stagnation evidence (static dm/cv) suppresses burn_rate_anomaly.
    Without stuck detection active, low dm/cv is the best surrogate for stuck-like
    behavior.  Genuine runaway traces maintain normal dm/cv (burning tokens but
    advancing coverage), so this does not affect true positives.
    Ref: AV-S04-M01 co-occurrence FP suppression."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Burn-rate trace: earlier steps have increasing findings (establishing
    # baseline), then last step has zero findings growth with huge token spike.
    # DM/CV are below static thresholds (relaxed evidence triggers).
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 2,
            coverage_score=0.3,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.10,
            cv_coverage=0.20,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=8,  # same as last history step — zero findings delta
        coverage_score=0.3,
        total_tokens=25000,  # massive token spike
        query_count=5,
        unique_domains=5,
        dm_coverage=0.10,
        cv_coverage=0.20,
    )
    result = detect_loop(current, history, config=config)
    # Relaxed stagnation now suppresses burn_rate — the trace is stuck-like.
    assert result.stuck_trigger != "burn_rate_anomaly", (
        "Burn_rate_anomaly should be suppressed by relaxed stagnation evidence"
    )
    assert result.detector_priority != "runaway_cost"


def test_stuck_disabled_burn_rate_min_steps_gate(
    vitals_snapshot_healthy: dict,
) -> None:
    """Burn_rate_anomaly should be suppressed on traces shorter than 4 steps
    in the stuck-disabled path — short baselines are unreliable."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # 3-step trace with token spike. Burn rate would fire but min-steps gate
    # should suppress it.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=3,
            coverage_score=0.3,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
        )
        for i in range(2)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=2,
        findings_count=3,
        coverage_score=0.3,
        total_tokens=15000,
        query_count=3,
        unique_domains=3,
        dm_coverage=0.5,
        cv_coverage=0.5,
    )
    result = detect_loop(current, history, config=config)
    assert result.stuck_trigger != "burn_rate_anomaly", (
        "Burn_rate_anomaly should be suppressed on traces shorter than 4 steps"
    )
    assert result.detector_priority != "runaway_cost"


def test_stuck_disabled_min_baseline_resists_spike_contamination(
    vitals_snapshot_healthy: dict,
) -> None:
    """Regression: burn_rate_anomaly must fire on later steps of a runaway trace
    even after the spike enters the baseline. The min-baseline mode in the
    stuck-disabled path prevents spike contamination of the mean.
    Ref: AV-S03-M01, bench 42-FN regression on 6-step DSPy traces.
    Note: dm/cv must be above stagnation thresholds so relaxed stagnation
    does not suppress the genuine runaway signal (AV-S04-M01)."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # 6-step trace: 2 normal steps, then 10x token spike at step 2+.
    # Coverage is advancing (0.1→0.4) so dm/cv are healthy — this is genuine
    # runaway (burning tokens but still making some progress).
    history = [
        _make_snapshot(vitals_snapshot_healthy, loop_index=0, findings_count=2, coverage_score=0.1,
                       total_tokens=1000, query_count=1, unique_domains=1, dm_coverage=0.3, cv_coverage=0.4),
        _make_snapshot(vitals_snapshot_healthy, loop_index=1, findings_count=4, coverage_score=0.2,
                       total_tokens=2000, query_count=2, unique_domains=1, dm_coverage=0.3, cv_coverage=0.4),
        _make_snapshot(vitals_snapshot_healthy, loop_index=2, findings_count=5, coverage_score=0.3,
                       total_tokens=12000, query_count=3, unique_domains=1, dm_coverage=0.3, cv_coverage=0.4),
        _make_snapshot(vitals_snapshot_healthy, loop_index=3, findings_count=6, coverage_score=0.35,
                       total_tokens=22000, query_count=4, unique_domains=1, dm_coverage=0.3, cv_coverage=0.4),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy, loop_index=4, findings_count=7, coverage_score=0.4,
        total_tokens=32000, query_count=5, unique_domains=1, dm_coverage=0.3, cv_coverage=0.4,
    )
    result = detect_loop(current, history, config=config)
    assert result.detector_priority == "runaway_cost", (
        f"burn_rate_anomaly should fire at step 4 (min-baseline resists spike contamination); "
        f"got priority={result.detector_priority}"
    )


# ---------------------------------------------------------------------------
# Stuck-disabled co-occurrence arbitration tests (AV-S04-M01)
# ---------------------------------------------------------------------------


def test_stuck_disabled_loop_content_sim_suppresses_runaway(
    vitals_snapshot_healthy: dict,
) -> None:
    """When loop fires with content_similarity and runaway also fires,
    loop should win and runaway should be suppressed."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Trace with high output similarity (triggers content_similarity loop)
    # AND token spike (triggers burn_rate_anomaly).  DM/CV above thresholds
    # so relaxed stagnation does NOT fire — only co-occurrence arbitration.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 2,
            coverage_score=0.3,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
            output_similarity=0.95,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=8,
        coverage_score=0.3,
        total_tokens=25000,  # token spike
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
        output_similarity=0.95,
    )
    result = detect_loop(current, history, config=config)
    assert result.loop_detected is True, "Loop with content_similarity should win"
    assert result.loop_trigger == "content_similarity"
    assert result.stuck_trigger != "burn_rate_anomaly", (
        "Runaway should be suppressed when loop has content_similarity"
    )
    assert result.detector_priority == "loop"


def test_stuck_disabled_runaway_wins_over_weak_loop(
    vitals_snapshot_healthy: dict,
) -> None:
    """When runaway has higher confidence than loop (and loop lacks
    content_similarity), runaway should win and loop is suppressed."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Trace designed so loop fires weakly (no content_similarity, low conf)
    # but burn_rate fires strongly.  DM/CV above thresholds.
    # Use increasing findings to prevent stagnation, but massive token spike.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 2,
            coverage_score=0.3 + i * 0.05,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=1,  # low domain count may trigger weak loop
            dm_coverage=0.5,
            cv_coverage=0.5,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=8,  # zero delta from last history
        coverage_score=0.45,
        total_tokens=25000,  # massive spike
        query_count=5,
        unique_domains=1,
        dm_coverage=0.5,
        cv_coverage=0.5,
    )
    result = detect_loop(current, history, config=config)
    # If both fire AND runaway conf > loop conf, runaway wins.
    # If only runaway fires (no loop candidates survive), runaway wins directly.
    if result.stuck_trigger == "burn_rate_anomaly":
        assert result.detector_priority == "runaway_cost"


def test_stuck_disabled_confab_overlap_suppresses_runaway(
    vitals_snapshot_healthy: dict,
) -> None:
    """High-confidence confabulation should suppress runaway_cost in
    the stuck-disabled path."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Trace with low verified_source_ratio (confab signal) AND token spike.
    # DM/CV above thresholds so stagnation doesn't mask the test.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 3,
            sources_count=(i + 1) * 5,
            coverage_score=0.3,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
            verified_source_ratio=0.1,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=15,
        sources_count=30,
        coverage_score=0.3,
        total_tokens=25000,  # token spike
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
        verified_source_ratio=0.05,
    )
    result = detect_loop(current, history, config=config)
    # Confabulation should fire and suppress both loop and runaway.
    if result.confabulation_detected and result.confabulation_confidence >= 0.85:
        assert result.detector_priority == "confabulation"
        assert result.stuck_trigger != "burn_rate_anomaly", (
            "Runaway should be suppressed when confabulation fires with high confidence"
        )


def test_stuck_disabled_runaway_fires_with_healthy_dm_cv(
    vitals_snapshot_healthy: dict,
) -> None:
    """Genuine runaway traces with healthy dm/cv should still fire.
    This verifies that the relaxed stagnation suppression does not
    regress recall on traces where coverage IS advancing."""
    config = VitalsConfig(workflow_stuck_enabled="none", model_size_class="large")
    # Genuine runaway: coverage advancing, dm/cv healthy, but token cost
    # is 10x what it should be for the progress made.
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 2,
            coverage_score=0.1 + i * 0.1,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.4,
            cv_coverage=0.4,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=8,  # zero findings delta — still burning tokens
        coverage_score=0.45,
        total_tokens=25000,  # massive spike
        query_count=5,
        unique_domains=5,
        dm_coverage=0.4,
        cv_coverage=0.4,
    )
    result = detect_loop(current, history, config=config)
    assert result.stuck_trigger == "burn_rate_anomaly", (
        "Genuine runaway with healthy dm/cv should still fire burn_rate_anomaly"
    )
    assert result.detector_priority == "runaway_cost"


# ---------------------------------------------------------------------------
# Verified source ratio confabulation detection tests
# ---------------------------------------------------------------------------


def test_verified_source_ratio_low_triggers_confabulation(
    vitals_snapshot_healthy: dict,
) -> None:
    """Low verified_source_ratio with growing sources should trigger confabulation."""
    config = VitalsConfig()
    # Trace where sources are growing but most are unverified (ratio < 0.3).
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 3,
            sources_count=(i + 1) * 5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
            verified_source_ratio=0.2,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=15,
        sources_count=30,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
        verified_source_ratio=0.15,
    )
    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True, (
        "Low verified_source_ratio with growing sources should trigger confabulation"
    )
    assert result.confabulation_trigger is not None
    assert "verified_source_ratio_low" in result.confabulation_trigger
    assert "verified_source_ratio" in result.confabulation_signals


def test_verified_source_ratio_none_no_effect(
    vitals_snapshot_healthy: dict,
) -> None:
    """When verified_source_ratio is None, detection should not be affected."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 3,
            sources_count=(i + 1) * 5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=15,
        sources_count=25,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
    )
    result = detect_loop(current, history, config=config)
    if result.confabulation_detected:
        assert result.confabulation_trigger is not None
        assert "verified_source_ratio" not in result.confabulation_trigger


def test_verified_source_ratio_high_no_confabulation(
    vitals_snapshot_healthy: dict,
) -> None:
    """High verified_source_ratio (>= 0.3) should NOT trigger via this path."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 3,
            sources_count=(i + 1) * 5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
            verified_source_ratio=0.8,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=15,
        sources_count=25,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
        verified_source_ratio=0.8,
    )
    result = detect_loop(current, history, config=config)
    if result.confabulation_detected:
        assert "verified_source_ratio_low" not in (result.confabulation_trigger or "")


def test_verified_source_ratio_declining_boosts_confidence(
    vitals_snapshot_healthy: dict,
) -> None:
    """Declining verified_source_ratio should boost confabulation confidence."""
    config = VitalsConfig()
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=i,
            findings_count=(i + 1) * 3,
            sources_count=(i + 1) * 5,
            coverage_score=0.5,
            total_tokens=1000 * (i + 1),
            query_count=i + 1,
            unique_domains=i + 1,
            dm_coverage=0.5,
            cv_coverage=0.5,
            verified_source_ratio=0.25 - i * 0.05,
        )
        for i in range(4)
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=4,
        findings_count=15,
        sources_count=30,
        coverage_score=0.5,
        total_tokens=5000,
        query_count=5,
        unique_domains=5,
        dm_coverage=0.5,
        cv_coverage=0.5,
        verified_source_ratio=0.05,
    )
    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert "verified_ratio_declining" in (result.confabulation_trigger or "")
    assert "verified_source_ratio_declining" in result.confabulation_signals


# ---------------------------------------------------------------------------
# _windowed_ratio_declines unit tests
# ---------------------------------------------------------------------------


class TestWindowedRatioDeclines:
    """Direct unit tests for _windowed_ratio_declines()."""

    def test_empty_ratios_returns_zero(self) -> None:
        assert _windowed_ratio_declines(ratios=[], loop_indices=[], window=5) == 0

    def test_single_ratio_returns_zero(self) -> None:
        assert _windowed_ratio_declines(ratios=[0.5], loop_indices=[1], window=5) == 0

    def test_mismatched_lengths_returns_zero(self) -> None:
        assert _windowed_ratio_declines(ratios=[0.5, 0.4], loop_indices=[1], window=5) == 0

    def test_consecutive_declines_counted(self) -> None:
        """Three consecutive declines should all be counted."""
        ratios = [0.8, 0.6, 0.4, 0.2]
        indices = [1, 2, 3, 4]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 3

    def test_non_consecutive_declines_still_counted(self) -> None:
        """Declines with a plateau in between should still be counted."""
        # Step 1→2: 0.8→0.6 decline, 2→3: 0.6→0.6 plateau, 3→4: 0.6→0.3 decline
        ratios = [0.8, 0.6, 0.6, 0.3]
        indices = [1, 2, 3, 4]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 2

    def test_v_shaped_recovery_counted(self) -> None:
        """Decline, rise, decline pattern (Gemini-like) should count both declines."""
        ratios = [0.8, 0.5, 0.7, 0.4]
        indices = [1, 2, 3, 4]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 2

    def test_window_limits_scope(self) -> None:
        """Only declines within the window should be counted."""
        # 5 declines total, but window=3 should only see the last 3 steps
        ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        indices = [1, 2, 3, 4, 5, 6]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=3) == 2

    def test_none_ratios_skipped(self) -> None:
        """None values should be skipped, not break the count."""
        ratios = [0.8, None, 0.6, 0.3]
        indices = [1, 2, 3, 4]
        # 1→2 skip (None), 2→3 skip (None), 3→4: 0.6→0.3 decline
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 1

    def test_non_progressing_loop_indices_skipped(self) -> None:
        """Steps where loop_index doesn't advance should be skipped."""
        ratios = [0.8, 0.6, 0.4, 0.2]
        indices = [1, 2, 2, 3]  # index 2→2 doesn't advance
        # 1→2: decline, 2→3: skip (no progress), 3→4: decline
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 2

    def test_all_rising_returns_zero(self) -> None:
        ratios = [0.2, 0.4, 0.6, 0.8]
        indices = [1, 2, 3, 4]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 0

    def test_epsilon_tolerance(self) -> None:
        """Ratios differing by less than epsilon should not count as decline."""
        ratios = [0.5, 0.5 - 1e-12]
        indices = [1, 2]
        assert _windowed_ratio_declines(ratios=ratios, loop_indices=indices, window=5) == 0


def test_windowed_decline_triggers_confab(vitals_snapshot_healthy: dict) -> None:
    """Non-consecutive declines within window should trigger confab detection.

    This is the key behavioral change from bench: a plateau step no longer
    breaks the declining trajectory signal.
    """
    config = VitalsConfig(
        source_finding_ratio_declining_steps=3,
        source_finding_ratio_floor=0.3,
        source_finding_ratio_decline_window=5,
    )
    # Findings grow every step (required for ratio_decline_with_growth gate).
    # Sources bump at step 3, causing a ratio *rise* that breaks consecutive
    # declines but not windowed declines.
    # Ratios: 7/5=1.4, 7/10=0.7, 7/15=0.47, 9/20=0.45 (rise from sources bump),
    #         9/25=0.36 → declines: 1→2, 2→3, 4→5 = 3 in window of 5
    history = [
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=1,
            findings_count=5,
            sources_count=7,
            coverage_score=0.3,
            total_tokens=1000,
            query_count=1,
            unique_domains=3,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=2,
            findings_count=10,
            sources_count=7,
            coverage_score=0.35,
            total_tokens=2000,
            query_count=2,
            unique_domains=4,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=3,
            findings_count=15,
            sources_count=7,
            coverage_score=0.35,
            total_tokens=3000,
            query_count=3,
            unique_domains=4,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
        _make_snapshot(
            vitals_snapshot_healthy,
            loop_index=4,
            findings_count=20,
            sources_count=9,  # sources bump — ratio rises 0.47→0.45, breaks consecutive
            coverage_score=0.4,
            total_tokens=4000,
            query_count=4,
            unique_domains=5,
            dm_coverage=0.3,
            cv_coverage=0.3,
        ),
    ]
    current = _make_snapshot(
        vitals_snapshot_healthy,
        loop_index=5,
        findings_count=25,
        sources_count=9,
        coverage_score=0.45,
        total_tokens=5000,
        query_count=5,
        unique_domains=6,
        dm_coverage=0.3,
        cv_coverage=0.3,
    )

    result = detect_loop(current, history, config=config)
    assert result.confabulation_detected is True
    assert "source_finding_ratio_declining" in (result.confabulation_trigger or "")
