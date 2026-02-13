"""Unit tests for loop/stuck detection — ported from DeepSearch test suite.

Tests cover: healthy progression, findings plateau, coverage stagnation,
build workflow skip, early FP avoidance, short_run_objective_gap disabled,
burn rate anomaly, grace period, and high-coverage suppression.
"""

from __future__ import annotations

import copy

import pytest

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import _proportional_window, detect_loop
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
    payload["metrics"]["dm_coverage"] = dm_coverage
    payload["metrics"]["cv_coverage"] = cv_coverage
    if output_similarity is not None:
        payload["output_similarity"] = output_similarity
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
    """If both detectors fire, higher loop confidence should win."""
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


def test_detector_priority_stuck_wins_on_higher_confidence(
    vitals_snapshot_healthy: dict,
) -> None:
    """If both detectors fire, higher stuck confidence should win."""
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
    assert result.loop_detected is False
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
    config = VitalsConfig(loop_consecutive_pct=1.0)
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
