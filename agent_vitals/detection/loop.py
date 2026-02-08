"""Loop/stuck detection helpers for Agent Vitals.

The loop breaker analyzes vitals snapshots over time and emits detection flags
that can be used for logging, calibration, and enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Optional, Sequence

from ..config import VitalsConfig, get_vitals_config
from ..schema import VitalsSnapshot

SHORT_RUN_MAX_STEPS = 4
SHORT_RUN_MIN_FINDINGS = 3
SHORT_RUN_OBJECTIVE_MAX = 3


@dataclass(frozen=True, slots=True)
class LoopDetectionResult:
    """Detection result for loop/stuck analysis."""

    loop_detected: bool = False
    loop_confidence: float = 0.0
    loop_trigger: Optional[str] = None

    stuck_detected: bool = False
    stuck_confidence: float = 0.0
    stuck_trigger: Optional[str] = None

    def as_snapshot_update(self) -> dict[str, object]:
        """Return a Pydantic-compatible update mapping for VitalsSnapshot."""

        return {
            "loop_detected": bool(self.loop_detected),
            "loop_confidence": float(_clip01(self.loop_confidence)),
            "loop_trigger": self.loop_trigger,
            "stuck_detected": bool(self.stuck_detected),
            "stuck_confidence": float(_clip01(self.stuck_confidence)),
            "stuck_trigger": self.stuck_trigger,
        }


def detect_loop(
    snapshot: VitalsSnapshot,
    history: Optional[Sequence[VitalsSnapshot]] = None,
    *,
    config: Optional[VitalsConfig] = None,
    workflow_type: str = "unknown",
) -> LoopDetectionResult:
    """Analyze a vitals snapshot for loop/stuck indicators.

    Args:
        snapshot: Current vitals snapshot.
        history: Previous snapshots (oldest → newest). The current snapshot
            should not be included.
        config: Optional VitalsConfig override (defaults to env-derived config).
        workflow_type: Workflow type hint ("research", "build", "unknown").

    Returns:
        LoopDetectionResult with detection flags, confidence, and triggers.
    """

    cfg = config or get_vitals_config()
    prior = list(history or [])
    series = prior + [snapshot]
    if len(series) < 2:
        return LoopDetectionResult()

    consecutive = max(1, int(cfg.loop_consecutive_count))
    normalized_workflow = _normalize_workflow_type(workflow_type)

    findings_counts = [int(item.signals.findings_count) for item in series]
    sources_counts: list[int] = []
    sources_missing = False
    objectives_counts: list[int] = []
    objectives_missing = False
    for item in series:
        sources_value = getattr(item.signals, "sources_count", None)
        if sources_value is None:
            sources_missing = True
        else:
            sources_counts.append(int(sources_value))
        objectives_value = getattr(item.signals, "objectives_covered", None)
        if objectives_value is None:
            objectives_missing = True
        else:
            objectives_counts.append(int(objectives_value))
    query_counts = [int(item.signals.query_count) for item in series]
    domain_counts = [int(item.signals.unique_domains) for item in series]
    coverage_scores = [float(item.signals.coverage_score) for item in series]
    token_totals = [float(item.signals.total_tokens) for item in series]

    findings_deltas = _deltas(findings_counts)
    query_deltas = _deltas(query_counts)
    domain_deltas = _deltas(domain_counts)
    coverage_deltas = _deltas(coverage_scores)
    token_deltas = _deltas(token_totals)

    loop_candidates: list[tuple[float, str]] = []

    if len(findings_deltas) >= consecutive:
        recent_findings = findings_deltas[-consecutive:]
        plateau = all(delta <= 0.0 for delta in recent_findings)

        coverage_flat = False
        if len(coverage_deltas) >= consecutive:
            coverage_flat = all(abs(delta) <= 1e-3 for delta in coverage_deltas[-consecutive:])

        if plateau and coverage_flat:
            loop_candidates.append((0.85, "findings_plateau+coverage_flat"))
        elif plateau:
            loop_candidates.append((0.75, "findings_plateau"))

        if (
            plateau
            and len(query_deltas) >= consecutive
            and len(domain_deltas) >= consecutive
            and all(delta > 0.0 for delta in query_deltas[-consecutive:])
            and all(delta <= 0.0 for delta in domain_deltas[-consecutive:])
        ):
            loop_candidates.append((0.9, "query_repetition_proxy"))

    loop_detected = False
    loop_confidence = 0.0
    loop_trigger: Optional[str] = None
    if loop_candidates:
        loop_confidence, loop_trigger = max(loop_candidates, key=lambda item: item[0])
        loop_detected = True

    if not _stuck_enabled_for_workflow(normalized_workflow, cfg.workflow_stuck_enabled):
        # Build workflows can skip stuck detection; thrash covers deterministic failures.
        return LoopDetectionResult(
            loop_detected=loop_detected,
            loop_confidence=_clip01(loop_confidence),
            loop_trigger=loop_trigger,
            stuck_detected=False,
            stuck_confidence=0.0,
            stuck_trigger=None,
        )

    stuck_candidates: list[tuple[float, str]] = []

    dm_coverage = float(snapshot.metrics.dm_coverage)
    cv_coverage = float(snapshot.metrics.cv_coverage)

    # Coverage stagnation: only evaluate once enough points exist for DM lookback.
    # Grace period of 5 steps to eliminate FPs on slow-converging models.
    # When dm_coverage is 0.0 for 5+ consecutive steps, treat as sufficient evidence of stuck.
    dm_coverage_series = [float(item.metrics.dm_coverage) for item in series]
    dm_zero_streak = 0
    for value in reversed(dm_coverage_series):
        if value == 0.0:
            dm_zero_streak += 1
        else:
            break

    if dm_zero_streak >= 5 and coverage_scores[-1] < 0.95:
        # Strict dm_coverage=0.0 gate after grace period: stuck regardless of cv_coverage.
        # Suppress when coverage is near-maximum (convergence, not stuck).
        stuck_candidates.append((0.75, "coverage_stagnation"))
    elif len(series) >= 5 and coverage_scores[-1] < 0.95:
        # Standard check: require both dm and cv below thresholds.
        # Suppress when coverage is near-maximum (convergence, not stuck).
        if dm_coverage <= float(cfg.stuck_dm_threshold) and cv_coverage <= float(cfg.stuck_cv_threshold):
            dm_factor = _relative_margin_below(dm_coverage, float(cfg.stuck_dm_threshold))
            cv_factor = _relative_margin_below(cv_coverage, float(cfg.stuck_cv_threshold))
            stuck_candidates.append((0.6 + 0.4 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation"))
    elif len(series) >= 4 and coverage_scores[-1] < 0.95:
        # Short-run severe stagnation — DM/CV both well below half-threshold.
        half_dm = float(cfg.stuck_dm_threshold) / 2.0
        half_cv = float(cfg.stuck_cv_threshold) / 2.0
        if dm_coverage <= half_dm and cv_coverage <= half_cv:
            dm_factor = _relative_margin_below(dm_coverage, half_dm)
            cv_factor = _relative_margin_below(cv_coverage, half_cv)
            stuck_candidates.append((0.6 + 0.3 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation"))

    # Zero progress: early detection for fixture-like failures with no activity.
    loop_index = int(snapshot.loop_index)
    findings_count = int(snapshot.signals.findings_count)
    total_tokens = int(snapshot.signals.total_tokens)
    if total_tokens == 0:
        if loop_index >= 1 and findings_count <= 1:
            stuck_candidates.append((0.80, "zero_progress"))
        elif loop_index >= 0 and findings_count == 0:
            stuck_candidates.append((0.80, "zero_progress"))

    # AV-20: short_run_objective_gap DISABLED.
    # The len(series)==N equality check fires on every trace passing through step N,
    # not just genuinely short runs. This caused FPs on all healthy traces at step N-1.
    # Thrash detection via derive_stop_signals (error_count threshold) catches the
    # retry_thrash cases this was designed for, with no recall loss.

    # Coverage flatline: repeated zero deltas with low variation suggests stagnation.
    # Suppress when coverage is near-maximum — flat coverage at 1.0 is convergence, not stuck.
    if len(coverage_deltas) >= consecutive and coverage_scores[-1] < 0.95:
        recent = coverage_deltas[-consecutive:]
        if cv_coverage <= float(cfg.stuck_cv_threshold) and all(abs(delta) <= 1e-3 for delta in recent):
            stuck_candidates.append((0.65, "coverage_flat"))

    # Findings plateau: no new findings for consecutive steps despite continued token usage.
    # Suppress when coverage is near-maximum — findings plateau at full coverage is convergence.
    plateau_window = 3
    if (
        len(series) >= 5
        and len(findings_deltas) >= plateau_window
        and len(token_deltas) >= plateau_window
        and coverage_scores[-1] < 0.95
    ):
        recent_findings = findings_deltas[-plateau_window:]
        recent_tokens = token_deltas[-plateau_window:]
        if all(delta == 0.0 for delta in recent_findings) and all(delta > 0.0 for delta in recent_tokens):
            stuck_candidates.append((0.7, "findings_plateau"))

    # Sources-zero: research runs with no sources after 5+ steps.
    if (
        not sources_missing
        and normalized_workflow == "research"
        and len(series) >= 5
        and sources_counts
        and max(sources_counts) == 0
    ):
        stuck_candidates.append((0.8, "sources_zero"))

    # Late-onset stagnation: coverage flattens/regresses with no findings gain.
    stagnation_window = 2
    if (
        len(series) >= 5
        and len(coverage_deltas) >= stagnation_window
        and len(findings_deltas) >= stagnation_window
        and len(token_deltas) >= stagnation_window
    ):
        recent_coverage = coverage_deltas[-stagnation_window:]
        recent_findings = findings_deltas[-stagnation_window:]
        recent_tokens = token_deltas[-stagnation_window:]
        coverage_regression = any(delta < 0.0 for delta in recent_coverage)
        objectives_plateau = False
        if not objectives_missing and len(objectives_counts) >= len(series):
            objectives_deltas = _deltas(objectives_counts)
            if len(objectives_deltas) >= stagnation_window:
                objectives_plateau = all(delta <= 0.0 for delta in objectives_deltas[-stagnation_window:])
        coverage_non_increasing = all(delta <= 0.0 for delta in recent_coverage)
        findings_stalled = any(delta <= 0.0 for delta in recent_findings)
        tokens_active = all(delta > 0.0 for delta in recent_tokens)
        # Suppress late-onset stagnation when coverage is near-maximum.
        coverage_ready = 0.5 <= coverage_scores[-1] < 0.95
        dm_low = dm_coverage <= float(cfg.stuck_dm_threshold)
        if coverage_non_increasing and findings_stalled and tokens_active and coverage_ready:
            if (coverage_regression and dm_low) or (not coverage_regression and objectives_plateau):
                stuck_candidates.append((0.7, "late_onset_stagnation"))

    # Token burn rate anomaly: tokens per new finding spikes vs baseline.
    # Apply token_scale_factor to normalize token counts before burn rate comparison.
    # Suppress when coverage is near-maximum — token burn at full coverage is expected.
    scale = max(0.01, float(cfg.token_scale_factor))
    if len(token_deltas) >= 1 and len(findings_deltas) >= 1 and coverage_scores[-1] < 0.95:
        current_tokens = token_deltas[-1] * scale
        current_findings = findings_deltas[-1]

        baseline_ratios: list[float] = []
        baseline_token_deltas: list[float] = []
        for index in range(0, len(findings_deltas) - 1):  # exclude current step
            d_findings = findings_deltas[index]
            d_tokens = token_deltas[index] * scale
            if d_tokens <= 0.0:
                continue
            baseline_token_deltas.append(d_tokens)
            if d_findings > 0.0:
                baseline_ratios.append(d_tokens / d_findings)

        if baseline_ratios and current_tokens > 0.0:
            baseline = float(mean(baseline_ratios))
            if baseline > 0.0:
                ratio = math.inf if current_findings <= 0.0 else float(current_tokens / current_findings)
                if ratio > float(cfg.burn_rate_multiplier) * baseline:
                    if current_findings <= 0.0 and baseline_token_deltas:
                        token_baseline = float(mean(baseline_token_deltas))
                        token_threshold = float(cfg.burn_rate_multiplier) * token_baseline
                        if current_tokens <= token_threshold:
                            pass
                        else:
                            factor = min(
                                1.0,
                                float(ratio / (float(cfg.burn_rate_multiplier) * baseline)),
                            )
                            stuck_candidates.append(
                                (min(1.0, 0.7 + 0.3 * factor), "burn_rate_anomaly")
                            )
                    else:
                        factor = min(
                            1.0,
                            float(ratio / (float(cfg.burn_rate_multiplier) * baseline)),
                        )
                        stuck_candidates.append((min(1.0, 0.7 + 0.3 * factor), "burn_rate_anomaly"))

    stuck_detected = False
    stuck_confidence = 0.0
    stuck_trigger: Optional[str] = None
    if stuck_candidates:
        stuck_confidence, stuck_trigger = max(stuck_candidates, key=lambda item: item[0])
        stuck_detected = True

    return LoopDetectionResult(
        loop_detected=loop_detected,
        loop_confidence=_clip01(loop_confidence),
        loop_trigger=loop_trigger,
        stuck_detected=stuck_detected,
        stuck_confidence=_clip01(stuck_confidence),
        stuck_trigger=stuck_trigger,
    )


# Backwards-compatible alias
detect_agent_loop = detect_loop


def _deltas(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return []
    return [float(values[index] - values[index - 1]) for index in range(1, len(values))]


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _relative_margin_below(value: float, threshold: float) -> float:
    """Return a normalized [0,1] margin for how far value is below threshold."""

    if not math.isfinite(value) or not math.isfinite(threshold):
        return 0.0
    denom = max(abs(threshold), 1e-9)
    margin = max(0.0, (threshold - value) / denom)
    return _clip01(margin)


def _normalize_workflow_type(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"research", "build", "unknown", "synthetic"}:
        return candidate
    return "unknown"


def _stuck_enabled_for_workflow(workflow_type: str, mode: str) -> bool:
    normalized_mode = str(mode or "").strip().lower().replace("_", "-")
    if normalized_mode in {"none", "off", "disabled"}:
        return False
    if normalized_mode in {"all", "both", "enabled"}:
        return True
    if normalized_mode in {"build-only", "build"}:
        return workflow_type == "build"
    if normalized_mode in {"research-only", "research"}:
        return workflow_type != "build"
    return workflow_type != "build"


__all__ = ["LoopDetectionResult", "detect_agent_loop", "detect_loop"]
