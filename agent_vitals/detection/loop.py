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
from .adaptive_threshold import AdaptiveThreshold

SHORT_RUN_MAX_STEPS = 4
SHORT_RUN_MIN_FINDINGS = 3
SHORT_RUN_OBJECTIVE_MAX = 3
# Legacy fallback; active window is now trace-length proportional.
FINDINGS_PLATEAU_WINDOW = 4
SOURCE_PRODUCTIVITY_MIN_SOURCES = 10
SOURCE_PRODUCTIVITY_MIN_FINDINGS = 5
SOURCES_STAGNATION_WINDOW = 2
UNIQUE_DOMAINS_STAGNATION_WINDOW = 3
SOURCES_STAGNATION_MAX_SOURCES = 3


@dataclass(frozen=True, slots=True)
class LoopDetectionResult:
    """Detection result for loop/stuck analysis."""

    loop_detected: bool = False
    loop_confidence: float = 0.0
    loop_trigger: Optional[str] = None

    confabulation_detected: bool = False
    confabulation_confidence: float = 0.0
    confabulation_trigger: Optional[str] = None
    confabulation_signals: tuple[str, ...] = ()

    stuck_detected: bool = False
    stuck_confidence: float = 0.0
    stuck_trigger: Optional[str] = None
    detector_priority: Optional[str] = None

    def as_snapshot_update(self) -> dict[str, object]:
        """Return a Pydantic-compatible update mapping for VitalsSnapshot."""

        return {
            "loop_detected": bool(self.loop_detected),
            "loop_confidence": float(_clip01(self.loop_confidence)),
            "loop_trigger": self.loop_trigger,
            "confabulation_detected": bool(self.confabulation_detected),
            "confabulation_confidence": float(_clip01(self.confabulation_confidence)),
            "confabulation_trigger": self.confabulation_trigger,
            "confabulation_signals": list(self.confabulation_signals),
            "stuck_detected": bool(self.stuck_detected),
            "stuck_confidence": float(_clip01(self.stuck_confidence)),
            "stuck_trigger": self.stuck_trigger,
            "detector_priority": self.detector_priority,
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
    min_evidence_steps = max(1, int(cfg.min_evidence_steps))
    if len(series) < max(2, min_evidence_steps):
        return LoopDetectionResult()

    # AV-28: adaptive loop threshold scales with trace length.
    consecutive = _proportional_window(
        trace_length=len(series),
        percentage=float(cfg.loop_consecutive_pct),
        minimum=2,
        fallback=max(1, int(cfg.loop_consecutive_count)),
    )
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
    loop_indices = [int(item.loop_index) for item in series]
    loop_index_deltas = _deltas(loop_indices)
    query_deltas = _deltas(query_counts)
    domain_deltas = _deltas(domain_counts)
    coverage_deltas = _deltas(coverage_scores)
    token_deltas = _deltas(token_totals)
    findings_count = int(snapshot.signals.findings_count)
    sources_count = int(snapshot.signals.sources_count)
    source_finding_ratio = _source_finding_ratio(
        sources_count=sources_count,
        findings_count=findings_count,
    )
    ratio_floor = max(0.0, float(cfg.source_finding_ratio_floor))
    ratio_declining_required = max(1, int(cfg.source_finding_ratio_declining_steps))
    ratio_series: list[Optional[float]] = []
    for item in series:
        snapshot_ratio = getattr(item, "source_finding_ratio", None)
        if snapshot_ratio is None:
            snapshot_ratio = _source_finding_ratio(
                sources_count=int(item.signals.sources_count),
                findings_count=int(item.signals.findings_count),
            )
        ratio_series.append(snapshot_ratio)
    ratio_declining_steps = _consecutive_ratio_declines(
        ratios=ratio_series,
        loop_indices=loop_indices,
    )
    source_productive = (
        sources_count >= SOURCE_PRODUCTIVITY_MIN_SOURCES
        and findings_count >= SOURCE_PRODUCTIVITY_MIN_FINDINGS
        and float(snapshot.signals.coverage_score) >= 0.5
    )
    dm_coverage_series = [float(item.metrics.dm_coverage) for item in series]
    cv_coverage_series = [float(item.metrics.cv_coverage) for item in series]
    dm_stagnation_alarm, dm_stagnation_threshold, _, _ = _adaptive_alarm_from_series(
        values=dm_coverage_series,
        direction="decrease",
        fallback_threshold=float(cfg.stuck_dm_threshold),
        config=cfg,
    )
    cv_stagnation_alarm, cv_stagnation_threshold, _, _ = _adaptive_alarm_from_series(
        values=cv_coverage_series,
        direction="decrease",
        fallback_threshold=float(cfg.stuck_cv_threshold),
        config=cfg,
    )
    token_variance_series = _token_variance_series(
        values=token_totals,
        window_size=max(2, int(cfg.spc_window_size)),
    )
    token_variance_alarm = False
    if token_variance_series:
        token_variance_alarm, _, _, _ = _adaptive_alarm_from_series(
            values=token_variance_series,
            direction="decrease",
            fallback_threshold=None,
            config=cfg,
        )

    loop_candidates: list[tuple[float, str]] = []
    confab_candidates: list[tuple[float, str, tuple[str, ...]]] = []

    if len(findings_deltas) >= consecutive:
        recent_findings = findings_deltas[-consecutive:]
        plateau = all(delta <= 0.0 for delta in recent_findings)
        loop_progressing = (
            len(loop_index_deltas) >= consecutive
            and all(delta > 0.0 for delta in loop_index_deltas[-consecutive:])
        )

        # Require token activity — plateau WITH active work = loop,
        # plateau WITHOUT active work = stuck (not repeating, just idle).
        tokens_active = (
            len(token_deltas) >= consecutive
            and all(delta > 0.0 for delta in token_deltas[-consecutive:])
        )

        coverage_flat = False
        if len(coverage_deltas) >= consecutive:
            coverage_flat = all(abs(delta) <= 1e-3 for delta in coverage_deltas[-consecutive:])
        coverage_nontrivial = coverage_scores[-1] >= 0.2

        if (
            plateau
            and coverage_flat
            and tokens_active
            and loop_progressing
            and coverage_nontrivial
            and not source_productive
        ):
            loop_candidates.append((0.85, "findings_plateau+coverage_flat"))
        elif (
            plateau
            and tokens_active
            and loop_progressing
            and coverage_nontrivial
            and not source_productive
        ):
            loop_candidates.append((0.75, "findings_plateau"))

        if (
            plateau
            and loop_progressing
            and len(query_deltas) >= consecutive
            and len(domain_deltas) >= consecutive
            and all(delta > 0.0 for delta in query_deltas[-consecutive:])
            and all(delta <= 0.0 for delta in domain_deltas[-consecutive:])
            and max(domain_counts) > 0  # agent must have found domains to stagnate
        ):
            loop_candidates.append((0.9, "query_repetition_proxy"))

    # Confabulation indicator: source-to-finding ratio and trajectory.
    # Primary signal is low ratio; stagnation signals boost confidence.
    sources_stagnation = False
    unique_domains_stagnation = False
    if (
        not sources_missing
        and len(sources_deltas := _deltas(sources_counts)) >= SOURCES_STAGNATION_WINDOW
        and len(findings_deltas) >= SOURCES_STAGNATION_WINDOW
        and sources_counts
        and max(sources_counts) > 0
        and int(sources_counts[-1]) <= SOURCES_STAGNATION_MAX_SOURCES
    ):
        recent_source_deltas = sources_deltas[-SOURCES_STAGNATION_WINDOW:]
        recent_finding_deltas = findings_deltas[-SOURCES_STAGNATION_WINDOW:]
        sources_flat = all(delta == 0.0 for delta in recent_source_deltas)
        findings_growing = sum(recent_finding_deltas) > 0.0 and all(
            delta >= 0.0 for delta in recent_finding_deltas
        )
        sources_stagnation = sources_flat and findings_growing
        if sources_stagnation and len(domain_counts) >= UNIQUE_DOMAINS_STAGNATION_WINDOW:
            recent_domains = domain_counts[-UNIQUE_DOMAINS_STAGNATION_WINDOW:]
            unique_domains_stagnation = all(int(value) <= 1 for value in recent_domains)

    has_source_evidence = bool(sources_counts) and max(sources_counts) > 0
    ratio_floor_breach = False
    ratio_spc_ready = False
    if has_source_evidence and source_finding_ratio is not None:
        observed_ratio_risk = [
            max(0.0, ratio_floor - float(value))
            for value in ratio_series
            if value is not None
        ]
        ratio_floor_breach, _, _, ratio_spc_ready = _adaptive_alarm_from_series(
            values=observed_ratio_risk,
            direction="increase",
            fallback_threshold=0.0,
            config=cfg,
        )
    ratio_decline_with_growth = False
    if (
        has_source_evidence
        and source_finding_ratio is not None
        and source_finding_ratio <= 1.0
        and
        ratio_declining_steps >= ratio_declining_required
        and len(findings_deltas) >= ratio_declining_required
        and len(loop_index_deltas) >= ratio_declining_required
    ):
        recent_findings = findings_deltas[-ratio_declining_required:]
        recent_loop_steps = loop_index_deltas[-ratio_declining_required:]
        ratio_decline_with_growth = all(delta > 0.0 for delta in recent_findings) and all(
            delta > 0.0 for delta in recent_loop_steps
        )

    if ratio_floor_breach or ratio_decline_with_growth:
        confab_confidence = 0.60
        trigger_parts: list[str] = []
        signal_parts: list[str] = []
        if ratio_floor_breach:
            trigger_parts.append("source_finding_ratio_low")
            signal_parts.append("source_finding_ratio")
        if ratio_decline_with_growth:
            trigger_parts.append("source_finding_ratio_declining")
            signal_parts.append("source_finding_ratio_declining")

        if "findings_count_delta" in set(getattr(snapshot, "cusum_alarm_metrics", []) or []):
            confab_confidence += 0.15
            signal_parts.append("cusum_findings_count_delta")

        if ratio_floor_breach and ratio_spc_ready:
            confab_confidence += 0.05
            signal_parts.append("spc_ratio_threshold")

        if sources_stagnation:
            confab_confidence += 0.10
            trigger_parts.append("sources_stagnation")
            signal_parts.append("sources_stagnation")
            if unique_domains_stagnation:
                confab_confidence += 0.10
                trigger_parts.append("unique_domains_stagnation")
                signal_parts.append("unique_domains_stagnation")

        confab_candidates.append(
            (
                min(0.95, confab_confidence),
                "+".join(trigger_parts),
                tuple(dict.fromkeys(signal_parts)),
            )
        )

    # Content similarity gate: high output similarity is an independent
    # loop signal (agent producing near-identical outputs across iterations).
    sim_threshold = float(cfg.loop_similarity_threshold)
    output_similarity = getattr(snapshot, "output_similarity", None)
    if output_similarity is not None:
        similarity = float(output_similarity)
        similarity_series = [
            float(item.output_similarity)
            for item in series
            if getattr(item, "output_similarity", None) is not None
        ]
        similarity_alarm, similarity_threshold, _, _ = _adaptive_alarm_from_series(
            values=similarity_series,
            direction="increase",
            fallback_threshold=sim_threshold,
            config=cfg,
        )
        similarity_fixed_hit = similarity >= sim_threshold
        if similarity_alarm or similarity_fixed_hit:
            # Exact or near-exact repeat — strong loop signal
            active_threshold = _clip01(min(similarity_threshold, sim_threshold))
            confidence = 0.80 + 0.15 * min(
                1.0,
                (similarity - active_threshold) / max(1e-9, 1.0 - active_threshold),
            )
            loop_candidates.append((confidence, "content_similarity"))

    # Suppress loop when errors are present — error-induced plateaus are
    # thrash behavior, not repetitive looping.
    error_count = int(snapshot.signals.error_count)
    if error_count > 0:
        loop_candidates.clear()

    loop_detected = False
    loop_confidence = 0.0
    loop_candidate_confidence = 0.0
    loop_trigger: Optional[str] = None
    if loop_candidates:
        loop_confidence, loop_trigger = max(loop_candidates, key=lambda item: item[0])
        loop_candidate_confidence = loop_confidence
        loop_detected = True

    confabulation_detected = False
    confabulation_confidence = 0.0
    confabulation_trigger: Optional[str] = None
    confabulation_signals: tuple[str, ...] = ()
    if confab_candidates:
        (
            confabulation_confidence,
            confabulation_trigger,
            confabulation_signals,
        ) = max(confab_candidates, key=lambda item: item[0])
        confabulation_detected = True

    if not _stuck_enabled_for_workflow(normalized_workflow, cfg.workflow_stuck_enabled):
        # Build workflows can skip stuck detection; thrash covers deterministic failures.
        detector_priority = "confabulation" if confabulation_detected else ("loop" if loop_detected else None)
        if confabulation_detected:
            loop_detected = False
            loop_confidence = 0.0
            loop_trigger = None
        return LoopDetectionResult(
            loop_detected=loop_detected,
            loop_confidence=_clip01(loop_confidence),
            loop_trigger=loop_trigger,
            confabulation_detected=confabulation_detected,
            confabulation_confidence=_clip01(confabulation_confidence),
            confabulation_trigger=confabulation_trigger,
            confabulation_signals=confabulation_signals,
            stuck_detected=False,
            stuck_confidence=0.0,
            stuck_trigger=None,
            detector_priority=detector_priority,
        )

    stuck_candidates: list[tuple[float, str]] = []

    dm_coverage = float(snapshot.metrics.dm_coverage)
    cv_coverage = float(snapshot.metrics.cv_coverage)
    dm_low_signal = dm_stagnation_alarm or dm_coverage <= float(cfg.stuck_dm_threshold)
    cv_low_signal = cv_stagnation_alarm or cv_coverage <= float(cfg.stuck_cv_threshold)
    dm_reference_threshold = (
        dm_stagnation_threshold if dm_stagnation_alarm else float(cfg.stuck_dm_threshold)
    )
    cv_reference_threshold = (
        cv_stagnation_threshold if cv_stagnation_alarm else float(cfg.stuck_cv_threshold)
    )

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

    if not source_productive and dm_zero_streak >= 5 and coverage_scores[-1] < 0.95:
        # Strict dm_coverage=0.0 gate after grace period: stuck regardless of cv_coverage.
        # Suppress when coverage is near-maximum (convergence, not stuck).
        stuck_candidates.append((0.75, "coverage_stagnation"))
    elif not source_productive and len(series) >= 5 and coverage_scores[-1] < 0.95:
        # Standard check: require both dm and cv below thresholds.
        # Suppress when coverage is near-maximum (convergence, not stuck).
        if dm_low_signal and cv_low_signal:
            dm_factor = _relative_margin_below(dm_coverage, dm_reference_threshold)
            cv_factor = _relative_margin_below(cv_coverage, cv_reference_threshold)
            stuck_candidates.append((0.6 + 0.4 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation"))
    elif not source_productive and len(series) >= 4 and coverage_scores[-1] < 0.95:
        # Short-run severe stagnation — DM/CV both well below half-threshold.
        half_dm = float(cfg.stuck_dm_threshold) / 2.0
        half_cv = float(cfg.stuck_cv_threshold) / 2.0
        if dm_coverage <= half_dm and cv_coverage <= half_cv:
            dm_factor = _relative_margin_below(dm_coverage, half_dm)
            cv_factor = _relative_margin_below(cv_coverage, half_cv)
            stuck_candidates.append((0.6 + 0.3 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation"))

    # Zero progress: early detection for fixture-like failures with no activity.
    loop_index = int(snapshot.loop_index)
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
        loop_progressing = (
            len(loop_index_deltas) >= consecutive
            and all(delta > 0.0 for delta in loop_index_deltas[-consecutive:])
        )
        if (
            cv_low_signal
            and loop_progressing
            and all(abs(delta) <= 1e-3 for delta in recent)
        ):
            stuck_candidates.append((0.65, "coverage_flat"))

    # Findings plateau: no new findings for consecutive steps despite continued token usage.
    # Suppress when coverage is near-maximum — findings plateau at full coverage is convergence.
    plateau_window = _proportional_window(
        trace_length=len(series),
        percentage=float(cfg.findings_plateau_pct),
        minimum=2,
        fallback=FINDINGS_PLATEAU_WINDOW,
    )
    if (
        len(series) >= (plateau_window + 1)
        and len(findings_deltas) >= plateau_window
        and len(loop_index_deltas) >= plateau_window
        and len(token_deltas) >= plateau_window
        and coverage_scores[-1] < 0.95
    ):
        recent_findings = findings_deltas[-plateau_window:]
        recent_loop_deltas = loop_index_deltas[-plateau_window:]
        recent_tokens = token_deltas[-plateau_window:]
        if (
            all(delta > 0.0 for delta in recent_loop_deltas)
            and all(delta == 0.0 for delta in recent_findings)
            and all(delta > 0.0 for delta in recent_tokens)
        ):
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
        and not source_productive
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
        dm_low = dm_low_signal
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

    if token_variance_alarm and coverage_scores[-1] < 0.95 and not source_productive:
        stuck_candidates.append((0.65, "token_usage_variance_flat"))

    # If loop signals exist (including "near miss" signals below the full
    # loop threshold), suppress stagnation-style stuck triggers. This avoids
    # misclassifying early loop formation as stuck.
    loop_signal_hint = _has_loop_signal_hint(
        loop_candidates=loop_candidates,
        findings_deltas=findings_deltas,
        token_deltas=token_deltas,
        query_deltas=query_deltas,
        domain_deltas=domain_deltas,
        domain_counts=domain_counts,
        coverage_scores=coverage_scores,
        output_similarity=output_similarity,
        sim_threshold=sim_threshold,
        consecutive=consecutive,
    )
    if stuck_candidates and loop_signal_hint:
        low_coverage = bool(coverage_scores) and float(coverage_scores[-1]) < 0.2
        stuck_candidates = [
            (conf, trigger)
            for conf, trigger in stuck_candidates
            if (
                trigger not in {"coverage_stagnation", "coverage_flat"}
                or (low_coverage and trigger == "coverage_stagnation")
            )
        ]

    # Low output similarity confirms stuck: the agent is producing varied
    # outputs (not repeating) but still making no progress — flailing.
    if output_similarity is not None and stuck_candidates:
        similarity = float(output_similarity)
        low_sim_threshold = 1.0 - sim_threshold  # e.g., 0.2 when threshold is 0.8
        if similarity <= low_sim_threshold:
            # Boost the best stuck candidate confidence by up to 0.10
            boost = 0.10 * min(1.0, (low_sim_threshold - similarity) / max(1e-9, low_sim_threshold))
            best = max(stuck_candidates, key=lambda item: item[0])
            adjusted: list[tuple[float, str]] = []
            for conf, trigger in stuck_candidates:
                if (conf, trigger) == best:
                    adjusted.append((_clip01(conf + boost), trigger))
                else:
                    adjusted.append((conf, trigger))
            stuck_candidates = adjusted

    # Cross-detector suppression: when the primary failure mode is already
    # identified as loop or thrash (error-induced), suppress overlapping stuck
    # triggers that are symptoms rather than independent diagnoses.  Keep only
    # triggers that represent genuinely independent stuck conditions.
    # Priority rule: loop candidates at confidence >= 0.5 take precedence.
    _INDEPENDENT_STUCK_TRIGGERS = {"zero_progress", "sources_zero"}
    if coverage_scores and float(coverage_scores[-1]) < 0.2:
        _INDEPENDENT_STUCK_TRIGGERS.add("coverage_stagnation")
    if stuck_candidates and (loop_candidate_confidence >= 0.5 or error_count > 0):
        stuck_candidates = [
            (conf, trigger)
            for conf, trigger in stuck_candidates
            if trigger in _INDEPENDENT_STUCK_TRIGGERS
        ]

    # Confabulation overlap handling: keep coverage-stagnation available for
    # mixed-mode traces, but apply a confidence penalty at high confabulation
    # confidence so confabulation still has priority.
    if confabulation_detected and confabulation_confidence >= 0.85:
        adjusted_candidates: list[tuple[float, str]] = []
        for conf, trigger in stuck_candidates:
            if trigger == "coverage_stagnation":
                adjusted_candidates.append((_clip01(conf - 0.15), trigger))
            else:
                adjusted_candidates.append((conf, trigger))
        stuck_candidates = adjusted_candidates

    stuck_detected = False
    stuck_confidence = 0.0
    stuck_trigger: Optional[str] = None
    if stuck_candidates:
        stuck_confidence, stuck_trigger = max(stuck_candidates, key=lambda item: item[0])
        stuck_detected = True

    # Burn-rate anomaly is an exclusive runaway-cost signal. Keep the trigger
    # so stop-rule derivation can emit runaway_cost, but suppress stuck.
    runaway_cost_from_stuck = bool(
        stuck_detected and stuck_trigger == "burn_rate_anomaly"
    )
    if runaway_cost_from_stuck:
        stuck_detected = False
        stuck_confidence = 0.0

    detector_priority: Optional[str] = None
    if confabulation_detected:
        loop_detected = False
        loop_confidence = 0.0
        loop_trigger = None
        detector_priority = "confabulation"
    elif loop_detected and stuck_detected:
        if loop_confidence >= stuck_confidence:
            stuck_detected = False
            stuck_confidence = 0.0
            stuck_trigger = None
            detector_priority = "loop"
        else:
            loop_detected = False
            loop_confidence = 0.0
            loop_trigger = None
            detector_priority = "stuck"
    elif loop_detected:
        detector_priority = "loop"
    elif stuck_detected:
        detector_priority = "stuck"
    elif runaway_cost_from_stuck:
        detector_priority = "runaway_cost"

    return LoopDetectionResult(
        loop_detected=loop_detected,
        loop_confidence=_clip01(loop_confidence),
        loop_trigger=loop_trigger,
        confabulation_detected=confabulation_detected,
        confabulation_confidence=_clip01(confabulation_confidence),
        confabulation_trigger=confabulation_trigger,
        confabulation_signals=confabulation_signals,
        stuck_detected=stuck_detected,
        stuck_confidence=_clip01(stuck_confidence),
        stuck_trigger=stuck_trigger,
        detector_priority=detector_priority,
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


def _source_finding_ratio(
    *,
    sources_count: int,
    findings_count: int,
) -> Optional[float]:
    findings = max(0, int(findings_count))
    if findings <= 0:
        return None
    sources = max(0, int(sources_count))
    return float(sources) / float(findings)


def _consecutive_ratio_declines(
    *,
    ratios: Sequence[Optional[float]],
    loop_indices: Sequence[int],
) -> int:
    """Return trailing count of strictly declining ratio steps with loop progress."""

    if len(ratios) < 2 or len(loop_indices) != len(ratios):
        return 0

    steps = 0
    epsilon = 1e-9
    for idx in range(len(ratios) - 1, 0, -1):
        current_ratio = ratios[idx]
        previous_ratio = ratios[idx - 1]
        if current_ratio is None or previous_ratio is None:
            break
        if int(loop_indices[idx]) <= int(loop_indices[idx - 1]):
            break
        if float(current_ratio) < (float(previous_ratio) - epsilon):
            steps += 1
            continue
        break
    return steps


def _adaptive_alarm_from_series(
    *,
    values: Sequence[float],
    direction: str,
    fallback_threshold: Optional[float],
    config: VitalsConfig,
) -> tuple[bool, float, bool, bool]:
    """Replay a scalar series through AdaptiveThreshold and return last alarm state."""

    normalized_direction = "decrease" if direction == "decrease" else "increase"
    tracker = AdaptiveThreshold(
        direction=normalized_direction,
        k_sigma=max(0.1, float(config.spc_k_sigma)),
        window_size=max(2, int(config.spc_window_size)),
        warmup_steps=max(1, int(config.spc_warmup_steps)),
        cooldown_steps=max(0, int(config.spc_cooldown_steps)),
        wma_decay=float(config.spc_wma_decay),
    )
    latest_alarm = False
    latest_threshold = float(fallback_threshold or 0.0)
    latest_suppressed = False
    latest_warmup_complete = False
    for value in values:
        update = tracker.update(
            float(value),
            fallback_threshold=fallback_threshold,
        )
        latest_alarm = bool(update.alarm)
        latest_threshold = float(update.threshold)
        latest_suppressed = bool(update.suppressed_by_cooldown)
        latest_warmup_complete = bool(update.warmup_complete)
    return latest_alarm, latest_threshold, latest_suppressed, latest_warmup_complete


def _token_variance_series(
    *,
    values: Sequence[float],
    window_size: int,
) -> list[float]:
    """Build rolling token-delta variance series for adaptive flat-usage detection."""

    if len(values) < 3:
        return []

    deltas = _deltas(values)
    variances: list[float] = []
    size = max(2, int(window_size))
    for idx in range(len(deltas)):
        segment = deltas[: idx + 1]
        recent = segment[-size:]
        variances.append(_variance(recent))
    return variances


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = float(sum(values) / len(values))
    return float(sum((float(value) - avg) ** 2 for value in values) / len(values))


def _proportional_window(
    *,
    trace_length: int,
    percentage: float,
    minimum: int,
    fallback: int,
) -> int:
    """Compute a trace-length proportional window with sane fallbacks."""

    length = max(0, int(trace_length))
    floor_min = max(1, int(minimum))
    floor_fallback = max(floor_min, int(fallback))
    if length <= 0:
        return floor_min
    if math.isfinite(percentage) and percentage > 0.0:
        return max(floor_min, int(math.floor(length * percentage)))
    return floor_fallback


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


def _has_loop_signal_hint(
    *,
    loop_candidates: Sequence[tuple[float, str]],
    findings_deltas: Sequence[float],
    token_deltas: Sequence[float],
    query_deltas: Sequence[float],
    domain_deltas: Sequence[float],
    domain_counts: Sequence[int],
    coverage_scores: Sequence[float],
    output_similarity: Optional[float],
    sim_threshold: float,
    consecutive: int,
) -> bool:
    """Return True when loop-like behavior is present, even below threshold."""

    if loop_candidates:
        return True

    hint_window = max(1, int(consecutive) - 1)

    coverage_ready = bool(coverage_scores) and float(coverage_scores[-1]) >= 0.3
    if coverage_ready and len(findings_deltas) >= hint_window and len(token_deltas) >= hint_window:
        recent_findings = findings_deltas[-hint_window:]
        recent_tokens = token_deltas[-hint_window:]
        if all(delta <= 0.0 for delta in recent_findings) and all(delta > 0.0 for delta in recent_tokens):
            return True

    if (
        coverage_ready
        and len(query_deltas) >= hint_window
        and len(domain_deltas) >= hint_window
        and domain_counts
        and max(domain_counts) > 0
    ):
        recent_queries = query_deltas[-hint_window:]
        recent_domains = domain_deltas[-hint_window:]
        if all(delta > 0.0 for delta in recent_queries) and all(delta <= 0.0 for delta in recent_domains):
            return True

    if output_similarity is not None:
        relaxed_threshold = _clip01(float(sim_threshold) - 0.05)
        if float(output_similarity) >= relaxed_threshold:
            return True

    return False


__all__ = ["LoopDetectionResult", "detect_agent_loop", "detect_loop"]
