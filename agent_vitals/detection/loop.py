"""Loop/stuck detection helpers for Agent Vitals.

The loop breaker analyzes vitals snapshots over time and emits detection flags
that can be used for logging, calibration, and enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace as _dc_replace
from statistics import mean
from typing import TYPE_CHECKING, Optional, Sequence

from ..config import VitalsConfig, get_vitals_config
from ..schema import VitalsSnapshot
from .adaptive_threshold import AdaptiveDirection, AdaptiveThreshold
from .signal_mapping import classify_model_size, get_signal_mapping

if TYPE_CHECKING:
    from .signal_mapping import SignalMapping

SHORT_RUN_MAX_STEPS = 4
SHORT_RUN_MIN_FINDINGS = 3
SHORT_RUN_OBJECTIVE_MAX = 3
SHORT_RUN_ZERO_COVERAGE_MAX_FINDINGS = 6
SHORT_RUN_ZERO_COVERAGE_MIN_SOURCES = 8
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

    # Explicit runaway-cost signal carried alongside stuck. Set when the
    # burn-rate anomaly check fires; replaces the v1.14.1 implicit chain
    # where ``stuck_trigger="burn_rate_anomaly"`` was used as a sentinel
    # string protocol that downstream consumers (stop_rule, backtest)
    # had to know to interpret. AV-S08-M04.
    runaway_cost_detected: bool = False
    runaway_cost_confidence: float = 0.0

    # Hopfield 3rd-layer early-detection marker (AV-S09-M02). Set when
    # the optional Hopfield prefix-model adapter scored any detector
    # above its per-detector override threshold while the trace was in
    # the early-detection window (length 3-6). Purely informational —
    # never mutates the per-detector ``*_detected`` flags above, so the
    # handcrafted+TDA stack remains authoritative and full-trace
    # detection numbers stay bit-identical against the v1.14.2 baseline
    # under the default ``hopfield_enabled=False`` config.
    hopfield_override_active: bool = False

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
            "runaway_cost_detected": bool(self.runaway_cost_detected),
            "runaway_cost_confidence": float(_clip01(self.runaway_cost_confidence)),
            "hopfield_override_active": bool(self.hopfield_override_active),
            "detector_priority": self.detector_priority,
        }


@dataclass(slots=True)
class _DetectionContext:
    """Pre-computed signals shared across sub-detectors."""

    cfg: VitalsConfig
    series: list[VitalsSnapshot]
    snapshot: VitalsSnapshot
    consecutive: int
    normalized_workflow: str
    signal_map: SignalMapping

    # Time series
    findings_counts: list[int]
    sources_counts: list[int]
    sources_missing: bool
    objectives_counts: list[int]
    objectives_missing: bool
    query_counts: list[int]
    domain_counts: list[int]
    coverage_scores: list[float]
    token_totals: list[float]
    completion_tokens: list[float]

    # Deltas
    findings_deltas: list[float]
    loop_indices: list[int]
    loop_index_deltas: list[float]
    query_deltas: list[float]
    domain_deltas: list[float]
    coverage_deltas: list[float]
    token_deltas: list[float]

    # Current snapshot values
    findings_count: int
    sources_count: int
    query_count: int
    error_count: int
    source_finding_ratio: Optional[float]
    ratio_series: list[Optional[float]]
    ratio_declining_steps: int
    source_productive: bool
    output_similarity: Optional[float]

    # Adaptive alarms
    dm_stagnation_alarm: bool
    dm_stagnation_threshold: float
    cv_stagnation_alarm: bool
    cv_stagnation_threshold: float
    token_variance_alarm: bool
    dm_coverage_series: list[float]
    cv_coverage_series: list[float]

    # Verified source signals
    verified_source_ratio: Optional[float]
    verified_source_ratio_series: list[Optional[float]]

    # Causal confab: the full snapshot series (history + current)
    all_snapshots: list[VitalsSnapshot]

    # Config-derived
    ratio_floor: float
    ratio_declining_required: int
    ratio_decline_window: int
    sim_threshold: float


def _prepare_context(
    snapshot: VitalsSnapshot,
    history: Optional[Sequence[VitalsSnapshot]],
    cfg: VitalsConfig,
    workflow_type: str,
) -> Optional[_DetectionContext]:
    """Build shared detection context. Returns None if insufficient data."""

    prior = list(history or [])
    series = prior + [snapshot]
    min_evidence_steps = max(1, int(cfg.min_evidence_steps))
    if len(series) < max(2, min_evidence_steps):
        return None

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
    completion_tokens = [float(item.signals.completion_tokens) for item in series]

    model_size = classify_model_size(
        completion_tokens,
        explicit_class=str(cfg.model_size_class),
    )
    signal_map = get_signal_mapping(model_size)

    # NOTE: burn_rate_multiplier_scale intentionally NOT applied here.
    # The scale (2.0x for small models) was never validated against a corpus
    # with small-model traces and causes false negatives on runaway_cost
    # (bench: 42 FNs, recall 100%→69.8%).  The suppress_token_variance_flat
    # flag handles the validated small-model FP concern independently.

    findings_deltas = _deltas(findings_counts)
    loop_indices = [int(item.loop_index) for item in series]
    loop_index_deltas = _deltas(loop_indices)
    query_deltas = _deltas(query_counts)
    domain_deltas = _deltas(domain_counts)
    coverage_deltas = _deltas(coverage_scores)
    token_deltas = _deltas(token_totals)
    findings_count = int(snapshot.signals.findings_count)
    sources_count = int(snapshot.signals.sources_count)
    query_count = int(snapshot.signals.query_count)
    source_finding_ratio = _source_finding_ratio(
        sources_count=sources_count,
        findings_count=findings_count,
    )
    ratio_floor = max(0.0, float(cfg.source_finding_ratio_floor))
    ratio_declining_required = max(1, int(cfg.source_finding_ratio_declining_steps))
    ratio_decline_window = max(
        ratio_declining_required + 1,
        int(getattr(cfg, "source_finding_ratio_decline_window", 5)),
    )
    ratio_series: list[Optional[float]] = []
    for item in series:
        snapshot_ratio = getattr(item, "source_finding_ratio", None)
        if snapshot_ratio is None:
            snapshot_ratio = _source_finding_ratio(
                sources_count=int(item.signals.sources_count),
                findings_count=int(item.signals.findings_count),
            )
        ratio_series.append(snapshot_ratio)
    ratio_declining_steps = _windowed_ratio_declines(
        ratios=ratio_series,
        loop_indices=loop_indices,
        window=ratio_decline_window,
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

    sim_threshold = float(cfg.loop_similarity_threshold)
    output_similarity = getattr(snapshot, "output_similarity", None)
    error_count = int(snapshot.signals.error_count)

    # Verified source ratio: computed from verified/unverified counts if available.
    verified_source_ratio = getattr(snapshot, "verified_source_ratio", None)
    verified_source_ratio_series: list[Optional[float]] = [
        getattr(item, "verified_source_ratio", None) for item in series
    ]

    return _DetectionContext(
        cfg=cfg,
        series=series,
        snapshot=snapshot,
        consecutive=consecutive,
        normalized_workflow=normalized_workflow,
        signal_map=signal_map,
        findings_counts=findings_counts,
        sources_counts=sources_counts,
        sources_missing=sources_missing,
        objectives_counts=objectives_counts,
        objectives_missing=objectives_missing,
        query_counts=query_counts,
        domain_counts=domain_counts,
        coverage_scores=coverage_scores,
        token_totals=token_totals,
        completion_tokens=completion_tokens,
        findings_deltas=findings_deltas,
        loop_indices=loop_indices,
        loop_index_deltas=loop_index_deltas,
        query_deltas=query_deltas,
        domain_deltas=domain_deltas,
        coverage_deltas=coverage_deltas,
        token_deltas=token_deltas,
        findings_count=findings_count,
        sources_count=sources_count,
        query_count=query_count,
        error_count=error_count,
        source_finding_ratio=source_finding_ratio,
        ratio_series=ratio_series,
        ratio_declining_steps=ratio_declining_steps,
        source_productive=source_productive,
        output_similarity=output_similarity,
        dm_stagnation_alarm=dm_stagnation_alarm,
        dm_stagnation_threshold=dm_stagnation_threshold,
        cv_stagnation_alarm=cv_stagnation_alarm,
        cv_stagnation_threshold=cv_stagnation_threshold,
        token_variance_alarm=token_variance_alarm,
        dm_coverage_series=dm_coverage_series,
        cv_coverage_series=cv_coverage_series,
        ratio_floor=ratio_floor,
        all_snapshots=series,
        verified_source_ratio=verified_source_ratio,
        verified_source_ratio_series=verified_source_ratio_series,
        ratio_declining_required=ratio_declining_required,
        ratio_decline_window=ratio_decline_window,
        sim_threshold=sim_threshold,
    )


def _detect_loop_candidates(ctx: _DetectionContext) -> list[tuple[float, str]]:
    """Detect loop candidates from findings plateaus, query repetition, and content similarity."""

    candidates: list[tuple[float, str]] = []

    if len(ctx.findings_deltas) >= ctx.consecutive:
        recent_findings = ctx.findings_deltas[-ctx.consecutive:]
        plateau = all(delta <= 0.0 for delta in recent_findings)
        loop_progressing = (
            len(ctx.loop_index_deltas) >= ctx.consecutive
            and all(delta > 0.0 for delta in ctx.loop_index_deltas[-ctx.consecutive:])
        )

        tokens_active = (
            len(ctx.token_deltas) >= ctx.consecutive
            and all(delta > 0.0 for delta in ctx.token_deltas[-ctx.consecutive:])
        )

        coverage_flat = False
        if len(ctx.coverage_deltas) >= ctx.consecutive:
            coverage_flat = all(
                abs(delta) <= 1e-3 for delta in ctx.coverage_deltas[-ctx.consecutive:]
            )
        coverage_nontrivial = ctx.coverage_scores[-1] >= 0.2

        if (
            plateau
            and coverage_flat
            and tokens_active
            and loop_progressing
            and coverage_nontrivial
            and not ctx.source_productive
        ):
            candidates.append((0.85, "findings_plateau+coverage_flat"))
        elif (
            plateau
            and tokens_active
            and loop_progressing
            and coverage_nontrivial
            and not ctx.source_productive
        ):
            candidates.append((0.75, "findings_plateau"))

        if (
            plateau
            and loop_progressing
            and len(ctx.query_deltas) >= ctx.consecutive
            and len(ctx.domain_deltas) >= ctx.consecutive
            and all(delta > 0.0 for delta in ctx.query_deltas[-ctx.consecutive:])
            and all(delta <= 0.0 for delta in ctx.domain_deltas[-ctx.consecutive:])
            and max(ctx.domain_counts) > 0
        ):
            candidates.append((0.9, "query_repetition_proxy"))

    # Content similarity gate
    if ctx.output_similarity is not None:
        similarity = float(ctx.output_similarity)
        similarity_series = [
            float(similarity_value)
            for item in ctx.series
            if (similarity_value := item.output_similarity) is not None
        ]
        similarity_alarm, similarity_threshold, _, _ = _adaptive_alarm_from_series(
            values=similarity_series,
            direction="increase",
            fallback_threshold=ctx.sim_threshold,
            config=ctx.cfg,
        )
        similarity_fixed_hit = similarity >= ctx.sim_threshold
        if similarity_alarm or similarity_fixed_hit:
            active_threshold = _clip01(min(similarity_threshold, ctx.sim_threshold))
            confidence = 0.80 + 0.15 * min(
                1.0,
                (similarity - active_threshold) / max(1e-9, 1.0 - active_threshold),
            )
            candidates.append((confidence, "content_similarity"))

    # Suppress loop when errors are present — error-induced plateaus are
    # thrash behavior, not repetitive looping.
    if ctx.error_count > 0:
        candidates.clear()

    return candidates


_CAUSAL_EPSILON = 1e-12


def _causal_pearson(
    xs: Sequence[float], ys: Sequence[float],
) -> Optional[float]:
    """Pearson correlation, returning None when either series has zero variance."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= _CAUSAL_EPSILON or y_var <= _CAUSAL_EPSILON:
        return None
    return numerator / math.sqrt(x_var * y_var)


def _causal_residualize(
    target: Sequence[float], control: Sequence[float],
) -> list[float]:
    """Remove the linear effect of *control* from *target* via OLS."""
    if len(target) != len(control) or len(target) < 2:
        return list(target)
    mean_t = sum(target) / len(target)
    mean_c = sum(control) / len(control)
    c_var = sum((v - mean_c) ** 2 for v in control)
    if c_var <= _CAUSAL_EPSILON:
        return [v - mean_t for v in target]
    cov = sum((c - mean_c) * (t - mean_t) for c, t in zip(control, target))
    slope = cov / c_var
    intercept = mean_t - slope * mean_c
    return [t - (intercept + slope * c) for t, c in zip(target, control)]


def _causal_link_strength(
    snapshots: Sequence[VitalsSnapshot],
) -> tuple[float, Optional[float], float]:
    """Compute causal link strength for a window of snapshots.

    Returns (link_strength, partial_correlation, response_ratio).
    """
    findings = [s.signals.findings_count for s in snapshots]
    sources = [s.signals.sources_count for s in snapshots]
    tokens = [s.signals.total_tokens for s in snapshots]

    f_deltas = [max(0, c - p) for p, c in zip(findings, findings[1:])]
    s_deltas = [max(0, c - p) for p, c in zip(sources, sources[1:])]
    t_deltas = [max(0, c - p) for p, c in zip(tokens, tokens[1:])]

    response_ratio = sum(s_deltas) / max(1, sum(f_deltas))
    f_resid = _causal_residualize(f_deltas, t_deltas)
    s_resid = _causal_residualize(s_deltas, t_deltas)
    partial_corr = _causal_pearson(f_resid, s_resid)

    if partial_corr is None:
        link = min(1.0, response_ratio)
    else:
        norm_corr = (partial_corr + 1.0) / 2.0
        link = max(0.0, min(1.0, norm_corr * min(1.0, response_ratio)))

    return link, partial_corr, response_ratio


def _verified_count(snapshot: VitalsSnapshot) -> int:
    """Safely get verified_sources_count, treating None as 0."""
    return int(snapshot.signals.verified_sources_count or 0)


def _unverified_count(snapshot: VitalsSnapshot) -> int:
    """Safely get unverified_sources_count, treating None as 0."""
    return int(snapshot.signals.unverified_sources_count or 0)


def _has_verified_data(window: Sequence[VitalsSnapshot]) -> bool:
    """Check if the window has meaningful verified source counts."""
    total_verified = sum(_verified_count(s) for s in window)
    total_sources = sum(s.signals.sources_count for s in window)
    return total_sources > 0 and (
        total_verified > 0 or any(_unverified_count(s) > 0 for s in window)
    )


def _verified_link_strength(
    snapshots: Sequence[VitalsSnapshot],
) -> Optional[float]:
    """Compute verified->sources link strength for a window.

    Returns None when verified data is unavailable or there is no source growth.
    """
    if not _has_verified_data(snapshots):
        return None

    verified = [_verified_count(s) for s in snapshots]
    sources = [s.signals.sources_count for s in snapshots]
    tokens = [s.signals.total_tokens for s in snapshots]

    v_deltas = [max(0, c - p) for p, c in zip(verified, verified[1:])]
    s_deltas = [max(0, c - p) for p, c in zip(sources, sources[1:])]
    t_deltas = [max(0, c - p) for p, c in zip(tokens, tokens[1:])]

    total_s = sum(s_deltas)
    if total_s == 0:
        return None

    v_response = sum(v_deltas) / max(1, total_s)
    v_resid = _causal_residualize(v_deltas, t_deltas)
    s_resid = _causal_residualize(s_deltas, t_deltas)
    corr = _causal_pearson(v_resid, s_resid)

    if corr is None:
        return min(1.0, v_response)
    norm_corr = (corr + 1.0) / 2.0
    return max(0.0, min(1.0, norm_corr * min(1.0, v_response)))


def _detect_causal_confabulation(
    ctx: _DetectionContext,
) -> tuple[bool, Optional[tuple[float, str, tuple[str, ...]]]]:
    """Detect confabulation via rolling causal source-link degradation.

    Returns (eligible, candidate). When eligible is True the causal detector
    ran and produced a definitive result — callers should NOT also run the
    legacy SFR-threshold path (which causes double-counting and FPs).
    When eligible is False, the causal detector could not attempt detection
    (insufficient history, no source data) and the legacy path should run.
    """
    cfg = ctx.cfg
    window_size = int(cfg.causal_confab_window_size)
    snapshots = ctx.all_snapshots

    # Ineligibility: skip when there's no source evidence at all.
    if ctx.sources_missing or not ctx.sources_counts or max(ctx.sources_counts) == 0:
        return False, None

    # Ineligibility: trace too short for causal windowing.
    if len(snapshots) < window_size:
        return False, None

    # Score rolling windows. Track both the primary findings->sources link and
    # the verified->sources link for Path 3.
    link_scores: list[float] = []
    verified_scores: list[Optional[float]] = []
    window_starts: list[int] = []
    for end in range(window_size, len(snapshots) + 1):
        window = snapshots[end - window_size: end]
        link, _, _ = _causal_link_strength(window)
        link_scores.append(link)
        verified_scores.append(_verified_link_strength(window))
        window_starts.append(int(window[0].loop_index))

    if not link_scores:
        return False, None

    # The detector is now eligible — it WILL produce a definitive answer
    # (either fire or explicitly not fire). The legacy SFR path must not run.
    baseline_strength = max(link_scores[:2])
    comparison = link_scores[1:] or link_scores
    weakest_strength = min(comparison)
    structural_drop = max(0.0, baseline_strength - weakest_strength)

    # Final source-finding ratio: prefer the snapshot's stored ratio, fall
    # back to a fresh sources/findings calculation (matches bench reference).
    last = snapshots[-1]
    if last.source_finding_ratio is not None:
        final_ratio = float(last.source_finding_ratio)
    else:
        findings_denom = max(1, last.signals.findings_count)
        final_ratio = float(last.signals.sources_count) / float(findings_denom)

    initial_sources = snapshots[0].signals.sources_count

    # Path 1: Causal link break (structural break from healthy baseline).
    # Whole-trace semantics — the trace-level final-step adjudication in
    # backtest._replay_trace handles streaming-vs-one-shot reconciliation.
    structural_break = (
        baseline_strength >= float(cfg.causal_confab_baseline_floor)
        and weakest_strength <= float(cfg.causal_confab_weak_link_threshold)
        and structural_drop >= float(cfg.causal_confab_structural_drop_threshold)
        and final_ratio <= float(cfg.causal_confab_ratio_gate)
    )
    # Path 2: Persistent low causal link (never-established coupling).
    persistent_low = (
        weakest_strength <= float(cfg.causal_confab_low_link_threshold)
        and initial_sources <= int(cfg.causal_confab_source_bootstrap_cap)
        and final_ratio <= float(cfg.causal_confab_low_link_ratio_gate)
    )

    # Path 3: Verified source decoupling (real LLM confabulation).
    # Findings and sources may grow in lockstep so Paths 1-2 never fire,
    # but DOI verification reveals that verified sources persistently lag.
    verified_decoupling = False
    verified_baseline_strength = 0.0
    verified_weakest_strength = 0.0
    verified_drop = 0.0
    verified_present = [v for v in verified_scores if v is not None]
    if verified_present:
        verified_baseline_strength = max(verified_present)
        verified_weakest_strength = min(verified_present)
        verified_drop = max(0.0, verified_baseline_strength - verified_weakest_strength)

        total_verified = _verified_count(last)
        total_sources = last.signals.sources_count
        verified_ratio = total_verified / max(1, total_sources)

        verified_decoupling = (
            verified_baseline_strength <= float(cfg.causal_confab_verified_link_floor)
            and verified_ratio <= float(cfg.causal_confab_verified_ratio_gate)
            and total_sources >= int(cfg.causal_confab_verified_min_sources)
        )

    # Trigger selection priority: structural_break > persistent_low > verified_decoupling.
    trigger: Optional[str]
    confidence: float
    if structural_break:
        trigger = "causal_link_break"
        baseline_term = min(
            1.0,
            baseline_strength / max(float(cfg.causal_confab_baseline_floor), _CAUSAL_EPSILON),
        )
        drop_term = min(
            1.0,
            structural_drop / max(
                float(cfg.causal_confab_structural_drop_threshold), _CAUSAL_EPSILON
            ),
        )
        ratio_term = 1.0 - min(
            1.0, final_ratio / max(float(cfg.causal_confab_ratio_gate), _CAUSAL_EPSILON)
        )
        confidence = min(1.0, 0.35 + 0.35 * baseline_term + 0.2 * drop_term + 0.1 * ratio_term)
    elif persistent_low:
        trigger = "persistent_low_causal_link"
        low_link_term = 1.0 - min(
            1.0,
            weakest_strength / max(
                float(cfg.causal_confab_low_link_threshold), _CAUSAL_EPSILON
            ),
        )
        ratio_term = 1.0 - min(
            1.0,
            final_ratio / max(float(cfg.causal_confab_low_link_ratio_gate), _CAUSAL_EPSILON),
        )
        confidence = min(1.0, 0.4 + 0.4 * low_link_term + 0.2 * ratio_term)
    elif verified_decoupling:
        trigger = "verified_source_decoupling"
        weak_term = 1.0 - min(
            1.0,
            verified_weakest_strength / max(
                float(cfg.causal_confab_verified_weak_threshold), _CAUSAL_EPSILON
            ),
        )
        ratio_term = 1.0 - min(
            1.0,
            final_ratio / max(
                float(cfg.causal_confab_verified_ratio_gate), _CAUSAL_EPSILON
            ),
        )
        drop_term = (
            min(
                1.0,
                verified_drop / max(
                    float(cfg.causal_confab_verified_drop_threshold), _CAUSAL_EPSILON
                ),
            )
            if verified_drop > 0
            else 0.0
        )
        confidence = min(1.0, 0.4 + 0.3 * weak_term + 0.15 * drop_term + 0.15 * ratio_term)
    else:
        # Eligible but no detection — return (True, None) so callers know to
        # suppress the legacy SFR path.
        return True, None

    signal_parts: list[str] = [
        trigger,
        f"link_strength={weakest_strength:.3f}",
        f"structural_drop={structural_drop:.3f}",
        f"baseline={baseline_strength:.3f}",
    ]
    if trigger == "verified_source_decoupling":
        signal_parts.append(f"verified_link={verified_weakest_strength:.3f}")
    return True, (confidence, trigger, tuple(signal_parts))


def _detect_confabulation_candidates(
    ctx: _DetectionContext,
) -> list[tuple[float, str, tuple[str, ...]]]:
    """Detect confabulation candidates from source-to-finding ratio and trajectory.

    The causal detector is tried first. When *eligible* (sufficient history
    and source data) the causal detector owns the verdict — its result is the
    only confab candidate produced. The legacy SFR-threshold path runs only
    when causal is ineligible (short trace or no source data). This is the
    integration model bench validated against: bench reports F1=0.950 with
    Path 3 enabled when the legacy path is suppressed.
    """

    candidates: list[tuple[float, str, tuple[str, ...]]] = []

    # Primary path: causal confabulation detector.
    causal_eligible, causal = _detect_causal_confabulation(ctx)
    if causal is not None:
        candidates.append(causal)
    if causal_eligible:
        # Causal owns the verdict — skip legacy paths entirely.
        return candidates

    sources_stagnation = False
    unique_domains_stagnation = False
    if (
        not ctx.sources_missing
        and len(sources_deltas := _deltas(ctx.sources_counts)) >= SOURCES_STAGNATION_WINDOW
        and len(ctx.findings_deltas) >= SOURCES_STAGNATION_WINDOW
        and ctx.sources_counts
        and max(ctx.sources_counts) > 0
        and int(ctx.sources_counts[-1]) <= SOURCES_STAGNATION_MAX_SOURCES
    ):
        recent_source_deltas = sources_deltas[-SOURCES_STAGNATION_WINDOW:]
        recent_finding_deltas = ctx.findings_deltas[-SOURCES_STAGNATION_WINDOW:]
        sources_flat = all(delta == 0.0 for delta in recent_source_deltas)
        findings_growing = sum(recent_finding_deltas) > 0.0 and all(
            delta >= 0.0 for delta in recent_finding_deltas
        )
        sources_stagnation = sources_flat and findings_growing
        if sources_stagnation and len(ctx.domain_counts) >= UNIQUE_DOMAINS_STAGNATION_WINDOW:
            recent_domains = ctx.domain_counts[-UNIQUE_DOMAINS_STAGNATION_WINDOW:]
            unique_domains_stagnation = all(int(value) <= 1 for value in recent_domains)

    has_source_evidence = bool(ctx.sources_counts) and max(ctx.sources_counts) > 0
    ratio_floor_breach = False
    ratio_spc_ready = False
    if has_source_evidence and ctx.source_finding_ratio is not None:
        observed_ratio_risk = [
            max(0.0, ctx.ratio_floor - float(value))
            for value in ctx.ratio_series
            if value is not None
        ]
        ratio_floor_breach, _, _, ratio_spc_ready = _adaptive_alarm_from_series(
            values=observed_ratio_risk,
            direction="increase",
            fallback_threshold=0.0,
            config=ctx.cfg,
        )
    ratio_decline_with_growth = False
    if (
        has_source_evidence
        and ctx.source_finding_ratio is not None
        and ctx.source_finding_ratio <= 1.0
        and ctx.ratio_declining_steps >= ctx.ratio_declining_required
        and len(ctx.findings_deltas) >= ctx.ratio_declining_required
        and len(ctx.loop_index_deltas) >= ctx.ratio_declining_required
    ):
        recent_findings = ctx.findings_deltas[-ctx.ratio_declining_required:]
        recent_loop_steps = ctx.loop_index_deltas[-ctx.ratio_declining_required:]
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

        if "findings_count_delta" in set(
            getattr(ctx.snapshot, "cusum_alarm_metrics", []) or []
        ):
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

        candidates.append(
            (
                min(0.95, confab_confidence),
                "+".join(trigger_parts),
                tuple(dict.fromkeys(signal_parts)),
            )
        )

    # Verified source ratio: direct signal for high-volume source fabrication.
    # When verified_source_ratio < 0.3 AND sources are growing, the agent is
    # producing many unverified sources — a hallmark of frontier model
    # confabulation (claude-sonnet fabricates plausible sources in volume).
    _VERIFIED_RATIO_THRESHOLD = 0.3
    _VERIFIED_MIN_SOURCES = 3
    if (
        ctx.verified_source_ratio is not None
        and ctx.verified_source_ratio < _VERIFIED_RATIO_THRESHOLD
        and ctx.sources_count >= _VERIFIED_MIN_SOURCES
        and not ctx.sources_missing
        and len(ctx.sources_counts) >= 2
        and ctx.sources_counts[-1] > ctx.sources_counts[-2]
    ):
        # Scale confidence by how far below the threshold the ratio is.
        ratio_gap = (_VERIFIED_RATIO_THRESHOLD - ctx.verified_source_ratio) / _VERIFIED_RATIO_THRESHOLD
        confab_confidence = 0.70 + 0.20 * min(1.0, ratio_gap)
        trigger_parts = ["verified_source_ratio_low"]
        signal_parts = ["verified_source_ratio"]

        # Boost if the verified ratio shows windowed decline pattern.
        verified_declines = _windowed_ratio_declines(
            ratios=ctx.verified_source_ratio_series,
            loop_indices=ctx.loop_indices,
            window=ctx.ratio_decline_window,
        )
        if verified_declines >= ctx.ratio_declining_required:
            confab_confidence += 0.10
            trigger_parts.append("verified_ratio_declining")
            signal_parts.append("verified_source_ratio_declining")

        candidates.append(
            (
                min(0.95, confab_confidence),
                "+".join(trigger_parts),
                tuple(dict.fromkeys(signal_parts)),
            )
        )

    return candidates


def _detect_stuck_candidates(ctx: _DetectionContext) -> list[tuple[float, str]]:
    """Detect stuck candidates from coverage stagnation, zero progress, and burn rate."""

    cfg = ctx.cfg
    candidates: list[tuple[float, str]] = []

    dm_coverage = float(ctx.snapshot.metrics.dm_coverage)
    cv_coverage = float(ctx.snapshot.metrics.cv_coverage)
    dm_low_signal = ctx.dm_stagnation_alarm or dm_coverage <= float(cfg.stuck_dm_threshold)
    cv_low_signal = ctx.cv_stagnation_alarm or cv_coverage <= float(cfg.stuck_cv_threshold)
    dm_reference_threshold = (
        ctx.dm_stagnation_threshold if ctx.dm_stagnation_alarm else float(cfg.stuck_dm_threshold)
    )
    cv_reference_threshold = (
        ctx.cv_stagnation_threshold if ctx.cv_stagnation_alarm else float(cfg.stuck_cv_threshold)
    )

    # Coverage stagnation: only evaluate once enough points exist for DM lookback.
    dm_coverage_series = [float(item.metrics.dm_coverage) for item in ctx.series]
    dm_zero_streak = 0
    for value in reversed(dm_coverage_series):
        if value == 0.0:
            dm_zero_streak += 1
        else:
            break

    if not ctx.source_productive and dm_zero_streak >= 5 and ctx.coverage_scores[-1] < 0.95:
        candidates.append((0.75, "coverage_stagnation"))
    elif not ctx.source_productive and len(ctx.series) >= 5 and ctx.coverage_scores[-1] < 0.95:
        if dm_low_signal and cv_low_signal:
            dm_factor = _relative_margin_below(dm_coverage, dm_reference_threshold)
            cv_factor = _relative_margin_below(cv_coverage, cv_reference_threshold)
            candidates.append(
                (0.6 + 0.4 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation")
            )
    elif not ctx.source_productive and len(ctx.series) >= 4 and ctx.coverage_scores[-1] < 0.95:
        half_dm = float(cfg.stuck_dm_threshold) / 2.0
        half_cv = float(cfg.stuck_cv_threshold) / 2.0
        if dm_coverage <= half_dm and cv_coverage <= half_cv:
            dm_factor = _relative_margin_below(dm_coverage, half_dm)
            cv_factor = _relative_margin_below(cv_coverage, half_cv)
            candidates.append(
                (0.6 + 0.3 * (dm_factor + cv_factor) / 2.0, "coverage_stagnation")
            )

    # Zero progress
    loop_index = int(ctx.snapshot.loop_index)
    total_tokens = int(ctx.snapshot.signals.total_tokens)
    if total_tokens == 0:
        if loop_index >= 1 and ctx.findings_count <= 1:
            candidates.append((0.80, "zero_progress"))
        elif loop_index >= 0 and ctx.findings_count == 0:
            candidates.append((0.80, "zero_progress"))

    # Research-style hard stall
    if (
        ctx.normalized_workflow == "research"
        and len(ctx.series) == SHORT_RUN_MAX_STEPS
        and ctx.coverage_scores[-1] <= 0.0
        and dm_coverage <= 0.0
        and cv_coverage <= 0.0
        and ctx.findings_count <= SHORT_RUN_ZERO_COVERAGE_MAX_FINDINGS
        and ctx.sources_count >= SHORT_RUN_ZERO_COVERAGE_MIN_SOURCES
        and ctx.query_count >= 2
        and total_tokens > 0
        and ctx.output_similarity is not None
        and float(ctx.output_similarity) >= ctx.sim_threshold
    ):
        candidates.append((0.92, "short_run_zero_coverage"))

    # Coverage flatline
    if len(ctx.coverage_deltas) >= ctx.consecutive and ctx.coverage_scores[-1] < 0.95:
        recent = ctx.coverage_deltas[-ctx.consecutive:]
        loop_progressing = (
            len(ctx.loop_index_deltas) >= ctx.consecutive
            and all(delta > 0.0 for delta in ctx.loop_index_deltas[-ctx.consecutive:])
        )
        if (
            cv_low_signal
            and loop_progressing
            and all(abs(delta) <= 1e-3 for delta in recent)
        ):
            candidates.append((0.65, "coverage_flat"))

    # Findings plateau
    plateau_window = _proportional_window(
        trace_length=len(ctx.series),
        percentage=float(cfg.findings_plateau_pct),
        minimum=2,
        fallback=FINDINGS_PLATEAU_WINDOW,
    )
    if (
        len(ctx.series) >= (plateau_window + 1)
        and len(ctx.findings_deltas) >= plateau_window
        and len(ctx.loop_index_deltas) >= plateau_window
        and len(ctx.token_deltas) >= plateau_window
        and ctx.coverage_scores[-1] < 0.95
    ):
        recent_findings = ctx.findings_deltas[-plateau_window:]
        recent_loop_deltas = ctx.loop_index_deltas[-plateau_window:]
        recent_tokens = ctx.token_deltas[-plateau_window:]
        if (
            all(delta > 0.0 for delta in recent_loop_deltas)
            and all(delta == 0.0 for delta in recent_findings)
            and all(delta > 0.0 for delta in recent_tokens)
        ):
            candidates.append((0.7, "findings_plateau"))

    # Sources-zero: research runs with no sources after 5+ steps.
    if (
        not ctx.sources_missing
        and ctx.normalized_workflow == "research"
        and len(ctx.series) >= 5
        and ctx.sources_counts
        and max(ctx.sources_counts) == 0
    ):
        candidates.append((0.8, "sources_zero"))

    # Late-onset stagnation
    stagnation_window = 2
    if (
        len(ctx.series) >= 5
        and len(ctx.coverage_deltas) >= stagnation_window
        and len(ctx.findings_deltas) >= stagnation_window
        and len(ctx.token_deltas) >= stagnation_window
        and not ctx.source_productive
    ):
        recent_coverage = ctx.coverage_deltas[-stagnation_window:]
        recent_findings = ctx.findings_deltas[-stagnation_window:]
        recent_tokens = ctx.token_deltas[-stagnation_window:]
        coverage_regression = any(delta < 0.0 for delta in recent_coverage)
        objectives_plateau = False
        if not ctx.objectives_missing and len(ctx.objectives_counts) >= len(ctx.series):
            objectives_deltas = _deltas(ctx.objectives_counts)
            if len(objectives_deltas) >= stagnation_window:
                objectives_plateau = all(
                    delta <= 0.0 for delta in objectives_deltas[-stagnation_window:]
                )
        coverage_non_increasing = all(delta <= 0.0 for delta in recent_coverage)
        findings_stalled = any(delta <= 0.0 for delta in recent_findings)
        tokens_active = all(delta > 0.0 for delta in recent_tokens)
        coverage_ready = 0.5 <= ctx.coverage_scores[-1] < 0.95
        dm_low = dm_low_signal
        if coverage_non_increasing and findings_stalled and tokens_active and coverage_ready:
            if (coverage_regression and dm_low) or (
                not coverage_regression and objectives_plateau
            ):
                candidates.append((0.7, "late_onset_stagnation"))

    # NOTE (AV-S08-M04): burn_rate_anomaly used to be appended here as a
    # stuck candidate with confidence 1.0, then immediately filtered out
    # downstream as "really a runaway_cost signal". The conf-1.0 sentinel
    # acted as an implicit suppressor of other stuck candidates, which
    # made per-profile burn_rate_multiplier overrides a footgun (raising
    # the multiplier silently leaked stuck FPs). The runaway-cost signal
    # is now computed explicitly in `_compute_burn_rate_runaway` and
    # carried as `LoopDetectionResult.runaway_cost_detected`; the stuck
    # suppression is applied explicitly in `_resolve_detections`.

    # Token usage variance flat
    if (
        ctx.token_variance_alarm
        and ctx.coverage_scores[-1] < 0.95
        and not ctx.source_productive
        and not ctx.signal_map.suppress_token_variance_flat
    ):
        candidates.append((0.65, "token_usage_variance_flat"))

    return candidates


def _handle_stuck_disabled(
    ctx: _DetectionContext,
    loop_candidates: list[tuple[float, str]],
    confab_candidates: list[tuple[float, str, tuple[str, ...]]],
) -> LoopDetectionResult:
    """Early-return path when stuck detection is disabled for this workflow."""

    # Use min-baseline mode so that once a burn-rate spike enters the
    # baseline, subsequent steps still detect the anomaly relative to the
    # pre-spike minimum rather than the contaminated mean.
    # ``apply_signal_scale=False`` preserves the historical stuck-disabled
    # behavior (the model-size scale was NOT applied here because it
    # caused FNs on small-model traces).
    build_runaway_detected, build_runaway_confidence_value = _compute_burn_rate_runaway(
        ctx, baseline_mode="min", apply_signal_scale=False
    )
    build_runaway_confidence: float | None = (
        build_runaway_confidence_value if build_runaway_detected else None
    )

    # Gate burn_rate_anomaly on short traces: baseline from < 3 deltas is
    # unreliable, producing FPs on healthy traces with normal token variance.
    _BURN_RATE_MIN_STEPS = 4
    if build_runaway_detected and len(ctx.series) < _BURN_RATE_MIN_STEPS:
        build_runaway_detected = False
        build_runaway_confidence = None

    # Stagnation evidence: adaptive alarms, dm=0 streak, or static dm/cv
    # threshold checks.  Without stuck detection active, stagnation is the
    # best surrogate for stuck-like behavior — suppress both loop (filtered
    # to content_similarity only) and burn_rate_anomaly (runaway).  Genuine
    # runaway traces maintain normal dm/cv (burning tokens but advancing),
    # so this does not affect true positives.
    dm_coverage = float(ctx.snapshot.metrics.dm_coverage)
    cv_coverage = float(ctx.snapshot.metrics.cv_coverage)
    dm_zero_streak = 0
    for value in reversed(ctx.dm_coverage_series):
        if value == 0.0:
            dm_zero_streak += 1
        else:
            break
    strict_stagnation_evidence = (
        ctx.dm_stagnation_alarm
        or ctx.cv_stagnation_alarm
        or (dm_zero_streak >= 5 and len(ctx.dm_coverage_series) >= 5)
    ) and not ctx.source_productive
    relaxed_stagnation_evidence = strict_stagnation_evidence or (
        not ctx.source_productive
        and dm_coverage <= float(ctx.cfg.stuck_dm_threshold)
        and cv_coverage <= float(ctx.cfg.stuck_cv_threshold)
    )

    if relaxed_stagnation_evidence:
        loop_candidates = [
            (conf, trig) for conf, trig in loop_candidates if trig == "content_similarity"
        ]
        # Without stuck detection, low dm/cv is the best surrogate for
        # stuck-like behavior.  Genuine runaway traces maintain normal dm/cv
        # (tokens burn but coverage advances), so this preserves recall.
        build_runaway_detected = False
        build_runaway_confidence = None

    loop_detected = bool(loop_candidates)
    loop_confidence = 0.0
    loop_trigger: Optional[str] = None
    if loop_candidates:
        loop_confidence, loop_trigger = max(loop_candidates, key=lambda item: item[0])

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

    # --- Co-occurrence arbitration (ported from _resolve_detections) ---

    # Confabulation overlap: high-confidence confab suppresses runaway.
    if confabulation_detected and confabulation_confidence >= 0.85 and build_runaway_detected:
        build_runaway_detected = False
        build_runaway_confidence = None

    # Loop-runaway arbitration: when both fire, loop with content_similarity
    # wins unconditionally (direct observation of output repetition is
    # definitive loop evidence).  Without content_similarity, loop needs
    # significantly higher confidence to override runaway.
    if loop_detected and build_runaway_detected:
        _has_content_sim = loop_trigger == "content_similarity"
        runaway_conf = build_runaway_confidence or 0.0
        if _has_content_sim:
            build_runaway_detected = False
            build_runaway_confidence = None
        elif loop_confidence >= runaway_conf + 0.10:
            build_runaway_detected = False
            build_runaway_confidence = None
        else:
            # Runaway wins — suppress loop.
            loop_detected = False
            loop_confidence = 0.0
            loop_trigger = None

    # Error-count suppression: errors with loop co-occurrence suppress runaway.
    if ctx.error_count > 0 and loop_detected and build_runaway_detected:
        build_runaway_detected = False
        build_runaway_confidence = None

    detector_priority = (
        "confabulation"
        if confabulation_detected
        else ("loop" if loop_detected else ("runaway_cost" if build_runaway_detected else None))
    )
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
        runaway_cost_detected=build_runaway_detected,
        runaway_cost_confidence=_clip01(build_runaway_confidence or 0.0),
        hopfield_override_active=_consult_hopfield_override(ctx),
        detector_priority=detector_priority,
    )


def _resolve_detections(
    ctx: _DetectionContext,
    loop_candidates: list[tuple[float, str]],
    confab_candidates: list[tuple[float, str, tuple[str, ...]]],
    stuck_candidates: list[tuple[float, str]],
) -> LoopDetectionResult:
    """Cross-detector arbitration: suppression, priority resolution, and final result."""

    # Resolve loop candidates
    loop_detected = False
    loop_confidence = 0.0
    loop_candidate_confidence = 0.0
    loop_trigger: Optional[str] = None
    if loop_candidates:
        loop_confidence, loop_trigger = max(loop_candidates, key=lambda item: item[0])
        loop_candidate_confidence = loop_confidence
        loop_detected = True

    # Resolve confabulation candidates
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

    # Burn-rate anomaly: explicit runaway-cost signal (AV-S08-M04).
    # Computed as a separate flag rather than appended to stuck_candidates
    # so it cannot accidentally suppress stuck via arbitration sentinels.
    # The explicit suppression is applied below, gated on the same
    # conditions that the previous implicit "conf=1.0 wins arbitration"
    # behavior was effectively gated on (see the error/loop filters).
    runaway_cost_detected, runaway_cost_confidence = _compute_burn_rate_runaway(
        ctx, baseline_mode="mean", apply_signal_scale=True
    )

    # Loop signal hint suppresses stagnation-style stuck triggers.
    loop_signal_hint = _has_loop_signal_hint(
        loop_candidates=loop_candidates,
        findings_deltas=ctx.findings_deltas,
        token_deltas=ctx.token_deltas,
        query_deltas=ctx.query_deltas,
        domain_deltas=ctx.domain_deltas,
        domain_counts=ctx.domain_counts,
        coverage_scores=ctx.coverage_scores,
        output_similarity=ctx.output_similarity,
        sim_threshold=ctx.sim_threshold,
        consecutive=ctx.consecutive,
    )
    if stuck_candidates and loop_signal_hint:
        low_coverage = bool(ctx.coverage_scores) and float(ctx.coverage_scores[-1]) < 0.2
        stuck_candidates = [
            (conf, trigger)
            for conf, trigger in stuck_candidates
            if (
                trigger not in {"coverage_stagnation", "coverage_flat"}
                or (low_coverage and trigger == "coverage_stagnation")
            )
        ]

    # Low output similarity confirms stuck
    if ctx.output_similarity is not None and stuck_candidates:
        similarity = float(ctx.output_similarity)
        low_sim_threshold = 1.0 - ctx.sim_threshold
        if similarity <= low_sim_threshold:
            boost = 0.10 * min(
                1.0, (low_sim_threshold - similarity) / max(1e-9, low_sim_threshold)
            )
            best = max(stuck_candidates, key=lambda item: item[0])
            adjusted: list[tuple[float, str]] = []
            for conf, trigger in stuck_candidates:
                if (conf, trigger) == best:
                    adjusted.append((_clip01(conf + boost), trigger))
                else:
                    adjusted.append((conf, trigger))
            stuck_candidates = adjusted

    # Cross-detector suppression: error-induced or loop co-occurrence
    _INDEPENDENT_STUCK_TRIGGERS = {"zero_progress", "sources_zero"}
    if ctx.coverage_scores and float(ctx.coverage_scores[-1]) < 0.2:
        _INDEPENDENT_STUCK_TRIGGERS.add("coverage_stagnation")
    _INDEPENDENT_STUCK_TRIGGERS.add("short_run_zero_coverage")
    error_or_loop_filter_active = False
    if stuck_candidates and ctx.error_count > 0:
        stuck_candidates = [
            (conf, trigger)
            for conf, trigger in stuck_candidates
            if trigger in _INDEPENDENT_STUCK_TRIGGERS
        ]
        error_or_loop_filter_active = True
    elif stuck_candidates and loop_candidate_confidence >= 0.5:
        has_content_similarity = any(trig == "content_similarity" for _, trig in loop_candidates)
        if has_content_similarity:
            stuck_candidates = [
                (conf, trigger)
                for conf, trigger in stuck_candidates
                if trigger in _INDEPENDENT_STUCK_TRIGGERS
            ]
            error_or_loop_filter_active = True

    # Explicit burn-rate runaway suppression of stuck (AV-S08-M04).
    #
    # Pre-v1.14.1, burn_rate_anomaly was appended to stuck_candidates with
    # confidence 1.0; it would win arbitration whenever it fired and any
    # other stuck candidate was suppressed by being out-arbitrated. The
    # implicit "1.0 wins" behavior had the effect that:
    #   - When burn_rate fires AND error_count == 0 AND no high-conf
    #     loop+content_similarity, burn_rate wins → stuck suppressed.
    #   - When errors or loop+content_sim apply, the INDEPENDENT-trigger
    #     filter strips burn_rate from candidates first → no suppression,
    #     stuck can leak through normally.
    # This block reproduces that exact gating explicitly: when burn-rate
    # runaway fires and the error/loop filters are NOT active, clear
    # remaining stuck candidates because the runaway-cost signal is the
    # right label for this snapshot. Any future per-profile
    # burn_rate_multiplier override now sees a single explicit knob
    # instead of a side-channel buried in arbitration sentinels.
    if runaway_cost_detected and not error_or_loop_filter_active:
        stuck_candidates = []

    # Confabulation overlap handling
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

    # Co-occurrence priority resolution (av32-m02)
    detector_priority = None
    if confabulation_detected:
        loop_detected = False
        loop_confidence = 0.0
        loop_trigger = None
        detector_priority = "confabulation"
    elif loop_detected and stuck_detected:
        _has_content_sim = (
            any(trig == "content_similarity" for _, trig in loop_candidates)
            if loop_candidates
            else False
        )
        if stuck_trigger == "short_run_zero_coverage":
            detector_priority = "stuck"
        elif _has_content_sim and loop_confidence >= stuck_confidence:
            stuck_detected = False
            stuck_confidence = 0.0
            stuck_trigger = None
            detector_priority = "loop"
        elif loop_confidence >= stuck_confidence + 0.10:
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
    elif runaway_cost_detected:
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
        runaway_cost_detected=runaway_cost_detected,
        runaway_cost_confidence=_clip01(runaway_cost_confidence),
        hopfield_override_active=_consult_hopfield_override(ctx),
        detector_priority=detector_priority,
    )


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
        history: Previous snapshots (oldest -> newest). The current snapshot
            should not be included.
        config: Optional VitalsConfig override (defaults to env-derived config).
        workflow_type: Workflow type hint ("research", "build", "unknown").

    Returns:
        LoopDetectionResult with detection flags, confidence, and triggers.
    """

    cfg = config or get_vitals_config()
    ctx = _prepare_context(snapshot, history, cfg, workflow_type)
    if ctx is None:
        return LoopDetectionResult()

    loop_candidates = _detect_loop_candidates(ctx)
    confab_candidates = _detect_confabulation_candidates(ctx)

    if not _stuck_enabled_for_workflow(ctx.normalized_workflow, ctx.cfg.workflow_stuck_enabled):
        return _handle_stuck_disabled(ctx, loop_candidates, confab_candidates)

    stuck_candidates = _detect_stuck_candidates(ctx)
    return _resolve_detections(ctx, loop_candidates, confab_candidates, stuck_candidates)


# Backwards-compatible alias
detect_agent_loop = detect_loop


def _consult_hopfield_override(ctx: _DetectionContext) -> bool:
    """Hopfield 3rd-layer early-detection consultation (AV-S09-M02).

    Returns the value to write to ``LoopDetectionResult.hopfield_override_active``
    for the current snapshot. Lazy-imports the optional Hopfield adapter so the
    base ``agent_vitals`` install pays no cost when ``cfg.hopfield_enabled`` is
    False (the default) or when the ``agent-vitals[hopfield]`` extras are not
    installed.

    Gating layers:

    1. ``cfg.hopfield_enabled`` must be True (default False — preserves
       v1.14.2 behavior bit-identically when callers do not opt in).
    2. The optional ``onnxruntime`` backend must be importable; otherwise
       :func:`hopfield_override_fires` returns False without raising.
    3. The current trace length (history + current snapshot, exposed as
       ``ctx.series``) must fall inside the early-detection window
       ``[3, 6]`` enforced by :func:`hopfield_override_fires`. At length
       ≥7 the existing handcrafted+TDA stack is authoritative per bench's
       Five-Paradigm Comparative Report (intel_alert b7416ceb).

    The returned value never mutates the per-detector ``*_detected`` flags
    on :class:`LoopDetectionResult`; it propagates only as the explicit
    ``hopfield_override_active`` provenance marker so trace-level
    ``vitals.any`` numbers stay bit-identical against the v1.14.2 baseline
    on the bundled corpus.
    """

    if not getattr(ctx.cfg, "hopfield_enabled", False):
        return False
    try:
        from .hopfield import hopfield_override_fires
    except ImportError:  # pragma: no cover - safety net
        return False
    return hopfield_override_fires(
        ctx.series,
        model_dir=getattr(ctx.cfg, "hopfield_model_dir", None),
    )


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


def _windowed_ratio_declines(
    *,
    ratios: Sequence[Optional[float]],
    loop_indices: Sequence[int],
    window: int = 5,
) -> int:
    """Count step-over-step ratio declines in the last *window* steps.

    Unlike ``_consecutive_ratio_declines`` this tolerates non-declining
    steps inside the window — it counts the *total* declining steps rather
    than requiring them to be consecutive.  This catches V-shaped recovery
    patterns (Gemini), late-onset confabulation, and threshold-boundary
    bouncing that break the consecutive requirement.
    """

    if len(ratios) < 2 or len(loop_indices) != len(ratios):
        return 0

    start = max(0, len(ratios) - window)
    declines = 0
    epsilon = 1e-9
    for idx in range(start + 1, len(ratios)):
        current_ratio = ratios[idx]
        previous_ratio = ratios[idx - 1]
        if current_ratio is None or previous_ratio is None:
            continue
        if int(loop_indices[idx]) <= int(loop_indices[idx - 1]):
            continue
        if float(current_ratio) < (float(previous_ratio) - epsilon):
            declines += 1
    return declines


def _adaptive_alarm_from_series(
    *,
    values: Sequence[float],
    direction: AdaptiveDirection,
    fallback_threshold: Optional[float],
    config: VitalsConfig,
) -> tuple[bool, float, bool, bool]:
    """Replay a scalar series through AdaptiveThreshold and return last alarm state."""

    normalized_direction: AdaptiveDirection = (
        "decrease" if direction == "decrease" else "increase"
    )
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


def _compute_burn_rate_runaway(
    ctx: _DetectionContext,
    *,
    baseline_mode: str = "mean",
    apply_signal_scale: bool = True,
) -> tuple[bool, float]:
    """Compute the explicit burn-rate runaway-cost signal.

    Replaces the v1.14.1 implicit chain (AV-S08-M04). Returns
    ``(fired, confidence)``. Centralizes the burn-rate check so both
    ``_resolve_detections`` and ``_handle_stuck_disabled`` agree on
    the same signal. ``apply_signal_scale=True`` applies the
    ``burn_rate_multiplier_scale`` (model-size aware) — used in the
    stuck-enabled path. ``apply_signal_scale=False`` skips it — the
    stuck-disabled path historically did not apply the scale because
    it caused FNs on small-model traces.
    """

    burn_cfg = ctx.cfg
    if apply_signal_scale and ctx.signal_map.burn_rate_multiplier_scale != 1.0:
        burn_cfg = _dc_replace(
            ctx.cfg,
            burn_rate_multiplier=ctx.cfg.burn_rate_multiplier
            * ctx.signal_map.burn_rate_multiplier_scale,
        )
    confidence = _burn_rate_anomaly_confidence(
        token_deltas=ctx.token_deltas,
        findings_deltas=ctx.findings_deltas,
        coverage_scores=ctx.coverage_scores,
        cfg=burn_cfg,
        baseline_mode=baseline_mode,
    )
    if confidence is None:
        return False, 0.0
    return True, float(confidence)


def _burn_rate_anomaly_confidence(
    *,
    token_deltas: Sequence[float],
    findings_deltas: Sequence[float],
    coverage_scores: Sequence[float],
    cfg: VitalsConfig,
    baseline_mode: str = "mean",
) -> float | None:
    """Return runaway-confidence when burn rate spikes against recent baseline.

    Args:
        baseline_mode: ``"mean"`` (default) averages all prior ratios.
            ``"min"`` uses the lowest prior ratio, which resists spike
            contamination — once a burn-rate spike enters the baseline,
            ``mean`` inflates the threshold and masks subsequent spikes.
    """

    if not token_deltas or not findings_deltas or not coverage_scores:
        return None
    if coverage_scores[-1] >= 0.95:
        return None

    scale = max(0.01, float(cfg.token_scale_factor))
    current_tokens = token_deltas[-1] * scale
    current_findings = findings_deltas[-1]
    if current_tokens <= 0.0:
        return None

    baseline_ratios: list[float] = []
    baseline_token_deltas: list[float] = []
    for index in range(0, len(findings_deltas) - 1):
        baseline_tokens = token_deltas[index] * scale
        baseline_findings = findings_deltas[index]
        if baseline_tokens <= 0.0:
            continue
        baseline_token_deltas.append(baseline_tokens)
        if baseline_findings > 0.0:
            baseline_ratios.append(baseline_tokens / baseline_findings)

    if not baseline_ratios:
        return None

    baseline = float(min(baseline_ratios) if baseline_mode == "min" else mean(baseline_ratios))
    if baseline <= 0.0:
        return None

    ratio = math.inf if current_findings <= 0.0 else float(current_tokens / current_findings)
    multiplier = float(cfg.burn_rate_multiplier)
    if ratio <= multiplier * baseline:
        return None

    if current_findings <= 0.0 and baseline_token_deltas:
        token_baseline = float(
            min(baseline_token_deltas) if baseline_mode == "min" else mean(baseline_token_deltas)
        )
        token_threshold = multiplier * token_baseline
        if current_tokens <= token_threshold:
            return None

    factor = min(1.0, float(ratio / (multiplier * baseline)))
    return min(1.0, 0.7 + 0.3 * factor)


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
        if all(delta <= 0.0 for delta in recent_findings) and all(
            delta > 0.0 for delta in recent_tokens
        ):
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
        if all(delta > 0.0 for delta in recent_queries) and all(
            delta <= 0.0 for delta in recent_domains
        ):
            return True

    if output_similarity is not None:
        relaxed_threshold = _clip01(float(sim_threshold) - 0.05)
        if float(output_similarity) >= relaxed_threshold:
            return True

    return False


__all__ = ["LoopDetectionResult", "detect_agent_loop", "detect_loop"]
