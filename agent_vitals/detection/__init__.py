"""Detection engine for Agent Vitals.

Re-exports the core detection components for convenient access.
"""

from .loop import LoopDetectionResult, detect_agent_loop, detect_loop
from .metrics import HysteresisConfig, TemporalMetrics
from .cusum import CUSUMTracker, CUSUMUpdate
from .adaptive_threshold import AdaptiveThreshold, AdaptiveThresholdUpdate
from .similarity import (
    SimilarityResult,
    compute_output_fingerprint,
    compute_pairwise_similarity,
    compute_similarity_scores,
)
from .stop_rule import StopRuleSignals, derive_stop_signals

__all__ = [
    "HysteresisConfig",
    "AdaptiveThreshold",
    "AdaptiveThresholdUpdate",
    "CUSUMTracker",
    "CUSUMUpdate",
    "LoopDetectionResult",
    "SimilarityResult",
    "StopRuleSignals",
    "TemporalMetrics",
    "compute_output_fingerprint",
    "compute_pairwise_similarity",
    "compute_similarity_scores",
    "derive_stop_signals",
    "detect_agent_loop",
    "detect_loop",
]
