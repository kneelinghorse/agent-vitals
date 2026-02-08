"""Detection engine for Agent Vitals.

Re-exports the core detection components for convenient access.
"""

from .loop import LoopDetectionResult, detect_agent_loop, detect_loop
from .metrics import HysteresisConfig, TemporalMetrics
from .stop_rule import StopRuleSignals, derive_stop_signals

__all__ = [
    "HysteresisConfig",
    "LoopDetectionResult",
    "StopRuleSignals",
    "TemporalMetrics",
    "derive_stop_signals",
    "detect_agent_loop",
    "detect_loop",
]
