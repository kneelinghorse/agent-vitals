"""Agent Vitals — Standalone agent health monitor.

Detect loops, stuck states, thrash, and runaway costs in any AI agent workflow.

Usage::

    from agent_vitals import AgentVitals, VitalsSnapshot, RawSignals

    monitor = AgentVitals(mission_id="my-task")
    snapshot = monitor.step(
        findings_count=5,
        coverage_score=0.6,
        total_tokens=12000,
        error_count=0,
    )
"""

from .config import VitalsConfig, get_vitals_config
from .detection.loop import LoopDetectionResult, detect_loop
from .detection.stop_rule import StopRuleSignals, derive_stop_signals
from .exceptions import AdapterError, BacktestError, ConfigurationError, ExportError, VitalsError
from .export import JSONLExporter, OTLPExporter, VitalsExporter
from .monitor import AgentVitals
from .schema import (
    HealthState,
    InterventionRecord,
    RawSignals,
    TemporalMetricsResult,
    VitalsSnapshot,
)

__version__ = "1.2.0"

__all__ = [
    "AdapterError",
    "AgentVitals",
    "BacktestError",
    "ConfigurationError",
    "ExportError",
    "HealthState",
    "InterventionRecord",
    "JSONLExporter",
    "LoopDetectionResult",
    "OTLPExporter",
    "RawSignals",
    "StopRuleSignals",
    "TemporalMetricsResult",
    "VitalsConfig",
    "VitalsError",
    "VitalsExporter",
    "VitalsSnapshot",
    "derive_stop_signals",
    "detect_loop",
    "get_vitals_config",
]
