"""Error hierarchy for Agent Vitals.

All agent-vitals exceptions inherit from VitalsError so callers can
catch the base class for broad error handling.
"""

from __future__ import annotations


class VitalsError(Exception):
    """Base exception for all agent-vitals errors."""


class ConfigurationError(VitalsError):
    """Invalid configuration or threshold values."""


class UnknownProfileError(ConfigurationError):
    """Raised when a requested framework profile is not registered.

    The exception message includes the list of known profile names so
    callers (especially external verifiers and agent integrators) can
    recover without reading source. Use ``VitalsConfig.list_profiles()``
    or ``VitalsConfig().profiles()`` for a soft lookup before calling
    ``profile_diff``.
    """


class AdapterError(VitalsError):
    """Signal extraction from agent state failed."""


class ExportError(VitalsError):
    """Export backend failed.

    Non-fatal by default — the monitor logs warnings but never crashes
    the host agent unless strict_export mode is enabled.
    """


class BacktestError(VitalsError):
    """Error during backtest evaluation."""


__all__ = [
    "AdapterError",
    "BacktestError",
    "ConfigurationError",
    "ExportError",
    "UnknownProfileError",
    "VitalsError",
]
