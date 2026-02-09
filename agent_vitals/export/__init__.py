"""Export integrations for Agent Vitals.

Provides the ``VitalsExporter`` protocol and concrete exporters.
Built-in exporters:
- ``JSONLExporter`` for local JSONL logs
- ``OTLPExporter`` for OpenTelemetry OTLP HTTP metrics
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import VitalsSnapshot


@runtime_checkable
class VitalsExporter(Protocol):
    """Protocol that all vitals exporters must satisfy.

    Exporters receive snapshots on each ``AgentVitals.step()`` call
    and are flushed/closed when the monitoring session ends.
    """

    def export(self, snapshot: VitalsSnapshot) -> None:
        """Export a single vitals snapshot."""
        ...

    def flush(self) -> None:
        """Flush any buffered data to the underlying sink."""
        ...

    def close(self) -> None:
        """Release resources held by this exporter."""
        ...


# Re-export concrete implementations
from .jsonl import JSONLExporter  # noqa: E402
from .otlp import OTLPExporter  # noqa: E402

__all__ = [
    "JSONLExporter",
    "OTLPExporter",
    "VitalsExporter",
]
