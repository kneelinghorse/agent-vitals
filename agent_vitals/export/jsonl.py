"""JSONL exporter for Agent Vitals snapshots.

Writes one JSON line per snapshot to a local file. Supports two layouts:

- **per_run**: ``{directory}/{mission_id}/{run_id}.jsonl``
- **append**: ``{directory}/{mission_id}.jsonl`` (all runs in one file)

Append mode supports ``max_bytes`` rotation: when the file would exceed
the threshold, it is renamed with an incrementing suffix and a fresh
file is started.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional

from ..schema import VitalsSnapshot

logger = logging.getLogger(__name__)

LayoutMode = Literal["per_run", "append"]

DEFAULT_MAX_BYTES = 10_000_000  # 10 MB


def _safe_filename(value: str) -> str:
    """Sanitize a string for use as a filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip())
    return cleaned.strip("-") or "default"


def _rotate(path: Path) -> None:
    """Rotate a file by renaming to the next available numeric suffix."""
    index = 1
    while True:
        candidate = Path(f"{path}.{index}")
        if not candidate.exists():
            path.rename(candidate)
            return
        index += 1


class JSONLExporter:
    """Write vitals snapshots as newline-delimited JSON to local files.

    Args:
        directory: Base directory for JSONL output.
        layout: ``"per_run"`` creates a subdirectory per mission with
            one file per run_id. ``"append"`` writes all runs for a
            mission into a single file.
        max_bytes: Maximum file size in bytes before rotation (append
            mode only). Set to 0 to disable rotation.

    Example::

        from agent_vitals.export import JSONLExporter

        exporter = JSONLExporter(directory="./vitals_logs", layout="per_run")
        monitor = AgentVitals(exporters=[exporter])
    """

    def __init__(
        self,
        *,
        directory: str | Path = "vitals_logs",
        layout: LayoutMode = "per_run",
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._directory = Path(directory)
        self._layout: LayoutMode = layout
        self._max_bytes = max(0, max_bytes)
        self._closed = False

    @property
    def directory(self) -> Path:
        """Return the base output directory."""
        return self._directory

    @property
    def layout(self) -> LayoutMode:
        """Return the layout mode."""
        return self._layout

    @property
    def max_bytes(self) -> int:
        """Return the rotation threshold."""
        return self._max_bytes

    def export(self, snapshot: VitalsSnapshot) -> None:
        """Write a single snapshot as a JSON line.

        Silently skips on I/O errors to avoid crashing the monitored agent.
        """
        if self._closed:
            logger.warning("JSONLExporter.export() called after close(); ignoring.")
            return

        path = self._resolve_path(snapshot)
        line = snapshot.model_dump_json()
        encoded = (line + "\n").encode("utf-8")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Cannot create JSONL directory %s: %s", path.parent, exc)
            return

        # Rotation (append mode only)
        if self._layout == "append" and self._max_bytes > 0:
            try:
                if path.exists() and path.stat().st_size + len(encoded) > self._max_bytes:
                    _rotate(path)
            except Exception as exc:
                logger.warning("JSONL rotation check failed: %s", exc)

        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            logger.warning("JSONL write failed for %s: %s", path, exc)

    def flush(self) -> None:
        """No-op for file-based exporter (writes are unbuffered)."""
        pass

    def close(self) -> None:
        """Mark this exporter as closed."""
        self._closed = True

    def _resolve_path(self, snapshot: VitalsSnapshot) -> Path:
        """Determine the output file path based on layout mode."""
        mission = _safe_filename(snapshot.mission_id)

        if self._layout == "per_run":
            run_id = _safe_filename(snapshot.run_id or "default")
            return self._directory / mission / f"{run_id}.jsonl"
        else:
            return self._directory / f"{mission}.jsonl"


__all__ = ["JSONLExporter"]
