"""Tests for the export module: VitalsExporter protocol and JSONLExporter."""

from __future__ import annotations

import json
from pathlib import Path


from agent_vitals import AgentVitals, JSONLExporter, VitalsExporter, VitalsSnapshot


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_jsonl_exporter_satisfies_protocol() -> None:
    """JSONLExporter must be a runtime-checkable VitalsExporter."""
    exporter = JSONLExporter(directory="/tmp/test")
    assert isinstance(exporter, VitalsExporter)


# ---------------------------------------------------------------------------
# JSONLExporter unit tests
# ---------------------------------------------------------------------------


def _make_snapshot(mission_id: str = "test-mission", run_id: str = "run-1") -> VitalsSnapshot:
    """Create a minimal valid VitalsSnapshot for testing."""
    from agent_vitals.schema import RawSignals, TemporalMetricsResult

    return VitalsSnapshot(
        mission_id=mission_id,
        run_id=run_id,
        loop_index=0,
        signals=RawSignals(
            findings_count=3,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.1,
            cv_findings_rate=0.2,
            dm_coverage=0.3,
            dm_findings=0.4,
            qpf_tokens=0.5,
            cs_effort=0.6,
        ),
        health_state="healthy",
    )


class TestJSONLExporterPerRun:
    """Test per_run layout mode."""

    def test_creates_file_per_run(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        snap = _make_snapshot(mission_id="m1", run_id="r1")
        exporter.export(snap)

        expected = tmp_path / "m1" / "r1.jsonl"
        assert expected.exists()
        lines = expected.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["mission_id"] == "m1"
        assert data["signals"]["findings_count"] == 3

    def test_multiple_steps_append_to_same_file(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        for i in range(5):
            snap = _make_snapshot(mission_id="m1", run_id="r1")
            exporter.export(snap)

        path = tmp_path / "m1" / "r1.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_different_runs_create_separate_files(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        exporter.export(_make_snapshot(mission_id="m1", run_id="r1"))
        exporter.export(_make_snapshot(mission_id="m1", run_id="r2"))

        assert (tmp_path / "m1" / "r1.jsonl").exists()
        assert (tmp_path / "m1" / "r2.jsonl").exists()


class TestJSONLExporterAppend:
    """Test append layout mode."""

    def test_writes_to_single_file(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="append")
        exporter.export(_make_snapshot(mission_id="m1", run_id="r1"))
        exporter.export(_make_snapshot(mission_id="m1", run_id="r2"))

        path = tmp_path / "m1.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_rotation_on_max_bytes(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="append", max_bytes=200)
        snap = _make_snapshot(mission_id="rot")

        # Write enough to trigger rotation
        exporter.export(snap)
        first_size = (tmp_path / "rot.jsonl").stat().st_size
        assert first_size > 100  # sanity check

        # Second write should trigger rotation
        exporter.export(snap)

        rotated = tmp_path / "rot.jsonl.1"
        assert rotated.exists(), "Rotated file should exist"
        assert (tmp_path / "rot.jsonl").exists(), "New file should exist after rotation"

    def test_rotation_disabled_when_max_bytes_zero(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="append", max_bytes=0)
        snap = _make_snapshot(mission_id="no-rot")

        for _ in range(10):
            exporter.export(snap)

        assert not (tmp_path / "no-rot.jsonl.1").exists()
        lines = (tmp_path / "no-rot.jsonl").read_text().strip().split("\n")
        assert len(lines) == 10


class TestJSONLExporterLifecycle:
    """Test flush/close lifecycle."""

    def test_close_prevents_further_writes(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        exporter.export(_make_snapshot())
        exporter.close()
        # Should not crash, just log warning
        exporter.export(_make_snapshot())

        # Only one line should exist (the one before close)
        path = tmp_path / "test-mission" / "run-1.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_flush_is_noop(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        exporter.flush()  # should not raise


class TestJSONLExporterFilenamesSafety:
    """Test that unsafe characters are sanitized in filenames."""

    def test_special_chars_in_mission_id(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="append")
        snap = _make_snapshot(mission_id="my/mission:v2!@#$")
        exporter.export(snap)

        # Should create a sanitized filename
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        assert "/" not in files[0].name
        assert ":" not in files[0].name


# ---------------------------------------------------------------------------
# Monitor + exporter integration
# ---------------------------------------------------------------------------


class TestMonitorExporterIntegration:
    """Integration test: AgentVitals wired with JSONLExporter."""

    def test_step_triggers_export(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        monitor = AgentVitals(
            mission_id="int-test",
            run_id="run-42",
            exporters=[exporter],
        )

        for i in range(3):
            monitor.step(
                findings_count=i + 1,
                coverage_score=0.1 * (i + 1),
                total_tokens=1000 * (i + 1),
                error_count=0,
            )

        path = tmp_path / "int-test" / "run-42.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify content correctness
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["loop_index"] == i
            assert data["signals"]["findings_count"] == i + 1
            assert "source_finding_ratio" in data
            assert "ratio_trend" in data
            assert "confabulation_confidence" in data
            assert "confabulation_signals" in data

    def test_context_manager_flushes_and_closes(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")

        with AgentVitals(
            mission_id="ctx-test",
            run_id="run-ctx",
            exporters=[exporter],
        ) as monitor:
            monitor.step(
                findings_count=1,
                coverage_score=0.5,
                total_tokens=500,
                error_count=0,
            )

        # After context manager, exporter should be closed
        assert exporter._closed is True

    def test_reset_closes_exporters(self, tmp_path: Path) -> None:
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        monitor = AgentVitals(
            mission_id="reset-test",
            exporters=[exporter],
        )

        monitor.step(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=500,
            error_count=0,
        )
        monitor.reset()

        assert exporter._closed is True

    def test_exporter_failure_does_not_crash_step(self, tmp_path: Path) -> None:
        """A broken exporter should not prevent step() from returning."""

        class BrokenExporter:
            def export(self, snapshot: VitalsSnapshot) -> None:
                raise RuntimeError("I'm broken")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        monitor = AgentVitals(
            mission_id="broken-test",
            exporters=[BrokenExporter()],  # type: ignore[list-item]
        )

        # Should not raise
        snapshot = monitor.step(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=500,
            error_count=0,
        )
        assert snapshot is not None
        assert snapshot.loop_index == 0

    def test_full_session_produces_valid_jsonl(self, tmp_path: Path) -> None:
        """End-to-end: full monitor session produces parseable JSONL with all fields."""
        exporter = JSONLExporter(directory=tmp_path, layout="per_run")
        monitor = AgentVitals(
            mission_id="e2e-session",
            run_id="full-run",
            exporters=[exporter],
        )

        # Simulate a 6-step healthy run
        for i in range(6):
            monitor.step(
                findings_count=i + 1,
                coverage_score=min(1.0, 0.15 * (i + 1)),
                total_tokens=500 * (i + 1),
                error_count=0,
            )

        path = tmp_path / "e2e-session" / "full-run.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 6

        # Every line must be valid JSON and deserialize to VitalsSnapshot
        for line in lines:
            data = json.loads(line)
            snap = VitalsSnapshot.model_validate(data)
            assert snap.mission_id == "e2e-session"
            assert snap.run_id == "full-run"
            assert snap.spec_version == "1.0.0"
