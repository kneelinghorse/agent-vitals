"""Tests for the TDA adjudication layer (av-s06-m02).

Covers:
- Module is importable without optional TDA dependencies installed
- ``is_tda_available`` reflects backend presence
- ``extract_tda_features`` and ``predict_runaway_cost`` short-circuit
  on traces shorter than ``min_steps`` without raising
- Hybrid integration in ``_replay_trace`` overrides handcrafted
  positives when TDA says "not runaway"
- Hybrid integration leaves the handcrafted verdict alone when TDA
  is disabled, the model is missing, or the backend is unavailable
"""

from __future__ import annotations

import pytest

from agent_vitals.backtest import _replay_trace
from agent_vitals.config import VitalsConfig
from agent_vitals.detection import tda as tda_module
from agent_vitals.detection.loop import LoopDetectionResult
from agent_vitals.detection.stop_rule import StopRuleSignals
from agent_vitals.detection.tda import (
    MissingTDADependencyError,
    TDARunawayPrediction,
    extract_tda_features,
    is_tda_available,
    predict_runaway_cost,
)
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


def _snapshot(loop_index: int = 0) -> VitalsSnapshot:
    return VitalsSnapshot(
        mission_id="t1",
        loop_index=loop_index,
        signals=RawSignals(
            findings_count=1,
            coverage_score=0.5,
            total_tokens=1000,
            error_count=0,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.1,
            cv_findings_rate=0.1,
            dm_coverage=0.3,
            dm_findings=0.3,
            qpf_tokens=0.5,
            cs_effort=0.5,
        ),
        health_state="healthy",
    )


class TestTDAModule:
    def test_module_importable_without_deps(self) -> None:
        """The module must import cleanly even when gtda is missing.

        The top-of-file import already proves this — if it failed the
        whole test module would refuse to collect. We additionally check
        the public surface here so a partial-import regression is caught.
        """
        assert hasattr(tda_module, "extract_tda_features")
        assert hasattr(tda_module, "predict_runaway_cost")
        assert hasattr(tda_module, "is_tda_available")
        assert hasattr(tda_module, "TDAConfig")
        assert hasattr(tda_module, "TDARunawayPrediction")
        assert hasattr(tda_module, "MissingTDADependencyError")

    def test_is_tda_available_returns_bool(self) -> None:
        assert isinstance(is_tda_available(), bool)

    def test_extract_features_short_trace_returns_none(self) -> None:
        """Traces with fewer than min_steps snapshots short-circuit to None."""
        if not is_tda_available():
            with pytest.raises(MissingTDADependencyError):
                extract_tda_features([_snapshot(i) for i in range(3)])
            return
        result = extract_tda_features([_snapshot(i) for i in range(3)])
        assert result is None

    def test_predict_short_trace_returns_none(self) -> None:
        if not is_tda_available():
            with pytest.raises(MissingTDADependencyError):
                predict_runaway_cost([_snapshot(i) for i in range(3)])
            return
        result = predict_runaway_cost([_snapshot(i) for i in range(3)])
        assert result is None


class TestHybridReplayIntegration:
    """End-to-end checks of the hybrid override in _replay_trace."""

    def test_tda_disabled_keeps_handcrafted_runaway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tda_enabled=False the handcrafted verdict must stand."""
        snapshots = [_snapshot(i) for i in range(6)]
        detections = iter([LoopDetectionResult() for _ in range(6)])
        # Final snapshot triggers handcrafted runaway via stop signals.
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(5)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        cfg = VitalsConfig()  # tda_enabled defaults to False
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is True

    def test_tda_override_flips_handcrafted_positive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With tda_enabled=True and TDA saying 'not runaway', flip to False."""
        snapshots = [_snapshot(i) for i in range(6)]
        detections = iter([LoopDetectionResult() for _ in range(6)])
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(5)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )
        # Stub out TDA prediction so this test runs without gtda.
        monkeypatch.setattr(
            "agent_vitals.detection.tda.predict_runaway_cost",
            lambda *_args, **_kwargs: TDARunawayPrediction(detected=False, probability=0.05),
        )

        cfg = VitalsConfig(tda_enabled=True)
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is False

    def test_tda_confirm_keeps_handcrafted_positive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When TDA confirms (detected=True) the runaway label stands."""
        snapshots = [_snapshot(i) for i in range(6)]
        detections = iter([LoopDetectionResult() for _ in range(6)])
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(5)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )
        monkeypatch.setattr(
            "agent_vitals.detection.tda.predict_runaway_cost",
            lambda *_args, **_kwargs: TDARunawayPrediction(detected=True, probability=0.95),
        )

        cfg = VitalsConfig(tda_enabled=True)
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is True

    def test_tda_short_trace_falls_back_to_handcrafted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Traces with fewer than 5 snapshots skip the TDA layer entirely."""
        snapshots = [_snapshot(i) for i in range(4)]
        detections = iter([LoopDetectionResult() for _ in range(4)])
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(3)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )
        # If TDA were called this stub would flip the verdict — assert it
        # is NOT called by checking the runaway label survives.
        monkeypatch.setattr(
            "agent_vitals.detection.tda.predict_runaway_cost",
            lambda *_args, **_kwargs: TDARunawayPrediction(detected=False, probability=0.05),
        )

        cfg = VitalsConfig(tda_enabled=True)
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is True

    def test_missing_tda_deps_falls_back_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the TDA backend raises MissingTDADependencyError the
        handcrafted verdict stands and no exception escapes."""
        snapshots = [_snapshot(i) for i in range(6)]
        detections = iter([LoopDetectionResult() for _ in range(6)])
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(5)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise MissingTDADependencyError("simulated missing backend")

        monkeypatch.setattr(
            "agent_vitals.detection.tda.predict_runaway_cost", _raise
        )

        cfg = VitalsConfig(tda_enabled=True)
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is True

    def test_missing_model_artifact_falls_back_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshots = [_snapshot(i) for i in range(6)]
        detections = iter([LoopDetectionResult() for _ in range(6)])
        stop_signals = iter(
            [StopRuleSignals(False, False, False, False) for _ in range(5)]
            + [StopRuleSignals(False, False, False, True)]
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.detect_loop",
            lambda *_args, **_kwargs: next(detections),
        )
        monkeypatch.setattr(
            "agent_vitals.backtest.derive_stop_signals",
            lambda *_args, **_kwargs: next(stop_signals),
        )

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError("simulated missing model artifact")

        monkeypatch.setattr(
            "agent_vitals.detection.tda.predict_runaway_cost", _raise
        )

        cfg = VitalsConfig(tda_enabled=True)
        fired = _replay_trace(snapshots, config=cfg, workflow_type="research")
        assert fired["runaway_cost"] is True
