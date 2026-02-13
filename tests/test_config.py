"""Tests for agent_vitals.config module."""

from agent_vitals.config import VitalsConfig


class TestVitalsConfig:
    """Tests for VitalsConfig."""

    def test_defaults(self) -> None:
        cfg = VitalsConfig()
        assert cfg.loop_consecutive_pct == 0.5
        assert cfg.findings_plateau_pct == 0.4
        assert cfg.min_evidence_steps == 3
        assert cfg.source_finding_ratio_floor == 0.3
        assert cfg.source_finding_ratio_declining_steps == 3
        assert cfg.loop_consecutive_count == 3
        assert cfg.stuck_dm_threshold == 0.15
        assert cfg.stuck_cv_threshold == 0.3
        assert cfg.burn_rate_multiplier == 3.0
        assert cfg.token_scale_factor == 1.0
        assert cfg.spc_k_sigma == 3.0
        assert cfg.spc_window_size == 5
        assert cfg.spc_warmup_steps == 2
        assert cfg.spc_cooldown_steps == 1
        assert cfg.spc_wma_decay == 0.7
        assert cfg.workflow_stuck_enabled == "research-only"

    def test_from_dict(self) -> None:
        cfg = VitalsConfig.from_dict({
            "loop_consecutive_pct": 0.45,
            "findings_plateau_pct": 0.35,
            "min_evidence_steps": 4,
            "source_finding_ratio_floor": 0.25,
            "source_finding_ratio_declining_steps": 4,
            "spc_k_sigma": 2.5,
            "spc_window_size": 7,
            "spc_warmup_steps": 3,
            "spc_cooldown_steps": 2,
            "spc_wma_decay": 0.8,
            "stuck_dm_threshold": 0.2,
            "enabled": True,
            "enforcement": False,
            "workflow_stuck_enabled": "all",
        })
        assert cfg.loop_consecutive_pct == 0.45
        assert cfg.findings_plateau_pct == 0.35
        assert cfg.min_evidence_steps == 4
        assert cfg.source_finding_ratio_floor == 0.25
        assert cfg.source_finding_ratio_declining_steps == 4
        assert cfg.spc_k_sigma == 2.5
        assert cfg.spc_window_size == 7
        assert cfg.spc_warmup_steps == 3
        assert cfg.spc_cooldown_steps == 2
        assert cfg.spc_wma_decay == 0.8
        assert cfg.stuck_dm_threshold == 0.2
        assert cfg.workflow_stuck_enabled == "all"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = VitalsConfig.from_dict({
            "unknown_key": "should_be_ignored",
            "loop_consecutive_count": 3,
        })
        assert cfg.loop_consecutive_count == 3

    def test_from_dict_handles_bad_types(self) -> None:
        cfg = VitalsConfig.from_dict({
            "loop_consecutive_count": "not_a_number",
        })
        # Should fall back to default
        assert cfg.loop_consecutive_count == 3

    def test_hysteresis_config(self) -> None:
        cfg = VitalsConfig(
            th_enter_warning=0.3,
            th_exit_warning=0.7,
            th_enter_critical=0.1,
            th_exit_critical=0.25,
        )
        hyst = cfg.hysteresis_config()
        assert hyst.enter_warning == 0.3
        assert hyst.exit_warning == 0.7

    def test_from_yaml_with_bundled_file(self) -> None:
        """Test loading from the bundled thresholds.yaml."""
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert cfg.loop_consecutive_pct == 0.5
        assert cfg.findings_plateau_pct == 0.4
        assert cfg.min_evidence_steps == 3
        assert cfg.source_finding_ratio_floor == 0.3
        assert cfg.source_finding_ratio_declining_steps == 3
        assert cfg.spc_k_sigma == 3.0
        assert cfg.spc_window_size == 5
        assert cfg.spc_warmup_steps == 2
        assert cfg.spc_cooldown_steps == 1
        assert cfg.spc_wma_decay == 0.7
        assert cfg.stuck_dm_threshold == 0.15


class TestVitalsConfigFromDict:
    """Tests for VitalsConfig.from_dict edge cases."""

    def test_bool_coercion(self) -> None:
        cfg = VitalsConfig.from_dict({"enabled": "true", "enforcement": 0})
        assert cfg.enabled is True
        assert cfg.enforcement is False

    def test_float_coercion(self) -> None:
        cfg = VitalsConfig.from_dict({"stuck_dm_threshold": "0.25"})
        assert cfg.stuck_dm_threshold == 0.25

    def test_path_coercion(self) -> None:
        from pathlib import Path
        cfg = VitalsConfig.from_dict({"jsonl_dir": "/tmp/vitals"})
        assert cfg.jsonl_dir == Path("/tmp/vitals")
