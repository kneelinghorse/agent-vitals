"""Tests for agent_vitals.config module."""

from pathlib import Path

import pytest

import agent_vitals.config as config_module
from agent_vitals.config import VitalsConfig, get_vitals_config


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
        assert cfg.burn_rate_multiplier == 2.5
        assert cfg.thrash_error_threshold == 1
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

    def test_get_vitals_config_uses_yaml_thresholds(self, tmp_path, monkeypatch) -> None:
        """get_vitals_config should load threshold values from YAML."""
        yaml_path = tmp_path / "thresholds.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "loop_consecutive_pct: 0.42",
                    "stuck_cv_threshold: 0.55",
                    "profiles:",
                    "  langgraph:",
                    "    loop_consecutive_pct: 0.33",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", yaml_path)
        for env_name in (
            "VITALS_LOOP_CONSECUTIVE_PCT",
            "VITALS_STUCK_CV_THRESHOLD",
        ):
            monkeypatch.delenv(env_name, raising=False)

        cfg = get_vitals_config()

        assert cfg.loop_consecutive_pct == 0.42
        assert cfg.stuck_cv_threshold == 0.55
        assert cfg.for_framework("langgraph").loop_consecutive_pct == 0.33

    def test_get_vitals_config_env_overrides_yaml(self, tmp_path, monkeypatch) -> None:
        """Explicit env vars should still override YAML-loaded thresholds."""
        yaml_path = tmp_path / "thresholds.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "loop_consecutive_pct: 0.42",
                    "stuck_cv_threshold: 0.55",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", yaml_path)
        monkeypatch.setenv("VITALS_STUCK_CV_THRESHOLD", "0.91")
        monkeypatch.delenv("VITALS_LOOP_CONSECUTIVE_PCT", raising=False)

        cfg = get_vitals_config()

        assert cfg.loop_consecutive_pct == 0.42
        assert cfg.stuck_cv_threshold == 0.91


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
        cfg = VitalsConfig.from_dict({"jsonl_dir": "/tmp/vitals"})
        assert cfg.jsonl_dir == Path("/tmp/vitals")


class TestCUSUMConfig:
    """Tests for CUSUM tracker config params."""

    def test_cusum_defaults(self) -> None:
        cfg = VitalsConfig()
        assert cfg.cusum_k_sigma == 0.5
        assert cfg.cusum_h_sigma == 4.0
        assert cfg.cusum_warmup_steps == 2
        assert cfg.cusum_min_sigma_similarity == 0.05
        assert cfg.cusum_min_sigma_tokens == 25.0
        assert cfg.cusum_min_sigma_findings == 0.5

    def test_cusum_from_dict(self) -> None:
        cfg = VitalsConfig.from_dict({
            "cusum_k_sigma": 1.0,
            "cusum_h_sigma": 5.0,
            "cusum_warmup_steps": 3,
            "cusum_min_sigma_similarity": 0.1,
            "cusum_min_sigma_tokens": 50.0,
            "cusum_min_sigma_findings": 1.0,
        })
        assert cfg.cusum_k_sigma == 1.0
        assert cfg.cusum_h_sigma == 5.0
        assert cfg.cusum_warmup_steps == 3
        assert cfg.cusum_min_sigma_similarity == 0.1
        assert cfg.cusum_min_sigma_tokens == 50.0
        assert cfg.cusum_min_sigma_findings == 1.0

    def test_cusum_from_yaml(self) -> None:
        """Bundled thresholds.yaml includes CUSUM params."""
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert cfg.cusum_k_sigma == 0.5
        assert cfg.cusum_h_sigma == 4.0
        assert cfg.cusum_warmup_steps == 2
        assert cfg.cusum_min_sigma_similarity == 0.05
        assert cfg.cusum_min_sigma_tokens == 25.0
        assert cfg.cusum_min_sigma_findings == 0.5

    def test_cusum_profile_override(self) -> None:
        """Framework profiles can override CUSUM params."""
        from agent_vitals.config import ThresholdProfile

        profiles = (ThresholdProfile(
            framework="custom",
            cusum_k_sigma=0.8,
            cusum_h_sigma=3.0,
            cusum_warmup_steps=5,
            cusum_min_sigma_similarity=0.02,
        ),)
        cfg = VitalsConfig(framework_profiles=profiles)
        resolved = cfg.for_framework("custom")
        assert resolved.cusum_k_sigma == 0.8
        assert resolved.cusum_h_sigma == 3.0
        assert resolved.cusum_warmup_steps == 5
        assert resolved.cusum_min_sigma_similarity == 0.02
        # Non-overridden fields keep defaults
        assert resolved.cusum_min_sigma_tokens == 25.0
        assert resolved.cusum_min_sigma_findings == 0.5


def test_version_matches_pyproject() -> None:
    """agent_vitals.__version__ should match the version in pyproject.toml."""
    # tomllib is stdlib only on Python 3.11+. Skip on 3.10 — that runner
    # still exercises the rest of the test module and the version check
    # runs on 3.11/3.12 which is sufficient drift detection.
    tomllib = pytest.importorskip("tomllib")

    import agent_vitals

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as fh:
        pyproject = tomllib.load(fh)
    expected = pyproject["project"]["version"]
    assert agent_vitals.__version__ == expected, (
        f"__version__={agent_vitals.__version__!r} != pyproject.toml version={expected!r}"
    )
