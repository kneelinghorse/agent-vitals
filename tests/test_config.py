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


class TestProfileIntrospectionAPI:
    """Tests for the av-s07-m01 profile introspection API.

    This API exists so external verifiers (notably the bench harness)
    can detect packaging regressions and inspect framework profile
    divergence without poking at ``dataclasses.fields()`` internals.

    The killer test in this class is
    ``test_v1_13_0_packaging_regression_would_be_caught`` — it asserts
    that *every* API entry point fires when the bundled thresholds.yaml
    is missing from the install. If that test ever stops passing, the
    API has regressed and bench's gate will silently miss the next
    packaging bug.
    """

    # ─── thresholds_yaml_path / is_yaml_loaded ───

    def test_thresholds_yaml_path_returns_absolute_path(self) -> None:
        path = VitalsConfig.thresholds_yaml_path()
        assert path.is_absolute()
        assert path.name == "thresholds.yaml"

    def test_is_yaml_loaded_true_for_bundled_yaml(self) -> None:
        assert VitalsConfig.is_yaml_loaded() is True

    def test_is_yaml_loaded_false_when_yaml_missing(self, tmp_path, monkeypatch) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", missing)
        assert VitalsConfig.is_yaml_loaded() is False

    def test_is_yaml_loaded_false_when_yaml_not_a_mapping(
        self, tmp_path, monkeypatch
    ) -> None:
        bad = tmp_path / "thresholds.yaml"
        bad.write_text("just a string", encoding="utf-8")
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", bad)
        assert VitalsConfig.is_yaml_loaded() is False

    # ─── assert_profiles_loaded ───

    def test_assert_profiles_loaded_passes_for_bundled_yaml(self) -> None:
        # Should not raise.
        VitalsConfig.assert_profiles_loaded()

    def test_assert_profiles_loaded_raises_when_yaml_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        from agent_vitals.exceptions import ConfigurationError

        missing = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", missing)
        with pytest.raises(ConfigurationError, match="missing or unreadable"):
            VitalsConfig.assert_profiles_loaded()

    def test_assert_profiles_loaded_raises_when_no_profiles_defined(
        self, tmp_path, monkeypatch
    ) -> None:
        from agent_vitals.exceptions import ConfigurationError

        bare = tmp_path / "thresholds.yaml"
        bare.write_text("loop_consecutive_pct: 0.5\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", bare)
        with pytest.raises(ConfigurationError, match="no framework profiles"):
            VitalsConfig.assert_profiles_loaded()

    # ─── list_profiles / profiles ───

    def test_list_profiles_returns_sorted_tuple_of_bundled_profiles(self) -> None:
        # The bundled thresholds.yaml ships with crewai, dspy, langgraph.
        assert VitalsConfig.list_profiles() == ("crewai", "dspy", "langgraph")

    def test_list_profiles_empty_when_yaml_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", missing)
        assert VitalsConfig.list_profiles() == ()

    def test_profiles_instance_method_matches_classmethod_for_default_load(
        self,
    ) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert cfg.profiles() == VitalsConfig.list_profiles()

    # ─── profile_diff ───

    def test_profile_diff_dspy_has_three_overrides(self) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        diff = cfg.profile_diff("dspy")
        assert set(diff.keys()) == {
            "loop_consecutive_pct",
            "stuck_dm_threshold",
            "workflow_stuck_enabled",
        }
        assert diff["loop_consecutive_pct"].default == 0.5
        assert diff["loop_consecutive_pct"].override == 0.7
        assert diff["stuck_dm_threshold"].default == 0.15
        assert diff["stuck_dm_threshold"].override == 0.1
        assert diff["workflow_stuck_enabled"].default == "research-only"
        assert diff["workflow_stuck_enabled"].override == "none"

    def test_profile_diff_crewai_excludes_loop_consecutive_pct_when_equal_to_default(
        self,
    ) -> None:
        # crewai's YAML lists loop_consecutive_pct: 0.5, which equals
        # the dataclass default. The diff must omit it because the
        # diff is a property of the YAML *vs the default*, not vs self.
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        diff = cfg.profile_diff("crewai")
        assert "loop_consecutive_pct" not in diff
        # The two real overrides are still present.
        assert set(diff.keys()) == {"burn_rate_multiplier", "token_scale_factor"}
        assert diff["burn_rate_multiplier"].default == 2.5
        assert diff["burn_rate_multiplier"].override == 3.0
        assert diff["token_scale_factor"].default == 1.0
        assert diff["token_scale_factor"].override == 0.7

    def test_profile_diff_returns_field_default_and_override(self) -> None:
        from agent_vitals.config import ProfileFieldDiff

        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        diff = cfg.profile_diff("langgraph")
        assert "loop_consecutive_pct" in diff
        entry = diff["loop_consecutive_pct"]
        assert isinstance(entry, ProfileFieldDiff)
        assert entry.field == "loop_consecutive_pct"
        assert entry.default == 0.5
        assert entry.override == 0.4

    def test_profile_diff_raises_unknown_profile_error_with_known_list_in_message(
        self,
    ) -> None:
        from agent_vitals.exceptions import ConfigurationError, UnknownProfileError

        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        with pytest.raises(UnknownProfileError) as exc_info:
            cfg.profile_diff("langchain")
        # Subclass of ConfigurationError so existing handlers catch it.
        assert isinstance(exc_info.value, ConfigurationError)
        msg = str(exc_info.value)
        assert "langchain" in msg
        # All three known profiles must appear in the recovery list.
        assert "crewai" in msg
        assert "dspy" in msg
        assert "langgraph" in msg

    def test_profile_diff_case_insensitive(self) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        # All three forms must resolve to the same diff.
        lower = cfg.profile_diff("dspy")
        upper = cfg.profile_diff("DSPY")
        mixed = cfg.profile_diff("DSpy")
        assert lower == upper == mixed

    def test_profile_diff_anchor_is_pure_defaults_not_self(
        self, tmp_path, monkeypatch
    ) -> None:
        # If profile_diff compared against `self`, then loading a YAML
        # with loop_consecutive_pct=0.99 at the top level would make
        # the dspy profile (loop_consecutive_pct=0.7) look like a
        # *down*-override. We want it to still report dspy's 0.7
        # against the dataclass default 0.5 — the diff is a property
        # of the YAML profile, not of the loaded config.
        custom = tmp_path / "thresholds.yaml"
        custom.write_text(
            "\n".join(
                [
                    "loop_consecutive_pct: 0.99",
                    "profiles:",
                    "  dspy:",
                    "    loop_consecutive_pct: 0.7",
                ]
            ),
            encoding="utf-8",
        )
        cfg = VitalsConfig.from_yaml(yaml_path=custom, allow_env_override=False)
        # Sanity check the load worked.
        assert cfg.loop_consecutive_pct == 0.99
        diff = cfg.profile_diff("dspy")
        # Anchor is pure default (0.5), not self (0.99).
        assert diff["loop_consecutive_pct"].default == 0.5
        assert diff["loop_consecutive_pct"].override == 0.7

    def test_profile_diff_empty_when_profile_only_lists_default_values(
        self, tmp_path, monkeypatch
    ) -> None:
        # Synthetic profile that redundantly sets a field to the default.
        custom = tmp_path / "thresholds.yaml"
        custom.write_text(
            "\n".join(
                [
                    "profiles:",
                    "  passthrough:",
                    "    loop_consecutive_pct: 0.5",  # equals default
                ]
            ),
            encoding="utf-8",
        )
        cfg = VitalsConfig.from_yaml(yaml_path=custom, allow_env_override=False)
        assert cfg.profiles() == ("passthrough",)
        diff = cfg.profile_diff("passthrough")
        assert diff == {}

    # ─── The killer test: would the API have caught the v1.13.0 bug? ───

    def test_v1_13_0_packaging_regression_would_be_caught(
        self, tmp_path, monkeypatch
    ) -> None:
        """Asserts every API entry point signals failure when thresholds.yaml
        is missing from the install — exactly the v1.13.0 packaging bug.

        If this test ever stops passing, the introspection API has
        regressed and bench's external gate will silently miss the next
        packaging regression. This test is load-bearing for the
        contract with all external verifiers; do not relax it.
        """
        from agent_vitals.exceptions import ConfigurationError, UnknownProfileError

        # Simulate a wheel built without thresholds.yaml.
        missing = tmp_path / "site-packages" / "agent_vitals" / "thresholds.yaml"
        monkeypatch.setattr(config_module, "THRESHOLDS_YAML_PATH", missing)

        # 1. Boolean check fails.
        assert VitalsConfig.is_yaml_loaded() is False

        # 2. List is empty.
        assert VitalsConfig.list_profiles() == ()

        # 3. Hard gate raises.
        with pytest.raises(ConfigurationError):
            VitalsConfig.assert_profiles_loaded()

        # 4. Loaded config has no profiles attached.
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert cfg.profiles() == ()

        # 5. profile_diff for any framework name raises UnknownProfileError
        #    instead of silently returning {} (which would have been the
        #    quietly-broken pre-API behavior).
        for fw in ("dspy", "crewai", "langgraph"):
            with pytest.raises(UnknownProfileError):
                cfg.profile_diff(fw)


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
