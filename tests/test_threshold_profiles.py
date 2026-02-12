"""Unit tests for framework-specific threshold profiles.

Tests cover: ThresholdProfile dataclass, profile loading from YAML
and dict, VitalsConfig.for_framework() selection/fallback/override,
auto-detection from adapter type, and explicit framework parameter.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_vitals.adapters import CrewAIAdapter, DSPyAdapter, LangGraphAdapter
from agent_vitals.config import (
    ADAPTER_FRAMEWORK_MAP,
    ThresholdProfile,
    VitalsConfig,
)
from agent_vitals.monitor import AgentVitals


# ---------------------------------------------------------------------------
# ThresholdProfile dataclass
# ---------------------------------------------------------------------------


class TestThresholdProfile:
    """Tests for ThresholdProfile creation and from_dict."""

    def test_defaults_to_none(self) -> None:
        """All override fields default to None."""
        profile = ThresholdProfile(framework="test")
        assert profile.framework == "test"
        assert profile.loop_consecutive_count is None
        assert profile.burn_rate_multiplier is None
        assert profile.token_scale_factor is None
        assert profile.workflow_stuck_enabled is None

    def test_from_dict_parses_overrides(self) -> None:
        """from_dict picks up known keys and ignores unknowns."""
        data = {
            "loop_consecutive_count": 8,
            "burn_rate_multiplier": 4.0,
            "token_scale_factor": 0.7,
            "unknown_key": "ignored",
        }
        profile = ThresholdProfile.from_dict("crewai", data)
        assert profile.framework == "crewai"
        assert profile.loop_consecutive_count == 8
        assert profile.burn_rate_multiplier == 4.0
        assert profile.token_scale_factor == 0.7
        assert profile.loop_similarity_threshold is None  # not in data

    def test_from_dict_coerces_types(self) -> None:
        """from_dict coerces string values to proper types."""
        data = {"loop_consecutive_count": "10", "burn_rate_multiplier": "3.5"}
        profile = ThresholdProfile.from_dict("dspy", data)
        assert profile.loop_consecutive_count == 10
        assert profile.burn_rate_multiplier == 3.5

    def test_from_dict_ignores_invalid(self) -> None:
        """from_dict silently ignores invalid values."""
        data = {"loop_consecutive_count": "not_a_number"}
        profile = ThresholdProfile.from_dict("test", data)
        assert profile.loop_consecutive_count is None

    def test_from_dict_normalizes_framework_name(self) -> None:
        """Framework name is lowercased and stripped."""
        profile = ThresholdProfile.from_dict("  LangGraph  ", {})
        assert profile.framework == "langgraph"

    def test_frozen(self) -> None:
        """ThresholdProfile is frozen (immutable)."""
        profile = ThresholdProfile(framework="test")
        with pytest.raises(AttributeError):
            profile.framework = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VitalsConfig.for_framework()
# ---------------------------------------------------------------------------


class TestForFramework:
    """Tests for VitalsConfig.for_framework() selection and override."""

    def test_no_profiles_returns_self(self) -> None:
        """Config without profiles returns self for any framework."""
        config = VitalsConfig()
        result = config.for_framework("langgraph")
        assert result is config

    def test_unknown_framework_returns_self(self) -> None:
        """Unknown framework name returns self (default fallback)."""
        profile = ThresholdProfile(framework="crewai", loop_consecutive_count=8)
        config = VitalsConfig(framework_profiles=(profile,))
        result = config.for_framework("unknown")
        assert result is config

    def test_matching_profile_applies_overrides(self) -> None:
        """Matching profile overrides only the specified fields."""
        profile = ThresholdProfile(
            framework="crewai",
            loop_consecutive_count=8,
            burn_rate_multiplier=4.0,
            token_scale_factor=0.7,
        )
        config = VitalsConfig(framework_profiles=(profile,))
        result = config.for_framework("crewai")

        # Overridden fields
        assert result.loop_consecutive_count == 8
        assert result.burn_rate_multiplier == 4.0
        assert result.token_scale_factor == 0.7

        # Non-overridden fields preserved
        assert result.loop_similarity_threshold == config.loop_similarity_threshold
        assert result.stuck_dm_threshold == config.stuck_dm_threshold
        assert result.stuck_cv_threshold == config.stuck_cv_threshold
        assert result.workflow_stuck_enabled == config.workflow_stuck_enabled

    def test_case_insensitive_lookup(self) -> None:
        """Framework lookup is case-insensitive."""
        profile = ThresholdProfile(framework="langgraph", loop_consecutive_count=5)
        config = VitalsConfig(framework_profiles=(profile,))
        result = config.for_framework("LangGraph")
        assert result.loop_consecutive_count == 5

    def test_preserves_non_threshold_config(self) -> None:
        """for_framework preserves non-threshold settings (enabled, paths, etc.)."""
        profile = ThresholdProfile(framework="dspy", loop_consecutive_count=10)
        config = VitalsConfig(
            enabled=False,
            enforcement=True,
            jsonl_layout="per_run",
            framework_profiles=(profile,),
        )
        result = config.for_framework("dspy")
        assert result.enabled is False
        assert result.enforcement is True
        assert result.jsonl_layout == "per_run"
        assert result.loop_consecutive_count == 10

    def test_preserves_framework_profiles_on_result(self) -> None:
        """The resulting config still carries framework_profiles."""
        profile = ThresholdProfile(framework="crewai", loop_consecutive_count=8)
        config = VitalsConfig(framework_profiles=(profile,))
        result = config.for_framework("crewai")
        assert len(result.framework_profiles) == 1

    def test_multiple_profiles_selects_correct(self) -> None:
        """Multiple profiles: correct one is selected."""
        profiles = (
            ThresholdProfile(framework="langgraph", loop_consecutive_count=5),
            ThresholdProfile(framework="crewai", loop_consecutive_count=8),
            ThresholdProfile(framework="dspy", loop_consecutive_count=10),
        )
        config = VitalsConfig(framework_profiles=profiles)

        assert config.for_framework("langgraph").loop_consecutive_count == 5
        assert config.for_framework("crewai").loop_consecutive_count == 8
        assert config.for_framework("dspy").loop_consecutive_count == 10

    def test_empty_profile_returns_self(self) -> None:
        """Profile with no overrides (all None) returns self."""
        profile = ThresholdProfile(framework="test")
        config = VitalsConfig(framework_profiles=(profile,))
        result = config.for_framework("test")
        assert result is config


# ---------------------------------------------------------------------------
# Profile loading from YAML
# ---------------------------------------------------------------------------


class TestYAMLProfileLoading:
    """Tests for loading framework profiles from YAML files."""

    def test_from_yaml_loads_profiles(self, tmp_path: Path) -> None:
        """from_yaml parses the profiles section."""
        yaml_content = textwrap.dedent("""\
            loop_consecutive_count: 6
            profiles:
              langgraph:
                loop_consecutive_count: 5
              crewai:
                loop_consecutive_count: 8
                token_scale_factor: 0.7
        """)
        yaml_file = tmp_path / "thresholds.yaml"
        yaml_file.write_text(yaml_content)

        config = VitalsConfig.from_yaml(yaml_file, allow_env_override=False)
        assert len(config.framework_profiles) == 2

        # Default
        assert config.loop_consecutive_count == 6

        # LangGraph override
        lg = config.for_framework("langgraph")
        assert lg.loop_consecutive_count == 5

        # CrewAI override
        cr = config.for_framework("crewai")
        assert cr.loop_consecutive_count == 8
        assert cr.token_scale_factor == 0.7

    def test_from_yaml_no_profiles_section(self, tmp_path: Path) -> None:
        """YAML without profiles section produces empty profiles."""
        yaml_content = "loop_consecutive_count: 6\n"
        yaml_file = tmp_path / "thresholds.yaml"
        yaml_file.write_text(yaml_content)

        config = VitalsConfig.from_yaml(yaml_file, allow_env_override=False)
        assert config.framework_profiles == ()

    def test_from_dict_loads_profiles(self) -> None:
        """from_dict parses profiles from a dictionary."""
        data = {
            "loop_consecutive_count": 6,
            "profiles": {
                "dspy": {
                    "loop_consecutive_count": 10,
                    "workflow_stuck_enabled": "none",
                },
            },
        }
        config = VitalsConfig.from_dict(data)
        assert len(config.framework_profiles) == 1

        dspy = config.for_framework("dspy")
        assert dspy.loop_consecutive_count == 10
        assert dspy.workflow_stuck_enabled == "none"

    def test_bundled_thresholds_yaml_has_profiles(self) -> None:
        """The bundled thresholds.yaml includes framework profiles."""
        config = VitalsConfig.from_yaml(allow_env_override=False)
        assert len(config.framework_profiles) >= 3

        # Verify the 3 shipped profiles exist
        frameworks = {p.framework for p in config.framework_profiles}
        assert "langgraph" in frameworks
        assert "crewai" in frameworks
        assert "dspy" in frameworks


# ---------------------------------------------------------------------------
# Auto-detection from adapter type
# ---------------------------------------------------------------------------


class TestAdapterAutoDetection:
    """Tests for auto-detecting framework from adapter class."""

    def test_adapter_framework_map_covers_all_adapters(self) -> None:
        """ADAPTER_FRAMEWORK_MAP covers all shipped adapters."""
        expected = {
            "LangChainAdapter", "LangGraphAdapter", "CrewAIAdapter",
            "AutoGenAdapter", "DSPyAdapter", "HaystackAdapter",
            "LangfuseAdapter", "LangSmithAdapter", "TelemetryAdapter",
        }
        assert set(ADAPTER_FRAMEWORK_MAP.keys()) == expected

    def test_auto_detect_langgraph(self) -> None:
        """LangGraphAdapter auto-resolves to 'langgraph' profile."""
        profiles = (ThresholdProfile(framework="langgraph", loop_consecutive_count=5),)
        config = VitalsConfig(framework_profiles=profiles)
        monitor = AgentVitals(
            mission_id="test",
            config=config,
            adapter=LangGraphAdapter(),
        )
        assert monitor._framework == "langgraph"
        assert monitor.config.loop_consecutive_count == 5

    def test_auto_detect_crewai(self) -> None:
        """CrewAIAdapter auto-resolves to 'crewai' profile."""
        profiles = (ThresholdProfile(framework="crewai", burn_rate_multiplier=4.0),)
        config = VitalsConfig(framework_profiles=profiles)
        monitor = AgentVitals(
            mission_id="test",
            config=config,
            adapter=CrewAIAdapter(),
        )
        assert monitor._framework == "crewai"
        assert monitor.config.burn_rate_multiplier == 4.0

    def test_auto_detect_dspy(self) -> None:
        """DSPyAdapter auto-resolves to 'dspy' profile."""
        profiles = (ThresholdProfile(framework="dspy", workflow_stuck_enabled="none"),)
        config = VitalsConfig(framework_profiles=profiles)
        monitor = AgentVitals(
            mission_id="test",
            config=config,
            adapter=DSPyAdapter(),
        )
        assert monitor._framework == "dspy"
        assert monitor.config.workflow_stuck_enabled == "none"

    def test_explicit_framework_overrides_adapter(self) -> None:
        """Explicit framework= parameter overrides adapter auto-detection."""
        profiles = (
            ThresholdProfile(framework="langgraph", loop_consecutive_count=5),
            ThresholdProfile(framework="crewai", loop_consecutive_count=8),
        )
        config = VitalsConfig(framework_profiles=profiles)
        # Adapter is LangGraph but explicit framework is crewai
        monitor = AgentVitals(
            mission_id="test",
            config=config,
            adapter=LangGraphAdapter(),
            framework="crewai",
        )
        assert monitor._framework == "crewai"
        assert monitor.config.loop_consecutive_count == 8

    def test_no_adapter_no_framework(self) -> None:
        """No adapter and no framework uses default config."""
        config = VitalsConfig()
        monitor = AgentVitals(mission_id="test", config=config)
        assert monitor._framework is None
        assert monitor.config is config

    def test_adapter_without_profile_uses_defaults(self) -> None:
        """Adapter with no matching profile uses default thresholds."""
        config = VitalsConfig()  # no profiles
        monitor = AgentVitals(
            mission_id="test",
            config=config,
            adapter=LangGraphAdapter(),
        )
        assert monitor._framework == "langgraph"
        assert monitor.config.loop_consecutive_count == config.loop_consecutive_count
