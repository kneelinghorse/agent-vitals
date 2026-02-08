"""Configuration helpers for the Agent Vitals subsystem.

This module centralizes configuration for vitals detection thresholds,
export settings, and runtime behavior. Configuration can come from:
1. Constructor kwargs (highest priority)
2. YAML file (file-based)
3. Environment variables (deployment)
4. Built-in defaults (lowest priority)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .schema import HysteresisConfig

logger = logging.getLogger(__name__)

THRESHOLDS_YAML_PATH = Path(__file__).parent / "thresholds.yaml"

VITALS_ENABLED_ENV = "VITALS_ENABLED"
VITALS_ENFORCEMENT_ENV = "VITALS_ENFORCEMENT"
VITALS_SHADOW_LOG_ENV = "VITALS_SHADOW_LOG"

VITALS_JSONL_DIR_ENV = "VITALS_JSONL_DIR"
VITALS_JSONL_MAX_BYTES_ENV = "VITALS_JSONL_MAX_BYTES"
VITALS_HISTORY_SIZE_ENV = "VITALS_HISTORY_SIZE"
VITALS_JSONL_LAYOUT_ENV = "VITALS_JSONL_LAYOUT"

VITALS_LOOP_SIMILARITY_THRESHOLD_ENV = "VITALS_LOOP_SIMILARITY_THRESHOLD"
VITALS_LOOP_CONSECUTIVE_COUNT_ENV = "VITALS_LOOP_CONSECUTIVE_COUNT"
VITALS_STUCK_DM_THRESHOLD_ENV = "VITALS_STUCK_DM_THRESHOLD"
VITALS_STUCK_CV_THRESHOLD_ENV = "VITALS_STUCK_CV_THRESHOLD"
VITALS_BURN_RATE_MULTIPLIER_ENV = "VITALS_BURN_RATE_MULTIPLIER"
VITALS_TOKEN_SCALE_FACTOR_ENV = "VITALS_TOKEN_SCALE_FACTOR"
WORKFLOW_STUCK_ENABLED_ENV = "WORKFLOW_STUCK_ENABLED"

VITALS_TH_ENTER_WARNING_ENV = "VITALS_TH_ENTER_WARNING"
VITALS_TH_EXIT_WARNING_ENV = "VITALS_TH_EXIT_WARNING"
VITALS_TH_ENTER_CRITICAL_ENV = "VITALS_TH_ENTER_CRITICAL"
VITALS_TH_EXIT_CRITICAL_ENV = "VITALS_TH_EXIT_CRITICAL"

VITALS_EXPORT_OTLP_ENV = "VITALS_EXPORT_OTLP"
VITALS_OTLP_ENDPOINT_ENV = "VITALS_OTLP_ENDPOINT"
VITALS_EXPORT_LANGFUSE_ENV = "VITALS_EXPORT_LANGFUSE"
VITALS_EXPORT_LANGSMITH_ENV = "VITALS_EXPORT_LANGSMITH"

DEFAULT_VITALS_JSONL_DIR = Path("checkpoints/vitals")
DEFAULT_VITALS_JSONL_MAX_BYTES = 10_000_000  # 10MB
DEFAULT_VITALS_HISTORY_SIZE = 20
DEFAULT_VITALS_JSONL_LAYOUT = "append"  # "append" | "per_run"

DEFAULT_LOOP_SIMILARITY_THRESHOLD = 0.8
DEFAULT_LOOP_CONSECUTIVE_COUNT = 6
DEFAULT_STUCK_DM_THRESHOLD = 0.15
DEFAULT_STUCK_CV_THRESHOLD = 0.5
DEFAULT_BURN_RATE_MULTIPLIER = 3.0
DEFAULT_TOKEN_SCALE_FACTOR = 1.0
DEFAULT_WORKFLOW_STUCK_ENABLED = "research-only"

DEFAULT_TH_ENTER_WARNING = 0.4
DEFAULT_TH_EXIT_WARNING = 0.6
DEFAULT_TH_ENTER_CRITICAL = 0.2
DEFAULT_TH_EXIT_CRITICAL = 0.35


@dataclass(frozen=True, slots=True)
class VitalsConfig:
    """Vitals module configuration for detection thresholds and behavior."""

    enabled: bool = True
    enforcement: bool = False
    shadow_log: bool = True

    jsonl_dir: Path = DEFAULT_VITALS_JSONL_DIR
    jsonl_max_bytes: int = DEFAULT_VITALS_JSONL_MAX_BYTES
    history_size: int = DEFAULT_VITALS_HISTORY_SIZE
    jsonl_layout: str = DEFAULT_VITALS_JSONL_LAYOUT

    loop_similarity_threshold: float = DEFAULT_LOOP_SIMILARITY_THRESHOLD
    loop_consecutive_count: int = DEFAULT_LOOP_CONSECUTIVE_COUNT
    stuck_dm_threshold: float = DEFAULT_STUCK_DM_THRESHOLD
    stuck_cv_threshold: float = DEFAULT_STUCK_CV_THRESHOLD
    burn_rate_multiplier: float = DEFAULT_BURN_RATE_MULTIPLIER
    token_scale_factor: float = DEFAULT_TOKEN_SCALE_FACTOR
    workflow_stuck_enabled: str = DEFAULT_WORKFLOW_STUCK_ENABLED

    th_enter_warning: float = DEFAULT_TH_ENTER_WARNING
    th_exit_warning: float = DEFAULT_TH_EXIT_WARNING
    th_enter_critical: float = DEFAULT_TH_ENTER_CRITICAL
    th_exit_critical: float = DEFAULT_TH_EXIT_CRITICAL

    export_otlp: bool = False
    otlp_endpoint: str = "http://localhost:4318"
    export_langfuse: bool = False
    export_langsmith: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VitalsConfig":
        """Build a VitalsConfig from a plain dictionary.

        Unknown keys are silently ignored. Type coercion is applied
        where possible.

        Args:
            data: Dictionary of configuration values.

        Returns:
            VitalsConfig with values from the dictionary merged over defaults.
        """

        kwargs: dict[str, Any] = {}

        for key, coerce in (
            ("enabled", _to_bool),
            ("enforcement", _to_bool),
            ("shadow_log", _to_bool),
            ("export_otlp", _to_bool),
            ("export_langfuse", _to_bool),
            ("export_langsmith", _to_bool),
        ):
            if key in data:
                kwargs[key] = coerce(data[key])

        for key in (
            "loop_similarity_threshold",
            "stuck_dm_threshold",
            "stuck_cv_threshold",
            "burn_rate_multiplier",
            "token_scale_factor",
            "th_enter_warning",
            "th_exit_warning",
            "th_enter_critical",
            "th_exit_critical",
        ):
            if key in data:
                try:
                    value = float(data[key])
                    if math.isfinite(value):
                        kwargs[key] = value
                except (TypeError, ValueError):
                    pass

        for key in ("loop_consecutive_count", "jsonl_max_bytes", "history_size"):
            if key in data:
                try:
                    kwargs[key] = int(data[key])
                except (TypeError, ValueError):
                    pass

        for key in ("workflow_stuck_enabled", "jsonl_layout", "otlp_endpoint"):
            if key in data:
                text = str(data[key]).strip()
                if text:
                    kwargs[key] = text

        if "jsonl_dir" in data:
            kwargs["jsonl_dir"] = Path(str(data["jsonl_dir"]))

        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Optional[Path] = None,
        *,
        allow_env_override: bool = True,
    ) -> "VitalsConfig":
        """Build a VitalsConfig from a YAML thresholds file.

        Loads threshold values from the YAML file, then applies any
        environment-variable overrides on top (unless disabled).

        Args:
            yaml_path: Path to thresholds YAML. Defaults to the
                bundled ``thresholds.yaml`` next to this module.
            allow_env_override: When True, environment variables
                take precedence over YAML values.

        Returns:
            VitalsConfig with merged YAML + env configuration.
        """

        path = yaml_path or THRESHOLDS_YAML_PATH
        data: Mapping[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping):
                data = raw
            else:
                logger.warning("Thresholds YAML at %s is not a mapping; using defaults", path)
        else:
            logger.warning("Thresholds YAML not found at %s; using defaults", path)

        def _yaml_float(key: str, default: float) -> float:
            value = data.get(key)
            if value is None:
                return default
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else default
            except (TypeError, ValueError):
                return default

        def _yaml_int(key: str, default: int) -> int:
            value = data.get(key)
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _yaml_str(key: str, default: str) -> str:
            value = data.get(key)
            if value is None:
                return default
            text = str(value).strip()
            return text if text else default

        yaml_kwargs = dict(
            loop_similarity_threshold=_yaml_float(
                "loop_similarity_threshold", DEFAULT_LOOP_SIMILARITY_THRESHOLD
            ),
            loop_consecutive_count=_yaml_int(
                "loop_consecutive_count", DEFAULT_LOOP_CONSECUTIVE_COUNT
            ),
            stuck_dm_threshold=_yaml_float(
                "stuck_dm_threshold", DEFAULT_STUCK_DM_THRESHOLD
            ),
            stuck_cv_threshold=_yaml_float(
                "stuck_cv_threshold", DEFAULT_STUCK_CV_THRESHOLD
            ),
            burn_rate_multiplier=_yaml_float(
                "burn_rate_multiplier", DEFAULT_BURN_RATE_MULTIPLIER
            ),
            token_scale_factor=_yaml_float(
                "token_scale_factor", DEFAULT_TOKEN_SCALE_FACTOR
            ),
            workflow_stuck_enabled=_yaml_str(
                "workflow_stuck_enabled", DEFAULT_WORKFLOW_STUCK_ENABLED
            ),
            th_enter_warning=_yaml_float(
                "th_enter_warning", DEFAULT_TH_ENTER_WARNING
            ),
            th_exit_warning=_yaml_float(
                "th_exit_warning", DEFAULT_TH_EXIT_WARNING
            ),
            th_enter_critical=_yaml_float(
                "th_enter_critical", DEFAULT_TH_ENTER_CRITICAL
            ),
            th_exit_critical=_yaml_float(
                "th_exit_critical", DEFAULT_TH_EXIT_CRITICAL
            ),
        )

        if allow_env_override:
            env_instance = cls.from_env()
            for field_name, env_var in (
                ("loop_similarity_threshold", VITALS_LOOP_SIMILARITY_THRESHOLD_ENV),
                ("loop_consecutive_count", VITALS_LOOP_CONSECUTIVE_COUNT_ENV),
                ("stuck_dm_threshold", VITALS_STUCK_DM_THRESHOLD_ENV),
                ("stuck_cv_threshold", VITALS_STUCK_CV_THRESHOLD_ENV),
                ("burn_rate_multiplier", VITALS_BURN_RATE_MULTIPLIER_ENV),
                ("token_scale_factor", VITALS_TOKEN_SCALE_FACTOR_ENV),
                ("workflow_stuck_enabled", WORKFLOW_STUCK_ENABLED_ENV),
                ("th_enter_warning", VITALS_TH_ENTER_WARNING_ENV),
                ("th_exit_warning", VITALS_TH_EXIT_WARNING_ENV),
                ("th_enter_critical", VITALS_TH_ENTER_CRITICAL_ENV),
                ("th_exit_critical", VITALS_TH_EXIT_CRITICAL_ENV),
            ):
                if _normalize(os.getenv(env_var)) is not None:
                    yaml_kwargs[field_name] = getattr(env_instance, field_name)

            return cls(
                enabled=env_instance.enabled,
                enforcement=env_instance.enforcement,
                shadow_log=env_instance.shadow_log,
                jsonl_dir=env_instance.jsonl_dir,
                jsonl_max_bytes=env_instance.jsonl_max_bytes,
                history_size=env_instance.history_size,
                jsonl_layout=env_instance.jsonl_layout,
                export_otlp=env_instance.export_otlp,
                otlp_endpoint=env_instance.otlp_endpoint,
                export_langfuse=env_instance.export_langfuse,
                export_langsmith=env_instance.export_langsmith,
                **yaml_kwargs,
            )

        return cls(**yaml_kwargs)

    @classmethod
    def from_env(cls) -> "VitalsConfig":
        """Build a VitalsConfig instance from environment variables."""

        enabled = _parse_bool(VITALS_ENABLED_ENV, default=True)
        enforcement = _parse_bool(VITALS_ENFORCEMENT_ENV, default=True)
        shadow_log = _parse_bool(VITALS_SHADOW_LOG_ENV, default=True)

        jsonl_dir = _parse_path(VITALS_JSONL_DIR_ENV, default=DEFAULT_VITALS_JSONL_DIR)
        jsonl_max_bytes = _parse_int(
            VITALS_JSONL_MAX_BYTES_ENV,
            default=DEFAULT_VITALS_JSONL_MAX_BYTES,
            min_value=1,
        )
        history_size = _parse_int(
            VITALS_HISTORY_SIZE_ENV,
            default=DEFAULT_VITALS_HISTORY_SIZE,
            min_value=1,
        )
        jsonl_layout = _parse_jsonl_layout(
            VITALS_JSONL_LAYOUT_ENV,
            default=DEFAULT_VITALS_JSONL_LAYOUT,
        )

        loop_similarity_threshold = _parse_float(
            VITALS_LOOP_SIMILARITY_THRESHOLD_ENV,
            default=DEFAULT_LOOP_SIMILARITY_THRESHOLD,
            min_value=0.0,
            max_value=1.0,
        )
        loop_consecutive_count = _parse_int(
            VITALS_LOOP_CONSECUTIVE_COUNT_ENV,
            default=DEFAULT_LOOP_CONSECUTIVE_COUNT,
            min_value=1,
        )
        stuck_dm_threshold = _parse_float(
            VITALS_STUCK_DM_THRESHOLD_ENV,
            default=DEFAULT_STUCK_DM_THRESHOLD,
            min_value=0.0,
            max_value=1.0,
        )
        stuck_cv_threshold = _parse_float(
            VITALS_STUCK_CV_THRESHOLD_ENV,
            default=DEFAULT_STUCK_CV_THRESHOLD,
            min_value=0.0,
        )
        burn_rate_multiplier = _parse_float(
            VITALS_BURN_RATE_MULTIPLIER_ENV,
            default=DEFAULT_BURN_RATE_MULTIPLIER,
            min_value=1.0,
        )
        token_scale_factor = _parse_float(
            VITALS_TOKEN_SCALE_FACTOR_ENV,
            default=DEFAULT_TOKEN_SCALE_FACTOR,
            min_value=0.01,
        )
        workflow_stuck_enabled = _parse_workflow_stuck_enabled(
            WORKFLOW_STUCK_ENABLED_ENV,
            default=DEFAULT_WORKFLOW_STUCK_ENABLED,
        )

        th_enter_warning = _parse_float(
            VITALS_TH_ENTER_WARNING_ENV,
            default=DEFAULT_TH_ENTER_WARNING,
            min_value=0.0,
            max_value=1.0,
        )
        th_exit_warning = _parse_float(
            VITALS_TH_EXIT_WARNING_ENV,
            default=DEFAULT_TH_EXIT_WARNING,
            min_value=0.0,
            max_value=1.0,
        )
        th_enter_critical = _parse_float(
            VITALS_TH_ENTER_CRITICAL_ENV,
            default=DEFAULT_TH_ENTER_CRITICAL,
            min_value=0.0,
            max_value=1.0,
        )
        th_exit_critical = _parse_float(
            VITALS_TH_EXIT_CRITICAL_ENV,
            default=DEFAULT_TH_EXIT_CRITICAL,
            min_value=0.0,
            max_value=1.0,
        )

        export_otlp = _parse_bool(VITALS_EXPORT_OTLP_ENV, default=False)
        otlp_endpoint = str(os.getenv(VITALS_OTLP_ENDPOINT_ENV) or "http://localhost:4318").strip()
        if not otlp_endpoint:
            otlp_endpoint = "http://localhost:4318"
        export_langfuse = _parse_bool(VITALS_EXPORT_LANGFUSE_ENV, default=False)
        export_langsmith = _parse_bool(VITALS_EXPORT_LANGSMITH_ENV, default=False)

        return cls(
            enabled=enabled,
            enforcement=enforcement,
            shadow_log=shadow_log,
            jsonl_dir=jsonl_dir,
            jsonl_max_bytes=jsonl_max_bytes,
            history_size=history_size,
            jsonl_layout=jsonl_layout,
            loop_similarity_threshold=loop_similarity_threshold,
            loop_consecutive_count=loop_consecutive_count,
            stuck_dm_threshold=stuck_dm_threshold,
            stuck_cv_threshold=stuck_cv_threshold,
            burn_rate_multiplier=burn_rate_multiplier,
            token_scale_factor=token_scale_factor,
            workflow_stuck_enabled=workflow_stuck_enabled,
            th_enter_warning=th_enter_warning,
            th_exit_warning=th_exit_warning,
            th_enter_critical=th_enter_critical,
            th_exit_critical=th_exit_critical,
            export_otlp=export_otlp,
            otlp_endpoint=otlp_endpoint,
            export_langfuse=export_langfuse,
            export_langsmith=export_langsmith,
        )

    def hysteresis_config(self) -> HysteresisConfig:
        """Return a HysteresisConfig using the configured thresholds."""

        return HysteresisConfig(
            enter_warning=self.th_enter_warning,
            exit_warning=self.th_exit_warning,
            enter_critical=self.th_enter_critical,
            exit_critical=self.th_exit_critical,
        )


def get_vitals_config() -> VitalsConfig:
    """Return a VitalsConfig instance built from the current environment."""

    return VitalsConfig.from_env()


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = value.strip()
    return candidate or None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _parse_bool(env_name: str, *, default: bool) -> bool:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
    return default


def _parse_int(
    env_name: str,
    *,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
        return default

    if min_value is not None and parsed < min_value:
        logger.warning("Out-of-range %s=%r; clamping to %s", env_name, raw, min_value)
        parsed = min_value
    if max_value is not None and parsed > max_value:
        logger.warning("Out-of-range %s=%r; clamping to %s", env_name, raw, max_value)
        parsed = max_value
    return parsed


def _parse_workflow_stuck_enabled(env_name: str, *, default: str) -> str:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return default
    text = raw.strip().lower().replace("_", "-")
    mapping = {
        "research-only": "research-only",
        "research": "research-only",
        "default": "research-only",
        "build-only": "build-only",
        "build": "build-only",
        "all": "all",
        "both": "all",
        "enabled": "all",
        "true": "all",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "false": "none",
    }
    if text in mapping:
        return mapping[text]
    logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
    return default


def _parse_float(
    env_name: str,
    *,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return float(default)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
        return float(default)

    if not math.isfinite(parsed):
        logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
        return float(default)

    if min_value is not None and parsed < min_value:
        logger.warning("Out-of-range %s=%r; using default=%s", env_name, raw, default)
        return float(default)
    if max_value is not None and parsed > max_value:
        logger.warning("Out-of-range %s=%r; using default=%s", env_name, raw, default)
        return float(default)
    return float(parsed)


def _parse_path(env_name: str, *, default: Path) -> Path:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return default
    return Path(raw)


def _parse_jsonl_layout(env_name: str, *, default: str) -> str:
    raw = _normalize(os.getenv(env_name))
    if raw is None:
        return str(default)
    value = raw.lower().replace("-", "_")
    if value in {"append", "per_run"}:
        return value
    logger.warning("Invalid %s=%r; using default=%s", env_name, raw, default)
    return str(default)


__all__ = [
    "DEFAULT_VITALS_HISTORY_SIZE",
    "DEFAULT_VITALS_JSONL_DIR",
    "DEFAULT_VITALS_JSONL_MAX_BYTES",
    "DEFAULT_VITALS_JSONL_LAYOUT",
    "THRESHOLDS_YAML_PATH",
    "VITALS_JSONL_LAYOUT_ENV",
    "VitalsConfig",
    "get_vitals_config",
]
