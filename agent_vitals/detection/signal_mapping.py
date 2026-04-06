"""Signal mapping for model-size-aware detection normalization.

Small models (4B/9B) produce structurally flat token variance regardless of
max_tokens settings. This module classifies model size from token distribution
characteristics and provides threshold adjustments to prevent false positives
from signals that are structural properties of the model rather than detection
indicators.

Sprint-07 learnings:
  L23: 9B models produce flat token variance regardless of max_tokens.
  L22: stuck and runaway_cost detectors fight over same traces due to
       shared token growth signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ModelSizeClass = Literal["small", "medium", "large"]

# CV(completion_tokens) thresholds for auto-classification.
# Derived from observed distributions:
#   4B/9B models: CV < 0.15 (near-uniform completion lengths)
#   27B models: CV 0.15-0.30 (moderate variation)
#   70B+ models: CV > 0.30 (high variation, responsive to max_tokens)
SMALL_MODEL_CV_THRESHOLD = 0.15
MEDIUM_MODEL_CV_THRESHOLD = 0.30

# Minimum steps needed for reliable CV estimation.
MIN_STEPS_FOR_CLASSIFICATION = 4


@dataclass(frozen=True, slots=True)
class SignalMapping:
    """Model-size-aware signal adjustments for detection normalization."""

    model_size_class: ModelSizeClass

    # When True, the token_usage_variance_flat trigger is suppressed
    # because flat variance is a structural property of the model.
    suppress_token_variance_flat: bool

    # Multiplier applied on top of the configured burn_rate_multiplier.
    # Small models need a higher effective threshold because their flatter
    # token output produces a lower baseline, making normal context
    # accumulation look like a burn-rate spike.
    burn_rate_multiplier_scale: float


# Pre-computed mappings for each size class.
_MAPPINGS: dict[ModelSizeClass, SignalMapping] = {
    "small": SignalMapping(
        model_size_class="small",
        suppress_token_variance_flat=True,
        burn_rate_multiplier_scale=2.0,
    ),
    "medium": SignalMapping(
        model_size_class="medium",
        suppress_token_variance_flat=False,
        burn_rate_multiplier_scale=1.5,
    ),
    "large": SignalMapping(
        model_size_class="large",
        suppress_token_variance_flat=False,
        burn_rate_multiplier_scale=1.0,
    ),
}


def classify_model_size(
    completion_tokens: Sequence[float],
    *,
    explicit_class: str = "auto",
) -> ModelSizeClass:
    """Classify model size from completion token distribution.

    When *explicit_class* is ``"auto"`` (default), the classification is
    inferred from the coefficient of variation (CV) of *completion_tokens*.
    Small models produce near-uniform completion lengths (low CV), while
    large models show high variability.

    Args:
        completion_tokens: Per-step completion token counts.
        explicit_class: If not ``"auto"``, returns this class directly
            after validation.

    Returns:
        Model size classification: ``"small"``, ``"medium"``, or ``"large"``.
    """

    if explicit_class != "auto":
        normalized = str(explicit_class).strip().lower()
        if normalized in ("small", "medium", "large"):
            return normalized  # type: ignore[return-value]
        return "large"

    # Need sufficient data for reliable CV estimation.
    if len(completion_tokens) < MIN_STEPS_FOR_CLASSIFICATION:
        return "large"  # conservative: no special handling

    # Filter out zero values (completion_tokens not provided).
    nonzero = [float(v) for v in completion_tokens if float(v) > 0]
    if len(nonzero) < MIN_STEPS_FOR_CLASSIFICATION:
        return "large"

    cv = _coefficient_of_variation(nonzero)
    if cv < SMALL_MODEL_CV_THRESHOLD:
        return "small"
    if cv < MEDIUM_MODEL_CV_THRESHOLD:
        return "medium"
    return "large"


def get_signal_mapping(model_size_class: ModelSizeClass) -> SignalMapping:
    """Return signal adjustments for the given model size class.

    Args:
        model_size_class: One of ``"small"``, ``"medium"``, ``"large"``.

    Returns:
        SignalMapping with threshold adjustments.
    """

    return _MAPPINGS.get(model_size_class, _MAPPINGS["large"])


def _coefficient_of_variation(values: Sequence[float]) -> float:
    """Compute CV (σ/μ) of *values*."""

    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    if avg <= 0.0:
        return 0.0
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return float((variance**0.5) / avg)


__all__ = [
    "MEDIUM_MODEL_CV_THRESHOLD",
    "MIN_STEPS_FOR_CLASSIFICATION",
    "ModelSizeClass",
    "SMALL_MODEL_CV_THRESHOLD",
    "SignalMapping",
    "classify_model_size",
    "get_signal_mapping",
]
