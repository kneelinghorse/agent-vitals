"""Adaptive SPC thresholding with WMA control limits and cooldown suppression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import pstdev
from typing import Literal, Optional, Sequence


AdaptiveDirection = Literal["increase", "decrease"]


@dataclass(frozen=True, slots=True)
class AdaptiveThresholdUpdate:
    """Single update result for AdaptiveThreshold."""

    value: float
    wma: float
    sigma: float
    threshold: float
    alarm: bool
    warmup_complete: bool
    suppressed_by_cooldown: bool


class AdaptiveThreshold:
    """Stateful adaptive threshold tracker using WMA +/- k*sigma limits."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        k_sigma: float = 3.0,
        warmup_steps: int = 2,
        cooldown_steps: int = 1,
        wma_decay: float = 0.7,
        min_sigma: float = 1e-6,
        direction: AdaptiveDirection = "increase",
    ) -> None:
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        if not math.isfinite(k_sigma) or k_sigma <= 0.0:
            raise ValueError("k_sigma must be finite and > 0")
        if not math.isfinite(wma_decay) or not (0.0 < wma_decay <= 1.0):
            raise ValueError("wma_decay must be finite and in (0, 1]")
        if not math.isfinite(min_sigma) or min_sigma <= 0.0:
            raise ValueError("min_sigma must be finite and > 0")

        self._direction = direction
        self._window_size = int(window_size)
        self._k_sigma = float(k_sigma)
        self._warmup_steps = int(warmup_steps)
        self._cooldown_steps = int(cooldown_steps)
        self._wma_decay = float(wma_decay)
        self._min_sigma = float(min_sigma)

        self._values: list[float] = []
        self._cooldown_remaining = 0

    def reset(self) -> None:
        """Reset internal rolling state."""
        self._values = []
        self._cooldown_remaining = 0

    def update(
        self,
        value: float,
        *,
        fallback_threshold: Optional[float] = None,
    ) -> AdaptiveThresholdUpdate:
        """Update with a new value and return thresholding result."""
        x = float(value)
        if not math.isfinite(x):
            x = 0.0
        self._values.append(x)

        warmup_complete = len(self._values) > self._warmup_steps
        history = self._values[-self._window_size :]
        baseline_values = history[:-1] if len(history) > 1 else history

        if warmup_complete and baseline_values:
            wma = _weighted_moving_average(baseline_values, decay=self._wma_decay)
            sigma = max(self._min_sigma, _safe_pstdev(baseline_values))
            if self._direction == "increase":
                threshold = float(wma + self._k_sigma * sigma)
                raw_alarm = x > threshold
            else:
                threshold = float(wma - self._k_sigma * sigma)
                raw_alarm = x < threshold
        else:
            wma = _weighted_moving_average(history, decay=self._wma_decay)
            sigma = max(self._min_sigma, _safe_pstdev(history))
            if fallback_threshold is None:
                threshold = float(wma)
                raw_alarm = False
            else:
                threshold = float(fallback_threshold)
                if self._direction == "increase":
                    raw_alarm = x > threshold
                else:
                    raw_alarm = x < threshold

        suppressed = False
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if raw_alarm:
                raw_alarm = False
                suppressed = True

        if raw_alarm:
            self._cooldown_remaining = self._cooldown_steps

        return AdaptiveThresholdUpdate(
            value=x,
            wma=float(wma),
            sigma=float(sigma),
            threshold=float(threshold),
            alarm=bool(raw_alarm),
            warmup_complete=bool(warmup_complete),
            suppressed_by_cooldown=bool(suppressed),
        )


def _weighted_moving_average(values: Sequence[float], *, decay: float) -> float:
    if not values:
        return 0.0
    weights: list[float] = []
    n = len(values)
    for idx in range(n):
        exponent = n - idx - 1
        weights.append(float(decay**exponent))
    denominator = sum(weights)
    if denominator <= 0.0:
        return float(values[-1])
    numerator = sum(weight * float(value) for weight, value in zip(weights, values))
    return float(numerator / denominator)


def _safe_pstdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


__all__ = [
    "AdaptiveDirection",
    "AdaptiveThreshold",
    "AdaptiveThresholdUpdate",
]
