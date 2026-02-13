"""CUSUM change-point tracking for early anomaly detection.

The tracker supports default SPC-style parameterization:
- ``k = 0.5 * sigma``
- ``H = 4.0 * sigma``

where ``sigma`` is estimated from a short warmup baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Literal, Optional


CUSUMDirection = Literal["increase", "decrease"]


@dataclass(frozen=True, slots=True)
class CUSUMUpdate:
    """Single update result from a CUSUM tracker."""

    value: float
    score: float
    target: float
    k: float
    h: float
    alarm: bool
    warmup_complete: bool


class CUSUMTracker:
    """Stateful one-sided CUSUM tracker.

    Formula:
        C_i = max(0, C_{i-1} + (x_i - target) - k)

    For ``direction="decrease"``, drift is inverted to:
        C_i = max(0, C_{i-1} + (target - x_i) - k)

    Args:
        target: Optional fixed target baseline. If ``None``, estimated from
            the warmup window mean.
        k: Optional fixed drift sensitivity. If ``None``, derived as
            ``k_sigma * sigma`` from warmup.
        h: Optional fixed alarm threshold. If ``None``, derived as
            ``h_sigma * sigma`` from warmup.
        warmup_steps: Number of observations used to initialize baseline.
        k_sigma: Multiplier for auto-derived ``k``.
        h_sigma: Multiplier for auto-derived ``h``.
        min_sigma: Floor for sigma to prevent degenerate zero-threshold setups.
        direction: Detect increasing or decreasing drift.
    """

    def __init__(
        self,
        *,
        target: Optional[float] = None,
        k: Optional[float] = None,
        h: Optional[float] = None,
        warmup_steps: int = 2,
        k_sigma: float = 0.5,
        h_sigma: float = 4.0,
        min_sigma: float = 1e-6,
        direction: CUSUMDirection = "increase",
    ) -> None:
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if not math.isfinite(k_sigma) or k_sigma <= 0.0:
            raise ValueError("k_sigma must be finite and > 0")
        if not math.isfinite(h_sigma) or h_sigma <= 0.0:
            raise ValueError("h_sigma must be finite and > 0")
        if not math.isfinite(min_sigma) or min_sigma <= 0.0:
            raise ValueError("min_sigma must be finite and > 0")

        self._direction = direction
        self._warmup_steps = int(warmup_steps)
        self._k_sigma = float(k_sigma)
        self._h_sigma = float(h_sigma)
        self._min_sigma = float(min_sigma)

        self._target_explicit = float(target) if target is not None else None
        self._k_explicit = float(k) if k is not None else None
        self._h_explicit = float(h) if h is not None else None

        self._target = self._target_explicit
        self._k = self._k_explicit
        self._h = self._h_explicit

        self._warmup_values: list[float] = []
        self._score = 0.0
        self._warmup_complete = False

    @property
    def score(self) -> float:
        """Current CUSUM score."""
        return float(self._score)

    @property
    def warmup_complete(self) -> bool:
        """Whether baseline initialization has completed."""
        return bool(self._warmup_complete)

    def reset(self) -> None:
        """Reset runtime state while preserving configuration."""
        self._target = self._target_explicit
        self._k = self._k_explicit
        self._h = self._h_explicit
        self._warmup_values = []
        self._score = 0.0
        self._warmup_complete = False

    def update(self, value: float) -> CUSUMUpdate:
        """Update tracker with a new observation."""
        x = float(value)
        if not math.isfinite(x):
            x = 0.0

        if not self._warmup_complete:
            self._warmup_values.append(x)
            if len(self._warmup_values) >= self._warmup_steps:
                self._initialize_from_warmup()
                self._warmup_complete = True
            return CUSUMUpdate(
                value=x,
                score=float(self._score),
                target=float(self._target or 0.0),
                k=float(self._k or 0.0),
                h=float(self._h or 0.0),
                alarm=False,
                warmup_complete=self._warmup_complete,
            )

        target = float(self._target or 0.0)
        k = float(self._k or 0.0)
        h = float(self._h or 0.0)

        if self._direction == "increase":
            drift = (x - target) - k
        else:
            drift = (target - x) - k

        self._score = max(0.0, self._score + drift)
        alarm = bool(self._score > h)
        return CUSUMUpdate(
            value=x,
            score=float(self._score),
            target=target,
            k=k,
            h=h,
            alarm=alarm,
            warmup_complete=True,
        )

    def _initialize_from_warmup(self) -> None:
        """Initialize baseline parameters from warmup data if unset."""
        baseline = self._warmup_values[-self._warmup_steps :]
        if self._target is None:
            self._target = float(mean(baseline))
        sigma = float(pstdev(baseline)) if len(baseline) > 1 else 0.0
        sigma = max(self._min_sigma, sigma)
        if self._k is None:
            self._k = float(self._k_sigma * sigma)
        if self._h is None:
            self._h = float(self._h_sigma * sigma)


__all__ = ["CUSUMDirection", "CUSUMTracker", "CUSUMUpdate"]
