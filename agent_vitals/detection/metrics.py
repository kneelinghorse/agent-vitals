"""Temporal metrics engine for Agent Vitals.

Implements Protocol Suite-inspired temporal diagnostics:
- Coefficient of Variation (CV)
- Directional Momentum (DM)
- Temporal Hysteresis (TH)
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable, List, Sequence, Tuple

from ..schema import HealthState, HysteresisConfig


class TemporalMetrics:
    """Compute temporal metrics from signal histories."""

    def coefficient_of_variation(self, values: Sequence[float], window: int = 5) -> float:
        """Return CV = σ/μ over a trailing window.

        Args:
            values: Signal time series.
            window: Trailing window size.

        Returns:
            Non-negative coefficient of variation. Returns 0.0 for insufficient
            data or zero mean.
        """

        if window <= 0:
            raise ValueError("window must be greater than zero")

        series = _coerce_floats(values)
        if len(series) < 2:
            return 0.0

        window_values = series[-window:]
        mu = mean(window_values)
        if abs(mu) < 1e-12:
            return 0.0
        sigma = pstdev(window_values)
        cv = float(sigma) / float(abs(mu))
        return max(0.0, cv)

    def directional_momentum(
        self,
        values: Sequence[float],
        *,
        alpha: float = 0.3,
        lookback: int = 3,
    ) -> float:
        """Return normalized EMA momentum in [-1, 1].

        DM is computed from the EMA delta over the lookback horizon and
        normalized by the mean magnitude of the EMA window to yield a bounded,
        scale-aware score.

        Args:
            values: Signal time series.
            alpha: EMA smoothing factor (0 < alpha <= 1).
            lookback: Number of steps to measure momentum over (>= 1).

        Returns:
            Momentum score in [-1, 1]. Returns 0.0 when insufficient data.
        """

        if lookback <= 0:
            raise ValueError("lookback must be greater than zero")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        series = _coerce_floats(values)
        if len(series) < lookback + 1:
            return 0.0

        ema = self._compute_ema(series, alpha=alpha)
        end = ema[-1]
        start = ema[-1 - lookback]
        delta = end - start
        window = ema[-(lookback + 1) :]
        scale = abs(mean(window))
        if scale < 1e-9:
            return 0.0
        score = delta / scale
        return float(_clip(score, -1.0, 1.0))

    def temporal_hysteresis(
        self,
        value: float,
        current_state: HealthState,
        thresholds: HysteresisConfig,
    ) -> Tuple[HealthState, bool]:
        """State machine with asymmetric enter/exit thresholds.

        Args:
            value: Current signal value (e.g., coverage score).
            current_state: Current health state.
            thresholds: Hysteresis threshold configuration.

        Returns:
            Tuple of (new_state, changed_flag).
        """

        if not math.isfinite(value):
            return current_state, False

        if current_state == "healthy":
            if value <= thresholds.enter_warning:
                return "warning", True
            return "healthy", False

        if current_state == "warning":
            if value >= thresholds.exit_warning:
                return "healthy", True
            if value <= thresholds.enter_critical:
                return "critical", True
            return "warning", False

        if current_state == "critical":
            if value >= thresholds.exit_critical:
                return "warning", True
            return "critical", False

        return "healthy", True

    @staticmethod
    def _compute_ema(values: Sequence[float], *, alpha: float) -> List[float]:
        series = list(values)
        if not series:
            return []

        ema: List[float] = [float(series[0])]
        for value in series[1:]:
            ema.append(alpha * float(value) + (1.0 - alpha) * ema[-1])
        return ema


def _coerce_floats(values: Iterable[float]) -> List[float]:
    series: List[float] = []
    for item in values:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            series.append(value)
    return series


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["HysteresisConfig", "TemporalMetrics"]

# Re-export for backwards compatibility
HysteresisConfig = HysteresisConfig
