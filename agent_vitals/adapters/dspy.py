"""DSPy adapter for converting program state into RawSignals.

Extracts token usage from DSPy's built-in tracking (``get_lm_usage()``,
``lm.history``), findings from prediction outputs, and coverage from
module completion state.

DSPy is an optional dependency — this module can be imported without
dspy installed.  The adapter operates on plain dict state payloads that
mirror the structures DSPy exposes at runtime.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..schema import RawSignals
from .base import BaseAdapter


class DSPyAdapter(BaseAdapter):
    """Extract Agent Vitals signals from DSPy program state dictionaries.

    Expected state keys (all optional, adapter degrades gracefully):

    - ``lm_usage``: dict from ``prediction.get_lm_usage()`` mapping
      model names to ``{completion_tokens, prompt_tokens, total_tokens}``.
    - ``history``: list from ``lm.history`` — each entry is a dict with
      keys like ``usage``, ``outputs``, ``cost``, ``timestamp``, etc.
    - ``predictions``: list of prediction output dicts or strings.
    - ``modules_completed``: int count of completed DSPy modules.
    - ``modules_total``: int total DSPy modules in the program.
    - ``errors``: list of error strings or dicts.
    - ``error_count``: int cumulative error count.
    - ``findings_count``: int explicit override for findings.
    - ``coverage_score``: float explicit override for coverage.
    - ``total_tokens``: int explicit override for total tokens.
    - ``query_count``: int explicit override for query count.
    - ``unique_domains``: int explicit override for domain count.
    """

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)

        # --- Token usage ---
        prompt_tokens, completion_tokens, total_tokens = self._extract_tokens(
            normalized
        )

        # --- Findings ---
        findings_count = self._extract_findings(normalized)

        # --- Coverage ---
        coverage_score = self._extract_coverage(normalized)

        # --- Errors ---
        error_count = self._safe_int(normalized.get("error_count", 0))
        if error_count == 0:
            error_count = self._safe_len(normalized.get("errors"), 0)

        # --- Query count (LM calls) ---
        query_count = self._safe_int(normalized.get("query_count", 0))
        if query_count == 0:
            history = normalized.get("history")
            if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
                query_count = len(history)

        return self.validate(
            RawSignals(
                findings_count=findings_count,
                sources_count=self._safe_int(normalized.get("sources_count", 0)),
                objectives_covered=self._safe_int(
                    normalized.get("objectives_covered", 0)
                ),
                coverage_score=coverage_score,
                confidence_score=self._safe_float(
                    normalized.get("confidence_score", 0.0)
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                api_calls=query_count,
                query_count=query_count,
                unique_domains=self._safe_int(
                    normalized.get("unique_domains", 0)
                ),
                refinement_count=self._safe_int(
                    normalized.get("refinement_count", 0)
                ),
                convergence_delta=self._safe_float(
                    normalized.get("convergence_delta", 0.0)
                ),
                error_count=error_count,
            )
        )

    # ------------------------------------------------------------------
    # Token extraction
    # ------------------------------------------------------------------

    def _extract_tokens(
        self, state: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        """Extract token usage, preferring explicit overrides then lm_usage then history."""

        # Explicit overrides
        explicit_total = state.get("total_tokens")
        if explicit_total is not None:
            prompt = self._safe_int(state.get("prompt_tokens", 0))
            completion = self._safe_int(state.get("completion_tokens", 0))
            total = self._safe_int(explicit_total)
            if total == 0 and (prompt + completion) > 0:
                total = prompt + completion
            return prompt, completion, total

        # Primary: get_lm_usage() dict — {model_name: {prompt_tokens, completion_tokens, total_tokens}}
        lm_usage = state.get("lm_usage")
        if isinstance(lm_usage, Mapping):
            return self._aggregate_lm_usage(lm_usage)

        # Fallback: lm.history list — each entry has a "usage" dict
        history = state.get("history")
        if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            return self._aggregate_history_usage(history)

        return 0, 0, 0

    def _aggregate_lm_usage(
        self, lm_usage: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage across all model keys in lm_usage dict."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for model_key, usage in lm_usage.items():
            if not isinstance(usage, Mapping):
                continue
            total_prompt += self._safe_int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
            )
            total_completion += self._safe_int(
                usage.get("completion_tokens", usage.get("output_tokens", 0))
            )
            total_tokens += self._safe_int(usage.get("total_tokens", 0))

        if total_tokens == 0 and (total_prompt + total_completion) > 0:
            total_tokens = total_prompt + total_completion

        return total_prompt, total_completion, total_tokens

    def _aggregate_history_usage(
        self, history: Sequence[Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage from lm.history entries."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for entry in history:
            if not isinstance(entry, Mapping):
                continue
            usage = entry.get("usage")
            if isinstance(usage, Mapping):
                total_prompt += self._safe_int(
                    usage.get("prompt_tokens", usage.get("input_tokens", 0))
                )
                total_completion += self._safe_int(
                    usage.get("completion_tokens", usage.get("output_tokens", 0))
                )
                total_tokens += self._safe_int(usage.get("total_tokens", 0))

        if total_tokens == 0 and (total_prompt + total_completion) > 0:
            total_tokens = total_prompt + total_completion

        return total_prompt, total_completion, total_tokens

    # ------------------------------------------------------------------
    # Findings extraction
    # ------------------------------------------------------------------

    def _extract_findings(self, state: Mapping[str, Any]) -> int:
        """Extract findings count from predictions or explicit override."""
        explicit = state.get("findings_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # Count non-empty prediction outputs
        predictions = state.get("predictions")
        if isinstance(predictions, Sequence) and not isinstance(
            predictions, (str, bytes)
        ):
            count = 0
            for pred in predictions:
                if pred is not None and pred != "" and pred != {}:
                    count += 1
            return count

        # Fallback: count non-empty outputs from history
        history = state.get("history")
        if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            seen: set[str] = set()
            for entry in history:
                if not isinstance(entry, Mapping):
                    continue
                outputs = entry.get("outputs")
                if isinstance(outputs, Sequence) and not isinstance(
                    outputs, (str, bytes)
                ):
                    for out in outputs:
                        key = str(out)[:200] if out else ""
                        if key and key not in seen:
                            seen.add(key)
            return len(seen)

        return 0

    # ------------------------------------------------------------------
    # Coverage extraction
    # ------------------------------------------------------------------

    def _extract_coverage(self, state: Mapping[str, Any]) -> float:
        """Derive coverage from module completion or explicit override."""
        explicit = state.get("coverage_score")
        if explicit is not None:
            return self._clip01(self._safe_float(explicit))

        modules_completed = state.get("modules_completed")
        modules_total = state.get("modules_total")
        if modules_completed is not None and modules_total is not None:
            completed = self._safe_int(modules_completed)
            total = self._safe_int(modules_total)
            if total > 0:
                return self._clip01(completed / total)

        return 0.0
