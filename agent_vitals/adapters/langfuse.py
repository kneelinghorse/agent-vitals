"""Langfuse adapter for converting trace/observation data into RawSignals.

Extracts token usage from Langfuse generation observations, findings from
generation outputs, errors from observation status/level, and coverage from
trace metadata or scores.

Langfuse is an optional dependency — this module can be imported without
langfuse installed.  The adapter operates on plain dict state payloads that
mirror the structures Langfuse exposes via its Python SDK and API.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ..schema import RawSignals
from .base import BaseAdapter


class LangfuseAdapter(BaseAdapter):
    """Extract Agent Vitals signals from Langfuse trace/observation data.

    Handles two primary state shapes:

    **Trace-centric** (from ``langfuse.get_trace()`` or callback data):
      - ``trace``: dict with ``input``, ``output``, ``metadata``, ``tags``.
      - ``observations``: list of observation dicts with ``type``
        (``GENERATION``, ``SPAN``, ``EVENT``), ``usage``, ``output``,
        ``level``, ``status_message``, ``model``, ``start_time``, ``end_time``.

    **Flat generation list** (from direct instrumentation):
      - ``generations``: list of generation dicts with ``usage``, ``output``,
        ``model``, ``input``, ``completion_start_time``.

    Expected state keys (all optional, adapter degrades gracefully):

    - ``trace``: dict with trace-level metadata.
    - ``observations``: list of observation dicts (spans, generations, events).
    - ``generations``: list of generation dicts (shorthand for generation-only).
    - ``scores``: list of score dicts with ``name`` and ``value``.
    - ``findings_count``: int explicit override for findings.
    - ``coverage_score``: float explicit override for coverage.
    - ``total_tokens``: int explicit override for total tokens.
    - ``query_count``: int explicit override for query count.
    - ``unique_domains``: int explicit override for domain count.
    - ``error_count``: int explicit override for error count.
    - ``errors``: list of error strings or dicts.
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
        error_count = self._extract_errors(normalized)

        # --- Query count (generation/LLM calls) ---
        query_count = self._safe_int(normalized.get("query_count", 0))
        if query_count == 0:
            query_count = self._count_generations(normalized)

        # --- Sources / domains ---
        unique_domains = self._safe_int(normalized.get("unique_domains", 0))
        sources_count = self._safe_int(normalized.get("sources_count", 0))
        if unique_domains == 0 or sources_count == 0:
            s_count, d_count = self._extract_sources(normalized)
            if sources_count == 0:
                sources_count = s_count
            if unique_domains == 0:
                unique_domains = d_count

        return self.validate(
            RawSignals(
                findings_count=findings_count,
                sources_count=sources_count,
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
                unique_domains=unique_domains,
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
        """Extract token usage from generations, observations, or explicit overrides."""

        # Explicit overrides
        explicit_total = state.get("total_tokens")
        if explicit_total is not None:
            prompt = self._safe_int(state.get("prompt_tokens", 0))
            completion = self._safe_int(state.get("completion_tokens", 0))
            total = self._safe_int(explicit_total)
            if total == 0 and (prompt + completion) > 0:
                total = prompt + completion
            return prompt, completion, total

        # Primary: observations list (includes generations)
        observations = state.get("observations")
        if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            result = self._aggregate_observation_usage(observations)
            if result[2] > 0:
                return result

        # Fallback: flat generations list
        generations = state.get("generations")
        if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
            result = self._aggregate_generation_usage(generations)
            if result[2] > 0:
                return result

        return 0, 0, 0

    def _aggregate_observation_usage(
        self, observations: Sequence[Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage from observation dicts with usage metadata."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            # Only count generations (LLM calls) for token usage
            obs_type = str(obs.get("type", "")).upper()
            if obs_type not in ("GENERATION", ""):
                continue
            p, c, t = self._extract_usage_from_dict(obs)
            total_prompt += p
            total_completion += c
            total_tokens += t

        if total_tokens == 0 and (total_prompt + total_completion) > 0:
            total_tokens = total_prompt + total_completion

        return total_prompt, total_completion, total_tokens

    def _aggregate_generation_usage(
        self, generations: Sequence[Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage from flat generation dicts."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for gen in generations:
            if not isinstance(gen, Mapping):
                continue
            p, c, t = self._extract_usage_from_dict(gen)
            total_prompt += p
            total_completion += c
            total_tokens += t

        if total_tokens == 0 and (total_prompt + total_completion) > 0:
            total_tokens = total_prompt + total_completion

        return total_prompt, total_completion, total_tokens

    def _extract_usage_from_dict(
        self, obj: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        """Extract token counts from a single observation/generation dict.

        Looks for usage in these locations (priority order):
        1. ``usage`` dict (standard Langfuse format)
        2. ``usage_details`` dict (newer Langfuse format)
        3. Top-level ``prompt_tokens``/``completion_tokens``/``total_tokens``
        """
        # Standard usage dict
        usage = obj.get("usage")
        if isinstance(usage, Mapping):
            return self._parse_usage_dict(usage)

        # Newer usage_details dict
        usage_details = obj.get("usage_details")
        if isinstance(usage_details, Mapping):
            return self._parse_usage_dict(usage_details)

        # Top-level keys
        prompt = self._safe_int(obj.get("prompt_tokens", 0))
        completion = self._safe_int(obj.get("completion_tokens", 0))
        total = self._safe_int(obj.get("total_tokens", 0))
        if total == 0 and (prompt + completion) > 0:
            total = prompt + completion
        return prompt, completion, total

    def _parse_usage_dict(self, usage: Mapping[str, Any]) -> tuple[int, int, int]:
        """Parse token counts from a usage/usage_details dict."""
        prompt = self._safe_int(
            usage.get("prompt_tokens", usage.get("input_tokens", usage.get("input", 0)))
        )
        completion = self._safe_int(
            usage.get("completion_tokens", usage.get("output_tokens", usage.get("output", 0)))
        )
        total = self._safe_int(usage.get("total_tokens", usage.get("total", 0)))
        if total == 0 and (prompt + completion) > 0:
            total = prompt + completion
        return prompt, completion, total

    # ------------------------------------------------------------------
    # Findings extraction
    # ------------------------------------------------------------------

    def _extract_findings(self, state: Mapping[str, Any]) -> int:
        """Extract findings count from generation outputs, trace output, or override."""
        explicit = state.get("findings_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # Count unique non-empty generation outputs from observations
        observations = state.get("observations")
        if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            count = self._count_unique_generation_outputs(observations)
            if count > 0:
                return count

        # Flat generations list
        generations = state.get("generations")
        if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
            count = self._count_unique_outputs(generations)
            if count > 0:
                return count

        # Trace-level output
        trace = state.get("trace")
        if isinstance(trace, Mapping):
            output = trace.get("output")
            if output is not None and output != "" and output != {}:
                return 1

        return 0

    def _count_unique_generation_outputs(self, observations: Sequence[Any]) -> int:
        """Count unique non-empty outputs from GENERATION observations."""
        seen: set[str] = set()
        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            obs_type = str(obs.get("type", "")).upper()
            if obs_type not in ("GENERATION", ""):
                continue
            output = obs.get("output")
            if output is not None and output != "" and output != {}:
                key = str(output)[:200]
                seen.add(key)
        return len(seen)

    def _count_unique_outputs(self, items: Sequence[Any]) -> int:
        """Count unique non-empty outputs from a list of dicts."""
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            output = item.get("output")
            if output is not None and output != "" and output != {}:
                key = str(output)[:200]
                seen.add(key)
        return len(seen)

    # ------------------------------------------------------------------
    # Coverage extraction
    # ------------------------------------------------------------------

    def _extract_coverage(self, state: Mapping[str, Any]) -> float:
        """Derive coverage from explicit override, scores, or trace metadata."""
        explicit = state.get("coverage_score")
        if explicit is not None:
            return self._clip01(self._safe_float(explicit))

        # From scores list (look for coverage-related score)
        scores = state.get("scores")
        if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
            for score in scores:
                if not isinstance(score, Mapping):
                    continue
                name = str(score.get("name", "")).lower()
                if name in ("coverage", "coverage_score", "progress", "completion"):
                    value = score.get("value")
                    if value is not None:
                        return self._clip01(self._safe_float(value))

        # From trace metadata
        trace = state.get("trace")
        if isinstance(trace, Mapping):
            metadata = self._as_mapping(trace.get("metadata"))
            cov = metadata.get("coverage_score") or metadata.get("progress")
            if cov is not None:
                return self._clip01(self._safe_float(cov))

        return 0.0

    # ------------------------------------------------------------------
    # Error extraction
    # ------------------------------------------------------------------

    def _extract_errors(self, state: Mapping[str, Any]) -> int:
        """Extract error count from observations, explicit count, or error list."""
        explicit = state.get("error_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # Count errors from explicit errors list
        errors_list = state.get("errors")
        if isinstance(errors_list, Sequence) and not isinstance(errors_list, (str, bytes)):
            if len(errors_list) > 0:
                return len(errors_list)

        # Count observations with ERROR level or error status
        observations = state.get("observations")
        if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            count = self._count_observation_errors(observations)
            if count > 0:
                return count

        # Count generations with errors
        generations = state.get("generations")
        if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
            count = self._count_generation_errors(generations)
            if count > 0:
                return count

        return 0

    def _count_observation_errors(self, observations: Sequence[Any]) -> int:
        """Count observations with ERROR level or non-empty status_message."""
        count = 0
        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            level = str(obs.get("level", "")).upper()
            status_message = obs.get("status_message")
            if level == "ERROR":
                count += 1
            elif status_message and str(status_message).strip():
                count += 1
        return count

    def _count_generation_errors(self, generations: Sequence[Any]) -> int:
        """Count generations with error indicators."""
        count = 0
        for gen in generations:
            if not isinstance(gen, Mapping):
                continue
            level = str(gen.get("level", "")).upper()
            status_message = gen.get("status_message")
            if level == "ERROR":
                count += 1
            elif status_message and str(status_message).strip():
                count += 1
        return count

    # ------------------------------------------------------------------
    # Query count (generation observations = LLM calls)
    # ------------------------------------------------------------------

    def _count_generations(self, state: Mapping[str, Any]) -> int:
        """Count generation observations as a proxy for LLM call count."""
        # From observations list
        observations = state.get("observations")
        if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            count = 0
            for obs in observations:
                if not isinstance(obs, Mapping):
                    continue
                obs_type = str(obs.get("type", "")).upper()
                if obs_type == "GENERATION":
                    count += 1
            if count > 0:
                return count

        # From flat generations list
        generations = state.get("generations")
        if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
            return len([g for g in generations if isinstance(g, Mapping)])

        return 0

    # ------------------------------------------------------------------
    # Sources / domains extraction
    # ------------------------------------------------------------------

    def _extract_sources(self, state: Mapping[str, Any]) -> tuple[int, int]:
        """Extract sources count and unique domain count from trace metadata or observations."""
        # From explicit sources list
        sources = state.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            return self._count_sources_and_domains(sources)

        # From trace metadata
        trace = state.get("trace")
        if isinstance(trace, Mapping):
            metadata = self._as_mapping(trace.get("metadata"))
            sources = metadata.get("sources")
            if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
                return self._count_sources_and_domains(sources)

        return 0, 0

    def _count_sources_and_domains(self, sources: Sequence[Any]) -> tuple[int, int]:
        """Count source items and unique domains from a sources list."""
        domains: set[str] = set()
        valid_count = 0
        for src in sources:
            if isinstance(src, str):
                valid_count += 1
                hostname = self._extract_hostname(src)
                if hostname:
                    domains.add(hostname)
            elif isinstance(src, Mapping):
                valid_count += 1
                url = src.get("url") or src.get("source") or src.get("link")
                if url is None:
                    meta = self._as_mapping(src.get("metadata") or src.get("meta"))
                    url = meta.get("url") or meta.get("source")
                if isinstance(url, str) and url:
                    hostname = self._extract_hostname(url)
                    if hostname:
                        domains.add(hostname)
        return valid_count, len(domains)

    @staticmethod
    def _extract_hostname(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return parsed.hostname or ""
