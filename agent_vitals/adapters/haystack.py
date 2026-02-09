"""Haystack 2.x adapter for converting pipeline/agent state into RawSignals.

Extracts token usage from Haystack's ``ChatMessage._meta.usage`` dicts,
findings from agent messages or pipeline component outputs, and coverage
from agent state or pipeline progress.

Haystack is an optional dependency — this module can be imported without
haystack-ai installed.  The adapter operates on plain dict state payloads
that mirror the structures Haystack exposes at runtime.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ..schema import RawSignals
from .base import BaseAdapter


class HaystackAdapter(BaseAdapter):
    """Extract Agent Vitals signals from Haystack 2.x pipeline/agent state.

    Handles two primary state shapes:

    **Agent state** (from ``Agent.run()``):
      - ``messages``: list of ChatMessage-like dicts with ``role``,
        ``content``, and ``_meta`` (containing ``usage`` dict).
      - ``state``: dict from Agent's ``state_schema`` dynamic keys.

    **Pipeline state** (from ``Pipeline.run()``):
      - ``component_outputs``: dict mapping component names to their
        output dicts (may include ``replies`` with usage metadata).

    Expected state keys (all optional, adapter degrades gracefully):

    - ``messages``: list of ChatMessage-like dicts.
    - ``replies``: list of reply dicts with ``_meta.usage``.
    - ``component_outputs``: dict of {component_name: output_dict}.
    - ``state``: dict with dynamic agent state keys.
    - ``findings_count``: int explicit override.
    - ``coverage_score``: float explicit override.
    - ``total_tokens``: int explicit override.
    - ``query_count``: int explicit override.
    - ``error_count``: int explicit override.
    - ``errors``: list of error strings or dicts.
    - ``sources``: list of source document dicts with ``url`` or ``meta.url``.
    - ``unique_domains``: int explicit override.
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

        # --- Query count ---
        query_count = self._safe_int(normalized.get("query_count", 0))
        if query_count == 0:
            query_count = self._count_assistant_messages(normalized)

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
        """Extract token usage from messages, replies, or component outputs."""

        # Explicit overrides
        explicit_total = state.get("total_tokens")
        if explicit_total is not None:
            prompt = self._safe_int(state.get("prompt_tokens", 0))
            completion = self._safe_int(state.get("completion_tokens", 0))
            total = self._safe_int(explicit_total)
            if total == 0 and (prompt + completion) > 0:
                total = prompt + completion
            return prompt, completion, total

        # Primary: messages list (Agent output)
        messages = state.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            result = self._aggregate_message_usage(messages)
            if result[2] > 0:
                return result

        # Fallback: replies list (Pipeline ChatGenerator output)
        replies = state.get("replies")
        if isinstance(replies, Sequence) and not isinstance(replies, (str, bytes)):
            result = self._aggregate_message_usage(replies)
            if result[2] > 0:
                return result

        # Fallback: component_outputs (Pipeline.run() output)
        component_outputs = state.get("component_outputs")
        if isinstance(component_outputs, Mapping):
            return self._extract_component_tokens(component_outputs)

        return 0, 0, 0

    def _aggregate_message_usage(
        self, messages: Sequence[Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage from ChatMessage-like dicts with _meta.usage."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            meta = msg.get("_meta") or msg.get("meta") or {}
            if not isinstance(meta, Mapping):
                continue
            usage = meta.get("usage")
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

    def _extract_component_tokens(
        self, component_outputs: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        """Extract tokens from Pipeline component outputs containing replies."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for _comp_name, output in component_outputs.items():
            if not isinstance(output, Mapping):
                continue
            replies = output.get("replies")
            if isinstance(replies, Sequence) and not isinstance(
                replies, (str, bytes)
            ):
                p, c, t = self._aggregate_message_usage(replies)
                total_prompt += p
                total_completion += c
                total_tokens += t

        if total_tokens == 0 and (total_prompt + total_completion) > 0:
            total_tokens = total_prompt + total_completion

        return total_prompt, total_completion, total_tokens

    # ------------------------------------------------------------------
    # Findings extraction
    # ------------------------------------------------------------------

    def _extract_findings(self, state: Mapping[str, Any]) -> int:
        """Extract findings count from messages, component outputs, or override."""
        explicit = state.get("findings_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # Count unique assistant message contents
        messages = state.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            seen: set[str] = set()
            for msg in messages:
                if not isinstance(msg, Mapping):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and content:
                    key = str(content)[:200]
                    seen.add(key)
            if seen:
                return len(seen)

        # Fallback: component_outputs reply count
        component_outputs = state.get("component_outputs")
        if isinstance(component_outputs, Mapping):
            count = 0
            for _comp_name, output in component_outputs.items():
                if not isinstance(output, Mapping):
                    continue
                replies = output.get("replies")
                if isinstance(replies, Sequence) and not isinstance(
                    replies, (str, bytes)
                ):
                    count += len(replies)
            if count > 0:
                return count

        return 0

    # ------------------------------------------------------------------
    # Coverage extraction
    # ------------------------------------------------------------------

    def _extract_coverage(self, state: Mapping[str, Any]) -> float:
        """Derive coverage from explicit override or agent state."""
        explicit = state.get("coverage_score")
        if explicit is not None:
            return self._clip01(self._safe_float(explicit))

        # From agent state dict
        agent_state = state.get("state")
        if isinstance(agent_state, Mapping):
            cov = agent_state.get("coverage_score") or agent_state.get("progress")
            if cov is not None:
                return self._clip01(self._safe_float(cov))

        # From pipeline progress: completed components / total
        component_outputs = state.get("component_outputs")
        components_total = state.get("components_total")
        if isinstance(component_outputs, Mapping) and components_total is not None:
            total = self._safe_int(components_total)
            if total > 0:
                return self._clip01(len(component_outputs) / total)

        return 0.0

    # ------------------------------------------------------------------
    # Query count (assistant messages = LLM calls)
    # ------------------------------------------------------------------

    def _count_assistant_messages(self, state: Mapping[str, Any]) -> int:
        """Count assistant messages as a proxy for query/LLM call count."""
        messages = state.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return 0
        count = 0
        for msg in messages:
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                count += 1
        return count

    # ------------------------------------------------------------------
    # Sources / domains extraction
    # ------------------------------------------------------------------

    def _extract_sources(self, state: Mapping[str, Any]) -> tuple[int, int]:
        """Extract sources count and unique domain count from source documents."""
        sources = state.get("sources")
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            return 0, 0

        domains: set[str] = set()
        valid_count = 0
        for src in sources:
            if not isinstance(src, Mapping):
                continue
            valid_count += 1
            url = src.get("url")
            if url is None:
                meta = self._as_mapping(src.get("meta"))
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
