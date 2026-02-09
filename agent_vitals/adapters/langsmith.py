"""LangSmith adapter for converting run trace data into RawSignals.

Extracts token usage from LangSmith run ``usage_metadata``, findings from
run outputs, errors from run ``error`` field and ``status``, and coverage
from feedback stats or metadata.

LangSmith is an optional dependency — this module can be imported without
langsmith installed.  The adapter operates on plain dict state payloads that
mirror the structures LangSmith exposes via its Python SDK and API.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ..schema import RawSignals
from .base import BaseAdapter


class LangSmithAdapter(BaseAdapter):
    """Extract Agent Vitals signals from LangSmith run trace data.

    Handles two primary state shapes:

    **Single run** (from ``client.read_run()`` or callback data):
      - ``run_type``: ``"llm"`` | ``"chain"`` | ``"tool"`` | ``"retriever"``
      - ``outputs``: dict of run output data.
      - ``usage_metadata``: dict with ``input_tokens``, ``output_tokens``,
        ``total_tokens``.
      - ``error``: str error message (None if success).
      - ``status``: ``"success"`` | ``"error"``.

    **Run with children** (chain run with nested LLM/tool runs):
      - ``child_runs``: list of child run dicts (token usage aggregated
        from ``run_type="llm"`` children).

    Expected state keys (all optional, adapter degrades gracefully):

    - ``run_type``: str run type hint.
    - ``outputs``: dict of run output data.
    - ``usage_metadata``: dict with token counts.
    - ``error``: str error message.
    - ``status``: str run status.
    - ``child_runs``: list of child run dicts.
    - ``feedback_stats``: dict of ``{score_name: {mean, count, ...}}``.
    - ``tags``: list of string tags.
    - ``extra``: dict with ``metadata`` sub-dict.
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

        # --- Query count (LLM runs) ---
        query_count = self._safe_int(normalized.get("query_count", 0))
        if query_count == 0:
            query_count = self._count_llm_runs(normalized)

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
        """Extract token usage from usage_metadata, child_runs, or overrides."""

        # Explicit overrides
        explicit_total = state.get("total_tokens")
        if explicit_total is not None:
            prompt = self._safe_int(state.get("prompt_tokens", 0))
            completion = self._safe_int(state.get("completion_tokens", 0))
            total = self._safe_int(explicit_total)
            if total == 0 and (prompt + completion) > 0:
                total = prompt + completion
            return prompt, completion, total

        # Primary: usage_metadata on the run itself
        usage = state.get("usage_metadata")
        if isinstance(usage, Mapping):
            result = self._parse_usage_metadata(usage)
            if result[2] > 0:
                return result

        # Fallback: aggregate from child_runs (LLM runs)
        child_runs = state.get("child_runs")
        if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes)):
            result = self._aggregate_child_run_tokens(child_runs)
            if result[2] > 0:
                return result

        # Fallback: extra.metadata with token fields
        extra = state.get("extra")
        if isinstance(extra, Mapping):
            metadata = self._as_mapping(extra.get("metadata"))
            usage_in_meta = metadata.get("usage_metadata") or metadata.get("usage")
            if isinstance(usage_in_meta, Mapping):
                result = self._parse_usage_metadata(usage_in_meta)
                if result[2] > 0:
                    return result

        return 0, 0, 0

    def _parse_usage_metadata(
        self, usage: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        """Parse token counts from a LangSmith usage_metadata dict."""
        prompt = self._safe_int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0))
        )
        completion = self._safe_int(
            usage.get("output_tokens", usage.get("completion_tokens", 0))
        )
        total = self._safe_int(usage.get("total_tokens", 0))
        if total == 0 and (prompt + completion) > 0:
            total = prompt + completion
        return prompt, completion, total

    def _aggregate_child_run_tokens(
        self, child_runs: Sequence[Any]
    ) -> tuple[int, int, int]:
        """Aggregate token usage from child runs (only LLM-type runs)."""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0

        for run in child_runs:
            if not isinstance(run, Mapping):
                continue
            run_type = str(run.get("run_type", "")).lower()
            if run_type not in ("llm", ""):
                continue
            usage = run.get("usage_metadata")
            if isinstance(usage, Mapping):
                p, c, t = self._parse_usage_metadata(usage)
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
        """Extract findings count from run outputs, child_run outputs, or override."""
        explicit = state.get("findings_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # From run outputs dict
        outputs = state.get("outputs")
        if isinstance(outputs, Mapping):
            count = self._count_output_findings(outputs)
            if count > 0:
                return count

        # From child_runs outputs (unique outputs from LLM runs)
        child_runs = state.get("child_runs")
        if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes)):
            count = self._count_child_run_findings(child_runs)
            if count > 0:
                return count

        return 0

    def _count_output_findings(self, outputs: Mapping[str, Any]) -> int:
        """Count findings from a run's outputs dict.

        Looks for common output patterns:
        - ``output``: single string/dict output → 1 if non-empty
        - ``outputs``: list of outputs → count non-empty
        - ``generations``: list of generation lists → count non-empty
        - ``result``: single result → 1 if non-empty
        """
        # List of outputs
        output_list = outputs.get("outputs") or outputs.get("results")
        if isinstance(output_list, Sequence) and not isinstance(output_list, (str, bytes)):
            seen: set[str] = set()
            for item in output_list:
                if item is not None and item != "" and item != {}:
                    key = str(item)[:200]
                    seen.add(key)
            if seen:
                return len(seen)

        # Generations (LLM output format)
        generations = outputs.get("generations")
        if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
            count = 0
            for gen_list in generations:
                if isinstance(gen_list, Sequence) and not isinstance(gen_list, (str, bytes)):
                    count += len([g for g in gen_list if g])
                elif gen_list:
                    count += 1
            if count > 0:
                return count

        # Single output
        output = outputs.get("output") or outputs.get("result") or outputs.get("text")
        if output is not None and output != "" and output != {}:
            return 1

        return 0

    def _count_child_run_findings(self, child_runs: Sequence[Any]) -> int:
        """Count unique non-empty outputs from child runs."""
        seen: set[str] = set()
        for run in child_runs:
            if not isinstance(run, Mapping):
                continue
            outputs = run.get("outputs")
            if isinstance(outputs, Mapping):
                output = outputs.get("output") or outputs.get("text") or outputs.get("result")
                if output and str(output).strip():
                    key = str(output)[:200]
                    seen.add(key)
            elif isinstance(outputs, str) and outputs.strip():
                seen.add(outputs[:200])
        return len(seen)

    # ------------------------------------------------------------------
    # Coverage extraction
    # ------------------------------------------------------------------

    def _extract_coverage(self, state: Mapping[str, Any]) -> float:
        """Derive coverage from explicit override, feedback_stats, or metadata."""
        explicit = state.get("coverage_score")
        if explicit is not None:
            return self._clip01(self._safe_float(explicit))

        # From feedback_stats (LangSmith evaluation scores)
        feedback_stats = state.get("feedback_stats")
        if isinstance(feedback_stats, Mapping):
            for key in ("coverage", "coverage_score", "progress", "completion"):
                stat = feedback_stats.get(key)
                if isinstance(stat, Mapping):
                    mean_val = stat.get("mean") or stat.get("avg")
                    if mean_val is not None:
                        return self._clip01(self._safe_float(mean_val))
                elif stat is not None:
                    return self._clip01(self._safe_float(stat))

        # From extra.metadata
        extra = state.get("extra")
        if isinstance(extra, Mapping):
            metadata = self._as_mapping(extra.get("metadata"))
            cov = metadata.get("coverage_score") or metadata.get("progress")
            if cov is not None:
                return self._clip01(self._safe_float(cov))

        return 0.0

    # ------------------------------------------------------------------
    # Error extraction
    # ------------------------------------------------------------------

    def _extract_errors(self, state: Mapping[str, Any]) -> int:
        """Extract error count from run error field, status, child_runs, or override."""
        explicit = state.get("error_count")
        if explicit is not None:
            return self._safe_int(explicit)

        # Count from explicit errors list
        errors_list = state.get("errors")
        if isinstance(errors_list, Sequence) and not isinstance(errors_list, (str, bytes)):
            if len(errors_list) > 0:
                return len(errors_list)

        count = 0

        # Run-level error field
        error = state.get("error")
        if error is not None and str(error).strip():
            count += 1

        # Run-level status
        status = str(state.get("status", "")).lower()
        if status == "error" and count == 0:
            count += 1

        # Child runs with errors
        child_runs = state.get("child_runs")
        if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes)):
            for run in child_runs:
                if not isinstance(run, Mapping):
                    continue
                child_error = run.get("error")
                child_status = str(run.get("status", "")).lower()
                if child_error and str(child_error).strip():
                    count += 1
                elif child_status == "error":
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Query count (LLM run count)
    # ------------------------------------------------------------------

    def _count_llm_runs(self, state: Mapping[str, Any]) -> int:
        """Count LLM-type runs as a proxy for LLM call count."""
        # If this is an LLM run itself
        run_type = str(state.get("run_type", "")).lower()
        if run_type == "llm":
            return 1

        # Count LLM child runs
        child_runs = state.get("child_runs")
        if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes)):
            count = 0
            for run in child_runs:
                if not isinstance(run, Mapping):
                    continue
                child_type = str(run.get("run_type", "")).lower()
                if child_type == "llm":
                    count += 1
            if count > 0:
                return count

        # Fallback: if usage_metadata exists, at least 1 LLM call happened
        if isinstance(state.get("usage_metadata"), Mapping):
            return 1

        return 0

    # ------------------------------------------------------------------
    # Sources / domains extraction
    # ------------------------------------------------------------------

    def _extract_sources(self, state: Mapping[str, Any]) -> tuple[int, int]:
        """Extract sources count and unique domain count from run data."""
        # From explicit sources list
        sources = state.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            return self._count_sources_and_domains(sources)

        # From extra.metadata
        extra = state.get("extra")
        if isinstance(extra, Mapping):
            metadata = self._as_mapping(extra.get("metadata"))
            sources = metadata.get("sources")
            if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
                return self._count_sources_and_domains(sources)

        # From retriever child_runs
        child_runs = state.get("child_runs")
        if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes)):
            return self._count_retriever_sources(child_runs)

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

    def _count_retriever_sources(self, child_runs: Sequence[Any]) -> tuple[int, int]:
        """Count documents from retriever child_runs."""
        domains: set[str] = set()
        total_docs = 0
        for run in child_runs:
            if not isinstance(run, Mapping):
                continue
            run_type = str(run.get("run_type", "")).lower()
            if run_type != "retriever":
                continue
            outputs = run.get("outputs")
            if not isinstance(outputs, Mapping):
                continue
            documents = outputs.get("documents") or outputs.get("results")
            if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
                continue
            for doc in documents:
                if not isinstance(doc, Mapping):
                    continue
                total_docs += 1
                meta = self._as_mapping(doc.get("metadata") or doc.get("meta"))
                url = meta.get("source") or meta.get("url")
                if isinstance(url, str) and url:
                    hostname = self._extract_hostname(url)
                    if hostname:
                        domains.add(hostname)
        return total_docs, len(domains)

    @staticmethod
    def _extract_hostname(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return parsed.hostname or ""
