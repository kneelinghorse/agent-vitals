"""LangChain adapter for converting agent state into RawSignals."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse

from ..schema import RawSignals
from .base import BaseAdapter


class LangChainAdapter(BaseAdapter):
    """Extract Agent Vitals signals from LangChain-style state dictionaries."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)
        token_usage = self._extract_token_usage(normalized)

        findings_count = self._safe_int(
            normalized.get(
                "findings_count",
                normalized.get("cumulative_outputs", normalized.get("outputs_count", 0)),
            )
        )
        if findings_count == 0:
            findings_count = self._safe_len(normalized.get("outputs"), 0)
        if findings_count == 0 and normalized.get("output") not in (None, ""):
            findings_count = 1

        sources_count = self._safe_int(
            normalized.get("sources_count", normalized.get("cumulative_sources", 0))
        )
        if sources_count == 0:
            sources_count = self._safe_len(normalized.get("source_documents"), 0)

        query_count = self._safe_int(
            normalized.get("query_count", normalized.get("cumulative_queries", 0))
        )
        if query_count == 0:
            query_count = self._safe_len(normalized.get("intermediate_steps"), 0)

        error_count = self._safe_int(
            normalized.get("error_count", normalized.get("cumulative_errors", 0))
        )
        if error_count == 0:
            error_count = self._safe_len(normalized.get("errors"), 0)

        prompt_tokens = self._safe_int(
            token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
        )
        completion_tokens = self._safe_int(
            token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
        )
        total_tokens = self._safe_int(
            normalized.get(
                "total_tokens",
                normalized.get(
                    "cumulative_tokens",
                    token_usage.get("total_tokens", prompt_tokens + completion_tokens),
                ),
            )
        )

        unique_domains = self._safe_int(normalized.get("unique_domains", 0))
        if unique_domains == 0:
            unique_domains = self._count_unique_domains(normalized.get("source_documents"))

        coverage_score = self._clip01(
            self._safe_float(
                normalized.get(
                    "coverage_score",
                    normalized.get("progress_score", normalized.get("coverage", 0.0)),
                )
            )
        )

        return self.validate(
            RawSignals(
                findings_count=findings_count,
                sources_count=sources_count,
                objectives_covered=self._safe_int(normalized.get("objectives_covered", 0)),
                coverage_score=coverage_score,
                confidence_score=self._safe_float(normalized.get("confidence_score", 0.0)),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                api_calls=query_count,
                query_count=query_count,
                unique_domains=unique_domains,
                refinement_count=self._safe_int(normalized.get("refinement_count", 0)),
                convergence_delta=self._safe_float(normalized.get("convergence_delta", 0.0)),
                error_count=error_count,
            )
        )

    def _extract_token_usage(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        direct = self._as_mapping(state.get("token_usage"))
        if direct:
            return direct

        usage_metadata = self._as_mapping(state.get("usage_metadata"))
        if usage_metadata:
            return usage_metadata

        llm_output = self._as_mapping(state.get("llm_output"))
        nested = self._as_mapping(llm_output.get("token_usage"))
        if nested:
            return nested

        return {}

    def _count_unique_domains(self, docs: Any) -> int:
        if not isinstance(docs, list):
            return 0

        domains: set[str] = set()
        for item in docs:
            if not isinstance(item, Mapping):
                continue
            source = item.get("source")
            if source is None:
                metadata = self._as_mapping(item.get("metadata"))
                source = metadata.get("source")
            if not isinstance(source, str):
                continue
            hostname = self._extract_hostname(source)
            if hostname:
                domains.add(hostname)
        return len(domains)

    @staticmethod
    def _extract_hostname(source: str) -> str:
        parsed = urlparse(source if "://" in source else f"https://{source}")
        return parsed.hostname or ""
