"""LangGraph adapter for converting TypedDict state into RawSignals."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse

from ..schema import RawSignals
from .base import BaseAdapter


class LangGraphAdapter(BaseAdapter):
    """Extract Agent Vitals signals from LangGraph-style state dictionaries."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)
        token_usage = self._as_mapping(normalized.get("token_usage"))

        findings_count = self._safe_int(
            normalized.get("findings_count", self._safe_len(normalized.get("findings"), 0))
        )
        sources_found = normalized.get("sources_found")
        sources_count = self._safe_int(
            normalized.get("sources_count", self._safe_len(sources_found, 0))
        )

        query_count = self._safe_int(
            normalized.get(
                "query_count",
                self._safe_len(normalized.get("queries"), self._safe_len(normalized.get("search_queries"), 0)),
            )
        )
        error_count = self._safe_int(
            normalized.get("error_count", self._safe_len(normalized.get("errors"), 0))
        )

        objectives_covered = self._safe_int(
            normalized.get(
                "objectives_covered", self._safe_len(normalized.get("covered_objectives"), 0)
            )
        )
        mission_objectives_total = self._safe_len(normalized.get("mission_objectives"), 0)
        coverage_score = self._derive_coverage(
            normalized,
            objectives_covered=objectives_covered,
            mission_objectives_total=mission_objectives_total,
        )

        prompt_tokens = self._safe_int(
            normalized.get("prompt_tokens", token_usage.get("prompt_tokens", 0))
        )
        completion_tokens = self._safe_int(
            normalized.get("completion_tokens", token_usage.get("completion_tokens", 0))
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
            unique_domains = self._count_unique_domains(sources_found)

        return self.validate(
            RawSignals(
                findings_count=findings_count,
                sources_count=sources_count,
                objectives_covered=objectives_covered,
                coverage_score=coverage_score,
                confidence_score=self._safe_float(normalized.get("confidence_score", 0.0)),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                api_calls=query_count,
                query_count=query_count,
                unique_domains=unique_domains,
                refinement_count=self._safe_int(
                    normalized.get("refinement_count", normalized.get("research_loop_count", 0))
                ),
                convergence_delta=self._safe_float(
                    normalized.get("convergence_delta", normalized.get("delta_coverage", 0.0))
                ),
                error_count=error_count,
            )
        )

    def _derive_coverage(
        self,
        state: Mapping[str, Any],
        *,
        objectives_covered: int,
        mission_objectives_total: int,
    ) -> float:
        explicit = state.get("coverage_score")
        if explicit is not None:
            return self._clip01(self._safe_float(explicit, 0.0))

        progress = state.get("progress_score", state.get("coverage"))
        if progress is not None:
            return self._clip01(self._safe_float(progress, 0.0))

        if mission_objectives_total > 0:
            return self._clip01(objectives_covered / mission_objectives_total)
        return 0.0

    def _count_unique_domains(self, sources: Any) -> int:
        if not isinstance(sources, list):
            return 0

        domains: set[str] = set()
        for item in sources:
            source = self._extract_source(item)
            if not source:
                continue
            parsed = urlparse(source if "://" in source else f"https://{source}")
            if parsed.hostname:
                domains.add(parsed.hostname)
        return len(domains)

    def _extract_source(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if not isinstance(item, Mapping):
            return ""
        for key in ("url", "source", "domain"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = self._as_mapping(item.get("metadata"))
        for key in ("url", "source", "domain"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
