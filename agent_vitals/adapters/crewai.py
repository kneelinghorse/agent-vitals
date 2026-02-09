"""CrewAI adapter for converting crew state into RawSignals."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import RawSignals
from .base import BaseAdapter

_COMPLETED_STATES = {"completed", "done", "success", "successful", "finished"}
_FAILED_STATES = {"failed", "error", "errored", "cancelled", "canceled", "timeout"}


class CrewAIAdapter(BaseAdapter):
    """Extract Agent Vitals signals from CrewAI-style state dictionaries."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)

        usage_metrics = self._extract_usage_metrics(normalized)
        prompt_tokens = self._safe_int(
            usage_metrics.get("prompt_tokens", usage_metrics.get("input_tokens", 0))
        )
        completion_tokens = self._safe_int(
            usage_metrics.get("completion_tokens", usage_metrics.get("output_tokens", 0))
        )
        total_tokens = self._safe_int(
            normalized.get(
                "total_tokens",
                usage_metrics.get("total_tokens", prompt_tokens + completion_tokens),
            )
        )

        tasks = self._extract_tasks(normalized)
        task_outputs = self._extract_task_outputs(normalized)

        findings_count = self._safe_int(normalized.get("findings_count", 0))
        if findings_count == 0:
            findings_count = self._safe_len(task_outputs, 0)

        completed_tasks = self._count_completed_tasks(tasks, task_outputs, normalized)
        total_tasks = self._safe_int(normalized.get("total_tasks", normalized.get("task_count", 0)))
        if total_tasks <= 0:
            total_tasks = self._safe_len(tasks, 0)
        if total_tasks <= 0:
            total_tasks = max(completed_tasks, self._safe_len(task_outputs, 0))

        explicit_coverage = normalized.get("coverage_score", normalized.get("coverage"))
        if explicit_coverage is not None:
            coverage_score = self._clip01(self._safe_float(explicit_coverage, 0.0))
        elif total_tasks > 0:
            coverage_score = self._clip01(completed_tasks / total_tasks)
        else:
            coverage_score = 0.0

        error_count = self._safe_int(normalized.get("error_count", normalized.get("cumulative_errors", -1)))
        if error_count < 0:
            error_count = self._derive_error_count(tasks, task_outputs, normalized)

        step_events = self._extract_step_events(normalized)
        api_calls = self._safe_int(normalized.get("api_calls", normalized.get("query_count", 0)))
        if api_calls == 0:
            api_calls = self._safe_len(step_events, 0)

        return self.validate(
            RawSignals(
                findings_count=findings_count,
                sources_count=self._safe_int(normalized.get("sources_count", 0)),
                objectives_covered=self._safe_int(normalized.get("objectives_covered", 0)),
                coverage_score=coverage_score,
                confidence_score=self._safe_float(normalized.get("confidence_score", 0.0)),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                api_calls=api_calls,
                query_count=api_calls,
                unique_domains=self._safe_int(normalized.get("unique_domains", 0)),
                refinement_count=self._safe_int(normalized.get("refinement_count", 0)),
                convergence_delta=self._safe_float(normalized.get("convergence_delta", 0.0)),
                error_count=error_count,
            )
        )

    def _extract_usage_metrics(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        direct = self._as_mapping(state.get("usage_metrics"))
        if direct:
            return direct

        crew = self._as_mapping_or_object(state.get("crew"))
        if crew:
            usage = self._as_mapping(self._value_from_mapping_or_object(crew, "usage_metrics"))
            if usage:
                return usage

        crew_output = self._as_mapping_or_object(state.get("crew_output", state.get("output")))
        if crew_output:
            for key in ("usage_metrics", "token_usage"):
                usage = self._as_mapping(self._value_from_mapping_or_object(crew_output, key))
                if usage:
                    return usage

        return {}

    def _extract_tasks(self, state: Mapping[str, Any]) -> list[Any]:
        direct = state.get("tasks")
        if isinstance(direct, list):
            return direct

        crew = self._as_mapping_or_object(state.get("crew"))
        tasks = self._value_from_mapping_or_object(crew, "tasks")
        if isinstance(tasks, list):
            return tasks

        return []

    def _extract_task_outputs(self, state: Mapping[str, Any]) -> list[Any]:
        for key in ("task_outputs", "outputs", "results"):
            direct = state.get(key)
            if isinstance(direct, list):
                return direct

        crew_output = self._as_mapping_or_object(state.get("crew_output", state.get("output")))
        if crew_output:
            for key in ("tasks_output", "task_outputs", "outputs", "results"):
                items = self._value_from_mapping_or_object(crew_output, key)
                if isinstance(items, list):
                    return items

        crew = self._as_mapping_or_object(state.get("crew"))
        items = self._value_from_mapping_or_object(crew, "tasks_output")
        if isinstance(items, list):
            return items

        return []

    def _extract_step_events(self, state: Mapping[str, Any]) -> list[Any]:
        callback = state.get("step_callback")
        if isinstance(callback, list):
            return callback
        if isinstance(callback, tuple):
            return list(callback)
        return []

    def _count_completed_tasks(
        self,
        tasks: list[Any],
        task_outputs: list[Any],
        state: Mapping[str, Any],
    ) -> int:
        explicit_completed = state.get("completed_tasks")
        if explicit_completed is not None:
            return max(0, self._safe_int(explicit_completed, 0))

        completed = 0
        for task in tasks:
            status = self._extract_status(task)
            if status in _COMPLETED_STATES:
                completed += 1

        if completed == 0 and task_outputs:
            completed = self._safe_len(task_outputs, 0)
        return completed

    def _derive_error_count(
        self,
        tasks: list[Any],
        task_outputs: list[Any],
        state: Mapping[str, Any],
    ) -> int:
        total = 0

        for task in tasks:
            status = self._extract_status(task)
            if status in _FAILED_STATES:
                total += 1

        for output in task_outputs:
            output_mapping = self._as_mapping_or_object(output)
            status = self._extract_status(output_mapping)
            if status in _FAILED_STATES:
                total += 1
                continue

            if self._has_error_payload(output_mapping):
                total += 1

        for event in self._extract_step_events(state):
            event_mapping = self._as_mapping_or_object(event)
            status = self._extract_status(event_mapping)
            if status in _FAILED_STATES or self._has_error_payload(event_mapping):
                total += 1

        errors_field = state.get("errors")
        if isinstance(errors_field, list):
            total += len(errors_field)

        return total

    def _extract_status(self, item: Any) -> str:
        mapping = self._as_mapping_or_object(item)
        if not mapping:
            return ""
        raw_status = self._value_from_mapping_or_object(
            mapping,
            "status",
            fallback_key="state",
        )
        if not isinstance(raw_status, str):
            return ""
        return raw_status.strip().lower()

    def _has_error_payload(self, item: Any) -> bool:
        mapping = self._as_mapping_or_object(item)
        if not mapping:
            return False
        for key in ("error", "exception", "failed"):
            value = self._value_from_mapping_or_object(mapping, key)
            if isinstance(value, bool) and value:
                return True
            if value not in (None, "", [], {}):
                return True
        return False

    def _as_mapping_or_object(self, value: Any) -> Mapping[str, Any] | Any:
        if isinstance(value, Mapping):
            return value
        if value is None:
            return {}
        return value

    def _value_from_mapping_or_object(
        self,
        value: Mapping[str, Any] | Any,
        key: str,
        *,
        fallback_key: str | None = None,
    ) -> Any:
        if isinstance(value, Mapping):
            if key in value:
                return value.get(key)
            if fallback_key is not None and fallback_key in value:
                return value.get(fallback_key)
            return None

        attr = getattr(value, key, None)
        if attr is not None:
            return attr
        if fallback_key is not None:
            return getattr(value, fallback_key, None)
        return None


__all__ = ["CrewAIAdapter"]
