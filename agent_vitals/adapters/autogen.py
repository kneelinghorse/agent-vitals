"""AutoGen/AG2 adapter for converting agent state into RawSignals."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import RawSignals
from .base import BaseAdapter

_FAILED_STATES = {"failed", "error", "errored", "cancelled", "canceled", "timeout"}


class AutoGenAdapter(BaseAdapter):
    """Extract Agent Vitals signals from AutoGen/AG2-style state dictionaries."""

    def extract(self, state: Mapping[str, Any]) -> RawSignals:
        normalized = self.normalize(state)

        usage_summary = self._extract_usage_summary(normalized)
        prompt_tokens = self._safe_int(
            usage_summary.get("prompt_tokens", usage_summary.get("input_tokens", 0))
        )
        completion_tokens = self._safe_int(
            usage_summary.get("completion_tokens", usage_summary.get("output_tokens", 0))
        )
        total_tokens = self._safe_int(
            normalized.get(
                "total_tokens",
                usage_summary.get("total_tokens", prompt_tokens + completion_tokens),
            )
        )

        outputs = self._extract_outputs(normalized)
        chat_messages = self._extract_chat_messages(normalized)

        findings_count = self._safe_int(normalized.get("findings_count", normalized.get("outputs_count", 0)))
        if findings_count == 0:
            findings_count = self._safe_len(outputs, 0)
        if findings_count == 0:
            findings_count = self._safe_len(chat_messages, 0)

        error_count = self._safe_int(normalized.get("error_count", normalized.get("cumulative_errors", -1)))
        if error_count < 0:
            error_count = self._derive_error_count(chat_messages, outputs, normalized)

        coverage_score = self._derive_coverage_score(
            normalized,
            findings_count=findings_count,
            chat_messages=chat_messages,
        )

        api_calls = self._safe_int(normalized.get("api_calls", normalized.get("query_count", 0)))
        if api_calls == 0:
            api_calls = self._safe_len(chat_messages, 0)

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

    def _extract_usage_summary(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        direct = self._flatten_usage_summary(self._as_mapping(state.get("usage_summary")))
        if direct:
            return direct

        token_usage = self._flatten_usage_summary(self._as_mapping(state.get("token_usage")))
        if token_usage:
            return token_usage

        agent = state.get("agent")
        if agent is not None:
            for method_name in ("get_total_usage", "get_actual_usage"):
                usage = self._call_agent_usage_method(agent, method_name)
                flattened = self._flatten_usage_summary(usage)
                if flattened:
                    return flattened

            agent_usage = self._flatten_usage_summary(self._as_mapping_or_object(agent))
            if agent_usage:
                return agent_usage

        agents = state.get("agents")
        if isinstance(agents, list):
            aggregated = self._aggregate_agents_usage(agents)
            if aggregated:
                return aggregated

        return {}

    def _extract_outputs(self, state: Mapping[str, Any]) -> list[Any]:
        for key in ("outputs", "results", "task_outputs", "responses"):
            value = state.get(key)
            if isinstance(value, list):
                return value
        return []

    def _extract_chat_messages(self, state: Mapping[str, Any]) -> list[Any]:
        direct = state.get("chat_messages")
        if isinstance(direct, list):
            return direct
        if isinstance(direct, Mapping):
            return self._flatten_chat_messages_mapping(direct)

        for key in ("messages", "conversation"):
            value = state.get(key)
            if isinstance(value, list):
                return value

        agent = state.get("agent")
        if agent is not None:
            messages = getattr(agent, "chat_messages", None)
            if isinstance(messages, list):
                return messages
            if isinstance(messages, Mapping):
                return self._flatten_chat_messages_mapping(messages)

        return []

    def _derive_error_count(
        self,
        chat_messages: list[Any],
        outputs: list[Any],
        state: Mapping[str, Any],
    ) -> int:
        total = 0

        for message in chat_messages:
            payload = self._as_mapping_or_object(message)
            if self._message_is_error(payload):
                total += 1

        for output in outputs:
            payload = self._as_mapping_or_object(output)
            if self._message_is_error(payload):
                total += 1

        errors_field = state.get("errors")
        if isinstance(errors_field, list):
            total += len(errors_field)

        return total

    def _derive_coverage_score(
        self,
        state: Mapping[str, Any],
        *,
        findings_count: int,
        chat_messages: list[Any],
    ) -> float:
        explicit = state.get("coverage_score", state.get("progress_score", state.get("coverage")))
        if explicit is not None:
            return self._clip01(self._safe_float(explicit, 0.0))

        mission_objectives_total = self._safe_len(state.get("mission_objectives"), 0)
        objectives_covered = self._safe_int(
            state.get(
                "objectives_covered",
                self._safe_len(state.get("covered_objectives"), 0),
            )
        )
        if mission_objectives_total > 0:
            return self._clip01(objectives_covered / mission_objectives_total)

        total_turns = self._safe_int(
            state.get("total_turns", state.get("max_turns", state.get("expected_turns", 0)))
        )
        completed_turns = self._safe_int(state.get("completed_turns", self._safe_len(chat_messages, 0)))
        if total_turns > 0:
            return self._clip01(completed_turns / total_turns)

        target_findings = self._safe_int(state.get("target_findings", state.get("expected_outputs", 0)))
        if target_findings > 0:
            return self._clip01(findings_count / target_findings)

        return 0.0

    def _aggregate_agents_usage(self, agents: list[Any]) -> Mapping[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for agent in agents:
            usage = self._call_agent_usage_method(agent, "get_total_usage")
            if not usage:
                usage = self._call_agent_usage_method(agent, "get_actual_usage")
            flattened = self._flatten_usage_summary(usage)
            if not flattened:
                continue
            totals["prompt_tokens"] += self._safe_int(
                flattened.get("prompt_tokens", flattened.get("input_tokens", 0))
            )
            totals["completion_tokens"] += self._safe_int(
                flattened.get("completion_tokens", flattened.get("output_tokens", 0))
            )
            totals["total_tokens"] += self._safe_int(
                flattened.get(
                    "total_tokens",
                    self._safe_int(flattened.get("prompt_tokens", 0))
                    + self._safe_int(flattened.get("completion_tokens", 0)),
                )
            )
        return totals

    def _flatten_usage_summary(self, summary: Mapping[str, Any]) -> Mapping[str, Any]:
        if not summary:
            return {}
        if self._looks_like_usage(summary):
            return summary

        nested_keys = (
            "total_usage",
            "actual_usage",
            "usage",
            "usage_summary",
            "usage_excluding_cached_inference",
            "usage_including_cached_inference",
        )
        for key in nested_keys:
            nested = self._as_mapping(summary.get(key))
            if not nested:
                continue
            flattened = self._flatten_usage_summary(nested)
            if flattened:
                return flattened

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        found = False
        for value in summary.values():
            nested = self._as_mapping(value)
            if not nested:
                continue
            flattened = self._flatten_usage_summary(nested)
            if not flattened:
                continue
            found = True
            prompt_tokens += self._safe_int(
                flattened.get("prompt_tokens", flattened.get("input_tokens", 0))
            )
            completion_tokens += self._safe_int(
                flattened.get("completion_tokens", flattened.get("output_tokens", 0))
            )
            total_tokens += self._safe_int(
                flattened.get(
                    "total_tokens",
                    self._safe_int(flattened.get("prompt_tokens", 0))
                    + self._safe_int(flattened.get("completion_tokens", 0)),
                )
            )

        if found:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        return {}

    def _call_agent_usage_method(self, agent: Any, method_name: str) -> Mapping[str, Any]:
        method = getattr(agent, method_name, None)
        if not callable(method):
            return {}
        try:
            value = method()
        except Exception:
            return {}
        return self._as_mapping(value)

    def _message_is_error(self, payload: Mapping[str, Any] | Any) -> bool:
        mapping = self._as_mapping_or_object(payload)
        status = self._extract_status(mapping)
        if status in _FAILED_STATES:
            return True

        for key in ("error", "exception", "failed", "is_error"):
            value = self._value_from_mapping_or_object(mapping, key)
            if isinstance(value, bool):
                if value:
                    return True
                continue
            if value not in (None, "", [], {}):
                return True

        return False

    def _extract_status(self, payload: Mapping[str, Any] | Any) -> str:
        value = self._value_from_mapping_or_object(payload, "status", fallback_key="state")
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    def _flatten_chat_messages_mapping(self, messages: Mapping[str, Any]) -> list[Any]:
        flattened: list[Any] = []
        for value in messages.values():
            if isinstance(value, list):
                flattened.extend(value)
        return flattened

    def _looks_like_usage(self, summary: Mapping[str, Any]) -> bool:
        keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        }
        return any(key in summary for key in keys)

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


__all__ = ["AutoGenAdapter"]
