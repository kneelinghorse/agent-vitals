# Agent Vitals

[![CI](https://github.com/kneelinghorse/agent-vitals/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kneelinghorse/agent-vitals/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/agent-vitals)](https://pypi.org/project/agent-vitals/)
[![Python](https://img.shields.io/pypi/pyversions/agent-vitals)](https://pypi.org/project/agent-vitals/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Standalone agent health monitor** — detect loops, stuck states, thrash, and runaway costs in any AI agent workflow.

Agent Vitals watches your LLM agent's vital signs in real time. Feed it four numbers per step and it tells you when your agent is looping, stuck, thrashing, or burning tokens for nothing.

## Install

```bash
pip install agent-vitals
```

```bash
# Optional framework integrations
pip install "agent-vitals[langchain,langgraph]"
```

```bash
# Optional observability export (OTLP)
pip install "agent-vitals[otlp]"
```

```bash
# Development and CI tooling (tests, coverage, lint/type checks)
pip install "agent-vitals[dev]"
```

## Quick Start

```python
from agent_vitals import AgentVitals

monitor = AgentVitals(mission_id="my-task")

for step in range(max_steps):
    result = call_llm(prompt)
    findings = extract_findings(result)

    snapshot = monitor.step(
        findings_count=len(findings),
        coverage_score=compute_coverage(findings),
        total_tokens=result.usage.total_tokens,
        error_count=error_tracker.count,
    )

    if snapshot.any_failure:
        print(f"Health issue at step {snapshot.loop_index}: "
              f"{snapshot.stuck_trigger or snapshot.loop_trigger}")
        break
```

## Features

- **4-field minimum**: Only `findings_count`, `coverage_score`, `total_tokens`, `error_count` required
- **Zero-config defaults**: `AgentVitals()` works out of the box with tuned thresholds
- **Framework-agnostic**: No dependency on LangChain, LangGraph, or any agent framework
- **Built-in adapters**: LangChain, LangGraph, CrewAI, AutoGen/AG2, DSPy, Haystack, Langfuse, and LangSmith signal extraction
- **Immutable snapshots**: Every `step()` returns a `VitalsSnapshot` with signals, metrics, and detection results
- **JSONL export**: Auto-log every snapshot to structured JSONL files
- **OTLP export**: Send metrics to Datadog, Grafana Cloud, or any OTLP backend
- **Backtest harness**: Offline evaluation of recorded trajectories with P/R/F1 metrics
- **Context manager**: `with AgentVitals(...) as monitor:` for clean resource management

## Detection Modes

| Detector | What it catches | Signal |
|---|---|---|
| **Loop** | Agent repeating actions without progress | Findings plateau over N steps |
| **Stuck** | Coverage stagnation despite continued work | Low DM + low CV on coverage |
| **Thrash** | Excessive errors indicating instability | Error count above threshold |
| **Runaway Cost** | Token burn with no output | Token spike with flat findings |

### Content-Based Loop Detection (v1.5.0)

When you pass `output_text` to `monitor.step()`, Agent Vitals computes content-level
similarity to distinguish loops from stuck states:

```python
snapshot = monitor.step(
    findings_count=5,
    coverage_score=0.6,
    total_tokens=12000,
    error_count=0,
    output_text="The agent's latest output text here...",
)

# New fields on VitalsSnapshot:
print(snapshot.output_similarity)    # 0.0–1.0 Jaccard similarity vs previous output
print(snapshot.output_fingerprint)   # SHA-256 hash for exact-match detection
```

- **High similarity** (≥0.85): Confirms loop — agent is producing repetitive outputs
- **Low similarity** with stagnant coverage: Confirms stuck — agent is producing varied but unproductive outputs
- **No output_text**: Detection falls back to signal-level heuristics (fully backward-compatible)

## API Overview

### Manual Integration (Recommended)

```python
from agent_vitals import AgentVitals

monitor = AgentVitals(mission_id="research-task")
snapshot = monitor.step(
    findings_count=5,
    coverage_score=0.6,
    total_tokens=12000,
    error_count=0,
)

print(snapshot.health_state)     # "healthy" | "warning" | "critical"
print(snapshot.any_failure)      # True if loop or stuck detected
print(snapshot.stuck_trigger)    # e.g. "coverage_stagnation", "burn_rate_anomaly"
```

### Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import TelemetryAdapter

monitor = AgentVitals(mission_id="my-task", adapter=TelemetryAdapter())
snapshot = monitor.step_from_state({
    "cumulative_outputs": 5,
    "coverage_score": 0.6,
    "cumulative_tokens": 12000,
    "cumulative_errors": 0,
})
```

### LangChain Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import LangChainAdapter

monitor = AgentVitals(mission_id="lc-agent", adapter=LangChainAdapter())
snapshot = monitor.step_from_state({
    "cumulative_outputs": 7,
    "coverage_score": 0.72,
    "llm_output": {"token_usage": {"prompt_tokens": 1200, "completion_tokens": 600, "total_tokens": 1800}},
    "cumulative_errors": 1,
    "intermediate_steps": [("search", "..."), ("summarize", "...")],
})
```

### LangGraph Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import LangGraphAdapter

monitor = AgentVitals(mission_id="lg-agent", adapter=LangGraphAdapter())
snapshot = monitor.step_from_state({
    "findings": ["f1", "f2"],
    "sources_found": [{"url": "https://example.com/a"}],
    "mission_objectives": ["o1", "o2", "o3"],
    "covered_objectives": ["o1", "o2"],
    "total_tokens": 4200,
    "errors": [],
})
```

### CrewAI Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import CrewAIAdapter

monitor = AgentVitals(mission_id="crewai-agent", adapter=CrewAIAdapter())
snapshot = monitor.step_from_state({
    "crew": {
        "usage_metrics": {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
        "tasks": [{"status": "completed"}, {"status": "failed"}, {"status": "completed"}],
    },
    "task_outputs": [{"result": "finding-a"}, {"result": "finding-b"}],
})
```

### AutoGen / AG2 Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import AutoGenAdapter

monitor = AgentVitals(mission_id="autogen-agent", adapter=AutoGenAdapter())
snapshot = monitor.step_from_state({
    "usage_summary": {
        "agent_a": {"prompt_tokens": 90, "completion_tokens": 40, "total_tokens": 130},
        "agent_b": {"prompt_tokens": 70, "completion_tokens": 35, "total_tokens": 105},
    },
    "chat_messages": [{"role": "user"}, {"role": "assistant"}, {"role": "assistant"}],
    "total_turns": 6,
})
```

### DSPy Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import DSPyAdapter

monitor = AgentVitals(mission_id="dspy-program", adapter=DSPyAdapter())
snapshot = monitor.step_from_state({
    "lm_usage": {
        "openai/gpt-4o-mini": {
            "prompt_tokens": 1200,
            "completion_tokens": 400,
            "total_tokens": 1600,
        },
    },
    "predictions": [{"answer": "Summary A"}, {"answer": "Analysis B"}],
    "modules_completed": 2,
    "modules_total": 3,
    "errors": [],
})
```

The DSPy adapter extracts tokens from `lm_usage` (preferred) or `lm.history` (fallback),
findings from `predictions` or history outputs, and coverage from module completion state.
No `dspy` dependency required.

### Haystack Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import HaystackAdapter

monitor = AgentVitals(mission_id="haystack-agent", adapter=HaystackAdapter())
snapshot = monitor.step_from_state({
    "messages": [
        {"role": "user", "content": "Research quantum computing"},
        {
            "role": "assistant",
            "content": "Quantum error correction advances...",
            "_meta": {"usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}},
        },
    ],
    "state": {"coverage_score": 0.6},
    "sources": [
        {"url": "https://arxiv.org/paper1"},
        {"url": "https://nature.com/article1"},
    ],
})
```

The Haystack adapter handles both Agent state (`messages` with `_meta.usage`) and
Pipeline state (`component_outputs` with `replies`). Extracts source URLs for domain
counting. No `haystack-ai` dependency required.

### Langfuse Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import LangfuseAdapter

monitor = AgentVitals(mission_id="langfuse-agent", adapter=LangfuseAdapter())
snapshot = monitor.step_from_state({
    "observations": [
        {
            "type": "GENERATION",
            "model": "gpt-4o",
            "output": "Analysis of market trends in Q4.",
            "usage": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            "level": "DEFAULT",
        },
        {
            "type": "SPAN",
            "name": "web_search",
            "output": {"results": ["result1", "result2"]},
        },
    ],
    "scores": [{"name": "coverage", "value": 0.65}],
    "sources": [
        {"url": "https://example.com/report"},
        {"url": "https://other.org/data"},
    ],
})
```

The Langfuse adapter extracts tokens from GENERATION observations (`usage` or
`usage_details`), findings from unique generation outputs, errors from observation
`level` ("ERROR") and `status_message`, and coverage from `scores` or trace metadata.
Also accepts flat `generations` lists. No `langfuse` dependency required.

### LangSmith Adapter Integration

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import LangSmithAdapter

monitor = AgentVitals(mission_id="langsmith-agent", adapter=LangSmithAdapter())
snapshot = monitor.step_from_state({
    "run_type": "chain",
    "usage_metadata": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
    "outputs": {"output": "Analysis of market trends in Q4."},
    "child_runs": [
        {
            "run_type": "llm",
            "usage_metadata": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
            "outputs": {"output": "Generated analysis."},
        },
        {
            "run_type": "retriever",
            "outputs": {
                "documents": [
                    {"metadata": {"source": "https://example.com/report"}},
                ],
            },
        },
    ],
    "feedback_stats": {"coverage": {"mean": 0.65}},
    "status": "success",
})
```

The LangSmith adapter extracts tokens from `usage_metadata` (preferred) or LLM
`child_runs` (fallback), findings from run `outputs`, errors from the `error` field
and `status`, and coverage from `feedback_stats` or `extra.metadata`. Retriever
child runs provide source/domain counts. No `langsmith` dependency required.

### LangChain Callback Integration

```python
from agent_vitals.callbacks import LangChainVitalsCallback

callback = LangChainVitalsCallback(
    mission_id="lc-callback",
    on_failure="log",            # "log" | "raise" | "callback"
    export_jsonl_dir="./vitals_logs",
)

# Pass callback into your LangChain runnable/agent callback list.
```

### LangGraph Node Integration

```python
from agent_vitals.callbacks import LangGraphVitalsNode

vitals_node = LangGraphVitalsNode(on_failure="force_finalize")

# Add `vitals_node` to your StateGraph as a normal callable node.
# Returned update includes:
#   - agent_vitals: snapshot payload
#   - force_finalize: True (when failure detected and mode is force_finalize)
```

### Pre-built Signals

```python
from agent_vitals import AgentVitals, RawSignals

monitor = AgentVitals(mission_id="my-task")
signals = RawSignals(findings_count=5, coverage_score=0.6, total_tokens=12000, error_count=0)
snapshot = monitor.step_from_signals(signals)
```

## Export

Log every snapshot to JSONL for offline analysis or observability pipelines.

```python
from agent_vitals import AgentVitals, JSONLExporter

exporter = JSONLExporter(
    directory="./vitals_logs",
    layout="per_run",       # or "append"
    max_bytes=10_000_000,   # rotation threshold (append mode)
)

with AgentVitals(mission_id="my-task", exporters=[exporter]) as monitor:
    for step in range(max_steps):
        monitor.step(findings_count=..., coverage_score=..., total_tokens=..., error_count=...)
# Exporter is automatically flushed and closed on exit
```

**Layouts:**
- `per_run`: `{directory}/{mission_id}/{run_id}.jsonl` — one file per run
- `append`: `{directory}/{mission_id}.jsonl` — all runs in one file, with rotation

### OTLP Export (Datadog / Grafana / OTLP-compatible)

```python
from agent_vitals import AgentVitals, OTLPExporter

otlp = OTLPExporter(
    endpoint="http://localhost:4318/v1/metrics",
    service_name="deepsearch-agent",
    mission_id="DRM.0.5",
    run_id="run-2026-02-09",
    workflow_type="research",
    export_interval_ms=5000,
)

with AgentVitals(mission_id="DRM.0.5", exporters=[otlp]) as monitor:
    monitor.step(findings_count=1, coverage_score=0.2, total_tokens=300, error_count=0)
```

Datadog example (delta temporality enabled):

```python
from agent_vitals import OTLPExporter

datadog = OTLPExporter(
    endpoint="https://otlp.datadoghq.com/v1/metrics",
    headers={"DD-API-KEY": "<datadog_api_key>"},
    service_name="agent-vitals",
    mission_id="DRM.0.5",
    run_id="run-42",
    workflow_type="research",
    delta_temporality=True,
)
```

Grafana Cloud example:

```python
from agent_vitals import OTLPExporter

grafana = OTLPExporter(
    endpoint="https://otlp-gateway-<region>.grafana.net/otlp/v1/metrics",
    headers={"Authorization": "Basic <base64(instance_id:api_key)>"},
    service_name="agent-vitals",
    mission_id="DRM.0.5",
    run_id="run-42",
    workflow_type="research",
)
```

## Configuration

```python
from agent_vitals import AgentVitals, VitalsConfig

# From constructor kwargs
monitor = AgentVitals(config=VitalsConfig(
    loop_consecutive_count=6,
    stuck_dm_threshold=0.15,
))

# From YAML file
monitor = AgentVitals.from_yaml("thresholds.yaml")

# From environment variables (VITALS_* prefix)
monitor = AgentVitals()  # auto-reads VITALS_LOOP_CONSECUTIVE_COUNT, etc.
```

### Key Thresholds

| Parameter | Default | Description |
|---|---|---|
| `loop_consecutive_count` | 5 | Steps of flat findings before loop detection |
| `stuck_dm_threshold` | 0.15 | DM below this → coverage stagnation |
| `stuck_cv_threshold` | 0.5 | CV below this → low variation |
| `burn_rate_multiplier` | 2.0 | Token spike ratio for burn rate anomaly |

### Framework-Specific Threshold Profiles (v1.5.0)

Different agent frameworks have different normal operating patterns. Framework profiles
automatically tune detection thresholds when you use a built-in adapter:

```python
from agent_vitals import AgentVitals
from agent_vitals.adapters import CrewAIAdapter

# Profile auto-detected from adapter type
monitor = AgentVitals(mission_id="crew-task", adapter=CrewAIAdapter())
# → Uses crewai profile: loop_consecutive_count=8, burn_rate_multiplier=4.0
```

Built-in profiles:

| Framework | `loop_consecutive_count` | `burn_rate_multiplier` | Notes |
|---|---|---|---|
| langgraph | 5 | 2.5 | Tighter loop detection for graph-based workflows |
| crewai | 8 | 4.0 | Higher burn rate tolerance for multi-agent crews |
| dspy | 10 | — | Higher consecutive count for optimization loops |

Override auto-detection with the `framework` parameter:

```python
monitor = AgentVitals(
    mission_id="task",
    adapter=LangGraphAdapter(),
    framework="crewai",  # Override: use crewai profile instead
)
```

Define custom profiles in `thresholds.yaml`:

```yaml
loop_consecutive_count: 6
profiles:
  langgraph:
    loop_consecutive_count: 5
    burn_rate_multiplier: 2.5
  crewai:
    loop_consecutive_count: 8
    burn_rate_multiplier: 4.0
    token_scale_factor: 0.7
```

## Backtest

Evaluate detection accuracy against labeled trajectory corpora.

```python
from agent_vitals.backtest import load_dataset, load_labels, run_backtest

dataset = load_dataset("path/to/traces/")
labels = load_labels("path/to/labels.json")
report = run_backtest(dataset, labels)

print(f"vitals.any: P={report.composite_any.precision:.3f} "
      f"R={report.composite_any.recall:.3f} "
      f"F1={report.composite_any.f1:.3f}")

for name, detector in report.detectors.items():
    print(f"  {name}: P={detector.precision:.3f} R={detector.recall:.3f}")
```

## CI Coverage Gate

CI enforces coverage with `pytest-cov`:

- Command: `pytest --cov=agent_vitals --cov-report=xml --cov-fail-under=85`
- Baseline measured on 2026-02-09: **85% total coverage**
- Coverage XML artifact is uploaded in GitHub Actions (`coverage.xml`)

## Session Summary

```python
monitor = AgentVitals(mission_id="my-task")
# ... run steps ...
summary = monitor.summary()
# {"mission_id": "my-task", "total_steps": 8, "health_state": "healthy",
#  "any_loop_detected": False, "any_stuck_detected": False, ...}

monitor.reset()  # Clear history for next run (also flushes exporters)
```

## Detection Precision

Agent Vitals has been validated against a 70-trace combined corpus spanning
DeepSearch (LangGraph/Ollama) and cross-agent (LangChain, raw OpenAI, CrewAI, AutoGen,
DSPy, Haystack — with GPT-4o-mini, DeepSeek-chat, Gemini, Llama, Mixtral, Claude, and
local OSS models) trajectories.

### Cross-Agent Corpus (40 traces, 7 frameworks, 7 models)

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| **vitals.any** | **1.000** | **1.000** | **1.000** |
| loop | 0.875 | 1.000 | 0.933 |
| stuck | 1.000 | 0.800 | 0.889 |
| thrash | 1.000 | 1.000 | 1.000 |

### Combined Corpus (70 traces)

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| **vitals.any** | **1.000** | **0.982** | **0.991** |
| loop | 0.909 | 0.870 | 0.889 |
| stuck | 0.824 | 0.737 | 0.778 |
| thrash | 1.000 | 1.000 | 1.000 |

The composite `vitals.any` signal — used for enforcement decisions — maintains perfect
precision across both corpora. Per-detector metrics are informational; the system
correctly identifies failures even in the 2 edge cases where loop and stuck signals
overlap. Content-based similarity (v1.5.0) addresses these edge cases for new traces
that provide `output_text`.

See `docs/vitals/av25-backtest-report.md` for the latest backtest analysis.

## License

MIT — see [LICENSE](LICENSE).
