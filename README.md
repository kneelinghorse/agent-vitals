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
- **Immutable snapshots**: Every `step()` returns a `VitalsSnapshot` with signals, metrics, and detection results
- **JSONL export**: Auto-log every snapshot to structured JSONL files
- **Backtest harness**: Offline evaluation of recorded trajectories with P/R/F1 metrics
- **Context manager**: `with AgentVitals(...) as monitor:` for clean resource management

## Detection Modes

| Detector | What it catches | Signal |
|---|---|---|
| **Loop** | Agent repeating actions without progress | Findings plateau over N steps |
| **Stuck** | Coverage stagnation despite continued work | Low DM + low CV on coverage |
| **Thrash** | Excessive errors indicating instability | Error count above threshold |
| **Runaway Cost** | Token burn with no output | Token spike with flat findings |

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

## Session Summary

```python
monitor = AgentVitals(mission_id="my-task")
# ... run steps ...
summary = monitor.summary()
# {"mission_id": "my-task", "total_steps": 8, "health_state": "healthy",
#  "any_loop_detected": False, "any_stuck_detected": False, ...}

monitor.reset()  # Clear history for next run (also flushes exporters)
```

## License

MIT — see [LICENSE](LICENSE).
