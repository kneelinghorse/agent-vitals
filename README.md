# Agent Vitals

[![CI](https://github.com/kneelinghorse/agent-vitals/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kneelinghorse/agent-vitals/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/agent-vitals)](https://pypi.org/project/agent-vitals/)
[![Python](https://img.shields.io/pypi/pyversions/agent-vitals)](https://pypi.org/project/agent-vitals/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The direct-integration health monitor for production AI agents** — detect loops, stuck states, confabulation, thrash, and runaway costs with four numbers per step.

Agent Vitals is the easy-onboarding option for teams that want loop / stuck / runaway detection inside their agent workflow without standing up a separate observability service. **Four fields per step, zero configuration to start, ~5 MB base install.** Optional ML detectors (TDA, Hopfield early-screen) live behind explicit extras so the base install stays light.

```python
from agent_vitals import AgentVitals

monitor = AgentVitals(mission_id="my-task")
snapshot = monitor.step(
    findings_count=5,
    coverage_score=0.6,
    total_tokens=12000,
    error_count=0,
)
if snapshot.any_failure:
    handle_failure(snapshot)
```

That's the whole onboarding surface. Adapters for LangChain, LangGraph, CrewAI, AutoGen/AG2, DSPy, Haystack, Langfuse, and LangSmith ship in the base install — no framework dependencies required.

## Install

```bash
# Base install — handcrafted detectors only, no ML deps
pip install agent-vitals
```

```bash
# Optional framework integrations
pip install "agent-vitals[langchain,langgraph]"
```

```bash
# Optional observability export (OTLP → Datadog / Grafana / any OTLP backend)
pip install "agent-vitals[otlp]"
```

```bash
# Optional TDA override layer (giotto-tda + sklearn, ~150 MB)
pip install "agent-vitals[tda]"
```

```bash
# Optional Hopfield early-screen layer (onnxruntime + numpy, ~50 MB)
pip install "agent-vitals[hopfield]"
```

```bash
# Development and CI tooling (tests, coverage, lint/type checks)
pip install "agent-vitals[dev]"
```

The base install ships only `pydantic` + `pyyaml`. ML-heavy detector stacks are explicitly opt-in and never imported unless the matching extra is installed.

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
- **Optional ML detector layers**: TDA override (`agent-vitals[tda]`) and Hopfield early-screen (`agent-vitals[hopfield]`) — base install stays light, ML stacks are explicitly opt-in

## Detection Modes

Agent Vitals ships five detectors. The composite `vitals.any` signal is what enforcement hooks fire on; per-detector flags are available for targeted handling.

| Detector | What it catches | Signal |
|---|---|---|
| **Loop** | Agent repeating actions without progress | Findings plateau over N steps + content similarity |
| **Stuck** | Coverage stagnation despite continued work | Low DM + low CV on coverage |
| **Confabulation** | Plausible-but-unsupported output | Coverage / similarity divergence |
| **Thrash** | Excessive errors indicating instability | Error count above threshold |
| **Runaway Cost** | Token burn with no output | Token spike with flat findings (CUSUM-tracked) |

## Detector Layers

Detectors are organized into three layers, each independently opt-in:

```
Layer 1 — Handcrafted (always on, base install)
    loop · stuck · confabulation · thrash · runaway_cost
            │
            ▼
Layer 2 — TDA override (optional, agent-vitals[tda])
    runaway_cost adjudication via persistent-homology features
            │
            ▼
Layer 3 — Hopfield early-screen (optional, agent-vitals[hopfield])
    early-window detection at step prefixes 3–5, where handcrafted
    signals lack evidence (informational marker; never overrides)
```

- **Layer 1 — Handcrafted** is the default and the source of truth. All five detectors run on the four-field input and produce immediate per-step verdicts. This is what `pip install agent-vitals` gets you.
- **Layer 2 — TDA override** plugs into `runaway_cost` adjudication for trajectories where the handcrafted heuristics produce ambiguous evidence. Installed via `agent-vitals[tda]`. See `docs/vitals/tda-detector-design.md` for the design.
- **Layer 3 — Hopfield early-screen** runs a small ONNX model trained on early-window prefixes (cutoffs 3 and 5) to surface failures before the handcrafted stack accumulates enough evidence. It propagates as an informational `hopfield_override_active` marker on the snapshot — it never mutates per-detector flags, so adding `[hopfield]` is bit-identical to baseline on existing detector cells. Trained and validated by [`agent-vitals-bench`](https://github.com/kneelinghorse/agent-vitals-bench) on a 1494-trace corpus (macro-F1 0.901 at p3 vs handcrafted 0.466 — Hopfield is the only paradigm with meaningful early-prefix signal).

### Content-Based Loop Detection

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

### Framework-Specific Threshold Profiles

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

Bundled-corpus numbers (v1.15.0, default config) from `python scripts/ci_backtest.py` over the three bundled corpora — 370 traces / 1898 snapshots spanning synthetic, real, and AV-31-reviewed trajectories:

| Detector | Precision | Recall | F1 | Gate status |
|---|---|---|---|---|
| **vitals.any** (composite) | **0.992** | **0.946** | **0.969** | composite gate PASS |
| loop | 0.977 | 1.000 | 0.988 | **HARD GATE PASS** |
| stuck | 0.916 | 0.813 | 0.861 | informational |
| confabulation | 1.000 | 0.682 | 0.811 | informational |
| thrash | 1.000 | 1.000 | 1.000 | informational |
| runaway_cost | 0.850 | 0.895 | 0.872 | informational |

The composite `vitals.any` signal — what enforcement hooks fire on — clears the CI gate at P≥0.90 / R≥0.85. Loop is promoted to **hard gate** status (Wilson lower bounds P_lb=0.947 / R_lb=0.982 over 213 positives). Run `python scripts/ci_backtest.py` for the live numbers; the script also emits `backtest-results.json` for artifact upload.

For cross-framework precision/recall over a much larger labeled corpus (1494 traces, 7 frameworks, 7 models), see [`agent-vitals-bench`](https://github.com/kneelinghorse/agent-vitals-bench) and its `eval-cross-framework-v1` artifact set. The bench corpus is the source of truth for cross-framework gates and updates faster than this README.

## License

MIT — see [LICENSE](LICENSE).
