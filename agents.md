# AI Agent Configuration

## Project Overview
**Project Name**: Agent Vitals  
**Project Type**: Python Library (pip package)  
**Primary Language**: Python 3.10+  
**Framework**: Standalone (pydantic + pyyaml core; optional framework adapters)  
**Version**: 1.15.0 (Production/Stable)  
**License**: MIT

**Description**: Framework-agnostic health monitoring library for AI agent workflows. Detects five critical failure modes — loops, stuck states, confabulation, thrash, and runaway costs — through temporal signal analysis, content similarity, and statistical process control.

---

## Build & Development Commands

### Installation & Setup
```bash
# Install in development mode with all extras
pip install -e ".[dev,all]"

# Install core only (no framework adapters)
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage (CI gate: 85%)
pytest --cov=agent_vitals --cov-fail-under=85

# Run a specific test file
pytest tests/test_detection_loop.py -v

# Run backtest evaluation against bundled corpora
python scripts/ci_backtest.py
```

### Linting & Formatting
```bash
# Lint (ruff, line-length=100, target py310)
ruff check agent_vitals/ tests/

# Auto-fix lint issues
ruff check --fix agent_vitals/ tests/

# Type checking (strict mode)
mypy agent_vitals/
```

### Building
```bash
# Build distribution
python -m build

# Verify package
twine check dist/*
```

---

## Project Structure & Navigation

### Directory Layout
```
agent-vitals/
├── agent_vitals/                  # Core library
│   ├── __init__.py                # Public API exports
│   ├── monitor.py                 # AgentVitals main class (stateful monitor)
│   ├── schema.py                  # Pydantic models (VitalsSnapshot, RawSignals)
│   ├── config.py                  # Configuration & threshold management
│   ├── thresholds.yaml            # YAML threshold definitions
│   ├── backtest.py                # Offline evaluation harness (P/R/F1)
│   ├── ci_gate.py                 # CI gate promotion logic (Wilson CI)
│   ├── exceptions.py              # Custom exceptions
│   ├── stratified.py              # Stratified corpus utilities
│   │
│   ├── detection/                 # Core detection engine
│   │   ├── loop.py                # Loop/stuck/confabulation detector (~1000 LOC)
│   │   ├── metrics.py             # Temporal metrics (CV, DM, hysteresis)
│   │   ├── similarity.py          # Jaccard similarity + fingerprinting
│   │   ├── cusum.py               # CUSUM change-point tracking (SPC)
│   │   ├── adaptive_threshold.py  # Dynamic control limits (WMA)
│   │   ├── signal_mapping.py      # Model-size-aware normalization
│   │   └── stop_rule.py           # Stop-rule signal derivation
│   │
│   ├── adapters/                  # Framework signal extractors
│   │   ├── base.py                # SignalAdapter protocol
│   │   ├── langchain.py, langgraph.py, crewai.py
│   │   ├── autogen.py, dspy.py, haystack.py
│   │   ├── langfuse.py, langsmith.py
│   │   └── __init__.py            # TelemetryAdapter (cross-agent)
│   │
│   ├── callbacks/                 # Framework callback implementations
│   │   ├── langchain.py           # LangChainVitalsCallback
│   │   └── langgraph.py           # LangGraphVitalsNode
│   │
│   └── export/                    # Observability exporters
│       ├── jsonl.py               # JSONLExporter
│       └── otlp.py               # OTLPExporter (Datadog/Grafana)
│
├── tests/                         # Test suite (~36 files, 85% coverage gate)
├── scripts/                       # CI and utility scripts
├── checkpoints/                   # Bundled backtest corpora
│   └── vitals_corpus/
│       ├── av05_synth/            # 70 synthetic traces
│       └── av26_real/             # Real agent trajectories
└── cmos/                          # Project management (DO NOT write code here)
```

### Key Files
- `agent_vitals/monitor.py` — Main `AgentVitals` class; public API entry point
- `agent_vitals/detection/loop.py` — Core multi-signal detection engine (most complex file)
- `agent_vitals/config.py` — Configuration priority chain (kwargs > YAML > env > defaults)
- `agent_vitals/thresholds.yaml` — All tunable detection parameters
- `agent_vitals/schema.py` — `VitalsSnapshot`, `RawSignals`, `TemporalMetricsResult`
- `agent_vitals/backtest.py` — Evaluation harness with Wilson CI gate math
- `scripts/ci_backtest.py` — CI entry point for backtest validation

---

## Coding Standards & Style

### Python Guidelines
- **Target**: Python 3.10+ (no walrus operator abuse, use `match` sparingly)
- **Line length**: 100 characters (ruff enforced)
- **Type checking**: mypy strict mode — all public functions fully typed
- **Models**: Pydantic v2 `BaseModel` for data schemas; `@dataclass` for internal state
- **Imports**: stdlib first, third-party second, local third (ruff isort)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants

### Architecture Patterns
- **Immutable snapshots**: `VitalsSnapshot` is frozen — no mutation after creation
- **Protocol-based adapters**: `SignalAdapter` protocol for framework integrations
- **Zero framework deps in core**: All framework imports are optional extras
- **Adaptive thresholds**: Scale detection parameters to trace length, not fixed counts
- **Multi-signal fusion**: Detectors combine multiple evidence signals before firing
- **Model-size awareness**: Auto-classify small/medium/large models for threshold adjustment

### Forbidden Patterns
- No framework imports in core `agent_vitals/` (adapters only)
- No mutable global state
- No hardcoded thresholds outside `thresholds.yaml` / `config.py`
- No raw SQL or database access (this is a library, not a service)
- No `print()` for logging — use structured snapshot fields

---

## Testing Preferences

### Framework & Tools
- **Framework**: pytest (>=8.0)
- **Async**: pytest-asyncio (>=0.24)
- **Coverage**: pytest-cov (>=4.1), target 85%
- **Test naming**: `test_*.py`, functions `test_<what>_<condition>`

### Test Categories
- **Unit tests**: `tests/test_detection_*.py`, `tests/test_adapter_*.py`
- **Integration tests**: `tests/test_*_integration.py` (similarity, cusum, confabulation)
- **Backtest tests**: `tests/test_backtest.py`, `tests/test_ci_gate.py`
- **Export tests**: `tests/test_export.py`, `tests/test_otlp_export.py`

### Testing Requirements
- All detection logic must have both positive (fires correctly) and negative (does not false-positive) test cases
- Backtest must pass on bundled corpora before release
- Framework adapter tests must not require the framework to be installed (mock the state dicts)

---

## Sibling Projects

### metrics_and_protocols (Research Origin)
**Path**: `/Users/mac-studio/projects/unified-system-state/metrics_and_protocols`  
**Relationship**: Foundational research repo. Protocol theory, metric definitions, threshold calibration research all originate here. Has extensive CMOS history (10+ sprints, 46 missions). When we need theoretical grounding or metric design decisions, check here.

### agent-vitals-bench (Validation Harness)
**Path**: `/Users/mac-studio/projects/unified-system-state/agent-vitals-bench`  
**Relationship**: Independent validation apparatus. 862+ labeled traces across synthetic, elicited, and legacy corpora. Runs gate evaluations (Wilson CI lower bounds: P_lb >= 0.80, R_lb >= 0.75). When traces reveal detector issues, fixes flow back here to agent-vitals.

### Feedback Loop
```
metrics_and_protocols (theory) --> agent-vitals (implementation) --> agent-vitals-bench (validation)
         ^                                                                    |
         +--------------------------------------------------------------------+
                           findings inform theory refinement
```

---

## Detector Status

Canonical per-detector and per-framework precision/recall numbers live in `agent-vitals-bench`'s `eval-cross-framework-v1` artifact set, not in this file (the bench corpus is the source of truth and updates faster than this doc). As of v1.15.0, all five detectors (loop, stuck, confabulation, runaway_cost, thrash) have shipped with hybrid handcrafted+TDA paths where applicable, and a new third-layer Hopfield early-screen adapter ships behind the optional `agent-vitals[hopfield]` extra (informational `hopfield_override_active` marker, never mutates per-detector flags — bit-identical to baseline on existing detector cells). The crewai × tda composite gate is 5/5 PASS after the v1.14.1 burn_rate_multiplier revert; default-config crewai stuck remains the known limitation (architectural follow-up deferred — see AV-S08-M04). Run `python scripts/ci_backtest.py` for bundled-corpus numbers, or check bench for cross-framework gates.

---

## CMOS Integration Notes

### When Working on Application Code
1. Read THIS `agents.md` for coding standards and project context
2. Write code to `agent_vitals/` source directory
3. Write tests to `tests/` directory
4. Never write application code to `cmos/`

### When Working on CMOS Operations
1. Read `cmos/tiers/build.md` for CMOS-specific instructions
2. Use mission/sprint/session tools for project management
3. Keep application code and CMOS management separate

### Before Completing Missions
- All tests pass (`pytest`)
- Lint clean (`ruff check`)
- Type check clean (`mypy agent_vitals/`)
- Coverage gate holds (`--cov-fail-under=85`)

---

## Commit Messages
```
[type]([scope]): [description]

Types: feat, fix, refactor, test, docs, release, chore
Scopes: detection, adapters, export, backtest, config, ci

Examples:
feat(detection): add model-size-aware threshold normalization
fix(detection): resolve stuck/loop co-occurrence arbitration
test(backtest): expand av31 corpus with manual label review
release: v1.11.0 multi-model corpus and loop gate promotion
```
