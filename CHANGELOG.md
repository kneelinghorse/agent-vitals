# Changelog

All notable changes to the `agent-vitals` package.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2026-02-13

### Added
- First-class backtest scoring for `confabulation` across the standalone and DeepSearch backtest stacks.
- Real confabulation corpus expansion (`av26_real`): added `AV29.C01` and `AV29.C05` with manual labels and onset metadata.
- CI backtest gate wrapper (`scripts/ci_backtest.py`) with JSON artifact output and per-detector CI annotations.
- Bundled CI corpora under `checkpoints/vitals_corpus/av05_synth` and `checkpoints/vitals_corpus/av26_real` for self-contained CI execution.

### Changed
- `scripts/backtest_av26.py` now reads `confabulation_from_loop` and reports confabulation P/R/F1 as a first-class detector (no proxy evaluation).
- Detector tables and per-trace reports now include `confabulation` alongside loop/stuck/thrash/runaway_cost.
- GitHub Actions CI (`.github/workflows/ci.yml`) now runs backtest gating on Python 3.11 after tests and uploads `backtest-results.json`.

### Validation
- Three-way backtest (synthetic 33 + real 16, combined 49 traces):
  - `vitals.any` combined `P/R/F1 = 1.000/1.000/1.000`
  - `confabulation` combined `P/R/F1 = 1.000/1.000/1.000` (first-class labels)
  - `loop` combined `P/R/F1 = 0.900/1.000/0.947`
  - `stuck` combined `P/R/F1 = 0.550/0.786/0.647`
- CI wrapper runtime on bundled corpus: ~`0.208s` (well below 60s target).

### Notes
- Release hygiene decision for this cut: skip retro-tagging `v1.6.0`/`v1.7.0` and proceed with `v1.8.0`.

## [1.7.0] - 2026-02-13

### Added
- Adaptive SPC thresholding (`AdaptiveThreshold`) integrated into loop/stuck confabulation signals.
- Confabulation detector promoted to first-class output with:
  - `source_finding_ratio` floor/trajectory signals,
  - multi-signal confidence aggregation,
  - explicit `confabulation_detected`, `confabulation_confidence`, `confabulation_trigger`, and `confabulation_signals` snapshot fields.
- Detector-priority routing updated to support confabulation-first arbitration in both runtime and backtest composite paths.

### Changed
- Loop/stuck windows are trace-length proportional (`loop_consecutive_pct`, `findings_plateau_pct`) to improve short-run behavior.
- `scripts/backtest_av26.py` now explicitly validates full corpus behavior with no `min_run_steps` filter.
- Calibration fixtures and label mappings refreshed for current detector taxonomy.

### Fixed
- Build retry vitals enforcement regression in DeepSearch integration (retry-loop detection now triggers early-stop deterministically).
- Local mono-repo import fallback for `agent_vitals` in DeepSearch shim environments where editable install is missing.
- Full DeepSearch test suite tech debt cleanup (pre-existing failures triaged and resolved).

### Validation
- Full DeepSearch suite: `1466 passed, 12 skipped`.
- Three-way backtest (full corpus, no min-run filter) completed with `vitals.any` combined `P/R/F1 = 1.000/1.000/1.000`.

## [1.6.0] - 2026-02-13

### Added
- `sources_stagnation` loop trigger for low-source confabulation patterns (including `unique_domains_stagnation` boost).
- `detector_priority` metadata on loop detection results and vitals snapshots for explicit detector winner tracking.

### Changed
- Stuck detector precision hardening:
  - lowered `stuck_cv_threshold` from `0.5` to `0.3`,
  - widened findings-plateau window to `4`,
  - added loop-signal-aware suppression for stagnation-style stuck paths,
  - retained low-coverage exceptions to avoid suppressing true stuck failures.
- Detector arbitration now applies mutual exclusion with confidence-aware winner selection when loop and stuck both fire.
- DeepSearch integration now records rolling assistant output text history so content similarity is evaluated in real mission loops.

### Fixed
- Content similarity probe wiring gap in DeepSearch (`output_similarity`/`output_fingerprint` now populated in emitted snapshots).
- False positive behavior on `AV26.R06` while preserving `AV26.F02` confabulation detection.

### Validation
- Combined corpus backtest: loop `P/R/F1 = 1.000/1.000/1.000`, stuck `P/R/F1 = 1.000/1.000/1.000`, `vitals.any P/R/F1 = 1.000/1.000/1.000`.

## [1.5.0] - 2026-02-11

### Added
- **Content-based loop detection**: Word-level Jaccard similarity (`compute_pairwise_similarity`) compares agent output text across iterations. `content_similarity` trigger fires when `output_similarity >= loop_similarity_threshold` (default 0.8).
- **Output fingerprinting**: `compute_output_fingerprint()` produces content hashes for deduplication.
- `output_similarity` and `output_fingerprint` fields on `VitalsSnapshot`.
- **Per-adapter threshold profiles**: Framework-specific threshold overrides via `thresholds.yaml` profiles section. Ships with LangGraph, CrewAI, and DSPy profiles.
- `VitalsConfig.for_framework(name)` method to apply framework-specific overrides.
- `similarity.py` module with tokenization, fingerprinting, and Jaccard computation.
- Integration tests for similarity pipeline (`test_similarity_integration.py`).

### Changed
- Loop detector now evaluates `content_similarity` as an independent signal alongside `findings_plateau`.
- `VitalsConfig` loads profiles from `thresholds.yaml` during initialization.

### Known Issues
- DeepSearch probe does not pass `output_text` to vitals; `content_similarity` trigger is inoperative in DeepSearch workflows. Standalone library usage via `AgentVitals.step(output_text=...)` works correctly.

## [1.4.0] - 2026-02-09

### Added
- **Langfuse adapter**: `LangfuseVitalsCallback` for Langfuse observability integration.
- **LangSmith adapter**: `LangSmithVitalsCallback` for LangSmith platform integration.
- Adapter tests for Langfuse and LangSmith.

### Changed
- DeepSearch migrated to depend on published `agent-vitals` package (eliminated dual-location detection code).
- Backtest corpus expanded with additional loop/stuck disambiguation traces.

## [1.3.0] - 2026-02-09

### Added
- **DSPy adapter**: `DSPyVitalsCallback` for DSPy optimization framework.
- **Haystack adapter**: `HaystackVitalsCallback` for Haystack pipeline framework.
- Synthetic cross-agent traces for DSPy and Haystack frameworks.

### Changed
- Loop detector precision improved via refined `findings_plateau` signal.
- Stuck detector precision improved via `coverage_stagnation` window adjustment.
- Cross-agent backtest expanded to cover DSPy and Haystack traces.

## [1.2.0] - 2026-02-09

### Added
- **OTLP exporter**: Export vitals snapshots as OpenTelemetry spans for Datadog/Grafana compatibility.
- **CrewAI adapter**: `CrewAIVitalsCallback` for CrewAI multi-agent framework.
- **AutoGen adapter**: `AutoGenVitalsCallback` for AutoGen framework.
- CI coverage gate (minimum 80% line coverage).
- Cross-agent backtest validation for CrewAI and AutoGen traces.

### Changed
- Thrash detector tuned for multi-agent handoff patterns.
- CI pipeline hardened with lint + type check + coverage gates.

## [1.1.0] - 2026-02-08

### Added
- **LangChain adapter**: `LangChainVitalsCallback` for LangChain LCEL chains.
- **LangGraph adapter**: `LangGraphVitalsCallback` with built-in state integration.
- Callback-based API for framework integration.
- GitHub Actions CI pipeline for standalone repo.

### Changed
- `AgentVitals` monitor class accepts `framework` parameter for adapter selection.

## [1.0.0] - 2026-02-08

### Added
- Standalone `agent-vitals` PyPI package extracted from DeepSearch.
- `AgentVitals` monitor class with `step()` API.
- Four detectors: loop (`findings_plateau`), stuck (`coverage_stagnation`, `zero_progress`, `findings_plateau`), thrash (`error_count`), runaway_cost (`burn_rate_anomaly`).
- `VitalsSnapshot` Pydantic model with signals, metrics, and detection state.
- `VitalsConfig` with YAML + environment variable configuration.
- `TemporalMetrics` engine: CV, DM, temporal hysteresis.
- `derive_stop_signals()` for enforcement integration.
- Backtest harness: `load_dataset()`, `run_backtest()`, `BacktestReport`.
- JSONL export for trace recording.
- `thresholds.yaml` for threshold configuration.
- Synthetic corpus (av05_synth, 24 traces) for regression testing.
- DeepSearch compatibility shim.
- 90%+ test coverage.

### Notes
- Initial release based on extraction from DeepSearch vitals subsystem (sprints av-01 through av-19).
- Evaluation: vitals.any P=1.000 R=1.000 on synthetic corpus; P=0.955 R=1.000 on cross-agent corpus.
