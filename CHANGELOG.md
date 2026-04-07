# Changelog

All notable changes to the `agent-vitals` package.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.14.0] - 2026-04-07

### Added
- **Profile introspection API on `VitalsConfig`** for external verifiers
  and agent integrators (av-s07-m01). Stable, agent-friendly surface
  for verifying packaging integrity and inspecting framework profile
  divergence — designed so external gates never need to poke at
  `dataclasses.fields()` over `VitalsConfig` internals. New surface:
  - `VitalsConfig.thresholds_yaml_path() -> Path` — install path of the
    bundled `thresholds.yaml`.
  - `VitalsConfig.is_yaml_loaded() -> bool` — `True` iff the bundled
    YAML exists and parses as a mapping. Canonical signal for the
    v1.13.0-style packaging regression where the wheel ships without
    its YAML data file.
  - `VitalsConfig.assert_profiles_loaded() -> None` — fail-loud
    one-line gate. Raises `ConfigurationError` when the YAML is
    missing or defines no framework profiles.
  - `VitalsConfig.list_profiles() -> tuple[str, ...]` — sorted names
    of every profile in the bundled YAML. Empty tuple when missing.
  - `VitalsConfig.profiles() -> tuple[str, ...]` — sorted names of
    profiles attached to *this* config instance (for custom YAML /
    `from_dict` callers).
  - `VitalsConfig.profile_diff(framework) -> dict[str, ProfileFieldDiff]`
    — every field where the named profile differs from pure
    `VitalsConfig()` defaults. Comparison anchor is dataclass
    defaults, **not** `self`, so the diff is reproducible across any
    caller's environment-variable configuration. Case-insensitive
    framework lookup. Raises `UnknownProfileError` (subclass of
    `ConfigurationError`) with the known-profile list in the message.
- New types `ProfileFieldDiff` (frozen dataclass: `field`, `default`,
  `override`) and `UnknownProfileError`, both re-exported from
  the `agent_vitals` top-level.

### Changed
- **Packaging hardening**: `[tool.setuptools.package-data]` now uses a
  `"*.yaml"` glob in addition to the explicit `thresholds.yaml` entry,
  so any future YAML data file is bundled by default. Defense in
  depth alongside the new introspection API.

### Fixed
- Detector logic, framework profiles, and TDA bundled model are
  bit-identical to v1.13.1. This release adds an API and tightens
  packaging; it does not change any detector behavior.

## [1.13.1] - 2026-04-07

### Fixed
- **`thresholds.yaml` missing from the v1.13.0 wheel** (packaging
  regression). Adding `[tool.setuptools.package-data]` in v1.13.0 to
  bundle `models/*.joblib` displaced setuptools' default data
  discovery, which had been implicitly bundling `thresholds.yaml`.
  Default-config users were unaffected (`VitalsConfig.from_yaml`
  graceful-degrades to dataclass defaults), but
  `VitalsConfig.for_framework("langgraph"|"crewai"|"dspy")` silently
  returned the default config instead of the framework-specific
  threshold overrides tuned across Sprints 9–14. Caught by bench
  smoke-test of the released wheel.
- One-line fix in `pyproject.toml`:
  ```toml
  [tool.setuptools.package-data]
  agent_vitals = ["models/*.joblib", "thresholds.yaml"]
  ```
- No code changes; detector logic is bit-identical to v1.13.0.
  Framework-profile users on v1.13.0 should upgrade.

## [1.13.0] - 2026-04-07

### Added
- **Hybrid runaway cost TDA detector** (`agent_vitals/detection/tda.py`):
  optional second-layer adjudicator for the handcrafted burn-rate detector.
  When enabled and the optional TDA backend is available, runs a 663-dim
  topological-data-analysis feature extraction (sliding-window persistent
  homology via `giotto-tda`) and a bundled GradientBoosting classifier to
  confirm or override the handcrafted verdict on the trace level.
  - Override-only / minimum-blast-radius: per-snapshot
    `runaway_cost_detected` flags are unchanged for early-warning consumers,
    only the trace label flips. Standalone TDA F1 = 1.000 on the eligible
    subset (1195 traces); zero wrong-direction overrides.
- **Bundled model artifact** (`agent_vitals/models/runaway_cost.joblib`):
  ships in the wheel via `[tool.setuptools.package-data]`, no separate
  download required.
- **`tda` optional dependency group** in `pyproject.toml`:
  `giotto-tda>=0.6.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`,
  `numpy>=1.24.0`. Requires Python ≤ 3.12. Base install is unchanged.
- **`VitalsConfig.tda_enabled`** (default `False`) and
  **`VitalsConfig.tda_model_path`** (default `None` → bundled artifact).

### Changed
- **Final-step adjudication for causal confabulation** in
  `backtest._replay_trace`: captures the last loop iteration's
  `confabulation_detected` flag (which is the bench-equivalent one-shot
  verdict — the call already passes `snapshots[-1]` with `snapshots[:-1]`
  as history) and uses it to override the streaming any-step aggregation.
  When the per-step aggregator says "fired" but the final-step verdict
  says "no", the trace label flips to `False`. Per-snapshot fields are
  unchanged.
- **Removed vestigial `latest_link` / `latest_verified` gates** from
  Paths 1, 2, and 3 of `_detect_causal_confabulation`. They were added in
  v1.12.0 as per-step transient suppressors but became redundant once the
  trace-level final-step adjudication landed and were costing 71 valid TPs
  on short synthetic confabs.

### Validation (bench v1, 1494 traces)
- **Causal confabulation** (M01, c39ae64):
  - F1: 0.944 → **0.9495** (exact bench prototype parity)
  - P: 0.954 → **0.986**
  - R: 0.935 → **0.9156**
  - FP: 14 → **4** (–10)
  - Per-trace diff against bench prototype: zero divergent traces
- **Runaway cost** (M02, 2fd3a25, TDA enabled):
  - F1: 0.9424 → **0.9765** (+0.034)
  - P: 0.891 → **0.9542** (+0.063)
  - R: **1.000** (preserved)
  - FP: 28 → **11** (–17)
  - Standalone TDA on eligible subset: **F1 = 1.000** (TP=199, FP=0)
- **Other detectors** (loop, stuck, thrash): parity with v1.12.0
- **Composite gate: PASS** in both TDA-on and TDA-off configurations
- **Graceful degradation**: with `tda_enabled=True` but the optional TDA
  backend not installed, the lazy loader probes correctly, the override
  skips silently, and runaway_cost metrics fall back bit-identically to
  the handcrafted-only baseline. Zero exceptions, zero overhead.

### Architecture Notes
- The two changes share the same minimum-blast-radius pattern: a
  trace-level override applied after the per-step replay loop, leaving
  per-snapshot fields untouched. `_replay_trace` owns trace-label
  semantics; per-call detectors (`_detect_causal_confabulation`,
  `predict_runaway_cost`) own per-call detection logic free of
  streaming-mode workarounds.
- TDA's full-corpus precision is bounded by where the model can speak.
  The override-only architecture by construction never adds TPs and
  never overrides TPs to FNs, so 11 length-6 traces remain as
  handcrafted FPs (TDA-blind, not TDA-wrong). Effective `min_steps` for
  `window_sizes=(3, 4, 5)` is 7, documented at the entry point in
  `TDAConfig.min_steps`.

## [1.12.0] - 2026-04-06

### Added
- **Causal confabulation detector** (`_detect_causal_confabulation`): replaces
  the SFR-threshold confab detector with a structural detector that monitors
  rolling partial correlation between findings and sources growth, residualized
  against token spend. Three detection paths:
  - **Path 1** (`causal_link_break`): healthy early baseline breaks later in the
    trace, indicating findings/sources decoupling.
  - **Path 2** (`persistent_low_causal_link`): coupling never established from a
    small bootstrap.
  - **Path 3** (`verified_source_decoupling`): verified sources persistently
    lag total source growth — catches real LLM confabulation via DOI
    verification (when `verified_sources_count` is populated).
- **Windowed ratio decline helper** (`_windowed_ratio_declines`): tolerates
  non-consecutive declining steps within a sliding window. Catches V-shaped
  recovery patterns and threshold-boundary bouncing that broke the legacy
  consecutive-decline requirement.
- 13 new config parameters for causal confab tuning
  (`causal_confab_window_size`, `baseline_floor`, `weak_link_threshold`,
  `structural_drop_threshold`, `ratio_gate`, `low_link_threshold`,
  `source_bootstrap_cap`, `low_link_ratio_gate`, `verified_link_floor`,
  `verified_weak_threshold`, `verified_drop_threshold`, `verified_ratio_gate`,
  `verified_min_sources`) plumbed through env / yaml / dict / profile paths.

### Changed
- `_detect_confabulation_candidates` integration: when the causal detector is
  *eligible* (sufficient history + source data) it owns the verdict — the
  legacy SFR-threshold path is suppressed entirely. The legacy path runs only
  as a fallback for short traces or no-source-data cases.
- Streaming-mode safeguard: Paths 1 and 2 require the latest window to confirm
  the weak-link condition, preventing transient per-step FPs from sticking via
  any-step trace label aggregation.

### Validation (bench v1, 1494 traces)
- Confab F1: 0.860 → 0.944 (+0.084)
- Confab P_lb: 0.812 → 0.954 (+0.142)
- Confab R_lb: 0.821 → 0.935 (+0.114)
- Confab FP: 45 → 14 (-31)
- All other detectors at parity
- Composite gate: PASS

### Known Limitations
- Path 3 streaming-vs-one-shot semantics gap: 10 FPs remain vs the bench
  one-shot reference because verified counts can rebound mid-trace
  (LLM cites a real paper after fabricated ones), but the streaming detector
  fires at the early-weak step and trace-level any-step aggregation makes
  the FP stick. Closing this requires final-step adjudication in
  `_replay_trace` (planned for S06).

## [1.11.0] - 2026-04-05

### Added
- **Verified source ratio confabulation signal** (`verified_source_ratio`):
  new `verified_sources_count` and `unverified_sources_count` fields on `RawSignals`
  enable ground-truth confabulation detection for frontier models (claude-sonnet,
  gpt-4o) that defeat the existing `source_finding_ratio` heuristic by producing
  fake sources in high volume.
- **Model-size-aware signal normalization** (`signal_mapping.py`): auto-classify
  traces as small/medium/large based on token variance, with per-size threshold
  adjustments. `burn_rate_multiplier_scale` applied in stuck-enabled path for
  FP protection on medium/small models.
- **Stratified corpus utilities** (`stratified.py`): helpers for corpus
  stratification and per-profile backtest evaluation.
- **Stop-rule signal derivation** (`stop_rule.py`): `derive_stop_signals()`
  extracts actionable stop signals from `VitalsSnapshot` for downstream consumers.

### Changed
- Backtest evaluation now infers per-trace workflow from corpus ids (`.bc.`/`.rc.`)
  so mixed AV-31 corpora score build and research traces with the correct stuck-detection mode.
- `_handle_stuck_disabled` now uses min-baseline mode for `burn_rate_anomaly`
  confidence, preventing spike contamination on multi-step DSPy traces.
  (S03 burn_rate regression fix.)
- `burn_rate_multiplier_scale` is path-specific: applied in `_detect_stuck_candidates`
  (stuck-enabled) for FP protection, NOT applied in `_handle_stuck_disabled`
  (stuck-disabled) to preserve recall.

### Fixed
- **AV-32 stuck/loop co-occurrence**: added a narrow `short_run_zero_coverage`
  stuck signal for research traces that hit repeated-output, zero-coverage stalls,
  and preserved that signal through loop arbitration via `detector_priority`.
- **burn_rate_anomaly regression** (S03): removed speculative `burn_rate_multiplier_scale`
  from `_prepare_context()` that doubled the threshold on small-model traces,
  causing 42 FNs on DSPy runaway_cost. Added min-baseline mode and path-specific
  scale to resolve without regressing LangGraph.
- **DSPy runaway_cost co-occurrence FPs** (S04): ported cross-detector arbitration
  from `_resolve_detections` to `_handle_stuck_disabled`. Four suppression rules:
  relaxed stagnation suppresses runaway, confabulation overlap, loop-runaway
  confidence arbitration (content_similarity wins unconditionally), error-count
  suppression. DSPy P_lb 0.693 → 0.814 on 862-trace bench corpus.
- DeepSearch backtest imports now delegate to the shared `agent_vitals.backtest`
  implementation so package CI, legacy DeepSearch tests, and the AV-31 report
  all evaluate the same replay logic.

### Validation
- All 4 framework profiles pass all enabled gates on 862-trace bench corpus.
- DSPy runaway_cost: P_lb=0.814 (was 0.693), R_lb=0.973 (unchanged).
- Default: P_lb=0.832, CrewAI: P_lb=0.940, LangGraph: P_lb=0.808 (no regressions).
- Loop hard gate maintained: P=0.986 [0.960], R=1.000 [0.982].
- Local: 505 tests, 90% coverage, lint clean.

## [1.10.0] - 2026-03-14

### Added
- **AV-31 multi-model corpus**: 373 manually-reviewed traces across 12 model providers
  (DeepSeek, Claude Sonnet, Claude Haiku, GPT-4o, GPT-4o-mini, Gemini Flash,
  Llama 70B, Llama 8B, Mistral Large, Mixtral, Qwen 72B, Command-R+).
- Manual label review pipeline (`scripts/manual_label_review.py`) with systematic
  heuristic review of all predicted positives, detector disagreement resolution,
  and 15-20% random negative sampling for hidden FN detection.
- Bundled av31_reviewed corpus (289 traces) for CI backtest validation.
- Full backtest report with per-detector 95% Wilson CI and three-way split analysis
  (build-only, research-only, combined).
- Gate promotion decision document with per-detector gap analysis.

### Changed
- **Loop detector promoted to hard CI gate**: P=0.986 [CI lower: 0.960],
  R=1.000 [CI lower: 0.982] on 370-trace combined corpus. First detector
  to meet all promotion thresholds (>=8 positives, P_lower >= 0.80, R_lower >= 0.75).
- `scripts/ci_backtest.py` now loads av31_reviewed corpus alongside existing
  av05_synth and av26_real corpora for expanded CI validation (370 traces total).

### Fixed
- **Thrash detector false positives**: All 59 thrash predictions in av31 corpus
  reclassified as FP — caused by single-step segmentation artifacts, not real
  thrash behavior. Thrash detector needs segmentation artifact filtering
  before gate promotion consideration.

### Validation
- Full suite: 370 traces, composite `vitals.any` P=0.919 R=0.986 (PASS).
- Loop hard gate: P=0.986 [0.960], R=1.000 [0.982] (PASS).
- Remaining soft gates: confabulation (P_lower=0.790, needs 0.800),
  stuck (suppressed by loop co-occurrence), thrash (FP from segmentation),
  runaway_cost (insufficient positives).
- Manual label review: 20.9% reclassification rate (78/373 entries).
  Dominant pattern: 59 thrash FPs from single-step segments.

### Notes
- Stuck detector recall is very low (0.8%) on the backtest because the
  engine suppresses stuck when loop co-fires. Most stuck-labeled traces
  also have loop labels; loop subsumes stuck in practice.
- Confabulation is 1 percentage point short of hard gate promotion
  (P_lower=0.790 vs 0.800 threshold). 2 more TP traces would likely qualify it.

## [1.9.0] - 2026-02-14

### Added
- Unfiltered three-way validation artifacts for the 81-trace corpus:
  - `docs/vitals/av30-m04-unfiltered-backtest.json`
  - `docs/vitals/av30-m04-unfiltered-backtest.md`
- Per-detector 95% Wilson confidence interval reporting for backtest decision-making.
- Per-detector CI gate policy helpers in `agent_vitals/ci_gate.py`.
- CI gate promotion decision artifacts:
  - `docs/vitals/av30-m05-ci-backtest.json`
  - `docs/vitals/av30-m05-gate-promotion.md`

### Changed
- `scripts/ci_backtest.py` now:
  - computes per-detector CI metrics,
  - applies promotion criteria (`>=8` positives, precision LB `>=0.80`, recall LB `>=0.75`),
  - emits hard/soft detector gate decisions in JSON,
  - enforces hard detector gates when detectors qualify.
- `.github/workflows/ci.yml` now passes explicit composite and detector-promotion thresholds to the backtest gate step.

### Validation
- Full suite: `pytest -q tests agent-vitals/tests` (`1474 passed`, `12 skipped`).
- CI backtest gate on full unfiltered corpus (`81` traces):
  - Composite `vitals.any` hard gate: `P=1.000`, `R=0.968` (PASS).
  - Per-detector hard promotions: none (all remain soft with statistical rationale).

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
