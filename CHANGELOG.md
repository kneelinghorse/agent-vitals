# Changelog

All notable changes to the `agent-vitals` package.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.18.0] - 2026-04-10

### Changed
- **Trace-level stuck+runaway co-occurrence suppression** (av-s12-m01).
  Suppress `stuck_detected` at trace level when `runaway_cost` co-fires
  on the same trace.  `findings_plateau` (the dominant stuck trigger on
  runaway-positive traces) is a natural precursor to cost runaway — the
  agent's output stagnates before costs escalate.  The stuck signal is a
  symptom, not root cause.

  Eliminates 34 stuck false positives on the bench v1 corpus
  (all runaway-elicited-positive traces with the same temporal pattern:
  stuck at step N, runaway at step N+1).  Combined with 4 bench label
  corrections, brings stuck FP from 38 to 0 on default profile.
  Zero TP regressions.

  Analogous to the existing stuck+loop content-similarity suppression
  at the same call site in `_replay_trace()`.

## [1.17.0] - 2026-04-10

### Changed
- **Per-step co-occurrence suppression for runaway_cost** (av-s11-m02).
  Suppress `runaway_cost_detected` when stuck fired at the previous step
  AND stuck candidates are present at the current step (cross-detector
  leakage pattern, window=1).  Eliminates 24 of 30 remaining
  default-mode runaway_cost false positives on the bench v1 corpus
  (runaway_cost P_lb lifts from 0.832 to 0.945 on default profile).
  6 confab→runaway FPs remain (no stuck involvement, out of scope).
  Zero TP regressions (229 → 229).  Zero stuck FP waterbed.

  The suppression uses a deferred-application pattern: the flag is
  computed before the burn-rate→stuck clearing block but applied after
  it, preventing the waterbed effect where early suppression would
  disable a downstream clearing step.  The `stuck_candidates` guard
  distinguishes cross-leakage (stuck signals present at runaway step)
  from temporal coincidence (genuine runaway following a stuck step).

  New `prior_stuck_detected` keyword-only parameter on `detect_loop()`
  (default `False`, backward compatible).  Wired automatically in
  `_replay_trace()` / `replay_trace()` and `AgentVitals.record()`.

  Bench validated on full v1 corpus (commit 6707d80):
  - default: runaway FP 30→6, TP 229→229, P_lb=0.945
  - langgraph: P_lb=0.939
  - crewai: P_lb=0.939
  - dspy: P_lb=0.820 (stuck excluded, suppression not applicable)

## [1.16.0] - 2026-04-10

### Added
- **Public `replay_trace()` API** (av-s10-m02). Stable public function
  on the `backtest` module for external consumers (e.g. agent-vitals-bench).
  Replays a trace through the full detector pipeline and returns
  `{"loop": bool, "stuck": bool, "confabulation": bool, "thrash": bool,
  "runaway_cost": bool}` — the canonical five detectors, no `any`
  composite.  Signature:
  `replay_trace(snapshots, config=None, workflow_type="unknown")`.
  Bench can replace `from agent_vitals.backtest import _replay_trace`
  with `from agent_vitals.backtest import replay_trace` and drop the
  `any`-field normalization in `evaluator/runner.py`.

### Changed
- **`burn_rate_multiplier` raised from 2.5 to 3.0** (av-s10-m01).
  Eliminates 20 of 52 default-mode runaway_cost false positives on the
  bench v1 corpus (1,494 traces) without losing any true positives.
  Default-mode runaway_cost precision lifts from P_lb=0.765 to
  P_lb=0.832, clearing the 0.80 hard-gate threshold on all four
  framework profiles without requiring TDA.  Updated in:
  `DEFAULT_BURN_RATE_MULTIPLIER` (config.py), `thresholds.yaml`
  (default + langgraph profile).

  Bench validated: 0/229 TPs lost at multiplier 3.0.  Multiplier 3.5
  was explicitly ruled out (would lose 40/229 TPs, 17.5% recall drop).
  32 FPs remain (24 stuck-synthetic cross-leakage, 6 confab
  cross-leakage, 2 other) — addressable via per-step co-occurrence
  arbitration in a future release.

### Public API surface
- `agent_vitals.backtest.replay_trace` — new stable function (v1.16.0+)
- `agent_vitals.backtest._replay_trace` — retained for internal use,
  not part of the public API contract

## [1.15.0] - 2026-04-08

### Release artifacts (canonical PyPI SHA256)

- `agent_vitals-1.15.0-py3-none-any.whl` —
  `632876d5e4482d08eb1c0a9089c507f9b44160b5fec9b5f95a882f9b36df8c30`
  (5,310,344 bytes)
- `agent_vitals-1.15.0.tar.gz` —
  `3b458de64ec2887bca231f1eec951bee53828c1e4d8e1c11f37c48f906540163`
  (5,359,295 bytes)

The bench AV-S09-M03 release-acceptance message (`d349872d`) flagged
that the agent-vitals release info_push (`1d02ce2f`) carried a wheel
SHA from an intermediate working build, not the actual PyPI artifact —
that SHA is `1d9f550a88cc508a8d7ed46e23f4bcaec6f6e403192d26ff73295f583c212910`
and matches a working-tree wheel that was never uploaded. The values
above are pulled directly from PyPI's authoritative JSON API at
`https://pypi.org/pypi/agent-vitals/1.15.0/json` and verified by bench
on download. **Operators verifying the released artifact should use
the values above.**

### Added
- **Hopfield 3rd-layer early-detection adapter** (av-s09-m01,
  av-s09-m02). Bench's Five-Paradigm Comparative Report
  (intel_alert `b7416ceb-2eba-4475-b119-10346beb077b`) showed Hopfield
  is the only paradigm with meaningful signal at trace prefixes 3-5
  (macro-F1 0.901 at p3 vs handcrafted 0.466). v1.15.0 ships the
  adapter as an optional layer that early-warning consumers can
  consult inside the early-detection window where the existing
  handcrafted+TDA stack has no view.

  - **New module `agent_vitals.detection.hopfield`** —
    `HopfieldEarlyDetector` facade plus the functional `predict()` and
    `hopfield_override_fires()` entry points. Mirrors the
    `agent_vitals.detection.tda` pattern: import-guarded
    `onnxruntime`, lazy artifact load via `lru_cache`, runtime prefix
    selector (`len <= 4 → p3` else `p5`), graceful degradation when
    the optional extras are missing.
  - **10 bundled ONNX prefix-models** under
    `agent_vitals/models/hopfield/` (5 detectors × {p3, p5}, ~5.6 MB
    total) plus per-model JSON sidecars carrying the canonical
    17-feature `mean`/`std`/`feature_order`/`max_steps`/`prefix_len`
    contract from the bench inference spec. Sidecar `feature_order`
    is asserted at load time so any future bench retrain that drifts
    the schema fails loudly instead of silently scoring garbage.
  - **New `agent-vitals[hopfield]` install extra** —
    `onnxruntime>=1.17` + `numpy>=1.24.0`. Compare to torch CPU
    (~200 MB) or torch CUDA (~750 MB+): the ONNX route saves
    150-700 MB per install and keeps the base footprint lightweight
    per the direct-integration positioning.
  - **Bundled artifacts ship via the wheel package-data glob**
    (`models/hopfield/*.onnx`, `models/hopfield/*.json`) — defense in
    depth against the v1.13.0 thresholds.yaml regression and the
    v1.14.0 package-data hardening pattern.

- **Hopfield wired into `_resolve_detections` as an informational
  marker** (av-s09-m02). Apply when (a) `cfg.hopfield_enabled=True`,
  (b) `onnxruntime` backend importable, (c) trace length ∈ [3, 6],
  (d) any per-detector probability ≥
  `DEFAULT_OVERRIDE_THRESHOLDS[detector]`. The thresholds are
  calibrated against the bench v1 corpus per AV-S09-M03 acceptance
  (intel response `0f771492`): `{loop: 0.80, stuck: 0.90,
  confabulation: 0.80, thrash: 0.70, runaway_cost: 0.90}`. The
  data-driven optimum maximizes early-window recall without dropping
  marker precision below 0.91 on any detector at any cutoff. The
  marker propagates from `LoopDetectionResult` through `monitor.py`
  into the constructed `VitalsSnapshot` so early-warning consumers
  (interventions, dashboards) can read it as provenance for early
  Hopfield evidence.

  - **Marker is purely informational** — it never mutates the
    per-detector `*_detected` flags. The handcrafted+TDA stack
    remains authoritative for trace-level detection decisions, so
    full-trace `vitals.any` numbers stay bit-identical against the
    v1.14.2 baseline on every bundled corpus regardless of whether
    `hopfield_enabled` is True or False. This is the structural M02
    contract that lets the override layer ship without altering
    composite cells.
  - **Why marker, not flag-flip**: under `_replay_trace`'s any-step
    trace-level semantics, an upgrade-only flag-flipping override at
    per-step would inevitably introduce new trace-level TPs/FPs in
    cells where handcrafted+TDA later doesn't catch the trace,
    breaking the bit-identical full-trace constraint at any threshold
    short of "never fires". Routing the signal into a separate
    channel preserves both halves of the spec — the override is
    observable and the existing detector cells are unchanged.
    Flag-flipping behavior remains an explicit followup decision
    pending bench validation evidence.

  ### Public API surface
  - **New on `LoopDetectionResult` (`agent_vitals.detection.loop`)**:
    `hopfield_override_active: bool` (additive, default `False`).
  - **New on `VitalsSnapshot` (`agent_vitals.schema`)**:
    `hopfield_override_active: bool` (additive, default `False`).
  - **New on `VitalsConfig` (`agent_vitals.config`)**:
    `hopfield_enabled: bool = False` and
    `hopfield_model_dir: Optional[Path] = None`. Default `False`
    mirrors `tda_enabled` so opting in is explicit.
  - **New module `agent_vitals.detection.hopfield`** exporting:
    `HOPFIELD_DETECTORS`, `HopfieldConfig`, `HopfieldPrediction`,
    `HopfieldEarlyDetector`, `MissingHopfieldDependencyError`,
    `N_FEATURES`, `DEFAULT_OVERRIDE_THRESHOLDS`,
    `is_hopfield_available`, `predict`, `select_prefix_variant`,
    `hopfield_override_fires`.
  - **New install extra `agent-vitals[hopfield]`**.
  - **Behavior**: with `hopfield_enabled=False` (the default),
    enabling no extras, or installing the wheel without the extras,
    the public API surface is additive only — every existing call
    site continues to behave bit-identically.

  ### Bundled-corpus impact (`scripts/ci_backtest.py`)
  - **Composite, loop, confabulation, stuck, thrash, runaway_cost:
    bit-identical to v1.14.2** (P=0.992 R=0.946 F1=0.969 composite;
    loop F1=0.988, confab F1=0.811, stuck F1=0.861, thrash F1=1.000,
    runaway_cost F1=0.872). The default config does not enable
    Hopfield consultation, so the wiring is dead code on the gate
    path.
  - Standalone bench-style replay against
    `checkpoints/vitals_corpus/av31_reviewed` (289 traces, 1091
    snapshots) with `hopfield_enabled=True` vs `hopfield_enabled=False`:
    every detector cell bit-identical (composite (203,2,12,72) →
    (203,2,12,72) on both sides). Strongest possible empirical proof
    of the M02 non-mutation contract.

### Bench validation cycle (AV-S09-M03)
- **Intel response `0f771492` — ALL FOUR GATING ASKS PASS** on the
  full bench v1 corpus (1494 traces, min_confidence ≥ 0.8). Bench
  re-ran `eval-hopfield-cross-framework-v1` against the v1.15.0rc
  wheel (SHA256 `abe987fbed1c40dbf9be0a464263496843ff03dcf6969b6ac23869207c045dc8`,
  matched bench-side to the byte) under three modes:

  1. **Pass A — `hopfield_enabled=False` × all 8 cells (40 cell × detector pairs)**:
     **40/40 bit-identical to v1.14.2 baseline.** Confirms the new
     code path is fully gated when the config knob is off.
  2. **Pass B — `hopfield_enabled=True` × 4 default-mode cells (20 pairs)**:
     **20/20 bit-identical to v1.14.2 baseline** with the Hopfield
     marker actively firing on traces in the [3,6] window. This is
     the structural M02 contract verified empirically at scale on
     the canonical bench corpus, not just the bundled av31_reviewed
     checkpoint (which agent-vitals had already verified upstream).
  3. **Marker firing reproduction** vs bench's prior p3/p5 PyTorch
     reference grid: max delta **0.0005 F1** (thrash @ cutoff=3),
     max relative **0.05%** — three cells exactly identical to four
     decimals. Consistent with the **6.2e-6** ONNX/PyTorch parity
     from the M01 export. Confirms zero integration drift between
     the bench prototype and the agent-vitals public adapter API.

  The one quadrant bench did NOT run was
  `hopfield_enabled=True × tda_enabled=True` (their main venv lacks
  the TDA backend; cross-repo discipline rule blocked an ad-hoc
  install). The structural argument for why it must hold (the marker
  and TDA override live on independent fields and cannot interfere)
  combined with agent-vitals' av31_reviewed verification of the
  AND-of-both case was accepted as sufficient.

- **Per-detector override thresholds**: agent-vitals adopted bench's
  data-driven recommendation (`{loop: 0.80, stuck: 0.90, confab: 0.80,
  thrash: 0.70, runaway_cost: 0.90}`) over the rc's uniform 0.90
  starting point in this release commit. Stuck and runaway_cost are
  pinned at 0.90 because both have a precision cliff between 0.70
  and 0.80 in the bench PR curves; the other three detectors buy
  meaningful early-window recall at acceptable precision floors when
  lowered. Full PR curves at cutoffs 3 and 5 for thresholds {0.5,
  0.6, 0.7, 0.8, 0.9, 0.95} live in the bench acceptance message.

- p7 prefix variant deferred to a future S10+ mission per
  `project_hopfield_p7_retrain_backlog.md` (training-procedure
  artifact diagnosed by bench, 10-line fix offered for retrain).

## [1.14.2] - 2026-04-08

### Changed
- **Replace implicit `burn_rate_anomaly → stuck` suppression with
  explicit logic** (av-s08-m04). The pre-v1.14.2 detection chain used
  the `stuck_trigger="burn_rate_anomaly"` string as a sentinel
  protocol: `loop._collect_stuck_candidates` appended burn-rate as a
  stuck candidate with confidence 1.0, it won arbitration via that
  sentinel, and downstream consumers (`stop_rule.derive_stop_signals`,
  `backtest._replay_trace`) filtered/converted on the magic string.
  This implicit chain made per-profile `burn_rate_multiplier` overrides
  a footgun (the v1.14.1 crewai regression).

  The refactor:
  - Adds `runaway_cost_detected: bool` and `runaway_cost_confidence:
    float` as explicit fields on `LoopDetectionResult` and
    `VitalsSnapshot`.
  - Removes the `burn_rate_anomaly` append from
    `_detect_stuck_candidates`. Burn-rate runaway is now computed
    independently in `_compute_burn_rate_runaway` and carried as the
    explicit field.
  - Replaces the implicit "conf=1.0 wins arbitration" suppressor with
    an explicit `stuck_candidates = []` clear in `_resolve_detections`,
    gated on the same conditions the old code's filters effectively
    gated it on (no error_count, no high-confidence loop+content_sim).
  - `_handle_stuck_disabled` now sets `runaway_cost_detected=True,
    stuck_detected=False, stuck_trigger=None` instead of the
    `stuck_detected=True, stuck_trigger="burn_rate_anomaly"` sentinel.
  - `backtest._replay_trace` reads `detection.stuck_detected` directly
    and passes `runaway_cost_detected` through to
    `derive_stop_signals` — the magic-string filter is gone.
  - `derive_stop_signals` keeps a legacy fallback for snapshots that
    still emit the old sentinel (external producers, older callers),
    documented inline for eventual removal.

  ### Public API surface
  - **New on `LoopDetectionResult` (`agent_vitals.detection.loop`)**:
    `runaway_cost_detected: bool`, `runaway_cost_confidence: float`.
  - **New on `VitalsSnapshot` (`agent_vitals.schema`)**:
    `runaway_cost_detected: bool`, `runaway_cost_confidence: float`.
  - **Behavior change on `LoopDetectionResult`**: when burn-rate
    runaway fires, `stuck_trigger` is now `None` instead of the
    `"burn_rate_anomaly"` sentinel. Consumers that read
    `stuck_trigger == "burn_rate_anomaly"` should switch to
    `runaway_cost_detected`. The legacy sentinel path in
    `derive_stop_signals` remains for back-compat.

  ### Bundled-corpus impact (`scripts/ci_backtest.py`)
  - loop / confabulation / stuck / thrash: bit-identical to v1.14.1.
  - **runaway_cost: TP 12 → 17, FN 7 → 2, FP unchanged at 3**
    (R 0.632 → 0.895, P 0.800 → 0.850, F1 0.706 → 0.872). The
    refactor recovers 5 traces where the old code silently dropped
    the runaway signal because the error/loop-suppression filters
    stripped `burn_rate_anomaly` from arbitration before it could
    win — and the magic-string protocol only kicked in when the
    sentinel survived. Composite gate still PASSES (P=0.992 R=0.946
    F1=0.969).

### Notes for bench
- Re-run of `eval-cross-framework-v1` is recommended. Loop / stuck /
  confabulation / thrash cells should remain bit-identical to v1.14.1.
  Runaway_cost cells may shift in the strictly-better direction
  (higher recall, same or lower FPs) on traces that hit the
  error/loop arbitration filters — this is the explicit recovery of
  signal that was silently lost pre-v1.14.2, not a regression.
- The crewai burn_rate_multiplier override regression test is still
  pinned in `tests/test_threshold_profiles.py`. The arbitration
  trade-off described in the v1.14.1 entry remains real; only the
  *implementation* of the suppression is now explicit and traceable
  in code without needing this CHANGELOG footnote.

## [1.14.1] - 2026-04-08

### Fixed
- **Crewai profile: remove `burn_rate_multiplier: 3.0` override**
  (av-s08-m01). A prior tune bumped the crewai threshold to 3.0
  targeting runaway_cost numbers, but the change silently disabled a
  stuck-suppression side-channel and produced **34 stuck FPs** on short
  elicited runaway-positive traces in bench's corpus
  (`eval-cross-framework-v1.md`: crewai stuck P_lb=0.7022, NO-GO).
  The crewai profile now inherits the default `burn_rate_multiplier=2.5`.

  **Mechanism.** `loop._collect_stuck_candidates()` appends
  `burn_rate_anomaly` as a stuck candidate with confidence 1.0 whenever
  the burn-rate gate passes. `backtest._replay_trace` filters out stuck
  whose winning trigger is `burn_rate_anomaly` (it's really a
  runaway_cost signal). Because confidence 1.0 beats every other stuck
  candidate in arbitration, `burn_rate_anomaly` wins, then gets masked
  by the replay filter — this double-ness acts as an **implicit
  stuck suppressor** on short runaway-positive traces. Raising
  `burn_rate_multiplier` above what the corpus's burn-rate spikes can
  clear stops the anomaly from firing, lets a lower-confidence
  non-burn_rate_anomaly stuck candidate win arbitration, and the replay
  filter no longer masks it — producing stuck FPs.

  **Impact on bench cross-framework gate (eval-cross-framework-v1):**
  - `crewai × tda` composite: **4/5 NO-GO → 5/5 PASS** ✓
  - `crewai stuck`: FP 35 → 1, P_lb 0.7022 → 0.9547 (clears 0.80 gate)
  - `crewai runaway_cost × tda`: FP 5 → 11, P_lb 0.9455 → 0.9111 (still
    clears; TDA adjudicator absorbs the extra FPs from the lower
    threshold). Under non-TDA mode, runaway_cost regresses to NO-GO —
    this is a documented trade-off that makes TDA load-bearing for
    crewai's paper-publishable composite PASS.
  - All other seven `(profile × mode)` cells bit-identical to v1.14.0.

### Added
- Regression guards pinning the fix:
  - `tests/test_threshold_profiles.py::test_crewai_profile_does_not_override_burn_rate_multiplier`
    — config-level pin on the shipped YAML so any re-addition of the
    override trips a test.
  - `tests/test_backtest.py::test_burn_rate_anomaly_trigger_does_not_fire_stuck`
    — replay-layer filter pin documenting the `burn_rate_anomaly`
    stuck-suppression side-channel so the filter cannot be weakened
    without an explicit review.
- Inline documentation of the trap in `agent_vitals/backtest.py`
  adjacent to the stuck_trigger filter and in
  `agent_vitals/thresholds.yaml` adjacent to the crewai profile block,
  so the next person to tune a profile threshold knows what to verify.

### Known Issues
- The implicit `burn_rate_anomaly → stuck` suppression side-channel is
  a latent architectural anti-pattern: any future per-profile
  `burn_rate_multiplier` override can unknowingly re-trip it. Replacing
  the implicit suppression with explicit logic is queued for a future
  sprint — not urgent, but worth doing before adding new per-profile
  burn-rate tuning.

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
