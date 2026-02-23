# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients (alpha/beta) or analytical roofline estimates.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Build and Run Commands

```bash
# Build
go build -o simulation_worker main.go

# Run with default model
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

## Testing

```bash
# Run all tests
go test ./...

# Run tests in a specific package
go test ./sim/...

# Run a single test by name
go test ./sim/... -run TestKVCache

# Run tests with verbose output
go test -v ./...

# Run tests with coverage
go test -cover ./...
```

## Code Architecture

### Core Simulation Engine (`sim/`)

The simulator uses a discrete-event architecture with a min-heap event queue:

- **config.go**: Module-scoped sub-config types (`KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig`) — composed into `SimConfig` via embedding (R16)
- **simulator.go**: `SimConfig` struct (composed of 6 embedded sub-configs + Horizon/Seed), `NewSimulator(SimConfig) (*Simulator, error)` constructor, `Simulator` struct and event loop (`Run()`), batch formation (delegated to `BatchFormation` interface), step execution with phased metric recording (`recordQueueSnapshots`, `recordKVUsageMetrics`, `recordRequestCompletion`), observation methods (`QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()`)
- **admission.go**: `AdmissionPolicy` interface (accepts `*RouterState`), `AlwaysAdmit`, `TokenBucket`, `RejectAll`, `NewAdmissionPolicy` factory
- **routing.go**: `RoutingPolicy` interface (accepts `*RouterState`), `RoutingSnapshot` (with `EffectiveLoad()` for canonical load calculation), `RoutingDecision` (with `Priority` hint), `RoundRobin`, `LeastLoaded`, `WeightedScoring` (composable scorer pipeline), `PrefixAffinity`, `AlwaysBusiest` templates, `NewRoutingPolicy` factory
- **routing_scorers.go**: `ScorerConfig`, scorer implementations (queue-depth, kv-utilization, load-balance), `ParseScorerConfigs`, `IsValidScorer`, `DefaultScorerConfigs`, `newScorerWithObserver` factory
- **routing_prefix_scorer.go**: Prefix-affinity scorer with router-side cache — proportional prefix match scoring via `PrefixCacheIndex`, observer hook for post-routing state updates
- **prefix_cache_index.go**: `PrefixCacheIndex` — per-instance LRU cache of hierarchical block hashes for router-side prefix matching
- **priority.go**: `PriorityPolicy` interface with `ConstantPriority`, `SLOBasedPriority`, and `InvertedSLO` templates, `NewPriorityPolicy` factory
- **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`, and `ReversePriority` templates, `NewScheduler` factory
- **latency_model.go**: `LatencyModel` interface (5 methods: StepTime, QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime), `BlackboxLatencyModel` (alpha/beta regression), `RooflineLatencyModel` (analytical FLOPs/bandwidth), `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)` factory
- **router_state.go**: `RouterState` bridge type (Snapshots + Clock) for cluster-level policy interfaces
- **bundle.go**: `PolicyBundle` struct with YAML loading (`LoadPolicyBundle`), validation (`Validate`)
- **event.go**: Event types (`ArrivalEvent`, `QueuedEvent`, `StepEvent`, `ScheduledEvent`, `RequestLeftEvent`, `PreemptionEvent`)
- **request.go**: `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`), Request lifecycle and state machine, `Priority` field for scheduler-aware ordering, `AssignedInstance` for cluster routing provenance (#181)
- **kvcache.go**: Block-based KV cache with LRU eviction and prefix caching, `CacheHits`/`CacheMisses` counters, transactional `AllocateKVBlocks` with `rollbackAllocation` on mid-loop failure
- **kv_store.go**: `KVStore` interface (11 methods: +`SetClock`, +`ConsumePendingTransferLatency`), `NewKVStore(KVCacheConfig)` factory with input validation (returns single-tier or tiered based on config)
- **kvcache_tiered.go**: `TieredKVCache` (GPU+CPU composition), `cpuTier`, `offloadedBlock`, offload/reload/transfer latency, `PendingTransferLatency()` (pure query), `ConsumePendingTransferLatency()` (read-and-clear)
- **batch.go**: Batch struct (group of requests processed in a single forward pass)
- **batch_formation.go**: `BatchFormation` interface, `BatchContext`/`BatchResult` types, `VLLMBatchFormation` (FCFS + chunked-prefill + preemption), `NewBatchFormation(LatencyModel)` factory
- **queue.go**: FIFO wait queue for pending requests

### Cluster Simulation (`sim/cluster/`)

Multi-replica extension using composition over the single-instance simulator:

- **instance.go**: `InstanceSimulator` wraps `sim.Simulator` via `NewInstanceSimulator(id, SimConfig)` with run-once guard; delegates to Simulator observation methods (`QueueDepth()`, `BatchSize()`, etc.) instead of direct field access
- **cluster.go**: `ClusterSimulator` orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation; `Run()` returns `error`
- **metrics.go**: `RawMetrics`, `Distribution`, `FitnessResult`, `CollectRawMetrics` (accepts `priorityPolicy` to suppress false-positive inversion detection for constant priority), `ComputeFitness` (returns `(FitnessResult, error)` — fails on unknown keys), anomaly detection, `ParseFitnessWeights` with NaN/Inf validation
- **deployment.go**: `DeploymentConfig` embeds `sim.SimConfig` + cluster-only fields; `ToSimConfig()` returns the embedded config
- **workload.go**: Centralized request generation (distribution-based or CSV traces) for cluster dispatch
- **counterfactual.go**: `computeCounterfactual()` for top-k candidate ranking and regret computation
- **evaluation.go**: `EvaluationResult` wrapper (RawMetrics + FitnessResult + trace + summary)

### Decision Tracing (`sim/trace/`)

Observation-only trace recording for cluster-level policy decisions:

- **trace.go**: `TraceLevel`, `TraceConfig`, `SimulationTrace`, `NewSimulationTrace`, recording methods
- **record.go**: `AdmissionRecord`, `RoutingRecord`, `CandidateScore` (pure data types, no `sim/` dependency)
- **summary.go**: `TraceSummary`, `Summarize()` aggregation

### Latency Estimation

Two modes, selected by `NewLatencyModel()` factory based on `--model-config-folder` presence:

1. **Blackbox mode** (default): Uses trained alpha/beta coefficients from `defaults.yaml`
   - Alpha coefficients: queueing time estimation
   - Beta coefficients: step time estimation based on batch features

2. **Roofline mode**: Analytical FLOPs/bandwidth estimation via `roofline_step.go`
   - Requires HuggingFace `config.json` in `model_configs/`
   - Requires `hardware_config.json` with GPU specs

### Configuration Loading

- **model_hardware_config.go**: `HFConfig` (raw HuggingFace config), `ModelConfig` (extracted params), `HardwareCalib` (GPU specs), `ValidateRooflineConfig` (validates all roofline denominator fields)
- **defaults.yaml**: Pre-trained coefficients, default GPU/TP/vLLM mappings, workload presets
- **cmd/default_config.go**: Loading and lookup functions for defaults.yaml

### Key Data Flow

```
Request Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion
                                            ↓              ↓
                                      KV Allocation   Latency Estimation (alpha/beta or roofline)
```
Note: Admission and Routing steps apply in cluster mode (multi-instance). Single-instance mode skips directly to WaitQueue.

## Development Guidelines

### Design Principles

BLIS follows a layered design document hierarchy. Each tier has a specific abstraction level and audience:

- **Design guidelines** (`docs/templates/design-guidelines.md`): Target architecture, DES foundations, module contracts, extension framework. Read this first when designing a new feature or extending BLIS.
- **Design docs** (per-feature): Behavioral specifications written per the guidelines. Describe what modules do and why, never how they're implemented. Four species: decision record, specification, problem analysis, system overview.
- **Macro plans** (multi-PR features): PR decomposition with module contracts and extension types. Written per `docs/templates/macro-plan.md`. May include frozen interface signatures (facts about merged code) but never method implementations (aspirations about unwritten code).
- **Micro plans** (single PR): Full implementation detail with behavioral contracts, TDD tasks, exact code. Written per `docs/templates/micro-plan.md`.

**The abstraction rule:** Design docs describe *what a module does and what it guarantees*. Macro plans describe *what to build and in what order*. Micro plans describe *how to implement each piece*. Go struct definitions, method implementations, and file:line references belong only in micro plans.

**Module architecture:** BLIS has a two-layer architecture — a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) and domain-specific modules (router, scheduler, KV cache manager, latency model, autoscaler, batch formation). Each module is defined by a behavioral contract with six aspects: what it observes, what it controls, what state it owns, what invariants it maintains, what events it produces/consumes, and its extension friction (how many files to add one more variant). See design guidelines Section 4 for the full module map and contract template.

**Extending BLIS:** Four extension types, each with a different recipe — policy template (new algorithm behind existing interface), subsystem module (new module with its own interface), backend swap (alternative implementation requiring interface extraction), tier composition (delegation wrapper). See design guidelines Section 5.

### BDD/TDD Development

> **Canonical source:** [`docs/standards/principles.md`](docs/standards/principles.md) (BDD/TDD section). If this section diverges, principles.md is authoritative.

This project follows BDD/TDD practices. When implementing features:

1. **Write behavioral contracts first**: Define invariants and expected behavior in Gherkin-style scenarios
2. **Implement tests before code**: Tests verify contracts hold
3. **Use table-driven tests**: Go's table-driven test pattern for comprehensive coverage
4. **Test laws, not just values**: Golden tests answer "did the output change?" but not "is the output correct?" Every golden test should have a companion invariant test that verifies a law the system must satisfy (conservation, causality, monotonicity)
5. **Refactor survival test**: Before accepting a test, ask: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?" If no, the test is structural — rewrite it to assert observable behavior instead of internal structure. See `docs/standards/principles.md` BDD/TDD section for prohibited/required assertion patterns.
6. **THEN clauses drive test quality**: A structural THEN clause produces a structural test. If a contract's THEN clause contains a concrete type name or internal field name, rewrite the THEN clause to describe observable behavior before writing the test.

### PR Workflow

Diligently follow the workflow in docs/process/pr-workflow.md. Before I approve any plan, validate it: 1) Check every task's dependencies — can each task actually start given what comes before it? 2) Verify all sections from the template are present and non-empty. 3) Read the executive summary as if you're a new team member — is it clear and human-readable? 4) Flag any tasks that seem under-specified for implementation. List all issues found.

For new features that introduce module boundaries or modify the architecture, a design doc (per the design guidelines) should exist before micro-planning begins. For smaller changes (bug fixes, new policy templates behind existing interfaces), a design doc is optional — proceed directly to micro-planning.

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize results concisely. Never try to synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental summaries per agent rather than a consolidated report.

### Task Agent Guidelines

When using Task agents: 1) Do NOT poll TaskList repeatedly — check at reasonable intervals (every 30-60 seconds, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.

### Code Review Standards

During PR reviews, check all Antipattern Prevention rules (1-20) below. Pay special attention to rules 8-10 (exported mutable maps, YAML pointer types, strict YAML parsing) which are easy to miss in new code. Always run `go test ./...` and lint after fixes.

### Macro Plan Updates

When asked to update the macro implementation plan, directly edit the document. Do NOT spend time re-reading all source documents or dispatching sub-agents to gather information you already have in context. Start writing immediately.

### Key Invariants to Maintain

> **Canonical source:** [`docs/standards/invariants.md`](docs/standards/invariants.md). If this section diverges, invariants.md is authoritative.

Full details (verification strategies, evidence): see [`docs/standards/invariants.md`](docs/standards/invariants.md).

- **INV-1 Request conservation**: `injected_requests == completed_requests + still_queued + still_running + dropped_unservable` at simulation end. Full pipeline: `num_requests == injected_requests + rejected_requests`.
- **INV-2 Request lifecycle**: Requests transition queued → running → completed; not completed before horizon remain in current state
- **INV-3 Clock monotonicity**: Simulation clock never decreases
- **INV-4 KV cache conservation**: `allocated_blocks + free_blocks = total_blocks` at all times
- **INV-5 Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`
- **INV-6 Determinism**: Same seed must produce byte-identical stdout across runs. Wall-clock timing goes to stderr.
- **INV-7 Signal freshness**: Routing snapshot signals have tiered freshness — PendingRequests (synchronous) vs KVUtilization (stale across batch steps). See `docs/standards/invariants.md` for the full hierarchy.
- **INV-8 Work-conserving**: After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while work is waiting.

### Engineering Principles

> **Canonical source:** [`docs/standards/principles.md`](docs/standards/principles.md). If this section diverges, principles.md is authoritative.

Full details: see [`docs/standards/principles.md`](docs/standards/principles.md).

**Separation of concerns:** `sim/` is a library (never terminates). Cluster-level policies see global state via `*RouterState`. Instance-level policies see only local data. Dependency direction: `cmd/ → sim/cluster/ → sim/`.

**Interface design:** Single-method interfaces. Pure query methods. Factory validation. Behavioral contracts, not implementation-specific (R13). Single-module methods (R14).

**Configuration design:** Group by module (R16). `SimConfig` composed of 6 embedded sub-configs. Factory signatures accept the narrowest sub-config: `NewKVStore(KVCacheConfig)`, `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`, `NewBatchFormation(LatencyModel)`. Each module's config independently validatable.

**Canonical constructors:** Struct literals in exactly one place (R4). Grep for ALL construction sites before adding fields.

**Output channel separation:** stdout (deterministic results), stderr (diagnostics via logrus).

**Error handling boundaries:** CLI → `logrus.Fatalf`. Library → `error` or `panic`. Never silent `continue` (R1).

### Antipattern Prevention

> **Canonical source:** [`docs/standards/rules.md`](docs/standards/rules.md). If this section diverges, rules.md is authoritative.

20 rules, each tracing to a real bug. Full details (evidence, checks, enforcement): see [`docs/standards/rules.md`](docs/standards/rules.md).

| # | Rule | One-sentence summary |
|---|------|---------------------|
| R1 | No silent data loss | Every error path must return error, panic, or increment counter — never silently drop data |
| R2 | Sort map keys | Map iteration feeding float sums or output ordering must sort keys first (determinism) |
| R3 | Validate CLI flags | Every numeric flag validated for zero, negative, NaN, Inf |
| R4 | Construction site audit | Adding a struct field? Grep for ALL literal construction sites, update every one |
| R5 | Transactional mutation | Resource-allocating loops must rollback on mid-loop failure |
| R6 | No Fatalf in library | `sim/` never terminates the process — return errors to callers |
| R7 | Invariant tests | Every golden test needs a companion invariant test verifying a system law |
| R8 | No exported maps | Validation maps unexported; expose via `IsValid*()` accessors |
| R9 | YAML pointer types | Use `*float64` when zero is a valid user value |
| R10 | Strict YAML parsing | `yaml.KnownFields(true)` — typos must cause errors |
| R11 | Guard division | Runtime-derived denominators must be checked for zero |
| R12 | Golden regeneration | Regenerate and document golden dataset when output changes |
| R13 | Multi-impl interfaces | New interfaces must work for >=2 backends |
| R14 | Single-module methods | No method spans scheduling + latency + metrics — extract concerns |
| R15 | Stale PR references | Grep for `planned for PR N` after completing PR N |
| R16 | Config by module | Group config parameters by module, not monolithic structs |
| R17 | Signal freshness | Document which routing signals are synchronously fresh vs stale |
| R18 | CLI flag precedence | defaults.yaml must not silently override user-provided CLI flags |
| R19 | Livelock protection | Unbounded retry/requeue loops must have circuit breakers |
| R20 | Degenerate detector inputs | Anomaly detectors must handle empty, skewed, or zero inputs explicitly |


### Current Implementation Focus

Active development: Composable Scorer Framework (see `docs/plans/2026-02-19-weighted-scoring-macro-plan.md`). PR17 (scorer framework + stateless scorers) and PR18 (prefix-affinity scorer + router-side cache) completed. Default weighted routing profile: `prefix-affinity:3,queue-depth:2,kv-utilization:2` (llm-d parity).

### Extension Recipes

Step-by-step guides for adding policies, scorers, latency model backends, KV tiers, trace records, and per-request metrics: see `docs/extension-recipes.md`.

### Adding New Scorers (Weighted Routing)

To add a new scoring dimension for the `weighted` routing policy (e.g., predicted-latency):

1. **Implement the scorer function** in `sim/routing_scorers.go` (stateless) or a new file (stateful) — a `scorerFunc` that takes `(*Request, []RoutingSnapshot)` and returns `map[string]float64` with scores in [0,1] per instance. Stateful scorers also return an `observerFunc` called after each routing decision.
2. **Register the scorer** in `sim/routing_scorers.go`: add to `validScorerNames` map + `newScorerWithObserver` factory switch
3. **Add behavioral tests** — monotonicity, boundary values, INV-1/INV-2 conformance
4. Extension friction: **2 touch points** (implementation + registration)

**Stateful scorers and observers:**
- Stateful scorers (like prefix-affinity) return an `observerFunc` alongside the `scorerFunc`. The `observerFunc` signature is `func(req *Request, targetInstance string)`.
- The observer is called by `WeightedScoring.Route()` after argmax selects the target instance but before returning the `RoutingDecision`. This lets the scorer update internal state (e.g., recording which blocks were routed to which instance).
- The scorer and observer share state via closure. See `newPrefixAffinityScorer` in `sim/routing_prefix_scorer.go`: both the scorer and observer close over the same `PrefixCacheIndex`, so the observer's `RecordBlocks` calls are visible to subsequent scorer invocations.

Examples:
- See `scoreLoadBalance` in `sim/routing_scorers.go` for a simple stateless scorer
- See `scoreQueueDepth` for a scorer with edge case handling (uniform load)
- See `newPrefixAffinityScorer` in `sim/routing_prefix_scorer.go` for a stateful scorer with observer and router-side cache

### Code Style

- Use composition over inheritance (e.g., `InstanceSimulator` wraps existing `sim` components)
- Timestamp-based event ordering via min-heap; cluster event queue uses `(timestamp, priority, seqID)` ordering; per-instance queues use timestamp-only; cluster-level instance ties broken by lowest instance index
- Partitioned RNG per subsystem to isolate randomness

### CI/CD

GitHub Actions CI runs on all PRs to main (`.github/workflows/ci.yml`):
- `go build ./...` - Build verification
- `golangci-lint run ./...` - Static analysis (v2.9.0)
- `go test ./...` - Test suite

Run lint locally before pushing: `golangci-lint run ./...`

## File Organization

```
inference-sim/
├── .github/workflows/         # CI configuration (build, lint, test)
├── main.go                    # CLI entry point (Cobra)
├── cmd/
│   ├── root.go                # CLI commands and flags (--num-instances, --policy-config, --routing-scorers, --workload-spec, --trace-level, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth, --kv-transfer-base-latency, --snapshot-refresh-interval)
│   ├── observe.go             # Real mode HTTP client (OpenAI-compatible, streaming + non-streaming)
│   └── default_config.go      # defaults.yaml loading
├── sim/                       # Core single-instance simulator
│   ├── config.go              # Module-scoped sub-config types (KVCacheConfig, BatchConfig, LatencyCoeffs, etc.)
│   ├── simulator.go           # SimConfig struct (composed of embedded sub-configs), NewSimulator(SimConfig), event loop, batch formation, step execution
│   ├── admission.go           # AdmissionPolicy interface, AlwaysAdmit, TokenBucket, NewAdmissionPolicy factory
│   ├── routing.go             # RoutingPolicy interface, RoutingSnapshot, RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity
│   ├── routing_scorers.go     # ScorerConfig, scorerFunc, stateless scorers, ParseScorerConfigs, newScorerWithObserver
│   ├── routing_prefix_scorer.go # Prefix-affinity scorer + observer (proportional prefix matching)
│   ├── prefix_cache_index.go  # PrefixCacheIndex: per-instance LRU of hierarchical block hashes
│   ├── priority.go            # PriorityPolicy interface, ConstantPriority, SLOBasedPriority, NewPriorityPolicy factory
│   ├── scheduler.go           # InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, NewScheduler factory
│   ├── latency_model.go       # LatencyModel interface, BlackboxLatencyModel, RooflineLatencyModel, NewLatencyModel factory
│   ├── router_state.go        # RouterState bridge type (Snapshots + Clock) for cluster-level policies
│   ├── bundle.go              # PolicyBundle YAML loading, LoadPolicyBundle, Validate
│   ├── event.go               # Event types (Arrival, Queued, Step, Scheduled, Preemption, RequestLeft)
│   ├── request.go             # Request state machine (queued → running → completed), Priority field, workload metadata (TenantID, SLOClass, etc.)
│   ├── kvcache.go             # Block-based KV cache with LRU eviction, prefix caching, transactional rollback
│   ├── kv_store.go            # KVStore interface, NewKVStore factory
│   ├── kvcache_tiered.go      # TieredKVCache: GPU+CPU composition, offload/reload, transfer latency
│   ├── batch.go               # Batch struct
│   ├── batch_formation.go     # BatchFormation interface, VLLMBatchFormation, NewBatchFormation factory
│   ├── queue.go               # FIFO wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct, NewRequestMetrics canonical constructor
│   ├── rng.go                 # PartitionedRNG for deterministic multi-subsystem simulation
│   ├── roofline_step.go       # Analytical FLOPs/bandwidth latency estimation
│   ├── model_hardware_config.go # HFConfig, ModelConfig, HardwareCalib, ValidateRooflineConfig
│   ├── workload_config.go     # CSV trace loading and distribution-based workload generation
│   └── internal/testutil/     # Shared test infrastructure (golden dataset loading)
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator wrapper with run-once guard
│   ├── cluster.go             # ClusterSimulator: shared-clock event loop, online routing, aggregation
│   ├── cluster_event.go       # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent
│   ├── snapshot.go            # CachedSnapshotProvider (returns sim.RoutingSnapshot), ObservabilityConfig
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, anomaly detection, per-SLO-class metrics, JainFairnessIndex
│   ├── deployment.go          # DeploymentConfig (embeds sim.SimConfig) + cluster-only fields
│   ├── workload.go            # Centralized request generation for cluster dispatch
│   └── evaluation.go          # EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
├── sim/workload/              # ServeGen-informed workload generation (PR10)
│   ├── spec.go                # WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, YAML loading
│   ├── arrival.go             # ArrivalSampler: Poisson, Gamma (Marsaglia-Tsang), Weibull (bisection)
│   ├── distribution.go        # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF, Constant
│   ├── client.go              # Rate normalization, prefix group management
│   ├── generator.go           # GenerateRequests pipeline with client decomposition
│   ├── servegen.go            # Native ServeGen data file loading (chunk-*-trace.csv + dataset.json)
│   ├── tracev2.go             # Trace v2 format (YAML header + CSV data)
│   ├── replay.go              # Trace v2 → sim.Request with synthetic token IDs
│   ├── calibrate.go           # CalibrationReport, PrepareCalibrationPairs, MAPE/Pearson r
│   ├── multimodal.go          # Multimodal token generation (text+image+audio+video)
│   ├── reasoning.go           # Reasoning multi-turn with context accumulation
│   ├── network.go             # Client-perspective latency (RTT + bandwidth)
│   ├── inference_perf.go      # inference-perf format: InferencePerfSpec, expansion, validation
│   └── scenarios.go           # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
├── sim/trace/                 # Decision trace recording (PR13)
│   ├── trace.go               # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go             # TraceSummary, Summarize()
├── model_configs/             # HuggingFace config.json files
├── defaults.yaml              # Trained coefficients, defaults
├── hardware_config.json       # GPU specifications
├── examples/                  # Example configuration files
├── hypotheses/                # Hypothesis experiment artifacts (run.sh, analyze.py, FINDINGS.md)
├── testdata/goldendataset.json # Golden dataset for regression tests
├── docs/
│   ├── standards/             # Canonical rules, invariants, principles, experiment standards
│   ├── process/               # Activity workflows (PR, design, macro-plan, hypothesis)
│   ├── templates/             # Artifact templates (micro-plan, macro-plan, design-guidelines, hypothesis)
│   ├── plans/                 # Active implementation plans
│   │   └── archive/           # Completed design docs (architectural reference)
│   ├── pr-history.md          # Completed PRs and design doc catalog
│   └── extension-recipes.md   # Step-by-step extension guides
└── CONTRIBUTING.md            # Contributor guide (references docs/standards/)
```

## Project Governance Documents

### Standards (what rules apply)

- `docs/standards/rules.md`: **20 antipattern rules** (R1-R20) — each with evidence, checks, enforcement locations
- `docs/standards/invariants.md`: **8 system invariants** (INV-1 through INV-8) — with verification strategies
- `docs/standards/principles.md`: **Engineering principles** — separation of concerns, interface design, BDD/TDD
- `docs/standards/experiments.md`: **Experiment standards** — hypothesis families (6 families × type classification), rigor requirements, root cause verification (RCV-1 through RCV-6), iterative review protocol, findings classification

### Process (how to do each activity)

- `docs/process/pr-workflow.md`: End-to-end PR workflow (worktree → plan → review → implement → audit → commit)
- `docs/process/design.md`: Design document creation process
- `docs/process/macro-plan.md`: Macro-level (multi-PR) planning process
- `docs/process/hypothesis.md`: Hypothesis experiment process

### Templates (what to produce)

- `docs/templates/design-guidelines.md`: **BLIS Design Guidelines** — DES foundations, module architecture, extension framework. **Start here when designing anything new.**
- `docs/templates/macro-plan.md`: Template for macro-level planning (multi-PR features)
- `docs/templates/micro-plan.md`: Template for micro-level (per-PR) planning with TDD tasks and behavioral contracts
- `docs/templates/hypothesis.md`: Template for hypothesis experiment artifacts

### Per-Feature Plans and PR History

See `docs/pr-history.md` for the full catalog of design documents, micro-plans, and completed PR summaries.
