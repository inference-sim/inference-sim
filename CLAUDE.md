# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference platforms (vLLM, SGLang). It models request arrival, KV-cache dynamics, scheduling, token generation, and latency using trained performance coefficients (alpha/beta) or analytical roofline models.

The simulator is CPU-only and designed for capacity planning, saturation analysis, and performance prediction without requiring real GPUs.

## Build and Run Commands

```bash
# Build
go build -o simulation_worker main.go

# Run with default model
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct

# Run with custom workload distribution
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload distribution \
  --rate 10 --num-requests 100 \
  --prompt-tokens 512 --output-tokens 256

# Run with trace replay (deterministic testing)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload traces --workload-traces-filepath traces.csv

# Run with roofline mode (no trained coefficients required)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json --hardware H100 --tp 1

# Run multi-instance with routing policy
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-cache-weight 0.6 --routing-load-weight 0.4

# Run with fitness evaluation and anomaly detection
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --fitness-weights "throughput:0.5,p99_ttft:0.3,mean_e2e:0.2"

# Run with decision tracing and counterfactual analysis
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --trace-level decisions --counterfactual-k 5 --summarize-trace

# Run with ServeGen-informed workload specification
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --workload-spec examples/servegen-language.yaml
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

- **simulator.go**: `SimConfig` struct, `NewSimulator(SimConfig) (*Simulator, error)` constructor, `Simulator` struct and event loop (`Run()`), batch formation (`makeRunningBatch`), step execution, observation methods (`QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()`)
- **admission.go**: `AdmissionPolicy` interface (accepts `*RouterState`), `AlwaysAdmit`, `TokenBucket`, `RejectAll`, `NewAdmissionPolicy` factory
- **routing.go**: `RoutingPolicy` interface (accepts `*RouterState`), `RoutingSnapshot` (with `EffectiveLoad()` for canonical load calculation), `RoutingDecision` (with `Priority` hint), `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity`, `AlwaysBusiest` templates, `NewRoutingPolicy` factory
- **priority.go**: `PriorityPolicy` interface with `ConstantPriority`, `SLOBasedPriority`, and `InvertedSLO` templates, `NewPriorityPolicy` factory
- **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`, and `ReversePriority` templates, `NewScheduler` factory
- **router_state.go**: `RouterState` bridge type (Snapshots + Clock) for cluster-level policy interfaces
- **bundle.go**: `PolicyBundle` struct with YAML loading (`LoadPolicyBundle`), validation (`Validate`)
- **event.go**: Event types (`ArrivalEvent`, `QueuedEvent`, `StepEvent`, `ScheduledEvent`, `RequestLeftEvent`, `PreemptionEvent`)
- **request.go**: `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`), Request lifecycle and state machine, `Priority` field for scheduler-aware ordering, `AssignedInstance` for cluster routing provenance (#181)
- **kvcache.go**: Block-based KV cache with LRU eviction and prefix caching, `CacheHits`/`CacheMisses` counters, transactional `AllocateKVBlocks` with `rollbackAllocation` on mid-loop failure
- **kv_store.go**: `KVStore` interface (11 methods: +`SetClock`, +`ConsumePendingTransferLatency`), `NewKVStore` factory with input validation (returns single-tier or tiered based on config)
- **kvcache_tiered.go**: `TieredKVCache` (GPU+CPU composition), `cpuTier`, `offloadedBlock`, offload/reload/transfer latency, `PendingTransferLatency()` (pure query), `ConsumePendingTransferLatency()` (read-and-clear)
- **batch.go**: Batch formation respecting token budgets and batch size limits
- **queue.go**: FIFO wait queue for pending requests

### Cluster Simulation (`sim/cluster/`)

Multi-replica extension using composition over the single-instance simulator:

- **instance.go**: `InstanceSimulator` wraps `sim.Simulator` via `NewInstanceSimulator(id, SimConfig)` with run-once guard; delegates to Simulator observation methods (`QueueDepth()`, `BatchSize()`, etc.) instead of direct field access
- **cluster.go**: `ClusterSimulator` orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation; `Run()` returns `error`
- **metrics.go**: `RawMetrics`, `Distribution`, `FitnessResult`, `CollectRawMetrics` (accepts `priorityPolicy` to suppress false-positive inversion detection for constant priority), `ComputeFitness` (returns `(FitnessResult, error)` — fails on unknown keys), anomaly detection, `ParseFitnessWeights` with NaN/Inf validation
- **deployment.go**: `DeploymentConfig` struct with `ToSimConfig()` for per-instance construction
- **workload.go**: Centralized request generation (distribution-based or CSV traces) for cluster dispatch
- **counterfactual.go**: `computeCounterfactual()` for top-k candidate ranking and regret computation
- **evaluation.go**: `EvaluationResult` wrapper (RawMetrics + FitnessResult + trace + summary)

### Decision Tracing (`sim/trace/`)

Observation-only trace recording for cluster-level policy decisions:

- **trace.go**: `TraceLevel`, `TraceConfig`, `SimulationTrace`, `NewSimulationTrace`, recording methods
- **record.go**: `AdmissionRecord`, `RoutingRecord`, `CandidateScore` (pure data types, no `sim/` dependency)
- **summary.go**: `TraceSummary`, `Summarize()` aggregation

### Latency Estimation

Two modes controlled by `--model-config-folder` presence:

1. **Blackbox mode** (default): Uses trained alpha/beta coefficients from `defaults.yaml`
   - Alpha coefficients: queueing time estimation
   - Beta coefficients: step time estimation based on batch features

2. **Roofline mode**: Analytical FLOPs/bandwidth estimation via `roofline_step.go`
   - Requires HuggingFace `config.json` in `model_configs/`
   - Requires `hardware_config.json` with GPU specs

### Configuration Loading

- **model_hardware_config.go**: `HFConfig` (raw HuggingFace config), `ModelConfig` (extracted params), `HardwareCalib` (GPU specs)
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

- **Design guidelines** (`docs/plans/2026-02-18-design-guidelines.md`): Target architecture, DES foundations, module contracts, extension framework. Read this first when designing a new feature or extending BLIS.
- **Design docs** (per-feature): Behavioral specifications written per the guidelines. Describe what modules do and why, never how they're implemented. Four species: decision record, specification, problem analysis, system overview.
- **Macro plans** (multi-PR features): PR decomposition with module contracts and extension types. Written per `docs/plans/macroplanprompt.md`. May include frozen interface signatures (facts about merged code) but never method implementations (aspirations about unwritten code).
- **Micro plans** (single PR): Full implementation detail with behavioral contracts, TDD tasks, exact code. Written per `docs/plans/prmicroplanprompt-v2.md`.

**The abstraction rule:** Design docs describe *what a module does and what it guarantees*. Macro plans describe *what to build and in what order*. Micro plans describe *how to implement each piece*. Go struct definitions, method implementations, and file:line references belong only in micro plans.

**Module architecture:** BLIS has a two-layer architecture — a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) and domain-specific modules (router, scheduler, KV cache manager, latency model, autoscaler, batch formation). Each module is defined by a behavioral contract with six aspects: what it observes, what it controls, what state it owns, what invariants it maintains, what events it produces/consumes, and its extension friction (how many files to add one more variant). See design guidelines Section 4 for the full module map and contract template.

**Extending BLIS:** Four extension types, each with a different recipe — policy template (new algorithm behind existing interface), subsystem module (new module with its own interface), backend swap (alternative implementation requiring interface extraction), tier composition (delegation wrapper). See design guidelines Section 5.

### BDD/TDD Development

This project follows BDD/TDD practices. When implementing features:

1. **Write behavioral contracts first**: Define invariants and expected behavior in Gherkin-style scenarios
2. **Implement tests before code**: Tests verify contracts hold
3. **Use table-driven tests**: Go's table-driven test pattern for comprehensive coverage
4. **Test laws, not just values**: Golden tests answer "did the output change?" but not "is the output correct?" Every golden test should have a companion invariant test that verifies a law the system must satisfy (conservation, causality, monotonicity)
5. **Refactor survival test**: Before accepting a test, ask: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?" If no, the test is structural — rewrite it to assert observable behavior instead of internal structure. Common structural traps: type assertions on factory returns (`policy.(*ConcreteType)`), exact formula reproduction (`assert.Equal(score, 0.6*x + 0.4*y)`), internal field access. See `prmicroplanprompt-v2.md` rules 9-10 for the full prohibited/required assertion patterns.
6. **THEN clauses drive test quality**: A structural THEN clause produces a structural test. If a contract's THEN clause contains a concrete type name or internal field name, rewrite the THEN clause to describe observable behavior before writing the test.

### PR Workflow

Diligently follow the workflow in docs/plans/prworkflow.md. Before I approve any plan, validate it: 1) Check every task's dependencies — can each task actually start given what comes before it? 2) Verify all sections from the template are present and non-empty. 3) Read the executive summary as if you're a new team member — is it clear and human-readable? 4) Flag any tasks that seem under-specified for implementation. List all issues found.

For new features that introduce module boundaries or modify the architecture, a design doc (per the design guidelines) should exist before micro-planning begins. For smaller changes (bug fixes, new policy templates behind existing interfaces), a design doc is optional — proceed directly to micro-planning.

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize results concisely. Never try to synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental summaries per agent rather than a consolidated report.

### Task Agent Guidelines

When using Task agents: 1) Do NOT poll TaskList repeatedly — check at reasonable intervals (every 30-60 seconds, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.

### Code Review Standards

During PR reviews, check all Antipattern Prevention rules (1-11) below. Pay special attention to rules 8-10 (exported mutable maps, YAML pointer types, strict YAML parsing) which are easy to miss in new code. Always run `go test ./...` and lint after fixes.

### Macro Plan Updates

When asked to update the macro implementation plan, directly edit the document. Do NOT spend time re-reading all source documents or dispatching sub-agents to gather information you already have in context. Start writing immediately.

### Key Invariants to Maintain

- **Request conservation**: `injected == completed + queued + running` at simulation end
- **Request lifecycle**: Requests transition queued → running → completed; requests not completed before horizon remain in current state
- **Clock monotonicity**: Simulation clock never decreases
- **KV cache conservation**: `allocated_blocks + free_blocks = total_blocks` at all times
- **Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`
- **Determinism**: Same seed must produce byte-identical output across runs

### Engineering Principles

**Separation of concerns:**
- `sim/` is a library — it must never call `os.Exit`, `logrus.Fatalf`, or terminate the process. Return errors to callers. Only `cmd/` may terminate.
- Cluster-level policies (admission, routing) receive `*RouterState` with global view. Instance-level policies (priority, scheduler) receive only local data. Never leak cluster state to instance-level code.
- Bridge types (`RouterState`, `RoutingSnapshot`) live in `sim/` to avoid import cycles. Conversion between package-specific types and bridge types happens at package boundaries.
- Each package has a unidirectional dependency: `cmd/ → sim/cluster/ → sim/` and `sim/cluster/ → sim/trace/`. The `sim/` package must never import subpackages.

**Interface design:**
- Policy interfaces should be single-method when possible (see `AdmissionPolicy`, `RoutingPolicy`, `PriorityPolicy`, `InstanceScheduler`).
- Query methods must be pure — no side effects, no state mutation, no destructive reads. If a method needs to both query and clear state, provide separate `Get()` and `Consume()` methods.
- Factory functions must validate their inputs. Follow the pattern: `IsValid*()` check + switch/case + panic on unknown. Never silently accept invalid configuration.
- Interfaces must be defined by behavioral contract (allocate, query, release), not by one implementation's data model. If an interface method only makes sense for one backend, the interface is too specific — it must accommodate at least two implementations.
- Individual methods should operate within a single module's responsibility. If a method spans scheduling, latency estimation, and metric collection, extract each concern into its module's interface.

**Configuration design:**
- Group configuration by module. A single config struct combining hardware identity, model parameters, simulation parameters, and policy choices creates shotgun surgery when adding parameters. Each module's config should be independently specifiable and validatable.

**Canonical constructors:**
- Every struct constructed in multiple places needs a canonical constructor (e.g., `NewRequestMetrics()`). Struct literals appear in exactly one place.
- Before adding a field to a struct, grep for ALL construction sites (`StructName{`). Update every site or refactor to use the canonical constructor.

**Output channel separation:**
- Simulation results (metrics JSON, fitness scores, anomaly counters, trace summaries) use `fmt.Println`/`fmt.Printf` to write to **stdout**. These are the program's primary output and must be visible regardless of `--log` level.
- Diagnostic messages (configuration echoes, progress markers, warnings, errors) use `logrus.*` to write to **stderr**, controlled by `--log`.
- Rule of thumb: if a user piping output to a file would want to capture it, use `fmt`. If it is operational context for debugging, use `logrus`.

**Error handling boundaries:**
- `cmd/root.go`: `logrus.Fatalf` for user input errors (this is the CLI boundary)
- `sim/`, `sim/cluster/`, `sim/workload/`: `panic()` for internal invariant violations that represent programming errors; `error` return for recoverable failures; `bool` return for expected conditions (e.g., KV allocation failure → preempt and retry)
- Never use `continue` in an error path without either propagating the error, counting the occurrence, or documenting why it's safe. Silent `continue` that drops data is the most common source of bugs in this codebase.

### Antipattern Prevention

Each rule traces to a real bug we found and fixed. Enforced by PR workflow (self-audit dimensions 7-9) and micro-plan template (Phase 8 checklist).

1. **No silent data loss**: Every error path must either return an error, panic with context, or increment a counter. A `continue` or early `return` that silently drops a request, metric, or allocation is a correctness bug.

2. **Sort map keys before float accumulation**: Go map iteration is non-deterministic. Any `for k, v := range someMap` that feeds a running sum (`total += v`) or determines output ordering must sort keys first. Unsorted iteration violates the determinism invariant.

3. **Validate ALL numeric CLI flags**: Every numeric flag (`--rate`, `--fitness-weights`, `--kv-cpu-blocks`, etc.) must be validated for: zero, negative, NaN, Inf, and empty string. Missing validation causes infinite loops (Rate=0) or wrong results (NaN weights).

4. **Construction site audit**: Before adding a field to a struct, find every place that struct is constructed as a literal. If there are multiple sites, either add a canonical constructor or update every site. Missing a site causes silent field-zero bugs.

5. **Transactional state mutation**: Any loop that allocates resources (blocks, slots, counters) must handle mid-loop failure by rolling back all mutations from previous iterations. A partial allocation that returns `false` without cleanup violates conservation invariants.

6. **No logrus.Fatalf in library code**: The `sim/` package tree must never terminate the process — return errors so callers can handle them. This enables embedding, testing, and adapters.

7. **Invariant tests alongside golden tests**: Golden tests (comparing against known-good output) are regression freezes, not correctness checks. If a bug exists when the golden values are captured, the golden test perpetuates the bug. Every subsystem that has golden tests must also have invariant tests that verify conservation laws, causality, and determinism.

8. **No exported mutable maps**: Validation lookup maps (e.g., `validRoutingPolicies`) must be unexported. Expose through `IsValid*()` accessor functions. Exported maps allow callers to mutate global state, breaking encapsulation and enabling hard-to-trace bugs.

9. **Pointer types for YAML zero-value ambiguity**: YAML config structs must use `*float64` (pointer) for fields where zero is a valid user-provided value, to distinguish "not set" (nil) from "set to zero" (0.0). Using bare `float64` causes silent misconfiguration when users intentionally set a value to zero.

10. **Strict YAML parsing**: Use `yaml.KnownFields(true)` or equivalent strict parsing for all YAML config loading. Typos in field names must cause parse errors, not silent acceptance of malformed config. A silently ignored typo produces default behavior that the user didn't intend.

11. **Guard division in runtime computation**: Any division where the denominator derives from runtime state (batch size, block count, request count, bandwidth) must guard against zero. CLI validation (rule 3) catches input zeros at the boundary; this rule catches intermediate zeros that arise during simulation (e.g., `utilization = usedBlocks / totalBlocks` when no blocks are configured, `avgLatency = sum / count` when count is zero). Use explicit guards or documented invariants proving the denominator is non-zero.

### Current Implementation Focus

Active development: Evolutionary Policy Optimization extension (see `docs/plans/2026-02-11-macro-implementation-plan-v2.md`):
- 16 PRs across 6 phases to extend BLIS to multi-replica cluster simulation
- **Research-ready checkpoint at ~5 weeks** (after Phase 2) enables early policy experiments
- **Completed:** PR1 (PartitionedRNG), PR2 (InstanceSimulator), PR3 (ClusterSimulator with shared-clock event loop, round-robin dispatch, metrics aggregation, golden dataset equivalence tests), PR4 (cluster control plane with online routing pipeline, SnapshotProvider, AdmissionPolicy with AlwaysAdmit + TokenBucket templates, cluster event queue), PR5 (architectural simplification: SimConfig struct, unified CLI path through ClusterSimulator, field privatization, AdmissionPolicy consolidated to `sim/admission.go`), PR6 (RoutingPolicy interface in `sim/routing.go` with RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity templates; RoutingSnapshot bridge type), PR7 (PriorityPolicy with ConstantPriority + SLOBasedPriority templates, InstanceScheduler with FCFS + PriorityFCFS + SJF templates, Priority field on Request, CLI flags `--priority-policy` and `--scheduler`), PR8 (RouterState bridge type in `sim/router_state.go`, PolicyBundle YAML config in `sim/bundle.go`, `--policy-config` CLI flag, AdmissionPolicy and RoutingPolicy accept `*RouterState`, `RoutingDecision.Priority` hint field, **INTERFACE FREEZE**), PR9 (RawMetrics with Distribution + FitnessResult, anomaly detection with priority inversion + HOL blocking counters, pathological templates: reject-all, inverted-slo, always-busiest, reverse-priority, `--fitness-weights` CLI flag, **RESEARCH-READY CHECKPOINT**)
- **Completed (cont'd):** PR13 (DecisionTrace with RoutingRecord, counterfactual analysis with top-k candidates and regret, TraceSummary, EvaluationResult wrapper, `--trace-level decisions --counterfactual-k --summarize-trace` CLI flags), PR10 (ServeGen-informed Workload Generator in `sim/workload/` with multi-client specs, Poisson/Gamma/Weibull arrivals, Gaussian/Exponential/ParetoLogNormal/EmpiricalPDF distributions, native ServeGen loading, trace v2 replay, CalibrationReport with MAPE/Pearson r, real-mode HTTP client in `cmd/observe.go`, per-SLO-class metrics with Jain fairness index, `--workload-spec` CLI flag), PR12 (TieredKVCache with GPU+CPU offload/reload, `KVStore` interface, `NewKVStore` factory, `--kv-cpu-blocks --kv-offload-threshold --kv-transfer-bandwidth` CLI flags)
- **Completed (hardening):** Phase 1 (structural helpers), Phase 2 (correctness fixes), Phase 3 (metric fixes), Phase 4 (invariant tests), Phase 5 (input validation), Phase 6 (modularity: observation methods, snapshot unification, SetClock on interface, PendingTransferLatency pure query, error returns, ValidPolicyNames, RequestState constants, NewKVStore validation)
- **Next:** PR11 (autoscaling), PR14 (P/D disaggregation, depends on PR12), PR15 (framework adapters), PR16 (integration tests)
- Each PR is CLI-exercisable immediately after merge (no scaffolding)

### Adding New Policy Templates

To add a new policy template (e.g., a new routing algorithm):

1. **Implement the interface** in the corresponding file:
   - `AdmissionPolicy` → `sim/admission.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `RoutingPolicy` → `sim/routing.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `PriorityPolicy` → `sim/priority.go` (instance-level: receives `req` + `clock` only)
   - `InstanceScheduler` → `sim/scheduler.go` (instance-level: receives `requests` + `clock` only)
   - Note: `RouterState` is a bridge type in `sim/` to avoid import cycles — see `sim/router_state.go`

2. **Register in two places** (both required):
   - Add policy name to valid names map in `sim/bundle.go` (e.g., `validRoutingPolicies`) and corresponding `IsValid*` function
   - Add `case` to factory function in the same policy file (e.g., `NewRoutingPolicy` in `sim/routing.go`)
   - CLI error messages auto-derive from `ValidAdmissionPolicyNames()` etc. — no manual update needed

3. **Add tests** following BDD naming: `TestMyPolicy_Scenario_Behavior`
   - Test observable behavior, not internal structure
   - Include empty-snapshots panic test for routing policies (defensive programming convention)
   - Use `&RouterState{Snapshots: snapshots, Clock: clock}` in test setup

4. **Update documentation**: CLAUDE.md file organization, README policy lists

**Important:** For load-based routing, use `snap.EffectiveLoad()` — never compute `QueueDepth + BatchSize + PendingRequests` inline. This ensures all routing policies use the same formula.

Examples:
- See `RejectAll` in `sim/admission.go` for a simple admission template (constant return)
- See `PrefixAffinity` in `sim/routing.go` for a stateful routing policy with LeastLoaded fallback

### Extending KV Cache Tiers

To add a new KV tier (e.g., NVMe offloading for 3-tier GPU+CPU+NVMe):

1. **Implement the `KVStore` interface** in `sim/kvcache_*.go` (11 methods: allocate, get cached, release, capacity queries, metrics, `SetClock`, `ConsumePendingTransferLatency`)
2. **Compose existing tiers** — e.g., wrap `TieredKVCache` (GPU+CPU) with NVMe logic, following the same delegation pattern
3. **Update `NewKVStore` factory** in `sim/kv_store.go` to instantiate your tier based on `SimConfig` fields
4. **Add CLI flags** in `cmd/root.go` for new parameters (e.g., `--kv-nvme-blocks`)
5. **Aggregate metrics** — combine hit/miss/thrashing counters from all tiers; see `TieredKVCache.CacheHitRate()` for the 2-tier pattern
6. **Add behavioral tests** in `sim/kvcache_*_test.go`
7. **Preserve rollback semantics** — `KVCacheState.AllocateKVBlocks` is transactional: on mid-loop failure, `rollbackAllocation()` undoes all mutations (UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, RequestMap). If your tier adds mutations beyond what delegation to `gpu.AllocateKVBlocks()` handles, you must roll those back too. See `cachedBlockMutation` and `newBlockMutation` types in `sim/kvcache.go`.
8. **`GetCachedBlocks` is a pure query** — it returns cached block IDs without side effects. `CacheHits` are counted by `AllocateKVBlocks` when cached blocks are committed to an allocation (and rolled back on failure). This was fixed in the Phase 3 hardening PR; the previous implementation incremented CacheHits in GetCachedBlocks, causing double-counting in tiered mode.

Examples:
- See `TieredKVCache` in `sim/kvcache_tiered.go` for 2-tier GPU+CPU composition
- See `KVCacheState` in `sim/kvcache.go` for single-tier baseline (also implements `KVStore`)
- See `docs/plans/pr12-architectural-predesign.md` for the design decisions behind the tiered architecture

### Adding New Trace Record Types

To add a new trace record type (e.g., `ScaleRecord` for autoscaling events):

1. **Define the record struct** in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. **Add a slice field** to `SimulationTrace` in `sim/trace/trace.go` (e.g., `Scales []ScaleRecord`)
3. **Add a recording method** to `SimulationTrace` (e.g., `RecordScale(ScaleRecord)`)
4. **Hook recording** into the cluster event pipeline in `sim/cluster/cluster_event.go` (guard with `if cs.trace != nil` for zero-overhead default)
5. **Update `Summarize()`** in `sim/trace/summary.go` to aggregate the new record type
6. **Add behavioral tests** in `sim/trace/*_test.go`

Examples:
- See `AdmissionRecord` in `sim/trace/record.go` for a simple record
- See `RoutingRecord` with `CandidateScore` for a record with nested counterfactual data
- See `computeCounterfactual()` in `sim/cluster/counterfactual.go` for derived computation that lives in `sim/cluster/` (not `sim/trace/`) because it needs `sim.RoutingSnapshot`

### Adding New Per-Request Metric Fields

To add a new field to per-request JSON output (appears in `--results-path` output):

1. **Add field to `Request`** in `sim/request.go` (runtime state, zero-value safe). When constructing `Request` structs, use `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`) — never bare strings.
2. **Add field to `RequestMetrics`** in `sim/metrics_utils.go` (JSON output struct, use `omitempty` for backward compatibility)
3. **Update `NewRequestMetrics()` constructor** in `sim/metrics_utils.go` to propagate the new field from `Request` to `RequestMetrics`
4. **Set the field** at the appropriate event (e.g., `RoutingDecisionEvent` for cluster-level, or completion for computed metrics)
5. **Add behavioral tests** covering multi-instance, single-instance, and standalone boundaries

Examples:
- See `HandledBy` (#181) — set by `RoutingDecisionEvent`, zero-value when used outside cluster pipeline (suppressed from JSON via `omitempty`)
- See `SLOClass`/`TenantID` (PR10) — set during workload generation, propagated at injection

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
│   ├── root.go                # CLI commands and flags (--num-instances, --policy-config, --workload-spec, --trace-level, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth)
│   ├── observe.go             # Real mode HTTP client (OpenAI-compatible, streaming + non-streaming)
│   └── default_config.go      # defaults.yaml loading
├── sim/                       # Core single-instance simulator
│   ├── simulator.go           # SimConfig struct, NewSimulator(SimConfig), event loop, batch formation, step execution
│   ├── admission.go           # AdmissionPolicy interface, AlwaysAdmit, TokenBucket, NewAdmissionPolicy factory
│   ├── routing.go             # RoutingPolicy interface, RoutingSnapshot, RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity
│   ├── priority.go            # PriorityPolicy interface, ConstantPriority, SLOBasedPriority, NewPriorityPolicy factory
│   ├── scheduler.go           # InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, NewScheduler factory
│   ├── router_state.go        # RouterState bridge type (Snapshots + Clock) for cluster-level policies
│   ├── bundle.go              # PolicyBundle YAML loading, LoadPolicyBundle, Validate
│   ├── event.go               # Event types (Arrival, Queued, Step, Scheduled, Preemption, RequestLeft)
│   ├── request.go             # Request state machine (queued → running → completed), Priority field, workload metadata (TenantID, SLOClass, etc.)
│   ├── kvcache.go             # Block-based KV cache with LRU eviction, prefix caching, transactional rollback
│   ├── kv_store.go            # KVStore interface, NewKVStore factory
│   ├── kvcache_tiered.go      # TieredKVCache: GPU+CPU composition, offload/reload, transfer latency
│   ├── batch.go               # Batch struct
│   ├── queue.go               # FIFO wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct, NewRequestMetrics canonical constructor
│   ├── rng.go                 # PartitionedRNG for deterministic multi-subsystem simulation
│   ├── roofline_step.go       # Analytical FLOPs/bandwidth latency estimation
│   ├── model_hardware_config.go # HFConfig, ModelConfig, HardwareCalib structs
│   ├── workload_config.go     # CSV trace loading and distribution-based workload generation
│   └── internal/testutil/     # Shared test infrastructure (golden dataset loading)
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator wrapper with run-once guard
│   ├── cluster.go             # ClusterSimulator: shared-clock event loop, online routing, aggregation
│   ├── cluster_event.go       # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent
│   ├── snapshot.go            # CachedSnapshotProvider (returns sim.RoutingSnapshot), ObservabilityConfig
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, anomaly detection, per-SLO-class metrics, JainFairnessIndex
│   ├── deployment.go          # DeploymentConfig struct
│   └── workload.go            # Centralized request generation for cluster dispatch
├── sim/workload/              # ServeGen-informed workload generation (PR10)
│   ├── spec.go                # WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, YAML loading
│   ├── arrival.go             # ArrivalSampler: Poisson, Gamma (Marsaglia-Tsang), Weibull (bisection)
│   ├── distribution.go        # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF
│   ├── client.go              # Rate normalization, prefix group management
│   ├── generator.go           # GenerateRequests pipeline with client decomposition
│   ├── servegen.go            # Native ServeGen data file loading (chunk-*-trace.csv + dataset.json)
│   ├── tracev2.go             # Trace v2 format (YAML header + CSV data)
│   ├── replay.go              # Trace v2 → sim.Request with synthetic token IDs
│   ├── calibrate.go           # CalibrationReport, PrepareCalibrationPairs, MAPE/Pearson r
│   ├── multimodal.go          # Multimodal token generation (text+image+audio+video)
│   ├── reasoning.go           # Reasoning multi-turn with context accumulation
│   ├── network.go             # Client-perspective latency (RTT + bandwidth)
│   └── scenarios.go           # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
├── sim/kv/                    # P/D cross-instance KV transfer (planned, PR14)
├── sim/trace/                 # Decision trace recording (PR13)
│   ├── trace.go               # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go             # TraceSummary, Summarize()
├── sim/adapter/               # Framework adapters (planned, Phase 5)
├── model_configs/             # HuggingFace config.json files
├── defaults.yaml              # Trained coefficients, defaults
├── hardware_config.json       # GPU specifications
├── examples/                  # Example configuration files (policy-config.yaml, servegen-language.yaml)
├── testdata/goldendataset.json # Golden dataset for regression tests
└── docs/plans/                # Design documents
```

## Design Documents

### Guidelines and Templates (read these first)

- `docs/plans/2026-02-18-design-guidelines.md`: **BLIS Design Guidelines** — DES foundations (model scoping, event design, V&V), design doc authoring rules (abstraction levels, staleness test, four species), module architecture (two-layer architecture, target module map, contract template, real-system correspondence), extension framework (four extension types with recipes), anti-patterns with evidence. **Start here when designing anything new.**
- `docs/plans/macroplanprompt.md`: Template for macro-level planning (multi-PR feature expansions). Requires design guidelines as prerequisite. Enforces module contracts, model scoping, extension type classification, and abstraction level boundaries.
- `docs/plans/prmicroplanprompt-v2.md`: Template for micro-level (per-PR) planning with TDD tasks and behavioral contracts. This is where full code detail belongs.
- `docs/plans/prworkflow.md`: End-to-end PR workflow (worktree → plan → review → implement → review → audit → commit)

### Design Documents (per-feature)

- `docs/plans/2026-02-06-evolutionary-policy-optimization-design.md`: Full technical specification for cluster simulation extension
- `docs/plans/2026-02-11-macro-implementation-plan-v2.md`: Macro-level implementation plan (v3.4, 16 PRs across 6 phases, online routing architecture)
- `docs/plans/2026-02-13-simplification-assessment.md`: Architectural simplification assessment (constructor collapse, unified CLI, field privatization, interface dedup)
- `docs/plans/2026-02-16-workload-generator-design.md`: ServeGen-informed workload generator design (multi-client specs, arrival processes, calibration)
- `docs/plans/2026-02-17-pr13-decision-traces.md`: PR13 decision trace design (RoutingRecord, counterfactual analysis, TraceSummary)
- `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md`: Hardening design — antipattern elimination, extension scenario analysis, modularity improvements
- `docs/plans/pr12-architectural-predesign.md`: PR12 architectural pre-design — 6 binding design decisions for tiered KV cache (gold standard for decision records)

### Micro-Level Implementation Plans

- `docs/plans/pr10-workload-generator-plan.md`: PR10 micro-level implementation plan (workload generator)
