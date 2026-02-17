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
  --rate 10 --max-prompts 100 \
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

- **simulator.go**: `SimConfig` struct, `NewSimulator(SimConfig)` constructor, `Simulator` struct and event loop (`Run()`), batch formation (`makeRunningBatch`), step execution
- **admission.go**: `AdmissionPolicy` interface (accepts `*RouterState`), `AlwaysAdmit`, `TokenBucket`, `RejectAll`, `NewAdmissionPolicy` factory
- **routing.go**: `RoutingPolicy` interface (accepts `*RouterState`), `RoutingSnapshot`, `RoutingDecision` (with `Priority` hint), `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity`, `AlwaysBusiest` templates, `NewRoutingPolicy` factory
- **priority.go**: `PriorityPolicy` interface with `ConstantPriority`, `SLOBasedPriority`, and `InvertedSLO` templates, `NewPriorityPolicy` factory
- **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`, and `ReversePriority` templates, `NewScheduler` factory
- **router_state.go**: `RouterState` bridge type (Snapshots + Clock) for cluster-level policy interfaces
- **bundle.go**: `PolicyBundle` struct with YAML loading (`LoadPolicyBundle`), validation (`Validate`)
- **event.go**: Event types (`ArrivalEvent`, `QueuedEvent`, `StepEvent`, `ScheduledEvent`, `RequestLeftEvent`, `PreemptionEvent`)
- **request.go**: Request lifecycle and state machine (queued → running → completed), `Priority` field for scheduler-aware ordering
- **kvcache.go**: Block-based KV cache with LRU eviction and prefix caching
- **batch.go**: Batch formation respecting token budgets and batch size limits
- **queue.go**: FIFO wait queue for pending requests

### Cluster Simulation (`sim/cluster/`)

Multi-replica extension using composition over the single-instance simulator:

- **instance.go**: `InstanceSimulator` wraps `sim.Simulator` via `NewInstanceSimulator(id, SimConfig)` with run-once guard and cluster-level accessors
- **cluster.go**: `ClusterSimulator` orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation
- **metrics.go**: `RawMetrics`, `Distribution`, `FitnessResult`, `CollectRawMetrics`, `ComputeFitness`, anomaly detection
- **deployment.go**: `DeploymentConfig` struct with `ToSimConfig()` for per-instance construction
- **workload.go**: Centralized request generation (distribution-based or CSV traces) for cluster dispatch

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
Request Arrival → WaitQueue → Batch Formation → Step Execution → Completion
                     ↓              ↓
               KV Allocation   Latency Estimation (alpha/beta or roofline)
```

## Development Guidelines

### BDD/TDD Development

This project follows BDD/TDD practices. When implementing features:

1. **Write behavioral contracts first**: Define invariants and expected behavior in Gherkin-style scenarios
2. **Implement tests before code**: Tests verify contracts hold
3. **Use table-driven tests**: Go's table-driven test pattern for comprehensive coverage

### PR Workflow

Diligently follow the workflow in docs/plans/prworkflow.md. Before I approve any plan, validate it: 1) Check every task's dependencies — can each task actually start given what comes before it? 2) Verify all sections from the template are present and non-empty. 3) Read the executive summary as if you're a new team member — is it clear and human-readable? 4) Flag any tasks that seem under-specified for implementation. List all issues found.

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize results concisely. Never try to synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental summaries per agent rather than a consolidated report.

### Task Agent Guidelines

When using Task agents: 1) Do NOT poll TaskList repeatedly — check at reasonable intervals (every 30-60 seconds, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.

### Code Review Standards

During PR reviews, check for: unexported mutable maps, missing pointer types for YAML ambiguity, lack of strict YAML parsing, NaN/Inf validation gaps, and preservation of structural tests. Always run `go test ./...` and lint after fixes.
Add under ## PR Workflow section or as a sub-section ## Plan Document Updates.

### Macro Plan Updates

When asked to update the macro implementation plan, directly edit the document. Do NOT spend time re-reading all source documents or dispatching sub-agents to gather information you already have in context. Start writing immediately.

### Key Invariants to Maintain

- **Request lifecycle**: Requests transition queued → running → completed; requests not completed before horizon remain in current state
- **Clock monotonicity**: Simulation clock never decreases
- **KV cache conservation**: `allocated_blocks + free_blocks = total_blocks`
- **Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`



### Current Implementation Focus

Active development: Evolutionary Policy Optimization extension (see `docs/plans/2026-02-11-macro-implementation-plan-v2.md`):
- 16 PRs across 6 phases to extend BLIS to multi-replica cluster simulation
- **Research-ready checkpoint at ~5 weeks** (after Phase 2) enables early policy experiments
- **Completed:** PR1 (PartitionedRNG), PR2 (InstanceSimulator), PR3 (ClusterSimulator with shared-clock event loop, round-robin dispatch, metrics aggregation, golden dataset equivalence tests), PR4 (cluster control plane with online routing pipeline, SnapshotProvider, AdmissionPolicy with AlwaysAdmit + TokenBucket templates, cluster event queue), PR5 (architectural simplification: SimConfig struct, unified CLI path through ClusterSimulator, field privatization, AdmissionPolicy consolidated to `sim/admission.go`), PR6 (RoutingPolicy interface in `sim/routing.go` with RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity templates; RoutingSnapshot bridge type), PR7 (PriorityPolicy with ConstantPriority + SLOBasedPriority templates, InstanceScheduler with FCFS + PriorityFCFS + SJF templates, Priority field on Request, CLI flags `--priority-policy` and `--scheduler`), PR8 (RouterState bridge type in `sim/router_state.go`, PolicyBundle YAML config in `sim/bundle.go`, `--policy-config` CLI flag, AdmissionPolicy and RoutingPolicy accept `*RouterState`, `RoutingDecision.Priority` hint field, **INTERFACE FREEZE**), PR9 (RawMetrics with Distribution + FitnessResult, anomaly detection with priority inversion + HOL blocking counters, pathological templates: reject-all, inverted-slo, always-busiest, reverse-priority, `--fitness-weights` CLI flag, **RESEARCH-READY CHECKPOINT**)
- **Next:** Policy research experiments (research-ready checkpoint reached), or PR10 (Workload Generator, optional) and subsequent parallel tracks
- Will add to `sim/kv/`, `sim/workload/`, `sim/trace/` packages
- Each PR is CLI-exercisable immediately after merge (no scaffolding)

### Adding New Policy Templates

To add a new policy template (e.g., a new routing algorithm):

1. **Implement the interface** in the corresponding file:
   - `AdmissionPolicy` → `sim/admission.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `RoutingPolicy` → `sim/routing.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `PriorityPolicy` → `sim/priority.go` (instance-level: receives `req` + `clock` only)
   - `InstanceScheduler` → `sim/scheduler.go` (instance-level: receives `requests` + `clock` only)
   - Note: `RouterState` is a bridge type in `sim/` to avoid import cycles — see `sim/router_state.go`

2. **Register in three places** (all required):
   - Add policy name to valid names map in `sim/bundle.go` (e.g., `validRoutingPolicies`) and corresponding `IsValid*` function
   - Add `case` to factory function in the same policy file (e.g., `NewRoutingPolicy` in `sim/routing.go`)
   - Update CLI validation error message in `cmd/root.go` to list the new policy name

3. **Add tests** following BDD naming: `TestMyPolicy_Scenario_Behavior`
   - Test observable behavior, not internal structure
   - Include empty-snapshots panic test for routing policies (defensive programming convention)
   - Use `&RouterState{Snapshots: snapshots, Clock: clock}` in test setup

4. **Update documentation**: CLAUDE.md file organization, README policy lists

Examples:
- See `RejectAll` in `sim/admission.go` for a simple admission template (constant return)
- See `PrefixAffinity` in `sim/routing.go` for a stateful routing policy with LeastLoaded fallback

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
│   ├── root.go                # CLI commands and flags (always uses ClusterSimulator, --num-instances defaults to 1, --priority-policy, --scheduler, --policy-config)
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
│   ├── request.go             # Request state machine (queued → running → completed), Priority field
│   ├── kvcache.go             # Block-based KV cache with LRU eviction and prefix caching
│   ├── batch.go               # Batch struct
│   ├── queue.go               # FIFO wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct
│   ├── rng.go                 # PartitionedRNG for deterministic multi-subsystem simulation
│   ├── roofline_step.go       # Analytical FLOPs/bandwidth latency estimation
│   ├── model_hardware_config.go # HFConfig, ModelConfig, HardwareCalib structs
│   ├── workload_config.go     # CSV trace loading and distribution-based workload generation
│   └── internal/testutil/     # Shared test infrastructure (golden dataset loading)
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator wrapper with run-once guard
│   ├── cluster.go             # ClusterSimulator: shared-clock event loop, online routing, aggregation
│   ├── cluster_event.go       # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent
│   ├── snapshot.go            # InstanceSnapshot, CachedSnapshotProvider, ObservabilityConfig
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, anomaly detection
│   ├── deployment.go          # DeploymentConfig struct
│   └── workload.go            # Centralized request generation for cluster dispatch
├── sim/kv/                    # Tiered KV cache (planned, Phase 4)
├── sim/workload/              # Enhanced workload generation (planned, Phase 3)
├── sim/trace/                 # Decision traces (planned, Phase 4)
├── sim/adapter/               # Framework adapters (planned, Phase 5)
├── model_configs/             # HuggingFace config.json files
├── defaults.yaml              # Trained coefficients, defaults
├── hardware_config.json       # GPU specifications
├── examples/                  # Example configuration files (policy-config.yaml)
├── testdata/goldendataset.json # Golden dataset for regression tests
└── docs/plans/                # Design documents
```

## Design Documents

- `docs/plans/2026-02-06-evolutionary-policy-optimization-design.md`: Full technical specification for cluster simulation extension
- `docs/plans/2026-02-11-macro-implementation-plan-v2.md`: Macro-level implementation plan (v3.0, 16 PRs across 6 phases, online routing architecture)
- `docs/plans/2026-02-13-simplification-assessment.md`: Architectural simplification assessment (constructor collapse, unified CLI, field privatization, interface dedup)
- `docs/plans/macroplanprompt.md`: Template for macro-level planning
- `docs/plans/prmicroplanprompt.md`: Template for micro-level (per-PR) planning with team-based agent process
