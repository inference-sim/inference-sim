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
- **admission.go**: `AdmissionPolicy` interface, `AlwaysAdmit`, `TokenBucket`, `NewAdmissionPolicy` factory
- **routing.go**: `RoutingPolicy` interface, `RoutingSnapshot`, `RoutingDecision`, `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity` templates, `NewRoutingPolicy` factory
- **priority.go**: `PriorityPolicy` interface with `ConstantPriority` and `SLOBasedPriority` templates, `NewPriorityPolicy` factory
- **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, and `SJFScheduler` templates, `NewScheduler` factory
- **event.go**: Event types (`ArrivalEvent`, `QueuedEvent`, `StepEvent`, `ScheduledEvent`, `RequestLeftEvent`, `PreemptionEvent`)
- **request.go**: Request lifecycle and state machine (queued → running → completed), `Priority` field for scheduler-aware ordering
- **kvcache.go**: Block-based KV cache with LRU eviction and prefix caching
- **batch.go**: Batch formation respecting token budgets and batch size limits
- **queue.go**: FIFO wait queue for pending requests

### Cluster Simulation (`sim/cluster/`)

Multi-replica extension using composition over the single-instance simulator:

- **instance.go**: `InstanceSimulator` wraps `sim.Simulator` via `NewInstanceSimulator(id, SimConfig)` with run-once guard and cluster-level accessors
- **cluster.go**: `ClusterSimulator` orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation
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

### Key Invariants to Maintain

- **Request lifecycle**: Requests transition queued → running → completed; requests not completed before horizon remain in current state
- **Clock monotonicity**: Simulation clock never decreases
- **KV cache conservation**: `allocated_blocks + free_blocks = total_blocks`
- **Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`

### Current Implementation Focus

Active development: Evolutionary Policy Optimization extension (see `docs/plans/2026-02-11-macro-implementation-plan-v2.md`):
- 16 PRs across 6 phases to extend BLIS to multi-replica cluster simulation
- **Research-ready checkpoint at ~5 weeks** (after Phase 2) enables early policy experiments
- **Completed:** PR1 (PartitionedRNG), PR2 (InstanceSimulator), PR3 (ClusterSimulator with shared-clock event loop, round-robin dispatch, metrics aggregation, golden dataset equivalence tests), PR4 (cluster control plane with online routing pipeline, SnapshotProvider, AdmissionPolicy with AlwaysAdmit + TokenBucket templates, cluster event queue), PR5 (architectural simplification: SimConfig struct, unified CLI path through ClusterSimulator, field privatization, AdmissionPolicy consolidated to `sim/admission.go`), PR6 (RoutingPolicy interface in `sim/routing.go` with RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity templates; RoutingSnapshot bridge type), PR7 (PriorityPolicy with ConstantPriority + SLOBasedPriority templates, InstanceScheduler with FCFS + PriorityFCFS + SJF templates, Priority field on Request, CLI flags `--priority-policy` and `--scheduler`)
- **Next:** PR8+ (policy bundles, raw metrics, tiered KV cache, decision traces)
- Will add to `sim/kv/`, `sim/workload/`, `sim/trace/` packages
- Each PR is CLI-exercisable immediately after merge (no scaffolding)

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
│   ├── root.go                # CLI commands and flags (always uses ClusterSimulator, --num-instances defaults to 1, --priority-policy, --scheduler)
│   └── default_config.go      # defaults.yaml loading
├── sim/                       # Core single-instance simulator
│   ├── simulator.go           # SimConfig struct, NewSimulator(SimConfig), event loop, batch formation, step execution
│   ├── admission.go           # AdmissionPolicy interface, AlwaysAdmit, TokenBucket, NewAdmissionPolicy factory
│   ├── routing.go             # RoutingPolicy interface, RoutingSnapshot, RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity
│   ├── priority.go            # PriorityPolicy interface, ConstantPriority, SLOBasedPriority, NewPriorityPolicy factory
│   ├── scheduler.go           # InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, NewScheduler factory
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
│   └── workload_config.go     # CSV trace loading and distribution-based workload generation
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator wrapper with run-once guard
│   ├── cluster.go             # ClusterSimulator: shared-clock event loop, online routing, aggregation
│   ├── cluster_event.go       # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent
│   ├── snapshot.go            # InstanceSnapshot, CachedSnapshotProvider, ObservabilityConfig
│   ├── deployment.go          # DeploymentConfig struct
│   └── workload.go            # Centralized request generation for cluster dispatch
├── sim/kv/                    # Tiered KV cache (planned, Phase 4)
├── sim/workload/              # Enhanced workload generation (planned, Phase 3)
├── sim/trace/                 # Decision traces (planned, Phase 4)
├── sim/adapter/               # Framework adapters (planned, Phase 5)
├── model_configs/             # HuggingFace config.json files
├── defaults.yaml              # Trained coefficients, defaults
├── hardware_config.json       # GPU specifications
├── testdata/goldendataset.json # Golden dataset for regression tests
└── docs/plans/                # Design documents
```

## Design Documents

- `docs/plans/2026-02-06-evolutionary-policy-optimization-design.md`: Full technical specification for cluster simulation extension
- `docs/plans/2026-02-11-macro-implementation-plan-v2.md`: Macro-level implementation plan (v3.0, 16 PRs across 6 phases, online routing architecture)
- `docs/plans/2026-02-13-simplification-assessment.md`: Architectural simplification assessment (constructor collapse, unified CLI, field privatization, interface dedup)
- `docs/plans/macroplanprompt.md`: Template for macro-level planning
- `docs/plans/prmicroplanprompt.md`: Template for micro-level (per-PR) planning with team-based agent process
