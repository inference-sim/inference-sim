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

- **simulator.go**: Main `Simulator` struct and event loop (`Run()`), batch formation (`makeRunningBatch`), step execution
- **event.go**: Event types (`ArrivalEvent`, `StepEvent`, `ScheduledEvent`, `RequestLeftEvent`, `PreemptionEvent`)
- **request.go**: Request lifecycle and state machine (queued → running → completed)
- **kvcache.go**: Block-based KV cache with LRU eviction and prefix caching
- **batch.go**: Batch formation respecting token budgets and batch size limits
- **queue.go**: FIFO wait queue for pending requests

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

- **Request lifecycle**: Every request reaches exactly one terminal state (COMPLETED, REJECTED, TIMED_OUT)
- **Clock monotonicity**: Simulation clock never decreases
- **KV cache conservation**: `allocated_blocks + free_blocks = total_blocks`
- **Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`

### Current Implementation Focus

Active development: Evolutionary Policy Optimization extension (see `docs/plans/2026-02-11-macro-implementation-plan-v2.md`):
- 24 PRs across 6 phases to extend BLIS to multi-replica cluster simulation
- **Research-ready checkpoint at ~5 weeks** (after Phase 2) enables early policy experiments
- Next steps: Phase 1 (Foundation) - PartitionedRNG, InstanceSimulator, DeploymentConfig, ClusterSimulator
- Will add `sim/cluster/`, `sim/policy/`, `sim/kv/`, `sim/workload/`, `sim/trace/` packages
- Each policy type includes pathological templates for baseline testing and anomaly detection validation

### Code Style

- Use composition over inheritance (e.g., `InstanceSimulator` wraps existing `sim` components)
- Explicit tie-breaking for determinism (timestamp → event type priority → event ID)
- Partitioned RNG per subsystem to isolate randomness

## File Organization

```
inference-sim/
├── main.go                    # CLI entry point (Cobra)
├── cmd/
│   ├── root.go                # CLI commands and flags
│   └── default_config.go      # defaults.yaml loading
├── sim/                       # Core single-instance simulator
│   ├── simulator.go           # Event loop, batch formation
│   ├── event.go               # Event types
│   ├── request.go             # Request state machine
│   ├── kvcache.go             # KV cache with prefix caching
│   ├── batch.go               # Batch formation
│   ├── queue.go               # Wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection
│   ├── roofline_step.go       # Analytical latency estimation
│   ├── model_hardware_config.go # HF config, hardware specs
│   └── workload_config.go     # Workload generation
├── sim/cluster/               # Multi-replica extension (planned)
├── sim/policy/                # Pluggable policies (planned)
├── sim/kv/                    # Tiered KV cache (planned, Phase 4)
├── sim/workload/              # Enhanced workload generation (planned, Phase 3)
├── sim/trace/                 # Decision traces (planned, Phase 4)
├── sim/adapter/               # Framework adapters (planned, Phase 5)
├── model_configs/             # HuggingFace config.json files
├── defaults.yaml              # Trained coefficients, defaults
├── hardware_config.json       # GPU specifications
└── docs/plans/                # Design documents
```

## Design Documents

- `docs/plans/2026-02-06-evolutionary-policy-optimization-design.md`: Full technical specification for cluster simulation extension
- `docs/plans/2026-02-11-macro-implementation-plan-v2.md`: Macro-level implementation plan (v2.1, 24 PRs across 6 phases, research-first ordering)
- `docs/plans/macroplanprompt.md`: Template for macro-level planning
- `docs/plans/prmicroplanprompt.md`: Template for micro-level (per-PR) planning
