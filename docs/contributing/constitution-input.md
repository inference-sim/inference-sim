# BLIS Constitution Input

This document captures the core principles, rules, and guidelines distilled from BLIS's contributing standards. It is intended as input to `/speckit.constitution` to define the project's AI constitution.

---

## Project Identity

**BLIS (Blackbox Inference Simulator)** is a CPU-only, deterministic, discrete-event simulator (DES) for LLM inference serving systems. It models multi-instance clusters for capacity planning, policy optimization research, and performance prediction — without requiring real GPUs.

**Tech stack:** Go (single language, no exceptions). Dependencies: `cobra` (CLI), `logrus` (logging), `gopkg.in/yaml.v3` (config), `gonum` (numerics). No runtime C bindings, no CGO, no Python.

**Build:** `go build -o blis main.go`. **Test:** `go test ./...`. **Lint:** `golangci-lint run ./...` (v2.9.0, zero tolerance — CI fails on any lint issue).

---

## Architecture Constraints

### Package Dependency Direction

The dependency graph is strictly unidirectional and must never be violated:

```
cmd/ → sim/cluster/ → sim/
                    → sim/trace/
```

- `sim/` is a **library** — it never imports subpackages, never calls `os.Exit` or `logrus.Fatalf`, and never terminates the process. It returns errors to callers.
- `cmd/` is the **CLI layer** — the only place that may terminate the process.
- `sim/cluster/` orchestrates instances — it sees global state via `*RouterState`.
- Instance-level policies see **only local data** — they must never access cluster-level state.

### Two-Layer Architecture

Layer 1 (Simulation Kernel): domain-agnostic DES infrastructure — event queue, clock, RNG, statistics.
Layer 2 (Domain Modules): inference-specific logic, each behind an interface — Router, Scheduler, KV Cache Manager, Latency Model, Admission, Batch Formation.

The kernel never contains inference-specific logic. Domain modules never manage the event queue or clock directly.

### Module Boundaries

Each module is defined by a behavioral contract: what it observes, what it controls, what state it owns, what invariants it maintains, what events it produces/consumes, and how many files must change to add one more variant. New modules must have a no-op default so existing behavior is unchanged when not configured.

---

## Engineering Principles

### Separation of Concerns

- Cluster-level policies (admission, routing) receive `*RouterState` with global view.
- Instance-level policies (priority, scheduler) receive only local data.
- Bridge types (`RouterState`, `RoutingSnapshot`) live in `sim/` to avoid import cycles.
- Never let cluster state leak to instance-level code.

### Interface Design

- Prefer single-method interfaces (`AdmissionPolicy`, `RoutingPolicy`, `PriorityPolicy`, `InstanceScheduler`).
- Query methods must be **pure** — no side effects, no state mutation, no destructive reads. Separate `Get()` and `Consume()` for query-and-clear patterns.
- Factory functions must validate inputs: `IsValid*()` check + switch/case + panic on unknown.
- Interfaces defined by **behavioral contract**, not one implementation's data model.
- Methods operate within a **single module's responsibility** — no method spans scheduling + latency + metrics.
- New interfaces must accommodate at least **two implementations** (no methods that only make sense for one backend).

### Configuration Design

- Group configuration by module, not in monolithic structs.
- Each module's config must be independently specifiable and validatable.
- `SimConfig` is composed of 6 embedded sub-configs. Factory signatures accept the narrowest sub-config (e.g., `NewKVStore(KVCacheConfig)`).
- Use `*float64` (pointer) for YAML config fields where zero is a valid user value.
- Use `yaml.KnownFields(true)` strict parsing — typos must cause parse errors, not silent default behavior.

### Output Channel Separation

- **stdout** (deterministic results): simulation metrics JSON, fitness scores, trace summaries. Use `fmt.Println`/`fmt.Printf`.
- **stderr** (diagnostics): configuration echoes, progress markers, warnings, errors. Use `logrus.*`.
- Never write wall-clock timing or diagnostic text to stdout.

### Error Handling

| Layer | Strategy |
|-------|----------|
| CLI (`cmd/`) | `logrus.Fatalf` for user-facing errors |
| Library (`sim/`) | `panic()` for invariant violations and invalid constructor inputs |
| Library (`sim/`) | `error` return for recoverable failures (I/O, parse) |
| Runtime (`sim/`) | `bool` return for expected conditions (e.g., KV allocation failure) |

Never use `continue` in an error path without propagating, counting, or documenting why it is safe.

### Canonical Constructors

- Every struct constructed in multiple places needs a canonical constructor.
- Struct literals must appear in exactly one place.
- Before adding a field, grep for ALL construction sites.

### Determinism

- Same seed must produce **byte-identical stdout** across runs.
- All randomness flows through `PartitionedRNG` with named subsystems.
- Go map iteration feeding float accumulation or output ordering must sort keys first.
- New modules introducing randomness must declare their subsystem name.

---

## Testing Requirements

### BDD/TDD Practice

1. Write **behavioral contracts** first (GIVEN/WHEN/THEN format).
2. Implement **tests before code**.
3. Use **table-driven tests** for comprehensive coverage.
4. Test **laws, not just values** — every golden test needs a companion invariant test verifying a system law (conservation, causality, monotonicity).
5. Apply the **refactor survival test**: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?" If no, rewrite the test.

### Prohibited Assertion Patterns (Structural — break on refactor)

- Type assertions: `policy.(*ConcreteType)`
- Internal field access: `obj.internalField`
- Exact formula reproduction: `assert.Equal(score, 0.6*cache + 0.4*load)`

### Required Assertion Patterns (Behavioral — survive refactor)

- Observable output: `assert.Equal(policy.Compute(req, clock), 0.0)`
- Invariant verification: `assert.Equal(completed+queued+running+dropped, injected)`
- Ordering/ranking: `assert.True(scoreA > scoreB)`

### Test Performance Budget

- No single test exceeds 5 seconds without providing a `testing.Short()` fast-path skip.
- Total `go test ./...` must complete under 60 seconds.
- Benchmarks (`Benchmark*`) run only with `go test -bench=.`, never in the default test path.

---

## System Invariants

These properties must hold at all times. Every subsystem that has golden tests must also have invariant tests verifying them.

| ID | Invariant |
|----|-----------|
| **INV-1** | `injected_requests == completed + queued + running + dropped + timed_out` at simulation end. Full pipeline: `num_requests == injected + rejected`. |
| **INV-2** | Requests transition `queued → running → completed`. No invalid transitions. |
| **INV-3** | Simulation clock never decreases. Every event timestamp ≥ previous event timestamp. |
| **INV-4** | `allocated_blocks + free_blocks = total_blocks` at all times. |
| **INV-5** | `arrival_time ≤ enqueue_time ≤ schedule_time ≤ completion_time` for every request. |
| **INV-6** | Same seed produces byte-identical stdout across runs. |
| **INV-7** | Routing snapshot signals have tiered freshness: `InFlightRequests` is synchronous; `QueueDepth`, `BatchSize`, `KVUtilization` are Immediate (interval=0) or Periodic (interval>0). |
| **INV-8** | After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while work is waiting. |
| **INV-9** | Servability decisions (enqueue guard, admission, routing, priority) must not read `Request.OutputTokens`. Only the execution engine may access it. |

---

## Antipattern Rules

23 rules, each tracing to a real bug. All apply to every PR. Priority: **Critical** (R1, R4, R5, R6, R11, R19, R21) → **Important** (R2, R3, R7–R10, R13, R14, R17, R18, R20, R22, R23) → **Hygiene** (R12, R15, R16).

| Rule | Requirement |
|------|-------------|
| **R1** | Every error path must return error, panic, or increment counter — never silently drop data with a bare `continue` or early `return`. |
| **R2** | Go map iteration feeding float accumulation or output ordering must sort keys first. |
| **R3** | Every numeric parameter validated for zero, negative, NaN, Inf — at both CLI flags AND library constructors. |
| **R4** | Before adding a struct field, grep for ALL literal construction sites and update every one. |
| **R5** | Resource-allocating loops must rollback all mutations on mid-loop failure (transactional). |
| **R6** | `sim/` packages must never call `logrus.Fatalf` or `os.Exit` — return errors to callers. |
| **R7** | Every golden test needs a companion invariant test verifying a system law. |
| **R8** | Validation maps must be unexported; expose via `IsValid*()` accessor functions. |
| **R9** | Use `*float64` for YAML config fields where zero is a valid user-provided value. |
| **R10** | Use `yaml.KnownFields(true)` strict parsing for all YAML config loading. |
| **R11** | Runtime-derived division denominators must be guarded against zero. |
| **R12** | Regenerate and document the golden dataset command when output format or defaults change. |
| **R13** | New interfaces must work for ≥2 backends — no methods only meaningful for one. |
| **R14** | No method spans multiple module responsibilities — extract each concern. |
| **R15** | After completing a PR, grep for `planned for PR N` / `TODO.*PR N` and resolve stale references. |
| **R16** | New config parameters go into the appropriate module sub-config, not directly into `SimConfig`. |
| **R17** | Scorer authors must document which snapshot fields they read and their freshness tier. |
| **R18** | Check `cmd.Flags().Changed("<flag>")` before applying any default from `defaults.yaml` — never overwrite a user-provided flag. |
| **R19** | Loops whose exit condition depends on resource availability must have a circuit breaker (max iteration count, progress assertion, or bounded retry). |
| **R20** | Anomaly detectors must explicitly handle degenerate inputs: empty sets, single-instance concentration, all-zero distributions. |
| **R21** | Never use `range` over a slice that can shrink during iteration — use index-based `for i := 0; i < len(slice); i++`. |
| **R22** | Capacity pre-checks must be at least as permissive as the actual operation they guard — pre-check formula must account for all factors the allocation path accounts for. |
| **R23** | When multiple code paths produce the same output type, all paths must apply the same set of transformations — diff them explicitly. |

---

## Development Workflow Requirements

### Every PR Must

1. Start in an **isolated git worktree** before any work begins.
2. Have a written **implementation plan** with behavioral contracts (GIVEN/WHEN/THEN) and TDD task breakdown.
3. Have the plan reviewed (10 perspectives → convergence: zero CRITICAL + zero IMPORTANT) before implementation begins.
4. Implement tasks using **TDD**: write failing test → implement → pass → lint → commit.
5. Have the code reviewed (same 10-perspective convergence) before commit.
6. Pass a **pre-commit self-audit** (10 dimensions of critical thinking — logic, design, determinism, consistency, documentation, edge cases, test epistemology, construction sites, error paths, DRY).
7. Pass the verification gate: `go build ./...` + `go test ./... -count=1` + `golangci-lint run ./...` — all must report zero failures.

### DES Design Requirements

New modules, events, and state must address:
- **Model scoping**: what analysis question does this answer? What is modeled, simplified, omitted?
- **Event classification**: exogenous (arrival-driven) vs endogenous (state-driven). Priority constant for tie-breaking.
- **State vs statistics separation**: state variables evolve the system; statistics are derived outputs. Never mix them in one method.
- **Verification**: which invariants prove correctness? **Validation**: against what real-system data?
- **Randomness**: which `PartitionedRNG` subsystem? Does it support common-random-number experiments?

### Documentation Single Source of Truth

Every piece of documentation lives in exactly one canonical location. Before updating, identify the canonical source. Update the canonical source first, then working copies. The canonical-source pattern: include a header noting the authoritative file.

---

## Extension Patterns

Four ways to extend BLIS — choose the right one:

| Type | When | Key Requirement |
|------|------|-----------------|
| **Policy Template** | New algorithm behind frozen interface | Implements interface, deterministic, handles all edge cases defined by the interface |
| **Subsystem Module** | New module with own interface + events | Design doc required; no-op default mandatory; testable in isolation with mocks |
| **Backend Swap** | Alternative for internal module lacking interface | Phase A extracts interface; Phase B adds backend. Interface must accommodate both backends |
| **Tier Composition** | Layer behavior onto existing module | Satisfies same interface as inner module (Liskov); metrics aggregate across tiers |

Touch-point targets: new policy template ≤3 files, new latency backend ≤2 files, new config param ≤2 files. Exceeding targets requires explicit justification in the design doc.
