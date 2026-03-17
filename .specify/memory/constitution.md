<!--
SYNC IMPACT REPORT
==================
Version change: (unversioned template) → 1.0.0
Modified principles: N/A (initial population from template)
Added sections:
  - Core Principles (I–VIII): Architecture & Layering, Determinism, Interface Design,
    BDD/TDD, Error Handling, Configuration Discipline, System Invariants,
    Antipattern Prevention
  - Development Workflow
  - Extension Framework
  - Governance
Removed sections: N/A
Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check section already present;
     principle names now concrete (no changes needed)
  ✅ .specify/templates/spec-template.md — no constitution-specific references; no change
  ✅ .specify/templates/tasks-template.md — no constitution-specific references; no change
Follow-up TODOs:
  - TODO(RATIFICATION_DATE): no historical adoption date found; set to initial-population date
    2026-03-13. Update if a prior adoption date is known.
-->

# BLIS Constitution

## Core Principles

### I. Architecture & Layering

BLIS uses a strict two-layer architecture. The **Simulation Kernel** (Layer 1) is a
domain-agnostic DES infrastructure — event queue, clock, RNG, statistics. **Domain
Modules** (Layer 2) implement inference-specific logic, each behind an interface:
Router, Scheduler, KV Cache Manager, Latency Model, Admission, Batch Formation.

Non-negotiable rules:
- The kernel MUST NOT contain inference-specific logic.
- Domain modules MUST NOT manage the event queue or clock directly.
- Package dependency direction MUST be strictly unidirectional:
  `cmd/ → sim/cluster/ → sim/` and `cmd/ → sim/cluster/ → sim/trace/`.
- `sim/` is a library; it MUST never call `os.Exit`, `logrus.Fatalf`, or terminate
  the process. It MUST return errors to callers.
- `cmd/` is the only layer permitted to terminate the process.
- Instance-level policies MUST access only local data; cluster-level state MUST NOT
  leak into instance-level code.
- Bridge types (`RouterState`, `RoutingSnapshot`) MUST live in `sim/` to prevent
  import cycles.

*Rationale*: Unidirectional dependencies eliminate circular imports and make modules
independently testable and replaceable.

### II. Determinism

Same seed MUST produce byte-identical stdout across runs.

Non-negotiable rules:
- All randomness MUST flow through `PartitionedRNG` with named per-module subsystems.
- Go map iteration feeding float accumulation or output ordering MUST sort keys first.
- New modules introducing randomness MUST declare their `PartitionedRNG` subsystem name.
- Wall-clock timing and diagnostics MUST go to stderr. Simulation metrics MUST go
  to stdout.

*Rationale*: Reproducibility is essential for capacity planning, policy optimization
comparisons, and regression detection.

### III. Interface & Module Design

Interfaces define behavioral contracts, not implementation convenience.

Non-negotiable rules:
- Prefer single-method interfaces (`AdmissionPolicy`, `RoutingPolicy`, `PriorityPolicy`,
  `InstanceScheduler`).
- Query methods MUST be **pure** — no side effects, no state mutation, no destructive
  reads. Use separate `Get()` and `Consume()` for query-and-clear patterns.
- Factory functions MUST validate inputs: `IsValid*()` check + switch/case + panic on
  unknown variant.
- Interfaces MUST be defined by behavioral contract, not one implementation's data
  model.
- Methods MUST operate within a single module's responsibility — no method spans
  scheduling + latency + metrics.
- New interfaces MUST accommodate at least two implementations (no methods that only
  make sense for one backend).
- Validation maps MUST be unexported; expose only via `IsValid*()` accessor functions.

*Rationale*: Single-method interfaces are the smallest stable unit of abstraction and
maximally composable.

### IV. Test-First (BDD/TDD) (NON-NEGOTIABLE)

Tests are behavioral contracts in code, written before implementation.

Non-negotiable rules:
1. Write **behavioral contracts** first (GIVEN/WHEN/THEN format) before any code.
2. Implement **tests before code**: tests must fail before implementation begins.
3. Use **table-driven tests** for comprehensive scenario coverage.
4. Test **laws, not values** — every golden test MUST have a companion invariant test
   verifying a system law (conservation, causality, monotonicity).
5. Apply the **refactor survival test**: "Would this test still pass if the
   implementation were completely rewritten but the behavior preserved?" If no,
   rewrite the test.
6. No single test MUST exceed 5 seconds without a `testing.Short()` fast-path skip.
7. Total `go test ./...` MUST complete under 60 seconds.
8. Benchmarks MUST run only with `go test -bench=.`, never in the default test path.

Prohibited assertion patterns (structural — break on refactor):
- Type assertions: `policy.(*ConcreteType)`
- Internal field access: `obj.internalField`
- Exact formula reproduction: `assert.Equal(score, 0.6*cache + 0.4*load)`

Required assertion patterns (behavioral — survive refactor):
- Observable output: `assert.Equal(policy.Compute(req, clock), 0.0)`
- Invariant verification: `assert.Equal(completed+queued+running+dropped, injected)`
- Ordering/ranking: `assert.True(scoreA > scoreB)`

*Rationale*: Tests that assert implementation internals break on every refactor,
creating friction instead of confidence.

### V. Error Handling

Error strategies are determined by layer, not convenience.

| Layer | Strategy |
|-------|----------|
| CLI (`cmd/`) | `logrus.Fatalf` for user-facing errors |
| Library (`sim/`) | `panic()` for invariant violations and invalid constructor inputs |
| Library (`sim/`) | `error` return for recoverable failures (I/O, parse) |
| Runtime (`sim/`) | `bool` return for expected conditions (e.g., KV allocation failure) |

Non-negotiable rules:
- `sim/` packages MUST NEVER call `logrus.Fatalf` or `os.Exit`.
- Every error path MUST return error, panic, or increment a counter — NEVER silently
  drop data with a bare `continue` or early `return`.
- Resource-allocating loops MUST rollback all mutations on mid-loop failure
  (transactional semantics).
- Loops whose exit condition depends on resource availability MUST have a circuit
  breaker (max iteration count, progress assertion, or bounded retry).
- Runtime-derived division denominators MUST be guarded against zero.

*Rationale*: Silent data loss and unbounded retry loops are the two most common
sources of subtle simulation bugs in this codebase.

### VI. Configuration Discipline

Configuration is structured by module, strictly parsed, and pointer-typed where zero
is meaningful.

Non-negotiable rules:
- Configuration MUST be grouped by module, not in monolithic structs.
- `SimConfig` is composed of exactly 6 embedded sub-configs; factory signatures
  MUST accept the narrowest sub-config (e.g., `NewKVStore(KVCacheConfig)`).
- New config parameters MUST go into the appropriate module sub-config, not directly
  into `SimConfig`.
- Use `*float64` (pointer) for YAML config fields where zero is a valid user value.
- Use `yaml.KnownFields(true)` strict parsing — typos MUST cause parse errors, not
  silent default behavior.
- `cmd.Flags().Changed("<flag>")` MUST be checked before applying any default from
  `defaults.yaml` — never overwrite a user-provided flag.

*Rationale*: Silent YAML field mismatch and zero-vs-unset ambiguity are reproducible
bug sources; strict parsing and pointer types eliminate them.

### VII. System Invariants

These properties MUST hold at all times. Every subsystem with golden tests MUST also
have invariant tests verifying them.

| ID | Invariant |
|----|-----------|
| **INV-1** | `injected_requests == completed + queued + running + dropped + timed_out` at simulation end. Full pipeline: `num_requests == injected + rejected`. |
| **INV-2** | Requests transition `queued → running → completed`. No invalid transitions. |
| **INV-3** | Simulation clock never decreases. Every event timestamp ≥ previous event timestamp. |
| **INV-4** | `allocated_blocks + free_blocks = total_blocks` at all times. |
| **INV-5** | `arrival_time ≤ enqueue_time ≤ schedule_time ≤ completion_time` for every request. |
| **INV-6** | Same seed produces byte-identical stdout across runs. |
| **INV-7** | Routing snapshot signals have tiered freshness: `InFlightRequests` is synchronous; `QueueDepth`, `BatchSize`, `KVUtilization` are Immediate (interval=0) or Periodic (interval>0). |
| **INV-8** | After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator MUST NOT idle while work is waiting. |
| **INV-9** | Servability decisions (enqueue guard, admission, routing, priority) MUST NOT read `Request.OutputTokens`. Only the execution engine may access it. |

*Rationale*: Invariants are the ground truth for correctness. Golden tests verify
output stability; invariant tests verify semantic correctness.

### VIII. Antipattern Prevention

23 rules, each tracing to a real bug. All apply to every PR.

Priority: **Critical** (R1, R4, R5, R6, R11, R19, R21) → **Important** (R2, R3,
R7–R10, R13, R14, R17, R18, R20, R22, R23) → **Hygiene** (R12, R15, R16).

| Rule | Requirement |
|------|-------------|
| **R1** | Every error path must return error, panic, or increment counter — never silently drop data. |
| **R2** | Map iteration feeding float accumulation or output ordering must sort keys first. |
| **R3** | Every numeric parameter validated for zero, negative, NaN, Inf — at CLI flags AND library constructors. |
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
| **R18** | Check `cmd.Flags().Changed("<flag>")` before applying any default from `defaults.yaml`. |
| **R19** | Loops whose exit condition depends on resource availability must have a circuit breaker. |
| **R20** | Anomaly detectors must explicitly handle degenerate inputs: empty sets, single-instance concentration, all-zero distributions. |
| **R21** | Never use `range` over a slice that can shrink during iteration — use index-based `for i := 0; i < len(slice); i++`. |
| **R22** | Capacity pre-checks must be at least as permissive as the actual operation they guard. |
| **R23** | When multiple code paths produce the same output type, all paths must apply the same set of transformations — diff them explicitly. |

## Development Workflow

Every PR MUST follow this sequence — no exceptions:

1. Start in an **isolated git worktree** before any work begins.
2. Have a written **implementation plan** with behavioral contracts (GIVEN/WHEN/THEN)
   and TDD task breakdown.
3. Have the plan reviewed (10 perspectives → convergence: zero CRITICAL + zero
   IMPORTANT) before implementation begins.
4. Implement tasks using **TDD**: write failing test → implement → pass → lint →
   commit.
5. Have the code reviewed (same 10-perspective convergence) before commit.
6. Pass a **pre-commit self-audit** (10 dimensions: logic, design, determinism,
   consistency, documentation, edge cases, test epistemology, construction sites,
   error paths, DRY).
7. Pass the verification gate: `go build ./...` + `go test ./... -count=1` +
   `golangci-lint run ./...` — all MUST report zero failures.

**DES Design Requirements** — new modules, events, and state MUST address:
- **Model scoping**: what analysis question does this answer? What is modeled,
  simplified, omitted?
- **Event classification**: exogenous (arrival-driven) vs endogenous (state-driven).
  Priority constant for tie-breaking.
- **State vs statistics**: state variables evolve the system; statistics are derived
  outputs. MUST NOT be mixed in one method.
- **Verification**: which invariants prove correctness? **Validation**: against what
  real-system data?
- **Randomness**: which `PartitionedRNG` subsystem? Does it support common-random-
  number experiments?

**Documentation single source of truth**: every piece of documentation lives in
exactly one canonical location. Before updating, identify the canonical source.
Update the canonical source first, then working copies.

## Extension Framework

Four extension types — choose the right one:

| Type | When | Key Requirement |
|------|------|-----------------|
| **Policy Template** | New algorithm behind frozen interface | Implements interface, deterministic, handles all edge cases |
| **Subsystem Module** | New module with own interface + events | Design doc required; no-op default mandatory; testable in isolation |
| **Backend Swap** | Alternative for internal module lacking interface | Phase A extracts interface; Phase B adds backend. Interface accommodates both backends |
| **Tier Composition** | Layer behavior onto existing module | Satisfies same interface as inner module (Liskov); metrics aggregate across tiers |

Touch-point targets: new policy template ≤3 files, new latency backend ≤2 files,
new config param ≤2 files. Exceeding targets requires explicit justification in the
design doc.

## Governance

This Constitution summarizes the project's core principles; canonical sources
(`rules.md`, `invariants.md`, `principles.md`) are authoritative for detailed
definitions. Amendments require documentation, team approval, and a migration
plan for affected artifacts.

**Amendment procedure**:
1. Propose change with rationale (new principle, rule update, or removal).
2. Convergence review (zero CRITICAL + zero IMPORTANT across 8 perspectives).
3. Update this constitution and all dependent templates atomically in a single PR.
4. Increment version per the versioning policy below.

**Versioning policy**:
- MAJOR: backward-incompatible principle removals or redefinitions.
- MINOR: new principle or section added, or materially expanded guidance.
- PATCH: clarifications, wording, typo fixes, non-semantic refinements.

**Compliance review**: all PRs and reviews MUST verify compliance with all 8
Principles (I–VIII) and all 23 antipattern rules (R1–R23). The plan-template
Constitution Check gate enforces this at plan time. Complexity violations MUST be
justified in a Complexity Tracking table.

**Tech stack constraint**: Go is the single implementation language — no exceptions.
No runtime C bindings, no CGO, no Python. Dependencies: `cobra`, `logrus`,
`gopkg.in/yaml.v3`, `gonum`. Build: `go build -o blis main.go`.
Lint: `golangci-lint run ./...` (v2.9.0, zero tolerance).

**Version**: 1.0.0 | **Ratified**: 2026-03-13 | **Last Amended**: 2026-03-13
