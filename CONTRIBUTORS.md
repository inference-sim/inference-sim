# Contributing to BLIS

This guide covers the engineering standards that keep BLIS (Blackbox Inference Simulator) correct and maintainable.

## Quick Start

```bash
# Build
go build -o simulation_worker main.go

# Test
go test ./...

# Lint
golangci-lint run ./...
```

All three must pass before submitting a PR.

## Development Workflow

Follow `docs/plans/prworkflow.md` for the complete PR lifecycle. The workflow applies to all PRs regardless of source (macro plan, issues, design docs, feature requests):

1. **Create worktree** — isolate your work from the main branch
2. **Write implementation plan** — behavioral contracts + TDD tasks using `docs/plans/prmicroplanprompt-v2.md`
3. **Review plan** — 5 focused passes (external review, cross-doc, architecture, codebase, structural)
4. **Implement** — test-first, one contract at a time
5. **Review code** — 4 focused passes (quality, test quality, getting-started, automated reviewer)
6. **Self-audit** — 9 dimensions of deliberate critical thinking (no automation)
7. **Commit, push, PR**

## Engineering Principles

### Test Laws, Not Just Values

Golden tests ("did the output change?") are necessary but insufficient. Pair each golden test with an **invariant test** that verifies a law the system must satisfy:

- **Request conservation**: injected == completed + queued + running
- **Request lifecycle**: queued -> running -> completed (no invalid transitions)
- **KV block conservation**: allocated + free == total
- **Causality**: arrival <= schedule <= completion
- **Clock monotonicity**: clock never decreases
- **Determinism**: same seed produces identical output

If your PR touches request lifecycle, KV cache, or metrics, add or extend invariant tests.

### BDD/TDD

1. Write behavioral contracts first (GIVEN/WHEN/THEN)
2. Write failing tests before implementation
3. Name tests `TestType_Scenario_Behavior`
4. Test observable behavior, not internal structure — tests should survive a refactor
5. Use table-driven tests for multiple scenarios

### Separation of Concerns

- **`sim/` is a library** — never call `os.Exit` or `logrus.Fatalf`. Return errors.
- **`cmd/` is the CLI** — validates user input, terminates on errors via `logrus.Fatalf`.
- **Cluster-level policies** (admission, routing) see global state via `*RouterState`.
- **Instance-level policies** (priority, scheduler) see only local data. Never leak cluster state.
- **Dependency direction**: `cmd/ → sim/cluster/ → sim/`. The `sim/` package never imports subpackages.

### Interface Design

- Prefer single-method interfaces (see `AdmissionPolicy`, `RoutingPolicy`)
- Query methods must be pure — no side effects, no destructive reads
- Factory functions must validate inputs (panic on programming errors, return error on user input)

### Error Handling

| Layer | Strategy | Example |
|-------|----------|---------|
| CLI (`cmd/`) | `logrus.Fatalf` for user errors | Invalid `--rate` value |
| Library (`sim/`) | `panic()` for invariant violations | Unknown policy name passed to factory |
| Library (`sim/`) | `error` return for recoverable failures | File I/O errors, parse errors |
| Runtime (`sim/`) | `bool` return for expected conditions | KV allocation failure (triggers preemption) |

**Never** use `continue` in an error path without propagating, counting, or documenting why it's safe.

## Antipattern Checklist

Before submitting a PR, verify:

- [ ] No silent `continue` or early `return` that drops data without error propagation
- [ ] No map iteration feeding float accumulation without sorted keys
- [ ] Every struct built in multiple places has a canonical constructor
- [ ] No `logrus.Fatalf` or `os.Exit` in library code (`sim/`, `sim/cluster/`, `sim/workload/`)
- [ ] Every new CLI flag validated for zero, negative, NaN, Inf
- [ ] Every loop that allocates resources handles mid-loop failure with rollback
- [ ] Invariant tests added or extended if request lifecycle, KV cache, or metrics are touched
- [ ] Golden dataset regenerated if output values changed (document regeneration command)

## Adding New Components

### New Policy Template

1. Implement the interface in the corresponding file (`sim/admission.go`, `sim/routing.go`, `sim/priority.go`, `sim/scheduler.go`)
2. Register in `sim/bundle.go` (valid names map + `IsValid*` function)
3. Add `case` to factory function
4. Add behavioral tests (`TestMyPolicy_Scenario_Behavior`)
5. Update CLAUDE.md and README

### New KV Cache Tier

1. Implement `KVStore` interface (9 methods; 10 after hardening PR adds `SetClock`)
2. Compose existing tiers using delegation pattern
3. Update `NewKVStore` factory with validation
4. Add CLI flags with validation (zero, negative, NaN/Inf guards)
5. Aggregate metrics from all tiers
6. Add conservation invariant tests

### New Trace Record Type

1. Define record struct in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. Add slice field to `SimulationTrace`
3. Add recording method
4. Hook into cluster event pipeline (`if cs.trace != nil`)
5. Update `Summarize()` aggregation
6. Add behavioral tests

## Code Style

- Composition over inheritance
- Timestamp-based event ordering via min-heap
- Partitioned RNG per subsystem for deterministic isolation
- BDD-style test naming: `TestType_Scenario_Behavior`
- Conventional commits: `feat(scope)`, `fix(scope)`, `refactor(scope)`, `test(scope)`, `docs(scope)`

## Detailed Reference

See `CLAUDE.md` for comprehensive project documentation including:
- Full code architecture and file organization
- CLI flags and configuration loading
- All key invariants
- Design document references
