# Micro Plan: Cohort Multi-Turn Support (Issue #847)

**Goal:** Add multi-turn / reasoning fields to `CohortSpec` so a single cohort entry can express N concurrent multi-turn sessions without duplicating `ClientSpec` entries.
**Source:** https://github.com/inference-sim/inference-sim/issues/847
**Closes:** #847
**PR size:** Medium (4 files changed)
**Date:** 2026-03-26

---

## Part 1 — Design Validation

### A. Executive Summary

`CohortSpec` is a structural subset of `ClientSpec`. The `ExpandCohorts` function builds each `ClientSpec` by copying only the fields that exist on `CohortSpec`; six fields (`PrefixLength`, `Reasoning`, `ClosedLoop`, `Timeout`, `Network`, `Multimodal`) that exist on `ClientSpec` have no counterpart on `CohortSpec` and are therefore silently dropped during expansion. This PR adds those six fields to `CohortSpec`, copies them in `ExpandCohorts`, validates the three that carry invariant constraints (`PrefixLength >= 0`, `Reasoning.MultiTurn.MaxRounds >= 1`, `Timeout >= 0`), and documents the new fields in the workloads guide.

### B. Behavioral Contracts

**BC-1 (Field propagation):** GIVEN a `CohortSpec` with any of `PrefixLength`, `Reasoning`, `ClosedLoop`, `Timeout`, `Network`, or `Multimodal` set, WHEN `ExpandCohorts` runs, THEN every expanded `ClientSpec` carries the same value for those fields. Pointer fields are shared (same pointer — safe because `GenerateRequests` only reads them).

**BC-2 (Reasoning validation):** GIVEN a `CohortSpec` with `reasoning.multi_turn.max_rounds < 1`, WHEN `spec.Validate()` runs, THEN it returns an error containing `"max_rounds must be >= 1"`.

**BC-3 (Timeout validation):** GIVEN a `CohortSpec` with `timeout < 0`, WHEN `spec.Validate()` runs, THEN it returns an error containing `"timeout must be non-negative"`.

**BC-4 (YAML round-trip):** GIVEN a cohort YAML with `reasoning.multi_turn` fields, WHEN unmarshalled into `WorkloadSpec` and expanded, THEN expanded clients have `Reasoning != nil` with correct `MaxRounds`.

**BC-5 (closed_loop propagation):** GIVEN a `CohortSpec` with `closed_loop: false`, WHEN expanded, THEN every client has `ClosedLoop` pointing to `false`.

**BC-6 (Nil fields = no injection):** GIVEN a `CohortSpec` with none of the new fields set (nil/zero), WHEN expanded, THEN all 6 new fields on every expanded `ClientSpec` are nil/zero (no accidental injection). The existing `TestExpandCohorts_Determinism_SameSeedSameOutput` covers the stronger INV-6 byte-identity guarantee.

**BC-7 (PrefixLength validation):** GIVEN a `CohortSpec` with `prefix_length < 0`, WHEN `spec.Validate()` runs, THEN it returns an error containing `"prefix_length must be non-negative"`.

### C. Component Interaction

`ExpandCohorts` is a pure function: `[]CohortSpec × int64 → []ClientSpec`. Adding fields to the struct and copying them in the loop is entirely local to `sim/workload/`. No other package is affected — callers in `sim/workload/generator.go` call `ExpandCohorts` and receive the richer `ClientSpec` transparently.

### D. Risks

- **R4 (construction site drift):** `CohortSpec` is constructed in `cohort_test.go` and `generator_test.go` — see Appendix for complete list. All sites use named fields so new fields default to nil/zero safely.
- **Pointer aliasing:** All new pointer fields are safe to share across expanded clients because `GenerateRequests` reads but never mutates `Reasoning`, `ClosedLoop`, `Timeout`, `Network`, `Multimodal`. A comment in `ExpandCohorts` at the copy site documents this invariant.

### E. Test Strategy

| Contract | Test | File |
|----------|------|------|
| BC-1 (all 6 fields) | `TestExpandCohorts_NewFieldsPropagate` | `cohort_test.go` |
| BC-2 (reasoning validation) | `TestCohortValidation_InvalidMaxRounds_ReturnsError` | `cohort_test.go` |
| BC-3 (timeout validation) | `TestCohortValidation_NegativeTimeout_ReturnsError` | `cohort_test.go` |
| BC-4 (YAML round-trip) | `TestCohortSpec_YAMLRoundTrip_Reasoning` | `cohort_test.go` |
| BC-5 (closed_loop) | embedded in `TestExpandCohorts_NewFieldsPropagate` | `cohort_test.go` |
| BC-6 (nil fields = no injection) | `TestExpandCohorts_NilNewFields_NoChange` | `cohort_test.go` |
| BC-7 (prefix_length validation) | `TestCohortValidation_NegativePrefixLength_ReturnsError` | `cohort_test.go` |

---

## Part 2 — Executable Tasks

### Task 1: Write failing tests

**Test file:** `sim/workload/cohort_test.go`

Add the following tests (requires `"gopkg.in/yaml.v3"` import alongside the existing `"strings"` import). They will fail to compile until Task 2 adds the struct fields.

```go
// BC-4: YAML round-trip for reasoning on cohort.
// Uses strict decoder (KnownFields=true) to match LoadWorkloadSpec production path (R10).
func TestCohortSpec_YAMLRoundTrip_Reasoning(t *testing.T) {
    input := `
version: "2"
seed: 42
aggregate_rate: 10
num_requests: 50
cohorts:
  - id: "chat"
    population: 3
    slo_class: "standard"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 256, std_dev: 64, min: 2, max: 1024 }
    output_distribution:
      type: gaussian
      params: { mean: 128, std_dev: 32, min: 2, max: 512 }
    reasoning:
      multi_turn:
        max_rounds: 5
        think_time_us: 3000000
`
    var spec WorkloadSpec
    decoder := yaml.NewDecoder(strings.NewReader(input))
    decoder.KnownFields(true)
    if err := decoder.Decode(&spec); err != nil {
        t.Fatalf("strict decode failed: %v", err)
    }
    if err := spec.Validate(); err != nil {
        t.Fatalf("validate failed: %v", err)
    }
    clients := ExpandCohorts(spec.Cohorts, spec.Seed)
    if len(clients) != 3 {
        t.Fatalf("expected 3 clients, got %d", len(clients))
    }
    for _, c := range clients {
        if c.Reasoning == nil {
            t.Errorf("client %s: Reasoning is nil", c.ID)
            continue
        }
        if c.Reasoning.MultiTurn == nil {
            t.Errorf("client %s: Reasoning.MultiTurn is nil", c.ID)
            continue
        }
        if c.Reasoning.MultiTurn.MaxRounds != 5 {
            t.Errorf("client %s: MaxRounds = %d, want 5", c.ID, c.Reasoning.MultiTurn.MaxRounds)
        }
        if c.Reasoning.MultiTurn.ThinkTimeUs != 3_000_000 {
            t.Errorf("client %s: ThinkTimeUs = %d, want 3000000", c.ID, c.Reasoning.MultiTurn.ThinkTimeUs)
        }
    }
}

// BC-1 + BC-5: All new fields propagate including closed_loop
func TestExpandCohorts_NewFieldsPropagate(t *testing.T) {
    closedLoop := false
    timeout := int64(5_000_000)
    cohorts := []CohortSpec{
        {
            ID: "multi-turn", Population: 3, RateFraction: 1.0,
            Arrival:    ArrivalSpec{Process: "poisson"},
            InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
            OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
            PrefixLength: 64,
            Reasoning: &ReasoningSpec{
                MultiTurn: &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1_000_000},
            },
            ClosedLoop: &closedLoop,
            Timeout:    &timeout,
            Network:    &NetworkSpec{RTTMs: 10.0},
            Multimodal: &MultimodalSpec{
                TextDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 64, "std_dev": 10, "min": 1, "max": 200}},
            },
        },
    }
    clients := ExpandCohorts(cohorts, 42)
    if len(clients) != 3 {
        t.Fatalf("expected 3 clients, got %d", len(clients))
    }
    for _, c := range clients {
        if c.PrefixLength != 64 {
            t.Errorf("client %s: PrefixLength = %d, want 64", c.ID, c.PrefixLength)
        }
        if c.Reasoning == nil || c.Reasoning.MultiTurn == nil || c.Reasoning.MultiTurn.MaxRounds != 3 {
            t.Errorf("client %s: Reasoning not propagated correctly", c.ID)
        }
        if c.ClosedLoop == nil || *c.ClosedLoop != false {
            t.Errorf("client %s: ClosedLoop not propagated correctly", c.ID)
        }
        if c.Timeout == nil || *c.Timeout != 5_000_000 {
            t.Errorf("client %s: Timeout not propagated correctly", c.ID)
        }
        if c.Network == nil || c.Network.RTTMs != 10.0 {
            t.Errorf("client %s: Network not propagated correctly", c.ID)
        }
        if c.Multimodal == nil {
            t.Errorf("client %s: Multimodal not propagated", c.ID)
        }
    }
}

// BC-6: nil new fields → no accidental injection into existing behavior
func TestExpandCohorts_NilNewFields_NoChange(t *testing.T) {
    cohorts := []CohortSpec{
        {
            ID: "baseline", Population: 2, RateFraction: 1.0,
            Arrival:    ArrivalSpec{Process: "poisson"},
            InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
            OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
            // new fields intentionally absent (zero/nil)
        },
    }
    clients := ExpandCohorts(cohorts, 42)
    for _, c := range clients {
        if c.Reasoning != nil {
            t.Errorf("client %s: Reasoning should be nil", c.ID)
        }
        if c.ClosedLoop != nil {
            t.Errorf("client %s: ClosedLoop should be nil", c.ID)
        }
        if c.Timeout != nil {
            t.Errorf("client %s: Timeout should be nil", c.ID)
        }
        if c.Network != nil {
            t.Errorf("client %s: Network should be nil", c.ID)
        }
        if c.Multimodal != nil {
            t.Errorf("client %s: Multimodal should be nil", c.ID)
        }
        if c.PrefixLength != 0 {
            t.Errorf("client %s: PrefixLength should be 0", c.ID)
        }
    }
}

// BC-2: reasoning.multi_turn.max_rounds < 1 fails validation
func TestCohortValidation_InvalidMaxRounds_ReturnsError(t *testing.T) {
    spec := &WorkloadSpec{
        Version:       "2",
        AggregateRate: 10.0,
        Cohorts: []CohortSpec{
            {
                ID: "bad", Population: 2, RateFraction: 1.0,
                Arrival:    ArrivalSpec{Process: "poisson"},
                InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
                OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
                Reasoning:  &ReasoningSpec{MultiTurn: &MultiTurnSpec{MaxRounds: 0}},
            },
        },
    }
    err := spec.Validate()
    if err == nil {
        t.Fatal("expected validation error for max_rounds=0")
    }
    if !strings.Contains(err.Error(), "max_rounds must be >= 1") {
        t.Errorf("unexpected error: %v", err)
    }
}

// BC-3: negative timeout fails validation
func TestCohortValidation_NegativeTimeout_ReturnsError(t *testing.T) {
    neg := int64(-1)
    spec := &WorkloadSpec{
        Version:       "2",
        AggregateRate: 10.0,
        Cohorts: []CohortSpec{
            {
                ID: "bad-timeout", Population: 2, RateFraction: 1.0,
                Arrival:    ArrivalSpec{Process: "poisson"},
                InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
                OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
                Timeout: &neg,
            },
        },
    }
    err := spec.Validate()
    if err == nil {
        t.Fatal("expected validation error for negative timeout")
    }
    if !strings.Contains(err.Error(), "timeout must be non-negative") {
        t.Errorf("unexpected error: %v", err)
    }
}

// BC-7: negative prefix_length fails validation
func TestCohortValidation_NegativePrefixLength_ReturnsError(t *testing.T) {
    spec := &WorkloadSpec{
        Version:       "2",
        AggregateRate: 10.0,
        Cohorts: []CohortSpec{
            {
                ID: "bad-prefix", Population: 2, RateFraction: 1.0,
                Arrival:      ArrivalSpec{Process: "poisson"},
                InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
                OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
                PrefixLength: -1,
            },
        },
    }
    err := spec.Validate()
    if err == nil {
        t.Fatal("expected validation error for negative prefix_length")
    }
    if !strings.Contains(err.Error(), "prefix_length must be non-negative") {
        t.Errorf("unexpected error: %v", err)
    }
}
```

**Run test to verify failure:**
```bash
go test ./sim/workload/... -run "TestCohortSpec_YAMLRoundTrip_Reasoning|TestExpandCohorts_NewFieldsPropagate|TestExpandCohorts_NilNewFields_NoChange|TestCohortValidation_InvalidMaxRounds|TestCohortValidation_NegativeTimeout|TestCohortValidation_NegativePrefixLength"
# Expected: compilation error — fields don't exist on CohortSpec yet
```

**Commit:** `test(workload): add failing tests for cohort multi-turn field propagation (BC-1..7)`

---

### Task 2: Add fields to CohortSpec, validation, and ExpandCohorts

**File 1: `sim/workload/spec.go`**

1. Add 6 fields to `CohortSpec` struct (after `Drain *DrainSpec` at line ~70, before closing brace):

```go
    PrefixLength int             `yaml:"prefix_length,omitempty"`
    Reasoning    *ReasoningSpec  `yaml:"reasoning,omitempty"`
    ClosedLoop   *bool           `yaml:"closed_loop,omitempty"`
    Timeout      *int64          `yaml:"timeout,omitempty"`
    Network      *NetworkSpec    `yaml:"network,omitempty"`
    Multimodal   *MultimodalSpec `yaml:"multimodal,omitempty"`
```

2. In `validateCohort` (after the `Drain` validation block, ~line 336, before the `return nil`), add:

```go
    // Validate PrefixLength (mirrors validateClient line 255)
    if c.PrefixLength < 0 {
        return fmt.Errorf("%s: prefix_length must be non-negative, got %d", prefix, c.PrefixLength)
    }
    // Validate Timeout if specified (mirrors validateClient line 265)
    if c.Timeout != nil && *c.Timeout < 0 {
        return fmt.Errorf("%s: timeout must be non-negative, got %d", prefix, *c.Timeout)
    }
    // Validate MaxRounds for reasoning/multi-turn (mirrors validateClient line 269)
    if c.Reasoning != nil && c.Reasoning.MultiTurn != nil && c.Reasoning.MultiTurn.MaxRounds < 1 {
        return fmt.Errorf("%s: reasoning.multi_turn.max_rounds must be >= 1, got %d", prefix, c.Reasoning.MultiTurn.MaxRounds)
    }
```

**File 2: `sim/workload/cohort.go`**

In the `client := ClientSpec{...}` literal (after `Streaming: cohort.Streaming`), add:

```go
                PrefixLength: cohort.PrefixLength,
                // Pointer fields shared across all expanded clients.
                // Safe: GenerateRequests reads but never mutates these fields.
                Reasoning:    cohort.Reasoning,
                ClosedLoop:   cohort.ClosedLoop,
                Timeout:      cohort.Timeout,
                Network:      cohort.Network,
                Multimodal:   cohort.Multimodal,
```

**Run tests to verify they pass:**
```bash
go test ./sim/workload/... -count=1
# Expected: PASS (all existing + new tests)
```

**Run lint:**
```bash
golangci-lint run ./sim/workload/...
# Expected: no issues
```

**Commit:** `feat(workload): add multi-turn fields to CohortSpec and propagate in ExpandCohorts (closes #847)`

---

### Task 3: Update workloads.md documentation

**File: `docs/guide/workloads.md`**

In the **Cohort Dynamics** section, after the existing traffic-pattern table and YAML example block, add a new subsection **Multi-Turn Sessions** before `## Advanced Features`:

```markdown
### Multi-Turn Sessions

Cohorts support the full set of `ClientSpec` behavioral fields, enabling multi-turn workloads without writing one client entry per session. The following fields are available on cohorts in addition to the traffic-pattern fields above:

| YAML key | Type | Purpose |
|----------|------|---------|
| `reasoning` | object | Multi-turn sessions; see [Reasoning](#reasoning-multi-turn-with-context-accumulation) for sub-fields |
| `closed_loop` | boolean | `false` = open-loop (all rounds pre-stamped at generation time); omit to use the default (closed-loop when `reasoning` is set) |
| `timeout` | integer (µs) | Per-request timeout; `0` = no timeout; omit to use the default (300 s) |
| `network` | object | Client-side network latency; see [Client-Side Network Latency](#client-side-network-latency) |
| `multimodal` | object | Multimodal token generation; see [Multimodal Requests](#multimodal-requests) |
| `prefix_length` | integer | Shared prefix token count prepended to every request |

**Example — 10 multi-turn sessions, each running 5 rounds with pre-stamped arrivals and a 30 s timeout:**

```yaml
version: "2"
seed: 42
aggregate_rate: 10
num_requests: 100

cohorts:
  - id: "chat-session"
    population: 10
    slo_class: "standard"
    rate_fraction: 1.0
    arrival:
      process: constant
    input_distribution:
      type: gaussian
      params: { mean: 256, std_dev: 64, min: 2, max: 1024 }
    output_distribution:
      type: gaussian
      params: { mean: 128, std_dev: 32, min: 2, max: 512 }
    reasoning:
      multi_turn:
        max_rounds: 5
        think_time_us: 3000000   # 3 s user think time between rounds
    closed_loop: false           # pre-stamp all rounds at generation time
    timeout: 30000000            # 30 s per-request timeout
```

> **Note:** Do not combine `spike` with `reasoning.multi_turn`. The spike lifecycle window constrains ALL round arrivals, not just session starts. With `spike.duration_us: 100000` (100 ms) and `think_time_us: 3000000` (3 s), rounds 1-4 fall outside the window and are silently suppressed — use the full simulation horizon instead.
```

**Commit:** `docs(workloads): document cohort multi-turn fields with example`

---

## Part 3 — Deviation Log

| ID | Type | Description |
|----|------|-------------|
| DEV-1 | CLARIFICATION | Issue table lists `PrefixLength` as `int` (not `*int`) consistent with `ClientSpec.PrefixLength` (line 104); treated as value type (zero = no prefix, which is unambiguous). |
| DEV-2 | CLARIFICATION | Issue says "at minimum Reasoning, ClosedLoop, and Timeout; the others are lower priority" — interpreting as all 6 should be added (the full struct snippet in the issue lists all 6). Adding all 6 is one coherent change and avoids a follow-up PR. |
| DEV-3 | CLARIFICATION | `Lifecycle *LifecycleSpec` is intentionally absent from `CohortSpec`. `Lifecycle` is a synthesized output field: `ExpandCohorts` builds it from `Diurnal`, `Spike`, and `Drain`. Exposing `Lifecycle` directly on `CohortSpec` would create two paths to the same effect and break the single-source invariant. |

---

## Part 3J — Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — 6 field copies, no new types or interfaces
- [x] No feature creep — strictly limited to fields listed in issue #847
- [x] No unexercised flags or interfaces
- [x] No partial implementations — all 6 fields added in one task
- [x] No breaking changes — additive only; existing cohorts with nil new fields behave identically
- [x] No hidden global state — `ExpandCohorts` is a pure function
- [x] All new code will pass golangci-lint — validation uses existing error patterns
- [x] CLAUDE.md: no new files, no file organization change, no new CLI flags — no update needed
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: this PR does not modify any canonical source in the source-of-truth map
- [x] Deviation log reviewed — DEV-1, DEV-2, DEV-3 all resolved
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (tests → implementation → docs)
- [x] All contracts mapped to specific tasks (BC-1..7 → Tasks 1+2)
- [x] Golden dataset regeneration: N/A — no golden datasets affected
- [x] Construction site audit completed — `cohort_test.go` and `generator_test.go` (Appendix)
- [x] Not part of a macro plan

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — `validateCohort` returns errors; `ExpandCohorts` has no error paths
- [x] R2: N/A — no float accumulation or ordered output in new code
- [x] R3: All 3 new numeric parameters validated (`PrefixLength >= 0`, `Timeout >= 0`, `MaxRounds >= 1`)
- [x] R4: All `CohortSpec` construction sites audited (cohort_test.go, generator_test.go — all named fields)
- [x] R5: N/A — no resource allocation loops
- [x] R6: No `logrus.Fatalf` / `os.Exit` in `sim/workload/` — validation returns `error`
- [x] R7: N/A — no golden datasets introduced
- [x] R8: N/A — no exported mutable maps
- [x] R9: `ClosedLoop *bool` and `Timeout *int64` correctly use pointer types for YAML zero-value ambiguity; `PrefixLength int` is unambiguous (zero = no prefix)
- [x] R10: BC-4 YAML round-trip test uses `yaml.NewDecoder` + `KnownFields(true)` — matches `LoadWorkloadSpec` production path
- [x] R11: N/A — no runtime-derived denominators
- [x] R12: N/A — no golden datasets
- [x] R13: N/A — no new interfaces
- [x] R14: N/A — no new methods spanning multiple responsibilities
- [x] R15: No stale PR references in modified files
- [x] R16: N/A — no new config structs (fields added to existing `CohortSpec`)
- [x] R17: N/A — no routing scorer signals
- [x] R18: N/A — no CLI flags
- [x] R19: N/A — no retry/requeue loops
- [x] R20: N/A — no detectors or analyzers
- [x] R21: N/A — no `range` over mutable slices
- [x] R22: N/A — no pre-check estimates
- [x] R23: N/A — no parallel code paths with transformation asymmetry

---

## Appendix — File-Level Details

### `sim/workload/spec.go` changes
- `CohortSpec` struct: add 6 fields after `Drain *DrainSpec` at line 70 (struct spans lines 56-71; insert before the closing brace at line 71)
- `validateCohort`: add 3 validation checks after the drain block, before `return nil` (~line 336)

### `sim/workload/cohort.go` changes
- `ExpandCohorts`: add 6 field copies (with pointer-sharing comment) in `client := ClientSpec{...}` literal (lines 31-42)

### `sim/workload/cohort_test.go` changes
- Add 6 new test functions for BC-1 through BC-7 (BC-1 and BC-5 combined in one test)
- Add `"gopkg.in/yaml.v3"` to imports (for `TestCohortSpec_YAMLRoundTrip_Reasoning`)

### `docs/guide/workloads.md` changes
- Add "Multi-Turn Sessions" subsection in Cohort Dynamics section with field table and example YAML

### Construction site check (R4)
`CohortSpec{...}` literals found via `grep 'CohortSpec{' sim/`:

| File | Lines | Named fields? | Safe? |
|------|-------|---------------|-------|
| `sim/workload/cohort_test.go` | all test functions | Yes | Yes — new fields default to nil/zero |
| `sim/workload/generator_test.go` | ~1538, ~1559 | Yes | Yes — new fields default to nil/zero |

No production code constructs `CohortSpec{}` — it is only created by YAML unmarshalling and test literals.
