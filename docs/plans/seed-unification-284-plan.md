# Seed Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `--seed` the master seed that controls all randomness, including workload generation from `--workload-spec` YAML files.

**The problem today:** When using `--workload-spec`, the CLI `--seed` flag does NOT control workload generation. The YAML file has its own `seed:` field that independently seeds the workload RNG. Running `--seed 42` vs `--seed 123` with the same workload-spec produces identical workloads — only simulation-level randomness varies. Users who run "3 seeds for statistical rigor" unknowingly get 3 identical runs.

**What this PR adds:**
1. CLI seed override — when `--seed` is explicitly passed, it overrides the YAML `seed:` field before workload generation, so different CLI seeds produce different workloads
2. Transparent logging — info messages report which seed controlled workload generation (CLI override vs YAML default)
3. Updated experiment standard — ED-4 updated to reflect the fix, removing the "generate per-seed YAML copies" workaround

**Why this matters:** This fixes a correctness issue (R18: CLI flag precedence) that silently undermines experiment validity. It also establishes INV-6a (seed supremacy) — the principle that `--seed` controls all randomness when explicitly provided.

**Architecture:** The fix is entirely in `cmd/root.go` at the CLI integration layer. The library (`sim/workload/`) is unchanged — `GenerateRequests` correctly uses `spec.Seed`, and the fix simply sets `spec.Seed = seed` when `cmd.Flags().Changed("seed")` is true. This follows the existing precedent used by `--horizon` and `--num-requests`.

**Source:** Design doc `docs/plans/2026-02-20-seed-unification-design.md`, GitHub issue #284

**Closes:** Fixes #284

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds 3 lines of seed-override logic to `cmd/root.go` (lines 204-236), following the exact `Changed()` pattern already used for `--horizon` and `--num-requests`. When `--seed` is explicitly passed and `--workload-spec` is in use, the CLI seed overrides the YAML seed before `GenerateRequests` is called. Two info-level log messages provide transparency. ED-4 in experiment standards is updated to reflect the fix.

Adjacent components: `sim/workload.GenerateRequests` (unchanged — already uses `spec.Seed`), `sim.PartitionedRNG` (unchanged), `cluster.DeploymentConfig.Seed` (unchanged — continues to control simulation RNG).

No deviations from the design doc.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Seed override
- GIVEN a workload-spec YAML with `seed: 42`
- WHEN the user runs with `--workload-spec w.yaml --seed 123`
- THEN the workload is generated using seed 123 (not 42)
- MECHANISM: `cmd.Flags().Changed("seed")` detects explicit CLI flag → `spec.Seed = seed`

BC-2: Different seeds produce different workloads
- GIVEN a workload-spec YAML with `seed: 42`
- WHEN the user runs twice with `--seed 100` and `--seed 200`
- THEN the two runs produce different request arrival times and token counts
- MECHANISM: Different seeds create different `PartitionedRNG` instances in `GenerateRequests`

BC-3: YAML seed preserved when --seed not specified
- GIVEN a workload-spec YAML with `seed: 42`
- WHEN the user runs with `--workload-spec w.yaml` (no explicit `--seed`)
- THEN the workload is generated using YAML seed 42 (backward compatible)
- MECHANISM: `cmd.Flags().Changed("seed")` returns false → spec.Seed untouched

BC-4: Determinism preserved
- GIVEN any combination of `--seed` and `--workload-spec`
- WHEN the same command is run twice with identical flags
- THEN the output is byte-identical (INV-6)
- MECHANISM: Same seed → same `PartitionedRNG` → same workload → same simulation

**Negative Contracts:**

BC-5: No backward compatibility break
- GIVEN existing scripts that run `--workload-spec w.yaml` without `--seed`
- WHEN this PR is deployed
- THEN their output MUST NOT change
- MECHANISM: `Changed("seed")` is false when `--seed` is not passed → no override

**Error Handling Contracts:**

BC-6: No new error paths
- GIVEN any valid or invalid combination of `--seed` and `--workload-spec`
- WHEN the command is executed
- THEN no new panic, fatal, or error paths are introduced
- MECHANISM: The override is a simple assignment (`spec.Seed = seed`) with no failure mode

### C) Component Interaction

```
CLI (cmd/root.go)
  │
  ├── --seed flag (int64, default 42)
  │     │
  │     ├── Changed("seed")? ──yes──▶ spec.Seed = seed  ──▶ GenerateRequests(spec, ...)
  │     │                                                        │
  │     └── no ──▶ spec.Seed unchanged (YAML value) ──▶ GenerateRequests(spec, ...)
  │                                                              │
  │                                                              ▼
  │                                                    sim.NewPartitionedRNG(spec.Seed)
  │                                                              │
  ├── DeploymentConfig.Seed = seed ──▶ ClusterSimulator RNG (routing, tie-breaking)
  │
  └── (These two RNG paths are now both controllable via --seed)
```

No new types, interfaces, or state. No API changes. The override is a 3-line in-place mutation in `cmd/root.go`.

Extension friction: 0 files to add a new CLI-overridable YAML field — the `Changed()` pattern is self-contained.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| (not mentioned) | Task 4 updates design doc status to "Implemented" | ADDITION: micro plan housekeeping convention — does not contradict any design decision |

### E) Review Guide

**The tricky part:** The `Changed("seed")` check. Since `--seed` has a default value of 42, and the YAML might also have `seed: 42`, the user who doesn't pass `--seed` gets the YAML value (correct). The user who passes `--seed 42` explicitly gets the CLI override (also correct, because `Changed()` detects explicit flags regardless of whether the value matches the default).

**What to scrutinize:** BC-2 test — verify it actually tests that different seeds produce different workloads, not just that the code runs without error.

**What's safe to skim:** BC-4 (determinism) — this invariant is already tested extensively in the workload package. BC-6 (no new error paths) — trivially true.

**Known debt:** The tests validate that different seeds produce different workloads (a `sim/workload` library property) but do not exercise the actual `cmd.Flags().Changed("seed")` branch in `cmd/root.go`. Testing Cobra flag parsing requires integration-level CLI invocation, which is disproportionate for a 3-line change. The CLI wiring is visually verifiable and follows the exact `Changed()` pattern used by `--horizon` and `--num-requests`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `cmd/root.go` (~lines 204-236) — add seed override logic + logging
- Create: `cmd/root_seed_test.go` — behavioral tests for seed override (new test file for CLI-level tests)
- Modify: `docs/standards/experiments.md` (~lines 95-103) — update ED-4
- Modify: `docs/plans/2026-02-20-seed-unification-design.md` — mark status as Implemented

**Key decisions:**
- Tests are at the `cmd/` level (not `sim/workload/`) because the bug is at the CLI integration layer
- Tests call `GenerateRequests` directly (not the full CLI) to keep them fast and focused
- No changes to `sim/` packages — the library is correct; only the CLI integration is wrong

### G) Task Breakdown

---

### Task 1: Seed Override Logic + Logging

**Contracts Implemented:** BC-1, BC-3, BC-5

**Files:**
- Modify: `cmd/root.go:204-236`

**Step 1: Implement the seed override and logging**

Context: We add 3 lines of seed override logic after `spec.Validate()` and before `GenerateRequests`, plus info logging. This follows the exact pattern of `--horizon` and `--num-requests` overrides at lines 213-221.

In `cmd/root.go`, after line 211 (`logrus.Fatalf("Invalid workload spec: %v", err)`), add the seed override block. The final workload-spec section should read:

```go
		// --workload-spec takes precedence over --workload if set
		if workloadSpecPath != "" {
			spec, err := workload.LoadWorkloadSpec(workloadSpecPath)
			if err != nil {
				logrus.Fatalf("Failed to load workload spec: %v", err)
			}
			if err := spec.Validate(); err != nil {
				logrus.Fatalf("Invalid workload spec: %v", err)
			}
			// Apply CLI --seed override: when explicitly passed, CLI seed controls
			// workload generation (R18: CLI flag precedence, INV-6a: seed supremacy).
			// When --seed is not passed, the YAML seed is used (backward compatible).
			if cmd.Flags().Changed("seed") {
				logrus.Infof("CLI --seed %d overrides workload-spec seed %d", seed, spec.Seed)
				spec.Seed = seed
			} else {
				logrus.Infof("Using workload-spec seed %d (CLI --seed not specified)", spec.Seed)
			}
			// Apply spec horizon as default; CLI --horizon flag overrides via Changed().
			if spec.Horizon > 0 && !cmd.Flags().Changed("horizon") {
				simulationHorizon = spec.Horizon
			}
```

**Step 2: Build to verify compilation**

Run: `go build ./...`
Expected: SUCCESS (no compilation errors)

**Step 3: Commit**

```bash
git add cmd/root.go
git commit -m "fix(cmd): CLI --seed overrides workload-spec YAML seed (#284, BC-1, BC-3, BC-5)

When --seed is explicitly passed and --workload-spec is in use, the CLI
seed now overrides the YAML seed field before workload generation. This
follows the existing Changed() pattern used by --horizon and --num-requests.

When --seed is not passed, the YAML seed is used as-is (backward compatible).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Behavioral Tests for Seed Override

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4

**Files:**
- Create: `cmd/root_seed_test.go`

**Step 1: Write tests for all seed override contracts**

Context: We test at the integration boundary — the spec seed mutation that happens in `cmd/root.go`. Since we can't easily invoke the full Cobra command in a unit test, we test the behavioral outcome: that `GenerateRequests` produces different results when given different seeds, and identical results when given the same seed. This validates the contracts without coupling to CLI internals.

```go
package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// makeTestSpec returns a minimal WorkloadSpec for seed tests.
func makeTestSpec(seed int64) *workload.WorkloadSpec {
	return &workload.WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "c1", TenantID: "t1", RateFraction: 1.0, SLOClass: "interactive",
			Arrival:    workload.ArrivalSpec{Process: "poisson"},
			InputDist:  workload.DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: workload.DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
}

// TestSeedOverride_DifferentSeeds_DifferentWorkloads verifies BC-1/BC-2:
// when the CLI seed overrides the YAML seed, different seeds produce
// different workloads (arrival times and token counts differ).
func TestSeedOverride_DifferentSeeds_DifferentWorkloads(t *testing.T) {
	// GIVEN a workload spec with YAML seed 42
	spec1 := makeTestSpec(42)
	spec2 := makeTestSpec(42)

	// WHEN CLI --seed overrides to different values
	spec1.Seed = 100 // simulates Changed("seed") → spec.Seed = 100
	spec2.Seed = 200 // simulates Changed("seed") → spec.Seed = 200

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(spec1, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(spec2, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN the workloads differ (at least one request has different arrival time)
	if len(r1) == 0 || len(r2) == 0 {
		t.Fatal("expected non-empty request sets")
	}
	anyDifferent := false
	minLen := len(r1)
	if len(r2) < minLen {
		minLen = len(r2)
	}
	for i := 0; i < minLen; i++ {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			anyDifferent = true
			break
		}
	}
	if len(r1) != len(r2) {
		anyDifferent = true
	}
	if !anyDifferent {
		t.Error("different seeds produced identical workloads — seed override is not working")
	}
}

// TestSeedOverride_SameSeed_IdenticalWorkload verifies BC-4:
// same seed produces byte-identical workload (determinism preserved).
func TestSeedOverride_SameSeed_IdenticalWorkload(t *testing.T) {
	// GIVEN two specs with the same seed (simulating CLI override to same value)
	spec1 := makeTestSpec(42)
	spec2 := makeTestSpec(42)
	spec1.Seed = 123
	spec2.Seed = 123

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(spec1, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(spec2, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN output is identical
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}

// TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified verifies BC-3/BC-5:
// when --seed is not explicitly passed, the YAML seed governs workload
// generation (backward compatibility).
func TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified(t *testing.T) {
	// GIVEN a spec with YAML seed 42 (no CLI override)
	specA := makeTestSpec(42)
	specB := makeTestSpec(42)

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(specA, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(specB, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN same YAML seed produces identical workload (YAML seed is the default)
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d — YAML seed not preserved", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./cmd/... -run TestSeedOverride -v`
Expected: All 3 tests PASS

**Step 3: Run lint**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add cmd/root_seed_test.go
git commit -m "test(cmd): add behavioral tests for seed override (BC-1 through BC-5)

- TestSeedOverride_DifferentSeeds_DifferentWorkloads (BC-1, BC-2)
- TestSeedOverride_SameSeed_IdenticalWorkload (BC-4)
- TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified (BC-3, BC-5)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update ED-4 Experiment Standard

**Contracts Implemented:** (documentation — supports BC-1 through BC-5)

**Files:**
- Modify: `docs/standards/experiments.md:95-103`

**Step 1: Update ED-4 to reflect the fix**

Context: Replace the "known issue" workaround with the resolved behavior.

Replace lines 95-103 of `docs/standards/experiments.md` with:

```markdown
### ED-4: Workload seed independence
**Resolved (#284):** CLI `--seed` now overrides the workload-spec YAML `seed:` field when explicitly passed. Behavior:
- `--seed N --workload-spec w.yaml` → workload uses seed N (CLI override)
- `--workload-spec w.yaml` (no `--seed`) → workload uses YAML `seed:` value (backward compatible)
- CLI-generated workloads (`--rate`, `--num-requests`) → `--seed` controls everything (unchanged)

For multi-seed experiments: simply vary `--seed` on the command line. No need to generate per-seed YAML copies.

**Note:** The YAML `seed:` field still serves as the default seed for the workload when `--seed` is not explicitly specified. This enables the "shareable workload" pattern — distributing a YAML file that always produces the same workload by default.
```

**Step 2: Commit**

```bash
git add docs/standards/experiments.md
git commit -m "docs(standards): update ED-4 — seed override resolves #284

Replace 'known issue' workaround with resolved behavior documentation.
CLI --seed now overrides YAML seed when explicitly passed.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Final Verification + Design Doc Status Update

**Contracts Implemented:** All (verification)

**Files:**
- Modify: `docs/plans/2026-02-20-seed-unification-design.md:3`

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 2: Run full lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Update design doc status**

Change line 3 of `docs/plans/2026-02-20-seed-unification-design.md` from:
```
**Status:** Draft (pending human review)
```
to:
```
**Status:** Implemented
```

**Step 4: Commit**

```bash
git add docs/plans/2026-02-20-seed-unification-design.md
git commit -m "docs(plans): mark seed unification design as implemented

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestSeedOverride_DifferentSeeds_DifferentWorkloads |
| BC-2 | Task 2 | Unit | TestSeedOverride_DifferentSeeds_DifferentWorkloads |
| BC-3 | Task 2 | Unit | TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified |
| BC-4 | Task 2 | Unit | TestSeedOverride_SameSeed_IdenticalWorkload |
| BC-5 | Task 2 | Unit | TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified |
| BC-6 | — | N/A | No new error paths — trivially satisfied |

No golden dataset changes (output format unchanged). No invariant test needed beyond BC-4 (determinism is the invariant). Existing workload tests in `sim/workload/generator_test.go` continue to pass unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| `Changed("seed")` returns true when user didn't intend override | Low | Medium | Cobra's `Changed()` only returns true for explicitly passed flags, not defaults. Well-tested in existing codebase. | Task 1 |
| Backward compatibility break for scripts without `--seed` | Low | High | BC-3/BC-5 test explicitly verifies YAML seed preserved when `--seed` not specified. | Task 2 |
| Spec mutation before `GenerateRequests` affects other callers | None | — | `spec` is a local variable in the Cobra RunE closure; no other code references it after mutation. | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — simple 3-line fix
- [x] No feature creep — strictly scoped to #284
- [x] No unexercised flags or interfaces — no new flags
- [x] No partial implementations — complete in Task 1
- [x] No breaking changes — BC-5 ensures backward compat
- [x] No hidden global state impact — `spec` is local
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: N/A (no shared helpers needed)
- [x] CLAUDE.md: No update needed (no new files, packages, or flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log: no deviations
- [x] Each task produces testable code
- [x] Task dependencies correctly ordered (1→2→3→4)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no changes needed
- [x] Construction site audit: no struct fields added
- [x] Not part of macro plan — standalone issue

**Antipattern rules:**
- [x] R1: No silent data loss — no new error paths
- [x] R2: No map iteration — N/A
- [x] R3: No new CLI flags — N/A
- [x] R4: No struct fields added — N/A
- [x] R5: No resource allocation — N/A
- [x] R6: No logrus.Fatalf in sim/ — change is in cmd/
- [x] R7: No golden tests — N/A
- [x] R8: No exported maps — N/A
- [x] R9: No YAML fields — N/A (reading existing field, not adding)
- [x] R10: Strict parsing — N/A (not changing YAML parsing)
- [x] R11: No division — N/A
- [x] R12: No output changes — N/A
- [x] R13: No new interfaces — N/A
- [x] R14: No multi-concern methods — N/A
- [x] R15: No stale PR references — N/A
- [x] R16: Config grouping — N/A
- [x] R17: Signal freshness — N/A
- [x] R18: **This PR fixes R18 for seed** — CLI flag precedence restored

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go`

**Purpose:** Add seed override logic in the `--workload-spec` section.

**Exact change location:** After `spec.Validate()` error check (line 211), before the `--horizon` override block (line 213).

**Code to insert (between lines 211 and 212):**

```go
			// Apply CLI --seed override: when explicitly passed, CLI seed controls
			// workload generation (R18: CLI flag precedence, INV-6a: seed supremacy).
			// When --seed is not passed, the YAML seed is used (backward compatible).
			if cmd.Flags().Changed("seed") {
				logrus.Infof("CLI --seed %d overrides workload-spec seed %d", seed, spec.Seed)
				spec.Seed = seed
			} else {
				logrus.Infof("Using workload-spec seed %d (CLI --seed not specified)", spec.Seed)
			}
```

**Key Implementation Notes:**
- The `seed` variable is the package-level `var seed int64` populated by Cobra from `--seed` flag
- `spec` is a local `*workload.WorkloadSpec` — mutating `spec.Seed` only affects this invocation
- `cmd` is the Cobra command parameter from `Run: func(cmd *cobra.Command, args []string)`
- Placement before the horizon override maintains the flag override ordering: seed → horizon → num-requests

### File: `cmd/root_seed_test.go`

**Purpose:** Behavioral tests for the seed override at the CLI integration boundary.

**Complete implementation:** See Task 2, Step 1 above (3 test functions + 1 helper).

### File: `docs/standards/experiments.md`

**Purpose:** Update ED-4 from "known issue with workaround" to "resolved with new behavior".

**Complete implementation:** See Task 3, Step 1 above.

### File: `docs/plans/2026-02-20-seed-unification-design.md`

**Purpose:** Mark design doc as implemented.

**Change:** Line 3: `Draft (pending human review)` → `Implemented`
