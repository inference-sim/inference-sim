# PR5 Micro-Design Plan: Architectural Simplification

**Date:** 2026-02-13
**PR:** 5 of 16 (v3.0)
**Branch:** `pr5`
**Based on:** Post-PR4 codebase, macro plan v3.0

---

## PART 1: Human Review

### A) Executive Summary

PR5 simplifies BLIS internals before adding policy features (PR6+). Four changes, zero behavioral impact:

1. **SimConfig struct** — Replaces 5 constructors (15-18 params each) with a single options struct
2. **Unified CLI path** — CLI always uses `ClusterSimulator` (even N=1), eliminating duplicate code paths
3. **Field privatization** — ~19 internal `Simulator` fields become unexported
4. **Interface dedup** — `AdmissionPolicy` interface + implementations consolidated (eliminate duplication between `sim/cluster/` and `sim/policy/`)

**SCOPE-WARNING:** Touches ~10 production files across 4 packages (`sim/`, `sim/cluster/`, `sim/policy/`, `cmd/`). Acceptable: all changes are mechanical refactoring with golden test verification.

**Golden test preservation:** Bit-exact output MUST be preserved. Verified via existing `TestSimulator_GoldenDataset`, `TestInstanceSimulator_GoldenDataset_Equivalence`, and `TestClusterSimulator_SingleInstance_GoldenEquivalence`.

---

### B) Behavioral Contracts

#### Positive Contracts (what MUST happen)

**BC-1: SimConfig Constructor Equivalence**
- GIVEN any valid parameter combination previously passed to `NewSimulator(17 params)`
- WHEN the same values are placed in a `SimConfig` struct and passed to the new constructor
- THEN the resulting `*Simulator` produces bit-identical simulation output
- MECHANISM: `SimConfig` is a value struct; constructor maps fields 1:1 to the same initialization in `newSimulatorBase`

**BC-2: SimConfig Without-Workload Equivalence**
- GIVEN any valid parameter combination previously passed to `NewSimulatorWithoutWorkload(15 params)`
- WHEN the same values are placed in a `SimConfig` and passed to the new constructor
- THEN the resulting `*Simulator` has empty EventQueue and identical field state
- MECHANISM: The workload/no-workload distinction becomes: if both `GuideLLMConfig` and `TracesWorkloadFilePath` are zero-valued, no workload is generated

**BC-3: Unified CLI Golden Equivalence**
- GIVEN a single-instance CLI invocation (`--num-instances 1` or default)
- WHEN the unified code path routes through `ClusterSimulator` with N=1
- THEN `SaveResults` output is bit-identical to the pre-PR5 single-instance path
- MECHANISM: `ClusterSimulator` with N=1, zero admission/routing latency, `AlwaysAdmit` policy produces identical event sequence as direct `InstanceSimulator.Run()`

**BC-4: SaveResults Output Format Change**
- GIVEN `--num-instances 1` (unified path) with `--results-path` specified
- WHEN `SaveResults` is called
- THEN: (a) Per-instance metrics print to stdout with ID `"instance_0"` (was `"default"`); (b) Aggregated metrics save to file with ID `"cluster"` (was `"default"`); (c) Metric **values** are bit-identical (aggregated N=1 = single instance)
- MECHANISM: The cluster path always uses `InstanceID(fmt.Sprintf("instance_%d", idx))`. File output uses "cluster" for aggregated results. Golden tests verify metric values, NOT instance ID strings. The ID string change is cosmetic.

**BC-5: Multi-Instance Path Unchanged**
- GIVEN `--num-instances N` where N > 1
- WHEN the cluster path executes
- THEN behavior is identical to pre-PR5
- MECHANISM: The cluster path code is not modified beyond constructor signature changes

**BC-6: AdmissionPolicy Interface Consolidation**
- GIVEN code importing `AdmissionPolicy` from either `sim/cluster/` or `sim/policy/`
- WHEN the interface is moved to `sim/` base package
- THEN both `cluster` and `policy` packages use the same interface type
- MECHANISM: `AdmissionPolicy` interface depends only on `*sim.Request` and `int64`, which are already in `sim/`

**BC-7: Field Privatization Non-Breakage**
- GIVEN all internal `Simulator` methods that access privatized fields
- WHEN fields are renamed from exported to unexported
- THEN all internal logic remains identical (same field, lowercase name)
- MECHANISM: Only fields NOT accessed outside `sim/` package are privatized. `InstanceSimulator` accesses `Clock`, `Horizon`, `Metrics`, `WaitQ`, `KVCache`, `RunningBatch` — these remain public.

#### Negative Contracts (what MUST NOT happen)

**NC-1: No RNG Stream Divergence**
- GIVEN the same seed value
- WHEN workload is generated through the cluster path (N=1)
- THEN the request sequence (IDs, arrival times, token counts) MUST be identical to the old single-instance path
- MECHANISM: Both paths use `PartitionedRNG.ForSubsystem(SubsystemWorkload)` which derives RNG from the master seed identically (`sim/rng.go:73-76`)

**NC-2: No Golden Test Breakage**
- All 3 golden test suites MUST pass: `TestSimulator_GoldenDataset` (`sim/simulator_test.go:16`), `TestInstanceSimulator_GoldenDataset_Equivalence` (`sim/cluster/instance_test.go:17`), `TestClusterSimulator_SingleInstance_GoldenEquivalence` (`sim/cluster/cluster_test.go:54`)

**NC-3: No Dead Code Introduced**
- MUST NOT leave unused constructors, unused type aliases, or backward-compat shims
- Old constructors MUST be deleted, not deprecated

#### Error Handling Contracts

**EC-1: No-Workload SimConfig is Valid**
- GIVEN a `SimConfig` with `GuideLLMConfig == nil` AND `TracesWorkloadFilePath == ""`
- WHEN `NewSimulator(cfg)` is called
- THEN the constructor succeeds and returns a Simulator with an empty EventQueue (no workload generated)
- MECHANISM: The unified constructor treats both-nil as "no workload" for the cluster injection path. The old `logrus.Fatalf` at `sim/simulator.go:164` is removed. The existing guard at `ClusterSimulator` level (`cluster.go:105-107`) already panics if both are nil — no new validation needed.

**EC-1b: ClusterSimulator Workload Guard Preserved**
- GIVEN `NewClusterSimulator(config, nil, "")` with nil workload and empty traces path
- WHEN the constructor is called
- THEN panic with `"ClusterSimulator: workload config is nil and no traces path provided"` (existing behavior at `cluster.go:106`)

**EC-2: Panic Preservation**
- All existing panics (double-run, zero instances, inject-after-run) MUST be preserved with identical messages

---

### C) Component Interaction

```
┌────────────────────────────────────────────────────────────────┐
│                        CLI (cmd/root.go)                       │
│  ALWAYS builds DeploymentConfig → NewClusterSimulator          │
│  (removes single-instance special case)                        │
└──────────────┬─────────────────────────────────────────────────┘
               │ DeploymentConfig
               ▼
┌────────────────────────────────────────────────────────────────┐
│              ClusterSimulator (sim/cluster/)                    │
│  Creates N InstanceSimulator via SimConfig                     │
│  SimConfig replaces 15-18 param constructors                   │
└──────────────┬─────────────────────────────────────────────────┘
               │ SimConfig
               ▼
┌────────────────────────────────────────────────────────────────┐
│              sim.Simulator                                      │
│  NewSimulator(cfg SimConfig) *Simulator                        │
│  ~19 fields now unexported (internal-only access)              │
│  sim.AdmissionPolicy interface (moved from cluster + policy)   │
└────────────────────────────────────────────────────────────────┘
```

**API Contracts:**
- `SimConfig` — New value struct in `sim/` package with all constructor params as named fields
- `NewSimulator(cfg SimConfig) *Simulator` — Single constructor, workload mode determined by `GuideLLMConfig`/`TracesWorkloadFilePath` presence
- `sim.AdmissionPolicy` — Interface moved to `sim/` (signature: `Admit(*Request, int64) (bool, string)`)
- `DeploymentConfig` gains a `ToSimConfig() SimConfig` method (or constructor uses it directly)

**State Changes:** None. No new mutable state. Field renaming only.

---

### D) Deviation Log

| Macro Plan Says | Code Actually Does | Category | Reason |
|---|---|---|---|
| "~80 LOC" changes to simulator.go | ~120 LOC needed (3 constructors + SimConfig + field privatization) | CORRECTION | Macro plan underestimated; 3 constructors + ~19 field renames |
| "Move AdmissionPolicy interface to sim/" | `AlwaysAdmit`, `TokenBucket`, `NewTokenBucket` also duplicated (not just interface) | ADDITION | Both packages duplicate full implementations at `cluster.go:18-72` and `policy/admission.go:11-67` |
| Single instance uses ID "default" | `cmd/root.go:204` creates `InstanceID("default")`; saves to file with ID "default" | ADDITION | Unified path: stdout prints ID "instance_0", file saves ID "cluster". Old path: single call with ID "default". Metric values identical; only ID strings change. Golden tests don't check ID strings. |
| Single-instance SaveResults: one call | `cmd/root.go:224` single SaveResults call for both stdout + file | ADDITION | Unified path makes two calls: per-instance stdout + aggregated file. Output values identical for N=1 but format differs (two JSON prints to stdout vs one). **Mitigation:** For N=1, suppress per-instance stdout print (it's identical to aggregated). Only print aggregated. See I-5 resolution below. |
| `policy.NewAdmissionPolicy("")` panics | `policy/admission.go:59` only handles `"always-admit"`, panics on `""` | ADDITION | New `sim.NewAdmissionPolicy("", ...)` returns `AlwaysAdmit` (matching `cluster.go:65` behavior). This is the intended behavior — callers pass config values that may be empty string for defaults. |
| "test files (~100 LOC)" | Test files have 12+ constructor call sites in instance_test.go alone | CORRECTION | Test file changes are ~150-200 LOC. Total PR LOC ~350-400 changed, ~200 net reduction. |
| "~10 LOC" changes to deployment.go | DeploymentConfig needs method to convert to SimConfig (~20 LOC) | CORRECTION | Slightly more than estimated |
| Macro plan "Files Changed" says "New: none" | Micro plan creates `sim/admission.go` (new file) | ADDITION | Macro plan's "no new files" interpreted as "no new packages." Adding a single file in the existing `sim/` package is cleaner than stuffing interface+implementations into `simulator.go` (already 600 lines). See Challenge 3. |
| Macro plan test: `TestUnifiedCLIPath_MatchesGoldenDataset` | Existing `TestClusterSimulator_SingleInstance_GoldenEquivalence` already covers this | SIMPLIFICATION | The existing test validates the same contract (N=1 cluster = golden). No need for a separate test since the CLI calls `ClusterSimulator` directly. |

---

### E) Review Guide

1. **THE TRICKY PART:** RNG parity between old single-instance path and new unified cluster path (NC-1). The cluster path creates a *separate* `PartitionedRNG` in `NewClusterSimulator` (`cluster.go:138`) that generates workload, then injects requests into instances that have *their own* RNG. The old single-instance path used a single RNG. Both produce identical streams because `SubsystemWorkload` uses master seed directly (`rng.go:73-76`).

2. **WHAT TO SCRUTINIZE:** BC-3 (unified CLI golden equivalence) and NC-1 (no RNG divergence). The `SaveResults` instance ID change from "default" to "instance_0" (BC-4) — verify golden tests don't check this.

3. **WHAT'S SAFE TO SKIM:** Field privatization (BC-7) — purely mechanical lowercase rename. AdmissionPolicy dedup (BC-6) — move + delete, no logic change.

4. **KNOWN DEBT:** `SaveResults` discards `json.MarshalIndent` error (`metrics.go:115-117`). Pre-existing, not in scope. Test helper duplication was resolved in PR1 with `sim/internal/testutil/` — no action needed.

---

## PART 2: Implementation Reference

### F) Implementation Plan

**Key Decisions:**
1. `SimConfig` uses flat struct (not nested) — matches existing constructor param style
2. Workload mode is implicit: `GuideLLMConfig != nil` → distribution; `TracesWorkloadFilePath != ""` → traces; both nil/empty → no workload
3. `AdmissionPolicy` moves to `sim/` (not a new package) — simplest, no new packages
4. Field privatization scope: only fields with zero external access (per assessment at `simplification-assessment.md:127-154`)

**Design-time assertions:**
- No dead code: old constructors deleted entirely after migration
- All new codepaths exercisable: `SimConfig` constructor tested via existing golden tests
- No unused abstractions: `SimConfig` is used by every constructor call site

#### Task Graph

**Batch 1: Types and Interfaces (foundation)**

IT-1: Define `SimConfig` struct and `AdmissionPolicy` interface in `sim/`
- Files: `sim/simulator.go` (add SimConfig struct), `sim/admission.go` (new — interface + AlwaysAdmit + TokenBucket)
- Depends on: nothing
- Contracts: BC-1, BC-2, BC-6
- Verification: `go build ./sim/...`
- Parallel: yes

IT-2: Add `NewSimulatorFromConfig(cfg SimConfig)` constructor alongside existing constructors
- Files: `sim/simulator.go`
- Depends on: IT-1
- Contracts: BC-1, BC-2, EC-1
- Verification: `go build ./sim/...`
- Parallel: no (depends on IT-1)
- Note: Temporary name `NewSimulatorFromConfig` avoids conflict with existing `NewSimulator(17 params)`. IT-7 renames to `NewSimulator` after old constructors are removed.

**Batch 1 checkpoint:**
- Build: `go build ./...`
- Tests: `go test ./sim/... -run TestSimulator_GoldenDataset` (must pass — old constructors still exist)
- Contracts verified: BC-1 (partially — new constructor exists alongside old)

**Batch 2: Call-site migration**

IT-3: Update `InstanceSimulator` constructors to accept `SimConfig`
- Files: `sim/cluster/instance.go`
- Depends on: IT-2
- Contracts: BC-1, BC-2, BC-7
- Verification: `go build ./sim/cluster/...`
- Parallel: yes (with IT-4)

IT-4: Update `ClusterSimulator` to use `SimConfig` and `sim.AdmissionPolicy`
- Files: `sim/cluster/cluster.go`, `sim/cluster/deployment.go`
- Depends on: IT-1, IT-2
- Contracts: BC-5, BC-6
- Verification: `go build ./sim/cluster/...`
- Parallel: yes (with IT-3)

IT-5: Remove duplicated `AdmissionPolicy` from `sim/cluster/cluster.go` and delete `sim/policy/` package
- Files: `sim/cluster/cluster.go` (remove interface + implementations + factory), delete `sim/policy/admission.go` and `sim/policy/` directory
- Depends on: IT-4
- Contracts: BC-6, NC-3
- Verification: `go build ./...`
- Parallel: no (depends on IT-4)

**Batch 2 checkpoint:**
- Build: `go build ./...`
- Tests: `go test ./... -run "Golden|Equivalence"` (all golden tests must pass)
- Lint: `golangci-lint run ./...`
- Contracts verified: BC-1, BC-2, BC-5, BC-6

**Batch 3: Unified CLI + cleanup**

IT-6: Unify CLI path — always use `ClusterSimulator` even for N=1
- Files: `cmd/root.go`
- Depends on: IT-4
- Contracts: BC-3, BC-4, BC-5, NC-1
- Verification: `go test ./... -run "Golden|Equivalence"`
- Parallel: no

IT-7: Remove old constructors and rename `NewSimulatorFromConfig` → `NewSimulator`
- Removes: `newSimulatorBase`, old `NewSimulator(17 params)`, `NewSimulatorWithoutWorkload`, old `NewInstanceSimulator(18 params)`, `NewInstanceSimulatorWithoutWorkload`
- Renames: `NewSimulatorFromConfig(cfg SimConfig)` → `NewSimulator(cfg SimConfig)`
- Also renames: old `NewInstanceSimulator(id, 17 params)` → `NewInstanceSimulator(id, cfg)` (if using temp name)
- Files: `sim/simulator.go`, `sim/cluster/instance.go`
- Depends on: IT-3, IT-6
- Contracts: NC-3
- Verification: `go build ./...`
- Parallel: no (depends on IT-3, IT-6)

**Batch 3 checkpoint:**
- Build: `go build ./...`
- Tests: `go test ./...` (full suite)
- Lint: `golangci-lint run ./...`
- Contracts verified: BC-3, BC-4, NC-1, NC-3

**Batch 4: Field privatization + tests + CLAUDE.md**

IT-8: Privatize ~19 internal Simulator fields
- Files: `sim/simulator.go`, `sim/event.go` (if StepEvent field referenced)
- Depends on: IT-7
- Contracts: BC-7
- Verification: `go build ./sim/...` (cluster/ and test files will temporarily fail — expected)
- Parallel: no (must precede IT-9; `sim/simulator_test.go` is same-package and references renamed fields)

IT-9: Update all test files to use `SimConfig`-based constructors AND fix privatized field references
- Files: `sim/simulator_test.go`, `sim/cluster/instance_test.go`, `sim/cluster/cluster_test.go`
- Depends on: IT-8
- Contracts: BC-1, BC-2, NC-2
- Verification: `go test ./...`
- Parallel: no (depends on IT-8 for field renames; `sim/simulator_test.go` references privatized fields)

IT-10: Update CLAUDE.md for post-PR5 state
- Files: `CLAUDE.md`
- Depends on: IT-8, IT-9
- Contracts: —
- Verification: visual review
- Parallel: no
- MUST update: (1) Remove `sim/policy/` from file organization (deleted in IT-5; PR6 will re-create it), (2) Add `sim/admission.go` to file organization, (3) Update constructor descriptions to reference `SimConfig`, (4) Update "Current Implementation Focus" to mark PR5 as completed

**Batch 4 checkpoint (FINAL):**
- Build: `go build ./...`
- Tests: `go test ./...`
- Lint: `golangci-lint run ./...`
- Golden: all 3 golden test suites pass
- Contracts verified: ALL (BC-1 through BC-7, NC-1 through NC-3, EC-1, EC-1b, EC-2)

---

### G) Exercisability Proof

| New Codepath | Exercise Method | Details |
|---|---|---|
| `SimConfig` struct | Existing golden tests | `TestSimulator_GoldenDataset` calls `NewSimulator(cfg)` after migration |
| `NewSimulatorFromConfig` | Existing golden tests | All 3 golden test suites call the new constructor |
| Unified CLI path (N=1 through cluster) | `TestClusterSimulator_SingleInstance_GoldenEquivalence` | Already validates N=1 cluster path matches golden output |
| `sim.AdmissionPolicy` in base package | Existing cluster tests | `TestClusterSimulator_*` tests use `AlwaysAdmit` via `newAdmissionPolicy` |
| Privatized fields | Existing tests | All internal field access unchanged; compile verification sufficient |

No new codepaths are introduced that cannot be exercised by existing tests. No CLI changes needed — the interface is identical.

---

### H) Test Strategy

| Contract | Test Type | Test Name / Description |
|---|---|---|
| BC-1 | Golden | `TestSimulator_GoldenDataset` — verifies SimConfig constructor equivalence |
| BC-2 | Unit | `TestNewSimulatorWithoutWorkload_RunsEmpty` — verifies no-workload SimConfig path |
| BC-3 | Golden | `TestClusterSimulator_SingleInstance_GoldenEquivalence` — N=1 cluster = golden. Note: macro plan names this `TestUnifiedCLIPath_MatchesGoldenDataset`; the existing test already validates this contract since the CLI path calls `ClusterSimulator` directly. No separate CLI-level test needed. |
| BC-4 | Unit | Verify instance ID is "instance_0" and stdout output format (new test: `TestUnifiedCLI_InstanceID_And_Output`) |
| BC-5 | Golden | `TestClusterSimulator_MultiInstance_Determinism` — multi-instance unchanged |
| BC-6 | Unit | Existing admission tests + compile-time verification of single interface |
| BC-7 | Build | `go build ./...` — compile verifies no external access to privatized fields |
| NC-1 | Golden | `TestClusterWorkloadGen_MatchesSimulator` — RNG parity verified |
| NC-2 | Golden | All 3 golden suites |
| NC-3 | Build | `go build ./...` — no unused exports |
| EC-1 | Unit | `TestNewSimulator_NoWorkload_EmptyQueue` (new) — SimConfig with both nil produces empty EventQueue |
| EC-1b | Failure | Existing `TestClusterSimulator_NilWorkload_Panics` — ClusterSimulator guard preserved |
| EC-2 | Failure | Existing `TestClusterSimulator_RunOnce_Panics`, `TestInstanceSimulator_RunOnce_PanicsOnSecondCall` |

**New tests:**
- `TestUnifiedCLI_InstanceID_And_Output` — verify N=1 cluster path creates instance with ID "instance_0" and stdout format
- `TestNewSimulator_NoWorkload_EmptyQueue` — verify SimConfig with both nil succeeds with empty EventQueue

**Shared test infrastructure:** Uses `sim/internal/testutil/golden.go` (existing). No new helpers needed.

**Golden dataset:** No changes to `testdata/goldendataset.json`. Format unchanged.

**Lint:** `golangci-lint run ./...` MUST pass with zero new issues.

---

### I) Verification Protocol

**Per-task verification:**
- After IT-1: `go build ./sim/...` — if fails, SimConfig or AdmissionPolicy type definition is wrong
- After IT-2: `go build ./sim/...` + `go test ./sim/... -run TestSimulator_GoldenDataset` — if fails, BC-1 violated
- After IT-3: `go build ./sim/cluster/...` — if fails, InstanceSimulator constructor signature mismatch
- After IT-4: `go build ./sim/cluster/...` — if fails, ClusterSimulator / DeploymentConfig conversion wrong
- After IT-5: `go build ./...` — if fails, dangling references to removed duplicates
- After IT-6: `go test ./... -run "Golden|Equivalence"` — if fails, BC-3 or NC-1 violated (most critical)
- After IT-7: `go build ./...` — if fails, call sites still reference old constructors
- After IT-8: `go build ./...` — if fails, external code accesses privatized field
- After IT-9: `go test ./...` — if fails, test call sites incorrect

**Per-batch verification:**
- Batch 1: `go build ./...` + `go test ./sim/... -run TestSimulator_GoldenDataset`
- Batch 2: `go build ./...` + `go test ./... -run "Golden|Equivalence"` + `golangci-lint run ./...`
- Batch 3: `go build ./...` + `go test ./...` + `golangci-lint run ./...`
- Batch 4 (FINAL): `go build ./...` + `go test ./...` + `golangci-lint run ./...`

**Failure rule:** On ANY verification failure, STOP and diagnose before proceeding. Never skip a failing checkpoint.

---

### J) Risk Analysis

**Risk 1: RNG Stream Divergence in Unified Path**
- Likelihood: LOW (verified by existing `TestClusterWorkloadGen_MatchesSimulator`)
- Impact: HIGH (golden test breakage)
- Mitigation: `SubsystemWorkload` uses master seed directly in both paths (`rng.go:73-76`). Existing test `TestClusterWorkloadGen_MatchesSimulator` (`cluster_test.go:564`) validates byte-level parity.

**Risk 2: SaveResults Instance ID Change**
- Likelihood: CERTAIN (ID changes from "default" to "instance_0")
- Impact: LOW (golden tests don't check instance ID strings; `SaveResults` uses ID for display only)
- Mitigation: Verify `testdata/goldendataset.json` does not contain instance IDs. If external tooling depends on "default" ID, document the change.

**Risk 3: Big PR Scope**
- Likelihood: MEDIUM (~10 files changed)
- Impact: MEDIUM (review fatigue, merge conflict risk)
- Mitigation: Atomic commits per batch. Each batch independently buildable and testable. Field privatization (Batch 4) can be deferred if time-constrained.

**Risk 4: Cluster Path Event Ordering for N=1**
- Likelihood: LOW
- Impact: HIGH (golden test breakage)
- Mitigation: `ClusterSimulator` with N=1 already passes golden equivalence test (`cluster_test.go:54`). The unified path adds zero admission/routing latency (defaults: `AdmissionLatency=0`, `RoutingLatency=0`), so event timestamps are unchanged.

**Risk 5: Import Cycle When Moving AdmissionPolicy**
- Likelihood: LOW
- Impact: MEDIUM (compile failure)
- Mitigation: `AdmissionPolicy` interface depends only on `*sim.Request` and `int64` — both in `sim/` package. Implementations (`AlwaysAdmit`, `TokenBucket`) also depend only on `sim.Request`. No import cycle. `cluster/` already imports `sim/`. `policy/` already imports `sim/`.

---

### K) Reviewer Challenges & Resolutions

**Challenge 1:** "The macro plan says `sim/policy/admission.go` should be ~10 LOC change but we're actually removing the entire file content and redirecting to `sim/`. Is this scope creep?"
- **Resolution:** ACCEPTED. The plan says "eliminate duplicated interface" — removing the duplicate implementations (`AlwaysAdmit`, `TokenBucket`) from `sim/policy/` is the natural consequence. The `sim/policy/` package remains as a home for future policy implementations. Alternatively, `sim/policy/admission.go` can re-export types from `sim/` for backward compat, but this violates NC-3 (no dead code).

**Challenge 2:** "BC-4 changes instance ID from 'default' to 'instance_0'. Is this a behavioral change?"
- **Resolution:** ACCEPTED as behavioral change, but LOW IMPACT. Golden tests verify metrics values, not instance ID strings. The ID appears only in `SaveResults` output. This is documented in the deviation log. If the user depends on "default" ID, they can use `--num-instances 1` and check `instance_0` instead.

**Challenge 3:** "IT-1 creates a new file `sim/admission.go`. The macro plan says 'No new files.' Should we put AdmissionPolicy in `sim/simulator.go` instead?"
- **Resolution:** ACCEPTED. Place `AdmissionPolicy` interface in a new file `sim/admission.go` for clarity — it's a distinct concept. The macro plan's "no new files" referred to no new packages, not no new files within existing packages. A single small file (interface + 2 implementations + factory) is cleaner than adding to the already-large `simulator.go`.

**Challenge 4:** "Field privatization in IT-8 — what if future PRs need external access to privatized fields?"
- **Resolution:** REJECTED. Fields being privatized are never accessed outside `sim/` (verified at `simplification-assessment.md:127-154`). If future PRs need access, they can add accessor methods at that time. Privatization is easily reversible.

**Challenge 5:** "The `newAdmissionPolicy` factory in `cluster.go:63-72` takes `DeploymentConfig`. After moving to `sim/`, it should take individual params."
- **Resolution:** ACCEPTED. The factory function in `sim/` will take `(name string, capacity, refillRate float64)` matching the existing `policy.NewAdmissionPolicy` signature. `ClusterSimulator` constructor calls it with `config.AdmissionPolicy`, `config.TokenBucketCapacity`, `config.TokenBucketRefillRate`.

---

### L) Design Sanity Checklist

- [x] No unnecessary abstractions — `SimConfig` is the only new type; replaces 5 constructors
- [x] No feature creep beyond PR scope — no new CLI flags, no new features
- [x] No unexercised flags or interfaces — `SimConfig` exercised by all constructors; `AdmissionPolicy` exercised by cluster tests
- [x] No partial implementations — all 4 simplifications (S1-S4) fully implemented
- [x] No breaking changes without explicit contract updates — BC-4 documents ID change
- [x] No hidden global state impact — all changes are local to struct initialization
- [x] All new code will pass golangci-lint — mechanical changes, no new patterns
- [x] Shared test helpers used from `sim/internal/testutil/` — no new test infrastructure needed
- [x] CLAUDE.md updated in IT-10 — new `SimConfig` struct, removed constructors, unified CLI path
- [x] No stale references left in CLAUDE.md — constructor list, file descriptions updated
- [x] Deviation log reviewed — 5 deviations, all resolved (CORRECTION or ADDITION)
- [x] All reviewer challenges addressed — 5 challenges, 4 accepted, 1 rejected with justification
- [x] Task graph has no circular dependencies — linear: IT-1→IT-2→IT-3/IT-4→IT-5→IT-6→IT-7→IT-8→IT-9→IT-10
- [x] Every contract (BC-N) covered by at least one task (IT-N) — see task graph
- [x] Every task has a verification command — see task graph
- [x] Parallelization claims correct — IT-3||IT-4 (different packages); IT-8→IT-9 sequential (same-package field references)
- [x] Batch checkpoints verify all contracts completed so far — see verification protocol

---

## PART 3: Execution Details

### M) Commit Strategy

One commit per batch:

1. **Batch 1 commit:** `refactor(sim): add SimConfig struct and sim.AdmissionPolicy interface`
   - Contracts: BC-1 (partial), BC-2 (partial), BC-6 (partial)
2. **Batch 2 commit:** `refactor(cluster): migrate constructors to SimConfig, consolidate AdmissionPolicy`
   - Contracts: BC-1, BC-2, BC-5, BC-6
3. **Batch 3 commit:** `refactor(cmd): unify CLI path through ClusterSimulator, remove old constructors`
   - Contracts: BC-3, BC-4, NC-1, NC-3
4. **Batch 4 commit:** `refactor(sim): privatize internal fields, update tests and CLAUDE.md`
   - Contracts: BC-7, NC-2, EC-1, EC-2

---

## APPENDIX

### N) File-Level Reference

#### sim/simulator.go

**Current constructors (to be replaced):**
- `newSimulatorBase(horizon, seed, totalKVBlocks, blockSizeTokens, maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64, betaCoeffs, alphaCoeffs []float64, modelConfig ModelConfig, hwConfig HardwareCalib, model, GPU string, tp int, roofline bool) *Simulator` — lines 117-148, 15 params
- `NewSimulator(horizon, seed, totalKVBlocks, blockSizeTokens, maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64, betaCoeffs, alphaCoeffs []float64, guideLLMConfig *GuideLLMConfig, modelConfig ModelConfig, hwConfig HardwareCalib, model, GPU string, tp int, roofline bool, tracesWorkloadFilePath string) *Simulator` — lines 150-171, 17 params
- `NewSimulatorWithoutWorkload(horizon, seed, totalKVBlocks, blockSizeTokens, maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64, betaCoeffs, alphaCoeffs []float64, modelConfig ModelConfig, hwConfig HardwareCalib, model, GPU string, tp int, roofline bool) *Simulator` — lines 175-182, 15 params

**New SimConfig struct (to be added):**
```go
// SimConfig holds all configuration for creating a Simulator.
type SimConfig struct {
    Horizon                   int64
    Seed                      int64
    TotalKVBlocks             int64
    BlockSizeTokens           int64
    MaxRunningReqs            int64
    MaxScheduledTokens        int64
    LongPrefillTokenThreshold int64
    BetaCoeffs                []float64
    AlphaCoeffs               []float64
    ModelConfig               ModelConfig
    HWConfig                  HardwareCalib
    Model                     string
    GPU                       string
    TP                        int
    Roofline                  bool
    // Workload config (optional — nil means no workload generation)
    GuideLLMConfig         *GuideLLMConfig
    TracesWorkloadFilePath string
}
```

**New constructor (replaces all 3):**

> **Note:** This sketch shows the FINAL state after IT-8 (field privatization). During Batch 1-2, field names remain PascalCase (e.g., `Clock`, `Horizon`). IT-8 renames them to lowercase. The temporary constructor name during IT-2 is `NewSimulatorFromConfig`; IT-7 renames it to `NewSimulator`.

```go
func NewSimulator(cfg SimConfig) *Simulator {
    s := &Simulator{
        clock:                     0,
        horizon:                   cfg.Horizon,
        eventQueue:                make(EventQueue, 0),
        waitQ:                     &WaitQueue{},
        kvCache:                   NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens),
        runningBatch:              &Batch{},
        Metrics:                   NewMetrics(),
        maxRunningReqs:            cfg.MaxRunningReqs,
        maxScheduledTokens:        cfg.MaxScheduledTokens,
        betaCoeffs:                cfg.BetaCoeffs,
        alphaCoeffs:               cfg.AlphaCoeffs,
        runningBatchFeatures:      RegressionFeatures{},
        longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
        stepCount:                 0,
        reqNumComputedTokens:      make(map[string]int64),
        modelConfig:               cfg.ModelConfig,
        hwConfig:                  cfg.HWConfig,
        model:                     cfg.Model,
        gpu:                       cfg.GPU,
        tp:                        cfg.TP,
        roofline:                  cfg.Roofline,
    }
    s.rng = NewPartitionedRNG(NewSimulationKey(cfg.Seed))

    // Generate workload if config provided
    if cfg.TracesWorkloadFilePath != "" && cfg.GuideLLMConfig == nil {
        s.tracesWorkloadFilePath = cfg.TracesWorkloadFilePath
        s.Metrics.RequestRate = 0.0
        s.generateWorkloadFromCSV()
    } else if cfg.GuideLLMConfig != nil {
        s.guideLLMConfig = cfg.GuideLLMConfig
        s.Metrics.RequestRate = cfg.GuideLLMConfig.Rate
        s.generateWorkloadDistribution()
    }
    // else: no workload — caller injects via InjectArrival

    return s
}
```

**Fields to privatize (currently exported, zero external access):**
| Field | Line | New Name |
|---|---|---|
| `MaxRunningReqs` | :90 | `maxRunningReqs` |
| `MaxScheduledTokens` | :92 | `maxScheduledTokens` |
| `BetaCoeffs` | :93 | `betaCoeffs` |
| `AlphaCoeffs` | :94 | `alphaCoeffs` |
| `RunningBatchFeatures` | :97 | `runningBatchFeatures` |
| `LongPrefillTokenThreshold` | :98 | `longPrefillTokenThreshold` |
| `StepEvent` | :99 | `stepEvent` |
| `StepCount` | :100 | `stepCount` |
| `ReqNumComputedTokens` | :102 | `reqNumComputedTokens` |
| `PreemptionHappened` | :103 | `preemptionHappened` |
| `GuideLLMConfig` | :104 | `guideLLMConfig` |
| `Model` | :105 | `model` |
| `GPU` | :106 | `gpu` |
| `TP` | :107 | `tp` |
| `Roofline` | :108 | `roofline` |
| `TracesWorkloadFilePath` | :109 | `tracesWorkloadFilePath` |
| `ModelConfig` | :110 | `modelConfig` |
| `HWConfig` | :111 | `hwConfig` |
| `EventQueue` | :76 | `eventQueue` |

**Fields remaining public (accessed by InstanceSimulator or tests):**
| Field | Accessor in InstanceSimulator | Reason |
|---|---|---|
| `Clock` | `instance.go:98` | `InstanceSimulator.Clock()` |
| `Horizon` | `instance.go:109` | `InstanceSimulator.Horizon()` |
| `Metrics` | `instance.go:104` | `InstanceSimulator.Metrics()` |
| `WaitQ` | `instance.go:158` | `InstanceSimulator.QueueDepth()` |
| `KVCache` | `instance.go:171-177` | `InstanceSimulator.KVUtilization()`, `FreeKVBlocks()` |
| `RunningBatch` | `instance.go:163-166` | `InstanceSimulator.BatchSize()` |

#### sim/admission.go (NEW FILE)

```go
package sim

import "fmt"

// AdmissionPolicy decides whether a request is admitted to the cluster.
type AdmissionPolicy interface {
    Admit(req *Request, clock int64) (admitted bool, reason string)
}

// AlwaysAdmit admits all requests unconditionally.
type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *Request, _ int64) (bool, string) {
    return true, ""
}

// TokenBucket implements rate-limiting admission control.
type TokenBucket struct {
    capacity      float64
    refillRate    float64
    currentTokens float64
    lastRefill    int64
}

func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
    return &TokenBucket{
        capacity:      capacity,
        refillRate:    refillRate,
        currentTokens: capacity,
    }
}

func (tb *TokenBucket) Admit(req *Request, clock int64) (bool, string) {
    elapsed := clock - tb.lastRefill
    if elapsed > 0 {
        refill := float64(elapsed) * tb.refillRate / 1e6
        tb.currentTokens = min(tb.capacity, tb.currentTokens+refill)
        tb.lastRefill = clock
    }
    cost := float64(len(req.InputTokens))
    if tb.currentTokens >= cost {
        tb.currentTokens -= cost
        return true, ""
    }
    return false, "insufficient tokens"
}

// NewAdmissionPolicy creates an admission policy by name.
func NewAdmissionPolicy(name string, capacity, refillRate float64) AdmissionPolicy {
    switch name {
    case "", "always-admit":
        return &AlwaysAdmit{}
    case "token-bucket":
        return NewTokenBucket(capacity, refillRate)
    default:
        panic(fmt.Sprintf("unknown admission policy %q", name))
    }
}
```

#### sim/cluster/instance.go

**Current constructors (to be replaced):**
- `NewInstanceSimulator(id, horizon, seed, ...)` — 18 params, lines 30-73
- `NewInstanceSimulatorWithoutWorkload(id, horizon, seed, ...)` — 16 params, lines 114-127

**New constructor:**
```go
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
    return &InstanceSimulator{
        id:  id,
        sim: sim.NewSimulator(cfg),
    }
}
```

#### sim/cluster/cluster.go

**Changes:**
- Remove `AdmissionPolicy` interface (lines 18-20), `AlwaysAdmit` (lines 22-27), `TokenBucket` (lines 29-60), `NewTokenBucket` (lines 38-44), `newAdmissionPolicy` (lines 63-72)
- `admissionPolicy` field type changes from `AdmissionPolicy` to `sim.AdmissionPolicy`
- `NewClusterSimulator` creates instances via `NewInstanceSimulator(id, simConfig)` instead of `NewInstanceSimulatorWithoutWorkload(id, 16 params)`

#### sim/cluster/deployment.go

**Add method:**
```go
// ToSimConfig converts DeploymentConfig to SimConfig for per-instance construction.
// GuideLLMConfig and TracesWorkloadFilePath are intentionally omitted:
// cluster mode generates workload centrally and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
    return sim.SimConfig{
        Horizon:                   d.Horizon,
        Seed:                      d.Seed,
        TotalKVBlocks:             d.TotalKVBlocks,
        BlockSizeTokens:           d.BlockSizeTokens,
        MaxRunningReqs:            d.MaxRunningReqs,
        MaxScheduledTokens:        d.MaxScheduledTokens,
        LongPrefillTokenThreshold: d.LongPrefillTokenThreshold,
        BetaCoeffs:                d.BetaCoeffs,
        AlphaCoeffs:               d.AlphaCoeffs,
        ModelConfig:               d.ModelConfig,
        HWConfig:                  d.HWConfig,
        Model:                     d.Model,
        GPU:                       d.GPU,
        TP:                        d.TP,
        Roofline:                  d.Roofline,
    }
}
```

#### cmd/root.go

**Current two code paths (lines 201-258) → single cluster path:**
```go
// Build deployment config (always)
config := cluster.DeploymentConfig{
    NumInstances:              numInstances,
    Horizon:                   simulationHorizon,
    Seed:                      seed,
    // ... all fields ...
    AdmissionPolicy:           admissionPolicy,
    AdmissionLatency:          admissionLatency,
    RoutingLatency:            routingLatency,
    TokenBucketCapacity:       tokenBucketCapacity,
    TokenBucketRefillRate:     tokenBucketRefillRate,
}
cs := cluster.NewClusterSimulator(config, guideLLMConfig, tracesWorkloadFilePath)
cs.Run()

// Save results
for _, inst := range cs.Instances() {
    inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, startTime, "")
}
cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, startTime, resultsPath)
```

Note: For N=1, `AggregatedMetrics()` returns the single instance's metrics. Per-instance stdout print MUST be suppressed when N=1 (to avoid duplicate output). Only the aggregated `SaveResults` call prints to stdout and saves to file. This preserves the old single-call behavior.

```go
// Unified CLI path (replaces both branches)
cs := cluster.NewClusterSimulator(config, guideLLMConfig, tracesWorkloadFilePath)
cs.Run()

if numInstances > 1 {
    // Print per-instance metrics to stdout (multi-instance only)
    for _, inst := range cs.Instances() {
        inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, startTime, "")
    }
}
// Save aggregated metrics (always prints to stdout + saves to file if resultsPath set)
cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, startTime, resultsPath)
```

#### sim/policy/admission.go

**After dedup:** Delete `sim/policy/admission.go` entirely. The `sim/policy/` package directory is also deleted. PR6 will re-create `sim/policy/` when adding `RoutingPolicy` — this avoids leaving an empty package that violates NC-3 (no dead code).

#### Test files

All test call sites that use old constructors MUST be updated to use `SimConfig`. Example migration:

```go
// Before (sim/simulator_test.go):
sim := NewSimulator(math.MaxInt64, tc.Seed, tc.TotalKVBlocks, ...)

// After:
sim := NewSimulator(SimConfig{
    Horizon:     math.MaxInt64,
    Seed:        tc.Seed,
    TotalKVBlocks: tc.TotalKVBlocks,
    // ...
})
```

Same pattern for `NewInstanceSimulator` and `NewInstanceSimulatorWithoutWorkload` call sites.
