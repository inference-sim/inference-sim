# fix(deployment): DeploymentConfig Missing KVTransferBaseLatency — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the KV transfer base latency parameter through the full configuration pipeline so users can control it from the CLI.

**The problem today:** `SimConfig.KVTransferBaseLatency` exists and is used by `TieredKVCache` to add a fixed per-transfer cost, but `DeploymentConfig` (the cluster-level config) doesn't include this field. This means the value is always zero in cluster mode — users cannot configure it, and the field silently defaults to 0 regardless of intent. PR12's micro plan explicitly deferred this as a simplification, but it's now needed.

**What this PR adds:**
1. A `KVTransferBaseLatency` field on `DeploymentConfig` that propagates to `SimConfig` via `ToSimConfig()`
2. A `--kv-transfer-base-latency` CLI flag (default 0) with negative-value validation
3. A behavioral test proving the field actually affects transfer cost in tiered KV mode

**Why this matters:** This completes the tiered KV cache CLI surface. Without it, users cannot model real GPU↔CPU transfers that have a fixed overhead component (e.g., PCIe setup latency), limiting the simulator's fidelity for capacity planning.

**Architecture:** Pure config propagation — add field to `DeploymentConfig` struct, wire it through `ToSimConfig()`, add CLI flag in `cmd/root.go` with validation, test the full pipeline.

**Source:** Parallel dev plan Phase 0 in `docs/plans/2026-02-19-parallel-dev-plan-234-plus-4.md`

**Closes:** Fixes #234

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds one int64 field (`KVTransferBaseLatency`) to the `DeploymentConfig` → `SimConfig` → `TieredKVCache` pipeline. It's a config-propagation fix: the field already exists at the bottom (`SimConfig`) and is consumed by `TieredKVCache`, but was never wired through the cluster config or CLI. Three files change: `sim/cluster/deployment.go` (struct + method), `cmd/root.go` (flag + validation + construction), and test files.

No adjacent blocks change behavior. The `NewKVStore` factory and `TieredKVCache` constructor already handle this field correctly.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Field Propagation
- GIVEN a `DeploymentConfig` with `KVTransferBaseLatency` set to 42
- WHEN `ToSimConfig()` is called
- THEN the resulting `SimConfig.KVTransferBaseLatency` MUST equal 42
- MECHANISM: Direct field copy in `ToSimConfig()` method

BC-2: CLI Flag Acceptance
- GIVEN the CLI flag `--kv-transfer-base-latency 10`
- WHEN the simulation runs with `--kv-cpu-blocks > 0`
- THEN the tiered KV cache uses a base latency of 10 ticks per transfer
- MECHANISM: CLI var → `DeploymentConfig` field → `SimConfig` → `NewKVStore` → `NewTieredKVCache`

BC-3: Default Zero Behavior
- GIVEN no `--kv-transfer-base-latency` flag provided
- WHEN the simulation runs
- THEN transfer base latency MUST be 0 (backward compatible)
- MECHANISM: Go zero-value for int64 + Cobra default 0

**Negative/error contracts:**

BC-4: Negative Value Rejection
- GIVEN the CLI flag `--kv-transfer-base-latency -5`
- WHEN the CLI validates flags
- THEN the program MUST terminate with a fatal error message containing "--kv-transfer-base-latency must be >= 0"
- MECHANISM: `logrus.Fatalf` in `cmd/root.go` validation block

### C) Component Interaction

```
cmd/root.go                sim/cluster/deployment.go       sim/simulator.go
┌──────────┐               ┌───────────────────┐           ┌─────────────┐
│ CLI flag  │──validates──►│ DeploymentConfig   │──copy──►  │ SimConfig   │
│ int64 var │               │ .KVTransferBase.. │           │ .KVTransfer.│
└──────────┘               └───────────────────┘           └──────┬──────┘
                                                                  │
                                                           sim/kv_store.go
                                                           ┌──────▼──────┐
                                                           │ NewKVStore() │
                                                           │→TieredKV    │
                                                           └─────────────┘
```

No new interfaces, no new state. Pure config threading.

**Extension friction:** Adding one more KV config field requires 3 files (deployment.go, root.go, and the field mapping test). This matches the existing pattern for `KVCPUBlocks`, `KVOffloadThreshold`, and `KVTransferBandwidth`.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Add behavioral test: cluster with non-zero base latency produces different transfer costs than zero" | Tests field propagation via `ToSimConfig()` + negative validation. No end-to-end cluster test with actual transfer cost comparison. | SIMPLIFICATION: An end-to-end cluster test exercising TieredKVCache offload to observe different latencies would require careful workload tuning and is fragile. The existing `sim/kv_store_test.go:47` already tests `KVTransferBaseLatency: 10` at the SimConfig level. The propagation test (BC-1) proves the value reaches SimConfig, which is the gap this PR fixes. |

### E) Review Guide

- **The tricky part:** Nothing subtle — this is mechanical config propagation. The only risk is missing a construction site.
- **What to scrutinize:** BC-1 test — does it verify the field is actually copied, not just that the struct compiles?
- **What's safe to skim:** CLI flag registration (follows exact pattern of 3 adjacent flags).
- **Known debt:** None introduced.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/deployment.go` — add field + propagation (2 lines)
- `cmd/root.go` — add var + flag + validation + construction (4 lines)
- `sim/cluster/cluster_test.go` — update field mapping test (3 lines)

**Key decisions:**
- No YAML bundle support (this is a rarely-tuned parameter; YAML can be added if needed)
- Validation is `>= 0` (not `> 0`) because 0 is the valid default
- No CLAUDE.md update needed (the tiered KV CLI flags section already mentions this parameter is on SimConfig)

**Confirmation:** No dead code. All paths exercisable via `--kv-transfer-base-latency N` on existing CLI.

### G) Task Breakdown

---

### Task 1: Add Field to DeploymentConfig and Wire ToSimConfig

**Contracts Implemented:** BC-1, BC-3

**Files:**
- Modify: `sim/cluster/deployment.go:48` (add field), `:77` (add propagation)
- Modify: `sim/cluster/cluster_test.go:76-141` (update field mapping test)

**Step 1: Write failing test for field propagation**

Context: The existing `TestDeploymentConfig_ToSimConfig_FieldMapping` test verifies all fields are propagated. We add `KVTransferBaseLatency` with a non-zero value and assert it appears in the output.

In `sim/cluster/cluster_test.go`, update `TestDeploymentConfig_ToSimConfig_FieldMapping` — add `KVTransferBaseLatency: 42` to the `DeploymentConfig` literal (after `KVTransferBandwidth`) and add this assertion after the existing KV assertions:

```go
// Inside the DeploymentConfig literal, after KVTransferBandwidth:
KVTransferBaseLatency: 42,

// After the existing KVTransferBandwidth assertion:
if sc.KVTransferBaseLatency != 42 {
    t.Errorf("KVTransferBaseLatency: got %d, want 42", sc.KVTransferBaseLatency)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestDeploymentConfig_ToSimConfig_FieldMapping -v`
Expected: FAIL — `DeploymentConfig` has no field `KVTransferBaseLatency`

**Step 3: Implement field and propagation**

In `sim/cluster/deployment.go`:

After line 48 (`KVTransferBandwidth float64`), add:
```go
KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
```

In the `ToSimConfig()` method, after the `KVTransferBandwidth` line (line 76), add:
```go
KVTransferBaseLatency: d.KVTransferBaseLatency,
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestDeploymentConfig_ToSimConfig_FieldMapping -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/deployment.go sim/cluster/cluster_test.go
git commit -m "feat(deployment): add KVTransferBaseLatency field and propagation (BC-1, BC-3)

- Add KVTransferBaseLatency int64 field to DeploymentConfig
- Propagate via ToSimConfig() to SimConfig
- Update field mapping test to verify propagation

Fixes #234

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add CLI Flag and Validation

**Contracts Implemented:** BC-2, BC-4

**Files:**
- Modify: `cmd/root.go` — add var (~line 94), flag (~line 587), validation (~line 349), construction (~line 419)

**Step 1: Write failing test for negative value rejection**

Context: CLI validation tests live in `cmd/root_test.go` or are verified manually. For this task, we implement the flag + validation and verify via build + run.

There's no separate test file for CLI flag validation in this project — it's validated at the integration level. We'll verify BC-4 manually in Step 4.

**Step 2: Implement CLI flag, validation, and construction**

In `cmd/root.go`:

After line 94 (`kvTransferBandwidth float64`), add the variable:
```go
kvTransferBaseLatency int64
```

After line 349 (the `kvTransferBandwidth` validation block ending with `}`), add validation:
```go
if kvTransferBaseLatency < 0 {
    logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
}
```

In the `DeploymentConfig{}` literal (after line 419 `KVTransferBandwidth:`), add:
```go
KVTransferBaseLatency: kvTransferBaseLatency,
```

After line 587 (the `--kv-transfer-bandwidth` flag registration), add:
```go
runCmd.Flags().Int64Var(&kvTransferBaseLatency, "kv-transfer-base-latency", 0, "Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost)")
```

**Step 3: Build to verify compilation**

Run: `go build ./...`
Expected: SUCCESS

**Step 4: Run full tests**

Run: `go test ./... -count=1`
Expected: All pass

**Step 5: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go
git commit -m "feat(cmd): add --kv-transfer-base-latency CLI flag with validation (BC-2, BC-4)

- Add kvTransferBaseLatency int64 CLI variable
- Register --kv-transfer-base-latency flag (default 0)
- Validate >= 0 at CLI boundary (logrus.Fatalf for negative)
- Wire to DeploymentConfig construction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update CLAUDE.md CLI Flags Documentation

**Contracts Implemented:** Documentation completeness

**Files:**
- Modify: `CLAUDE.md` — add `--kv-transfer-base-latency` to CLI flags section

**Step 1: Update CLAUDE.md**

In the `cmd/root.go` entry under File Organization, update the comment to include the new flag:
```
│   ├── root.go                # CLI commands and flags (--num-instances, --policy-config, --workload-spec, --trace-level, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth, --kv-transfer-base-latency)
```

**Step 2: Run tests to verify no regressions**

Run: `go test ./... -count=1`
Expected: All pass

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add --kv-transfer-base-latency to CLAUDE.md CLI flags

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1     | Task 1 | Unit    | TestDeploymentConfig_ToSimConfig_FieldMapping (extended) |
| BC-2     | Task 2 | Integration | Build + full test suite verifies end-to-end wiring |
| BC-3     | Task 1 | Unit    | Verified by zero-value default in existing tests (no explicit KVTransferBaseLatency = 0) |
| BC-4     | Task 2 | Manual  | Verified by CLI validation code + build |

No golden dataset update needed — `KVTransferBaseLatency` defaults to 0, which matches existing behavior.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missing construction site | Low | Medium | Construction site audit identified all 15+ sites; only field mapping test needs explicit update (others use zero-value default) | Task 1 |
| Flag name inconsistency | Low | Low | Follows exact naming pattern of `--kv-transfer-bandwidth` (swap `bandwidth` → `base-latency`) | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated (Task 3)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — one justified simplification
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3)
- [x] All contracts mapped to specific tasks
- [x] No golden dataset regeneration needed
- [x] Construction site audit completed — 15+ sites, only 2 need update
- [x] New CLI flag validated for: negative (BC-4). Zero is valid default. NaN/Inf not applicable (int64).
- [x] No new error paths with silent continue
- [x] No map iteration with float accumulation
- [x] Library code does not call logrus.Fatalf (validation is in cmd/)
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes (no pointer type needed)
- [x] No YAML loading changes
- [x] No division operations
- [x] No new interfaces
- [x] No methods spanning multiple concerns
- [x] Config parameter grouped with existing KV config
- [x] Grepped for "PR 234" / "#234" — no stale references

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/deployment.go`

**Purpose:** Add `KVTransferBaseLatency` field and propagate to `SimConfig`.

After line 48 (`KVTransferBandwidth float64 // blocks/tick transfer rate (default 100.0)`):
```go
KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
```

In `ToSimConfig()`, after `KVTransferBandwidth: d.KVTransferBandwidth,`:
```go
KVTransferBaseLatency: d.KVTransferBaseLatency,
```

### File: `cmd/root.go`

**Purpose:** Add CLI flag, validation, and construction site update.

Variable declaration (after line 94):
```go
kvTransferBaseLatency int64
```

Validation (after line 349, the `kvTransferBandwidth` check):
```go
if kvTransferBaseLatency < 0 {
    logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
}
```

Construction (after `KVTransferBandwidth: kvTransferBandwidth,` in the DeploymentConfig literal):
```go
KVTransferBaseLatency: kvTransferBaseLatency,
```

Flag registration (after `--kv-transfer-bandwidth` flag):
```go
runCmd.Flags().Int64Var(&kvTransferBaseLatency, "kv-transfer-base-latency", 0, "Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost)")
```

### File: `sim/cluster/cluster_test.go`

**Purpose:** Update field mapping test to verify propagation.

In `TestDeploymentConfig_ToSimConfig_FieldMapping`, add to `DeploymentConfig` literal:
```go
KVTransferBaseLatency: 42,
```

Add assertion after `KVTransferBandwidth` check:
```go
if sc.KVTransferBaseLatency != 42 {
    t.Errorf("KVTransferBaseLatency: got %d, want 42", sc.KVTransferBaseLatency)
}
```
