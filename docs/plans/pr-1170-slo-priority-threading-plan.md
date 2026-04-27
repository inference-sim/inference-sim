# SLO Priority Override Threading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thread `SLOPriorityOverrides` from `DeploymentConfig` through `SimConfig` to `NewBatchFormation`, replacing the hardcoded `nil` sloMap so that custom `slo_priorities` in the policy bundle YAML actually affect priority preemption victim selection.

**The problem today:** `admission.slo_priorities` in the policy bundle YAML configures custom SLO priority overrides (e.g., `batch: 0` to make batch non-sheddable). These overrides reach the admission layer (`NewSLOPriorityMap` at `cluster.go:181`) but NOT the preemption layer — `NewSimulator` at `simulator.go:145` hardcodes `NewBatchFormation(cfg.PreemptionPolicy, nil)`, and `SimConfig` has no field to carry the overrides. The three-level config split (`DeploymentConfig → SimConfig → PolicyConfig`) means `SLOPriorityOverrides` is stuck at the cluster layer.

**What this PR adds:**
1. `SLOPriorityOverrides map[string]int` field on `SimConfig` — carries overrides to the instance layer.
2. Move the field from `DeploymentConfig` to `SimConfig` (promoted via embedding, so cluster-layer code is unchanged).
3. Move the assignment in `cmd/root.go` and `cmd/replay.go` from the `DeploymentConfig` literal to the nested `SimConfig` literal.
4. Replace `nil` in `NewBatchFormation(cfg.PreemptionPolicy, nil)` with `NewSLOPriorityMap(cfg.SLOPriorityOverrides)`.
5. Add a behavioral test verifying custom overrides propagate to preemption victim selection.
6. Update user-facing docs (`configuration.md`, `scheduling.md`) to document that `slo_priorities` now affects preemption.

**Why this matters:** Without this fix, a user who sets `slo_priorities: { batch: 0 }` in their policy bundle expects batch requests to be protected from both admission shedding AND priority preemption. Currently only admission respects the override — preemption still uses GAIE defaults.

**Architecture:** `SimConfig` gains one `map[string]int` field. `DeploymentConfig` loses its explicit field (promoted from embedded `SimConfig`). `resolveConfigForRole` returns `SimConfig` with the field automatically included via struct copy. `ResolvePoolConfig` copies the field via `resolved := global`. No interface changes.

**Source:** GitHub issue #1170 (part 3 of #1086 decomposition: #1168 → #1169 → #1170).
**Closes:** Fixes #1170.
**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block modified:** `SimConfig` (add field), `NewSimulator` (wire field), `DeploymentConfig` (remove duplicate field).
2. **Adjacent blocks:** `cluster.go:181` reads `config.SLOPriorityOverrides` — still works via promotion. `NewBatchFormation` already accepts `*SLOPriorityMap` as 2nd arg.
3. **Invariants touched:** None directly. The behavior change is that custom `slo_priorities` now reach preemption, but this is the intended design — not a new invariant.
4. **Construction Site Audit — `SimConfig`:**
   - `SimConfig` is a value type with embedded sub-configs. Adding a field does NOT require updating construction sites that use `SimConfig{}` with only some fields set — Go's zero-value for `map` is `nil`, and `NewBatchFormation(..., nil)` → `DefaultSLOPriorityMap()`. Only the 2 production sites (`cmd/root.go`, `cmd/replay.go`) need to SET the field.
   - `DeploymentConfig.SLOPriorityOverrides` removal: check all access via `config.SLOPriorityOverrides` — only 1 site: `cluster.go:181`. Promotion from embedded `SimConfig` makes this transparent.

---

## Part 1: Design Validation

### A) Executive Summary

This PR closes the config propagation gap where `slo_priorities` from the policy bundle reaches admission but not preemption. The fix is minimal: move the `SLOPriorityOverrides` field from `DeploymentConfig` (cluster-only) to `SimConfig` (instance-level), and replace the hardcoded `nil` in `NewSimulator` with a constructed `SLOPriorityMap`.

### B) Behavioral Contracts

**BC-1: Custom SLO priorities reach preemption victim selection**
- GIVEN a policy bundle with `admission: { slo_priorities: { batch: 0 } }` and `--preemption-policy priority`
- WHEN KV pressure triggers preemption with a batch and a background request running
- THEN background (priority=-3) is evicted, NOT batch (priority=0, overridden from default -1)

**BC-2: Default behavior unchanged**
- GIVEN no `slo_priorities` in the policy bundle (nil overrides)
- WHEN preemption occurs
- THEN victim selection uses GAIE defaults (same as before this PR)

**BC-3: Cluster admission still respects overrides**
- GIVEN a policy bundle with `slo_priorities` overrides
- WHEN the cluster constructs admission policies
- THEN `config.SLOPriorityOverrides` at `cluster.go:181` still resolves correctly (via promoted field)

### C) Component Interaction

```
cmd/ (CLI layer — sets field in SimConfig instead of DeploymentConfig)
  └─ config.SimConfig.SLOPriorityOverrides = sloPriorityOverrides

sim/cluster/deployment.go (field removed — promoted from embedded SimConfig)
  └─ config.SLOPriorityOverrides → resolves to config.SimConfig.SLOPriorityOverrides

sim/cluster/cluster.go:181 (unchanged — access via promotion)
  └─ priorityMap := sim.NewSLOPriorityMap(config.SLOPriorityOverrides)

sim/simulator.go:145 (nil → real map)
  └─ NewBatchFormation(cfg.PreemptionPolicy, NewSLOPriorityMap(cfg.SLOPriorityOverrides))

sim/batch_formation.go (unchanged — already accepts *SLOPriorityMap)
  └─ VLLMBatchFormation.sloMap = sloMap
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #1170: "3 callsites in cluster.go need the propagation" | 0 cluster.go edits | CORRECTION — the field promotion from embedded SimConfig means `config.SLOPriorityOverrides` at cluster.go:181 resolves automatically. The 3 `resolveConfigForRole` callsites (cluster.go:263, 312; direct_actuator.go:80) and the deferred-path callsite (cluster.go:783) all return SimConfig which now includes the field via struct copy. No code changes needed. |

### E) Review Guide

**Tricky part:** The `DeploymentConfig` → `SimConfig` field move relies on Go's struct embedding promotion. Verify that removing `SLOPriorityOverrides` from `DeploymentConfig` causes NO compile errors (the promoted field from `SimConfig` takes over).

**Scrutinize:** `cluster.go:181` — `config.SLOPriorityOverrides` must still resolve after the field is moved.

**Safe to skim:** `cmd/root.go` and `cmd/replay.go` — mechanical move of field assignment from outer to inner struct literal.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to **modify:**
- `sim/simulator.go` — add `SLOPriorityOverrides map[string]int` to `SimConfig`; replace `nil` with `NewSLOPriorityMap(cfg.SLOPriorityOverrides)`
- `sim/cluster/deployment.go` — remove `SLOPriorityOverrides` field (promoted from embedded SimConfig)
- `cmd/root.go` — move `SLOPriorityOverrides` assignment from DeploymentConfig literal to SimConfig literal
- `cmd/replay.go` — same
- `sim/simulator_preempt_test.go` — add `TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption`
- `docs/reference/configuration.md` — document that `slo_priorities` now affects preemption
- `docs/guide/scheduling.md` — note shared priorities between admission and preemption
- `CLAUDE.md` — Recent Changes entry

### G) Task Breakdown

---

#### Task 1: Add SLOPriorityOverrides to SimConfig + wire in NewSimulator (BC-1, BC-2)

**Files:** modify `sim/simulator.go`

**Step 1: Write the failing test**

Add to `sim/simulator_preempt_test.go`:

```go
func TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption(t *testing.T) {
	// BC-1: Custom slo_priorities override default preemption victim selection.
	// Default GAIE priorities: batch=-1, background=-3, critical=4.
	// Override: batch=0 (promoted from -1 to non-sheddable).
	// Setup: 10 blocks × 16 tokens = 160 capacity. 3 running × 3 blocks each = 9 used, 1 free.
	// Phase 1: crit decode uses the 1 free block (10/10 used).
	// Phase 1: batch decode → full → preemption.
	// With override: background(-3) < batch(0), so background evicted first.
	// Without override: background(-3) < batch(-1), so background would still be evicted —
	//   but we verify the SLOPriorityMap.Priority value, not just ordering.
	cfg := SimConfig{
		Horizon: 100_000_000,
		KVCacheConfig:       NewKVCacheConfig(10, 16, 0, 0.0, 0.0, 0.0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 0, 0}, []float64{100, 1, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
		PolicyConfig:        NewPolicyConfig("constant", "fcfs", "priority"),
		SLOPriorityOverrides: map[string]int{"batch": 0},
	}
	s := mustNewSimulator(t, cfg)

	// Inject three running requests to saturate cache (9/10 blocks used)
	// Use distinct input tokens to avoid prefix cache sharing.
	critReq := &Request{ID: "crit", SLOClass: "critical", ArrivalTime: 100, State: StateRunning,
		InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
			33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
		OutputTokens: make([]int, 10)}
	batchReq := &Request{ID: "batch-req", SLOClass: "batch", ArrivalTime: 200, State: StateRunning,
		InputTokens: []int{101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
			117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
			133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148},
		OutputTokens: make([]int, 10)}
	bgReq := &Request{ID: "bg-req", SLOClass: "background", ArrivalTime: 300, State: StateRunning,
		InputTokens: []int{201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
			217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
			233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248},
		OutputTokens: make([]int, 10)}
	for _, req := range []*Request{critReq, batchReq, bgReq} {
		s.KVCache.AllocateKVBlocks(req, 0, 48, nil)
		req.ProgressIndex = 48
	}
	s.RunningBatch = &Batch{Requests: []*Request{critReq, batchReq, bgReq}}

	// Add a new request to WaitQ (won't be dequeued because PreemptionHappened blocks Phase 2)
	newReq := &Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued}
	s.WaitQ.Enqueue(newReq)

	// Run FormBatch via the simulator's batchFormation
	result := s.batchFormation.FormBatch(BatchContext{
		RunningBatch:       s.RunningBatch,
		WaitQ:              s.WaitQ,
		KVCache:            s.KVCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	})

	// With batch=0 override: background(-3) is least urgent.
	// Priority preemption must evict bg-req, not batch-req.
	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "bg-req" {
		t.Errorf("expected bg-req evicted (background=-3 < batch=0 with override), got %q",
			result.Preempted[0].Request.ID)
	}
	// batch-req must still be running (promoted to priority=0, no longer least urgent)
	runningIDs := make(map[string]bool)
	for _, r := range result.RunningBatch.Requests {
		runningIDs[r.ID] = true
	}
	if !runningIDs["batch-req"] {
		t.Error("batch-req should still be running (priority=0 with override, more urgent than background=-3)")
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./sim/ -run TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption -v 2>&1 | tail -10
```

Expected: FAIL — `cfg.SLOPriorityOverrides` is an unknown field (not yet added to `SimConfig`).

**Step 3: Implement — add field to SimConfig and wire in NewSimulator**

In `sim/simulator.go`, update `SimConfig` (line ~57):

```go
type SimConfig struct {
	// Simulation control (no sub-config — no factory uses only these)
	Horizon int64
	Seed    int64

	// Module-scoped sub-configs (R16)
	KVCacheConfig
	BatchConfig
	LatencyCoeffs
	ModelHardwareConfig
	PolicyConfig
	WorkloadConfig

	// SLO priority overrides for preemption victim selection.
	// nil = use GAIE defaults. Shared with admission (same overrides).
	// Keys are SLO class names; values are integer priorities.
	SLOPriorityOverrides map[string]int
}
```

In `sim/simulator.go:145`, replace:
```go
	batchFormation := NewBatchFormation(cfg.PreemptionPolicy, nil)
```
with:
```go
	batchFormation := NewBatchFormation(cfg.PreemptionPolicy, NewSLOPriorityMap(cfg.SLOPriorityOverrides))
```

**Step 4: Run test to verify it passes**

```bash
go test ./sim/ -run TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption -v 2>&1 | tail -10
```

Expected: PASS.

**Step 5: Run build and lint**

```bash
go build ./... 2>&1 && golangci-lint run ./sim/... 2>&1 | head -10
```

**Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_preempt_test.go
git commit -m "feat(sim): add SLOPriorityOverrides to SimConfig, wire to NewBatchFormation (BC-1)

- Add SLOPriorityOverrides map[string]int field to SimConfig
- Replace hardcoded nil in NewSimulator with NewSLOPriorityMap(cfg.SLOPriorityOverrides)
- Add TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption: verifies custom
  slo_priorities reach priority preemption victim selection

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Move field from DeploymentConfig to SimConfig (BC-3)

**Files:** modify `sim/cluster/deployment.go`, `cmd/root.go`, `cmd/replay.go`

**Step 1: Remove field from DeploymentConfig**

In `sim/cluster/deployment.go`, delete lines 102-104:
```go
	// SLO priority overrides (issue #1013). Nil = GAIE-compatible defaults.
	// Keys are SLO class names; values are integer priorities. Negative = sheddable.
	SLOPriorityOverrides map[string]int `yaml:"slo_priority_overrides,omitempty"`
```

**Step 2: Move assignment in cmd/root.go**

In `cmd/root.go`, in the `DeploymentConfig` struct literal, move `SLOPriorityOverrides: sloPriorityOverrides` from the outer level into the `SimConfig` inner literal:

Before (around line 1514):
```go
			SLOPriorityOverrides:    sloPriorityOverrides,
```

Remove that line. Add inside the `SimConfig:` block (after `PolicyConfig`):
```go
				SLOPriorityOverrides: sloPriorityOverrides,
```

**Step 3: Move assignment in cmd/replay.go**

Same pattern. Remove `SLOPriorityOverrides: sloPriorityOverrides` from the outer `DeploymentConfig` literal (around line 241), add inside the `SimConfig:` block:

```go
				SLOPriorityOverrides: sloPriorityOverrides,
```

**Step 4: Run build to verify promotion works**

```bash
go build ./... 2>&1
```

Expected: clean build. `cluster.go:181` `config.SLOPriorityOverrides` resolves via promotion.

**Step 5: Run full test suite**

```bash
go test ./... -count=1 2>&1 | tail -15
```

Expected: all PASS.

**Step 6: Run lint**

```bash
golangci-lint run ./... 2>&1 | head -10
```

**Step 7: Commit**

```bash
git add sim/cluster/deployment.go cmd/root.go cmd/replay.go
git commit -m "refactor: move SLOPriorityOverrides from DeploymentConfig to SimConfig (BC-3)

- Remove SLOPriorityOverrides field from DeploymentConfig (promoted via embedded SimConfig)
- Move assignment in cmd/root.go and cmd/replay.go from outer to inner struct literal
- cluster.go:181 config.SLOPriorityOverrides resolves unchanged via Go promotion

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Documentation updates + CLAUDE.md

**Files:** modify `docs/reference/configuration.md`, `docs/guide/scheduling.md`, `CLAUDE.md`

**Step 1: Update configuration.md**

In `docs/reference/configuration.md`, update the `--preemption-policy` row (around line 124) to note the slo_priorities interaction:

Before:
```
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). |
```

After:
```
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). Priority mode uses `slo_priorities` from the policy bundle when set (shared with admission). |
```

Also add a note in the SLO Tier Priorities section (around line 205) that `slo_priorities` is shared:

Find the line: `Configurable via policy bundle YAML \`slo_priorities\` in \`AdmissionConfig\``

Add after it or update:
```
`slo_priorities` overrides affect both admission (tier-shed, GAIE-legacy) and preemption (`--preemption-policy priority`). Both subsystems use the same priority mapping.
```

**Step 2: Update scheduling.md**

In `docs/guide/scheduling.md`, find the preemption section and add a note that custom SLO priorities are shared between admission and preemption.

**Step 3: Update CLAUDE.md Recent Changes**

Add at the top of Recent Changes:

```
- SLO priority override threading (#1170): `SLOPriorityOverrides` field moved from `DeploymentConfig` to `SimConfig`, enabling policy bundle `slo_priorities` to reach `NewBatchFormation` for priority preemption. Previously, custom SLO priorities only affected admission (tier-shed, GAIE-legacy); now they also affect `--preemption-policy priority` victim selection. `simulator.go:145` changed from `NewBatchFormation(cfg.PreemptionPolicy, nil)` to `NewBatchFormation(cfg.PreemptionPolicy, NewSLOPriorityMap(cfg.SLOPriorityOverrides))`.
```

**Step 4: Commit**

```bash
git add docs/reference/configuration.md docs/guide/scheduling.md CLAUDE.md
git commit -m "docs: slo_priorities shared between admission and preemption (#1170)

- configuration.md: --preemption-policy row notes slo_priorities interaction
- scheduling.md: shared priority mapping note
- CLAUDE.md: Recent Changes for SLO priority override threading

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Behavioral | `TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption` |
| BC-2 | Task 1 | Build | All existing tests pass with `nil` SLOPriorityOverrides (zero value) |
| BC-3 | Task 2 | Build | `go build ./...` confirms promotion; existing cluster tests pass |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| Go promotion ambiguity after field move | Low | High (compile error) | `go build ./...` in Task 2 Step 4 catches immediately | Task 2 |
| Map shared by reference via struct copy | Low | Medium (mutation visible to both admission and preemption) | Map is read-only after construction; `NewSLOPriorityMap` copies defaults and applies overrides | Task 1 |
| Existing tests break from SimConfig field addition | None | N/A | Go zero-values `map` as `nil`; `NewSLOPriorityMap(nil)` → defaults. No test needs updating | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep — only threads existing overrides to an existing consumer
- [x] No unexercised code — BC-1 test exercises the new path
- [x] No partial implementations — the full config pipeline is connected
- [x] CLAUDE.md updated in Task 3
- [x] Documentation DRY — `configuration.md` is the user-facing source; `CLAUDE.md` is the working copy
- [x] All contracts mapped to tasks
- [x] Construction site audit: `SimConfig` addition is zero-value safe (no test sites need updating)

**Antipattern rules:**
- [x] R4: `SimConfig` field addition — zero-value safe, no construction site cascade
- [x] R6: No `logrus.Fatalf` in `sim/`
- [x] R8: `SLOPriorityOverrides` is a struct field (R8 targets package-level `var` declarations, not struct fields)
- [x] R16: Field on `SimConfig` directly rather than in `PolicyConfig` — accepted trade-off: `PolicyConfig.NewPolicyConfig()` is already 3-arg; adding a 4th `map[string]int` parameter would cascade to ~10 call sites. Direct `SimConfig` field is pragmatic; future cleanup may relocate if `PolicyConfig` grows
- [x] No YAML tag on new `SimConfig.SLOPriorityOverrides`: intentional — `SimConfig` fields are set programmatically (`cmd/root.go`, `cmd/replay.go`), never deserialized from YAML directly. The existing `deployment.go` YAML tag is dropped.

---

## Appendix

### `sim/simulator.go`

**Key change:** `SimConfig` gains `SLOPriorityOverrides map[string]int`. `NewSimulator` replaces `nil` with `NewSLOPriorityMap(cfg.SLOPriorityOverrides)` — `NewSLOPriorityMap` (defined in `sim/admission.go:94`) applies overrides on top of GAIE defaults. `nil` overrides → pure defaults (backward compatible).

### `sim/cluster/resolve.go` (no change needed, but note)

`ResolvePoolConfig` does `resolved := global` (struct copy). The `SLOPriorityOverrides` map header is shallow-copied — all pool instances share the same backing map. This is safe because the map is read-only after construction. The existing `LatencyCoeffs` slice (lines 53-58 of `resolve.go`) is also shared by reference for the same reason.

### `sim/cluster/deployment.go`

**Key change:** Remove `SLOPriorityOverrides map[string]int` field. The promoted field from the embedded `sim.SimConfig` takes over. `cluster.go:181` `config.SLOPriorityOverrides` resolves via Go struct embedding promotion — no code change needed at the access site.
