# Embed SimConfig in DeploymentConfig — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the field-by-field copy between DeploymentConfig and SimConfig so that adding a new simulator parameter never requires updating a manual field mapping again.

**The problem today:** `DeploymentConfig` duplicates 21 of `SimConfig`'s 23 fields and `ToSimConfig()` copies them one-by-one. This field-by-field copy already had a live bug (#234 — `KVTransferBaseLatency` was missing, causing silent zero-value in cluster mode). Every new parameter added to SimConfig requires touching 5 files: CLI flag + DeploymentConfig + ToSimConfig() + SimConfig + consumption site.

**What this PR adds:**
1. **Structural embedding** — `DeploymentConfig` embeds `sim.SimConfig` directly, eliminating 21 duplicated field declarations
2. **Trivial ToSimConfig()** — the method becomes `return d.SimConfig` (one line, zero drift risk)
3. **Reduced extension friction** — adding a new instance-level parameter drops from 5 touch-points to 3 (CLI + SimConfig + consumer)

**Why this matters:** Every planned feature (PR11 AutoScaler, PR14 P/D Disaggregation, PR15 Adapters) will add config fields. Getting the config structure right now prevents a class of silent zero-value bugs in all future PRs.

**Architecture:** `DeploymentConfig` (in `sim/cluster/`) embeds `sim.SimConfig` as an anonymous field. Cluster-only fields (`NumInstances`, routing pipeline, trace config, snapshot staleness) remain directly on `DeploymentConfig`. Go's field promotion ensures existing code that reads/writes `config.Horizon` continues to work unchanged. The ~37 SimConfig construction sites across test files are completely untouched.

**Source:** GitHub issue #247

**Relates to:** #247 (this PR delivers Phase 1: embedding. Phase 2: sub-config decomposition will be tracked in a follow-up issue.)

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR restructures `DeploymentConfig` to embed `SimConfig` instead of duplicating its fields. The change is purely structural — no new behavior, no new types, no new interfaces. All existing tests pass with only construction-site syntax changes (struct literal nesting).

The refactor touches 17 DeploymentConfig construction sites (1 production in cmd/root.go, 16 in tests) while leaving all ~37 SimConfig construction sites completely unchanged.

Adjacent blocks: `cmd/root.go` (CLI construction), `sim/cluster/cluster.go` (reads config fields), all cluster test files that construct `DeploymentConfig`.

No DEVIATION flags from Phase 0 — the issue proposed full sub-config decomposition but explicitly marked it as a "Proposed Approach" open to refinement.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Embedding Preserves Field Values
- GIVEN a DeploymentConfig with all SimConfig-equivalent fields set to non-zero values
- WHEN ToSimConfig() is called
- THEN the returned SimConfig has identical values for all 21 instance-level fields
- MECHANISM: ToSimConfig() returns d.SimConfig (the embedded struct itself)

BC-2: Promoted Field Access
- GIVEN a DeploymentConfig with an embedded SimConfig
- WHEN accessing instance-level fields (e.g., config.Horizon, config.Seed, config.PriorityPolicy)
- THEN the values are the same as config.SimConfig.Horizon, config.SimConfig.Seed, etc.
- MECHANISM: Go's struct embedding promotes fields for both read and write access

BC-3: Post-Construction Field Assignment
- GIVEN a DeploymentConfig constructed via a helper function
- WHEN a test assigns `config.Scheduler = "sjf"` or `config.Horizon = 500000`
- THEN the embedded SimConfig field is updated and ToSimConfig() reflects the change
- MECHANISM: Go promoted field assignment writes directly to the embedded struct

BC-4: Workload Fields Zero-Valued in Cluster Config
- GIVEN a DeploymentConfig constructed for cluster mode
- WHEN ToSimConfig() is called
- THEN GuideLLMConfig is nil and TracesWorkloadFilePath is ""
- MECHANISM: Embedded SimConfig's workload fields are never set (workload is passed separately to NewClusterSimulator)

**Negative Contracts:**

BC-5: No Cluster-Only Field Leakage
- GIVEN a DeploymentConfig with cluster-only fields set (NumInstances, AdmissionPolicy, TraceLevel, etc.)
- WHEN ToSimConfig() is called
- THEN the returned SimConfig contains NO cluster-only information
- MECHANISM: Cluster-only fields are declared directly on DeploymentConfig, not on SimConfig

BC-6: No Field Shadowing
- GIVEN the DeploymentConfig struct definition
- WHEN inspected via reflection
- THEN no directly-declared DeploymentConfig field shares a name with any SimConfig field
- MECHANISM: Explicit removal of PriorityPolicy and Scheduler from DeploymentConfig (they were duplicates)

**Error Handling Contracts:**

BC-7: All Existing Tests Pass
- GIVEN the full test suite
- WHEN `go test ./...` is run after the refactor
- THEN all tests pass with zero failures
- MECHANISM: Only construction-site syntax changes; no behavioral modifications

### C) Component Interaction

```
cmd/root.go
    │
    ├─ Constructs: cluster.DeploymentConfig{
    │      SimConfig: sim.SimConfig{...},  ← instance-level fields nested here
    │      NumInstances: N,                ← cluster-only fields flat here
    │      AdmissionPolicy: "...",
    │      ...
    │  }
    │
    ▼
cluster.NewClusterSimulator(config, guideLLMConfig, tracesPath)
    │
    ├─ config.ToSimConfig() → returns config.SimConfig (trivial)
    │       │
    │       ▼
    │   sim.NewSimulator(simCfg) → creates per-instance simulators
    │
    ├─ config.NumInstances → cluster orchestration
    ├─ config.AdmissionPolicy → admission factory
    ├─ config.RoutingPolicy → routing factory
    └─ config.TraceLevel → trace configuration
```

**State ownership:** No change. DeploymentConfig is read-only after construction. SimConfig is read-only after NewSimulator consumes it.

**Extension friction after refactor:** Adding a new instance-level parameter requires **3 files**: SimConfig definition + CLI flag in cmd/root.go (SimConfig literal is in the same function) + consumption site. Down from 5.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Decompose into ~6-7 sub-config structs (KVCacheConfig, LatencyConfig, BatchConfig, etc.) | Embed SimConfig in DeploymentConfig without internal sub-configs | SIMPLIFICATION: Embedding alone eliminates the #1 bug class (ToSimConfig drift) and reduces touch-points from 5→3. Sub-config decomposition adds construction-site verbosity across ~37 SimConfig test sites for a narrower benefit (factory signature scoping). Phase 2 follow-up recommended. |
| Issue says "Large — touches every construction site" | Only 17 DeploymentConfig sites touched; ~37 SimConfig sites unchanged | SIMPLIFICATION: Embedding approach has much smaller blast radius than full decomposition. |

### E) Review Guide

**The tricky part:** The `PriorityPolicy` and `Scheduler` fields exist on both SimConfig AND DeploymentConfig today. After embedding, they must be removed from DeploymentConfig to avoid ambiguous field promotion. Verify these are deleted, not duplicated.

**What to scrutinize:** The field mapping test rewrite (Task 5). The old test verified each field individually; the new test should use reflection to guarantee no drift is possible.

**What's safe to skim:** Tasks 3-4 (test construction site updates) are purely mechanical — adding `SimConfig: sim.SimConfig{...}` nesting to struct literals. The field names and values don't change.

**Known debt:** The sub-factories `NewLatencyModel(cfg SimConfig)` and `NewKVStore(cfg SimConfig)` still receive the full SimConfig rather than scoped sub-configs. Filed as a follow-up to #247 (Phase 2).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/deployment.go` — embed SimConfig, remove 21 duplicate fields + 2 shadowed fields, simplify ToSimConfig()
- `cmd/root.go` — restructure DeploymentConfig literal to use nested SimConfig
- `sim/cluster/cluster_test.go` — update helper + 4 inline literals + rewrite field mapping tests
- `sim/cluster/pending_requests_test.go` — update 4 inline literals
- `sim/cluster/cluster_trace_test.go` — update 4 inline literals
- `sim/cluster/evaluation_test.go` — update 1 inline literal
- `sim/cluster/prefix_routing_test.go` — update helper
- `sim/cluster/workload_test.go` — update 1 inline literal
- `CLAUDE.md` — update extension friction count and DeploymentConfig description

**Key decisions:**
- Embed approach (not full decomposition) — rationale in Deviation Log
- PriorityPolicy/Scheduler removed from DeploymentConfig (conflict resolution)
- Reflection-based shadowing test added for structural safety

**Confirmation:** No dead code. All paths exercisable after each task.

### G) Task Breakdown

---

#### Task 1: Restructure DeploymentConfig to embed SimConfig

**Contracts Implemented:** BC-1, BC-5, BC-6

**Files:**
- Modify: `sim/cluster/deployment.go`

**Step 1: Modify DeploymentConfig struct**

Context: Remove the 21 SimConfig-equivalent fields (including PriorityPolicy and Scheduler which are duplicates) and replace with an embedded `sim.SimConfig`.

In `sim/cluster/deployment.go`, replace the entire struct and ToSimConfig with:

```go
// DeploymentConfig describes a cluster where all instances share identical
// hardware and model configuration. NumInstances must be >= 1.
type DeploymentConfig struct {
	sim.SimConfig // Embeds all instance-level config (horizon, seed, KV, batch, latency, policy)

	NumInstances int

	// Online routing pipeline configuration (PR4+)
	AdmissionPolicy       string  // "always-admit" (default) or "token-bucket"
	AdmissionLatency      int64   // microseconds, default 0
	RoutingLatency        int64   // microseconds, default 0
	TokenBucketCapacity   float64 // max tokens, default 10000
	TokenBucketRefillRate float64 // tokens/second, default 1000

	// Routing policy configuration (PR6, evolved in PR17)
	RoutingPolicy        string             // "round-robin" (default), "least-loaded", "weighted", "prefix-affinity"
	RoutingScorerConfigs []sim.ScorerConfig // for weighted routing scorer pipeline (nil = use defaults)

	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // number of counterfactual candidates, default 0

	// Snapshot staleness configuration (H3 experiment)
	SnapshotRefreshInterval int64 // microseconds, 0 = Immediate (default)
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// GuideLLMConfig and TracesWorkloadFilePath are accessible but should be
// zero-valued: cluster mode generates workload centrally and injects
// requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}
```

**Step 2: Verify compilation fails (expected)**

Run: `go build ./sim/cluster/...`
Expected: FAIL — construction sites still use flat field syntax.

**Step 3: Commit structural change**

```bash
git add sim/cluster/deployment.go
git commit -m "refactor(cluster): embed SimConfig in DeploymentConfig (BC-1, BC-5, BC-6)

- Replace 21 duplicated fields with sim.SimConfig embedding
- Remove PriorityPolicy/Scheduler duplicate declarations (BC-6)
- Simplify ToSimConfig() to return d.SimConfig (BC-1)
- Cluster-only fields remain directly on DeploymentConfig (BC-5)

Note: compilation broken until construction sites updated (Tasks 2-4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Update production construction site (cmd/root.go)

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `cmd/root.go:412-445`

**Step 1: Restructure DeploymentConfig literal**

Context: The single production construction site in cmd/root.go needs to nest instance-level fields inside `SimConfig:`.

In `cmd/root.go`, replace lines 412-445 with:

```go
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon:                   simulationHorizon,
				Seed:                      seed,
				TotalKVBlocks:             totalKVBlocks,
				BlockSizeTokens:           blockSizeTokens,
				MaxRunningReqs:            maxRunningReqs,
				MaxScheduledTokens:        maxScheduledTokens,
				LongPrefillTokenThreshold: longPrefillTokenThreshold,
				BetaCoeffs:                betaCoeffs,
				AlphaCoeffs:               alphaCoeffs,
				ModelConfig:               modelConfig,
				HWConfig:                  hwConfig,
				Model:                     model,
				GPU:                       gpu,
				TP:                        tensorParallelism,
				Roofline:                  roofline,
				PriorityPolicy:            priorityPolicy,
				Scheduler:                 scheduler,
				KVCPUBlocks:               kvCPUBlocks,
				KVOffloadThreshold:        kvOffloadThreshold,
				KVTransferBandwidth:       kvTransferBandwidth,
				KVTransferBaseLatency:     kvTransferBaseLatency,
			},
			NumInstances:            numInstances,
			AdmissionPolicy:         admissionPolicy,
			AdmissionLatency:        admissionLatency,
			RoutingLatency:          routingLatency,
			TokenBucketCapacity:     tokenBucketCapacity,
			TokenBucketRefillRate:   tokenBucketRefillRate,
			RoutingPolicy:           routingPolicy,
			RoutingScorerConfigs:    parsedScorerConfigs,
			TraceLevel:              traceLevel,
			CounterfactualK:         counterfactualK,
			SnapshotRefreshInterval: snapshotRefreshInterval,
		}
```

**Step 2: Verify build passes**

Run: `go build ./...`
Expected: PASS (production code compiles; tests still fail)

**Step 3: Commit**

```bash
git add cmd/root.go
git commit -m "refactor(cmd): update DeploymentConfig construction for SimConfig embedding

- Nest instance-level fields inside SimConfig: sim.SimConfig{...}
- Cluster-only fields remain at top level

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Update test helper functions

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Modify: `sim/cluster/cluster_test.go` (newTestDeploymentConfig helper, ~line 15)
- Modify: `sim/cluster/prefix_routing_test.go` (baseDeploymentConfig helper, ~line 58)

**Step 1: Update newTestDeploymentConfig()**

In `sim/cluster/cluster_test.go`, replace the helper:

```go
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:            math.MaxInt64,
			Seed:               42,
			TotalKVBlocks:      10000,
			BlockSizeTokens:    16,
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
			BetaCoeffs:         []float64{1000, 10, 5},
			AlphaCoeffs:        []float64{100, 1, 100},
			Model:              "test-model",
			GPU:                "H100",
			TP:                 1,
		},
		NumInstances: numInstances,
	}
}
```

**Step 2: Update baseDeploymentConfig()**

In `sim/cluster/prefix_routing_test.go`, replace the helper:

```go
func baseDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:            50000000,
			Seed:               42,
			TotalKVBlocks:      2000,
			BlockSizeTokens:    16,
			MaxRunningReqs:     64,
			MaxScheduledTokens: 65536,
			BetaCoeffs:         []float64{1000, 10, 5},
			AlphaCoeffs:        []float64{100, 50, 25},
			Model:              "test-model",
		},
		NumInstances: numInstances,
		TraceLevel:   "decisions",
	}
}
```

**Step 3: Run tests that use these helpers**

Run: `go test ./sim/cluster/... -run "TestCluster|TestPrefix" -v -count=1 2>&1 | tail -20`
Expected: Some tests pass (those using helpers); others still fail (inline construction)

**Step 4: Commit**

```bash
git add sim/cluster/cluster_test.go sim/cluster/prefix_routing_test.go
git commit -m "refactor(cluster/test): update test helper construction for SimConfig embedding (BC-2, BC-3)

- newTestDeploymentConfig: nest instance fields in SimConfig
- baseDeploymentConfig: nest instance fields in SimConfig
- Post-construction field assignments (config.Horizon = ...) unchanged (promotion)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Update all inline test construction sites

**Contracts Implemented:** BC-2, BC-7

**Files:**
- Modify: `sim/cluster/cluster_test.go` (~lines 76, 148, 185, 863)
- Modify: `sim/cluster/pending_requests_test.go` (~lines 14, 62, 140, 214)
- Modify: `sim/cluster/cluster_trace_test.go` (~lines 12, 44, 91, 135)
- Modify: `sim/cluster/evaluation_test.go` (~line 13)
- Modify: `sim/cluster/workload_test.go` (~line 16)

**Step 1: Update all inline DeploymentConfig{} literals**

Context: Each inline literal needs instance-level fields moved into a nested `SimConfig: sim.SimConfig{...}` block. Cluster-only fields (AdmissionPolicy, RoutingPolicy, TraceLevel, CounterfactualK, etc.) stay at the top level. PriorityPolicy and Scheduler move INTO the SimConfig block.

The pattern for every construction site is the same mechanical transformation:

**Before:**
```go
config := DeploymentConfig{
    NumInstances: 2, Horizon: 10000000, Seed: 42,
    TotalKVBlocks: 100, BlockSizeTokens: 16,
    MaxRunningReqs: 10, MaxScheduledTokens: 2048,
    BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
    TraceLevel: "decisions", CounterfactualK: 2,
}
```

**After:**
```go
config := DeploymentConfig{
    SimConfig: sim.SimConfig{
        Horizon: 10000000, Seed: 42,
        TotalKVBlocks: 100, BlockSizeTokens: 16,
        MaxRunningReqs: 10, MaxScheduledTokens: 2048,
        BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
    },
    NumInstances: 2,
    TraceLevel: "decisions", CounterfactualK: 2,
}
```

Apply this transformation to all 14 inline construction sites across the 5 test files listed above.

**Instance-level fields** (move into SimConfig block): Horizon, Seed, TotalKVBlocks, BlockSizeTokens, MaxRunningReqs, MaxScheduledTokens, LongPrefillTokenThreshold, BetaCoeffs, AlphaCoeffs, ModelConfig, HWConfig, Model, GPU, TP, Roofline, PriorityPolicy, Scheduler, KVCPUBlocks, KVOffloadThreshold, KVTransferBandwidth, KVTransferBaseLatency

**Cluster-only fields** (keep at top level): NumInstances, AdmissionPolicy, AdmissionLatency, RoutingLatency, TokenBucketCapacity, TokenBucketRefillRate, RoutingPolicy, RoutingScorerConfigs, TraceLevel, CounterfactualK, SnapshotRefreshInterval

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/cluster/cluster_test.go sim/cluster/pending_requests_test.go sim/cluster/cluster_trace_test.go sim/cluster/evaluation_test.go sim/cluster/workload_test.go
git commit -m "refactor(cluster/test): update inline DeploymentConfig literals for embedding (BC-2, BC-7)

- 14 inline construction sites across 5 test files
- Instance-level fields nested in SimConfig block
- Cluster-only fields remain at top level
- All tests pass — pure mechanical transformation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Rewrite and strengthen field mapping tests

**Contracts Implemented:** BC-1, BC-4, BC-6

**Files:**
- Modify: `sim/cluster/cluster_test.go` (TestDeploymentConfig_ToSimConfig_FieldMapping, TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields)

**Step 1: Rewrite field mapping tests**

Context: The old tests verified each field individually (structural). Replace with behavioral tests that verify embedding identity and the no-shadowing invariant.

Replace `TestDeploymentConfig_ToSimConfig_FieldMapping` with:

```go
// TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig verifies that
// ToSimConfig() returns exactly the embedded SimConfig (BC-1).
func TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig(t *testing.T) {
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: 999, Seed: 7, TotalKVBlocks: 500, BlockSizeTokens: 32,
			MaxRunningReqs: 128, MaxScheduledTokens: 4096,
			LongPrefillTokenThreshold: 512,
			BetaCoeffs: []float64{1, 2, 3}, AlphaCoeffs: []float64{4, 5, 6},
			Model: "test-model", GPU: "H100", TP: 2, Roofline: true,
			PriorityPolicy: "slo-based", Scheduler: "priority-fcfs",
			KVTransferBaseLatency: 42,
		},
		NumInstances:    3,
		AdmissionPolicy: "token-bucket",
		TraceLevel:      "decisions",
	}

	sc := dc.ToSimConfig()

	// BC-1: ToSimConfig returns exactly the embedded SimConfig
	// Note: SimConfig contains slices (BetaCoeffs, AlphaCoeffs) so direct
	// == comparison won't compile. Use reflect.DeepEqual instead.
	if !reflect.DeepEqual(sc, dc.SimConfig) {
		t.Errorf("ToSimConfig() returned a different SimConfig than the embedded one")
	}

	// BC-4: Workload fields zero-valued (cluster generates workload centrally)
	if sc.GuideLLMConfig != nil {
		t.Error("GuideLLMConfig should be nil (workload generated centrally)")
	}
	if sc.TracesWorkloadFilePath != "" {
		t.Error("TracesWorkloadFilePath should be empty (workload generated centrally)")
	}
}
```

Replace `TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields` with:

```go
// TestDeploymentConfig_NoFieldShadowing verifies that no directly-declared
// DeploymentConfig field shares a name with any SimConfig field (BC-6).
func TestDeploymentConfig_NoFieldShadowing(t *testing.T) {
	dcType := reflect.TypeOf(DeploymentConfig{})
	scType := reflect.TypeOf(sim.SimConfig{})

	// Build set of SimConfig field names
	simFields := make(map[string]bool)
	for i := 0; i < scType.NumField(); i++ {
		simFields[scType.Field(i).Name] = true
	}

	// Check each directly-declared DeploymentConfig field (skip embedded SimConfig)
	for i := 0; i < dcType.NumField(); i++ {
		field := dcType.Field(i)
		if field.Anonymous {
			continue // skip the embedded SimConfig itself
		}
		if simFields[field.Name] {
			t.Errorf("DeploymentConfig field %q shadows SimConfig field — use promoted access instead", field.Name)
		}
	}
}
```

Add `"reflect"` to imports.

**Step 2: Run the new tests**

Run: `go test ./sim/cluster/... -run "TestDeploymentConfig_ToSimConfig|TestDeploymentConfig_NoFieldShadowing" -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 4: Lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): add embedding identity and no-shadowing tests (BC-1, BC-4, BC-6)

- TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig: verify identity
- TestDeploymentConfig_NoFieldShadowing: reflection-based structural safety
- Replace old field-by-field mapping tests (no longer needed with embedding)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: Update documentation

**Contracts Implemented:** (documentation only)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Update the DeploymentConfig description in the Code Architecture section. Key changes:
- `deployment.go` description: "DeploymentConfig embeds sim.SimConfig + cluster-only fields; ToSimConfig() returns the embedded config"
- Remove "duplicates 24 of these fields" language
- Update extension friction: "Adding one new parameter requires touching: CLI flag (cmd/root.go) + SimConfig + consumption site = **3 files**"

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for SimConfig embedding (#247)

- DeploymentConfig now embeds SimConfig (no field duplication)
- Extension friction reduced from 5 to 3 touch-points
- ToSimConfig() is now trivial (returns embedded config)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 5 | Unit | TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig |
| BC-2 | Task 4 | Existing | All existing tests pass unchanged (construction-site-only changes) |
| BC-3 | Task 3 | Existing | Tests using helpers with post-construction `config.X = ...` |
| BC-4 | Task 5 | Unit | TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig (workload check) |
| BC-5 | Task 5 | Unit | TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig (cluster fields not in SC) |
| BC-6 | Task 5 | Unit | TestDeploymentConfig_NoFieldShadowing |
| BC-7 | Task 4 | Suite | `go test ./... -count=1` — all tests pass |

**Golden dataset:** No changes. Output format and metrics are unaffected by this structural refactor.

**Shared test infrastructure:** No new helpers needed. Existing `sim/internal/testutil/` unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed construction site → compilation failure | Low | Low | Comprehensive grep audit (done in exploration) | Tasks 3-4 |
| PriorityPolicy/Scheduler shadowing after embedding | Medium | High | Explicit removal in Task 1 + reflection test in Task 5 | Tasks 1, 5 |
| Future developer re-adds duplicate field | Low | Medium | Reflection-based TestDeploymentConfig_NoFieldShadowing catches at test time | Task 5 |
| Post-construction assignments break | Very Low | Medium | Go promotes embedded fields for assignment — verified in exploration | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (embedding only, no new types)
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (existing newTestDeploymentConfig updated, not duplicated)
- [x] CLAUDE.md updated (Task 6)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — one deviation justified (embed-only vs full decomposition)
- [x] Each task produces working, testable code (Tasks 1-2 are compilation-progressive)
- [x] Task dependencies correctly ordered (1→2→3→4→5→6)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: not needed (no output changes)
- [x] Construction site audit: 17 DeploymentConfig sites enumerated and covered

**Antipattern rules:**
- [x] R1: No error paths affected
- [x] R2: No map iteration affected
- [x] R3: No new CLI flags
- [x] R4: Construction site audit complete (17 sites)
- [x] R5: No resource allocation loops
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: Reflection-based invariant test added
- [x] R8: No exported mutable maps
- [x] R9: No new YAML fields
- [x] R10: No YAML parsing changes
- [x] R11: No division changes
- [x] R12: No golden dataset changes
- [x] R13: N/A — no new interfaces
- [x] R14: N/A — no methods modified
- [x] R15: No stale PR references
- [x] R16: Config grouping improved (embedding is module-scoped)
- [x] R17: No routing scorer changes

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/deployment.go`

**Purpose:** Define DeploymentConfig with embedded SimConfig, trivial ToSimConfig()

**Changes:** Replace entire struct definition (32→12 directly-declared fields). Replace ToSimConfig() body (21 lines → 1 line).

**Key note:** PriorityPolicy and Scheduler are REMOVED from DeploymentConfig (they exist on SimConfig and are promoted via embedding).

### File: `cmd/root.go`

**Purpose:** CLI entry point — single production DeploymentConfig construction site

**Changes:** Lines 412-445 — wrap 21 instance-level fields in `SimConfig: sim.SimConfig{...}` block.

### File: `sim/cluster/cluster_test.go`

**Purpose:** Test helpers + field mapping tests + inline construction sites

**Changes:**
- `newTestDeploymentConfig()` (~line 15): nest fields in SimConfig
- `TestDeploymentConfig_ToSimConfig_FieldMapping` (~line 76): rewrite as embedding identity test
- `TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields` (~line 148): replace with NoFieldShadowing test
- Inline literals (~lines 185, 863): nest fields in SimConfig

### File: `sim/cluster/pending_requests_test.go`

**Purpose:** 4 inline DeploymentConfig construction sites

**Changes:** Lines 14, 62, 140, 214 — mechanical nesting transformation.

### File: `sim/cluster/cluster_trace_test.go`

**Purpose:** 4 inline DeploymentConfig construction sites

**Changes:** Lines 12, 44, 91, 135 — mechanical nesting transformation.

### File: `sim/cluster/evaluation_test.go`

**Purpose:** 1 inline DeploymentConfig construction site

**Changes:** Line 13 — mechanical nesting transformation.

### File: `sim/cluster/prefix_routing_test.go`

**Purpose:** Test helper baseDeploymentConfig()

**Changes:** Line 59 — nest fields in SimConfig.

### File: `sim/cluster/workload_test.go`

**Purpose:** 1 partial DeploymentConfig construction site

**Changes:** Line 16 — `DeploymentConfig{Horizon: 1000}` → `DeploymentConfig{SimConfig: sim.SimConfig{Horizon: 1000}}`
