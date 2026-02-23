# SimConfig Decomposition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the monolithic 23-field `SimConfig` into focused sub-configs grouped by module concern, so that factories express their actual dependencies and future field additions touch fewer files.

**The problem today:** `SimConfig` mixes 8 unrelated module concerns (simulation params, KV cache, batch limits, latency coefficients, hardware/model identity, policies, workload) in a single 23-field struct. Every factory (`NewKVStore`, `NewLatencyModel`, `NewBatchFormation`) receives all 23 fields when it only uses 2–6. Adding a new field requires manually auditing 61 construction sites across 17 files — a violation of R4 (canonical constructors) and R16 (config grouped by module).

**What this PR adds:**
1. **Module-scoped sub-configs** — 6 focused config structs (`KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig`) each grouping related fields
2. **Composed SimConfig** — `SimConfig` becomes a composition of embedded sub-configs, preserving all field access via Go's promoted fields
3. **Narrowed factory signatures** — `NewKVStore(KVCacheConfig)`, `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`, `NewBatchFormation(LatencyModel)` (drops unused `SimConfig` param)
4. **Consolidated test helpers** — `newTestSimConfig()` and `newTestDeploymentConfig()` updated once; inline literal sites updated to use embedded sub-config syntax

**Why this matters:** Every future PR that adds a config parameter (AutoScaler PR11, P/D disaggregation PR14) drops from 5–6 touch points to 2–3, and factory signatures document their actual dependencies.

**Architecture:** All sub-config types defined in `sim/config.go` (new file). `SimConfig` in `sim/simulator.go` recomposed via embedding. Factory signatures in `sim/kv_store.go`, `sim/latency_model.go`, `sim/batch_formation.go` narrowed. `DeploymentConfig` in `sim/cluster/deployment.go` inherits changes automatically via its existing `sim.SimConfig` embedding.

**Source:** GitHub issue #350

**Closes:** Fixes #350

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR decomposes `SimConfig` from a flat 23-field struct into a composition of 6 embedded sub-config structs grouped by module concern. The refactoring is purely structural — zero behavioral change, zero output change. All existing tests pass without modification to assertions (only construction-site syntax changes).

The PR sits at the foundation of the config layer. It is consumed by `NewSimulator`, `NewKVStore`, `NewLatencyModel`, `NewBatchFormation`, `NewInstanceSimulator`, and `NewClusterSimulator`. `DeploymentConfig` inherits changes automatically because it embeds `SimConfig`.

No deviations from issue #350 except: issue proposed 8 named sub-configs. This plan implements 6 sub-config types by (a) merging `SimulationParams` (Horizon, Seed) directly into SimConfig as non-embedded fields, since no factory takes only simulation params, (b) renaming `LatencyModelConfig` to `LatencyCoeffs`, and (c) merging `HardwareConfig` + `ModelParams` into a single `ModelHardwareConfig` since the latency model factory needs fields from both.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Field access preservation
- GIVEN any code accessing `cfg.TotalKVBlocks` (or any other SimConfig field) via promoted field syntax
- WHEN SimConfig is decomposed into embedded sub-configs
- THEN all existing field access expressions compile and return the same value
- MECHANISM: Go promotes embedded struct fields to the parent; `cfg.TotalKVBlocks` resolves to `cfg.KVCacheConfig.TotalKVBlocks`

BC-2: Factory signature narrowing — KVStore
- GIVEN `NewKVStore` currently accepts `SimConfig` but only uses 6 KV-related fields
- WHEN its signature changes to `NewKVStore(KVCacheConfig)`
- THEN it produces identical KVStore instances for the same field values
- MECHANISM: Factory body unchanged; only the parameter type narrows

BC-3: Factory signature narrowing — LatencyModel
- GIVEN `NewLatencyModel` currently accepts `SimConfig` but only uses coefficients + hardware fields
- WHEN its signature changes to `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`
- THEN it produces identical LatencyModel instances for the same field values
- MECHANISM: Factory body unchanged; parameter type splits into two sub-configs

BC-4: Factory cleanup — BatchFormation
- GIVEN `NewBatchFormation(_ SimConfig, latencyModel LatencyModel)` ignores its SimConfig parameter
- WHEN its signature changes to `NewBatchFormation(latencyModel LatencyModel)`
- THEN it produces identical BatchFormation instances
- MECHANISM: Unused parameter removed

BC-5: DeploymentConfig embedding preservation
- GIVEN `DeploymentConfig` embeds `sim.SimConfig`
- WHEN SimConfig is decomposed
- THEN `DeploymentConfig` promotes all sub-config fields (e.g., `config.TotalKVBlocks` still works)
- MECHANISM: Go's embedding promotion is transitive — DeploymentConfig embeds SimConfig, which embeds KVCacheConfig, so fields promote through two levels

BC-6: Golden dataset equivalence
- GIVEN all golden dataset test cases
- WHEN run with the decomposed SimConfig
- THEN output matches golden values exactly (byte-identical)
- MECHANISM: Pure refactoring — no computational logic changes

BC-7: No field shadowing in DeploymentConfig
- GIVEN the existing `TestDeploymentConfig_NoFieldShadowing` test
- WHEN SimConfig is decomposed into embedded sub-configs
- THEN the test continues to pass (no sub-config field names collide with DeploymentConfig field names)
- MECHANISM: Sub-config field names are disjoint from DeploymentConfig field names (verified in plan Phase 0)

**Negative Contracts:**

BC-8: No exported mutable maps
- GIVEN this PR introduces new types
- WHEN any type has map-typed fields
- THEN they MUST NOT be exported (R8)
- MECHANISM: No map-typed fields in any sub-config

BC-9: No behavioral change
- GIVEN any simulation run with any configuration
- WHEN run before vs. after this PR
- THEN stdout output MUST be byte-identical
- MECHANISM: This is a pure type restructuring; no computation logic is modified

### C) Component Interaction

```
cmd/root.go
  └── constructs DeploymentConfig { SimConfig { sub-configs... } }
        │
        ├── sim/cluster/cluster.go: NewClusterSimulator(DeploymentConfig, ...)
        │     └── calls config.ToSimConfig() → passes to NewInstanceSimulator
        │
        └── sim/simulator.go: NewSimulator(SimConfig)
              ├── NewKVStore(cfg.KVCacheConfig)          ← narrowed
              ├── NewLatencyModel(cfg.LatencyCoeffs,     ← narrowed
              │                   cfg.ModelHardwareConfig)
              └── NewBatchFormation(latencyModel)        ← dropped cfg param
```

**State changes:** None. Sub-configs are pure value types (no pointers, no maps, no channels). No new mutable state.

**Extension friction:** After this PR, adding a new KV config field touches 2 files: `sim/config.go` (type definition) + `cmd/root.go` (CLI flag wiring). Test files only need updating if they override that specific field — most go through `newTestSimConfig()`.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "~6-7 focused sub-configs" | 6 sub-config types (KVCacheConfig, BatchConfig, LatencyCoeffs, ModelHardwareConfig, PolicyConfig, WorkloadConfig) | ADDITION: WorkloadConfig was missing from original issue; added in our update |
| Separate `SimulationParams` type | Horizon and Seed remain top-level SimConfig fields | SIMPLIFICATION: No factory takes only (Horizon, Seed) — a sub-config type adds indirection with no narrowing benefit |
| `LatencyModelConfig` (AlphaCoeffs, BetaCoeffs) | `LatencyCoeffs` (same fields) | SIMPLIFICATION: Shorter name; "Coeffs" is more precise than "Config" for two coefficient slices |
| `HardwareConfig` + `ModelParams` (2 separate types) | `ModelHardwareConfig` (1 merged type) | SIMPLIFICATION: Latency model factory needs fields from both; merging avoids a 3-param factory signature |
| "~37 construction sites" | 61 actual sites in 17 Go files | CORRECTION: Updated in issue #350 |

### E) Review Guide

**The tricky part:** The `TestDeploymentConfig_NoFieldShadowing` test uses `reflect.TypeOf(sim.SimConfig{})` and iterates top-level fields. After embedding, `SimConfig{}`'s direct fields are the embedded sub-config types + Horizon + Seed — NOT the promoted fields. The test logic (checking `Anonymous` flag to skip embedded types) should still work, but verify this carefully.

**What to scrutinize:** BC-5 (transitive promotion through DeploymentConfig). Go promotes fields through one level of embedding. SimConfig embeds KVCacheConfig; DeploymentConfig embeds SimConfig. Does `deploymentConfig.TotalKVBlocks` still work? Yes — Go promotes through all embedding levels.

**What's safe to skim:** The 60 test-file construction site updates. These are mechanical syntax changes (`SimConfig{TotalKVBlocks: 100}` → `SimConfig{KVCacheConfig: KVCacheConfig{TotalKVBlocks: 100}}`). The compiler catches any mistakes.

**Known debt:** The `reflect.TypeOf` field-shadowing test will need updating to handle nested embedded structs (it currently only checks one level). Addressed in Task 6.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/config.go` — 6 sub-config type definitions

**Files to modify:**
- `sim/simulator.go` — Recompose SimConfig via embedding
- `sim/kv_store.go` — Narrow `NewKVStore` signature
- `sim/latency_model.go` — Narrow `NewLatencyModel` signature
- `sim/batch_formation.go` — Drop unused `SimConfig` param from `NewBatchFormation`
- `sim/simulator.go` — Update `NewSimulator` to pass sub-configs to narrowed factories
- `cmd/root.go` — Update construction site
- `sim/simulator_test.go` — Update `newTestSimConfig` + inline sites
- `sim/batch_formation_test.go` — Update 7 inline sites
- `sim/kv_store_test.go` — Update 5 inline sites
- `sim/latency_model_test.go` — Update 5 inline sites
- `sim/scheduler_test.go` — Update 4 inline sites
- `sim/simulator_preempt_test.go` — Update 2 inline sites
- `sim/simulator_decode_test.go` — Update 1 inline site
- `sim/model_hardware_config_test.go` — Update 3 inline sites
- `sim/cluster/deployment.go` — No changes needed (embedding transitive)
- `sim/cluster/cluster_test.go` — Update helper + 5 inline sites + field-shadowing test
- `sim/cluster/instance_test.go` — Update 2 helpers + 2 inline sites
- `sim/cluster/pending_requests_test.go` — Update 4 inline sites
- `sim/cluster/cluster_trace_test.go` — Update 4 inline sites
- `sim/cluster/evaluation_test.go` — Update 1 inline site
- `sim/cluster/prefix_routing_test.go` — Update 1 inline site
- `sim/cluster/snapshot_test.go` — Update 1 inline site
- `sim/cluster/workload_test.go` — Update 1 inline site
- `CLAUDE.md` — Update SimConfig description

**Key decisions:**
- Embedding (not named fields) for sub-configs — preserves field access syntax
- Horizon and Seed stay as top-level SimConfig fields — no factory uses only these two
- Sub-config names: `KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig` — avoid collision with existing type `ModelConfig`

### G) Task Breakdown

---

### Task 1: Define sub-config types and recompose SimConfig

**Contracts Implemented:** BC-1, BC-5, BC-7, BC-8

**Files:**
- Create: `sim/config.go`
- Modify: `sim/simulator.go:96-123`

**Step 1: Create `sim/config.go` with sub-config type definitions**

Context: We define 6 sub-config types (KVCacheConfig, BatchConfig, LatencyCoeffs, ModelHardwareConfig, PolicyConfig, WorkloadConfig). Horizon and Seed remain top-level in SimConfig since no factory uses only simulation params.

In `sim/config.go`:
```go
package sim

// KVCacheConfig groups KV cache parameters for NewKVStore.
type KVCacheConfig struct {
	TotalKVBlocks         int64   // GPU tier capacity in blocks (must be > 0)
	BlockSizeTokens       int64   // tokens per block (must be > 0)
	KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
	KVOffloadThreshold    float64 // GPU utilization threshold for offload (default 0.9)
	KVTransferBandwidth   float64 // blocks/tick transfer rate (default 100.0)
	KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
}

// BatchConfig groups batch formation parameters.
type BatchConfig struct {
	MaxRunningReqs            int64 // max requests in RunningBatch
	MaxScheduledTokens        int64 // max total new tokens across all requests in RunningBatch
	LongPrefillTokenThreshold int64 // threshold for long prefill chunking
}

// LatencyCoeffs groups regression coefficients for the latency model.
type LatencyCoeffs struct {
	BetaCoeffs  []float64 // regression coefficients for step time (≥3 elements required)
	AlphaCoeffs []float64 // regression coefficients for queueing time (≥3 elements required)
}

// ModelHardwareConfig groups model identity and hardware specification.
type ModelHardwareConfig struct {
	ModelConfig ModelConfig   // HuggingFace model parameters (for roofline mode)
	HWConfig    HardwareCalib // GPU specifications (for roofline mode)
	Model       string        // model name (e.g., "meta-llama/llama-3.1-8b-instruct")
	GPU         string        // GPU type (e.g., "H100")
	TP          int           // tensor parallelism degree
	Roofline    bool          // true = analytical roofline mode, false = blackbox regression
}

// PolicyConfig groups scheduling and priority policy selection.
type PolicyConfig struct {
	PriorityPolicy string // "constant" (default) or "slo-based"
	Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
}

// WorkloadConfig groups workload generation parameters.
// Both fields zero-valued means no workload generation (caller injects via InjectArrival).
type WorkloadConfig struct {
	GuideLLMConfig         *GuideLLMConfig // distribution-based workload (optional)
	TracesWorkloadFilePath string          // CSV trace file path (optional)
}
```

**Step 2: Recompose SimConfig in `sim/simulator.go`**

Replace the flat SimConfig definition (lines 96-123) with embedded composition:

```go
// SimConfig holds all configuration for creating a Simulator.
// Sub-configs are embedded so fields are accessible via promotion
// (e.g., cfg.TotalKVBlocks resolves to cfg.KVCacheConfig.TotalKVBlocks).
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
}
```

**Step 3: Verify build compiles (expect failures in construction sites only)**

Run: `go build ./sim/... 2>&1 | head -5`
Expected: Compilation errors in test files (construction sites using flat syntax). This is expected — we fix them in Tasks 2-4.

**Step 4: Commit type definitions**

```bash
git add sim/config.go sim/simulator.go
git commit -m "refactor(sim): define module-scoped sub-configs and recompose SimConfig (BC-1)

- Create sim/config.go with 6 sub-config types: KVCacheConfig, BatchConfig,
  LatencyCoeffs, ModelHardwareConfig, PolicyConfig, WorkloadConfig
- Recompose SimConfig via embedding (R16: config grouped by module)
- Horizon and Seed remain top-level (no factory uses only these)
- NOTE: Compilation broken in test files until construction sites updated (Tasks 2-4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Update sim/ package test construction sites

**Contracts Implemented:** BC-1, BC-6, BC-9

**Files:**
- Modify: `sim/simulator_test.go`
- Modify: `sim/batch_formation_test.go`
- Modify: `sim/kv_store_test.go`
- Modify: `sim/latency_model_test.go`
- Modify: `sim/scheduler_test.go`
- Modify: `sim/simulator_preempt_test.go`
- Modify: `sim/simulator_decode_test.go`
- Modify: `sim/model_hardware_config_test.go`

**Step 1: Update the `newTestSimConfig()` helper in `sim/simulator_test.go`**

Context: This helper is the canonical test config. Update it to use embedded sub-config syntax. All tests that call this helper get the fix for free.

Replace the helper (lines 232-247):
```go
func newTestSimConfig() SimConfig {
	return SimConfig{
		Horizon: math.MaxInt64,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   10000,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{1000, 10, 5},
			AlphaCoeffs: []float64{100, 1, 100},
		},
		ModelHardwareConfig: ModelHardwareConfig{
			Model: "test-model",
			GPU:   "H100",
			TP:    1,
		},
	}
}
```

**Step 2: Update all inline `SimConfig{}` sites in sim/ test files**

Apply the same pattern to every `SimConfig{` literal in the test files listed above. Each site groups fields into their sub-config. For example, `kv_store_test.go` sites that only set KV fields:

```go
// Before:
NewKVStore(SimConfig{TotalKVBlocks: 0, BlockSizeTokens: 16})
// After:
NewKVStore(SimConfig{KVCacheConfig: KVCacheConfig{TotalKVBlocks: 0, BlockSizeTokens: 16}})
```

For sites that set fields across multiple sub-configs (e.g., golden dataset tests setting KV + batch + latency + model fields), group all fields into their respective sub-config structs.

**Step 3: Verify sim/ package compiles and tests pass**

Run: `go test ./sim/... -count=1 -v 2>&1 | tail -5`
Expected: All tests PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/*_test.go
git commit -m "refactor(sim): update sim/ test construction sites for sub-config embedding (BC-1, BC-6)

- Update newTestSimConfig() helper with embedded sub-config syntax
- Update all 38 inline SimConfig{} sites in sim/ test files
- All assertions unchanged — pure syntax migration

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update sim/cluster/ test construction sites

**Contracts Implemented:** BC-1, BC-5, BC-6, BC-9

**Files:**
- Modify: `sim/cluster/cluster_test.go`
- Modify: `sim/cluster/instance_test.go`
- Modify: `sim/cluster/pending_requests_test.go`
- Modify: `sim/cluster/cluster_trace_test.go`
- Modify: `sim/cluster/evaluation_test.go`
- Modify: `sim/cluster/prefix_routing_test.go`
- Modify: `sim/cluster/snapshot_test.go`
- Modify: `sim/cluster/workload_test.go`

**Step 1: Update test helpers in `sim/cluster/`**

In `sim/cluster/cluster_test.go`, update `newTestDeploymentConfig()`:
```go
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: math.MaxInt64,
			Seed:    42,
			KVCacheConfig: sim.KVCacheConfig{
				TotalKVBlocks:   10000,
				BlockSizeTokens: 16,
			},
			BatchConfig: sim.BatchConfig{
				MaxRunningReqs:     256,
				MaxScheduledTokens: 2048,
			},
			LatencyCoeffs: sim.LatencyCoeffs{
				BetaCoeffs:  []float64{1000, 10, 5},
				AlphaCoeffs: []float64{100, 1, 100},
			},
			ModelHardwareConfig: sim.ModelHardwareConfig{
				Model: "test-model",
				GPU:   "H100",
				TP:    1,
			},
		},
		NumInstances: numInstances,
	}
}
```

In `sim/cluster/instance_test.go`, update `newTestSimConfigWithWorkload()` and `newTestInstanceSimConfig()` similarly.

**Step 2: Update all inline `sim.SimConfig{}` sites in sim/cluster/ test files**

Same pattern as Task 2 — group fields into sub-config structs.

**Step 3: Verify cluster package compiles and tests pass**

Run: `go test ./sim/cluster/... -count=1 -v 2>&1 | tail -5`
Expected: All tests PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/*_test.go
git commit -m "refactor(cluster): update cluster/ test construction sites for sub-config embedding (BC-5, BC-6)

- Update newTestDeploymentConfig(), newTestSimConfigWithWorkload(),
  newTestInstanceSimConfig() helpers
- Update all 18 inline sim.SimConfig{} sites in cluster/ test files
- All assertions unchanged — pure syntax migration

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update cmd/root.go construction site

**Contracts Implemented:** BC-1, BC-9

**Files:**
- Modify: `cmd/root.go:412-447`

**Step 1: Update the single production construction site**

Replace the DeploymentConfig literal (lines 412-447):
```go
config := cluster.DeploymentConfig{
	SimConfig: sim.SimConfig{
		Horizon: simulationHorizon,
		Seed:    seed,
		KVCacheConfig: sim.KVCacheConfig{
			TotalKVBlocks:         totalKVBlocks,
			BlockSizeTokens:       blockSizeTokens,
			KVCPUBlocks:           kvCPUBlocks,
			KVOffloadThreshold:    kvOffloadThreshold,
			KVTransferBandwidth:   kvTransferBandwidth,
			KVTransferBaseLatency: kvTransferBaseLatency,
		},
		BatchConfig: sim.BatchConfig{
			MaxRunningReqs:            maxRunningReqs,
			MaxScheduledTokens:        maxScheduledTokens,
			LongPrefillTokenThreshold: longPrefillTokenThreshold,
		},
		LatencyCoeffs: sim.LatencyCoeffs{
			BetaCoeffs:  betaCoeffs,
			AlphaCoeffs: alphaCoeffs,
		},
		ModelHardwareConfig: sim.ModelHardwareConfig{
			ModelConfig: modelConfig,
			HWConfig:    hwConfig,
			Model:       model,
			GPU:         gpu,
			TP:          tensorParallelism,
			Roofline:    roofline,
		},
		PolicyConfig: sim.PolicyConfig{
			PriorityPolicy: priorityPolicy,
			Scheduler:      scheduler,
		},
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

Note: `GuideLLMConfig` and `TracesWorkloadFilePath` are NOT set in the SimConfig here — they're passed separately to `NewClusterSimulator`. The `WorkloadConfig` sub-config stays zero-valued in this construction site.

**Step 2: Verify full build and all tests pass**

Run: `go build ./... && go test ./... -count=1 2>&1 | tail -10`
Expected: Build succeeds, all tests PASS

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit**

```bash
git add cmd/root.go
git commit -m "refactor(cmd): update CLI construction site for sub-config embedding (BC-1)

- Group SimConfig fields into KVCacheConfig, BatchConfig, LatencyCoeffs,
  ModelHardwareConfig, PolicyConfig sub-configs in cmd/root.go
- WorkloadConfig zero-valued (cluster mode passes workload separately)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Narrow factory signatures

**Contracts Implemented:** BC-2, BC-3, BC-4

**Files:**
- Modify: `sim/kv_store.go`
- Modify: `sim/latency_model.go`
- Modify: `sim/batch_formation.go`
- Modify: `sim/simulator.go` (call sites in NewSimulator)
- Modify: `sim/kv_store_test.go` (5 `NewKVStore(SimConfig{...})` → `NewKVStore(KVCacheConfig{...})`)
- Modify: `sim/latency_model_test.go` (5 `NewLatencyModel(cfg)` → `NewLatencyModel(coeffs, hw)`)
- Modify: `sim/batch_formation_test.go` (7 `NewLatencyModel(cfg)` + 7 `NewBatchFormation(cfg, lm)` + 7 `NewKVStore(cfg)` = 21 factory calls)
- Modify: `sim/simulator_preempt_test.go` (2 `NewLatencyModel(config)` + 2 `NewBatchFormation(config, lm)` + 2 `NewKVStore(config)` = 6 factory calls)

**Step 1: Narrow all three factory signatures + update all call sites atomically**

Context: In Go, compilation is all-or-nothing per package. We cannot write a test for the new signature while the old signature exists — it won't compile. Therefore, Steps 1-3 (signature change + call site updates + new tests) must happen atomically in one step. The compiler verifies completeness: if any call site is missed, the package won't compile.

**Step 3: Narrow `NewKVStore` signature**

In `sim/kv_store.go`, change:
```go
// NewKVStore creates a KVStore from KVCacheConfig.
// Returns *KVCacheState for single-tier (KVCPUBlocks <= 0, the default).
// Returns *TieredKVCache for tiered mode (KVCPUBlocks > 0).
func NewKVStore(cfg KVCacheConfig) KVStore {
	if cfg.TotalKVBlocks <= 0 {
		panic(fmt.Sprintf("KVStore: TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("KVStore: BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
```

**Step 4: Narrow `NewLatencyModel` signature**

In `sim/latency_model.go`, change:
```go
// NewLatencyModel creates the appropriate LatencyModel based on config.
// Returns RooflineLatencyModel if hw.Roofline is true, BlackboxLatencyModel otherwise.
// Returns error if coefficient slices are too short or roofline config validation fails.
func NewLatencyModel(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error) {
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if hw.Roofline {
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", hw.TP)
		}
		if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		return &RooflineLatencyModel{
			modelConfig: hw.ModelConfig,
			hwConfig:    hw.HWConfig,
			tp:          hw.TP,
			alphaCoeffs: coeffs.AlphaCoeffs,
		}, nil
	}
	if len(coeffs.BetaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: BetaCoeffs requires at least 3 elements, got %d", len(coeffs.BetaCoeffs))
	}
	return &BlackboxLatencyModel{
		betaCoeffs:  coeffs.BetaCoeffs,
		alphaCoeffs: coeffs.AlphaCoeffs,
	}, nil
}
```

**Step 5: Drop unused param from `NewBatchFormation`**

In `sim/batch_formation.go`, change:
```go
// NewBatchFormation creates the default BatchFormation.
// Currently returns VLLMBatchFormation (the only implementation).
func NewBatchFormation(latencyModel LatencyModel) BatchFormation {
	return &VLLMBatchFormation{
		latencyModel: latencyModel,
	}
}
```

**Step 6: Update `NewSimulator` call sites**

In `sim/simulator.go`, update the factory calls in `NewSimulator`:
```go
// Change:
latencyModel, err := NewLatencyModel(cfg)
// To:
latencyModel, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)

// Change:
batchFormation := NewBatchFormation(cfg, latencyModel)
// To:
batchFormation := NewBatchFormation(latencyModel)

// Change:
KVCache: NewKVStore(cfg),
// To:
KVCache: NewKVStore(cfg.KVCacheConfig),
```

**Step 7: Update kv_store_test.go call sites (5 sites)**

The existing `NewKVStore(SimConfig{...})` calls need updating to `NewKVStore(KVCacheConfig{...})`:
```go
// Before:
NewKVStore(SimConfig{KVCacheConfig: KVCacheConfig{TotalKVBlocks: 0, BlockSizeTokens: 16}})
// After:
NewKVStore(KVCacheConfig{TotalKVBlocks: 0, BlockSizeTokens: 16})
```

**Step 8: Update latency_model_test.go call sites (5 sites)**

The existing `NewLatencyModel(cfg)` calls (where `cfg` is `SimConfig`) need updating:
```go
// Before:
cfg := SimConfig{...}
lm, err := NewLatencyModel(cfg)
// After:
lm, err := NewLatencyModel(
	LatencyCoeffs{BetaCoeffs: ..., AlphaCoeffs: ...},
	ModelHardwareConfig{...},
)
```

**Step 8b: Update batch_formation_test.go factory call sites (21 sites)**

This file calls all three narrowed factories. Each of the 7 test functions has 3 calls to update:
```go
// Before (each test function):
lm, err := NewLatencyModel(cfg)
bf := NewBatchFormation(cfg, lm)
KVCache: NewKVStore(cfg),

// After:
lm, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
bf := NewBatchFormation(lm)
KVCache: NewKVStore(cfg.KVCacheConfig),
```

**Step 8c: Update simulator_preempt_test.go factory call sites (6 sites)**

This file has 2 test functions, each calling all three factories directly:
```go
// Before (each test function):
lm, _ := NewLatencyModel(config)
bf := NewBatchFormation(config, lm)
sim.KVCache = NewKVStore(config)

// After:
lm, _ := NewLatencyModel(config.LatencyCoeffs, config.ModelHardwareConfig)
bf := NewBatchFormation(lm)
sim.KVCache = NewKVStore(config.KVCacheConfig)
```

**Step 9: Verify all tests pass**

Run: `go test ./... -count=1 2>&1 | tail -10`
Expected: All tests PASS

**Step 10: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 11: Commit**

```bash
git add sim/kv_store.go sim/latency_model.go sim/batch_formation.go sim/simulator.go \
  sim/kv_store_test.go sim/latency_model_test.go sim/batch_formation_test.go \
  sim/simulator_preempt_test.go
git commit -m "refactor(sim): narrow factory signatures to module-scoped sub-configs (BC-2, BC-3, BC-4)

- NewKVStore(KVCacheConfig) — takes only KV fields (was SimConfig)
- NewLatencyModel(LatencyCoeffs, ModelHardwareConfig) — takes coefficients + hardware
- NewBatchFormation(LatencyModel) — drop unused SimConfig param
- Update NewSimulator to pass sub-configs to narrowed factories
- Update test call sites to use narrowed signatures

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update DeploymentConfig field-shadowing test

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/cluster/cluster_test.go:108-130`

**Step 1: Update `TestDeploymentConfig_NoFieldShadowing`**

Context: The current test uses `reflect.TypeOf(sim.SimConfig{})` and iterates top-level fields. After embedding, SimConfig's top-level fields are: `Horizon`, `Seed`, and the 6 embedded sub-config types (which are anonymous). The test's `field.Anonymous` check skips the embedded sub-configs correctly, but the `simFields` map only contains top-level field names (`Horizon`, `Seed`, `KVCacheConfig`, etc.) — not the promoted fields. We need to recursively collect all promoted field names.

```go
func TestDeploymentConfig_NoFieldShadowing(t *testing.T) {
	dcType := reflect.TypeOf(DeploymentConfig{})
	scType := reflect.TypeOf(sim.SimConfig{})

	// Recursively collect all field names from SimConfig (including promoted from embedded structs)
	simFields := make(map[string]bool)
	var collectFields func(t reflect.Type)
	collectFields = func(t reflect.Type) {
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if field.Anonymous {
				collectFields(field.Type)
			} else {
				simFields[field.Name] = true
			}
		}
	}
	collectFields(scType)

	// Check each directly-declared DeploymentConfig field (skip embedded SimConfig)
	for i := 0; i < dcType.NumField(); i++ {
		field := dcType.Field(i)
		if field.Anonymous {
			continue
		}
		if simFields[field.Name] {
			t.Errorf("DeploymentConfig field %q shadows SimConfig field — use promoted access instead", field.Name)
		}
	}
}
```

**Step 2: Run the specific test**

Run: `go test ./sim/cluster/... -run TestDeploymentConfig_NoFieldShadowing -v`
Expected: PASS

**Step 3: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "refactor(cluster): update field-shadowing test for nested sub-config embedding (BC-7)

- Recursively collect promoted field names from SimConfig's embedded sub-configs
- Ensures DeploymentConfig fields don't shadow any promoted SimConfig field

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Update CLAUDE.md and run full verification

**Contracts Implemented:** BC-6, BC-9

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md SimConfig description**

In the "Core Simulation Engine" section, update the `simulator.go` entry:
```
- **simulator.go**: `SimConfig` struct (composed of embedded `KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig` sub-configs), ...
```

In the "Configuration design" principle, add:
```
SimConfig composed of 6 embedded sub-configs (R16). Factory signatures accept the narrowest sub-config: `NewKVStore(KVCacheConfig)`, `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`, `NewBatchFormation(LatencyModel)`.
```

Add `sim/config.go` to the file tree:
```
│   ├── config.go              # Module-scoped sub-config types (KVCacheConfig, BatchConfig, etc.)
```

**Step 2: Run full verification suite**

Run: `go build ./... && go test ./... -count=1 && golangci-lint run ./...`
Expected: Build succeeds, all tests pass, zero lint issues

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for SimConfig sub-config decomposition

- Document new sim/config.go and sub-config types
- Update factory signature documentation
- Update file tree

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 2-4 | Compilation | All existing tests compile with new syntax |
| BC-2 | Task 5 | Unit | `TestNewKVStore_AcceptsKVCacheConfig` |
| BC-3 | Task 5 | Unit | `TestNewLatencyModel_AcceptsSubConfigs` |
| BC-4 | Task 5 | Compilation | `NewBatchFormation(latencyModel)` compiles |
| BC-5 | Task 3 | Compilation | DeploymentConfig field access works transitively |
| BC-6 | Task 2-3 | Golden | `TestSimulator_GoldenDataset`, `TestClusterSimulator_SingleInstance_GoldenEquivalence` |
| BC-7 | Task 6 | Unit | `TestDeploymentConfig_NoFieldShadowing` (updated) |
| BC-9 | Task 7 | Integration | Full `go test ./... -count=1` pass |

**Golden dataset:** No regeneration needed. Output is byte-identical (pure refactoring).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed construction site causes compile error | Low | Low | Go compiler catches all; `go build ./...` in each task | All |
| Sub-config field name collides with DeploymentConfig field | Low | Medium | Verified in Phase 0: no collisions exist | Task 1 |
| Reflect-based field-shadowing test breaks | Medium | Low | Updated test to recursively collect promoted fields | Task 6 |
| Merge conflicts with parallel branches | Medium | Medium | No parallel branches currently modifying SimConfig | N/A |

### E) Review Guide

**The tricky part:** Task 5 (factory signature narrowing) changes public API. Callers outside this repo that import `sim.NewKVStore(SimConfig)` will break. Since BLIS has no external consumers today, this is safe — but worth noting.

**What to scrutinize:** The `TestDeploymentConfig_NoFieldShadowing` update (Task 6) — the recursive field collection must handle the embedded sub-configs correctly.

**What's safe to skim:** Tasks 2-4 are mechanical construction-site updates. The compiler validates correctness.

**Known debt:** Sub-configs don't have canonical constructors yet (e.g., `NewKVCacheConfig`). This is intentional — adding constructors is a separate R4 follow-up when there are enough construction sites per sub-config to warrant it. Currently most tests go through `newTestSimConfig()`.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — sub-configs match 1:1 with existing module groupings
- [x] No feature creep — pure refactoring, no new behavior
- [x] No unexercised flags or interfaces — all sub-config types used by SimConfig
- [x] No partial implementations — all 61 construction sites accounted for
- [x] No breaking changes without contract updates — BC-1 through BC-9 cover all changes
- [x] No hidden global state impact — sub-configs are pure value types
- [x] All new code will pass golangci-lint
- [x] Shared test helpers updated (`newTestSimConfig`, `newTestDeploymentConfig`)
- [x] CLAUDE.md updated (Task 7)
- [x] Deviation log reviewed — 3 deviations documented and justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 → 2,3,4 → 5 → 6 → 7)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed (pure refactoring)
- [x] Construction site audit: 61 sites listed, all covered by Tasks 2-5

**Antipattern rules:**
- [x] R1: No new error paths
- [x] R2: No new map iteration
- [x] R3: No new CLI flags
- [x] R4: Construction sites audited (61 sites, all updated)
- [x] R5: No new resource allocation loops
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: No new golden tests (existing ones preserved)
- [x] R8: No exported mutable maps
- [x] R9: No new YAML fields
- [x] R10: No new YAML parsing
- [x] R11: No new division
- [x] R12: Golden dataset unchanged
- [x] R13: No new interfaces
- [x] R14: No new multi-module methods
- [x] R15: No stale PR references
- [x] R16: ✅ This PR's primary purpose — config grouped by module
- [x] R17: No new routing scorers

---

## Appendix: File-Level Implementation Details

### File: `sim/config.go` (NEW)

**Purpose:** Define 6 module-scoped sub-config types that compose into SimConfig.

See Task 1, Step 1 for complete implementation.

### File: `sim/simulator.go` (MODIFY lines 96-123)

**Purpose:** Replace flat SimConfig with embedded composition.

See Task 1, Step 2 for the new SimConfig definition.
See Task 5, Step 6 for the updated NewSimulator factory calls.

### File: `sim/kv_store.go` (MODIFY)

**Purpose:** Narrow `NewKVStore` from `SimConfig` to `KVCacheConfig`.

See Task 5, Step 3 for complete implementation.

### File: `sim/latency_model.go` (MODIFY lines 121-151)

**Purpose:** Narrow `NewLatencyModel` from `SimConfig` to `(LatencyCoeffs, ModelHardwareConfig)`.

See Task 5, Step 4 for complete implementation.

### File: `sim/batch_formation.go` (MODIFY line 183)

**Purpose:** Drop unused `SimConfig` parameter from `NewBatchFormation`.

See Task 5, Step 5 for complete implementation.

### File: `cmd/root.go` (MODIFY lines 412-447)

**Purpose:** Update single production construction site with sub-config grouping.

See Task 4, Step 1 for complete implementation.

### File: `sim/cluster/cluster_test.go` (MODIFY lines 108-130)

**Purpose:** Update field-shadowing test for recursive promoted field collection.

See Task 6, Step 1 for complete implementation.
