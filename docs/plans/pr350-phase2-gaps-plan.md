# PR #350 Phase 2 Gaps: Canonical Constructors, Step Decomposition, Doc Update

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the remaining standards gaps from the SimConfig decomposition (#350) — canonical constructors prevent silent field omission (R4), Step() decomposition separates concerns (R14), and the design guidelines module map is corrected (stale §4.2).

**The problem today:** The 6 sub-config types (`KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`, `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig`) have ~60 SimConfig construction sites (each containing 3-6 sub-config struct literals, totalling ~180+ individual replacements) across test and production code. Adding a new field to any sub-config compiles silently with zero-value — the compiler can't catch missed sites. The `Step()` method is 152 lines mixing 4 concerns. The design guidelines document says KVStore has 9 methods (it has 11) and config extension costs 5-6 files (it costs 2).

**What this PR adds:**
1. **Canonical constructors for all 6 sub-configs** — `NewKVCacheConfig(...)`, `NewBatchConfig(...)`, etc. Adding a parameter to a constructor is a compile error at every call site.
2. **Step() decomposed into 4 named phase methods** — `scheduleBatch`, `executeBatchStep`, `processCompletions`, `scheduleNextStep`. Each handles exactly one concern.
3. **Design guidelines §4.2 and §6.2 corrected** — KVStore method count, config touch-point count, monolith method reference, and other stale entries updated.

**Why this matters:** R4 compliance prevents the #181-class bug (missed construction site) from recurring at the sub-config level. R14 compliance makes the simulator's hot path readable and each phase independently modifiable. Accurate documentation prevents misleading future contributors.

**Architecture:** Pure refactoring — no behavioral change, no new interfaces, no new features. Constructors live in `sim/config.go` alongside their types. Phase methods live in `sim/simulator.go` alongside `Step()`. All existing tests must pass without assertion changes.

**Source:** GitHub issue #350 (remaining gaps section: R4, R14, §4.2)

**Closes:** Fixes #350

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR addresses the three remaining standards gaps from issue #350 after the SimConfig decomposition (#381):

1. **R4 (Canonical constructors):** Adds `New*Config()` constructors for all 6 sub-config types and migrates all ~60 Go construction sites. This ensures that adding a field to any sub-config is a compiler-enforced change at every call site.

2. **R14 (Step decomposition):** Extracts the 4 phases of `Step()` into named private methods. The orchestrator becomes 4 lines. Each phase method handles exactly one concern (scheduling, execution, completion, next-step).

3. **§4.2 (Doc accuracy):** Corrects stale entries in `docs/templates/design-guidelines.md` — KVStore method count (9→11), config parameter touch-point count (5-6→2), and Monolith Method line count reference (134→151).

**Adjacent blocks:** `sim/config.go` (constructors), `sim/simulator.go` (Step decomposition), `cmd/root.go` (production construction site), `docs/templates/design-guidelines.md` (documentation).

**No deviations from #350's recommended sequence.** R13/KVStore interface redesign is explicitly deferred (#246).

### B) Behavioral Contracts

**Positive Contracts (what MUST happen):**

**BC-1: Constructor output equivalence**
- GIVEN any combination of valid field values for a sub-config type
- WHEN the corresponding `New*Config()` constructor is called with those values
- THEN the returned struct MUST have all fields set to exactly the values passed
- MECHANISM: Each constructor is a direct struct literal return — no transformation, no defaults, no validation

**BC-2: Behavioral preservation (constructors)**
- GIVEN the existing test suite passes before constructor migration
- WHEN all struct literal construction sites are replaced with constructor calls using the same field values
- THEN all existing tests MUST pass without any assertion changes
- MECHANISM: Constructors produce identical values to the struct literals they replace

**BC-3: Behavioral preservation (Step decomposition)**
- GIVEN the existing test suite passes before Step() decomposition
- WHEN Step() is decomposed into 4 named phase methods
- THEN all existing tests MUST pass without any assertion changes
- MECHANISM: The extracted methods contain the exact same code in the same order; only the function boundary changes

**Negative Contracts (what MUST NOT happen):**

**BC-4: No default injection**
- GIVEN a `New*Config()` constructor
- WHEN called with zero-value arguments
- THEN the constructor MUST NOT inject non-zero defaults — all fields MUST be exactly the passed values
- MECHANISM: Constructors are trivial passthrough (no `if x == 0 { x = default }` logic)

**BC-5: Two-pass ordering preserved**
- GIVEN a running batch with zero-output-token requests
- WHEN Step() executes
- THEN `executeBatchStep` (TTFT recording) MUST complete before `processCompletions` (E2E recording)
- MECHANISM: The decomposition preserves the existing two-pass design (Phase 2 loop before Phase 3 loop)

**BC-6: Design guidelines accuracy**
- GIVEN the design guidelines module map (§4.2)
- WHEN a contributor reads the KVStore entry
- THEN the method count MUST match the actual `KVStore` interface (11 methods)
- MECHANISM: Direct documentation update

### C) Component Interaction

```
                    sim/config.go (6 constructors)
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      cmd/root.go    sim/*_test.go  sim/cluster/*_test.go
      (1 production   (~38 test      (~22 test
       site)           sites)         sites)

                    sim/simulator.go
                         │
    Step(now) ───> scheduleBatch(now)
                   executeBatchStep(now) int64
                   processCompletions(now, advance) []*Request
                   scheduleNextStep(now, advance, remaining)
```

**API Contracts:**
- 6 new exported constructors in `sim/` package — each takes all fields as positional parameters matching struct field order
- 4 new unexported methods on `*Simulator` — private, called only from `Step()`
- No new interfaces, no new types, no new events

**State Changes:** None. All methods operate on the same `*Simulator` state as before.

**Extension Friction:** Adding a field to a sub-config now costs: 1 file (constructor + struct in `sim/config.go`) + the compiler catches all N call sites. Down from "grep and hope."

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #350 lists R4, R13, R14, §4.2 as remaining gaps | This plan implements R4, R14, §4.2 only | DEFERRAL: R13 (KVStore interface redesign) is tracked separately as #246 — too large for this PR |
| Issue #350 recommends "each sub-config should get a canonical constructor" | Constructors take all fields as positional params | SIMPLIFICATION: Functional options or builder patterns are over-engineering for 2-6 field structs |
| `GuideLLMConfig` has a canonical constructor (`NewGuideLLMConfig`) | ~33 bare `GuideLLMConfig{}` struct literals NOT migrated | DEFERRAL: GuideLLMConfig is not a sub-config type (it's a workload generation struct). Its 11-parameter constructor makes test sites very noisy. Migration is a natural follow-up. |

### E) Review Guide

**The tricky part:** The Step() decomposition in Task 5 must preserve exact code behavior including the two-pass invariant (Phase 2 before Phase 3). Verify that `executeBatchStep` records TTFT before `processCompletions` records E2E.

**What to scrutinize:** BC-5 (two-pass ordering) — trace through a zero-output-token request scenario mentally to verify TTFT is set before E2E.

**What's safe to skim:** Tasks 2-4 are purely mechanical replacements (struct literal → constructor call). If Task 1's constructor tests pass and Task 2-4's `go test` passes, the migration is correct.

**Known debt:**
- `SimConfig` itself still has no canonical constructor (60 construction sites remain as struct literals composing sub-configs). This is acceptable — sub-config constructors are the first step; a `NewSimConfig()` with 8 parameters (6 sub-configs + Horizon + Seed) is the natural follow-up.
- ~33 `GuideLLMConfig{}` bare struct literals not migrated to `NewGuideLLMConfig` — the existing constructor has 11 parameters, making test sites very noisy. Natural follow-up.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/config_test.go` — constructor equivalence tests (BC-1, BC-4)

**Files to modify:**
- `sim/config.go` — add 6 canonical constructors + clarify "default" comments (BC-1, BC-4)
- `cmd/root.go` — replace 6 sub-config struct literals with constructor calls (BC-2)
- `sim/simulator_test.go` — replace ~16 construction sites (BC-2)
- `sim/batch_formation_test.go` — replace ~8 construction sites (BC-2)
- `sim/scheduler_test.go` — replace ~4 construction sites (BC-2)
- `sim/simulator_preempt_test.go` — replace 2 construction sites (BC-2)
- `sim/simulator_decode_test.go` — replace 1 construction site (BC-2)
- `sim/model_hardware_config_test.go` — replace 3 construction sites (BC-2)
- `sim/latency_model_test.go` — replace ~7 standalone + ~3 SimConfig sites (BC-2)
- `sim/kv_store_test.go` — replace ~6 standalone KVCacheConfig sites (BC-2)
- `sim/cluster/cluster_test.go` — replace ~7 construction sites (BC-2)
- `sim/cluster/instance_test.go` — replace ~4 construction sites (BC-2)
- `sim/cluster/pending_requests_test.go` — replace 4 construction sites (BC-2)
- `sim/cluster/cluster_trace_test.go` — replace 4 construction sites (BC-2)
- `sim/cluster/evaluation_test.go` — replace 1 construction site (BC-2)
- `sim/cluster/prefix_routing_test.go` — replace 1 construction site (BC-2)
- `sim/cluster/snapshot_test.go` — replace 1 construction site (BC-2)
- `sim/simulator.go` — extract Step() into 4 named methods (BC-3, BC-5)
- `docs/templates/design-guidelines.md` — correct stale §4.2 and §6.2 entries (BC-6)

**Key decisions:**
- Constructor parameter order matches struct field order (convention from `NewGuideLLMConfig`)
- Constructors are trivial passthrough (no validation, no defaults) — validation belongs in factories
- Step() phase methods are unexported (private implementation detail)

### G) Task Breakdown

---

#### Task 1: Add canonical constructors for all 6 sub-config types

**Contracts Implemented:** BC-1, BC-4

**Files:**
- Modify: `sim/config.go` (add 6 constructors after each type)
- Create: `sim/config_test.go` (constructor equivalence tests)

**Step 1: Write failing test for constructor equivalence**

Context: Verify each constructor produces the exact struct it replaces. Table-driven test covers all 6 types.

In `sim/config_test.go`:
```go
package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewKVCacheConfig_FieldEquivalence(t *testing.T) {
	got := NewKVCacheConfig(100, 16, 50, 0.9, 100.0, 500)
	want := KVCacheConfig{
		TotalKVBlocks:         100,
		BlockSizeTokens:       16,
		KVCPUBlocks:           50,
		KVOffloadThreshold:    0.9,
		KVTransferBandwidth:   100.0,
		KVTransferBaseLatency: 500,
	}
	assert.Equal(t, want, got)
}

func TestNewBatchConfig_FieldEquivalence(t *testing.T) {
	got := NewBatchConfig(10, 1000, 200)
	want := BatchConfig{
		MaxRunningReqs:            10,
		MaxScheduledTokens:        1000,
		LongPrefillTokenThreshold: 200,
	}
	assert.Equal(t, want, got)
}

func TestNewLatencyCoeffs_FieldEquivalence(t *testing.T) {
	beta := []float64{1000, 10, 2}
	alpha := []float64{500, 1, 1000}
	got := NewLatencyCoeffs(beta, alpha)
	want := LatencyCoeffs{BetaCoeffs: beta, AlphaCoeffs: alpha}
	assert.Equal(t, want, got)
}

func TestNewModelHardwareConfig_FieldEquivalence(t *testing.T) {
	mc := ModelConfig{NumLayers: 32}
	hw := HardwareCalib{NumGPUs: 2}
	got := NewModelHardwareConfig(mc, hw, "llama", "H100", 2, true)
	want := ModelHardwareConfig{
		ModelConfig: mc,
		HWConfig:    hw,
		Model:       "llama",
		GPU:         "H100",
		TP:          2,
		Roofline:    true,
	}
	assert.Equal(t, want, got)
}

func TestNewPolicyConfig_FieldEquivalence(t *testing.T) {
	got := NewPolicyConfig("slo-based", "priority-fcfs")
	want := PolicyConfig{PriorityPolicy: "slo-based", Scheduler: "priority-fcfs"}
	assert.Equal(t, want, got)
}

func TestNewWorkloadConfig_FieldEquivalence(t *testing.T) {
	glm := &GuideLLMConfig{Rate: 1.0, NumRequests: 100}
	got := NewWorkloadConfig(glm, "/path/to/trace.csv")
	want := WorkloadConfig{GuideLLMConfig: glm, TracesWorkloadFilePath: "/path/to/trace.csv"}
	assert.Equal(t, want, got)
}

func TestNewKVCacheConfig_ZeroValues_NoDefaults(t *testing.T) {
	// BC-4: Zero-value arguments must NOT inject non-zero defaults
	got := NewKVCacheConfig(0, 0, 0, 0, 0, 0)
	assert.Equal(t, KVCacheConfig{}, got)
}

func TestNewWorkloadConfig_NilGuideLLM(t *testing.T) {
	got := NewWorkloadConfig(nil, "")
	assert.Equal(t, WorkloadConfig{}, got)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNew.*Config -v`
Expected: FAIL — constructors don't exist yet

**Step 3: Implement constructors**

In `sim/config.go`, first fix the misleading "default" comments on KVCacheConfig:
```go
// BEFORE:
KVOffloadThreshold    float64 // GPU utilization threshold for offload (default 0.9)
KVTransferBandwidth   float64 // blocks/tick transfer rate (default 100.0)
// AFTER:
KVOffloadThreshold    float64 // GPU utilization threshold for offload (CLI default: 0.9, zero-value: 0)
KVTransferBandwidth   float64 // blocks/tick transfer rate (CLI default: 100.0, zero-value: 0)
```

Then add canonical constructors after each type definition:

```go
// NewKVCacheConfig creates a KVCacheConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks int64,
	kvOffloadThreshold, kvTransferBandwidth float64,
	kvTransferBaseLatency int64) KVCacheConfig {
	return KVCacheConfig{
		TotalKVBlocks:         totalKVBlocks,
		BlockSizeTokens:       blockSizeTokens,
		KVCPUBlocks:           kvCPUBlocks,
		KVOffloadThreshold:    kvOffloadThreshold,
		KVTransferBandwidth:   kvTransferBandwidth,
		KVTransferBaseLatency: kvTransferBaseLatency,
	}
}

// NewBatchConfig creates a BatchConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64) BatchConfig {
	return BatchConfig{
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
	}
}

// NewLatencyCoeffs creates a LatencyCoeffs with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewLatencyCoeffs(betaCoeffs, alphaCoeffs []float64) LatencyCoeffs {
	return LatencyCoeffs{
		BetaCoeffs:  betaCoeffs,
		AlphaCoeffs: alphaCoeffs,
	}
}

// NewModelHardwareConfig creates a ModelHardwareConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewModelHardwareConfig(modelConfig ModelConfig, hwConfig HardwareCalib,
	model, gpu string, tp int, roofline bool) ModelHardwareConfig {
	return ModelHardwareConfig{
		ModelConfig: modelConfig,
		HWConfig:    hwConfig,
		Model:       model,
		GPU:         gpu,
		TP:          tp,
		Roofline:    roofline,
	}
}

// NewPolicyConfig creates a PolicyConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewPolicyConfig(priorityPolicy, scheduler string) PolicyConfig {
	return PolicyConfig{
		PriorityPolicy: priorityPolicy,
		Scheduler:      scheduler,
	}
}

// NewWorkloadConfig creates a WorkloadConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewWorkloadConfig(guideLLMConfig *GuideLLMConfig, tracesWorkloadFilePath string) WorkloadConfig {
	return WorkloadConfig{
		GuideLLMConfig:         guideLLMConfig,
		TracesWorkloadFilePath: tracesWorkloadFilePath,
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestNew.*Config -v`
Expected: PASS (all 8 tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/config.go sim/config_test.go
git commit -m "feat(sim): add canonical constructors for 6 sub-config types (R4, BC-1)

- Add NewKVCacheConfig, NewBatchConfig, NewLatencyCoeffs,
  NewModelHardwareConfig, NewPolicyConfig, NewWorkloadConfig
- Each constructor takes all fields as positional params (R4 compliance)
- Add equivalence and zero-value tests (BC-1, BC-4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Migrate production code (cmd/root.go) to use constructors

**Contracts Implemented:** BC-2

**Files:**
- Modify: `cmd/root.go:435-463` (replace 6 sub-config struct literals)

**Step 1: Write failing test — N/A** (existing cmd tests serve as regression guard)

**Step 2: Verify existing tests pass before migration**

Run: `go test ./cmd/... -v`
Expected: PASS

**Step 3: Replace struct literals with constructor calls**

In `cmd/root.go`, replace the `SimConfig` construction (around line 435-463). The current code:
```go
KVCacheConfig: sim.KVCacheConfig{
    TotalKVBlocks:         totalKVBlocks,
    BlockSizeTokens:       blockSizeTokens,
    KVCPUBlocks:           kvCPUBlocks,
    KVOffloadThreshold:    kvOffloadThreshold,
    KVTransferBandwidth:   kvTransferBandwidth,
    KVTransferBaseLatency: kvTransferBaseLatency,
},
```

Becomes:
```go
KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
    kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
```

Apply the same pattern for all 6 sub-configs:
- `sim.KVCacheConfig{...}` → `sim.NewKVCacheConfig(...)`
- `sim.BatchConfig{...}` → `sim.NewBatchConfig(...)`
- `sim.LatencyCoeffs{...}` → `sim.NewLatencyCoeffs(...)`
- `sim.ModelHardwareConfig{...}` → `sim.NewModelHardwareConfig(...)`
- `sim.PolicyConfig{...}` → `sim.NewPolicyConfig(...)`

Note: `WorkloadConfig` is not set in `cmd/root.go` (workload is passed separately in cluster mode).

**Step 4: Run tests**

Run: `go test ./cmd/... -v && go build ./...`
Expected: PASS and build succeeds

**Step 5: Run lint**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go
git commit -m "refactor(cmd): use canonical sub-config constructors in root.go (BC-2)

- Replace 5 sub-config struct literals with constructor calls
- No behavioral change — same field values

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Migrate sim/ package test files to use constructors

**Contracts Implemented:** BC-2

**Files to modify (all in `sim/`):**
- `simulator_test.go` (~16 SimConfig sites)
- `batch_formation_test.go` (~8 sites: 7 SimConfig + 1 standalone KVCacheConfig)
- `scheduler_test.go` (~4 SimConfig sites)
- `simulator_preempt_test.go` (2 SimConfig sites)
- `simulator_decode_test.go` (1 SimConfig site)
- `model_hardware_config_test.go` (3 SimConfig sites)
- `latency_model_test.go` (~10 sites: 3 SimConfig + 4 standalone LatencyCoeffs + 3 standalone ModelHardwareConfig)
- `kv_store_test.go` (~6 standalone KVCacheConfig sites)

**Replacement patterns** (apply mechanically across all files):

For each `KVCacheConfig{...}` struct literal, replace with `NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks, kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency)`. Fields not set in the original literal are 0/0.0.

Example:
```go
// BEFORE:
KVCacheConfig: KVCacheConfig{
    TotalKVBlocks:   10000,
    BlockSizeTokens: 16,
},
// AFTER:
KVCacheConfig: NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
```

For `BatchConfig{...}`:
```go
// BEFORE:
BatchConfig: BatchConfig{
    MaxRunningReqs:            100,
    MaxScheduledTokens:        1000,
    LongPrefillTokenThreshold: 100,
},
// AFTER:
BatchConfig: NewBatchConfig(100, 1000, 100),
```

For `LatencyCoeffs{...}`:
```go
// BEFORE:
LatencyCoeffs: LatencyCoeffs{
    BetaCoeffs:  tc.BetaCoeffs,
    AlphaCoeffs: tc.AlphaCoeffs,
},
// AFTER:
LatencyCoeffs: NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
```

For `ModelHardwareConfig{...}`:
```go
// BEFORE:
ModelHardwareConfig: ModelHardwareConfig{
    Model: tc.Model,
    GPU:   tc.Hardware,
    TP:    tc.TP,
},
// AFTER:
ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
```

For `PolicyConfig{...}`:
```go
// BEFORE:
PolicyConfig: PolicyConfig{
    PriorityPolicy: "slo-based",
    Scheduler:      "priority-fcfs",
},
// AFTER:
PolicyConfig: NewPolicyConfig("slo-based", "priority-fcfs"),
```

For `WorkloadConfig{...}`:
```go
// BEFORE:
WorkloadConfig: WorkloadConfig{
    GuideLLMConfig: &GuideLLMConfig{...},
},
// AFTER:
WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{...}, ""),
```

**Step 1: Apply replacements** across all `sim/` test files listed above.

**Step 2: Run tests**

Run: `go test ./sim/... -count=1`
Expected: ALL PASS (BC-2 — no behavioral change)

**Step 3: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/*_test.go
git commit -m "refactor(sim): migrate test construction sites to canonical constructors (BC-2)

- Replace ~50 sub-config struct literals with constructor calls
- Files: simulator_test, batch_formation_test, scheduler_test,
  simulator_preempt_test, simulator_decode_test, model_hardware_config_test,
  latency_model_test, kv_store_test
- No assertion changes — pure mechanical replacement

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Migrate sim/cluster/ test files to use constructors

**Contracts Implemented:** BC-2

**Files to modify (all in `sim/cluster/`):**
- `cluster_test.go` (~7 SimConfig sites)
- `instance_test.go` (~4 SimConfig sites)
- `pending_requests_test.go` (4 SimConfig sites)
- `cluster_trace_test.go` (4 SimConfig sites)
- `evaluation_test.go` (1 SimConfig site)
- `prefix_routing_test.go` (1 SimConfig site)
- `snapshot_test.go` (1 SimConfig site)

**Replacement patterns:** Same as Task 3, but with `sim.` prefix:
```go
// BEFORE:
KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 10000, BlockSizeTokens: 16},
// AFTER:
KVCacheConfig: sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
```

**Step 1: Apply replacements** across all `sim/cluster/` test files listed above.

**Step 2: Run tests**

Run: `go test ./sim/cluster/... -count=1`
Expected: ALL PASS

**Step 3: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/cluster/*_test.go
git commit -m "refactor(cluster): migrate test construction sites to canonical constructors (BC-2)

- Replace ~22 sub-config struct literals with constructor calls
- Files: cluster_test, instance_test, pending_requests_test,
  cluster_trace_test, evaluation_test, prefix_routing_test, snapshot_test
- No assertion changes — pure mechanical replacement

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Extract Step() into 4 named phase methods (R14)

**Contracts Implemented:** BC-3, BC-5

**Files:**
- Modify: `sim/simulator.go:366-517` (decompose Step method)

**Step 1: Add zero-output-token regression test for BC-5**

Context: The two-pass invariant (BC-5) is critical for zero-output-token requests but has no dedicated test. Add one before decomposing to serve as a regression guard.

In `sim/simulator_test.go`, add:
```go
func TestStep_ZeroOutputTokens_TTFTBeforeE2E(t *testing.T) {
	// BC-5: TTFT must be recorded before E2E for zero-output-token requests.
	// This tests the two-pass invariant: executeBatchStep (Phase 2) records TTFT,
	// then processCompletions (Phase 3) records E2E.
	cfg := SimConfig{
		Horizon: 100_000_000,
		KVCacheConfig: NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(100, 10000, 100),
		LatencyCoeffs: NewLatencyCoeffs(
			[]float64{1000, 10, 2},
			[]float64{500, 1, 1000},
		),
	}
	sim, err := NewSimulator(cfg)
	assert.NoError(t, err)

	// Create a request with zero output tokens
	req := &Request{
		ID:           "zero-output",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: []int{}, // zero output tokens
		State:        StateQueued,
	}
	sim.InjectArrival(req)

	// Run until completion
	sim.Run()

	// TTFT must be recorded
	ttft, hasTTFT := sim.Metrics.RequestTTFTs[req.ID]
	assert.True(t, hasTTFT, "TTFT must be recorded for zero-output request")
	assert.Greater(t, ttft, 0.0, "TTFT must be positive")

	// E2E must be recorded
	e2e, hasE2E := sim.Metrics.RequestE2Es[req.ID]
	assert.True(t, hasE2E, "E2E must be recorded for zero-output request")
	assert.Greater(t, e2e, 0.0, "E2E must be positive")

	// Request must have completed
	assert.Equal(t, 1, sim.Metrics.CompletedRequests)
}
```

Run: `go test ./sim/... -run TestStep_ZeroOutputTokens -v`
Expected: PASS (verifies existing behavior before decomposition)

**Step 2: Verify all existing tests pass before decomposition**

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: ALL PASS

**Step 3: Extract 4 phase methods and refactor Step()**

Replace the current `Step()` method (lines 366-517) with the decomposed version.

The new `Step()` orchestrator:
```go
// Step simulates a single vllm step(): batch scheduling, model execution, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (3) process completions, (4) schedule next step.
func (sim *Simulator) Step(now int64) {
	sim.scheduleBatch(now)
	currStepAdvance := sim.executeBatchStep(now)
	remaining := sim.processCompletions(now, currStepAdvance)
	sim.scheduleNextStep(now, currStepAdvance, remaining)
}
```

The 4 extracted phase methods (add below Step):

```go
// scheduleBatch handles Phase 1: priority assignment, queue reordering, batch formation,
// and event scheduling for preemptions and newly scheduled requests.
func (sim *Simulator) scheduleBatch(now int64) {
	sim.stepCount += 1

	// Synchronize KV cache clock for thrashing detection (no-op for single-tier KVCacheState)
	sim.KVCache.SetClock(now)

	// Assign priorities to queued requests and order queue per scheduler policy
	for _, req := range sim.WaitQ.Items() {
		req.Priority = sim.priorityPolicy.Compute(req, now)
	}
	sim.WaitQ.Reorder(func(reqs []*Request) {
		sim.scheduler.OrderQueue(reqs, now)
	})

	// Delegate batch composition to the pluggable BatchFormation strategy.
	// Event scheduling and metrics recording happen after FormBatch returns (kernel concerns).
	batchCtx := BatchContext{
		RunningBatch:          sim.RunningBatch,
		WaitQ:                 sim.WaitQ,
		KVCache:               sim.KVCache,
		MaxScheduledTokens:    sim.maxScheduledTokens,
		MaxRunningReqs:        sim.maxRunningReqs,
		PrefillTokenThreshold: sim.longPrefillTokenThreshold,
		Now:                   now,
		StepCount:             sim.stepCount,
		ComputedTokens:        sim.reqNumComputedTokens,
	}
	batchResult := sim.batchFormation.FormBatch(batchCtx)

	// Apply result: update running batch
	sim.RunningBatch = batchResult.RunningBatch

	// Schedule events for preempted requests and record preemption metrics
	for _, p := range batchResult.Preempted {
		sim.Schedule(&PreemptionEvent{
			time:    now + p.PreemptionDelay,
			Request: p.Request,
		})
		sim.Metrics.PreemptionCount++
	}

	// Schedule events for newly scheduled requests and record scheduling metrics
	for _, s := range batchResult.NewlyScheduled {
		sim.Schedule(&ScheduledEvent{
			time:    now + s.ScheduledDelay,
			Request: s.Request,
		})
		sim.Metrics.RequestSchedulingDelays[s.Request.ID] = now + s.ScheduledDelay - s.Request.ArrivalTime
	}

	// Record queue depth observations after batch formation
	sim.recordQueueSnapshots()
}

// executeBatchStep handles Phase 2: model execution (prefill + decode) for all requests
// in the running batch. Returns the step time advance in ticks.
func (sim *Simulator) executeBatchStep(now int64) int64 {
	// Estimate step time via LatencyModel (blackbox or roofline, selected at construction)
	currStepAdvance := sim.latencyModel.StepTime(sim.RunningBatch.Requests)

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	// Note: TotalOutputTokens++ and TTFT metrics are recorded inline (not extracted to helpers)
	// because they are tightly coupled to the prefill/decode state transitions in this loop.
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < Len64(req.InputTokens) {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
		}
		if req.ProgressIndex == Len64(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			sim.Metrics.TTFTSum += req.FirstTokenTime // in microsec
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime)
		}
	}

	// Record KV cache usage observations after execution
	sim.recordKVUsageMetrics(currStepAdvance)

	return currStepAdvance
}

// processCompletions handles Phase 3: identifies completed requests, performs state
// transitions, releases KV blocks, and records completion metrics.
// Returns the remaining (non-completed) requests.
//
// IMPORTANT: This MUST run as a separate pass after executeBatchStep (BC-5).
// For zero-output-token requests, both "prefill completed" and "request completed"
// conditions are true in the same step. The two-pass design ensures prefill metrics
// (TTFT) are recorded before completion metrics (E2E). If these were ever
// consolidated into a single pass, both branches would fire for the same request
// in the same step.
func (sim *Simulator) processCompletions(now, currStepAdvance int64) []*Request {
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == Len64(req.InputTokens)+max(Len64(req.OutputTokens), 1)-1 {
			// State transitions
			req.State = StateCompleted
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
			if len(req.OutputTokens) > 0 {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					logrus.Errorf("[tick %07d] KV allocation failed for completing request %s (request will still complete) — this indicates a cache accounting bug", now, req.ID)
					sim.Metrics.KVAllocationFailures++
				}
			}
			// ReleaseKVBlocks is safe even when the final-token allocation failed:
			// AllocateKVBlocks only modifies RequestMap on success, so Release
			// frees exactly the blocks from prior successful allocations.
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})

			// Record completion metrics
			sim.recordRequestCompletion(req)
		} else {
			remaining = append(remaining, req)
		}
	}
	return remaining
}

// scheduleNextStep handles Phase 4: schedules the next step event based on
// remaining requests, or starts a new batch if only WaitQ has pending work
// (work-conserving property, INV-8).
func (sim *Simulator) scheduleNextStep(now, currStepAdvance int64, remaining []*Request) {
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		// estimate queue overhead from LR (sim.features)
		//
		pbe := StepEvent{time: now + currStepAdvance}
		sim.Schedule(&pbe)
		sim.stepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.stepEvent = nil
		// Work-conserving: if WaitQ has pending requests, immediately
		// schedule a new step to form the next batch. Without this,
		// queued requests are stranded until the next arrival event
		// triggers a QueuedEvent — violating the work-conserving
		// property that real vLLM maintains.
		if sim.WaitQ.Len() > 0 {
			pbe := StepEvent{time: now + currStepAdvance}
			sim.Schedule(&pbe)
			sim.stepEvent = &pbe
		}
	}
}
```

**Step 4: Run tests to verify behavioral preservation**

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: ALL PASS (BC-3 — identical behavior, including new zero-output test)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_test.go
git commit -m "refactor(sim): decompose Step() into 4 named phase methods (R14, BC-3, BC-5)

- Extract scheduleBatch(), executeBatchStep(), processCompletions(), scheduleNextStep()
- Step() becomes a 4-line orchestrator
- Preserves two-pass invariant (BC-5: TTFT before E2E for zero-output requests)
- No behavioral change — all existing tests pass unchanged

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: Update design guidelines §4.2 and §6.2

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/templates/design-guidelines.md` (§4.2 module map, §4.5 touch-point table, §6.2 anti-patterns)

**Step 1: Fix stale entries in design-guidelines.md**

In §4.2 module map table, update KVStore entry:
```
// BEFORE:
| **KV Cache Manager** | Allocate/release/cache KV blocks | `KVStore` (9 methods) | Implemented |
// AFTER:
| **KV Cache Manager** | Allocate/release/cache KV blocks | `KVStore` (11 methods) | Implemented |
```

In §4.5 touch-point table, update config parameter entry:
```
// BEFORE:
| New config parameter | ~3 files | 5-6 files (exceeds — known friction) |
// AFTER:
| New config parameter | ~2 files | 2 files (meets target — post-#381) |
```

In §6.2, update THREE entries:

Entry 1 — Interface Leaking Implementation (line ~444):
```
// BEFORE:
| **Interface Leaking Implementation** | `KVStore` interface has 9 methods, several exposing block-level semantics. ...
// AFTER:
| **Interface Leaking Implementation** | `KVStore` interface has 11 methods, several exposing block-level semantics. ...
```

Entry 2 — Monolith Method (line ~445):
```
// BEFORE:
| **Monolith Method** | `Simulator.Step()` is 134 lines mixing scheduling, latency estimation, token generation, completion, and metrics. Impossible to swap the latency model without modifying this method. | ...
// AFTER:
| **Monolith Method** | `Simulator.Step()` was 152 lines mixing 4 concerns. Decomposed into named phase methods (`scheduleBatch`, `executeBatchStep`, `processCompletions`, `scheduleNextStep`). | Each module's logic should be callable through its interface. When a method contains logic for multiple modules, extract each into its module's interface method. |
```

Entry 3 — Config Mixing Concerns (line ~446):
```
// BEFORE:
| **Config Mixing Concerns** | `SimConfig` combines hardware identity, model parameters, simulation parameters, and policy choices. Adding one autoscaling parameter requires understanding the entire struct. | ...
// AFTER:
| **Config Mixing Concerns** | `SimConfig` combined 23 fields from 8 concerns. Decomposed into 6 embedded sub-configs with canonical constructors (`NewKVCacheConfig`, etc.). Adding a field now touches 2 files; the compiler catches all call sites. | Group configuration by module. Each module's config should be independently specifiable and validatable. |
```

**Step 2: Run build to ensure no issues**

Run: `go build ./...`
Expected: Success

**Step 4: Commit**

```bash
git add docs/templates/design-guidelines.md
git commit -m "docs: update design guidelines §4.2 and §6.2 for #350 completion (BC-6)

- Fix KVStore method count: 9 → 11 (§4.2 and §6.2)
- Fix config parameter touch-point: 5-6 files → 2 files (§4.5)
- Update monolith method and config mixing entries for completed work (§6.2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 1 | Unit | TestNew*Config_FieldEquivalence (6 tests) |
| BC-4 | Task 1 | Unit | TestNewKVCacheConfig_ZeroValues_NoDefaults |
| BC-4 | Task 1 | Unit | TestNewWorkloadConfig_NilGuideLLM |
| BC-2 | Tasks 2-4 | Regression | All existing tests pass unchanged |
| BC-3 | Task 5 | Regression | All existing tests pass unchanged |
| BC-5 | Task 5 | Unit | TestStep_ZeroOutputTokens_TTFTBeforeE2E (new — dedicated regression guard) |
| BC-6 | Task 6 | Manual | Visual inspection of doc changes |

**Golden dataset:** No update needed — this PR is a pure refactor with no behavioral change.

**Invariant tests:** No new invariant tests needed — existing INV-1 through INV-8 tests validate that the refactoring preserves all system invariants.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Constructor parameter order mismatch | Medium | High (silent wrong values) | Convention: param order matches struct field order; equivalence tests catch mismatches | Task 1 |
| Mechanical replacement error (wrong zero value) | Low | Medium (test failure) | Tests catch immediately; ~60 sites but each is mechanical | Tasks 3-4 |
| Step() decomposition changes behavior | Low | High (metric regression) | Exact code preservation; all existing tests validate behavior | Task 5 |
| Two-pass ordering broken by decomposition | Low | High (wrong TTFT for zero-output requests) | BC-5 contract; processCompletions doc comment explains invariant | Task 5 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — constructors are minimal, no validation or defaults
- [x] No feature creep — only R4, R14, §4.2 from #350
- [x] No unexercised flags or interfaces
- [x] No partial implementations — all 6 sub-configs get constructors
- [x] No breaking changes — pure refactor
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: not needed (constructors are the helpers)
- [x] CLAUDE.md: no update needed (remaining-gaps note is in MEMORY.md, not CLAUDE.md)
- [x] No stale references after completion
- [x] Deviation log reviewed — R13 deferral justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4, 1→5, 6 last)
- [x] All contracts mapped to tasks
- [x] No golden dataset update needed (pure refactor)
- [x] Construction site audit: ~60 sites identified and covered by Tasks 2-4

**Antipattern rules:**
- [x] R1: No silent data loss (no error paths added)
- [x] R2: No map iteration changes
- [x] R3: No new CLI flags
- [x] R4: This PR IS the R4 fix — canonical constructors added
- [x] R5: No resource allocation changes
- [x] R6: No logrus.Fatalf in sim/ (constructors don't validate)
- [x] R7: No new golden tests
- [x] R8: No exported maps
- [x] R14: This PR IS the R14 fix — Step() decomposed

---

## Appendix: File-Level Implementation Details

### File: `sim/config.go`

**Purpose:** Add 6 canonical constructors after their respective type definitions.

**Constructor signatures (parameter order matches struct field order):**

| Constructor | Parameters | Return |
|---|---|---|
| `NewKVCacheConfig` | `(totalKVBlocks, blockSizeTokens, kvCPUBlocks int64, kvOffloadThreshold, kvTransferBandwidth float64, kvTransferBaseLatency int64)` | `KVCacheConfig` |
| `NewBatchConfig` | `(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64)` | `BatchConfig` |
| `NewLatencyCoeffs` | `(betaCoeffs, alphaCoeffs []float64)` | `LatencyCoeffs` |
| `NewModelHardwareConfig` | `(modelConfig ModelConfig, hwConfig HardwareCalib, model, gpu string, tp int, roofline bool)` | `ModelHardwareConfig` |
| `NewPolicyConfig` | `(priorityPolicy, scheduler string)` | `PolicyConfig` |
| `NewWorkloadConfig` | `(guideLLMConfig *GuideLLMConfig, tracesWorkloadFilePath string)` | `WorkloadConfig` |

### File: `sim/simulator.go`

**Purpose:** Decompose `Step()` into 4 named phase methods.

**Method signatures:**

| Method | Signature | Phase |
|---|---|---|
| `Step` | `(sim *Simulator) Step(now int64)` | Orchestrator (4 lines) |
| `scheduleBatch` | `(sim *Simulator) scheduleBatch(now int64)` | Phase 1: priority, reorder, batch formation, events |
| `executeBatchStep` | `(sim *Simulator) executeBatchStep(now int64) int64` | Phase 2: prefill/decode execution, returns step advance |
| `processCompletions` | `(sim *Simulator) processCompletions(now, currStepAdvance int64) []*Request` | Phase 3: completion processing, returns remaining |
| `scheduleNextStep` | `(sim *Simulator) scheduleNextStep(now, currStepAdvance int64, remaining []*Request)` | Phase 4: next step scheduling |

**Behavioral note:** Phase 3 (`processCompletions`) MUST be called after Phase 2 (`executeBatchStep`). The two-pass design is documented in the method's doc comment and enforced by the caller order in `Step()`.

### File: `docs/templates/design-guidelines.md`

**Purpose:** Correct 3 stale entries.

| Section | Before | After |
|---|---|---|
| §4.2 KVStore row | `KVStore` (9 methods) | `KVStore` (11 methods) |
| §4.5 config parameter row | 5-6 files (exceeds) | 2 files (meets target — post-#381) |
| §6.2 Monolith Method row | 134 lines | Decomposed into named phase methods |
