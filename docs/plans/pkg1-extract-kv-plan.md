# PKG-1: Extract sim/kv/ + sim/internal/ Utilities — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move KV cache implementations out of the flat `sim/` package into a dedicated `sim/kv/` sub-package, establishing the pattern for future module extractions.

**The problem today:** The `sim/` package contains 26 source files (~5,800 LOC) covering ~10 modules in a single flat namespace. The KV cache module (4 files, ~850 LOC) has a clean interface boundary (`KVStore`, 11 methods) but shares a package with unrelated modules like routing, scheduling, and metrics. This prevents Go's import system from enforcing module boundaries — any file in `sim/` can reach into KV cache internals.

**What this PR adds:**
1. **`sim/kv/` package** — KV cache implementations (`KVCacheState`, `TieredKVCache`) in their own package with compile-time boundary enforcement. (`PrefixCacheIndex` stays in `sim/` — used by routing code, no KV type dependencies)
2. **`sim/internal/hash/` package** — shared SHA256 hashing utilities (`HashTokens`, `HashBlock`, `ComputeBlockHashes`) used by both KV cache and routing, behind Go's `internal/` visibility constraint
3. **`sim/internal/util/` package** — shared generic utility (`Len64[T]`) used by 4+ files across the codebase
4. **Constructor injection** — `NewSimulator` accepts pre-built `KVStore` and `LatencyModel` instead of creating them internally, enabling the one-directional import pattern (`sim/kv/` → `sim/`, never reverse)

**Why this matters:** This is the foundation PR for the package decomposition (#404). It establishes the extraction pattern (interfaces in `sim/`, implementations in sub-packages, constructor injection) that PKG-2 (`sim/latency/`) will follow, and that v4 modules (`sim/infra/`, `sim/autoscale/`) will adopt.

**Architecture:** `sim/kv/` imports `sim/` for `Request`, `KVCacheConfig`, and the `KVStore` interface. `sim/` NEVER imports `sim/kv/` — the caller (`sim/cluster/instance.go`) imports both and wires them via constructor injection. `cmd/root.go` calls `cluster.NewClusterSimulator` which handles KV store creation internally. Hash utilities live in `sim/internal/hash/` (importable by any `sim/` subtree package, not externally). `Len64` lives in `sim/internal/util/`.

**Source:** GitHub issues #404 (tracking), #407 (this PR)

**Closes:** Fixes #407

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR extracts the KV cache module (4 source files, 4 test files) from the flat `sim/` package into `sim/kv/`. It also creates two shared internal utility packages (`sim/internal/hash/`, `sim/internal/util/`) and widens the `NewSimulator` constructor to accept pre-built dependencies (constructor injection pattern).

**Where it fits:** This is the first of 3 PRs in the package decomposition (#404). PKG-2 (`sim/latency/`) depends on `sim/internal/util/` created here. PKG-4 (cleanup) updates documentation after both extractions.

**Adjacent blocks:** `sim/simulator.go` (uses `KVStore` interface), `sim/batch_formation.go` (uses `KVStore` in `BatchContext`), `sim/routing.go` (uses hash functions for prefix affinity), `sim/cluster/instance.go` (constructs instances with KV stores).

**DEVIATION:** Issue #407 originally said `NewKVStore` factory moves to `sim/kv/factory.go`. After Phase 0 analysis, `NewKVStore` stays in `sim/kv_store.go` (package `sim`) because the `KVStore` interface is defined there and must stay in `sim/`. Only implementations and their constructors (`NewKVCacheState`, `NewTieredKVCache`) move to `sim/kv/`. The `NewKVStore` factory dispatches to `kv.NewKVCacheState` or `kv.NewTieredKVCache` — but this means `sim/` imports `sim/kv/`, creating a cycle. Resolution: eliminate the `NewKVStore` factory entirely. The caller constructs the KV store directly using `kv.NewKVCacheState` or `kv.NewTieredKVCache` and passes it to `NewSimulator`.

### B) Behavioral Contracts

**Positive Contracts (what MUST happen):**

**BC-1: Zero behavioral change**
- GIVEN any SimConfig that produced results before this PR
- WHEN the same simulation is run with the same seed after this PR
- THEN stdout output is byte-identical to before
- MECHANISM: Only import paths and constructor call sites change; no logic changes

**BC-2: KV cache conservation (INV-4)**
- GIVEN any simulation run
- WHEN the simulation completes
- THEN `allocated_blocks + free_blocks = total_blocks` holds
- MECHANISM: `KVCacheState.AllocateKVBlocks` and `ReleaseKVBlocks` preserve conservation (unchanged logic)

**BC-3: Hash consistency**
- GIVEN the same token sequence
- WHEN hashed by `sim/internal/hash/HashTokens` from any call site (KV cache or routing)
- THEN the hash value is identical
- MECHANISM: Single source of truth in `sim/internal/hash/` — no duplication

**BC-4: Import direction is acyclic**
- GIVEN the package dependency graph
- WHEN `go build ./...` runs
- THEN compilation succeeds with no import cycle errors
- MECHANISM: `sim/kv/` imports `sim/` (for Request, KVCacheConfig). `sim/` never imports `sim/kv/`. Callers import both.

**Negative Contracts (what MUST NOT happen):**

**BC-5: No new public API beyond package boundaries**
- GIVEN the `sim/internal/hash/` and `sim/internal/util/` packages
- WHEN code outside the `sim/` module tree attempts to import them
- THEN Go refuses to compile
- MECHANISM: Go's `internal/` package visibility enforcement

**BC-6: No test behavior change**
- GIVEN the full test suite
- WHEN `go test ./...` runs before and after this PR
- THEN the same tests pass with the same results
- MECHANISM: Tests move with their source files (same-package access preserved)

**Error Handling Contracts:**

**BC-7: Factory validation preserved**
- GIVEN invalid KV cache config (zero blocks, zero block size)
- WHEN `kv.NewKVCacheState` is called
- THEN it panics with the same error message as before
- MECHANISM: Validation logic moves unchanged to `sim/kv/`

### C) Component Interaction

```
cmd/root.go ──creates──> kv.NewKVCacheState (or kv.NewTieredKVCache)
    |                         |
    |                         v
    |                    sim.KVStore (interface, stays in sim/)
    |                         |
    +──passes to──> sim.NewSimulator(cfg, kvStore, latencyModel)
                              |
                         sim.Simulator
                              |
                    uses KVStore interface for:
                    - AllocateKVBlocks (batch formation)
                    - ReleaseKVBlocks (completion)
                    - BlockSize, UsedBlocks, TotalCapacity (metrics)
                    - SetClock, ConsumePendingTransferLatency (tiered)
```

**State ownership:** No change. `KVCacheState` owns block allocation state. `TieredKVCache` owns GPU+CPU tier state. `Simulator` holds a `KVStore` interface field.

**Extension friction:** Adding a new KV cache tier: 2 files in `sim/kv/` (implementation + test). Same as current.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| #407: `NewKVStore` factory moves to `sim/kv/factory.go` | `NewKVStore` eliminated; callers construct directly | CORRECTION: Moving factory to `sim/kv/` while `KVStore` interface stays in `sim/` creates import cycle (`sim/` would import `sim/kv/` for factory). Constructor injection eliminates the factory. |
| #404: `sim/cluster/` imports `sim/kv/` directly for instance construction | `NewInstanceSimulator` creates KV store internally, importing `sim/kv/` | SIMPLIFICATION: Only `instance.go` and `cmd/root.go` need to import `sim/kv/`; cluster test calls are unaffected. |
| #407: `hashTokens`/`hashBlock` exported from `sim/kv/` | Moved to `sim/internal/hash/` | CORRECTION: Hash functions have no KV-specific types — they take `[]int` and `string`. `internal/hash/` is cleaner and avoids `sim/` importing `sim/kv/`. |

### E) Review Guide

1. **THE TRICKY PART:** The import cycle avoidance. Verify that `sim/` has ZERO imports of `sim/kv/` after the PR. The only packages importing `sim/kv/` should be `cmd/root.go`, `sim/cluster/instance.go`, and test files.
2. **WHAT TO SCRUTINIZE:** BC-4 (acyclic imports) — grep for `"sim/kv"` in `sim/*.go` non-test files. Must be zero. BC-3 (hash consistency) — verify `sim/internal/hash/` is the single source.
3. **WHAT'S SAFE TO SKIM:** File moves (kvcache.go → kv/cache.go etc.) are mechanical. Test file moves are mechanical.
4. **KNOWN DEBT:** `KVStore` has 11 methods (#246). `KVCacheState` has exported fields (`RequestMap`, `HashToBlock`, `FreeHead`) that should arguably be unexported. Not addressed in this PR — scope is extraction only.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/internal/util/util.go` — `Len64[T]` generic helper
- `sim/internal/hash/hash.go` — `HashTokens`, `HashBlock`, `ComputeBlockHashes`
- `sim/kv/cache.go` — `KVCacheState` + `NewKVCacheState` (moved from `sim/kvcache.go`; add validation panics for zero blocks/blockSize)
- `sim/kv/tiered.go` — `TieredKVCache` + `NewTieredKVCache` (moved from `sim/kvcache_tiered.go`)

**Files to modify:**
- `sim/kv_store.go` — Remove `NewKVStore` factory; keep `KVStore` interface only
- `sim/simulator.go` — Widen `NewSimulator` to accept `kvStore KVStore, latencyModel LatencyModel`; remove lines 155-160 (TotalKVBlocks/BlockSizeTokens panics — now validated by `kv.NewKVCacheState`)
- `sim/kvcache.go` — Delete (contents moved to `sim/kv/cache.go`)
- `sim/kvcache_tiered.go` — Delete (contents moved to `sim/kv/tiered.go`)
- `sim/routing.go` — Update hash function calls to use `hash.HashTokens`, `hash.HashBlock`, `hash.ComputeBlockHashes`
- `sim/prefix_cache_index.go` — Update `hashBlock` calls to `hash.HashBlock` (file STAYS in `sim/`)
- `sim/batch_formation.go` — Update `Len64` to `util.Len64`
- `sim/latency_model.go` — Update `Len64` to `util.Len64`
- `sim/cluster/instance.go` — Import `sim/kv`, create KV store (single-tier or tiered based on config) before calling `NewSimulator`
- `sim/kvcache_test.go` — Update `hashTokens` calls to `hash.HashTokens` (Task 3), then move to `sim/kv/cache_test.go` (Task 5)
- `sim/routing_test.go` — Update `hashTokens` calls to `hash.HashTokens` (Task 3)
- `sim/batch_formation_test.go` — Replace 8 `NewKVStore(...)` calls with `NewKVCacheState(...)` (Task 4)
- `sim/simulator_preempt_test.go` — Replace 2 `NewKVStore(...)` calls with `NewKVCacheState(...)` (Task 4)

NOTE: `cmd/root.go` does NOT need changes — it calls `cluster.NewClusterSimulator` which handles KV construction internally via `NewInstanceSimulator`.

NOTE: `PrefixCacheIndex` and `prefix_cache_index.go` STAY in `sim/` — they are used by routing code and have no KV type dependencies. Moving them would create an import cycle.

**Files to move (tests):**
- `sim/kvcache_test.go` → `sim/kv/cache_test.go` (after Task 3 hash updates)
- `sim/kvcache_tiered_test.go` → `sim/kv/tiered_test.go`
- `sim/kv_store_test.go` → `sim/kv/store_test.go` (ADAPT: delete `NewKVStore` tests, update `KVStore` to `sim.KVStore`, update `*Request` to `*sim.Request`, update `NewKVCacheConfig` to `sim.NewKVCacheConfig`)

**Key decisions:**
- `NewKVStore` factory eliminated (constructor injection instead)
- `Len64` to `sim/internal/util/` (not duplicated)
- Hash functions to `sim/internal/hash/` (not duplicated, not in `sim/kv/`)
- Option A for blast radius: widen only `NewSimulator`; `NewInstanceSimulator` creates KV+latency internally

**No dead code.** Every file, type, and function is exercisable immediately.

### G) Task Breakdown

---

### Task 1: Create `sim/internal/util/` with `Len64`

**Contracts Implemented:** BC-5 (internal visibility)

**Files:**
- Create: `sim/internal/util/util.go`
- Create: `sim/internal/util/util_test.go`

**Step 1: Write failing test**

Context: `Len64` is a trivial generic but we verify it works in the new package.

```go
// sim/internal/util/util_test.go
package util

import "testing"

func TestLen64_IntSlice(t *testing.T) {
	got := Len64([]int{1, 2, 3})
	if got != 3 {
		t.Errorf("Len64([]int{1,2,3}) = %d, want 3", got)
	}
}

func TestLen64_EmptySlice(t *testing.T) {
	got := Len64([]int{})
	if got != 0 {
		t.Errorf("Len64([]int{}) = %d, want 0", got)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/internal/util/... -v`
Expected: FAIL (package does not exist yet)

**Step 3: Implement**

```go
// sim/internal/util/util.go
package util

// Len64 returns the length of a slice as int64.
func Len64[T any](v []T) int64 { return int64(len(v)) }
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/internal/util/... -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/internal/util/...`
Expected: No issues

**Step 6: Commit**

```bash
git add sim/internal/util/
git commit -m "refactor(sim): extract Len64 to sim/internal/util (BC-5)

- Create sim/internal/util/ package with Len64[T] generic helper
- Internal visibility prevents external import

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create `sim/internal/hash/` with hash functions

**Contracts Implemented:** BC-3 (hash consistency), BC-5 (internal visibility)

**Files:**
- Create: `sim/internal/hash/hash.go`
- Create: `sim/internal/hash/hash_test.go`

**Step 1: Write failing test**

Context: Verify hash functions produce deterministic output for known inputs.

```go
// sim/internal/hash/hash_test.go
package hash

import "testing"

func TestHashTokens_Deterministic(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5}
	h1 := HashTokens(tokens)
	h2 := HashTokens(tokens)
	if h1 != h2 {
		t.Errorf("HashTokens not deterministic: %q != %q", h1, h2)
	}
	if h1 == "" {
		t.Error("HashTokens returned empty string")
	}
}

func TestHashBlock_ChainsDeterministically(t *testing.T) {
	tokens := []int{10, 20, 30}
	h1 := HashBlock("", tokens)
	h2 := HashBlock("", tokens)
	if h1 != h2 {
		t.Errorf("HashBlock not deterministic: %q != %q", h1, h2)
	}
	// Different prev hash produces different result
	h3 := HashBlock("abc", tokens)
	if h1 == h3 {
		t.Error("HashBlock should produce different result with different prevHash")
	}
}

func TestComputeBlockHashes_MatchesManualChaining(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	blockSize := 4
	hashes := ComputeBlockHashes(blockSize, tokens)
	if len(hashes) != 2 {
		t.Fatalf("expected 2 block hashes, got %d", len(hashes))
	}
	// Manual verification: first block hashed with empty prev, second with first's hash
	manual0 := HashBlock("", tokens[0:4])
	manual1 := HashBlock(manual0, tokens[4:8])
	if hashes[0] != manual0 {
		t.Errorf("block 0: got %q, want %q", hashes[0], manual0)
	}
	if hashes[1] != manual1 {
		t.Errorf("block 1: got %q, want %q", hashes[1], manual1)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/internal/hash/... -v`
Expected: FAIL

**Step 3: Implement**

Move `hashTokens` from `sim/kvcache.go:119-136`, `hashBlock` from `sim/prefix_cache_index.go:69-77`, and `computeBlockHashes` from `sim/routing.go:248-262` into a single file:

```go
// sim/internal/hash/hash.go
package hash

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

// HashTokens computes a SHA256 hash of a token sequence.
// Used for KV cache prefix matching and routing prefix affinity.
func HashTokens(tokens []int) string {
	var sb strings.Builder
	for i, t := range tokens {
		if i > 0 {
			sb.WriteString("|")
		}
		sb.WriteString(strconv.Itoa(t))
	}
	h := sha256.New()
	h.Write([]byte(sb.String()))
	return hex.EncodeToString(h.Sum(nil))
}

// HashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
// This creates hierarchical block hashes for prefix caching.
// IMPORTANT: byte format must match original prefix_cache_index.go:69-77 exactly:
// prevHash + "token1" + "|" + "token2" + "|" + ... (pipe AFTER each token)
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	for _, t := range tokens {
		h.Write([]byte(strconv.Itoa(t)))
		h.Write([]byte("|"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each hash chains with the previous block's hash, enabling prefix matching.
func ComputeBlockHashes(blockSize int, tokens []int) []string {
	numBlocks := len(tokens) / blockSize
	if numBlocks == 0 {
		return nil
	}
	hashes := make([]string, numBlocks)
	prevHash := ""
	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		hashes[i] = HashBlock(prevHash, tokens[start:end])
		prevHash = hashes[i]
	}
	return hashes
}
```

**Step 4: Run test**

Run: `go test ./sim/internal/hash/... -v`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/internal/hash/...`
Expected: No issues

**Step 6: Commit**

```bash
git add sim/internal/hash/
git commit -m "refactor(sim): extract hash functions to sim/internal/hash (BC-3, BC-5)

- Move hashTokens, hashBlock, computeBlockHashes to shared internal package
- Single source of truth for KV cache and routing prefix hashing
- Internal visibility prevents external import

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update `sim/` to use internal packages (`Len64`, hash functions)

**Contracts Implemented:** BC-1 (zero behavioral change), BC-3 (hash consistency)

**Files:**
- Modify: `sim/kvcache.go` — replace `Len64` definition with import, replace `hashTokens` with `hash.HashTokens`
- Modify: `sim/prefix_cache_index.go` — replace `hashBlock` with `hash.HashBlock`
- Modify: `sim/routing.go` — replace `hashTokens`, `hashBlock`, `computeBlockHashes` with hash package calls
- Modify: `sim/batch_formation.go` — replace `Len64` with `util.Len64`
- Modify: `sim/simulator.go` — replace `Len64` with `util.Len64`
- Modify: `sim/latency_model.go` — replace `Len64` with `util.Len64`

**Step 1: No new test** — this is a mechanical refactor. Existing tests verify behavior.

**Step 2: Make changes to source files**

In `sim/kvcache.go`:
- Remove `Len64` function definition (lines 45-47)
- Remove `hashTokens` function definition (lines 119-136)
- Add imports: `"github.com/inference-sim/inference-sim/sim/internal/hash"` and `"github.com/inference-sim/inference-sim/sim/internal/util"`
- Replace all `Len64(` with `util.Len64(` (16 call sites)
- Replace all `hashTokens(` with `hash.HashTokens(` (3 call sites: lines 145, 225, 270)

In `sim/prefix_cache_index.go`:
- Remove `hashBlock` function definition (lines 69-77)
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/hash"`
- Replace `hashBlock(` with `hash.HashBlock(` (1 call site: line 62)

In `sim/routing.go`:
- Remove `computeBlockHashes` function definition (lines 248-262)
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/hash"`
- Replace `hashTokens(` with `hash.HashTokens(` (1 call site: line 223)
- Replace `computeBlockHashes(` with `hash.ComputeBlockHashes(` (1 call site: line 217)

In `sim/batch_formation.go`:
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/util"`
- Replace `Len64(` with `util.Len64(` (5 call sites)

In `sim/simulator.go`:
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/util"`
- Replace `Len64(` with `util.Len64(` (3 call sites: lines 442, 452, 480)

In `sim/latency_model.go`:
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/util"`
- Replace `Len64(` with `util.Len64(` (2 call sites: lines 41, 90)

**Step 3: Update TEST files that use removed functions**

In `sim/kvcache_test.go`:
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/hash"`
- Replace `hashTokens(` with `hash.HashTokens(` (3 call sites: lines 82, 102, 103)

In `sim/routing_test.go`:
- Add import: `"github.com/inference-sim/inference-sim/sim/internal/hash"`
- Replace `hashTokens(` with `hash.HashTokens(` (3 call sites: lines 463, 464, 470)

**Step 4: Build and test**

Run: `go build ./sim/...`
Expected: PASS

Run: `go test ./sim/... -count=1`
Expected: All tests pass (same count as before)

**Step 5: Lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/kvcache.go sim/prefix_cache_index.go sim/routing.go sim/batch_formation.go sim/simulator.go sim/latency_model.go sim/kvcache_test.go sim/routing_test.go
git commit -m "refactor(sim): use internal/util and internal/hash packages (BC-1, BC-3)

- Replace inline Len64, hashTokens, hashBlock, computeBlockHashes
  with imports from sim/internal/util and sim/internal/hash
- Update test files that called removed unexported functions
- Zero behavioral change — all existing tests pass

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Widen `NewSimulator` constructor (constructor injection)

**Contracts Implemented:** BC-4 (acyclic imports), BC-7 (factory validation preserved)

**Files:**
- Modify: `sim/simulator.go` — change `NewSimulator` signature
- Modify: `sim/kv_store.go` — remove `NewKVStore` factory
- Modify: `sim/simulator_test.go` — update `mustNewSimulator` helper
- Modify: `sim/cluster/instance.go` — create KV store + latency model before calling NewSimulator
- Modify: `cmd/root.go` — no change needed (calls NewClusterSimulator, not NewSimulator directly)

**Step 1: No new test** — existing tests validate behavior through updated call sites.

**Step 2: Widen NewSimulator**

In `sim/simulator.go`, change the constructor from:
```go
func NewSimulator(cfg SimConfig) (*Simulator, error) {
    latencyModel, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
    if err != nil {
        return nil, fmt.Errorf("NewSimulator: %w", err)
    }
    batchFormation := NewBatchFormation(latencyModel)
    // ...
    KVCache: NewKVStore(cfg.KVCacheConfig),
```

To:
```go
func NewSimulator(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*Simulator, error) {
    if kvStore == nil {
        return nil, fmt.Errorf("NewSimulator: kvStore must not be nil")
    }
    if latencyModel == nil {
        return nil, fmt.Errorf("NewSimulator: latencyModel must not be nil")
    }
    batchFormation := NewBatchFormation(latencyModel)
    // ...
    KVCache: kvStore,
```

Remove `NewLatencyModel` and `NewKVStore` calls from inside `NewSimulator`. `NewBatchFormation(latencyModel)` stays internal.

In `sim/kv_store.go`, remove the `NewKVStore` factory function (lines 24-38). Keep the `KVStore` interface definition only.

In `sim/simulator.go`, also remove the TotalKVBlocks/BlockSizeTokens validation panics at lines 155-160 (now validated by `NewKVCacheState` constructor).

In `sim/batch_formation_test.go`, replace 8 calls to `NewKVStore(cfg.KVCacheConfig)` with `NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)`. Also replace `NewKVStore(NewKVCacheConfig(...))` at line 414 with `NewKVCacheState(4, 4)`.

In `sim/simulator_preempt_test.go`, replace 2 calls to `NewKVStore(config.KVCacheConfig)` with `NewKVCacheState(config.TotalKVBlocks, config.BlockSizeTokens)`.

In `sim/simulator_test.go`, update `mustNewSimulator`:
```go
func mustNewSimulator(t *testing.T, cfg SimConfig) *Simulator {
    t.Helper()
    kvStore := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
    latencyModel, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
    if err != nil {
        t.Fatalf("NewLatencyModel: %v", err)
    }
    s, err := NewSimulator(cfg, kvStore, latencyModel)
    if err != nil {
        t.Fatalf("NewSimulator: %v", err)
    }
    return s
}
```

Note: For now, `NewKVCacheState` and `NewLatencyModel` are still in `sim/`. They will move to sub-packages in later tasks (KV) and PKG-2 (latency).

Update all direct `NewSimulator(cfg)` calls in test files to pass KV store and latency model. There are ~8 direct calls outside the helper (in `simulator_test.go` and `model_hardware_config_test.go`).

In `sim/cluster/instance.go`, update `NewInstanceSimulator` to construct KV store (replicating the dispatch logic from the removed `NewKVStore`):
```go
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
    // Create KV store (single-tier or tiered based on config)
    gpu := sim.NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
    var kvStore sim.KVStore = gpu
    if cfg.KVCPUBlocks > 0 {
        kvStore = sim.NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold, cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
    }
    latencyModel, err := sim.NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
    if err != nil {
        panic(fmt.Sprintf("NewInstanceSimulator(%s): NewLatencyModel: %v", id, err))
    }
    s, err := sim.NewSimulator(cfg, kvStore, latencyModel)
```

Note: `NewKVCacheState` and `NewTieredKVCache` are still in `sim/` at this point. Task 5 will move them to `sim/kv/`.

Add validation panics to `NewKVCacheState` (currently has none — the removed `NewKVStore` validated these):
```go
// In sim/kvcache.go NewKVCacheState, add at the top:
if totalBlocks <= 0 {
    panic(fmt.Sprintf("KVStore: TotalKVBlocks must be > 0, got %d", totalBlocks))
}
if blockSizeTokens <= 0 {
    panic(fmt.Sprintf("KVStore: BlockSizeTokens must be > 0, got %d", blockSizeTokens))
}
```

Also update the direct `sim.NewSimulator` call in `sim/cluster/cluster_test.go:864`.

**Step 3: Build and test**

Run: `go build ./...`
Expected: PASS

Run: `go test ./... -count=1`
Expected: All tests pass

**Step 4: Lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator.go sim/kv_store.go sim/simulator_test.go sim/model_hardware_config_test.go sim/cluster/instance.go sim/cluster/cluster_test.go sim/simulator_preempt_test.go
git commit -m "refactor(sim): constructor injection for NewSimulator (BC-4, BC-7)

- Widen NewSimulator to accept kvStore and latencyModel parameters
- Remove NewKVStore factory (callers construct directly)
- Update mustNewSimulator helper and all direct call sites
- NewInstanceSimulator creates KV store and latency model internally
- Enables one-directional imports for package extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Move KV cache files to `sim/kv/`

**Contracts Implemented:** BC-1 (zero behavioral change), BC-2 (conservation), BC-4 (acyclic imports), BC-6 (test behavior preserved)

**Files:**
- Create: `sim/kv/cache.go` (from `sim/kvcache.go`)
- Create: `sim/kv/tiered.go` (from `sim/kvcache_tiered.go`)
- Delete: `sim/kvcache.go`, `sim/kvcache_tiered.go`
- Move: `sim/kvcache_test.go` → `sim/kv/cache_test.go`
- Move: `sim/kvcache_tiered_test.go` → `sim/kv/tiered_test.go`
- Move+Adapt: `sim/kv_store_test.go` → `sim/kv/store_test.go` (delete NewKVStore tests, update type references to sim.KVStore/sim.Request/sim.NewKVCacheConfig)
- Modify: `sim/cluster/instance.go` — change `sim.NewKVCacheState` to `kv.NewKVCacheState`, `sim.NewTieredKVCache` to `kv.NewTieredKVCache`
- NOTE: `prefix_cache_index.go` and `prefix_cache_index_test.go` STAY in `sim/` (no change)
- NOTE: `batch_formation_test.go` does NOT need a stub — `BlackboxLatencyModel` stays in package `sim` (defer stub to PKG-2)

**Step 1: Move source files**

For each source file (`kvcache.go`, `kvcache_tiered.go`, `prefix_cache_index.go`):
1. Change `package sim` to `package kv`
2. Add import `"github.com/inference-sim/inference-sim/sim"` for `*sim.Request`, `sim.KVCacheConfig`
3. Replace `*Request` with `*sim.Request` in method signatures
4. Replace `hash.HashTokens` with just `hash.HashTokens` (already imported)
5. Replace `util.Len64` with just `util.Len64` (already imported)
6. Move to `sim/kv/` directory

For `kv_store.go`: The `KVStore` interface stays in `sim/kv_store.go` (package `sim`). Only the interface definition remains — the `NewKVStore` factory was removed in Task 4.

For test files:
1. Change `package sim` to `package kv`
2. Add import for `sim` package where `Request`, `KVCacheConfig`, etc. are used
3. Replace `NewKVCacheConfig(...)` with `sim.NewKVCacheConfig(...)` where needed
4. Move to `sim/kv/` directory

In `sim/cluster/instance.go`:
- Add import `"github.com/inference-sim/inference-sim/sim/kv"`
- Change `sim.NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)` to `kv.NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)`
- For tiered cache (when `cfg.KVCPUBlocks > 0`), also handle `kv.NewTieredKVCache` construction

In `sim/routing_prefix_scorer.go`:
- Add import `"github.com/inference-sim/inference-sim/sim/kv"`
- Change `NewPrefixCacheIndex(blockSize, defaultLRUCapacity)` to `kv.NewPrefixCacheIndex(blockSize, defaultLRUCapacity)`

In `sim/batch_formation_test.go:440`:
- Replace `&BlackboxLatencyModel{betaCoeffs: []float64{0, 0, 0}}` with a local test stub:
```go
type stubLatencyModel struct{}
func (s *stubLatencyModel) StepTime(_ []*Request) int64 { return 1000 }
func (s *stubLatencyModel) QueueingTime(_ *Request) int64 { return 0 }
func (s *stubLatencyModel) OutputTokenProcessingTime() int64 { return 0 }
func (s *stubLatencyModel) SchedulingProcessingTime() int64 { return 0 }
func (s *stubLatencyModel) PreemptionProcessingTime() int64 { return 0 }
```

**Step 2: Build**

Run: `go build ./...`
Expected: PASS (no import cycles)

**Step 3: Test**

Run: `go test ./... -count=1`
Expected: All tests pass

**Step 4: Verify import direction**

Run: `grep -r '"github.com/inference-sim/inference-sim/sim/kv"' sim/*.go | grep -v _test.go`
Expected: Only `sim/routing_prefix_scorer.go` (which imports kv for PrefixCacheIndex). `sim/simulator.go`, `sim/batch_formation.go` etc. must NOT import `sim/kv/`.

Wait — `sim/routing_prefix_scorer.go` is in package `sim`. If it imports `sim/kv/`, and `sim/kv/` imports `sim/`, that's a cycle!

**CORRECTION:** `PrefixCacheIndex` and `NewPrefixCacheIndex` must stay in `sim/` (they're used by routing code in `sim/`). Only `KVCacheState`, `TieredKVCache`, and their constructors move to `sim/kv/`. `PrefixCacheIndex` has NO dependency on `Request` or other KV types — it is self-contained. Moving it to `sim/kv/` would force `sim/` to import `sim/kv/`.

Revised plan: `prefix_cache_index.go` stays in `sim/`. Only 3 files move:
- `kvcache.go` → `sim/kv/cache.go`
- `kvcache_tiered.go` → `sim/kv/tiered.go`
- `kv_store_test.go` → `sim/kv/store_test.go` (tests the factory which now lives in kv/)

`prefix_cache_index.go` stays in `sim/` (no change needed for routing).

**Step 5: Lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/kv/ sim/kvcache.go sim/kvcache_tiered.go sim/cluster/instance.go sim/routing_prefix_scorer.go sim/batch_formation_test.go
git commit -m "refactor(sim): extract KV cache to sim/kv/ package (BC-1, BC-2, BC-4, BC-6)

- Move KVCacheState and TieredKVCache to sim/kv/
- KVStore interface stays in sim/ (consumed there)
- PrefixCacheIndex stays in sim/ (used by routing, no KV type deps)
- sim/kv/ imports sim/ for Request and KVCacheConfig (one-way)
- sim/ never imports sim/kv/ (verified by grep)
- Constructor injection: callers create KV store via kv.NewKVCacheState
- batch_formation_test uses stub LatencyModel instead of struct literal

Fixes #407

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final verification and determinism check

**Contracts Implemented:** BC-1 (byte-identical output), BC-6 (all tests pass)

**Step 1: Full build**

Run: `go build ./...`
Expected: PASS

**Step 2: Full test suite**

Run: `go test ./... -count=1`
Expected: All packages pass

**Step 3: Lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Determinism check**

Run:
```bash
go build -o blis main.go
./blis run --model meta-llama/llama-3.1-8b-instruct --seed 42 --num-requests 50 2>/dev/null > /tmp/pkg1_run1.json
./blis run --model meta-llama/llama-3.1-8b-instruct --seed 42 --num-requests 50 2>/dev/null > /tmp/pkg1_run2.json
diff /tmp/pkg1_run1.json /tmp/pkg1_run2.json
```
Expected: No diff (byte-identical)

**Step 5: Verify import graph**

Run: `grep -rn 'sim/kv' sim/*.go | grep -v _test.go | grep -v 'sim/kv_store.go'`
Expected: Only `sim/routing_prefix_scorer.go` if PrefixCacheIndex stayed. Actually — `routing_prefix_scorer.go` should NOT import `sim/kv/` (PrefixCacheIndex stays in sim/). So this grep should return ZERO results for non-test sim/ files.

Run: `grep -rn '"github.com/inference-sim/inference-sim/sim/kv"' sim/*.go | grep -v _test.go`
Expected: ZERO results — `sim/` never imports `sim/kv/`

**Step 6: Commit (if any cleanup needed)**

```bash
git add -A
git commit -m "refactor(sim): final verification for PKG-1 extraction

- All tests pass, lint clean, determinism verified
- Import graph verified: sim/ does not import sim/kv/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|---|---|---|---|
| BC-1 | Task 6 | Determinism | Manual: byte-identical output before/after |
| BC-2 | Task 5 | Invariant | Existing `TestKVConservation_AfterRun` (moved to kv/) |
| BC-3 | Task 2 | Unit | `TestHashTokens_Deterministic`, `TestComputeBlockHashes_MatchesManualChaining` |
| BC-4 | Task 5 | Build | `go build ./...` succeeds (no import cycles) |
| BC-5 | Task 1,2 | Build | `internal/` enforced by Go compiler |
| BC-6 | Task 5 | Integration | `go test ./... -count=1` — all tests pass |
| BC-7 | Task 5 | Failure | Existing `TestNewKVStore_InvalidConfig_Panics` (moved to kv/) |

**Golden dataset:** No output format changes. Golden dataset test continues to pass at new location. No regeneration needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| Import cycle discovered late | Low (design verified in 4 rounds) | High (blocks PR) | Grep for `sim/kv` in sim/*.go after each task | Task 5 |
| Test file move breaks white-box access | Low (same-package tests) | Medium | Tests use `package kv`, same as source | Task 5 |
| `batch_formation_test.go` stub doesn't match LatencyModel | Low | Medium | Stub implements all 5 interface methods | Task 5 |
| `Len64` or hash drift between packages | None (single source) | High | `sim/internal/` is the only definition | Task 1, 2 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (no new interfaces, just moving existing code)
- [x] No feature creep (extraction only, no behavioral changes)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without contract updates (NewSimulator signature widens — BC-4)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (sim/internal/testutil stays)
- [x] CLAUDE.md updated (PKG-4 cleanup PR handles this)
- [x] Deviation log reviewed — 3 deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed
- [x] Construction site audit: `NewSimulator` call sites enumerated (8 direct + helper covering ~20)

**Antipattern rules:**
- [x] R1: No silent data loss (no error path changes)
- [x] R2: No new map iteration
- [x] R3: No new CLI flags
- [x] R4: `NewSimulator` signature change — all 10 call sites updated in Task 4
- [x] R5: No new resource allocation loops
- [x] R6: No `logrus.Fatalf` in sim/kv/ (only panic for invariant violations)
- [x] R7: No new golden tests (existing invariant tests preserved)
- [x] R8: No exported mutable maps in sim/kv/
- [x] R9: No new YAML fields
- [x] R10: No YAML parsing changes
- [x] R11: No new division
- [x] R12: No golden dataset changes
- [x] R13: No new interfaces (KVStore preserved, 2 implementations)
- [x] R14: No multi-module methods
- [x] R15: No stale PR references (grep after completion)
- [x] R16: Config grouping preserved
- [x] R17: No routing signal changes

---

## APPENDIX: File-Level Implementation Details

See individual task steps above for complete code. Key behavioral notes:

**sim/kv/cache.go:**
- `package kv` (not `package sim`)
- Imports `"github.com/inference-sim/inference-sim/sim"` for `*sim.Request`
- `AllocateKVBlocks(req *sim.Request, ...)` — parameter type changes from `*Request` to `*sim.Request`
- All internal unexported methods (`popFreeBlock`, `appendToFreeList`, etc.) stay unexported in package `kv`
- `TieredKVCache` accesses `KVCacheState` internals (unexported methods) — both in same package, access preserved

**sim/kv/tiered.go:**
- Same package `kv` — can access `KVCacheState` unexported fields/methods
- `gpu *KVCacheState` field type unchanged (same package)

**sim/kv_store.go (stays in sim/):**
- `KVStore` interface definition only
- `NewKVStore` factory REMOVED (callers construct directly)
- No import of `sim/kv/`

**sim/simulator.go:**
- `NewSimulator(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*Simulator, error)`
- nil checks for both injected dependencies
- `NewBatchFormation(latencyModel)` still called internally
- No import of `sim/kv/` or `sim/latency/`

**sim/cluster/instance.go:**
- Imports `"github.com/inference-sim/inference-sim/sim/kv"`
- Creates KV store: `kvStore := kv.NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)` (or `kv.NewTieredKVCache(...)` when CPU blocks > 0)
- Creates latency model: `latencyModel, err := sim.NewLatencyModel(...)` (stays in sim/ until PKG-2)
- Passes both to `sim.NewSimulator(cfg, kvStore, latencyModel)`
