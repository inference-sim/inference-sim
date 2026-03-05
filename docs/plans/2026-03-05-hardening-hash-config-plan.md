# Hardening Batch: Dead Code Removal, Hash Performance, BatchConfig Validation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead hashing code, optimize hash hot-path allocations for reasoning workloads, and close the BatchConfig library-level validation gap.

**The problem today:** (1) `HashTokens` is dead code — zero production callers after #537 migrated to hierarchical `HashBlock`. It uses an incompatible format (pipe-between vs pipe-after) which is confusing to keep. (2) `HashBlock` and `ComputeBlockHashes` allocate a new string per token via `strconv.Itoa`, creating thousands of small allocations per request for reasoning workloads (20K+ tokens). No benchmarks exist to measure this. (3) `NewBatchConfig(0, 0, 0)` silently produces an invalid config — library callers bypass CLI validation (R3).

**What this PR adds:**
1. Dead code removal — delete `HashTokens` and its test (zero callers, incompatible format)
2. Hash benchmarks — `BenchmarkHashBlock` and `BenchmarkComputeBlockHashes` for baseline measurement
3. Hash allocation optimization — replace per-token `strconv.Itoa` + `[]byte(...)` with stack-buffered `strconv.AppendInt`, eliminating per-token strconv allocations
4. Constructor validation — `NewBatchConfig` panics on invalid values, matching the `NewKVCacheState` pattern

**Why this matters:** Items 1-3 improve maintainability and performance for the KV cache hashing hot path, which is called on every routing decision and every KV allocation. Item 4 closes a library-safety gap — R3 requires validation at library constructors, not just CLI flags.

**Architecture:** All changes are in existing files — `sim/internal/hash/hash.go` (items 1-3), `sim/internal/hash/hash_test.go` (items 1-2), `sim/config.go` (item 4), `sim/config_test.go` (item 4), `sim/simulator_test.go` (item 4 test fix). No new types, interfaces, or packages.

**Source:** GitHub issues #538, #542, #539, #382

**Closes:** Fixes #538, fixes #542, fixes #539, fixes #382

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

1. **Building blocks modified:** Hash utilities (`sim/internal/hash/`), sub-config constructors (`sim/config.go`), test infrastructure
2. **Adjacent blocks:** `sim/kv/cache.go` (calls `HashBlock`), `sim/prefix_cache_index.go` (calls `ComputeBlockHashes`), `sim/simulator.go` (consumes `BatchConfig` — already validates)
3. **Invariants touched:** None — hash optimization produces byte-identical hashes (same bytes written to SHA256). BatchConfig validation is pre-construction, not runtime.
4. **Construction Site Audit:**
   - `NewBatchConfig` — 1 production site (`cmd/root.go:685`), ~60 test sites (all use valid values except `TestNewSimulator_BatchConfigValidation` which uses zero/negative values to test error paths)
   - No struct field additions — only adding validation logic to existing constructor

**Confirmed facts:**
- `HashTokens` has zero non-test callers (confirmed via grep)
- `HashBlock` format: `prevHash` bytes then `"tokenN|"` per token (pipe AFTER)
- `HashTokens` format: `"token1|token2|token3"` (pipe BETWEEN) — incompatible
- `ComputeBlockHashes` inlines `HashBlock` logic for hasher reuse (line 71 comment)
- `NewBatchConfig` at `sim/config.go:38-44` — pure value constructor, no validation
- `NewSimulator` at `sim/simulator.go:94-102` — already validates MaxRunningReqs > 0, MaxScheduledTokens > 0, LongPrefillTokenThreshold >= 0
- `TestNewSimulator_BatchConfigValidation` at `sim/simulator_test.go:247-279` — tests zero/negative values via `NewBatchConfig`. Must update to use struct literals to avoid constructor panic.
- `NewKVCacheState` panics on invalid inputs (`sim/kv/cache.go:52-55`) — this is the pattern to match

---

## Part 1: Design Validation

### A) Executive Summary

Four independent hardening items bundled into one PR because they're all small, library-internal changes with no user-facing behavior change and no golden dataset impact:

1. **Dead code removal**: Delete unused `HashTokens` function and its test
2. **Benchmarks**: Add hash hot-path benchmarks to enable future performance measurement
3. **Allocation reduction**: Replace per-token string allocations with a reusable stack buffer
4. **Constructor validation**: Make `NewBatchConfig` reject invalid values at construction time

No runtime behavior changes for valid inputs. Hash outputs are byte-identical (same bytes written to SHA256 hasher). No golden dataset regeneration needed.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: HashBlock produces identical output after optimization
- GIVEN any prevHash string and token slice
- WHEN HashBlock is called with the optimized implementation
- THEN the returned hash string is identical to the pre-optimization implementation
- MECHANISM: Same byte sequence written to SHA256 hasher; only allocation strategy changes
```

```
BC-2: ComputeBlockHashes produces identical output after optimization
- GIVEN any blockSize and token slice
- WHEN ComputeBlockHashes is called with the optimized implementation
- THEN each returned hash string is identical to calling HashBlock sequentially
- MECHANISM: Existing TestComputeBlockHashes_MatchesManualChaining enforces this
```

```
BC-3: NewBatchConfig rejects non-positive MaxRunningReqs
- GIVEN maxRunningReqs <= 0
- WHEN NewBatchConfig is called
- THEN it panics with a message containing "MaxRunningReqs" and the invalid value
- MECHANISM: Explicit check before struct construction
```

```
BC-4: NewBatchConfig rejects non-positive MaxScheduledTokens
- GIVEN maxScheduledTokens <= 0
- WHEN NewBatchConfig is called
- THEN it panics with a message containing "MaxScheduledTokens" and the invalid value
- MECHANISM: Explicit check before struct construction
```

```
BC-5: NewBatchConfig rejects negative LongPrefillTokenThreshold
- GIVEN longPrefillTokenThreshold < 0
- WHEN NewBatchConfig is called
- THEN it panics with a message containing "LongPrefillTokenThreshold" and the invalid value
- MECHANISM: Explicit check before struct construction
```

```
BC-6: NewBatchConfig accepts valid inputs unchanged
- GIVEN maxRunningReqs > 0, maxScheduledTokens > 0, longPrefillTokenThreshold >= 0
- WHEN NewBatchConfig is called
- THEN it returns a BatchConfig with all fields set to the provided values
- MECHANISM: Existing TestNewBatchConfig_FieldEquivalence validates this
```

**Negative contracts:**

```
BC-7: Hash optimization does not change allocation count to zero
- GIVEN a benchmark with N tokens
- WHEN BenchmarkHashBlock runs before and after optimization
- THEN allocs/op MUST decrease (not necessarily to zero — SHA256 hasher allocates internally)
- MECHANISM: Benchmark comparison
```

```
BC-8: Dead code removal does not break any production caller
- GIVEN HashTokens is removed
- WHEN go build ./... is run
- THEN compilation succeeds
- MECHANISM: Zero production callers (confirmed by grep)
```

### C) Component Interaction

```
sim/internal/hash/hash.go
  ├── HashBlock(prevHash, tokens)      ← called by sim/kv/cache.go (3 sites)
  ├── ComputeBlockHashes(blockSize, tokens) ← called by sim/prefix_cache_index.go
  └── [DELETED] HashTokens(tokens)     ← zero production callers

sim/config.go
  └── NewBatchConfig(maxRunning, maxTokens, prefillThresh)
        ├── called by cmd/root.go:685 (production)
        └── called by ~60 test sites (all valid values)
              └── EXCEPT TestNewSimulator_BatchConfigValidation (uses zero/negative → must update)
```

No new state, no new interfaces, no cross-boundary changes.

**Extension friction:** Adding a field to `BatchConfig` still touches 2 files (config.go + cmd/root.go) — unchanged by this PR.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #382 filed as "BatchConfig zero validation" | Panics in constructor, not error returns | CORRECTION: `NewSimulator` already validates BatchConfig fields (simulator.go:94-102) with error returns. This plan adds *complementary* constructor-level panics matching the `NewKVCacheState` pattern — defense-in-depth, not replacement. |
| Existing hardening-batch-plan.md covers #382 with #508, #509, #383, #384 | This plan covers #382 only (plus hash items) | SCOPE_CHANGE: Different PR scope. #508/#509/#383 are separate KV cache validation items. #384 is already resolved (empty WorkloadConfig). Note: `hardening-batch-plan.md` also claims `Fixes #382` in its Closes field — should be updated after this PR merges (R15). |

### E) Review Guide

**The tricky part:** The `TestNewSimulator_BatchConfigValidation` update (Task 4, Step 3). It currently uses `NewBatchConfig(0, ...)` to create invalid configs. After adding panics to `NewBatchConfig`, those calls would panic. The fix uses struct literals (`BatchConfig{MaxRunningReqs: 0, ...}`) to bypass the constructor, which is intentional — the test verifies `NewSimulator`'s defense-in-depth, not the constructor. A comment explains this.

**What to scrutinize:** BC-1 and BC-2 — that the optimized hash implementation produces byte-identical output. The existing `TestComputeBlockHashes_MatchesManualChaining` tests enforce this, but verify the optimization doesn't accidentally change the byte format.

**What's safe to skim:** Task 1 (dead code deletion) — trivially correct. Task 2 (benchmarks) — standard Go benchmark pattern.

**Known debt:** None introduced. The `fmt` import in hash.go remains required after removing HashTokens — it's used by `ComputeBlockHashes`'s panic message.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files modified:**
- `sim/internal/hash/hash.go` — remove `HashTokens`, optimize `HashBlock` and `ComputeBlockHashes`
- `sim/internal/hash/hash_test.go` — remove `TestHashTokens_Deterministic`, add benchmarks
- `sim/config.go` — add validation panics to `NewBatchConfig`
- `sim/config_test.go` — add `TestNewBatchConfig_PanicsOnInvalid`
- `sim/simulator_test.go` — update `TestNewSimulator_BatchConfigValidation` to use struct literals

**Key decisions:**
- Stack-allocated `[20]byte` buffer for `strconv.AppendInt` (max int64 = 19 digits + pipe = 20 bytes)
- Panics (not errors) in `NewBatchConfig` — matches `NewKVCacheState` pattern
- Struct literals in test for invalid values — intentionally bypasses constructor for defense-in-depth testing

### G) Task Breakdown

---

#### Task 1: Remove dead HashTokens function (#538)

**Contracts Implemented:** BC-8

**Files:**
- Modify: `sim/internal/hash/hash.go` (delete lines 14-33, remove `strings` import)
- Modify: `sim/internal/hash/hash_test.go` (delete lines 5-15)

**Step 1: Delete `HashTokens` and its test**

In `sim/internal/hash/hash.go`, delete the `HashTokens` function (lines 14-33) and remove the `strings` import.

In `sim/internal/hash/hash_test.go`, delete `TestHashTokens_Deterministic` (lines 5-15).

The resulting `hash.go` imports should be:
```go
import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
)
```

**Step 2: Verify build and tests pass**

Run: `go test ./sim/internal/hash/... -v`
Expected: PASS — all remaining tests pass (HashBlock, ComputeBlockHashes tests unaffected)

Run: `go build ./...`
Expected: SUCCESS — zero production callers of HashTokens

**Step 3: Run lint**

Run: `golangci-lint run ./sim/internal/hash/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/internal/hash/hash.go sim/internal/hash/hash_test.go
git commit -m "refactor(hash): remove dead HashTokens function (#538)

- Delete HashTokens (zero production callers after #537 hierarchical migration)
- Delete TestHashTokens_Deterministic
- Remove unused strings import
- HashTokens used incompatible format (pipe-between vs pipe-after)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Add benchmark tests for hash hot path (#542)

**Contracts Implemented:** BC-7 (baseline measurement)

**Files:**
- Modify: `sim/internal/hash/hash_test.go` (add benchmarks)

**Step 1: Write benchmark functions**

Add to `sim/internal/hash/hash_test.go`:

```go
func BenchmarkHashBlock(b *testing.B) {
	// Simulate a typical block: 16 tokens with values in [0, 128000]
	tokens := make([]int, 16)
	for i := range tokens {
		tokens[i] = i * 1000
	}
	prevHash := "abc123"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HashBlock(prevHash, tokens)
	}
}

func BenchmarkComputeBlockHashes(b *testing.B) {
	// Simulate a reasoning workload: 2048 tokens, block size 16 = 128 blocks
	tokens := make([]int, 2048)
	for i := range tokens {
		tokens[i] = i * 100
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeBlockHashes(16, tokens)
	}
}

func BenchmarkComputeBlockHashes_LargeContext(b *testing.B) {
	// Simulate a long reasoning context: 20480 tokens, block size 16 = 1280 blocks
	tokens := make([]int, 20480)
	for i := range tokens {
		tokens[i] = i
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeBlockHashes(16, tokens)
	}
}
```

**Step 2: Run benchmarks to establish baseline**

Run: `go test ./sim/internal/hash/... -bench=. -benchmem -count=3`
Expected: Benchmarks run successfully, showing allocs/op > 0

Record the output (allocs/op and ns/op) — this is the pre-optimization baseline.

**Step 3: Run lint**

Run: `golangci-lint run ./sim/internal/hash/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/internal/hash/hash_test.go
git commit -m "test(hash): add benchmark tests for HashBlock and ComputeBlockHashes (#542)

- BenchmarkHashBlock: single 16-token block
- BenchmarkComputeBlockHashes: 128 blocks (2048 tokens, typical workload)
- BenchmarkComputeBlockHashes_LargeContext: 1280 blocks (20480 tokens, reasoning)
- Establishes baseline for allocation optimization (#539)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Reduce strconv allocations in hash functions (#539)

**Contracts Implemented:** BC-1, BC-2, BC-7

**Files:**
- Modify: `sim/internal/hash/hash.go` (optimize HashBlock and ComputeBlockHashes)

**Step 1: Optimize HashBlock**

Replace the current per-token string allocation pattern with a stack-buffered approach:

Current:
```go
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	for _, t := range tokens {
		h.Write([]byte(strconv.Itoa(t)))
		h.Write([]byte("|"))
	}
	return hex.EncodeToString(h.Sum(nil))
}
```

Optimized:
```go
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	var buf [20]byte // stack buffer: max int64 (19 digits) + pipe
	for _, t := range tokens {
		b := strconv.AppendInt(buf[:0], int64(t), 10)
		b = append(b, '|')
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}
```

**Step 2: Optimize ComputeBlockHashes (inlined version)**

Apply the same optimization to the inlined loop:

Current inner loop:
```go
		h.Write([]byte(prevHash))
		for _, t := range tokens[start:end] {
			h.Write([]byte(strconv.Itoa(t)))
			h.Write([]byte("|"))
		}
```

Optimized inner loop (declare `var buf [20]byte` before the outer loop):
```go
	var buf [20]byte
	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		h.Reset()
		h.Write([]byte(prevHash))
		for _, t := range tokens[start:end] {
			b := strconv.AppendInt(buf[:0], int64(t), 10)
			b = append(b, '|')
			h.Write(b)
		}
		hashes[i] = hex.EncodeToString(h.Sum(nil))
		prevHash = hashes[i]
	}
```

Remove the now-unused `strconv` import's `Itoa` usage — but `strconv.AppendInt` still requires `strconv`, so the import stays.

**Step 3: Run existing tests to verify hash output is unchanged**

Run: `go test ./sim/internal/hash/... -v`
Expected: PASS — all tests pass, confirming BC-1 and BC-2 (byte-identical output)

Run: `go test ./sim/... -v -count=1`
Expected: PASS — KV cache and routing tests confirm identical behavior

**Step 4: Run benchmarks to measure improvement**

Run: `go test ./sim/internal/hash/... -bench=. -benchmem -count=3`
Expected: allocs/op decreases compared to baseline (BC-7)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/internal/hash/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/internal/hash/hash.go
git commit -m "perf(hash): reduce per-token strconv allocations in HashBlock (#539)

- Replace strconv.Itoa + []byte conversion with strconv.AppendInt into stack buffer
- Stack-allocated [20]byte buffer reused across tokens (fewer per-token heap allocs)
- Applied to both HashBlock and ComputeBlockHashes (inlined path)
- Hash output is byte-identical (same bytes written to SHA256 hasher)
- Implements BC-1, BC-2, BC-7

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Add BatchConfig validation to NewBatchConfig (#382)

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-6

**Files:**
- Modify: `sim/config.go` (add validation to `NewBatchConfig`)
- Modify: `sim/config_test.go` (add `TestNewBatchConfig_PanicsOnInvalid`)
- Modify: `sim/simulator_test.go` (update `TestNewSimulator_BatchConfigValidation`)

**Step 1: Write failing test for constructor validation**

Add to `sim/config_test.go`:

```go
func TestNewBatchConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name          string
		maxRunning    int64
		maxTokens     int64
		prefillThresh int64
		wantContains  string
	}{
		{"zero_max_running", 0, 2048, 0, "MaxRunningReqs"},
		{"negative_max_running", -1, 2048, 0, "MaxRunningReqs"},
		{"zero_max_tokens", 256, 0, 0, "MaxScheduledTokens"},
		{"negative_max_tokens", 256, -1, 0, "MaxScheduledTokens"},
		{"negative_prefill", 256, 2048, -1, "LongPrefillTokenThreshold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic")
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, tc.wantContains) {
					t.Errorf("panic message %q should contain %q", msg, tc.wantContains)
				}
			}()
			NewBatchConfig(tc.maxRunning, tc.maxTokens, tc.prefillThresh)
		})
	}
}
```

Add `"fmt"` and `"strings"` to `config_test.go` imports.

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNewBatchConfig_PanicsOnInvalid -v`
Expected: FAIL — `NewBatchConfig` does not panic yet

**Step 3: Implement validation in NewBatchConfig**

In `sim/config.go`, add validation to `NewBatchConfig`:

```go
func NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64) BatchConfig {
	if maxRunningReqs <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxRunningReqs must be > 0, got %d", maxRunningReqs))
	}
	if maxScheduledTokens <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxScheduledTokens must be > 0, got %d", maxScheduledTokens))
	}
	if longPrefillTokenThreshold < 0 {
		panic(fmt.Sprintf("NewBatchConfig: LongPrefillTokenThreshold must be >= 0, got %d", longPrefillTokenThreshold))
	}
	return BatchConfig{
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
	}
}
```

Add an import block to `sim/config.go` (which currently has no imports):
```go
import "fmt"
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestNewBatchConfig_PanicsOnInvalid -v`
Expected: PASS

**Step 5: Fix TestNewSimulator_BatchConfigValidation**

The existing test at `sim/simulator_test.go:247-279` calls `NewBatchConfig(0, ...)` which now panics. Update it to use struct literals for invalid values — this tests `NewSimulator`'s defense-in-depth validation (which accepts `BatchConfig` struct directly).

Replace the test's inner body (lines 262-278) to use struct literal assignment:

```go
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig()
			// Use struct literal to bypass NewBatchConfig validation — this tests
			// NewSimulator's defense-in-depth, not the constructor.
			cfg.BatchConfig = BatchConfig{
				MaxRunningReqs:            tc.maxRunning,
				MaxScheduledTokens:        tc.maxTokens,
				LongPrefillTokenThreshold: tc.prefillThresh,
			}
			kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
			latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
			if err != nil {
				t.Fatalf("MustNewLatencyModel: %v", err)
			}
			_, err = NewSimulator(cfg, kvStore, latencyModel)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantErrContain) {
				t.Errorf("error %q should contain %q", err.Error(), tc.wantErrContain)
			}
		})
	}
```

**Step 6: Run all tests to verify nothing breaks**

Run: `go test ./sim/... -v -count=1`
Expected: PASS — all tests pass

Run: `go test ./... -count=1`
Expected: PASS — full test suite passes (cluster tests use valid NewBatchConfig values)

**Step 7: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 8: Commit**

```bash
git add sim/config.go sim/config_test.go sim/simulator_test.go
git commit -m "fix(sim): add validation to NewBatchConfig (R3) (#382)

- NewBatchConfig panics on MaxRunningReqs <= 0, MaxScheduledTokens <= 0,
  LongPrefillTokenThreshold < 0 (matches NewKVCacheState pattern)
- Add TestNewBatchConfig_PanicsOnInvalid with table-driven test cases
- Update TestNewSimulator_BatchConfigValidation to use struct literals
  for invalid values (tests NewSimulator defense-in-depth)
- Implements BC-3, BC-4, BC-5, BC-6

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Unit (existing) | TestComputeBlockHashes_MatchesManualChaining |
| BC-1 | Task 3 | Unit (existing) | TestComputeBlockHashes_FourBlocks_MatchesManualChaining |
| BC-2 | Task 3 | Unit (existing) | TestComputeBlockHashes_MatchesManualChaining |
| BC-3 | Task 4 | Unit (panic) | TestNewBatchConfig_PanicsOnInvalid/zero_max_running |
| BC-4 | Task 4 | Unit (panic) | TestNewBatchConfig_PanicsOnInvalid/zero_max_tokens |
| BC-5 | Task 4 | Unit (panic) | TestNewBatchConfig_PanicsOnInvalid/negative_prefill |
| BC-6 | Task 4 | Unit (existing) | TestNewBatchConfig_FieldEquivalence |
| BC-7 | Tasks 2-3 | Benchmark | BenchmarkHashBlock, BenchmarkComputeBlockHashes |
| BC-8 | Task 1 | Build | go build ./... succeeds after deletion |

**Golden dataset:** No changes needed — hash output is byte-identical, BatchConfig validation is pre-construction.

**Invariant tests:** Not applicable — no runtime behavior changes. Existing invariant tests (request conservation, KV block conservation, clock monotonicity) remain valid and unmodified.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Hash optimization changes output | Low | High (breaks KV cache matching) | TestComputeBlockHashes_MatchesManualChaining verifies byte equivalence; full test suite run | Task 3 |
| NewBatchConfig panic breaks a test we didn't find | Low | Medium (CI failure) | Grepped all NewBatchConfig call sites; only TestNewSimulator_BatchConfigValidation uses invalid values | Task 4 |
| Stack buffer overflow for large token IDs | None | N/A | Max int64 = 19 digits; token IDs are in [0, 128000] (6 digits max). 20-byte buffer is sufficient. | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing packages
- [x] CLAUDE.md: no updates needed (no new files/packages, no new CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 2 before Task 3 for benchmark baseline)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed
- [x] Construction site audit: `NewBatchConfig` — all 60+ call sites use valid values (verified by grep), only test-specific invalid-value test updated
- [x] R1: No silent continue/return — panics on invalid input
- [x] R2: No map iteration changes
- [x] R3: NewBatchConfig now validates (this PR's purpose)
- [x] R4: No struct field additions — only validation logic
- [x] R5: No resource allocation loops
- [x] R6: No logrus.Fatalf in sim/ — panics are acceptable for constructor validation
- [x] R7: No golden tests added
- [x] R8: No exported mutable maps
- [x] R9: No YAML fields changed
- [x] R10: No YAML parsing changes
- [x] R11: No division changes
- [x] R12: No golden dataset changes
- [x] R13: No new interfaces
- [x] R14: No multi-concern methods
- [x] R15: No stale PR references
- [x] R16: Config grouped by module (unchanged)
- [x] R17: No routing signal changes
- [x] R18: No CLI flag changes
- [x] R19: No retry loops
- [x] R20: No detector changes
- [x] R21: No range over mutable slices
- [x] R22: No pre-check changes
- [x] R23: No parallel code paths

---

## Appendix: File-Level Implementation Details

### File: `sim/internal/hash/hash.go`

**Purpose:** SHA256 hashing utilities for KV cache prefix matching and routing prefix affinity.

**After all changes (Tasks 1 + 3):**

```go
// Package hash provides SHA256 hashing utilities for KV cache prefix matching
// and routing prefix affinity. These functions are shared between sim/ (routing)
// and sim/kv/ (cache) to ensure hash consistency (BC-3).
package hash

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
)

// HashBlock computes a SHA256 hash of a token block chained with the previous block's hash.
// Format: prevHash bytes, then for each token: "tokenN" + "|" (pipe AFTER each token).
// This creates hierarchical block hashes for prefix caching.
// Also inlined in ComputeBlockHashes for hasher reuse.
// TestComputeBlockHashes_MatchesManualChaining guards consistency between the two paths.
func HashBlock(prevHash string, tokens []int) string {
	h := sha256.New()
	h.Write([]byte(prevHash))
	var buf [20]byte // stack buffer: max int64 (19 digits) + pipe
	for _, t := range tokens {
		b := strconv.AppendInt(buf[:0], int64(t), 10)
		b = append(b, '|')
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ComputeBlockHashes returns hierarchical block hashes for a token sequence.
// Each hash chains with the previous block's hash, enabling prefix matching.
// Tokens that don't fill a complete block are ignored.
// Produces the same output as calling HashBlock sequentially, but reuses a
// single SHA256 hasher instance across blocks to reduce allocations.
// Output equivalence is enforced by TestComputeBlockHashes_MatchesManualChaining.
func ComputeBlockHashes(blockSize int, tokens []int) []string {
	if blockSize <= 0 {
		panic(fmt.Sprintf("ComputeBlockHashes: blockSize must be > 0, got %d", blockSize))
	}
	numBlocks := len(tokens) / blockSize
	if numBlocks == 0 {
		return nil
	}
	hashes := make([]string, numBlocks)
	h := sha256.New()
	prevHash := ""
	var buf [20]byte // stack buffer: reused across all tokens in all blocks
	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		h.Reset()
		// Inlines HashBlock logic for hasher reuse — keep in sync with HashBlock above.
		h.Write([]byte(prevHash))
		for _, t := range tokens[start:end] {
			b := strconv.AppendInt(buf[:0], int64(t), 10)
			b = append(b, '|')
			h.Write(b)
		}
		hashes[i] = hex.EncodeToString(h.Sum(nil))
		prevHash = hashes[i]
	}
	return hashes
}
```

### File: `sim/config.go`

**Purpose:** Module-scoped sub-config types with canonical constructors (R4).

**NewBatchConfig after changes (Task 4):**

```go
// NewBatchConfig creates a BatchConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Panics on invalid values: MaxRunningReqs and MaxScheduledTokens must be > 0,
// LongPrefillTokenThreshold must be >= 0 (0 means disabled).
func NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64) BatchConfig {
	if maxRunningReqs <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxRunningReqs must be > 0, got %d", maxRunningReqs))
	}
	if maxScheduledTokens <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxScheduledTokens must be > 0, got %d", maxScheduledTokens))
	}
	if longPrefillTokenThreshold < 0 {
		panic(fmt.Sprintf("NewBatchConfig: LongPrefillTokenThreshold must be >= 0, got %d", longPrefillTokenThreshold))
	}
	return BatchConfig{
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
	}
}
```

**Note:** Requires adding `"fmt"` to the import block in `sim/config.go`.
