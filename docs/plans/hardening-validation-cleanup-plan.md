# Hardening: CLI Validation, Canonical Constructor, Dead Field Removal, Tiered Cache Test

> **NOTE (R15):** `NewKVCapacityParams` signature was extended from 4 to 6 positional args
> in PR #559 (MoE roofline). Code snippets in this completed plan reference the original
> 4-arg signature. See `sim/latency/kv_capacity.go` for the current signature.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close four independent hardening gaps: CLI validation for distribution-mode token flags, canonical constructor for KV capacity params, removal of a dead `Streaming` field from the simulator, and a round-trip test proving the tiered KV cache reload mechanism exercises hierarchical hashes.

**The problem today:** (1) Distribution-mode CLI flags (`--prompt-tokens-min`, `--prompt-tokens-max`, `--output-tokens-min`, `--output-tokens-max`, stdev) accept negative values — the only validation is a cross-field range check that can be bypassed with coordinated negative inputs (e.g., `--prompt-tokens-min -100 --prompt-tokens-max -50 --prompt-tokens -75`). (2) `KVCapacityParams` has no canonical constructor — adding a field won't produce a compiler error at construction sites. (3) `Request.Streaming` is populated by workload generation but never read by the simulator — users configuring `streaming: true` get a silent no-op. (4) The tiered KV cache's CPU reload mechanism has no test exercising the offload→reload path with hierarchical hashes from #537.

**What this PR adds:**
1. CLI standalone validation — negative/zero `--prompt-tokens-min/max`, `--output-tokens-min/max`, and negative stdev values are rejected at startup with clear error messages
2. `NewKVCapacityParams` canonical constructor — all construction sites compile-time enforced, matching R4 conventions
3. Dead field removal — `Request.Streaming` removed from `sim/request.go`, all workload generation sites updated, test assertions adjusted
4. Tiered cache reload mechanism test — verifies `tryReloadFromCPU` is triggered and executes during the offload→reload path, proving the mechanism works with hierarchical hashes

**Why this matters:** Items 1-2 close R3/R4 compliance gaps identified in reviews. Item 3 removes user-visible confusion (streaming config has zero effect). Item 4 provides regression safety for the hashing migration (#537) and documents a pre-existing design limitation in `tryReloadFromCPU`.

**Architecture:** All changes are in existing files. Item 1: `cmd/root.go` (6 new `logrus.Fatalf` checks). Item 2: `sim/latency/kv_capacity.go` (new constructor), `sim/latency/kv_capacity_test.go` (update construction sites). Item 3: `sim/request.go` (field removal), `sim/workload/generator.go`, `sim/workload/replay.go`, `sim/workload/generator_test.go` (remove Streaming propagation); `sim/workload/spec.go`, `sim/workload/cohort.go`, `sim/workload/scenarios.go`, `sim/workload/tracev2.go` are unchanged — `ClientSpec`/`CohortSpec`/`TraceRecord` preserve their `Streaming` fields. Item 4: `sim/kv/tiered_test.go` (new test function).

**Source:** GitHub issues #527, #533, #494, #541

**Closes:** Fixes #527, fixes #533, fixes #494, fixes #541

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

1. **Building blocks modified:** CLI validation layer (`cmd/root.go`), KV capacity estimation (`sim/latency/kv_capacity.go`), request data model (`sim/request.go`), workload generation pipeline (`sim/workload/`), tiered KV cache tests (`sim/kv/tiered_test.go`)
2. **Adjacent blocks:** `sim/simulator.go` (consumes Request — does NOT read Streaming), `sim/workload/generator.go` (produces Request), `sim/kv/tiered.go` (offload/reload logic — unchanged)
3. **Invariants touched:** None directly. No runtime behavior changes for valid inputs.
4. **Construction Site Audit:**
   - `KVCapacityParams` — 1 production function builds it field-by-field (`ExtractKVCapacityParams`, lines 255-297, with 3 success-return paths migrated to constructor). Error-return zero-value sites: 2 in `ExtractKVCapacityParamsFromFile` (lines 241/245, wrapper function — untouched) + 1 in `ExtractKVCapacityParams` (line 291). 8+ test struct-literal sites (`sim/latency/kv_capacity_test.go:45,334,363,478,516,522,554,560`)
   - `Request.Streaming` removal touches 2 production construction sites: `sim/workload/generator.go:272`, `sim/workload/replay.go:54`. Note: `sim/workload/cohort.go:41` sets `ClientSpec.Streaming` (not `sim.Request`) — preserved per BC-6.

**Confirmed facts:**
- `promptTokensMin` declared at `cmd/root.go:47`, flag registered at line 871 (default 2)
- `promptTokensMax` declared at `cmd/root.go:48`, flag registered at line 872 (default 7000)
- `outputTokensMin` declared at `cmd/root.go:51`, flag registered at line 875 (default 2)
- `outputTokensMax` declared at `cmd/root.go:52`, flag registered at line 876 (default 7000)
- `promptTokensStdev` declared at `cmd/root.go:46`, flag registered at line 870 (default 256)
- `outputTokensStdev` declared at `cmd/root.go:50`, flag registered at line 874 (default 256)
- Cross-field range check at `cmd/root.go:467-471` — only check currently present
- `Request.Streaming` at `sim/request.go:52` — `bool` field, never read by `simulator.go`, `batch_formation.go`, or any latency model
- `Streaming` in `ClientSpec` at `sim/workload/spec.go:105` — YAML field `streaming`
- `Streaming` in `CohortSpec` at `sim/workload/spec.go:66` — YAML field `streaming,omitempty`
- `Streaming` in `TraceRecord` at `sim/workload/tracev2.go:53` — used in trace v2 CSV export at line 122
- `KVCapacityParams` at `sim/latency/kv_capacity.go:14-19` — 4 fields, no constructor
- Tiered cache `tryReloadFromCPU` at `sim/kv/tiered.go:120-172` — copies `blk.Hash` string opaquely
- `maybeOffload` at `sim/kv/tiered.go:228-258` — does NOT reduce GPU free count (removeFromFreeList + appendToFreeList as empty block)
- `tryReloadFromCPU` single-slot overwrite limitation at `sim/kv/tiered.go:115-119` — with fewer free GPU blocks than CPU blocks, earlier hashes are destroyed by `popFreeBlock` when recycling slots

**Deviations from issue descriptions:**
- #494 says "either remove the field or implement streaming behavior." We choose removal — implementing streaming simulation is a feature, not a hardening fix. The field survives in `TraceRecord`, `ClientSpec`, and `CohortSpec` (trace/spec YAML schema) because those describe real-world data formats. Only the `sim.Request` field and the lines that copy it from spec→request are removed.

---

## Part 1: Design Validation

### A) Executive Summary

Four independent hardening items bundled into one PR because they're all small, library-internal or CLI-layer changes with no runtime behavior change for valid inputs and no golden dataset impact:

1. **CLI validation** (#527): Add standalone positivity checks for 6 distribution-mode flags before they reach the cross-field range check
2. **Canonical constructor** (#533): Add `NewKVCapacityParams` per R4 so future field additions cause compiler errors
3. **Dead field removal** (#494): Remove `Request.Streaming` that is never consumed by the simulator
4. **Tiered cache test** (#541): Add test verifying the offload→reload mechanism is exercised with hierarchical hashes

No file overlaps between the four items. No golden dataset regeneration needed.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: CLI standalone validation — min/max less than 1
- GIVEN distribution mode is active (`--workload distribution`)
- WHEN `--prompt-tokens-min`, `--prompt-tokens-max`, `--output-tokens-min`, or `--output-tokens-max` is less than 1 (zero or negative)
- THEN the CLI MUST exit with a `logrus.Fatalf` error message naming the offending flag and value
- MECHANISM: Standalone `< 1` check before the cross-field range check at `cmd/root.go:467`

BC-2: CLI standalone validation — negative stdev
- GIVEN distribution mode is active
- WHEN `--prompt-tokens-stdev` or `--output-tokens-stdev` is negative
- THEN the CLI MUST exit with a `logrus.Fatalf` error message naming the offending flag and value
- MECHANISM: Standalone `< 0` check (zero stdev is valid — means constant distribution). Note: the pre-existing cross-field check at `cmd/root.go:467` additionally enforces `stdev >= min`, so `--prompt-tokens-stdev 0` is rejected when `--prompt-tokens-min >= 1` (default 2). This is a pre-existing constraint outside this PR's scope.

BC-3: Canonical constructor enforces field completeness
- GIVEN `KVCapacityParams` has N fields
- WHEN a caller constructs it via `NewKVCapacityParams(isMoE, numLocalExperts, tieWordEmbeddings, hiddenAct)`
- THEN all N fields MUST be provided as arguments (adding a field causes a compiler error at every call site)
- MECHANISM: Positional constructor function returning `KVCapacityParams`

BC-4: ExtractKVCapacityParams uses canonical constructor
- GIVEN the `ExtractKVCapacityParams` function builds a `KVCapacityParams`
- WHEN a new field is added to `KVCapacityParams`
- THEN `ExtractKVCapacityParams` MUST fail to compile until updated
- MECHANISM: Replace field-by-field construction with `NewKVCapacityParams(...)` calls

BC-5: Streaming field removed from simulator request
- GIVEN a `sim.Request` struct
- WHEN any code attempts to read `req.Streaming`
- THEN it MUST fail to compile
- MECHANISM: Remove the `Streaming` field from `sim/request.go`

BC-6: Workload spec YAML schema preserves streaming
- GIVEN a workload spec YAML with `streaming: true` in a client spec
- WHEN the spec is loaded
- THEN the `ClientSpec.Streaming` field MUST still be parsed correctly
- MECHANISM: `ClientSpec.Streaming` and `CohortSpec.Streaming` are preserved (schema contract)

BC-7: Trace v2 format preserves streaming column
- GIVEN a trace v2 CSV file with a streaming column
- WHEN the trace is loaded
- THEN the `TraceRecord.Streaming` field MUST still be parsed correctly
- MECHANISM: `TraceRecord.Streaming` is preserved (trace format contract)

BC-8: Tiered cache reload mechanism exercised
- GIVEN a TieredKVCache with prefix blocks offloaded to CPU and GPU under pressure (free < needed)
- WHEN a request for the same prefix triggers `AllocateKVBlocks`
- THEN `tryReloadFromCPU` MUST execute (cpuHitCount > 0), proving the offload→reload mechanism works
- MECHANISM: GPU direct alloc fails (1 free < 2 needed), triggering `tryReloadFromCPU`. Note: due to the single-slot overwrite limitation (tiered.go:115-119), the hierarchical hash chain may break — the test verifies the mechanism, not the chain outcome.

**Negative contracts:**

BC-9: No runtime behavior change for valid inputs
- GIVEN valid positive values for all distribution-mode flags
- WHEN the simulation runs
- THEN output MUST be byte-identical to pre-PR behavior
- MECHANISM: New checks reject only invalid inputs; no changes to simulation logic

BC-10: No golden dataset impact
- GIVEN the golden dataset uses default workload settings (not distribution mode)
- WHEN tests run
- THEN `testdata/goldendataset.json` MUST NOT need regeneration
- MECHANISM: None of the changes affect default-mode code paths

### C) Component Interaction

```
cmd/root.go (CLI layer)
  ├── validates --prompt-tokens-min/max/stdev, --output-tokens-min/max/stdev [BC-1, BC-2]
  └── passes valid values to workload.SynthesizeFromDistribution()

sim/latency/kv_capacity.go
  ├── NewKVCapacityParams(isMoE, numLocalExperts, tieWordEmbeddings, hiddenAct) [BC-3]
  └── ExtractKVCapacityParams() calls NewKVCapacityParams() [BC-4]

sim/request.go
  └── Request struct — Streaming field REMOVED [BC-5]

sim/workload/{generator,replay}.go
  └── Request construction sites — Streaming line removed [BC-5]

sim/kv/tiered_test.go
  └── TestTieredKVCache_OffloadReload_MechanismExercised [BC-8]
```

Extension friction: Adding a field to `KVCapacityParams` after this PR touches 2 files (constructor + `ExtractKVCapacityParams`). Before this PR it touches 1 file but silently misses test sites. Net improvement.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #494: "either remove or implement streaming" | Remove from `sim.Request` only; keep in `ClientSpec`, `CohortSpec`, `TraceRecord` | SIMPLIFICATION — the YAML schema and trace format describe real-world data; only the sim-internal field is dead |
| #527: lists stdev flags as missing validation | Validates stdev ≥ 0 (not > 0) | CORRECTION — zero stdev is valid (constant distribution), though pre-existing cross-field check still rejects it when min ≥ 1 |
| #533: "Add NewKVCapacityParams constructor" | Also migrate all test construction sites | ADDITION — R4 requires ALL sites use the constructor |
| #541: "add tiered KV cache round-trip test with hierarchical hashing" | Test verifies reload mechanism (cpuHitCount > 0), not full hash chain survival | SCOPE_CHANGE — pre-existing `tryReloadFromCPU` single-slot overwrite limitation (tiered.go:115-119) makes full chain verification impossible with the current code. File follow-up issue. |

### E) Review Guide

**The tricky part:** Task 4's tiered cache test. `maybeOffload` does NOT reduce GPU free count (remove + re-append as empty). `tryReloadFromCPU` has a single-slot overwrite limitation: with fewer free GPU blocks than CPU blocks, `popFreeBlock` destroys earlier-loaded hashes when recycling slots. Hierarchical hashing then breaks (block 0 hash lost → entire chain unresolvable). The test verifies the reload *mechanism* (cpuHitCount > 0), not the chain *outcome* (GetCachedBlocks hits). Three existing tests in `tiered_test.go` have the same silent allocation failure.

**What to scrutinize:** The stdev validation bound (BC-2): is `< 0` correct? Yes — zero stdev means constant distribution, which is semantically valid. The pre-existing cross-field check rejects it anyway when min ≥ 1, but that's not this PR's concern.

**What's safe to skim:** BC-3/BC-4 (canonical constructor) is pure mechanical refactoring. Tasks 1-3 are straightforward.

**Known debt:** (1) `Request.Model` is in a similar situation to `Streaming` — "carried through the pipeline but not read by any routing policy" (request.go:67). Out of scope. (2) `tryReloadFromCPU` overwrite limitation with hierarchical hashing — file follow-up issue.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `cmd/root.go` — Add 6 standalone validation checks (BC-1, BC-2)
- `sim/latency/kv_capacity.go` — Add `NewKVCapacityParams` constructor, migrate `ExtractKVCapacityParams` (BC-3, BC-4)
- `sim/latency/kv_capacity_test.go` — Migrate test construction sites to use constructor (BC-3)
- `sim/request.go` — Remove `Streaming` field (BC-5)
- `sim/workload/generator.go` — Remove `Streaming: client.Streaming` line (BC-5)
- `sim/workload/replay.go` — Remove `Streaming: rec.Streaming` line (BC-5)
- `sim/workload/generator_test.go` — Remove `req.Streaming` assertion (BC-5)
- `sim/kv/tiered_test.go` — Add reload mechanism test (BC-8)

**Key decisions:**
- Stdev validation uses `< 0` (zero is valid for constant distribution)
- Min/max validation uses `< 1` (at least 1 token required)
- `NewKVCapacityParams` is a pure value constructor (no validation needed — fields are booleans, int, string)
- `Streaming` preserved in `ClientSpec`, `CohortSpec`, `TraceRecord` (YAML/trace schema stability)
- Task 4 test follows the established pattern from `TestTieredKVCache_ThrashingDetected` (10 GPU blocks, blockSize=2, 3 fillers, 1 free) and verifies mechanism (cpuHitCount > 0), not chain outcome

**Confirmation:** No dead code. All paths exercisable. No golden dataset impact.

### G) Task Breakdown

---

### Task 1: CLI standalone validation for distribution token flags

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `cmd/root.go:462-471` (before cross-field range check)

**Step 1: Implement validation checks**

Context: Add standalone positivity checks for the 6 distribution-mode flags. These must go BEFORE the existing cross-field range check (line 467) and INSIDE the `workloadType == "distribution"` branch (line 462).

In `cmd/root.go`, insert after line 466 (`rate` validation closing brace) and before line 467 (the cross-field check):

```go
		// R3: Standalone validation for distribution token bounds (BC-1, BC-2)
		if promptTokensMin < 1 {
			logrus.Fatalf("--prompt-tokens-min must be >= 1, got %d", promptTokensMin)
		}
		if promptTokensMax < 1 {
			logrus.Fatalf("--prompt-tokens-max must be >= 1, got %d", promptTokensMax)
		}
		if outputTokensMin < 1 {
			logrus.Fatalf("--output-tokens-min must be >= 1, got %d", outputTokensMin)
		}
		if outputTokensMax < 1 {
			logrus.Fatalf("--output-tokens-max must be >= 1, got %d", outputTokensMax)
		}
		if promptTokensStdev < 0 {
			logrus.Fatalf("--prompt-tokens-stdev must be >= 0, got %d", promptTokensStdev)
		}
		if outputTokensStdev < 0 {
			logrus.Fatalf("--output-tokens-stdev must be >= 0, got %d", outputTokensStdev)
		}
```

**Step 2: Build to verify no syntax errors**

Run: `go build ./...`
Expected: Success

**Step 3: Run existing tests to verify no regressions**

Run: `go test ./cmd/... -v -count=1`
Expected: All tests PASS (existing tests use valid values)

**Step 4: Run lint**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add cmd/root.go
git commit -m "fix(cmd): add R3 standalone validation for distribution token flags (BC-1, BC-2)

- Reject --prompt-tokens-min/max, --output-tokens-min/max < 1
- Reject --prompt-tokens-stdev, --output-tokens-stdev < 0
- Checks run before cross-field range check, preventing bypass via coordinated negatives

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Canonical constructor for KVCapacityParams

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/latency/kv_capacity.go:14-19` (add constructor after struct)
- Modify: `sim/latency/kv_capacity.go:255-297` (migrate `ExtractKVCapacityParams`)
- Modify: `sim/latency/kv_capacity_test.go` (migrate test construction sites)

**Step 1: Add constructor and migrate production site**

Context: Add `NewKVCapacityParams` as a positional constructor. Then migrate `ExtractKVCapacityParams` to use it. The constructor is a pure value function — no validation needed since fields are `bool`, `int`, and `string` with no invalid states at the type level (MoE expert count validation happens in `CalculateKVBlocks`).

In `sim/latency/kv_capacity.go`, add after the struct definition (after line 19):

```go
// NewKVCapacityParams creates a KVCapacityParams. Positional arguments ensure
// that adding a field causes a compiler error at every construction site (R4).
func NewKVCapacityParams(isMoE bool, numLocalExperts int, tieWordEmbeddings bool, hiddenAct string) KVCapacityParams {
	return KVCapacityParams{
		IsMoE:             isMoE,
		NumLocalExperts:   numLocalExperts,
		TieWordEmbeddings: tieWordEmbeddings,
		HiddenAct:         hiddenAct,
	}
}
```

In `ExtractKVCapacityParams`, replace the body (lines 255-297) to collect values first, then construct via `NewKVCapacityParams` at return points:

```go
func ExtractKVCapacityParams(hf *HFConfig) (KVCapacityParams, error) {
	hiddenAct := hf.MustGetString("hidden_act", "")
	tieWordEmbeddings := false
	if tied, ok := hf.GetBool("tie_word_embeddings"); ok {
		tieWordEmbeddings = tied
	}

	// MoE detection: check multiple field names used by different architectures.
	numLocalExperts := hf.MustGetInt("num_local_experts", 0)
	if numLocalExperts > 1 {
		return NewKVCapacityParams(true, numLocalExperts, tieWordEmbeddings, hiddenAct), nil
	}

	// Fallback MoE indicators
	for _, key := range []string{"n_routed_experts", "num_experts"} {
		if v := hf.MustGetInt(key, 0); v > 1 {
			return NewKVCapacityParams(true, v, tieWordEmbeddings, hiddenAct), nil
		}
	}
	// Activation-count fields signal MoE but don't provide total expert count
	for _, key := range []string{"n_shared_experts", "num_experts_per_tok"} {
		if v := hf.MustGetInt(key, 0); v > 0 {
			return KVCapacityParams{}, fmt.Errorf(
				"model appears to be MoE (%s=%d) but num_local_experts is missing; "+
					"cannot estimate weight size accurately. Set --total-kv-blocks explicitly", key, v)
		}
	}

	return NewKVCapacityParams(false, 0, tieWordEmbeddings, hiddenAct), nil
}
```

Note: Error-return sites keep `KVCapacityParams{}` zero-value — they're error paths returning an invalid sentinel. The 2 sites in `ExtractKVCapacityParamsFromFile` (lines 241/245) are in a wrapper function and also untouched.

**Step 2: Migrate test construction sites**

In `sim/latency/kv_capacity_test.go`, replace the `validDenseKVParams` helper (lines 44-51):

```go
func validDenseKVParams() latency.KVCapacityParams {
	return latency.NewKVCapacityParams(false, 0, false, "silu")
}
```

Replace all other test construction sites with `latency.NewKVCapacityParams(...)`:
- Line 334: `latency.KVCapacityParams{HiddenAct: "silu"}` → `latency.NewKVCapacityParams(false, 0, false, "silu")`
- Line 363: `latency.KVCapacityParams{HiddenAct: "relu"}` → `latency.NewKVCapacityParams(false, 0, false, "relu")`
- Line 478: `latency.KVCapacityParams{HiddenAct: "silu", IsMoE: true, NumLocalExperts: 8}` → `latency.NewKVCapacityParams(true, 8, false, "silu")`
- Line 516: `latency.KVCapacityParams{HiddenAct: "silu"}` → `latency.NewKVCapacityParams(false, 0, false, "silu")`
- Line 522: `latency.KVCapacityParams{IsMoE: true, NumLocalExperts: 8, HiddenAct: "silu"}` → `latency.NewKVCapacityParams(true, 8, false, "silu")`
- Line 554: `latency.KVCapacityParams{HiddenAct: "silu", TieWordEmbeddings: false}` → `latency.NewKVCapacityParams(false, 0, false, "silu")`
- Line 560: `latency.KVCapacityParams{HiddenAct: "silu", TieWordEmbeddings: true}` → `latency.NewKVCapacityParams(false, 0, true, "silu")`

**Step 3: Build and test**

Run: `go build ./... && go test ./sim/latency/... -v -count=1`
Expected: Build succeeds, all latency tests PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/latency/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/latency/kv_capacity.go sim/latency/kv_capacity_test.go
git commit -m "refactor(latency): add NewKVCapacityParams canonical constructor (BC-3, BC-4)

- Add positional constructor per R4 conventions
- Migrate ExtractKVCapacityParams and all test sites
- Adding a KVCapacityParams field now causes compiler errors at every site

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Remove Request.Streaming from simulator

**Contracts Implemented:** BC-5, BC-6, BC-7

**Files:**
- Modify: `sim/request.go:52` (remove field)
- Modify: `sim/workload/generator.go:272` (remove Streaming line)
- Modify: `sim/workload/replay.go:54` (remove Streaming line)
- Modify: `sim/workload/generator_test.go:409` (remove Streaming assertion)

**Step 1: Remove the field from Request struct**

In `sim/request.go`, remove line 52:
```go
	Streaming       bool    // Whether this request uses streaming response mode
```

**Step 2: Remove from workload generation construction sites**

In `sim/workload/generator.go`, remove the `Streaming` line from the Request literal (around line 272):
```go
				Streaming:        client.Streaming,
```

In `sim/workload/replay.go`, remove the `Streaming` line from the Request literal (around line 54):
```go
			Streaming:        rec.Streaming,
```

**Step 3: Update test assertion**

In `sim/workload/generator_test.go`, line 409, change:
```go
	if req.Streaming || req.RoundIndex != 0 || req.ReasonRatio != 0 {
```
to:
```go
	if req.RoundIndex != 0 || req.ReasonRatio != 0 {
```

And update the corresponding error message on line 410:
```go
		t.Error("new bool/int/float fields should have zero-value defaults")
```
to:
```go
		t.Error("new int/float fields should have zero-value defaults")
```

**Step 4: Build to verify all construction sites updated**

Run: `go build ./...`
Expected: Success (no remaining references to `Request.Streaming` in production code)

**Step 5: Run all tests**

Run: `go test ./sim/... ./sim/workload/... -count=1`
Expected: All PASS

**Step 6: Run lint**

Run: `golangci-lint run ./sim/... ./sim/workload/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/request.go sim/workload/generator.go sim/workload/replay.go sim/workload/generator_test.go
git commit -m "fix(sim): remove dead Request.Streaming field (BC-5, BC-6, BC-7)

- Remove Streaming from sim.Request (never read by simulator)
- Preserve Streaming in ClientSpec/CohortSpec/TraceRecord (YAML schema stability)
- Users configuring streaming: true in workload spec still parse correctly;
  the field just no longer propagates to the internal Request type

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Tiered KV cache reload mechanism test

**Contracts Implemented:** BC-8

**Files:**
- Modify: `sim/kv/tiered_test.go` (add new test function)

**Step 1: Write the reload mechanism test**

Context: This test verifies that the offload→reload mechanism works with hierarchical hashes. It follows the established pattern from `TestTieredKVCache_ThrashingDetected` (10 GPU blocks, blockSize=2, 3 fillers, 1 free GPU block).

Key design notes:
- `maybeOffload` does NOT reduce GPU free count (removeFromFreeList + appendToFreeList as empty)
- `tryReloadFromCPU` has a single-slot overwrite limitation (tiered.go:115-119): with 1 free GPU block and 2 CPU blocks, `popFreeBlock` destroys block 0's hash when recycling the slot for block 1. Hierarchical hashing then breaks (block 0 hash lost → entire chain unresolvable via `GetCachedBlocks`).
- The overall allocation fails (needs 2, has 1 after reload). This matches existing tests.
- We verify the mechanism (cpuHitCount > 0), not the chain outcome.

Add to `sim/kv/tiered_test.go`:

```go
func TestTieredKVCache_OffloadReload_MechanismExercised(t *testing.T) {
	// BC-8: GIVEN a TieredKVCache with prefix blocks offloaded to CPU and GPU under pressure
	// WHEN a request for the same prefix triggers AllocateKVBlocks
	// THEN tryReloadFromCPU MUST execute (cpuHitCount > 0)
	//
	// Follows the established pattern from TestTieredKVCache_ThrashingDetected.
	//
	// Block arithmetic (10 GPU blocks, blockSize=2):
	//   Step 1: target [1,2,3,4] (2 blk) + 3 others (6 blk) = 8 used, 2 free
	//   Step 2: release target → 6 used, 4 free. maybeOffload (util 0.6 > 0.3)
	//           offloads 2 hashed free blocks to CPU. Free count UNCHANGED: 6 used, 4 free.
	//   Step 3: 3 fillers (3 blk) → 9 used, 1 free
	//   Step 4: reload request needs 2 new (0 cached), has 1 free → GPU FAILS.
	//           tryReloadFromCPU: 2 CPU blocks, 1 free GPU slot.
	//           Reload block 0 into slot → append. Reload block 1 → pop same slot,
	//           destroy block 0 hash, fill with block 1. cpuHitCount = 2.
	//           GetCachedBlocks: block 0 hash miss (overwritten). newCached = 0.
	//           Retry: needs 2, has 1 → fails. ok = false (expected — known limitation).
	//
	// Note: the allocation failure is due to the single-slot overwrite limitation
	// in tryReloadFromCPU (tiered.go:115-119) combined with hierarchical hashing.
	// This test verifies the reload MECHANISM works, not the chain outcome.

	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 0)
	tiered.SetClock(100)

	// Step 1: Allocate target prefix [1,2,3,4] (2 blocks) + fill GPU to 80%
	tokens := []int{1, 2, 3, 4}
	target := &sim.Request{ID: "target", InputTokens: tokens}
	if !tiered.AllocateKVBlocks(target, 0, 4, []int64{}) {
		t.Fatal("target allocation should succeed")
	}
	others := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		others[i] = &sim.Request{ID: fmt.Sprintf("o%d", i), InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13}}
		if !tiered.AllocateKVBlocks(others[i], 0, 4, []int64{}) {
			t.Fatalf("other allocation %d should succeed", i)
		}
	}
	// GPU: 8 used, 2 free

	// Verify target's prefix is cached before offload
	cachedBefore := tiered.GetCachedBlocks(tokens)
	if len(cachedBefore) != 2 {
		t.Fatalf("expected 2 cached blocks before offload, got %d", len(cachedBefore))
	}

	// Step 2: Release target → offload triggered (util 6/10=0.6 > 0.3)
	tiered.ReleaseKVBlocks(target)
	if tiered.offloadCount == 0 {
		t.Fatal("setup error: offload should have triggered")
	}
	// After offload: 6 used, 4 free (offload preserves free count), 2 blocks on CPU

	// Verify cache miss after offload (hashes removed from GPU hash table)
	cachedAfterOffload := tiered.GetCachedBlocks(tokens)
	if len(cachedAfterOffload) != 0 {
		t.Fatalf("expected 0 cached blocks after offload, got %d", len(cachedAfterOffload))
	}

	// Step 3: Fill GPU to leave exactly 1 free block
	tiered.SetClock(5000) // well past thrashing window
	fillers := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		fillers[i] = &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		if !tiered.AllocateKVBlocks(fillers[i], 0, 2, []int64{}) {
			t.Fatalf("filler allocation %d should succeed", i)
		}
	}
	// GPU: 9 used, 1 free. Target needs 2 blocks → GPU direct alloc FAILS.

	// Step 4: Re-request the SAME prefix — GPU fails, triggers CPU→GPU reload
	reloadReq := &sim.Request{ID: "reload", InputTokens: tokens}
	cached := tiered.GetCachedBlocks(tokens)
	start := int64(len(cached)) * tiered.BlockSize()
	tiered.AllocateKVBlocks(reloadReq, start, 4, cached)
	// Allocation return value intentionally unchecked — may fail due to single-slot
	// overwrite limitation in tryReloadFromCPU (see test comment above).

	// Step 5: Verify CPU reload path was exercised
	if tiered.cpuHitCount == 0 {
		t.Fatal("expected CPU reload to occur (cpuHitCount == 0 means reload path was never triggered)")
	}

	// Clean up
	for _, o := range others {
		tiered.ReleaseKVBlocks(o)
	}
	for _, f := range fillers {
		tiered.ReleaseKVBlocks(f)
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/kv/... -run TestTieredKVCache_OffloadReload_MechanismExercised -v`
Expected: PASS

**Step 3: Run all KV tests**

Run: `go test ./sim/kv/... -v -count=1`
Expected: All PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/kv/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/kv/tiered_test.go
git commit -m "test(kv): add tiered cache reload mechanism test with hierarchical hashing (BC-8)

- Verify offload→CPU→reload→GPU mechanism executes (cpuHitCount > 0)
- Follows established pattern from TestTieredKVCache_ThrashingDetected
- Documents single-slot overwrite limitation with hierarchical hashing
  (tryReloadFromCPU destroys block 0 hash when recycling for block 1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1 | N/A (CLI layer — tested by absence of panic with valid defaults) | Existing `cmd/` tests verify valid paths |
| BC-2 | Task 1 | N/A (CLI layer) | Same |
| BC-3 | Task 2 | Unit (compile-time) | All `kv_capacity_test.go` sites use `NewKVCapacityParams` — compiler enforces |
| BC-4 | Task 2 | Unit | Existing `TestExtractKVCapacityParams_*` tests verify behavior preserved |
| BC-5 | Task 3 | Unit (compile-time) | `go build ./...` fails if any code references `req.Streaming` |
| BC-6 | Task 3 | Unit | Existing `TestWorkloadSpec_*` tests verify YAML parsing still works |
| BC-7 | Task 3 | Unit | Existing `TestTraceV2_*` tests verify trace parsing still works |
| BC-8 | Task 4 | Unit | `TestTieredKVCache_OffloadReload_MechanismExercised` |

Golden dataset: **No update needed**. None of the changes affect simulation output.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missing a `Request.Streaming` construction site | Low | Medium (build failure) | `go build ./...` catches all; confirmed 2 production sites via grep | Task 3 |
| Stdev=0 rejection breaks constant-distribution users | Low | High | BC-2 specifies `< 0` (zero is valid); pre-existing cross-field check would reject it anyway when min ≥ 1 | Task 1 |
| Task 4 test setup doesn't trigger reload path | Low | Low | Follows proven pattern from `TestTieredKVCache_ThrashingDetected`; cpuHitCount assertion verifies path was taken | Task 4 |
| Test sites for KVCapacityParams missed during migration | Low | Low (build catches) | Exhaustive grep found 8 sites; all listed in Task 2 | Task 2 |

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
- [x] CLAUDE.md: no updates needed (no new files/packages/CLI flags)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (tasks are independent — any order works)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed
- [x] Construction site audit completed

**Antipattern rules:**
- [x] R1: No silent data loss (validation uses `logrus.Fatalf`, not silent `continue`)
- [x] R2: N/A (no map iteration changes)
- [x] R3: This PR ADDS validation for 6 previously unvalidated parameters
- [x] R4: `NewKVCapacityParams` constructor added; all sites migrated
- [x] R5: N/A (no resource allocation loops)
- [x] R6: Validation is in `cmd/` (CLI layer), not `sim/` — `logrus.Fatalf` is correct
- [x] R7: N/A (no golden test changes)
- [x] R8–R23: N/A (no changes in scope)

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go`

**Purpose:** Add 6 standalone validation checks for distribution-mode token flags.

**Insertion point:** After line 466 (end of `rate > 0` check), before line 467 (cross-field range check). Inside the `} else if workloadType == "distribution" {` branch.

**Code:** See Task 1, Step 1.

### File: `sim/latency/kv_capacity.go`

**Purpose:** Add canonical constructor and migrate `ExtractKVCapacityParams`.

**New function after line 19:** `NewKVCapacityParams(isMoE bool, numLocalExperts int, tieWordEmbeddings bool, hiddenAct string) KVCapacityParams`

**Modified function:** `ExtractKVCapacityParams` — see Task 2, Step 1 for complete replacement.

### File: `sim/latency/kv_capacity_test.go`

**Purpose:** Migrate 8 test construction sites to `latency.NewKVCapacityParams(...)`.

**Changes:** See Task 2, Step 2 for the full mapping.

### File: `sim/request.go`

**Purpose:** Remove `Streaming` field. Delete line 52.

### File: `sim/workload/generator.go`

**Purpose:** Remove `Streaming: client.Streaming,` from Request construction (line 272).

### File: `sim/workload/replay.go`

**Purpose:** Remove `Streaming: rec.Streaming,` from Request construction (line 54).

### File: `sim/workload/generator_test.go`

**Purpose:** Remove `req.Streaming ||` from condition (line 409). Update error message (line 410).

### File: `sim/kv/tiered_test.go`

**Purpose:** Add `TestTieredKVCache_OffloadReload_MechanismExercised` — see Task 4, Step 1 for complete code.

**Key notes:**
- Test uses 10 GPU blocks (blockSize=2), 10 CPU blocks, threshold 0.3
- 2-block target (4 tokens) → 3 others → release → offload → 3 fillers → 1 free → reload request
- `maybeOffload` preserves free count (remove + re-append as empty)
- Single-slot overwrite: block 0 hash destroyed when recycled for block 1 → chain breaks
- Allocation fails (expected). `cpuHitCount > 0` proves reload mechanism executed.
- Follow-up issue: `tryReloadFromCPU` overwrite limitation with hierarchical hashing
