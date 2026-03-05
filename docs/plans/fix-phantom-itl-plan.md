# Fix Phantom ITL Entry in processCompletions — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix a systematic off-by-one in ITL (inter-token latency) counting that inflates E2E and ITL metrics for every completed request.

**The problem today:** `processCompletions()` appends an ITL entry for the completion step, but `executeBatchStep()` already appended one for the same decode step earlier in the same `Step()` call. This produces N ITL entries for N output tokens instead of the correct N-1 (the number of inter-token gaps). E2E is overestimated by `currStepAdvance + OutputTokenProcessingTime()` per request (one full step time plus OTP), and per-request `itl_ms` is overestimated by a factor of N/(N-1). For short-output workloads (N=2-5), this is 25-100% overestimate.

**What this PR adds:**
1. Removes the phantom ITL append from `processCompletions` — eliminates the double-count
2. Adds an ITL count invariant test — asserts `len(req.ITL) == max(N-1, 0)` for all completed requests
3. Regenerates the golden dataset — updates E2E and ITL metric values to reflect corrected accounting

**Why this matters:** BLIS metrics are used for capacity planning and calibration against real vLLM servers. An inflated E2E/ITL makes the simulator appear slower than reality, leading to over-provisioning in capacity plans and poor calibration fits.

**Architecture:** Single-line removal in `sim/simulator.go` (line 398), plus test additions in `sim/metrics_substrate_test.go` and golden dataset updates in `testdata/goldendataset.json`. No new types, interfaces, or packages.

**Source:** GitHub issue #524

**Closes:** Fixes #524

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a metrics accounting bug in the core simulator's step execution pipeline. The `processCompletions` method (Phase 3 of `Step()`) appends an ITL entry for the final decode step, but `executeBatchStep` (Phase 2) already recorded the same step's ITL. Both use the same `currStepAdvance` from the same `Step()` call. The fix is a single-line removal.

The bug affects every completed request with output tokens. It does not affect TTFT, request conservation, KV block accounting, or scheduling delays. Only E2E and ITL metrics change.

Adjacent blocks: `recordRequestCompletion()` (consumes `req.ITL`), golden dataset tests (assert exact metric values), metrics substrate tests (verify E2E/ITL relationships).

No deviations from issue #524.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: ITL Count Matches Inter-Token Gaps
- GIVEN a completed request with N output tokens (N ≥ 1) that was not preempted during execution
- WHEN examining `req.ITL` after completion
- THEN `len(req.ITL)` MUST equal `N - 1`
- MECHANISM: Only `executeBatchStep` appends ITL entries (one per decode step after the first output token). `processCompletions` does not append.
- NOTE: For preempted requests, `req.ITL` accumulates entries across execution attempts (ProgressIndex resets to 0 on preemption but ITL slice is not cleared). This is pre-existing behavior and out of scope for this fix. The BC-MS-15 test exercises only non-preempted scenarios.

BC-2: ITL Count for Zero-Output Requests
- GIVEN a completed request with 0 output tokens
- WHEN examining `req.ITL` after completion
- THEN `len(req.ITL)` MUST equal 0
- MECHANISM: The `processCompletions` guard `if len(req.OutputTokens) > 0` prevents ITL append; executeBatchStep's decode branch never fires for zero-output requests.

BC-3: E2E Equals TTFT Plus Decode Time
- GIVEN a completed request with N output tokens
- WHEN computing E2E from `recordRequestCompletion`
- THEN `E2E = TTFT + sum(req.ITL)`, where BC-1 guarantees `len(req.ITL) == N-1`. Each ITL entry equals `currStepAdvance + OutputTokenProcessingTime()` for its respective decode step.
- MECHANISM: `recordRequestCompletion` sums `req.ITL` (now N-1 entries) and adds `FirstTokenTime`.

BC-4: Golden Dataset Consistency
- GIVEN the corrected ITL accounting
- WHEN running golden dataset test cases with the same seed and parameters
- THEN all golden metric assertions pass with updated E2E and ITL values
- MECHANISM: Golden dataset regenerated with corrected simulator output.

**Negative Contracts:**

NC-1: No Phantom ITL Entries
- GIVEN a request completing in `processCompletions`
- WHEN the completion state transition fires
- THEN `processCompletions` MUST NOT append to `req.ITL`
- MECHANISM: The `req.ITL = append(...)` line removed from `processCompletions`.

NC-2: TTFT Unchanged
- GIVEN the same simulation parameters and seed
- WHEN comparing TTFT before and after this fix
- THEN all TTFT values MUST be identical (TTFT is unaffected by ITL accounting)
- MECHANISM: TTFT recording in `executeBatchStep` (lines 363-368) is not modified.

NC-3: Request Conservation Unchanged
- GIVEN the same simulation parameters
- WHEN running with corrected ITL
- THEN `completed_requests`, `total_input_tokens`, `total_output_tokens` MUST be identical to pre-fix values
- MECHANISM: ITL accounting does not affect request lifecycle or token counting.

### C) Component Interaction

```
Step()
  ├── scheduleBatch()       [unchanged]
  ├── executeBatchStep()    [unchanged — already appends ITL correctly]
  │     └── for each request:
  │           prefill branch: advance ProgressIndex, no ITL
  │           decode branch:  ProgressIndex++, ITL append ← sole ITL source
  │           TTFT check:     set FirstTokenTime if prefill complete
  └── processCompletions()  [MODIFIED — remove ITL append at line 398]
        └── for each request:
              completion check → state transition, KV release
              ❌ REMOVED: req.ITL = append(...)  ← was double-counting
              ✅ KEPT: KV allocation for final token
              ✅ KEPT: recordRequestCompletion()
```

No new interfaces, types, or state changes. Extension friction: 0 files (no new fields or types).

### D) Deviation Log

No deviations from issue #524.

### E) Review Guide

**The tricky part:** Verifying the N=1 edge case. With one output token, prefill generates token_0 (TTFT set), then `processCompletions` fires immediately. After the fix, N=1 requests have 0 ITL entries → E2E = TTFT + 0 = TTFT. This is correct (a 1-token response completes at first-token time), but it changes the `recordRequestCompletion` formula: `reqTotalOutput = lat - FirstTokenTime = 0`, so `RequestITLs[id] = 0 / max(0, 1) = 0`.

**What to scrutinize:** The golden dataset values after regeneration — verify that TTFT values are unchanged, and that E2E values decreased by roughly `currStepAdvance + OutputTokenProcessingTime()` per request.

**What's safe to skim:** The test structure follows existing BC-MS patterns in `metrics_substrate_test.go`.

**Known debt:** The per-request `ITL` field in JSON output (line 142 of `metrics.go`) reports the per-request average (`sum(ITL) / max(N-1, 1)`), which is TPOT but labeled as "ITL". This naming inconsistency predates this PR and is out of scope.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator.go:398` — Remove phantom ITL append (1 line)
- `sim/simulator.go:394-396` — Update comment to reflect removal
- `sim/metrics_substrate_test.go` — Add BC-MS-15: ITL count invariant test
- `testdata/goldendataset.json` — Regenerate E2E and ITL metric values

**Key decisions:**
- Remove rather than move the ITL append (executeBatchStep already handles it)
- Add the invariant test as a new behavioral contract (BC-MS-15) in the existing substrate test file
- Regenerate goldens by running tests, capturing actual values from failure output

### G) Task Breakdown

---

#### Task 1: Add ITL Count Invariant Test (BC-1, BC-2)

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/metrics_substrate_test.go` (add new test function)

**Step 1: Write failing test for ITL count invariant**

Context: We need a test that asserts `len(req.ITL) == max(N-1, 0)` for completed requests. This test will FAIL with the current code because the phantom entry produces N entries instead of N-1.

In `sim/metrics_substrate_test.go`, add at the end of the file:

```go
// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-15: ITL Count Invariant
//
// For every completed request with N output tokens:
//   len(req.ITL) == max(N-1, 0)
// This is the number of inter-token gaps: N tokens produce N-1 gaps.
// Zero-output requests have 0 ITL entries.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_ITLCount_MatchesInterTokenGaps(t *testing.T) {
	tests := []struct {
		name      string
		outputLen int
		wantITL   int
	}{
		{"zero-output", 0, 0},
		{"single-token", 1, 0},
		{"two-tokens", 2, 1},
		{"five-tokens", 5, 4},
		{"ten-tokens", 10, 9},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := msConfig(math.MaxInt64)
			s := mustNewSimulator(t, cfg)
			req := &Request{
				ID:           "itl-count",
				InputTokens:  msMakeTokens(32),
				OutputTokens: msMakeTokens(tc.outputLen),
				ArrivalTime:  0,
				State:        StateQueued,
			}
			s.InjectArrival(req)
			s.Run()

			if s.Metrics.CompletedRequests != 1 {
				t.Fatalf("expected 1 completed, got %d", s.Metrics.CompletedRequests)
			}

			gotITL := len(s.Metrics.AllITLs)
			if gotITL != tc.wantITL {
				t.Errorf("BC-MS-15 violated: len(AllITLs) = %d, want %d for %d output tokens",
					gotITL, tc.wantITL, tc.outputLen)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestMetrics_ITLCount -v`
Expected: FAIL — cases with outputLen >= 1 will show `len(AllITLs) = N, want N-1`

**Step 3: No implementation yet — this is the red phase of TDD**

The fix comes in Task 2. This test establishes the invariant that will drive the code change.

**Step 4: Commit the failing test (commented out with skip)**

Since TDD requires a failing test but we can't commit broken tests, add `t.Skip("BC-MS-15: enable after removing phantom ITL append from processCompletions")` as the first line of the test function body.

```bash
git add sim/metrics_substrate_test.go
git commit -m "test(sim): add ITL count invariant test BC-MS-15 (skipped, #524)

- Add TestMetrics_ITLCount_MatchesInterTokenGaps
- Asserts len(ITL) == max(N-1, 0) for completed requests
- Currently skipped: will be enabled after fix in next commit

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Remove Phantom ITL Append (NC-1)

**Contracts Implemented:** NC-1, BC-1, BC-3

**Files:**
- Modify: `sim/simulator.go:394-398` — Remove ITL append and update comment

**Step 1: The test from Task 1 defines the expected behavior**

Remove the `t.Skip(...)` line from `TestMetrics_ITLCount_MatchesInterTokenGaps`.

**Step 2: Run test to verify it fails (red)**

Run: `go test ./sim/... -run TestMetrics_ITLCount -v`
Expected: FAIL — phantom entry still present

**Step 3: Implement the fix**

In `sim/simulator.go`, modify `processCompletions`. Remove the ITL append at line 398 and update the comment at lines 394-396.

Replace:
```go
			// Zero-output requests complete at prefill end with no decode phase.
			// Both the completion-step ITL and the final-token KV allocation
			// only apply to requests that have output tokens.
			if len(req.OutputTokens) > 0 {
				req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
```

With:
```go
			// Zero-output requests complete at prefill end with no decode phase.
			// The final-token KV allocation only applies to requests that have
			// output tokens. ITL is NOT appended here — executeBatchStep already
			// recorded it for this decode step (fix for #524 phantom ITL entry).
			if len(req.OutputTokens) > 0 {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
```

**Step 4: Run test to verify it passes (green)**

Run: `go test ./sim/... -run TestMetrics_ITLCount -v`
Expected: PASS — all cases now have correct ITL count

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/metrics_substrate_test.go
git commit -m "fix(sim): remove phantom ITL entry from processCompletions (#524)

- Remove req.ITL append from processCompletions (line 398)
- executeBatchStep already records ITL for the final decode step
- Enable BC-MS-15 invariant test: len(ITL) == max(N-1, 0)
- Fixes E2E overestimate of currStepAdvance + OTP per request
- Fixes per-request itl_ms overestimate of N/(N-1) factor

Fixes #524

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Verify Existing Behavioral Contracts Still Hold (BC-3, NC-2, NC-3)

**Contracts Implemented:** BC-3, NC-2, NC-3

**Files:**
- No new files — run existing tests to verify

**Step 1: Run metrics substrate tests**

Run: `go test ./sim/... -run TestMetrics_ -v`

Expected: All BC-MS-1 through BC-MS-14 tests PASS. These verify internal consistency (E2E = TTFT + sum(ITL), mean ITL consistency, zero-output behavior). The relationships are still valid — they're just computed with N-1 entries instead of N.

**Step 2: Run causality tests**

Run: `go test ./sim/... -run TestSimulator_Causality -v`

Expected: PASS — all ITL values still non-negative, E2E >= TTFT.

**Step 3: Run full test suite (expect golden failures)**

Run: `go test ./sim/... -v 2>&1 | grep -E "FAIL|PASS|---"`

Expected: Golden dataset tests FAIL (E2E and ITL values changed). All other tests PASS. This confirms NC-2 (TTFT unchanged) and NC-3 (request counts unchanged) — golden failures will only be on E2E/ITL fields.

No commit for this task — it's a verification step.

---

#### Task 4: Regenerate Golden Dataset (BC-4)

**Contracts Implemented:** BC-4

**Files:**
- Modify: `testdata/goldendataset.json` — Update E2E and ITL metric values

**Step 1: Capture corrected golden values**

Run the golden test with verbose output to see actual vs expected:

Run: `go test ./sim/... -run TestSimulator_GoldenDataset -v 2>&1`

The `AssertFloat64Equal` helper prints both expected and actual values on failure. Capture the actual values for each test case's E2E and ITL fields.

**Step 2: Update golden dataset JSON**

For each of the 5 test cases in `testdata/goldendataset.json`, update the following fields with the actual values from Step 1:
- `e2e_mean_ms`
- `e2e_p90_ms`
- `e2e_p95_ms`
- `e2e_p99_ms`
- `itl_mean_ms`
- `itl_p90_ms`
- `itl_p95_ms`
- `itl_p99_ms`

**Important:** Do NOT modify these fields (they should be unchanged):
- `completed_requests`, `total_input_tokens`, `total_output_tokens`
- `vllm_estimated_duration_s`, `responses_per_sec`, `tokens_per_sec`
- `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms`
- `scheduling_delay_p99_ms`

**Step 3: Verify golden tests pass**

Run: `go test ./sim/... -run Golden -v`
Expected: ALL golden tests PASS (sim, cluster/instance, cluster/cluster)

**Step 4: Verify TTFT values are unchanged**

Manually compare the `ttft_*` fields in the JSON — they must be identical to the pre-fix values.

**Step 5: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL tests PASS

**Step 6: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 7: Commit**

```bash
git add testdata/goldendataset.json
git commit -m "test(golden): regenerate dataset after phantom ITL fix (#524)

- Update E2E and ITL metrics for all 5 golden test cases
- E2E values decreased by ~one step time + OTP per request
- ITL percentiles shifted due to N-1 entries instead of N
- TTFT, request counts, throughput metrics unchanged (verified)
- Regeneration triggered by fix in previous commit (R12)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Invariant | TestMetrics_ITLCount_MatchesInterTokenGaps |
| BC-2 | Task 1 | Invariant | TestMetrics_ITLCount_MatchesInterTokenGaps (zero-output case) |
| BC-3 | Task 3 | Invariant | TestMetrics_E2E_Identity_SingleRequest (existing BC-MS-1) |
| BC-4 | Task 4 | Golden | TestSimulator_GoldenDataset (existing) |
| NC-1 | Task 2 | Invariant | TestMetrics_ITLCount_MatchesInterTokenGaps (all cases) |
| NC-2 | Task 4 | Golden | TTFT fields unchanged in goldendataset.json |
| NC-3 | Task 4 | Golden | Request count fields unchanged in goldendataset.json |

**Golden dataset update:** Task 4. Regeneration by running `go test ./sim/... -run TestSimulator_GoldenDataset -v` and capturing actual values.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| N=1 edge case breaks E2E calculation | Low | High | Table-driven test covers N=0,1,2,5,10 | Task 1 |
| Golden dataset TTFT values accidentally modified | Low | Medium | Manual verification step in Task 4 | Task 4 |
| Cluster-level golden tests fail unexpectedly | Low | Medium | Task 3 runs full suite before golden regen | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — single-line removal + invariant test
- [x] No feature creep — strictly issue #524 scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — metrics change but API is unchanged
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (`msConfig`, `mustNewSimulator`, `msMakeTokens` from existing substrate tests)
- [x] CLAUDE.md — no updates needed (no new files/packages/flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — no canonical sources modified
- [x] Deviation log reviewed — no deviations
- [x] Each task produces testable code
- [x] Task dependencies correctly ordered (1→2→3→4)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration documented (Task 4)
- [x] Construction site audit — no struct fields added

**Antipattern rules:**
- [x] R1: No silent data loss — removing an incorrect append, not adding a silent drop
- [x] R4: No struct field changes
- [x] R7: Invariant test added (BC-MS-15) alongside golden test update
- [x] R11: No new division — existing `max(N-1, 1)` denominator guard unchanged
- [x] R12: Golden dataset regenerated (Task 4)

---

## Appendix: File-Level Implementation Details

### File: `sim/simulator.go`

**Purpose:** Remove phantom ITL append from `processCompletions` (line 398)

**Exact change** (lines 394-398):

Before:
```go
// Zero-output requests complete at prefill end with no decode phase.
// Both the completion-step ITL and the final-token KV allocation
// only apply to requests that have output tokens.
if len(req.OutputTokens) > 0 {
    req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
    ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
```

After:
```go
// Zero-output requests complete at prefill end with no decode phase.
// The final-token KV allocation only applies to requests that have
// output tokens. ITL is NOT appended here — executeBatchStep already
// recorded it for this decode step (fix for #524 phantom ITL entry).
if len(req.OutputTokens) > 0 {
    ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
```

### File: `sim/metrics_substrate_test.go`

**Purpose:** Add BC-MS-15 ITL count invariant test

**New test function** (added at end of file):

```go
func TestMetrics_ITLCount_MatchesInterTokenGaps(t *testing.T) {
	tests := []struct {
		name      string
		outputLen int
		wantITL   int
	}{
		{"zero-output", 0, 0},
		{"single-token", 1, 0},
		{"two-tokens", 2, 1},
		{"five-tokens", 5, 4},
		{"ten-tokens", 10, 9},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := msConfig(math.MaxInt64)
			s := mustNewSimulator(t, cfg)
			req := &Request{
				ID:           "itl-count",
				InputTokens:  msMakeTokens(32),
				OutputTokens: msMakeTokens(tc.outputLen),
				ArrivalTime:  0,
				State:        StateQueued,
			}
			s.InjectArrival(req)
			s.Run()

			if s.Metrics.CompletedRequests != 1 {
				t.Fatalf("expected 1 completed, got %d", s.Metrics.CompletedRequests)
			}

			gotITL := len(s.Metrics.AllITLs)
			if gotITL != tc.wantITL {
				t.Errorf("BC-MS-15 violated: len(AllITLs) = %d, want %d for %d output tokens",
					gotITL, tc.wantITL, tc.outputLen)
			}
		})
	}
}
```

### File: `testdata/goldendataset.json`

**Purpose:** Update E2E and ITL metric values after fix

Only the following fields change per test case:
- `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms`
- `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms`

All other fields (TTFT, request counts, throughput, scheduling delay) remain identical.
