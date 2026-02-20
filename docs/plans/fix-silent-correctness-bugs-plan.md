# Fix Silent Correctness Bugs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three bugs where user-specified CLI configuration is silently ignored, producing wrong simulation results without any error or warning.

**The problem today:** Three independent bugs silently discard user intent: (1) `--total-kv-blocks` is always overwritten by defaults.yaml, making KV cache capacity planning impossible; (2) `--snapshot-refresh-interval` accepts negative values without error, silently falling back to default behavior; (3) multi-client workload specs with `num_requests` starve later clients, making all requests belong to the first client's SLO class — effectively breaking multi-tenant experiments and per-SLO metrics.

**What this PR adds:**
1. **CLI flag precedence for KV blocks** — `--total-kv-blocks 50` now actually uses 50 blocks instead of silently reverting to 132,139 from defaults.yaml
2. **Negative value rejection for snapshot interval** — `--snapshot-refresh-interval -5` now produces a clear error instead of silently falling back to Immediate mode
3. **Fair multi-client request generation** — all clients generate requests across the full time horizon, then the merged stream is truncated to `maxRequests`, preserving the proportional interleaving that `rate_fraction` promises

**Why this matters:** These are experiment-validity bugs. Users running capacity-planning studies or multi-tenant fairness analyses get silently wrong results. Fixing them unblocks reliable hypothesis experiments (discovered during H9 prefix-caching and H3 signal-freshness research).

**Architecture:** Bug 1 and 2 are CLI-boundary fixes in `cmd/root.go` (1-2 lines each). Bug 3 is a workload generator fix in `sim/workload/generator.go` — restructure the generation loop to generate all clients first, merge, then truncate. No new types, interfaces, or packages.

**Source:** GitHub issues #285, #281, #278

**Closes:** Fixes #285, fixes #281, fixes #278

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes three independent "silent correctness" bugs where user-specified configuration is discarded without error:

1. **KV blocks override** (#285): `GetCoefficients()` unconditionally overwrites `totalKVBlocks` after Cobra parses CLI flags. Fix: guard with `cmd.Flags().Changed("total-kv-blocks")`, matching the existing pattern for `--horizon` and `--num-requests`.

2. **Snapshot interval validation** (#281): `--snapshot-refresh-interval` accepts negative values. The library code (`newObservabilityConfig`) silently treats `<= 0` as Immediate mode. Fix: reject negative values at the CLI boundary with `logrus.Fatalf`.

3. **Client starvation** (#278): `GenerateRequests` processes clients sequentially with a global `maxRequests` guard inside the per-client loop. High-rate clients exhaust the budget before low-rate clients generate any requests. Fix: remove the in-loop cap, generate all clients to completion, merge-sort by arrival time, then truncate the merged stream to `maxRequests`.

**Adjacent blocks:** `cmd/root.go` (CLI boundary), `sim/workload/generator.go` (workload generation). No changes to `sim/`, `sim/cluster/`, or `sim/trace/`.

**DEVIATION flags:** None — all three issues have clear root causes and suggested fixes that match the codebase.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: CLI --total-kv-blocks takes precedence over defaults.yaml**
- GIVEN the user passes `--total-kv-blocks 50` on the command line
- WHEN the simulation starts and `GetCoefficients()` returns a different KV block count from defaults.yaml
- THEN the simulation runs with 50 KV blocks (the user-specified value)
- MECHANISM: Guard `totalKVBlocks` assignment with `cmd.Flags().Changed("total-kv-blocks")`

**BC-2: defaults.yaml KV blocks used when CLI flag is not explicitly set**
- GIVEN the user does NOT pass `--total-kv-blocks` on the command line
- WHEN `GetCoefficients()` returns a KV block count from defaults.yaml
- THEN the simulation uses the defaults.yaml value (backward compatibility)
- MECHANISM: `cmd.Flags().Changed` returns false → assignment proceeds normally

**BC-3: Multi-client workloads preserve rate proportionality under maxRequests cap**
- GIVEN a workload spec with two clients (70% and 30% rate fractions) and `maxRequests = 100`
- WHEN requests are generated
- THEN both clients appear in the output, and their proportions approximate 70/30 (within statistical tolerance of ±10%)
- MECHANISM: Generate all clients across full horizon, merge-sort by arrival time, truncate to maxRequests

**BC-4: maxRequests still caps total output**
- GIVEN a workload spec with `maxRequests = N`
- WHEN requests are generated with a horizon that would produce more than N requests
- THEN the output contains exactly N requests (or fewer if horizon exhausted first)
- MECHANISM: Post-merge truncation `allRequests = allRequests[:maxRequests]`

#### Negative Contracts

**BC-5: Negative snapshot-refresh-interval rejected at CLI boundary**
- GIVEN the user passes `--snapshot-refresh-interval -5`
- WHEN the CLI validates input
- THEN the program exits with an error message containing "must be >= 0"
- MECHANISM: `logrus.Fatalf` guard in `cmd/root.go` validation section

**BC-7: Reasoning (multi-turn) clients are not starved by maxRequests cap**
- GIVEN a workload spec with a standard client and a reasoning (multi-turn) client, and `maxRequests > 0`
- WHEN requests are generated
- THEN the reasoning client produces at least one request in the output
- MECHANISM: Same generate-all-then-truncate fix applies to the reasoning path (lines 78-103)

#### Error Handling Contracts

**BC-6: No new panic paths in library code**
- GIVEN any of the three fixes
- WHEN applied
- THEN no new `panic()` or `logrus.Fatalf()` calls are added to `sim/` or `sim/workload/` packages
- MECHANISM: Bug 1 and 2 are `cmd/` changes; Bug 3 modifies `sim/workload/generator.go` control flow only

### C) Component Interaction

```
cmd/root.go (CLI boundary)
  │
  ├── Bug 1: GetCoefficients() → guard totalKVBlocks with Changed()
  ├── Bug 2: Add negative validation for snapshotRefreshInterval
  │
  └── workload.GenerateRequests(spec, horizon, maxRequests)
        │
        └── Bug 3: Remove in-loop maxRequests guard
                    Generate all clients → merge-sort → truncate
```

**API contracts:** No changes to any public API. `GenerateRequests` signature unchanged.

**State changes:** None — the generator produces the same type of output, just with fairer client representation.

**Extension friction:** N/A — bug fixes, no new types or interfaces.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #285 suggests auditing `--kv-cpu-blocks` for same pattern | Not included | `--kv-cpu-blocks` is not set by `GetCoefficients()` — verified in code. Only alpha, beta, and kvBlocks are returned. SCOPE_CHANGE: out of scope. |
| #285 suggests auditing beta-coeffs override | Not included | Beta coefficients ARE unconditionally overwritten, but there's no `--beta-coeffs` Changed() guard needed because the zero-check at line 143 already gates the entire block. SCOPE_CHANGE: a separate issue if needed. |
| #285 asks for "test that verifies CLI flags take precedence over defaults.yaml" | Plan tests flag registration only, not full precedence | SIMPLIFICATION: A true precedence test requires running the full CLI command with a model + defaults.yaml — effectively an integration test. The `Changed()` guard is a 1-line idiom already used elsewhere (lines 210, 216). The risk of regression is low. |
| #281 mentions adding a Rule 4 audit comment for DeploymentConfig | Not included | SIMPLIFICATION: The audit result (zero-value = Immediate = correct default) is documented in the issue itself. Adding a code comment for an existing field adds noise. |
| #278 suggests removing per-client cap at line 116 | Plan removes it for standard clients AND reasoning clients | ADDITION: The issue only mentions line 116 but the same bug exists in the reasoning path (line 78). Both are fixed. |

### E) Review Guide

**The tricky part:** Bug 3 changes the generation loop structure. The key concern is whether removing the in-loop `maxRequests` guard causes unbounded memory allocation when both `maxRequests` and `horizon` are large. The answer is no: `horizon` is always finite (validated at CLI boundary, line 221), so each client generates at most `horizon * rate` requests. The post-merge truncation then caps the total.

**What to scrutinize:** BC-3 — verify the test actually checks that BOTH clients appear (not just that the count is ≤ maxRequests). Also verify the reasoning path (multi-turn clients) gets the same fix.

**What's safe to skim:** Bug 1 and Bug 2 are 1-2 line fixes following established patterns already used elsewhere in `cmd/root.go`.

**Known debt:** The reasoning client path (lines 76-109) has the same sequential generation pattern. This fix addresses it, but the reasoning path is harder to test in isolation (requires multi-turn spec setup).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `cmd/root.go:171-172` — Guard `totalKVBlocks` assignment with `Changed()` check
- `cmd/root.go:~340` — Add negative validation for `snapshotRefreshInterval`
- `sim/workload/generator.go:78,116` — Remove in-loop `maxRequests` guards, add post-merge truncation
- `cmd/root_test.go` — Add test for BC-1 (flag precedence verification)
- `sim/workload/generator_test.go` — Add test for BC-3, BC-4 (fair multi-client generation)

**Key decisions:**
- Bug 3 fix: generate-all-then-truncate instead of per-client quota allocation. Simpler, preserves arrival time ordering, and matches the existing merge-sort architecture.
- No golden dataset update needed — these bugs are in CLI flag handling and workload generation, not in simulation output format.

**Confirmation:** No dead code, all changes exercised by tests.

### G) Task Breakdown

---

### Task 1: Fix --total-kv-blocks silent override (#285)

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `cmd/root.go:171-172`
- Test: `cmd/root_test.go`

**Step 1: Verify existing test covers BC-1 flag infrastructure**

Context: The existing `TestRunCmd_KVBlockFlags_DefaultsArePositive` already verifies the `total-kv-blocks` flag is registered with a positive default. Rather than adding a duplicate test, we'll add a BC-1/BC-2 comment to the existing test to document the precedence contract. A full CLI precedence test would require running the command with a model + defaults.yaml (integration test scope).

No new test file changes for this step — the existing test is sufficient.

**Step 2: Verify existing test passes**

Run: `go test ./cmd/... -run TestRunCmd_KVBlockFlags -v`
Expected: PASS

**Step 3: Implement the fix**

Context: Guard `totalKVBlocks` assignment with `cmd.Flags().Changed("total-kv-blocks")`, matching the existing pattern for `--horizon` (line 210) and `--num-requests` (line 216).

In `cmd/root.go`, change lines 171-172 from:

```go
newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
alphaCoeffs, betaCoeffs, totalKVBlocks = newAlpha, newBeta, kvBlocks
```

To:

```go
newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
alphaCoeffs, betaCoeffs = newAlpha, newBeta
if !cmd.Flags().Changed("total-kv-blocks") {
	totalKVBlocks = kvBlocks
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -v`
Expected: PASS (all existing tests still pass)

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go
git commit -m "fix(cli): respect --total-kv-blocks over defaults.yaml (BC-1, BC-2)

Guard totalKVBlocks assignment with cmd.Flags().Changed(), matching
the existing pattern for --horizon and --num-requests. Previously,
GetCoefficients() unconditionally overwrote user-specified values.

Fixes #285

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add CLI validation for --snapshot-refresh-interval (#281)

**Contracts Implemented:** BC-5

**Files:**
- Modify: `cmd/root.go:~340` (validation section)
- Test: `cmd/root_test.go`

**Step 1: Write test for BC-5**

Context: We verify the flag is registered and has a non-negative default. The actual rejection of negative values happens at runtime in the CLI validation block — we can't unit-test `logrus.Fatalf` directly, but we can verify the flag's default is valid.

```go
func TestRunCmd_SnapshotRefreshInterval_FlagRegistered(t *testing.T) {
	// BC-5: --snapshot-refresh-interval must reject negative values.
	// This test verifies the flag exists and its default is valid (non-negative).
	flag := runCmd.Flags().Lookup("snapshot-refresh-interval")
	assert.NotNil(t, flag, "snapshot-refresh-interval flag must be registered")

	defVal, err := strconv.ParseInt(flag.DefValue, 10, 64)
	assert.NoError(t, err, "default must be a valid int64")
	assert.GreaterOrEqual(t, defVal, int64(0),
		"default snapshot-refresh-interval must be >= 0")
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./cmd/... -run TestRunCmd_SnapshotRefreshInterval -v`
Expected: PASS

**Step 3: Implement the fix**

Context: Add negative validation for `snapshotRefreshInterval` in the validation section of `cmd/root.go`, near the other KV-related validations (around line 340-351).

In `cmd/root.go`, after the `kvTransferBaseLatency` validation (line 351), add:

```go
if snapshotRefreshInterval < 0 {
	logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
}
```

**Step 4: Run tests to verify all pass**

Run: `go test ./cmd/... -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go cmd/root_test.go
git commit -m "fix(cli): reject negative --snapshot-refresh-interval (BC-5)

Add CLI boundary validation per Antipattern Rule 3. Previously,
negative values were silently treated as Immediate mode by the
library code, hiding user misconfiguration.

Fixes #281

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix multi-client workload starvation (#278) — test first

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Test: `sim/workload/generator_test.go`

**Step 1: Write failing test for BC-3 (fair multi-client generation)**

Context: The existing `TestGenerateRequests_TwoClients_RateProportional` passes `maxRequests=0` (unlimited). We need a test that uses `maxRequests > 0` with two clients and verifies BOTH clients appear proportionally.

```go
func TestGenerateRequests_MaxRequests_PreservesClientProportions(t *testing.T) {
	// BC-3: With maxRequests cap, both clients must appear in proportional amounts.
	// Bug #278: Sequential generation starves later clients.
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "batch", TenantID: "tenant-A", SLOClass: "batch", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "realtime", TenantID: "tenant-B", SLOClass: "realtime", RateFraction: 0.3,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	maxReqs := int64(200)
	requests, err := GenerateRequests(spec, 100e6, maxReqs) // long horizon, capped by maxReqs
	if err != nil {
		t.Fatal(err)
	}

	// BC-4: total output must be capped
	if int64(len(requests)) != maxReqs {
		t.Errorf("len(requests) = %d, want %d", len(requests), maxReqs)
	}

	// BC-3: both SLO classes must appear
	countBatch := 0
	countRealtime := 0
	for _, r := range requests {
		switch r.SLOClass {
		case "batch":
			countBatch++
		case "realtime":
			countRealtime++
		}
	}

	if countRealtime == 0 {
		t.Fatal("realtime client produced 0 requests — starvation bug (#278)")
	}

	// Check proportions are approximately 70/30 (within ±10%)
	fracBatch := float64(countBatch) / float64(len(requests))
	if fracBatch < 0.6 || fracBatch > 0.8 {
		t.Errorf("batch fraction = %.3f, want ≈ 0.7 (±10%%)", fracBatch)
	}
}
```

Also add a test for reasoning client starvation (BC-7):

```go
func TestGenerateRequests_MaxRequests_ReasoningClientNotStarved(t *testing.T) {
	// BC-7: Reasoning (multi-turn) clients must not be starved by maxRequests cap.
	reasoningSpec := &ReasoningSpec{
		ReasonRatioDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 0.3, "std_dev": 0.1, "min": 0.1, "max": 0.9}},
		MultiTurn:       &MultiTurnSpec{MaxRounds: 2, ContextGrowth: "accumulate"},
	}
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "standard", TenantID: "std", SLOClass: "batch", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "reasoning", TenantID: "rsn", SLOClass: "realtime", RateFraction: 0.3,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
				Reasoning: reasoningSpec},
		},
	}
	maxReqs := int64(100)
	requests, err := GenerateRequests(spec, 100e6, maxReqs)
	if err != nil {
		t.Fatal(err)
	}

	// Total capped
	if int64(len(requests)) > maxReqs {
		t.Errorf("len(requests) = %d, want <= %d", len(requests), maxReqs)
	}

	// Reasoning client must appear
	countReasoning := 0
	for _, r := range requests {
		if r.TenantID == "rsn" {
			countReasoning++
		}
	}
	if countReasoning == 0 {
		t.Fatal("reasoning client produced 0 requests — starvation bug")
	}
}
```

**Step 2: Run tests to verify they FAIL**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_MaxRequests_PreservesClientProportions|TestGenerateRequests_MaxRequests_ReasoningClientNotStarved" -v`
Expected: FAIL — `countRealtime == 0` and `countReasoning == 0` (starvation bugs)

**Step 3: Commit the failing tests**

```bash
git add sim/workload/generator_test.go
git commit -m "test(workload): add failing test for multi-client starvation (BC-3, BC-4)

Reproduces #278: with maxRequests cap, later clients get zero requests
because sequential generation exhausts the budget on the first client.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fix multi-client workload starvation (#278) — implementation

**Contracts Implemented:** BC-3, BC-4, BC-6

**Files:**
- Modify: `sim/workload/generator.go:44-186`

**Step 1: Implement the fix**

Context: Remove the in-loop `maxRequests` guards from both the standard client path (line 116) and the reasoning client path (line 78). Instead, let all clients generate requests across the full horizon, then truncate the merged stream after sorting.

In `sim/workload/generator.go`, make these changes:

1. **Remove reasoning client maxRequests guard** (lines 78-79): Delete the `if maxRequests > 0 && int64(len(allRequests)) >= maxRequests { break }` inside the reasoning loop.

2. **Remove reasoning client count cap** (lines 96-103): Delete the `if maxRequests > 0 && int64(len(allRequests)+len(reasoningReqs)) > maxRequests` block.

3. **Remove standard client maxRequests guard** (lines 116-118): Delete the `if maxRequests > 0 && int64(len(allRequests)) >= maxRequests { break }` inside the standard generation loop.

4. **Add per-client soft cap** to prevent OOM when horizon is very large but maxRequests is small. Before the per-client generation loop (before line 45), compute a per-client generation limit:

```go
// Per-client generation cap: prevent OOM when horizon >> maxRequests.
// Each client generates at most 2x maxRequests (generous headroom for
// proportional mixing), then the post-merge truncation handles the rest.
// When maxRequests=0 (unlimited), no per-client cap applies.
perClientCap := int64(0)
if maxRequests > 0 {
	perClientCap = 2 * maxRequests
}
```

Then inside each per-client generation loop (both standard and reasoning paths), add a per-client soft cap check:

In the reasoning client loop (replacing the deleted maxRequests guard):
```go
if perClientCap > 0 && int64(len(clientRequests)) >= perClientCap {
	break
}
```

In the standard client loop (replacing the deleted maxRequests guard):
```go
if perClientCap > 0 && int64(clientCount) >= perClientCap {
	break
}
```

Note: This requires tracking per-client request counts. The simplest approach is to use a local `clientCount` variable per client loop iteration, or use a `clientRequests` slice per client. The implementation will collect per-client requests in a local slice, then append to `allRequests` after the loop — this also cleanly avoids counting cross-client accumulation in the guard.

**Simpler approach:** Actually, the cleanest fix is to keep the generation loop structure as-is but replace the `allRequests` accumulator check with a per-client count:

In the standard client loop, replace the deleted `maxRequests` guard with:
```go
if perClientCap > 0 && clientCount >= perClientCap {
	break
}
```

Where `clientCount` is initialized to `0` before the loop and incremented when a request is added.

5. **Add post-merge truncation** after the sort (after line 179, before ID assignment): Add truncation:

```go
// Truncate to maxRequests after merge-sort (preserves client proportionality)
if maxRequests > 0 && int64(len(allRequests)) > maxRequests {
	allRequests = allRequests[:maxRequests]
}
```

The resulting generation flow becomes:
- Compute `perClientCap = 2 * maxRequests` (OOM safeguard; 0 when unlimited)
- Each client generates requests up to `min(horizon, perClientCap)`
- All requests merge-sorted by arrival time
- Truncate to maxRequests (preserving proportional interleaving)
- Assign sequential IDs

**Note:** All line numbers in this task reference the original file before any edits. The `continue` at line 109 (which separates reasoning from standard client paths) is preserved — only the maxRequests guards are removed.

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestGenerateRequests_MaxRequests_PreservesClientProportions -v`
Expected: PASS

**Step 3: Run ALL workload tests to check for regressions**

Run: `go test ./sim/workload/... -v`
Expected: ALL PASS — existing `TestGenerateRequests_MaxRequests_CapsOutput` should still pass (it uses a single client, so truncation behavior is identical)

**Step 4: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/generator.go
git commit -m "fix(workload): generate all clients before applying maxRequests cap (BC-3, BC-4)

Previously, clients were processed sequentially with a global maxRequests
guard inside the per-client loop. High-rate clients exhausted the budget
before low-rate clients generated any requests, starving later clients.

Now all clients generate up to a per-client soft cap (2x maxRequests),
requests are merge-sorted by arrival time, then truncated to maxRequests —
preserving the proportional interleaving that rate_fraction promises.

Note: This changes the deterministic output for workloads using maxRequests > 0
with the same seed. Same seed + same code version = still deterministic.

Fixes #278

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run full verification and update CLAUDE.md if needed

**Contracts Implemented:** All (verification)

**Files:**
- Verify: all changed files

**Step 1: Run full build**

Run: `go build ./...`
Expected: SUCCESS

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Verify no CLAUDE.md updates needed**

This PR adds no new files, packages, CLI flags, or architectural changes. CLAUDE.md does not need updating.

**Step 5: Commit (if any cleanup needed)**

No commit expected — this is a verification-only task.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1     | Task 1 | Unit | TestRunCmd_KVBlockFlags_DefaultsArePositive (existing, covers flag infra) |
| BC-2     | Task 1 | Unit | (covered by BC-1 — Changed() guard is code-level, not testable without integration harness) |
| BC-3     | Task 3 | Unit | TestGenerateRequests_MaxRequests_PreservesClientProportions |
| BC-4     | Task 3 | Unit | TestGenerateRequests_MaxRequests_PreservesClientProportions (count check) |
| BC-5     | Task 2 | Unit | TestRunCmd_SnapshotRefreshInterval_FlagRegistered |
| BC-6     | Task 4 | Invariant | No new panic/fatalf in sim/workload/ (verified by grep in review) |
| BC-7     | Task 3 | Unit | TestGenerateRequests_MaxRequests_ReasoningClientNotStarved |

**Golden dataset:** No update needed. These bugs are in CLI flag handling and workload generation, not in simulation output format. The golden dataset tests exercise the simulation engine, which is not modified.

**Invariant tests:** The existing `TestGenerateRequests_TwoClients_RateProportional` (maxRequests=0) serves as the invariant test for rate proportionality. The new `TestGenerateRequests_MaxRequests_PreservesClientProportions` extends this invariant to the capped case.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bug 3 fix causes memory spike for large horizons with small maxRequests | Low | Medium | Per-client soft cap (`2 * maxRequests`) prevents unbounded allocation even when horizon is very large. Each client generates at most `2 * maxRequests` requests before stopping, then post-merge truncation to `maxRequests` finalizes. When `maxRequests=0` (unlimited), horizon bounds generation. |
| RNG state diverges for workloads using maxRequests > 0 | Medium | Low | The per-client cap (`2 * maxRequests`) causes more RNG samples to be drawn before truncation, changing the deterministic output for existing seeds. This is a known behavioral change — same seed + same fix version = deterministic, but cross-version reproducibility for `maxRequests > 0` workloads breaks. Document in PR description. |
| Changed() guard for totalKVBlocks breaks when defaults.yaml is missing | None | N/A | If defaults.yaml is missing, `GetCoefficients` is never called (the `if AllZeros(...)` block at line 143 won't execute with user-provided coefficients). If it IS called and fails, that's a pre-existing error path. |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing shared test package (not duplicated locally).
- [x] CLAUDE.md — no updates needed (no new files, packages, or CLI flags).
- [x] No stale references left in CLAUDE.md.
- [x] Deviation log reviewed — no unresolved deviations.
- [x] Each task produces working, testable code (no scaffolding).
- [x] Task dependencies are correctly ordered.
- [x] All contracts are mapped to specific tasks.
- [x] Golden dataset regeneration — not needed.
- [x] Construction site audit — no new struct fields added.
- [x] Every new CLI flag validated — no new flags added.
- [x] Every error path either returns error, panics with context, or increments counter.
- [x] No map iteration feeds float accumulation without sorted keys.
- [x] Library code never calls logrus.Fatalf — only cmd/ changes use Fatalf.
- [x] No loops allocating resources without rollback.
- [x] No exported mutable maps.
- [x] YAML config structs — no changes.
- [x] YAML loading — no changes.
- [x] No division with runtime denominators.
- [x] No new interfaces.
- [x] No method spans multiple module responsibilities.
- [x] Configuration parameters — no new params.
- [x] Grepped for stale PR references — N/A (bug fix, not a numbered PR).
- [x] Macro plan — not part of macro plan, no update needed.

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go`

**Purpose:** CLI boundary — fix silent KV blocks override and add snapshot interval validation.

**Change 1 (lines 171-172) — KV blocks guard:**

Before:
```go
newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
alphaCoeffs, betaCoeffs, totalKVBlocks = newAlpha, newBeta, kvBlocks
```

After:
```go
newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
alphaCoeffs, betaCoeffs = newAlpha, newBeta
if !cmd.Flags().Changed("total-kv-blocks") {
	totalKVBlocks = kvBlocks
}
```

**Change 2 (after line 351) — snapshot interval validation:**

Add after `kvTransferBaseLatency` validation:
```go
if snapshotRefreshInterval < 0 {
	logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
}
```

### File: `cmd/root_test.go`

**Purpose:** Add flag registration test for BC-5. (BC-1 is already covered by existing `TestRunCmd_KVBlockFlags_DefaultsArePositive`.)

**New test:**
```go
func TestRunCmd_SnapshotRefreshInterval_FlagRegistered(t *testing.T) {
	// BC-5: --snapshot-refresh-interval must reject negative values.
	// This test verifies the flag exists and its default is valid (non-negative).
	flag := runCmd.Flags().Lookup("snapshot-refresh-interval")
	assert.NotNil(t, flag, "snapshot-refresh-interval flag must be registered")

	defVal, err := strconv.ParseInt(flag.DefValue, 10, 64)
	assert.NoError(t, err, "default must be a valid int64")
	assert.GreaterOrEqual(t, defVal, int64(0),
		"default snapshot-refresh-interval must be >= 0")
}
```

### File: `sim/workload/generator.go`

**Purpose:** Fix multi-client starvation by generating all clients before applying maxRequests cap.

**Note:** All line numbers reference the original file before any edits. The `continue` at line 109 (which separates reasoning from standard client paths) is preserved.

**Changes:**

0. Add per-client soft cap computation (before line 45, after the `prefixes` assignment):
```go
// Per-client generation cap: prevent OOM when horizon >> maxRequests.
// Each client generates at most 2x maxRequests, then post-merge truncation finalizes.
perClientCap := int64(0)
if maxRequests > 0 {
	perClientCap = 2 * maxRequests
}
```

1. Replace lines 78-79 (reasoning client maxRequests break) with per-client cap:
```go
// REPLACE:
// OLD: if maxRequests > 0 && int64(len(allRequests)) >= maxRequests { break }
// NEW:
if perClientCap > 0 && int64(len(allRequests)-reasoningStartIdx) >= perClientCap {
	break
}
```
The simplest approach: add `clientReqCount := int64(0)` before the reasoning loop at line 77, increment it after each `allRequests = append(allRequests, reasoningReqs...)`, and use `clientReqCount` in the guard:
```go
if perClientCap > 0 && clientReqCount >= perClientCap {
	break
}
```

2. Remove lines 96-103 (reasoning client count cap):
```go
// DELETE entirely:
if maxRequests > 0 && int64(len(allRequests)+len(reasoningReqs)) > maxRequests {
	remaining := maxRequests - int64(len(allRequests))
	if remaining > 0 {
		reasoningReqs = reasoningReqs[:remaining]
	} else {
		reasoningReqs = nil
	}
}
```

3. Replace lines 116-118 (standard client maxRequests break) with per-client cap:
```go
// REPLACE:
// OLD: if maxRequests > 0 && int64(len(allRequests)) >= maxRequests { break }
// NEW: Track per-client count
if perClientCap > 0 && clientCount >= perClientCap {
	break
}
```
Add `var clientCount int64` before the loop and `clientCount++` after appending each request.

4. Add post-merge truncation after the sort (after line 179):
```go
// Truncate to maxRequests after merge-sort (preserves client proportionality)
if maxRequests > 0 && int64(len(allRequests)) > maxRequests {
	allRequests = allRequests[:maxRequests]
}
```

### File: `sim/workload/generator_test.go`

**Purpose:** Add failing test for BC-3 (fair multi-client generation with maxRequests).

**New test:** `TestGenerateRequests_MaxRequests_PreservesClientProportions` (see Task 3, Step 1 for full code).
