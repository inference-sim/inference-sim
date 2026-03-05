# fix(workload): reasoning path prefix prepend + per-round lifecycle filtering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two bugs where reasoning/multi-turn workload generation silently produces incorrect requests — missing shared prefix tokens and leaking rounds past lifecycle window boundaries.

**The problem today:** Reasoning clients configured with `prefix_group` (shared system prompts) never get the prefix prepended to their requests. Additionally, multi-session reasoning clients with lifecycle windows allow individual rounds to extend past the window's `EndUs` boundary — the window check only gates session starts, not per-round arrivals.

**What this PR adds:**
1. Prefix prepend for reasoning paths — both SingleSession and multi-session reasoning requests now include the shared prefix tokens from `prefix_group`, matching the standard request path behavior.
2. Per-round lifecycle filtering for multi-session reasoning — individual rounds that fall outside any active window are suppressed, matching the SingleSession path behavior added in #514.

**Why this matters:** Users with inference-perf specs combining `system_prompt_len > 0` with multi-turn chat get undercounted input tokens and miss prefix cache opportunities. Users with lifecycle windows get requests outside their intended activity periods, corrupting capacity planning results.

**Architecture:** Both fixes are in `sim/workload/generator.go`. Fix 1 passes the resolved `prefix` slice into the reasoning code paths and prepends it to each round's `InputTokens` after `GenerateReasoningRequests` returns. Fix 2 adds per-round `isInActiveWindow` filtering in the multi-session loop (mirroring the existing SingleSession pattern from lines 144-152).

**Source:** GitHub issues #515, #516

**Closes:** Fixes #515, fixes #516

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes two workload generation bugs in the reasoning/multi-turn path of `sim/workload/generator.go`. Both bugs cause the reasoning path to deviate from the standard request path's behavior:

1. The `prefix` variable (resolved at generator.go:109-112) is only applied in the standard path (line 231-233). Both reasoning branches (`SingleSession` at lines 118-153 and multi-session at lines 155-188) skip to `continue` before reaching the prefix prepend code.

2. The multi-session reasoning loop (lines 157-188) checks `isInActiveWindow` for the session start time (line 170) but bulk-appends all rounds from `GenerateReasoningRequests` without per-round filtering (line 182). The SingleSession path correctly filters per-round (lines 144-152), creating an asymmetry.

No deviations from the issue descriptions. Adjacent blocks: `GenerateReasoningRequests` in `reasoning.go` (unchanged — the prefix prepend happens in the caller), `isInActiveWindow` helper (unchanged, reused).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Prefix prepend in SingleSession reasoning path
- GIVEN a reasoning client with `SingleSession: true` and a non-empty `prefix_group`
- WHEN `GenerateRequests` produces requests for this client
- THEN every request's `InputTokens` MUST start with the shared prefix tokens from the prefix group
- MECHANISM: After `GenerateReasoningRequests` returns, prepend `prefix` to each request's `InputTokens` before horizon/lifecycle filtering

BC-2: Prefix prepend in multi-session reasoning path
- GIVEN a reasoning client with `SingleSession: false` and a non-empty `prefix_group`
- WHEN `GenerateRequests` produces requests for this client
- THEN every request's `InputTokens` MUST start with the shared prefix tokens from the prefix group
- MECHANISM: After `GenerateReasoningRequests` returns, prepend `prefix` to each request's `InputTokens` before appending to `allRequests`

BC-3: Per-round lifecycle filtering in multi-session path
- GIVEN a multi-session reasoning client with lifecycle windows
- WHEN a session's later rounds have `ArrivalTime` outside all active windows
- THEN those rounds MUST be excluded from the output
- MECHANISM: After `GenerateReasoningRequests` returns, iterate rounds and skip any where `!isInActiveWindow(req.ArrivalTime, client.Lifecycle)`

BC-4: Per-round horizon filtering in multi-session path
- GIVEN a multi-session reasoning client
- WHEN a session's later rounds have `ArrivalTime >= horizon`
- THEN those rounds MUST be excluded from the output
- MECHANISM: After `GenerateReasoningRequests` returns, break on `req.ArrivalTime >= horizon` (rounds are chronological)

**Backward Compatibility Contracts:**

BC-5: No-prefix reasoning clients unchanged
- GIVEN a reasoning client with no `prefix_group`
- WHEN `GenerateRequests` produces requests
- THEN the output MUST be byte-identical to the current behavior (prefix slice is nil, no prepend)

BC-6: Multi-session without lifecycle unchanged
- GIVEN a multi-session reasoning client with no lifecycle windows
- WHEN `GenerateRequests` produces requests
- THEN the output MUST be byte-identical to the current behavior (no filtering applied)

**Negative Contracts:**

BC-7: No modification of GenerateReasoningRequests
- The `reasoning.go` `GenerateReasoningRequests` function MUST NOT be modified. Prefix prepend and lifecycle filtering happen in the caller (`generator.go`).

BC-8: Prefix + context accumulation token layout
- GIVEN a reasoning client with `prefix_group`, `SingleSession: true`, and `ContextGrowth: "accumulate"`
- WHEN `GenerateRequests` produces a 2-round session
- THEN round 0's `InputTokens` MUST be `prefix + newInput_r0`, and round 1's `InputTokens` MUST be `prefix + [newInput_r0 + output_r0] + newInput_r1`
- MECHANISM: The prefix is prepended by the caller AFTER `GenerateReasoningRequests` builds context from raw `newInputTokens`. The prefix is NOT part of the accumulated context — it is re-prepended fresh each round.

### C) Component Interaction

```
generator.go:GenerateRequests
  ├── reasoning.go:GenerateReasoningRequests  [unchanged — produces raw rounds]
  ├── prefix prepend loop                     [NEW — applied to reasoning rounds]
  ├── per-round lifecycle filter              [NEW for multi-session; exists for SingleSession]
  └── client.go:generatePrefixTokens          [unchanged — produces prefix map]
```

No new types, interfaces, or state. The fix is purely control-flow: adding prefix prepend + filtering to the reasoning paths to match the standard path.

Extension friction: 0 files — this is a bug fix in existing code, no new abstractions.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #516 suggests Option 2 (prepend in caller after GenerateReasoningRequests) | Implements Option 2 | Less invasive, keeps reasoning.go unchanged |
| Original multi-session bulk-appends all rounds | New code filters per-round but keeps `clientReqCount` counting all generated rounds | ADDITION: Per-round filtering is new. Cap counting preserved for R19 safety (9/10 review perspectives flagged amplification risk). |

### E) Review Guide

1. **THE TRICKY PART:** Context accumulation + prefix interaction. In `reasoning.go`, `contextPrefix` accumulates `newInputTokens` — the raw sampled tokens, NOT the prefix-prepended `req.InputTokens`. So the prefix is NOT part of the accumulated context. After the caller prepends the prefix, the token layout for each round is: `[prefix | conversation_history | new_user_input]`. This matches the LLM serving model where the system prompt is always at position 0 and hits the KV prefix cache. **Add a defensive comment in the code** near the prefix prepend noting that reasoning.go builds context from raw `newInputTokens`, so the prefix must NOT be passed into `GenerateReasoningRequests` to avoid double-prepend.
2. **WHAT TO SCRUTINIZE:** BC-3/BC-4 — verify multi-session filtering mirrors the SingleSession pattern exactly (lines 144-152). Within one `GenerateReasoningRequests` call, rounds ARE chronological, so `break` on horizon is correct per-session. `continue` for lifecycle (not `break`) because a round in a gap between windows could have later rounds inside a subsequent window.
3. **WHAT'S SAFE TO SKIM:** BC-5/BC-6 backward compat — trivially satisfied by nil-checking prefix and lifecycle.
4. **KNOWN DEBT:** (a) The existing `TestGenerateRequests_ReasoningClient_RespectsLifecycleWindows` uses `ThinkTimeUs: 10000` which is too short to cross a 1s window boundary. (b) Pre-existing `MultiTurnSpec` fields (`MaxRounds`, `ThinkTimeUs`, `ContextGrowth`) have no validation — file as a separate hardening issue. (c) The SingleSession path has no `perClientCap` guard — it is bounded by `MaxRounds` (a user-input constant), not by the `clientReqCount` mechanism that protects the multi-session path. This is pre-existing and acceptable because SingleSession generates exactly one session.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/workload/generator.go` — add prefix prepend + per-round filtering to both reasoning paths
- `sim/workload/generator_test.go` — add tests for prefix + lifecycle in reasoning paths

**Key decisions:**
- Prefix prepend happens in the caller (generator.go), not in reasoning.go — matching issue #516's Option 2
- Multi-session per-round filtering mirrors the SingleSession pattern exactly
- No changes to reasoning.go or any other file

**Confirmation:** No dead code. All new code exercised by new tests.

### G) Task Breakdown

---

### Task 1: Add prefix prepend to both reasoning paths in generator.go

**Contracts Implemented:** BC-1, BC-2, BC-7, BC-8

**Files:**
- Modify: `sim/workload/generator.go:118-188` (both reasoning paths)
- Test: `sim/workload/generator_test.go`

**Step 1: Write failing tests for prefix prepend in reasoning paths**

Context: We need to verify that reasoning clients with a prefix_group get the shared prefix prepended. Currently they don't — the prefix application only happens in the standard path (line 231-233). We also need to verify the interaction with context accumulation.

```go
func TestGenerateRequests_ReasoningClient_PrependsPrefixTokens(t *testing.T) {
	// BC-1/BC-2: Reasoning paths must prepend shared prefix tokens, just like
	// the standard request path. Both SingleSession and multi-session must work.
	for _, singleSession := range []bool{true, false} {
		name := "multi-session"
		if singleSession {
			name = "single-session"
		}
		t.Run(name, func(t *testing.T) {
			spec := &WorkloadSpec{
				Version: "2", Seed: 42, AggregateRate: 10.0,
				Clients: []ClientSpec{{
					ID: "reasoning-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
					PrefixGroup:  "system-prompt",
					PrefixLength: 20,
					Arrival:      ArrivalSpec{Process: "constant"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
					Reasoning: &ReasoningSpec{
						ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
						MultiTurn: &MultiTurnSpec{
							MaxRounds:     2,
							ThinkTimeUs:   10_000,
							ContextGrowth: "",
							SingleSession: singleSession,
						},
					},
				}},
			}
			horizon := int64(2_000_000)
			requests, err := GenerateRequests(spec, horizon, 0)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(requests) < 2 {
				t.Fatalf("expected at least 2 requests, got %d", len(requests))
			}

			// Each request must have 20 (prefix) + 50 (input) = 70 tokens for round 0.
			// Verify all requests start with the same 20-token prefix.
			prefixLen := 20
			for i, req := range requests {
				if len(req.InputTokens) < prefixLen {
					t.Errorf("request %d: input too short (%d < %d)", i, len(req.InputTokens), prefixLen)
					continue
				}
				if req.RoundIndex == 0 && len(req.InputTokens) != prefixLen+50 {
					t.Errorf("request %d (round 0): input len %d, want %d (prefix %d + input 50)",
						i, len(req.InputTokens), prefixLen+50, prefixLen)
				}
			}
			// All requests share the same prefix tokens (first 20 tokens identical)
			firstPrefix := requests[0].InputTokens[:prefixLen]
			for i := 1; i < len(requests); i++ {
				for j := 0; j < prefixLen; j++ {
					if requests[i].InputTokens[j] != firstPrefix[j] {
						t.Errorf("request %d: prefix token %d = %d, want %d (shared prefix mismatch)",
							i, j, requests[i].InputTokens[j], firstPrefix[j])
						break
					}
				}
			}
		})
	}
}
```

Also add a test for the prefix + context accumulation interaction (BC-8):

```go
func TestGenerateRequests_ReasoningClient_PrefixWithAccumulation(t *testing.T) {
	// BC-8: With prefix + context accumulation, the token layout per round is:
	//   Round 0: [prefix | newInput_r0]
	//   Round 1: [prefix | newInput_r0 + output_r0 | newInput_r1]
	// The prefix is NOT part of the accumulated context — it's re-prepended fresh.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "accum-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			PrefixGroup:  "sys",
			PrefixLength: 10,
			Arrival:      ArrivalSpec{Process: "constant"},
			InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
			OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 15}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     2,
					ThinkTimeUs:   10_000,
					ContextGrowth: "accumulate",
					SingleSession: true,
				},
			},
		}},
	}
	horizon := int64(5_000_000)
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	prefixLen := 10
	inputLen := 20
	outputLen := 15

	// Round 0: prefix(10) + newInput(20) = 30
	r0 := requests[0]
	if len(r0.InputTokens) != prefixLen+inputLen {
		t.Errorf("round 0: input len %d, want %d", len(r0.InputTokens), prefixLen+inputLen)
	}

	// Round 1: prefix(10) + context(newInput_r0=20 + output_r0=15) + newInput(20) = 65
	r1 := requests[1]
	expectedR1Len := prefixLen + (inputLen + outputLen) + inputLen // 10 + 35 + 20 = 65
	if len(r1.InputTokens) != expectedR1Len {
		t.Errorf("round 1: input len %d, want %d", len(r1.InputTokens), expectedR1Len)
	}

	// The prefix tokens must be the same in both rounds
	for j := 0; j < prefixLen; j++ {
		if r0.InputTokens[j] != r1.InputTokens[j] {
			t.Errorf("prefix token %d: round 0 has %d, round 1 has %d",
				j, r0.InputTokens[j], r1.InputTokens[j])
			break
		}
	}

	// Round 1's context must start with round 0's newInput (after the prefix)
	// Round 0's newInput = r0.InputTokens[prefixLen:]
	r0NewInput := r0.InputTokens[prefixLen:]
	for j := 0; j < inputLen; j++ {
		if r1.InputTokens[prefixLen+j] != r0NewInput[j] {
			t.Errorf("round 1 context token %d: got %d, want %d (round 0's newInput)",
				j, r1.InputTokens[prefixLen+j], r0NewInput[j])
			break
		}
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_ReasoningClient_Prepends|TestGenerateRequests_ReasoningClient_PrefixWith" -v`
Expected: FAIL — reasoning requests don't have prefix tokens currently

**Step 3: Implement prefix prepend in both reasoning paths**

Context: Add prefix prepend after `GenerateReasoningRequests` returns, before filtering. This mirrors the standard path's `append(append([]int{}, prefix...), inputTokens...)` pattern.

In `sim/workload/generator.go`, modify the SingleSession path (after line 143, before the filtering loop):

Replace the SingleSession reasoning block (lines 118-153) with code that prepends prefix to each round's InputTokens before filtering.

Replace the multi-session reasoning block (lines 157-188) with code that prepends prefix to each round's InputTokens before appending.

Specifically, in the **SingleSession** path, after `GenerateReasoningRequests` returns and before the filtering for-loop, add:
```go
// Prepend shared prefix to each round's input (BC-1)
if len(prefix) > 0 {
	for _, req := range reasoningReqs {
		req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
	}
}
```

The multi-session path prefix prepend is handled in Task 2 as part of the complete replacement block (which includes both prefix prepend and per-round filtering).

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestGenerateRequests_ReasoningClient_PrependsPrefixTokens -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/generator.go sim/workload/generator_test.go
git commit -m "fix(workload): prepend shared prefix to reasoning request paths (BC-1, BC-2)

- SingleSession and multi-session reasoning paths now prepend prefix_group
  tokens to each round's InputTokens, matching the standard request path
- Fixes #516: reasoning path does not prepend shared prefix tokens

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add per-round lifecycle + horizon filtering to multi-session path

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/workload/generator.go:157-188` (multi-session reasoning path)
- Test: `sim/workload/generator_test.go`

**Step 1: Write failing test for per-round lifecycle filtering**

Context: The multi-session path currently bulk-appends all rounds from `GenerateReasoningRequests`. We need to verify that rounds crossing lifecycle window boundaries are suppressed.

```go
func TestGenerateRequests_MultiSession_PerRoundLifecycleFiltering(t *testing.T) {
	// BC-3: Multi-session reasoning must filter individual rounds against
	// lifecycle windows, not just session start times. A session starting
	// inside a window can have later rounds that cross the window boundary.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "ms-lc", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     10,
					ThinkTimeUs:   500_000, // 500ms between rounds → 10 rounds spans 5s
					ContextGrowth: "",
					SingleSession: false,
				},
			},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 2_000_000}}, // 2s window
			},
		}},
	}
	horizon := int64(10_000_000) // 10s
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// ALL requests must be within the lifecycle window [0, 2_000_000)
	for i, req := range requests {
		if req.ArrivalTime >= 2_000_000 {
			t.Errorf("request %d: ArrivalTime=%d >= window end 2000000 (BC-3 violation)",
				i, req.ArrivalTime)
		}
	}
	// With rate=1.0 (constant), first session starts at ~1s.
	// 10 rounds at 500ms spacing spans 5s, but window only 2s.
	// So later rounds of this session must be suppressed.
	// Total rounds should be significantly fewer than 10.
	if len(requests) >= 10 {
		t.Errorf("expected fewer than 10 requests due to lifecycle window truncation, got %d", len(requests))
	}
}
```

Also add a test for BC-4 (horizon filtering in isolation — no lifecycle windows):

```go
func TestGenerateRequests_MultiSession_PerRoundHorizonFiltering(t *testing.T) {
	// BC-4: Multi-session reasoning must filter individual rounds against
	// horizon, not just the session start time. No lifecycle windows — exercises
	// the `break` on horizon independently of lifecycle filtering.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "ms-hz", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     20,
					ThinkTimeUs:   500_000, // 500ms → 20 rounds spans ~10s
					ContextGrowth: "",
					SingleSession: false,
				},
			},
			// No Lifecycle — horizon is the only filter
		}},
	}
	horizon := int64(3_000_000) // 3s — session starts at 1s, rounds at 1.0s, 1.5s, 2.0s, 2.5s, 3.0s...
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// All requests must be before horizon
	for i, req := range requests {
		if req.ArrivalTime >= horizon {
			t.Errorf("request %d: ArrivalTime=%d >= horizon %d (BC-4 violation)",
				i, req.ArrivalTime, horizon)
		}
	}
	// Should have fewer than 20 requests (horizon truncation)
	if len(requests) >= 20 {
		t.Errorf("expected fewer than 20 requests due to horizon truncation, got %d", len(requests))
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_MultiSession_PerRound" -v`
Expected: FAIL — multi-session path currently bulk-appends all rounds

**Step 3: Implement per-round filtering in multi-session path**

Context: Replace the bulk `allRequests = append(allRequests, reasoningReqs...)` with a per-round filtering loop that mirrors the SingleSession pattern.

In `sim/workload/generator.go`, replace the multi-session bulk-append (line 182-183):
```go
allRequests = append(allRequests, reasoningReqs...)
clientReqCount += int64(len(reasoningReqs))
```

With the prefix prepend + per-round filtering block shown below.

**Note on `clientReqCount`:** The original code counted all generated rounds with `clientReqCount += int64(len(reasoningReqs))`. The new per-round filtering loop still needs to count all generated rounds (not just accepted ones) to preserve `perClientCap` as an R19 safety valve. Counting only accepted rounds would weaken the cap when lifecycle windows filter most rounds, causing unbounded generation amplification.

The implementation keeps `clientReqCount += int64(len(reasoningReqs))` on the original line, separate from the per-round append loop:

```go
// Prepend shared prefix to each round's input (BC-2, #516)
if len(prefix) > 0 {
	for _, req := range reasoningReqs {
		req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
	}
}
// Count all generated rounds for perClientCap safety (R19)
clientReqCount += int64(len(reasoningReqs))
// Filter individual rounds against horizon and lifecycle windows (BC-3, BC-4, #515)
for _, req := range reasoningReqs {
	if req.ArrivalTime >= horizon {
		break // rounds are in chronological order (BC-4)
	}
	if client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, client.Lifecycle) {
		continue // suppress rounds outside lifecycle windows (BC-3)
	}
	allRequests = append(allRequests, req)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestGenerateRequests_MultiSession_PerRoundLifecycleFiltering -v`
Expected: PASS

**Step 5: Run all workload tests to verify no regressions**

Run: `go test ./sim/workload/... -v`
Expected: All PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 7: Commit with contract reference**

```bash
git add sim/workload/generator.go sim/workload/generator_test.go
git commit -m "fix(workload): add per-round lifecycle + horizon filtering to multi-session reasoning (BC-3, BC-4)

- Multi-session reasoning path now filters individual rounds against
  lifecycle windows and horizon, matching the SingleSession path
- Previously only session start times were checked, allowing later
  rounds to leak past window boundaries
- Fixes #515: reasoning multi-session path lacks per-round lifecycle
  window filtering

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Verify backward compatibility and run full test suite

**Contracts Implemented:** BC-5, BC-6

**Files:**
- Test: `sim/workload/generator_test.go`

**Step 1: Write backward-compatibility tests**

Context: Verify that reasoning clients without prefix_group or lifecycle windows produce identical output to the pre-fix behavior.

```go
func TestGenerateRequests_ReasoningClient_NoPrefixGroup_Unchanged(t *testing.T) {
	// BC-5: Reasoning clients without prefix_group must produce identical output.
	spec := &WorkloadSpec{
		Version: "2", Seed: 99, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "no-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     3,
					ThinkTimeUs:   10_000,
					ContextGrowth: "",
					SingleSession: false,
				},
			},
			// No PrefixGroup, no Lifecycle
		}},
	}
	horizon := int64(2_000_000)
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// With no prefix, input tokens should be exactly 50 (constant sampler) for round 0.
	// This verifies no prefix is spuriously prepended.
	for _, req := range requests {
		if req.RoundIndex == 0 && len(req.InputTokens) != 50 {
			t.Errorf("round 0 request: input len %d, want 50 (no prefix should be added)", len(req.InputTokens))
		}
	}
}
```

**Step 2: Run backward-compatibility test**

Run: `go test ./sim/workload/... -run TestGenerateRequests_ReasoningClient_NoPrefixGroup_Unchanged -v`
Expected: PASS (no prefix applied when prefix_group is empty)

**Step 3: Run full test suite**

Run: `go test ./... -count=1`
Expected: All PASS

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit backward-compat tests**

```bash
git add sim/workload/generator_test.go
git commit -m "test(workload): add backward-compatibility tests for reasoning prefix/lifecycle (BC-5, BC-6)

- Verify no-prefix reasoning clients produce unchanged output
- Confirms no regression from prefix prepend or lifecycle filtering changes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestGenerateRequests_ReasoningClient_PrependsPrefixTokens/single-session |
| BC-2 | Task 1 | Unit | TestGenerateRequests_ReasoningClient_PrependsPrefixTokens/multi-session |
| BC-3 | Task 2 | Unit | TestGenerateRequests_MultiSession_PerRoundLifecycleFiltering |
| BC-4 | Task 2 | Unit | TestGenerateRequests_MultiSession_PerRoundHorizonFiltering |
| BC-5 | Task 3 | Unit | TestGenerateRequests_ReasoningClient_NoPrefixGroup_Unchanged |
| BC-6 | Task 3 | Unit | (covered by BC-5 test — multi-session reasoning without lifecycle/prefix produces unchanged round 0 length) |
| BC-7 | — | Structural | Verified by inspection — reasoning.go is not modified |
| BC-8 | Task 1 | Unit | TestGenerateRequests_ReasoningClient_PrefixWithAccumulation |

No golden dataset changes — workload generation tests are standalone and don't use `testdata/goldendataset.json`.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Prefix prepend changes RNG sequence for existing reasoning tests | Low | Medium | Prefix prepend doesn't consume RNG — it copies tokens. Existing tests without prefix_group are unaffected (BC-5). |
| Per-round filtering changes request count for existing multi-session tests | Low | Low | Existing test uses short ThinkTimeUs (10ms) within a 1s window — all rounds fit. BC-6 ensures no change. |
| Context accumulation interacts with prefix prepend | Low | Medium | The prefix is prepended to each round's InputTokens AFTER `GenerateReasoningRequests` builds context. The context in `reasoning.go` accumulates `newInputTokens` (raw, without prefix). So round 0 final = `prefix + newInput_r0`. Round 1 final = `prefix + [newInput_r0 + output_r0] + newInput_r1`. The prefix is NOT in the accumulated context — it's re-prepended fresh each round by the caller. This matches the LLM serving semantic: the system prompt (prefix) appears at position 0, the conversation history follows, and new user input is at the end. The KV cache sees the same prefix tokens every round and cache-hits them. |

(See Section E above for review guide.)

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure bug fix in existing control flow
- [x] No feature creep — only fixes #515 and #516
- [x] No unexercised flags or interfaces
- [x] No partial implementations — both bugs fully fixed
- [x] No breaking changes — BC-5, BC-6 ensure backward compat
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: N/A (standalone tests)
- [x] CLAUDE.md: No update needed (no new files, flags, or packages)
- [x] Documentation DRY: No canonical sources modified
- [x] Deviation log reviewed — 2 deviations documented
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3)
- [x] All contracts mapped to tasks
- [x] Golden dataset: not affected
- [x] Construction site audit: no new struct fields
- [x] R1: No silent continue/return — filtered rounds are intentionally suppressed per lifecycle spec
- [x] R4: No new struct fields
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: No golden tests added
- [x] R19: No unbounded loops added

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/generator.go`

**Purpose:** Fix reasoning paths to prepend prefix tokens and filter per-round lifecycle windows.

**Changes (SingleSession path, ~lines 118-153):**

After `GenerateReasoningRequests` returns (line 143) and before the filtering loop (line 144), insert:
```go
// Prepend shared prefix to each round's input (BC-1, #516)
if len(prefix) > 0 {
	for _, req := range reasoningReqs {
		req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
	}
}
```

**Changes (multi-session path, ~lines 157-188):**

Replace the bulk append (line 182-183):
```go
allRequests = append(allRequests, reasoningReqs...)
clientReqCount += int64(len(reasoningReqs))
```

With prefix prepend + per-round filtering (keeping `clientReqCount` on all generated rounds for R19 safety):
```go
// Prepend shared prefix to each round's input (BC-2, #516)
if len(prefix) > 0 {
	for _, req := range reasoningReqs {
		req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
	}
}
// Count all generated rounds for perClientCap safety (R19)
clientReqCount += int64(len(reasoningReqs))
// Filter individual rounds against horizon and lifecycle windows (BC-3, BC-4, #515)
for _, req := range reasoningReqs {
	if req.ArrivalTime >= horizon {
		break // rounds are in chronological order (BC-4)
	}
	if client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, client.Lifecycle) {
		continue // suppress rounds outside lifecycle windows (BC-3)
	}
	allRequests = append(allRequests, req)
}
```

**Key Implementation Notes:**
- RNG: No change — prefix prepend copies tokens, doesn't consume RNG
- Metrics: No impact — workload generation is pre-simulation
- State mutation: `req.InputTokens` field is overwritten on each `*sim.Request` (pointer semantics — `reasoningReqs` is `[]*sim.Request`). The underlying token slice is new (no aliasing with `reasoning.go`'s `contextPrefix`).
- Error handling: No new error paths
- Defensive comment: Add near prefix prepend: `// NOTE: reasoning.go builds contextPrefix from raw newInputTokens, NOT from req.InputTokens. The prefix must be prepended here in the caller, not passed into GenerateReasoningRequests, to avoid double-prepend with context accumulation.`
