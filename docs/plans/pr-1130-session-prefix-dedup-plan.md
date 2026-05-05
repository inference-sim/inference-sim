# Session Prefix Deduplication Fix (Issue #1130) Implementation Plan

**Goal:** Fix prefix token duplication in `SessionManager.OnComplete` when `context_growth: accumulate` is combined with a non-empty prefix — every follow-up round currently receives the prefix twice.
**Source:** https://github.com/inference-sim/inference-sim/issues/1130
**Closes:** Fixes #1130

---

## Part 1: Design Validation

### A. Executive Summary

When a session uses `context_growth: accumulate` with a non-empty `Prefix`, round-0's `InputTokens` is `[prefix... | conversation...]`. `OnComplete` currently accumulates ALL of `req.InputTokens` (including the prefix) into `contextTokens`, then the follow-up generation prepends the prefix again (line 199), producing `[prefix | prefix | conversation | output | newInput]` on round 1.

This PR fixes `session.go` to strip the prefix before accumulation, matching the invariant already followed by `reasoning.go`: **`contextTokens` must be prefix-free at all times**. The fix is two lines in the accumulate block; no interface changes.

**PR tier:** Small (2 files: `session.go`, `session_test.go`; no new interfaces/types/CLI flags).

### B. Behavioral Contracts

**BC-1: prefix-free accumulation**
- GIVEN a session with `ContextGrowth="accumulate"` and `Prefix=[P×5]`
- WHEN `OnComplete` is called for round 0 with `InputTokens=[P×5|C×10]`, `ProgressIndex=20` (10 input + 5 actual output)
- THEN the round-1 follow-up has `len(InputTokens) == 30` (`[P×5 | C×10 | O×5 | newC×10]`), not 35 — the prefix appears exactly once

**BC-2: no cross-round corruption**
- GIVEN a session as in BC-1, running to round 2
- WHEN `OnComplete` is called for round 1 (with the correct 30-token round-1 request)
- THEN the round-2 follow-up has `len(InputTokens) == 50` (`[P×5 | C×10 | O×5 | newC×10 | O×5 | newC×10]`), and `contextTokens` does not contain the prefix — corruption does not compound

**BC-3: no-prefix sessions unaffected**
- GIVEN a session with `ContextGrowth="accumulate"` and `Prefix=nil`
- WHEN `OnComplete` is called (any round)
- THEN behaviour is identical to pre-fix: existing `TestSession_ContextAccumulation` and `TestSession_ContextAccumulation_MultiStep` still pass unchanged

**BC-4: non-accumulate sessions unaffected**
- GIVEN a session with `ContextGrowth=""` (no accumulation) and any `Prefix`
- WHEN `OnComplete` is called
- THEN the follow-up InputTokens equals `[prefix... | freshInput...]` — the prefix-prepend block (lines 199-201) is the only prefix source, as before

### C. Component Interaction

Single module change. `OnComplete` in `SessionManager` (`sim/workload/session.go`) is the only touch point. No interface signatures change. No callers change.

The invariant adopted matches `reasoning.go:87-90` (accumulates `newInputTokens + outputTokens`, never the prefix) and the explicit warning at `generator.go:178-182` about double-prepend. Both `blis run` and `blis replay` use `SessionManager` and benefit from the fix. `blis observe` does not use `SessionManager.OnComplete`.

### D. Deviation Log

| # | Section | Source says | Plan does | Reason |
|---|---------|-------------|-----------|--------|
| 1 | Proposed solution | Fix "subsumes PR #1046's quadratic-growth fix" | Preserves #1046's suffix-only append; applies it to prefix-stripped input | CORRECTION — "subsumes" is wrong: both fixes are independently required. Removing the suffix-only append would reintroduce quadratic growth. |

### E. Review Guide

Reviewers should focus on:
1. BC-1/BC-2: The slice expression `rawConversation[len(sess.contextTokens):]` — verify the offset is still correct after stripping the prefix.
2. BC-3: Confirm the `len(bp.Prefix) == 0` path is a no-op (slice of full `req.InputTokens`, same as before).
3. Guard: verify `len(req.InputTokens) >= len(bp.Prefix)` is always true by invariant (it is: `generator.go` always prepends the full prefix to round-0 InputTokens, and `session.go` builds follow-ups with prefix prepended).

---

## Part 2: Executable Tasks

### F. Implementation Overview

One code change, two new tests, one updated comment.

Files modified:
- `sim/workload/session.go` — strip prefix in accumulate block
- `sim/workload/session_test.go` — add BC-1 and BC-2 tests

### G–I. TDD Tasks

---

#### Task 1: Failing tests for BC-1 and BC-2 (session_test.go)

**Files:** modify `sim/workload/session_test.go`

**Test (add after `TestSession_ContextAccumulation_MultiStep`):**

```go
// TestSession_ContextAccumulation_WithPrefix verifies BC-1:
// when ContextGrowth="accumulate" and Prefix is non-empty, the prefix
// appears exactly once in the follow-up — not twice (once from accumulated
// contextTokens and once from the prefix-prepend block).
//
// Invariant: len(round1.InputTokens) == len(prefix) + len(input0) + len(output0) + len(newInput1)
//            = 5 + 10 + 5 + 10 = 30, not 35.
func TestSession_ContextAccumulation_WithPrefix(t *testing.T) {
	bp := makeTestBlueprint("sess-prefix", 3, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(7)), 5) // 5 prefix tokens

	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: InputTokens = [prefix(5) | content(10)] = 15 tokens, actual output = 5
	prefixCopy := append([]int{}, bp.Prefix...)
	content0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(8)), 10)
	inputR0 := append(prefixCopy, content0...)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(9)), 5)

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-prefix", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 15 + 5 = 20
		InputTokens:   inputR0,
		OutputTokens:  outputR0,
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}

	r1 := follow[0]
	// prefix(5) + content0(10) + output0(5) + newInput(10) = 30
	wantLen := 5 + 10 + 5 + 10
	if len(r1.InputTokens) != wantLen {
		t.Errorf("BC-1: round 1 input length = %d, want %d (prefix appears once, not twice)",
			len(r1.InputTokens), wantLen)
	}

	// Verify prefix appears at position 0
	for i, tok := range bp.Prefix {
		if i >= len(r1.InputTokens) || r1.InputTokens[i] != tok {
			t.Errorf("BC-1: round 1 token[%d] = %v, want prefix token %v", i, r1.InputTokens[i], tok)
		}
	}
}

// TestSession_ContextAccumulation_WithPrefix_MultiStep verifies BC-2:
// corruption does not compound across rounds — round 2 has the correct
// token count and contextTokens remains prefix-free throughout.
func TestSession_ContextAccumulation_WithPrefix_MultiStep(t *testing.T) {
	bp := makeTestBlueprint("sess-prefix-multi", 4, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(10)), 5) // 5 prefix tokens

	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: [prefix(5) | content(10)] = 15; actual output = 5
	inputR0 := append(append([]int{}, bp.Prefix...), sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(11)), 10)...)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(12)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-prefix-multi", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 20
		InputTokens:   inputR0, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("round 0: expected 1 follow-up, got %d", len(follow1))
	}
	// Round 1 must be 30 tokens: prefix(5)+content0(10)+output0(5)+newInput(10)
	if len(follow1[0].InputTokens) != 30 {
		t.Errorf("BC-2: round 1 input length = %d, want 30", len(follow1[0].InputTokens))
	}

	// Round 1 completion
	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-prefix-multi", RoundIndex: 1,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(follow1[0].InputTokens) + len(follow1[0].OutputTokens)), // 30+5=35
		InputTokens:   follow1[0].InputTokens, OutputTokens: follow1[0].OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 10000)
	if len(follow2) != 1 {
		t.Fatalf("round 1: expected 1 follow-up, got %d", len(follow2))
	}
	// Round 2: prefix(5)+content0(10)+output0(5)+newInput1(10)+output1(5)+newInput2(10) = 45
	// contextTokens after round 1 = content0(10)+output0(5)+newInput1(10)+output1(5) = 30 (prefix-free)
	// round 2 InputTokens = prefix(5) + contextTokens(30) + newInput2(10) = 45
	if len(follow2[0].InputTokens) != 45 {
		t.Errorf("BC-2: round 2 input length = %d, want 45 (no compounding corruption)", len(follow2[0].InputTokens))
	}
}
```

**Verify (expect FAIL):**
```
go test ./sim/workload/... -run "TestSession_ContextAccumulation_WithPrefix" -v
```

**Commit:** (after impl passes — see Task 2)

---

#### Task 2: Implement fix in session.go (BC-1, BC-2, BC-3, BC-4)

**Files:** modify `sim/workload/session.go`

**Impl — replace the accumulate block (currently lines 175–196):**

```go
	var inputTokens []int
	if bp.ContextGrowth == "accumulate" {
		// contextTokens is prefix-free (invariant: matches reasoning.go:87-90 and
		// generator.go:178-182). req.InputTokens = [prefix... | conversation...],
		// so strip the prefix before computing the new suffix to avoid
		// double-counting the prefix block in contextTokens.
		rawConversation := req.InputTokens[len(bp.Prefix):]
		if len(rawConversation) > len(sess.contextTokens) {
			sess.contextTokens = append(sess.contextTokens, rawConversation[len(sess.contextTokens):]...)
		}
		if actualOutputLen > 0 && len(req.OutputTokens) > 0 {
			outTokens := req.OutputTokens
			if actualOutputLen < len(outTokens) {
				outTokens = outTokens[:actualOutputLen]
			}
			sess.contextTokens = append(sess.contextTokens, outTokens...)
		}
		inputTokens = append(append([]int{}, sess.contextTokens...), newInputTokens...)
	} else {
		inputTokens = newInputTokens
	}
```

When `bp.Prefix` is nil/empty, `req.InputTokens[0:]` == `req.InputTokens` — BC-3 holds (no-prefix path unchanged). The suffix-only append from PR #1046 is preserved — BC-2 holds (no quadratic growth).

**Verify (expect PASS):**
```
go test ./sim/workload/... -run "TestSession_ContextAccumulation" -v
```

**Lint:**
```
golangci-lint run ./sim/workload/...
```

**Commit:**
```
fix(session): strip prefix before context accumulation to prevent double-prepend (BC-1, BC-2, closes #1130)
```

---

## Sanity Checklist

- [ ] R1 (no silent continue): no new early returns that drop state
- [ ] R4 (construction sites): no struct literal changes — SessionBlueprint unchanged
- [ ] R14 (single-module methods): change is entirely within `session.go`
- [ ] INV-6 (determinism): fix is deterministic — no new RNG or map iteration
- [ ] INV-10 (session causality): arrival times unchanged — fix only affects token content
- [ ] INV-11 (session completeness): all terminal paths unchanged
- [ ] BC-3 guard: `req.InputTokens[len(nil):]` == `req.InputTokens[0:]` == `req.InputTokens` ✓
- [ ] cross-path parity: `blis run` and `blis replay` both affected (shared `session.go`); `blis observe` not affected ✓
