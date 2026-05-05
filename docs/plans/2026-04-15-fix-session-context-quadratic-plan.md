# fix(session): quadratic context growth in accumulate mode — Implementation Plan

**Goal:** Fix a bug where `accumulate` mode doubles the accumulated context on every round, causing O(N²) token growth in multi-turn sessions.
**Source:** Discovered during health-scorer evaluation work; no GitHub issue.
**Closes:** N/A (standalone bug fix)

## Background

In `accumulate` mode, `OnComplete` builds each follow-up request's input as:

```
inputTokens = append(contextTokens, newSuffix...)
```

Then stores context with the buggy line:

```go
sess.contextTokens = append(sess.contextTokens, req.InputTokens...)
```

`req.InputTokens` already contains all prior `contextTokens` plus the new suffix — appending it in full re-includes the accumulated context, causing ~2× growth per round. A 3-round session with 10-token inputs grows to 55 tokens instead of the correct 40.

## Behavioral Contracts

**BC-1: No-double-count accumulation**
- GIVEN a session in `accumulate` mode with `contextTokens` of length C
- WHEN a round completes with `req.InputTokens` of length C+S (C prior context + S new suffix) and `actualOutputLen` output tokens
- THEN `contextTokens` grows by exactly `S + actualOutputLen` (the new suffix plus actual output), not by `C+S`

**BC-2: Multi-round size invariant**
- GIVEN a session in `accumulate` mode with constant sampler (10 input, 5 output)
- WHEN rounds 0, 1, and 2 each complete without truncation
- THEN round 1 input length = 25 (15 accumulated + 10 new), round 2 input length = 40 (30 accumulated + 10 new)

**BC-3: Non-accumulate mode unaffected**
- GIVEN a session without `ContextGrowth == "accumulate"`
- WHEN a round completes
- THEN the follow-up input length equals only the freshly-sampled token count (no history prepended)

## Tasks

### Task 1: Fix contextTokens suffix-only append (BC-1, BC-2)

**Files:** modify `sim/workload/session.go`, update `sim/workload/session_test.go`

**Test (already in place — `TestSession_ContextAccumulation_MultiStep`):**

Verifies BC-1 + BC-2: a 3-round accumulate session produces round-1 input of 25 tokens and round-2 input of 40 tokens (not 55 under the buggy append-all behavior).

**Impl (already in place):**

Replace the buggy `append(contextTokens, req.InputTokens...)` with suffix-only append:

```go
// Only append the new suffix (req.InputTokens[len(contextTokens):]).
// req.InputTokens was built as append(contextTokens, newSuffix...),
// so re-appending it in full double-counts the prior context.
if len(req.InputTokens) > len(sess.contextTokens) {
    sess.contextTokens = append(sess.contextTokens, req.InputTokens[len(sess.contextTokens):]...)
}
```

**Verify:** `go test ./sim/workload/... -run TestSession_ContextAccumulation -count=1 -v`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(session): suffix-only append prevents quadratic context growth in accumulate mode (BC-1, BC-2)`

## Sanity Checklist

- [x] R1 (silent continue): no new error paths with silent `continue`
- [x] R4 (construction sites): no struct fields added; no construction sites to audit
- [x] R7 (golden tests): `TestSession_ContextAccumulation_MultiStep` expected values (25, 40) independently verified by the BC-2 invariant (not just "because the code produced them")
- [x] R12 (behavioral tests): all THEN clauses check observable output lengths, not internal struct fields
- [x] INV-6 (determinism): fix is deterministic — no map iteration, no float accumulation
- [x] INV-10 (session causality): arrival times unaffected by this fix
- [x] Non-accumulate path: `else` branch returns only `newInputTokens`; unmodified
