# Fix: blis observe per-request min_tokens enforcement — Implementation Plan

**Goal:** Make `blis observe` set `min_tokens = max_tokens` per request so the observed output-token distribution matches the workload spec, just as `blis run` already does.

**The problem today:** `requestToPending` accepts a global `minTokens int` constant and applies it identically to every request. Because the real server's EOS fires before `max_tokens` unless `min_tokens` prevents it, the actual output-token distribution is a left-truncated, compressed version of the workload spec. This silently biases every downstream calibration metric.

**What this PR adds:**
1. `requestToPending` derives `MinTokens` from `req.MaxOutputLen` per-request (not a global flag), ensuring exact-length enforcement for every request in normal mode.
2. `--unconstrained-output` correctly bypasses min_tokens (server decides length freely, same as before).
3. Removes the now-redundant `--min-tokens` CLI flag, `clampRequestsToMinTokens`, and `validateMinTokensMean` — the infrastructure added to work around the original bug.
4. Adds `.gitignore` entries for external-repo sibling directories that appear in `git status`.

**Why this matters:** The observe→calibrate loop is only as good as the fidelity of the observed data. Distribution compression in the trace silently biases every TTFT, ITL, and E2E calibration metric.

**Architecture:** All changes are in `cmd/` (never `sim/`). `requestToPending` is the single construction site for `PendingRequest`. The fix is a two-line semantic change in that function plus removal of the global-flag infrastructure that was the workaround for the original bug.

**Source:** inference-sim/inference-sim#1277

**Closes:** Fixes #1277

**Behavioral Contracts:** See Section B below.

---

## Phase 0: Component Context

1. **Building block modified:** `requestToPending` in `cmd/observe_cmd.go` — converts `*sim.Request` to `*PendingRequest` for HTTP dispatch.
2. **Adjacent blocks:** `runObserveOrchestrator` (calls `requestToPending`), `RealClient.Send` (sends `PendingRequest` over HTTP), `warnOnFinishReason` (uses `PendingRequest.MinTokens` for diagnostics). `warnOnFinishReason` logic is unaffected: it receives per-request values and continues to work correctly.
3. **Invariants touched:** None (observe is the CLI/HTTP layer, not the DES core).
4. **Construction site audit:** `PendingRequest{}` is constructed in exactly one place: `requestToPending` in `cmd/observe_cmd.go`. No field is added or removed — only `MinTokens` population changes. R4 is satisfied.

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a silent distribution-fidelity bug in `blis observe`. The root cause: `requestToPending` accepted a global `minTokens int` parameter instead of deriving `MinTokens` from `req.MaxOutputLen`. Because vLLM's EOS fires before `max_tokens` unless `min_tokens` suppresses it, the observed output distribution was compressed relative to the workload spec.

The fix: remove the `minTokens` parameter from `requestToPending` (and from `runObserveOrchestrator` which threaded it through), and compute `MinTokens = req.MaxOutputLen` at the point of `PendingRequest` construction. For unconstrained mode, `MinTokens` remains 0. The global `--min-tokens` flag (and its validation infrastructure: `validateMinTokensMean`, `clampRequestsToMinTokens`) becomes unnecessary and is removed.

The PR also adds `.gitignore` entries for external-repo sibling directories that appear in `git status` as untracked.

No adjacent modules change. `PendingRequest` struct is unchanged. `warnOnFinishReason` works correctly with per-request values.

### B) Behavioral Contracts

**BC-1: Per-request exact-length enforcement (normal mode)**
- GIVEN a `sim.Request` with `MaxOutputLen = N` and `unconstrained = false`
- WHEN `requestToPending` converts it to a `PendingRequest`
- THEN `PendingRequest.MinTokens == N` (equal to `MaxOutputLen`)
- MECHANISM: `requestToPending` sets `MinTokens = req.MaxOutputLen` when `!unconstrained`

**BC-2: Unconstrained mode bypasses min_tokens**
- GIVEN a `sim.Request` with any `MaxOutputLen` and `unconstrained = true`
- WHEN `requestToPending` converts it to a `PendingRequest`
- THEN `PendingRequest.MinTokens == 0` (server decides output length freely)
- MECHANISM: `requestToPending` sets `MinTokens = 0` when `unconstrained`. `Send()` omits `min_tokens` from the HTTP body when `MinTokens == 0` — confirmed by `TestSend_MinTokensInBody`. The field is not sent as `0`; it is absent entirely.

**BC-3: No global min_tokens flag accepted**
- GIVEN a `blis observe` invocation
- WHEN the user tries to pass `--min-tokens N`
- THEN the command rejects the flag as unrecognized (flag does not exist)
- MECHANISM: `--min-tokens` flag registration, `observeMinTokens` variable, `validateMinTokensMean`, and `clampRequestsToMinTokens` are all removed

**BC-4: Zero-length request keeps MinTokens == 0**
- GIVEN a `sim.Request` with `MaxOutputLen = 0` and `unconstrained = false`
- WHEN `requestToPending` converts it
- THEN `PendingRequest.MinTokens == 0` (zero is the natural value; `Send()` applies its `defaultMaxOutputTokens = 2048` fallback for `MaxOutputTokens <= 0`, so the server receives `max_tokens: 2048, min_tokens: (omitted)`)
- MECHANISM: `MinTokens = req.MaxOutputLen` → 0; `Send()` omits `min_tokens` when 0 and uses defaultMaxOutputTokens fallback for max_tokens; vLLM constraint satisfied

### C) Component Interaction

```
cmd/observe_cmd.go
 └── runObserve (entrypoint)
      ├── generates wl.Requests from workload spec
      └── runObserveOrchestrator(ctx, ..., unconstrained, ...)   [minTokens arg REMOVED]
           └── dispatch goroutine
                └── requestToPending(req, idx, noStreaming, unconstrained, ...)  [minTokens arg REMOVED]
                     └── PendingRequest{MinTokens: req.MaxOutputLen or 0}
                          └── RealClient.Send(ctx, pending)
                               └── HTTP POST /v1/completions|/v1/chat/completions
                                    └── vLLM: min_tokens suppresses EOS until MaxOutputLen tokens generated
```

**Removed from data flow:**
- `observeMinTokens int` global flag var (was passed: `runObserve → runObserveOrchestrator → requestToPending`)
- `clampRequestsToMinTokens(wl.Requests, observeMinTokens)` call in `runObserve`
- `validateMinTokensMean` validation call in `runObserve`

**Unchanged:** `warnOnFinishReason` receives per-request `minTokens` from `PendingRequest.MinTokens` via `Send()` — it already operated on per-request values; the fix aligns the population path, not the diagnostic path.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Consider removing [--min-tokens] or keeping it as an explicit opt-out" | Removes `--min-tokens` entirely | SIMPLIFICATION: The per-request behavior is always correct; the flag was a manual workaround. An "opt-out" flag for spec-decoding servers is future scope if needed (#1277 documents the known limitation). |
| "Update the call site at observe_cmd.go:607 to remove the minTokens argument" | Also updates all test call sites for `runObserveOrchestrator` (~10 sites) | CORRECTION: Source only mentioned the production call site; all test call sites must also be updated for the code to compile. |
| No mention of `.gitignore` | Adds `.gitignore` entries for external sibling dirs | ADDITION: User-requested in same PR session; untracked directories are noise in `git status`. |

No clarifications needed — source document is complete and unambiguous on the fix direction.

### E) Review Guide

**Tricky part:** The `minTokens` parameter threads through three function signatures (`runObserveOrchestrator`, `requestToPending`) and has ~10 test call sites. The diff will be noisy from mechanical argument removal; the behavioral change is localized to `requestToPending`.

**Scrutinize:**
1. The unconstrained branch in `requestToPending` — ensure `MinTokens = 0` not `req.MaxOutputLen` for unconstrained mode.
2. BC-4 (zero `MaxOutputLen`) — vLLM requires `min_tokens <= max_tokens`; zero satisfies this since `min_tokens = 0 = max_tokens = 0`.
3. `warnOnFinishReason` behavior after fix — `exactLengthMode` will be true for all normal requests now (since `MinTokens == MaxOutputLen`). This is correct: "length" finish_reason means the server hit the target length, not that output was truncated.

**Safe to skim:** All `runObserveOrchestrator` test call site updates (mechanical argument removal), `.gitignore` additions, CLAUDE.md update.

**Known debt:** Spec-decoding servers reject `min_tokens > 1` (vLLM raises ValueError at engine input_processor.py:168-173). After this fix, every normal-mode request will send `min_tokens = max_tokens`. If the observed server uses spec decoding, requests will fail with HTTP 400. The existing `warnOnFinishReason` detection (finish_reason=stop with outputTokens < minTokens) surfaces this as a warning. A dedicated error message mentioning spec-decoding as a likely cause is deferred to a follow-up issue. The `--unconstrained-output` flag provides an escape hatch for this scenario.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Key Change |
|------|--------|-----------|
| `cmd/observe_cmd.go` | Modify | Remove `minTokens int` from `requestToPending` + `runObserveOrchestrator`; derive `MinTokens` from `req.MaxOutputLen`; remove `--min-tokens` flag/var, `validateMinTokensMean`, `clampRequestsToMinTokens` |
| `cmd/observe_cmd_test.go` | Modify | Add `TestRequestToPending_MinTokensEqualsMaxOutputLen`; update `TestRequestToPending_MinTokensPropagated`; update all `runObserveOrchestrator` call sites; remove `TestValidateMinTokensMean`, `TestClampRequestsToMinTokens` |
| `CLAUDE.md` | Modify | Update observe example: remove `--min-tokens 2048` from "exact output length control" example; update comment to reflect automatic behavior |
| `.gitignore` | Modify | Add 12 external sibling directory entries |

No dead code introduced. No new abstractions. All struct construction sites (just one: `requestToPending`) are covered by Task 1.

### G) Task Breakdown

---

#### Task 1: Implement per-request MinTokens in requestToPending (BC-1, BC-2, BC-3, BC-4)

**Contracts:** BC-1, BC-2, BC-3, BC-4
**Files:** modify `cmd/observe_cmd.go`, modify `cmd/observe_cmd_test.go`

**Step 1 — Write failing test:**

Add to `cmd/observe_cmd_test.go` (after `TestRequestToPending_MinTokensPropagated`):

```go
func TestRequestToPending_MinTokensEqualsMaxOutputLen(t *testing.T) {
	tests := []struct {
		name         string
		maxOutputLen int
		unconstrained bool
		wantMinTokens int
	}{
		{name: "normal mode: MinTokens equals MaxOutputLen", maxOutputLen: 256, unconstrained: false, wantMinTokens: 256},
		{name: "unconstrained mode: MinTokens is zero", maxOutputLen: 256, unconstrained: true, wantMinTokens: 0},
		{name: "zero MaxOutputLen in normal mode: MinTokens is zero", maxOutputLen: 0, unconstrained: false, wantMinTokens: 0},
		{name: "large MaxOutputLen in normal mode", maxOutputLen: 4096, unconstrained: false, wantMinTokens: 4096},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &sim.Request{
				ID:           "r",
				InputTokens:  make([]int, 5),
				MaxOutputLen: tc.maxOutputLen,
			}
			p := requestToPending(req, 0, false, tc.unconstrained, nil, nil, 1.0)
			if p.MinTokens != tc.wantMinTokens {
				t.Errorf("MinTokens = %d, want %d", p.MinTokens, tc.wantMinTokens)
			}
		})
	}
}
```

Also update the existing `TestRequestToPending_MinTokensPropagated` to test the new invariant (MinTokens follows req.MaxOutputLen, not a global parameter):

```go
func TestRequestToPending_MinTokensPropagated(t *testing.T) {
	// BC-1: MinTokens must equal MaxOutputLen per-request (not a global constant).
	req := &sim.Request{
		ID:           "per-req-min-tok",
		InputTokens:  make([]int, 5),
		MaxOutputLen: 128,
	}
	p := requestToPending(req, 0, false, false, nil, nil, 1.0)
	if p.MinTokens != req.MaxOutputLen {
		t.Errorf("MinTokens = %d, want %d (== MaxOutputLen)", p.MinTokens, req.MaxOutputLen)
	}
}
```

**Step 2 — Verify test fails:**
```
go test ./cmd/... -run TestRequestToPending_MinTokensEqualsMaxOutputLen 2>&1
# Expected: compilation error — requestToPending called with wrong number of arguments
```

**Step 3 — Implement: change requestToPending signature and body:**

In `cmd/observe_cmd.go`, change the function signature and implementation:

```go
// requestToPending converts a sim.Request to a PendingRequest for HTTP dispatch.
// prefixes maps prefix-group name to a pre-built prefix string; prefixLengths maps
// prefix-group name to the target token count for the prefix (not word count;
// see buildPrefixStrings). Both may be nil if no prefix groups exist.
// tokensPerWord is the calibrated ratio from calibratePrefixTokenRatio; it scales
// word count so the server tokenizes the prompt to approximately len(InputTokens) tokens.
func requestToPending(req *sim.Request, reqIndex int, noStreaming, unconstrained bool, prefixes map[string]string, prefixLengths map[string]int, tokensPerWord float64) *PendingRequest {
	// Scale token count to word count using calibrated ratio (BC-3, BC-6).
	if tokensPerWord <= 0 {
		tokensPerWord = 1.0
	}
	wordCount := int(math.Round(float64(len(req.InputTokens)) / tokensPerWord))
	if wordCount <= 0 {
		wordCount = 1
	}

	var prompt string
	if req.PrefixGroup != "" && prefixes != nil {
		if prefix, ok := prefixes[req.PrefixGroup]; ok {
			prefixLen := prefixLengths[req.PrefixGroup]
			suffixTokens := len(req.InputTokens) - prefixLen
			if suffixTokens < 1 {
				suffixTokens = 1
			}
			suffixWords := int(math.Round(float64(suffixTokens) / tokensPerWord))
			if suffixWords < 1 {
				suffixWords = 1
			}
			suffixStart := len(req.InputTokens) - suffixTokens
			if suffixStart < 0 {
				suffixStart = 0
			}
			if suffixStart > len(req.InputTokens) {
				suffixStart = len(req.InputTokens)
			}
			prompt = prefix + tokensToPrompt(req.InputTokens[suffixStart:], suffixWords)
		} else {
			prompt = tokensToPrompt(req.InputTokens, wordCount)
		}
	} else {
		prompt = tokensToPrompt(req.InputTokens, wordCount)
	}

	// BC-1: set min_tokens = max_tokens per-request so the server generates exactly
	// MaxOutputLen tokens (matching what blis run produces). BC-2: skip in unconstrained
	// mode — the server decides output length freely.
	minTokens := req.MaxOutputLen
	if unconstrained {
		minTokens = 0
	}

	return &PendingRequest{
		RequestID:       reqIndex,
		InputTokens:     len(req.InputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		Model:           req.Model,
		Streaming:       req.Streaming && !noStreaming,
		ClientID:        req.ClientID,
		TenantID:        req.TenantID,
		SLOClass:        req.SLOClass,
		PrefixGroup:     req.PrefixGroup,
		PrefixLength:    req.PrefixLength,
		Prompt:          prompt,
		Unconstrained:   unconstrained,
		MinTokens:       minTokens,
		DeadlineUs:      req.Deadline,
	}
}
```

**Change runObserveOrchestrator signature** — remove `minTokens int` parameter:

```go
func runObserveOrchestrator(
	ctx context.Context,
	client *RealClient,
	recorder *Recorder,
	sessionMgr *workload.SessionManager,
	requests []*sim.Request,
	noStreaming bool,
	maxConcurrency int,
	warmupCount int,
	prefixes map[string]string,
	prefixLengths map[string]int,
	unconstrained bool,
	recordITL bool,
	tokensPerWord float64,
) {
```

**Update the dispatch closure inside runObserveOrchestrator** (was passing `minTokens` as 5th positional):

```go
pending := requestToPending(req, idx, noStreaming, unconstrained, prefixes, prefixLengths, tokensPerWord)
```

**Update the production call site** at `runObserve` (was passing `observeMinTokens`):

```go
runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, observeNoStreaming, observeMaxConcur, observeWarmup, prefixes, prefixLengths, observeUnconstrainedOutput, observeRecordITL, tokensPerWord)
```

**Remove global flag infrastructure** from `cmd/observe_cmd.go`:

1. Remove `observeMinTokens int` from the `var` block (observe_cmd.go:60).

2. Remove `--min-tokens` flag registration (observe_cmd.go:148):
   ```go
   // DELETE this line:
   observeCmd.Flags().IntVar(&observeMinTokens, "min-tokens", 0, "Set min_tokens in request body (requests server to generate at least N tokens before EOS; 0 = omit field)")
   ```

3. Remove validation of `--min-tokens` (observe_cmd.go:282-284):
   ```go
   // DELETE these lines:
   if observeMinTokens < 0 {
       logrus.Fatalf("--min-tokens must be >= 0, got %d", observeMinTokens)
   }
   ```

4. Remove `validateMinTokensMean` check (observe_cmd.go:288-293):
   ```go
   // DELETE these lines:
   if observeMinTokens > 0 && !observeUnconstrainedOutput &&
       observeWorkloadSpec == "" && observeWorkload == "" &&
       cmd.Flags().Changed("output-tokens") {
       if msg := validateMinTokensMean(observeMinTokens, observeOutputTokens); msg != "" {
           logrus.Fatalf("%s", msg)
       }
   }
   ```

5. Remove `clampRequestsToMinTokens` call (observe_cmd.go:384-391):
   ```go
   // DELETE these lines:
   // Clamp each request's MaxOutputLen to min_tokens so no request reaches the server
   // with max_tokens < min_tokens (which vLLM rejects with HTTP 400).
   if observeMinTokens > 0 && !observeUnconstrainedOutput {
       if n := clampRequestsToMinTokens(wl.Requests, observeMinTokens); n > 0 {
           logrus.Infof("Clamped max_tokens floor to min_tokens=%d on %d/%d requests (distribution left tail truncated)",
               observeMinTokens, n, len(wl.Requests))
       }
   }
   ```

6. Remove `validateMinTokensMean` function definition (~observe_cmd.go:767-777).

7. Remove `clampRequestsToMinTokens` function definition (~observe_cmd.go:779-800).

**Update test call sites for runObserveOrchestrator** — remove the `minTokens` (12th) argument (currently `0` in all test calls). There are ~10 occurrences; each call changes from 14 to 13 arguments. Example:

```go
// Before:
runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false, 0, false, 1.0)
// After:
runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false, false, 1.0)
```

**Remove obsolete tests** from `cmd/observe_cmd_test.go`:
- `TestValidateMinTokensMean` (function removed)
- `TestClampRequestsToMinTokens` (function removed)

**Step 4 — Verify tests pass:**
```
go test ./cmd/... -run "TestRequestToPending" -v 2>&1
# Expected: all TestRequestToPending_* pass
go test ./cmd/... -count=1 2>&1
# Expected: all tests pass
```

**Step 5 — Lint:**
```
golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "fix(observe): set min_tokens per-request to enforce exact output length (BC-1, BC-2, BC-3)

Fixes #1277

- requestToPending: derive MinTokens from req.MaxOutputLen (not global flag)
- unconstrained mode: MinTokens=0 (server decides length freely, BC-2)
- remove --min-tokens flag, clampRequestsToMinTokens, validateMinTokensMean
- update runObserveOrchestrator signature (remove minTokens param)

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Update CLAUDE.md and .gitignore

**Contracts:** (documentation/config — no behavioral contracts)
**Files:** modify `CLAUDE.md`, modify `.gitignore`

**Step 1 — No failing test (documentation/config change).**

**Step 2 — Implement: update CLAUDE.md**

In CLAUDE.md, replace the old observe example with `--min-tokens`:

```
# Observe with exact output length control (min_tokens=output_tokens defers EOS to target length; vLLM/compatible servers only)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --rate 10 --num-requests 100 --output-tokens 2048 --min-tokens 2048 \
  --trace-header trace.yaml --trace-data trace.csv
```

Remove this block entirely (the behavior is now the automatic default — no flag needed).

**Step 3 — Implement: add .gitignore entries**

Append to `.gitignore`:
```
# External-repo sibling directories cloned alongside inference-sim
ServeGen/
aiconfigurator/
gateway-api-inference-extension/
llm-d-inference-payload-processor/
llm-d-inference-scheduler/
llm-d-kv-cache/
llm-d-pd-utils/
llm-d-routing-sidecar/
llm-d/
sarathi-serve/
vidur/
vllm/
```

**Step 4 — Verify build still passes:**
```
go build ./... 2>&1
```

**Step 5 — Lint:**
```
golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
git add CLAUDE.md .gitignore
git commit -m "chore: update CLAUDE.md + add sibling dirs to .gitignore

- Remove --min-tokens example (min_tokens is now set automatically per-request)
- Add .gitignore entries for external-repo sibling directories

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit/Behavioral | `TestRequestToPending_MinTokensEqualsMaxOutputLen` (case: normal) |
| BC-2 | Task 1 | Unit/Behavioral | `TestRequestToPending_MinTokensEqualsMaxOutputLen` (case: unconstrained) |
| BC-3 | Task 1 | Unit/Behavioral | Removal of `TestValidateMinTokensMean` + `TestClampRequestsToMinTokens` (dead code gone) |
| BC-4 | Task 1 | Unit/Behavioral | `TestRequestToPending_MinTokensEqualsMaxOutputLen` (case: zero MaxOutputLen) |
| BC-1 (regression) | Task 1 | Unit/Behavioral | `TestRequestToPending_MinTokensPropagated` (updated) |

No golden dataset changes. No new invariant violations (observe is not in the DES core).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| Compilation error from missing minTokens arg in test call sites | High (known) | Medium | Task 1 explicitly lists all ~10 call sites and the exact replacement | Task 1 |
| warnOnFinishReason logic subtly broken by per-request change | Low | Low | exactLengthMode will now always be true for normal requests; "length" warning suppressed; this is correct behavior and covered by existing TestWarnOnFinishReason tests | Task 1 |
| Spec-decoding server rejects min_tokens > 1 | Low | Low (known limitation) | Documented in issue; existing warnOnFinishReason detection surfaces the failure; no new detection needed | — |
| Zero MaxOutputLen request sent with min_tokens=max_tokens=0 | Low | None | BC-4 covers this; vLLM accepts min_tokens=0 (no constraint) | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — removing, not adding.
- [x] No feature creep — strictly the bug fix + .gitignore housekeeping as requested.
- [x] No unexercised flags or interfaces — `--min-tokens` flag is removed, not left as dead code.
- [x] No partial implementations — all call sites updated in Task 1.
- [x] No breaking changes without explicit contract updates — BC-3 explicitly contracts the flag removal.
- [x] No hidden global state impact — `observeMinTokens` is a file-scoped var in `cmd/`; removing it has no cross-package effect.
- [x] All new code will pass golangci-lint — no new variables, no unused imports.
- [x] Shared test helpers used from existing shared test package.
- [x] CLAUDE.md updated — `--min-tokens` example removed in Task 2.
- [x] No stale references left in CLAUDE.md after Task 2.
- [x] Documentation DRY — no canonical source file (rules.md, invariants.md, principles.md) is modified; no source-of-truth map updates needed.
- [x] Deviation log reviewed — two deviations, both justified.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered — Task 2 depends on Task 1 (CLAUDE.md update references behavior established in Task 1).
- [x] All contracts mapped to specific tasks — see Test Strategy table.
- [x] No golden dataset changes needed.
- [x] Construction site audit completed — `PendingRequest` constructed only in `requestToPending`.
- [x] Not part of a macro plan.

**Antipattern rules:**
- [x] R1: No silent continue/return — no error paths changed.
- [x] R2: No map iteration in ordered output.
- [x] R3: No new numeric parameters — we are removing one.
- [x] R4: PendingRequest has one construction site (requestToPending); covered.
- [x] R5: No resource allocation loops.
- [x] R6: No sim/ code touched.
- [x] R7: No golden tests added.
- [x] R8: No exported mutable maps.
- [x] R9: No YAML fields added.
- [x] R10: No YAML parsing changed.
- [x] R11: No division by runtime-derived denominators.
- [x] R12: No golden dataset changes.
- [x] R13: No new interfaces.
- [x] R14: requestToPending already single-responsibility; not changing that.
- [x] R15: No stale PR references.
- [x] R16: No config parameter changes.
- [x] R17: No routing scorer changes.
- [x] R18: No CLI flag defaults.yaml interaction.
- [x] R19: No retry loops.
- [x] R20: No detectors/analyzers changed.
- [x] R21: No range over slices that shrink.
- [x] R22: No pre-check estimates.
- [x] R23: No parallel code paths with asymmetric transforms.

---

## Appendix: File-Level Implementation Details

### File: `cmd/observe_cmd.go`

**Purpose:** Implements `blis observe` command. Changes: remove global min_tokens infrastructure, update two function signatures.

**Changes by location:**

**1. Remove `observeMinTokens` from var block (~line 60):**
```go
// DELETE:
observeMinTokens           int
```

**2. Remove `--min-tokens` flag (~line 148):**
```go
// DELETE:
observeCmd.Flags().IntVar(&observeMinTokens, "min-tokens", 0, "Set min_tokens in request body (requests server to generate at least N tokens before EOS; 0 = omit field)")
```

**3. Remove --min-tokens validation (~lines 282-284):**
```go
// DELETE:
if observeMinTokens < 0 {
    logrus.Fatalf("--min-tokens must be >= 0, got %d", observeMinTokens)
}
```

**4. Remove validateMinTokensMean check (~lines 288-293):**
```go
// DELETE:
if observeMinTokens > 0 && !observeUnconstrainedOutput &&
    observeWorkloadSpec == "" && observeWorkload == "" &&
    cmd.Flags().Changed("output-tokens") {
    if msg := validateMinTokensMean(observeMinTokens, observeOutputTokens); msg != "" {
        logrus.Fatalf("%s", msg)
    }
}
```

**5. Remove clampRequestsToMinTokens call (~lines 384-391):**
```go
// DELETE:
// Clamp each request's MaxOutputLen to min_tokens so no request reaches the server
// with max_tokens < min_tokens (which vLLM rejects with HTTP 400).
if observeMinTokens > 0 && !observeUnconstrainedOutput {
    if n := clampRequestsToMinTokens(wl.Requests, observeMinTokens); n > 0 {
        logrus.Infof("Clamped max_tokens floor to min_tokens=%d on %d/%d requests (distribution left tail truncated)",
            observeMinTokens, n, len(wl.Requests))
    }
}
```

**6. Update runObserveOrchestrator call (~line 470):**
```go
// Before:
runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, observeNoStreaming, observeMaxConcur, observeWarmup, prefixes, prefixLengths, observeUnconstrainedOutput, observeMinTokens, observeRecordITL, tokensPerWord)
// After:
runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, observeNoStreaming, observeMaxConcur, observeWarmup, prefixes, prefixLengths, observeUnconstrainedOutput, observeRecordITL, tokensPerWord)
```

**7. Update runObserveOrchestrator signature (~line 537):**
```go
// Before:
func runObserveOrchestrator(
    ...
    unconstrained bool,
    minTokens int,
    recordITL bool,
    tokensPerWord float64,
) {
// After:
func runObserveOrchestrator(
    ...
    unconstrained bool,
    recordITL bool,
    tokensPerWord float64,
) {
```

**8. Update requestToPending call inside dispatch closure (~line 607):**
```go
// Before:
pending := requestToPending(req, idx, noStreaming, unconstrained, minTokens, prefixes, prefixLengths, tokensPerWord)
// After:
pending := requestToPending(req, idx, noStreaming, unconstrained, prefixes, prefixLengths, tokensPerWord)
```

**9. Update requestToPending signature and body (~line 824):**
See full implementation code in Task 1 Step 3 above.

**10. Remove validateMinTokensMean function (~lines 767-777):**
```go
// DELETE entire function:
func validateMinTokensMean(minTokens, outputMean int) string { ... }
```

**11. Remove clampRequestsToMinTokens function (~lines 779-800):**
```go
// DELETE entire function:
func clampRequestsToMinTokens(requests []*sim.Request, minTokens int) int { ... }
```

---

### File: `cmd/observe_cmd_test.go`

**Purpose:** Tests for observe command. Changes: add new test, update existing test, remove dead tests, update ~10 call sites.

**1. Update TestRequestToPending_MinTokensPropagated (~line 1081):**
Replace existing test with new version (see Task 1 Step 3 above).

**2. Add TestRequestToPending_MinTokensEqualsMaxOutputLen (after line 1090):**
Full test code in Task 1 Step 1 above.

**3. Update all runObserveOrchestrator call sites (remove 12th argument `0` for minTokens):**

| Line | Before (14 args) | After (13 args) |
|------|-----------------|-----------------|
| ~151 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~218 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~285 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~317 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~345 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~384 | `..., false, 0, true, 1.0)` | `..., false, true, 1.0)` |
| ~425 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~465 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~523 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |
| ~567 | `..., false, 0, false, 1.0)` | `..., false, false, 1.0)` |

**4. Remove TestValidateMinTokensMean (~lines 1518-1566):** Delete entire function.

**5. Remove TestClampRequestsToMinTokens (~lines 1569-1637):** Delete entire function.

---

### File: `CLAUDE.md`

**Change:** Remove the `--min-tokens` example block entirely (the `# Observe with exact output length control...` comment + the 3-line command). The behavior it describes (exact output length) is now the automatic default and requires no flag.

---

### File: `.gitignore`

**Change:** Append at end of file:
```
# External-repo sibling directories cloned alongside inference-sim
ServeGen/
aiconfigurator/
gateway-api-inference-extension/
llm-d-inference-payload-processor/
llm-d-inference-scheduler/
llm-d-kv-cache/
llm-d-pd-utils/
llm-d-routing-sidecar/
llm-d/
sarathi-serve/
vidur/
vllm/
```
