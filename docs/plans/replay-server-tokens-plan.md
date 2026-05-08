# Plan: Use server_input_tokens for replay token generation

**Goal:** Eliminate the systematic TTFT under-prediction that occurs when replaying chat-API traces. `blis observe --api-format chat` records both a client-side `input_tokens` count and the server-reported `server_input_tokens` (which includes hidden chat-template formatting tokens). Replay currently uses the smaller client-side count, causing the simulator to model a faster prefill than the server actually performed.

**Source:** [inference-sim/inference-sim#1269](https://github.com/inference-sim/inference-sim/issues/1269)

**Closes:** #1269

---

## Behavioral Contracts

**BC-1 (ServerInputTokens priority — no prefix group):** GIVEN a `TraceRecord` with `ServerInputTokens > 0` and `PrefixGroup == ""`, WHEN `LoadTraceV2Requests` constructs the `sim.Request`, THEN `len(req.InputTokens) == rec.ServerInputTokens`.

**BC-2 (Fallback to InputTokens):** GIVEN a `TraceRecord` with `ServerInputTokens == 0` OR `PrefixGroup != ""`, WHEN `LoadTraceV2Requests` constructs the `sim.Request`, THEN `len(req.InputTokens)` uses `rec.InputTokens` (plus prefix length if applicable). Zero `ServerInputTokens` means "not recorded" (generated traces). Non-empty `PrefixGroup` means `ServerInputTokens` already contains the prefix length — using it as a suffix count would double-count the prefix, so we fall back to `InputTokens`.

**BC-3 (Session round-0 priority — no prefix group):** GIVEN a session trace where round-0 has `ServerInputTokens > 0` and `PrefixGroup == ""`, WHEN `LoadTraceV2SessionBlueprints` builds the initial request, THEN `len(req.InputTokens) == round0.ServerInputTokens`.

**BC-4 (Session sampler priority — no prefix group):** GIVEN a session trace where round N>0 has `ServerInputTokens > 0` and `PrefixGroup == ""`, WHEN the blueprint's `InputSampler` is queried for that round, THEN it returns `ServerInputTokens` for that round (not `InputTokens`). Rounds with `PrefixGroup != ""` use `InputTokens` unchanged.

**BC-5 (Non-session path parity, R23):** GIVEN a non-session record in a session-mode trace with `ServerInputTokens > 0` and `PrefixGroup == ""`, WHEN `LoadTraceV2SessionBlueprints` processes it, THEN `len(req.InputTokens) == rec.ServerInputTokens`.

**BC-6 (sim.Request has no ServerInputTokens field):** `sim.Request` does not gain a `ServerInputTokens` field. The server-reported count is used only to size the synthetic token ID slice; it is not stored as a separate field on the request struct. This is unchanged from the original design — `server_input_tokens` is a trace-layer artifact, not a simulation primitive.

---

## Tasks

### Task 1: Add failing tests for BC-1, BC-2, BC-3, BC-4, BC-5

**File:** `sim/workload/replay_test.go`

Write tests before implementing the fix to confirm they fail.

**Test A — BC-1: `LoadTraceV2Requests` uses `ServerInputTokens` when present**

```go
func TestLoadTraceV2Requests_ServerInputTokens_UsedWhenPresent(t *testing.T) {
    // GIVEN a trace record where ServerInputTokens > InputTokens (chat template overhead)
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 0, InputTokens: 512, ServerInputTokens: 530,
                OutputTokens: 64, ArrivalTimeUs: 0, Status: "ok"},
        },
    }
    requests, err := LoadTraceV2Requests(trace, 42)
    if err != nil {
        t.Fatal(err)
    }
    // BC-1: len(InputTokens) reflects server-reported count, not client-side count
    if len(requests[0].InputTokens) != 530 {
        t.Errorf("input token count = %d, want 530 (server-reported)", len(requests[0].InputTokens))
    }
}
```

**Test B — BC-2: `LoadTraceV2Requests` falls back to `InputTokens` when `ServerInputTokens == 0`**

```go
func TestLoadTraceV2Requests_ServerInputTokens_Zero_FallsBackToInputTokens(t *testing.T) {
    // GIVEN a trace record with ServerInputTokens == 0 (generated trace, not observed)
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 0, InputTokens: 256, ServerInputTokens: 0,
                OutputTokens: 32, ArrivalTimeUs: 0, Status: "ok"},
        },
    }
    requests, err := LoadTraceV2Requests(trace, 42)
    if err != nil {
        t.Fatal(err)
    }
    // BC-2: fallback to InputTokens when ServerInputTokens not recorded
    if len(requests[0].InputTokens) != 256 {
        t.Errorf("input token count = %d, want 256 (fallback)", len(requests[0].InputTokens))
    }
}
```

**Test C — BC-3: session round-0 uses `ServerInputTokens`**

```go
func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Round0(t *testing.T) {
    // GIVEN a 2-round session where round-0 has server overhead tokens
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0,
                InputTokens: 512, ServerInputTokens: 530,
                OutputTokens: 64, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 1,
                InputTokens: 256, ServerInputTokens: 274,
                OutputTokens: 32, ArrivalTimeUs: 5000},
        },
    }
    requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
    if err != nil {
        t.Fatal(err)
    }
    if len(requests) != 1 {
        t.Fatalf("expected 1 round-0 request, got %d", len(requests))
    }
    // BC-3: round-0 token count uses ServerInputTokens
    if len(requests[0].InputTokens) != 530 {
        t.Errorf("round-0 input token count = %d, want 530 (server-reported)", len(requests[0].InputTokens))
    }
}
```

**Test D — BC-4: blueprint sampler uses `ServerInputTokens` for subsequent rounds**

```go
func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Sampler(t *testing.T) {
    // GIVEN a 3-round session with ServerInputTokens on rounds 1 and 2
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0,
                InputTokens: 512, ServerInputTokens: 530, OutputTokens: 64, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 1,
                InputTokens: 256, ServerInputTokens: 274, OutputTokens: 32, ArrivalTimeUs: 5000},
            {RequestID: 3, SessionID: "A", RoundIndex: 2,
                InputTokens: 128, ServerInputTokens: 0, OutputTokens: 16, ArrivalTimeUs: 10000},
        },
    }
    _, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
    if err != nil {
        t.Fatal(err)
    }
    bp := blueprints[0]
    // BC-4: round-1 sampler returns ServerInputTokens (274 > 256)
    got1 := bp.InputSampler.Sample(nil)
    if got1 != 274 {
        t.Errorf("round-1 sampler value = %d, want 274 (server-reported)", got1)
    }
    // BC-2: round-2 sampler falls back to InputTokens (ServerInputTokens == 0)
    got2 := bp.InputSampler.Sample(nil)
    if got2 != 128 {
        t.Errorf("round-2 sampler value = %d, want 128 (fallback)", got2)
    }
}
```

**Test E — BC-5: non-session path in `LoadTraceV2SessionBlueprints`**

```go
func TestLoadTraceV2SessionBlueprints_ServerInputTokens_NonSessionRecord(t *testing.T) {
    // GIVEN a non-session record with ServerInputTokens > InputTokens
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "", InputTokens: 512, ServerInputTokens: 530,
                OutputTokens: 64, ArrivalTimeUs: 0},
        },
    }
    requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
    if err != nil {
        t.Fatal(err)
    }
    if len(requests) != 1 {
        t.Fatalf("expected 1 request, got %d", len(requests))
    }
    // BC-5: non-session path uses ServerInputTokens
    if len(requests[0].InputTokens) != 530 {
        t.Errorf("non-session input token count = %d, want 530", len(requests[0].InputTokens))
    }
}
```

**Test F — BC-2 (PrefixGroup fallback): `LoadTraceV2Requests` ignores `ServerInputTokens` when `PrefixGroup` is set**

```go
func TestLoadTraceV2Requests_ServerInputTokens_PrefixGroup_FallsBackToInputTokens(t *testing.T) {
    // GIVEN a prefix-group record with ServerInputTokens > InputTokens
    // ServerInputTokens includes the prefix length — applying it as suffix count would double-count.
    // WHEN LoadTraceV2Requests constructs the request
    // THEN the suffix uses InputTokens, not ServerInputTokens (prefix prepended separately)
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 0, InputTokens: 100, PrefixGroup: "shared", PrefixLength: 128,
                ServerInputTokens: 246, // = PrefixLength(128) + InputTokens(100) + overhead(18)
                OutputTokens: 32, ArrivalTimeUs: 0, Status: "ok"},
        },
    }
    requests, err := LoadTraceV2Requests(trace, 42)
    if err != nil {
        t.Fatal(err)
    }
    // BC-2: total = PrefixLength(128) + InputTokens(100) = 228, not 128+246=374
    if len(requests[0].InputTokens) != 228 {
        t.Errorf("input token count = %d, want 228 (prefix 128 + suffix 100, not ServerInputTokens 246)",
            len(requests[0].InputTokens))
    }
}
```

Run to confirm they fail:
```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr-replay-server-tokens
go test ./sim/workload/... -run "TestLoadTraceV2Requests_ServerInputTokens|TestLoadTraceV2SessionBlueprints_ServerInputTokens" -v 2>&1 | head -60
```
Expected: FAIL (BC-1, BC-3, BC-4, BC-5 fail; BC-2 tests may coincidentally pass; Test F may fail if guard is missing).

---

### Task 2: Implement fix in `LoadTraceV2Requests`

**File:** `sim/workload/replay.go`

Replace line 35:
```go
inputTokens := sim.GenerateRandomTokenIDs(rng, rec.InputTokens)
```
With:
```go
nInput := rec.InputTokens
if rec.ServerInputTokens > 0 && rec.PrefixGroup == "" {
    nInput = rec.ServerInputTokens
}
inputTokens := sim.GenerateRandomTokenIDs(rng, nInput)
```

The `PrefixGroup == ""` guard is required because `ServerInputTokens` is the server's full prompt token count, which for prefix-group records already includes the `PrefixLength` prefix tokens. Using it as the *suffix* count and then prepending prefix tokens would double-count. For non-prefix records (the `--api-format chat` case this fix targets), no prefix is prepended so the full `ServerInputTokens` is the correct total length.

Remove the now-stale comment on line 70:
```go
// ServerInputTokens: not propagated to sim.Request (calibration-only field, BC-7)
```
Replace with nothing (the field doesn't exist on sim.Request; that's structurally enforced — no comment needed).

Run the BC-1 and BC-2 tests to confirm they pass:
```bash
go test ./sim/workload/... -run "TestLoadTraceV2Requests_ServerInputTokens" -v
```
Expected: PASS.

Lint:
```bash
golangci-lint run ./sim/workload/...
```
Expected: no issues.

---

### Task 3: Implement fix in `LoadTraceV2SessionBlueprints`

**File:** `sim/workload/replay.go`

**3a. Fix `inputSeq` build loop** (around line 159):
```go
for i, rec := range rounds {
    nInput := rec.InputTokens
    if rec.ServerInputTokens > 0 && rec.PrefixGroup == "" {
        nInput = rec.ServerInputTokens
    }
    inputSeq[i] = nInput
    outputSeq[i] = rec.OutputTokens
}
```

**3b. Fix round-0 request construction** (around line 188):
```go
nInput := r0.InputTokens
if r0.ServerInputTokens > 0 && r0.PrefixGroup == "" {
    nInput = r0.ServerInputTokens
}
inputTokens := sim.GenerateRandomTokenIDs(sessionRNG, nInput)
```

**3c. Fix non-session records loop** (around line 246):
```go
nInput := rec.InputTokens
if rec.ServerInputTokens > 0 && rec.PrefixGroup == "" {
    nInput = rec.ServerInputTokens
}
inputTokens := sim.GenerateRandomTokenIDs(rng, nInput)
```

Run all new tests:
```bash
go test ./sim/workload/... -run "TestLoadTraceV2SessionBlueprints_ServerInputTokens" -v
```
Expected: PASS.

Run full package:
```bash
go test ./sim/workload/... -v 2>&1 | tail -20
```
Expected: all pass.

Lint:
```bash
golangci-lint run ./sim/workload/...
```

---

### Task 4: Update stale comments in `replay_test.go`

**File:** `sim/workload/replay_test.go`

In `TestLoadTraceV2Requests_ModelAndDeadline`, the record at index 0 has `ServerInputTokens: 300, InputTokens: 100`. After Task 2's fix, `LoadTraceV2Requests` will generate 300 tokens (not 100) for that record. The existing comment is misleading:

Old:
```go
ServerInputTokens: 300, // must NOT appear on sim.Request
```
New:
```go
ServerInputTokens: 300, // used as token count for InputTokens generation (> InputTokens: 100)
```

Also update the function docstring from "BC-3, BC-4, BC-5, BC-6, BC-7" to "BC-3, BC-4, BC-5, BC-6" (BC-7 is now replaced by BC-1/BC-2 from this plan).

Add an assertion to `TestLoadTraceV2Requests_ModelAndDeadline` to make the behavior explicit (so it becomes a canary if someone reverts the fix):
```go
// BC-1: ServerInputTokens (300) used as token count, not InputTokens (100)
if len(requests[0].InputTokens) != 300 {
    t.Errorf("request 0 input token count = %d, want 300 (server-reported)", len(requests[0].InputTokens))
}
```

Run to confirm:
```bash
go test ./sim/workload/... -run TestLoadTraceV2Requests_ModelAndDeadline -v
```
Expected: PASS.

Final full test run and build:
```bash
go test ./... 2>&1 | tail -20
go build ./...
```

Commit:
```bash
git add sim/workload/replay.go sim/workload/replay_test.go docs/plans/replay-server-tokens-plan.md
git commit -m "fix(replay): use server_input_tokens for token generation to eliminate TTFT bias

- BC-1: when TraceRecord.ServerInputTokens > 0, use it as token count for
  sim.Request.InputTokens generation instead of InputTokens (chat template overhead)
- BC-2: ServerInputTokens == 0 falls back to InputTokens (generated traces)
- Applies to LoadTraceV2Requests, LoadTraceV2SessionBlueprints (round-0,
  inputSeq sampler, and non-session path) for full code-path parity (R23)
- Removes stale 'calibration-only' comment; sim.Request still has no
  ServerInputTokens field (BC-6: structurally enforced, not comment-enforced)

Closes #1269

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Sanity Checklist

- [ ] BC-2 (zero fallback) is guarded by `> 0`, not `!= 0` — negative values treated as "not recorded" (consistent with R3 validation in `ParseTraceRecord` which already rejects negatives)
- [ ] BC-2 (PrefixGroup guard): condition is `rec.ServerInputTokens > 0 && rec.PrefixGroup == ""` — the `PrefixGroup == ""` guard prevents double-counting prefix tokens (`ServerInputTokens` on prefix-group records already includes `PrefixLength`)
- [ ] INV-6 (determinism): `ServerInputTokens` is a deterministic field from the trace CSV — using it doesn't introduce non-determinism
- [ ] R23 (code path parity): all three call sites in `LoadTraceV2SessionBlueprints` (inputSeq, round-0, non-session) get the same `PrefixGroup == ""` guard as `LoadTraceV2Requests`
- [ ] R4 (construction sites): no new struct fields added; `nInput` is a local variable
- [ ] `sim.Request` still has no `ServerInputTokens` field (BC-6); verified by grep: `grep -n "ServerInputTokens" sim/request.go` returns nothing

---

## Deviation Log

**CLARIFICATION-1:** Issue says "use it instead of `rec.InputTokens` as the token count for the sim.Request." This could mean either (a) store it as a new sim.Request field, or (b) use it only as the count for GenerateRandomTokenIDs. Interpretation: (b). `sim.Request` has no `ServerInputTokens` field — this was explicitly designed in the original observe-replay plan (BC-7 there). We preserve that design: server-reported count is a trace artifact used only to determine synthetic token ID slice length. BC-6 above documents this explicitly.

**CLARIFICATION-3:** Issue does not mention prefix-group traces. For records with `PrefixGroup != ""`, `ServerInputTokens` reflects the server's full prompt token count including the shared prefix tokens. The replay code prepends prefix tokens separately on top of the suffix generated by `GenerateRandomTokenIDs`. Applying `ServerInputTokens` as the suffix count would double-count the prefix. Decision: guard with `PrefixGroup == ""` and fall back to `InputTokens` for prefix-group records. This is conservative and safe; the issue's stated use case (`--api-format chat` traces) does not combine prefix groups with chat template overhead.

**CLARIFICATION-2:** The existing `TestLoadTraceV2Requests_ModelAndDeadline` test uses `ServerInputTokens: 300, InputTokens: 100`. After the fix, `len(req.InputTokens)` becomes 300 instead of 100. The test doesn't currently assert on this length, so it wouldn't catch a regression if the fix were reverted. Added an explicit assertion (Task 4) to make the test a genuine canary.
