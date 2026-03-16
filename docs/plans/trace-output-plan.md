# TraceV2 Export for `blis run` — Micro Plan

- **Goal:** Enable `blis run` to export generated workload requests as TraceV2 files for replay-based A/B policy comparisons.
- **The problem today:** `blis run` generates synthetic workload requests but discards the structured workload data after simulation. Users who want to compare policies on identical workloads must manually construct TraceV2 files or use `blis observe` against a live server. There's no way to round-trip generated workloads through `blis replay`.
- **What this PR adds:**
  1. `--trace-output <prefix>` CLI flag that exports simulation requests as TraceV2 files (`<prefix>.yaml` + `<prefix>.csv`)
  2. Three new metadata fields on `sim.Request` (`ClientID`, `PrefixGroup`, `Streaming`) populated during workload generation
  3. A library function `RequestsToTraceRecords` in `sim/workload/` that converts `[]*sim.Request` → `[]TraceRecord`
  4. Session follow-up request collection via `onRequestDone` callback wrapper for complete trace export
  5. Round-trip and invariant tests: field preservation, record count conservation, timing causality
- **Why this matters:** Enables deterministic A/B testing — generate one workload, replay it under different routing/scheduling policies, and compare results with identical input.
- **Architecture:** Adds 3 workload metadata fields to `sim.Request` (following existing TenantID/SLOClass pattern), a conversion function in `sim/workload/tracev2.go`, and CLI integration in `cmd/root.go`. Uses existing `ExportTraceV2` infrastructure.
- **Source:** GitHub issue #656
- **Closes:** `Fixes #656`
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** TraceV2 export pipeline (existing: `observe.go` → `Recorder` → `ExportTraceV2`; new: `run` → `RequestsToTraceRecords` → `ExportTraceV2`)
2. **Adjacent blocks:** Workload generator (`sim/workload/generator.go`), Request lifecycle (`sim/request.go`), CLI (`cmd/root.go`), Replay loader (`sim/workload/replay.go`), Session manager (`sim/workload/session.go`)
3. **Invariants touched:** INV-6 (determinism — new fields are zero-value safe, no stdout change), INV-5 (causality — projected into trace timing assertions)
4. **Construction Site Audit for `sim.Request`:**
   - `sim/workload/generator.go:269` — main request construction in `GenerateRequests` → **UPDATE: populate ClientID, PrefixGroup, Streaming**
   - `sim/workload/replay.go:44` — `LoadTraceV2Requests` construction → **UPDATE: map ClientID, PrefixGroup, Streaming from TraceRecord**
   - `sim/workload/session.go:168` — `SessionManager.OnComplete` follow-up construction → **UPDATE: propagate ClientID from SessionBlueprint** (blueprint already carries `ClientID` at line 30 but doesn't propagate it)
   - `sim/workload/reasoning.go:68` — reasoning request construction → **UPDATE: set ClientID from `clientID` parameter** (function already receives `clientID` at line 19 but doesn't set it)
   - Test files: various `Request{}` literals → zero-value safe, no update needed

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a `--trace-output` flag to `blis run` that exports generated workload requests as TraceV2 format (YAML header + CSV data). The conversion function `RequestsToTraceRecords` lives in `sim/workload/` alongside existing TraceV2 infrastructure. Three workload metadata fields (`ClientID`, `PrefixGroup`, `Streaming`) are added to `sim.Request` following the established pattern of TenantID/SLOClass. The replay loader (`LoadTraceV2Requests`) already reads these fields from TraceV2 — this PR closes the generation→export loop. Session follow-up requests propagate `ClientID` from `SessionBlueprint` for metadata continuity across rounds.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: TraceV2 File Generation
- GIVEN a successful `blis run` with `--trace-output myprefix`
- WHEN simulation completes
- THEN two files are created: `myprefix.yaml` (YAML header with version=2, mode="generated", time_unit="microseconds") and `myprefix.csv` (CSV with one row per request)

BC-2: Request Field Mapping
- GIVEN a completed simulation with N requests
- WHEN trace is exported
- THEN each trace record's input and output token counts match the pre-determined workload specification, multimodal breakdown is preserved, arrival time is preserved, and tenant/SLO/session metadata is preserved
- MECHANISM: InputTokens = len(req.InputTokens), OutputTokens = len(req.OutputTokens) (pre-determined count for ALL requests — the trace records workload input, not simulation results, so replayed workloads are identical regardless of policy differences)

BC-3: Timing Field Population
- GIVEN a completed request with known first-token time and inter-token latencies
- WHEN converted to a trace record
- THEN the arrival timestamp is preserved, the first-chunk timestamp reflects when the first token was generated, and the last-chunk timestamp reflects when the last token was generated (excluding server-side post-decode overhead). For requests without TTFT (timed out during prefill, or incomplete), both chunk timestamps are 0.
- MECHANISM: ArrivalTimeUs = req.ArrivalTime, FirstChunkTimeUs = ArrivalTime + FirstTokenTime (only if TTFTSet), LastChunkTimeUs = ArrivalTime + FirstTokenTime + sum(ITL) (only if TTFTSet). Both chunk timestamps guarded by TTFTSet to avoid producing `LastChunkTimeUs = ArrivalTime` for prefill-timeout requests. This deliberately excludes PostDecodeFixedOverhead — the trace records client-observable token delivery times, not server-side E2E.

BC-4: Round-Trip Fidelity
- GIVEN an exported TraceV2 from `blis run`
- WHEN loaded via `LoadTraceV2()`
- THEN all token counts, workload metadata, and arrival times match the original requests
- MECHANISM: Token sequences are NOT preserved (LoadTraceV2Requests generates new random token IDs). Only counts and metadata round-trip.

BC-5: Workload Metadata Preservation
- GIVEN requests generated from a workload spec with multiple clients
- WHEN converted to trace records
- THEN ClientID and Streaming fields from the workload spec are preserved on each record (including session follow-up requests for ClientID). PrefixGroup is intentionally cleared because InputTokens already includes prefix — setting PrefixGroup would cause double-prepend on replay.

BC-9: Record Count Conservation
- GIVEN N requests passed to the conversion function
- WHEN converted to trace records
- THEN exactly N trace records are produced, one per request
```

**Negative contracts:**

```
BC-6: No stdout Impact
- GIVEN `--trace-output` is set
- WHEN simulation runs
- THEN stdout output is byte-identical to a run without `--trace-output` (INV-6 preserved)

BC-7: No Export Without Flag
- GIVEN `--trace-output` is NOT set
- WHEN simulation completes
- THEN no trace files are written
```

**Error handling:**

```
BC-8: Export Failure Reporting
- GIVEN `--trace-output` points to an unwritable path
- WHEN export is attempted
- THEN a fatal error is logged to stderr with the path and underlying error
```

### C) Component Interaction

```
WorkloadSpec (YAML)
    │
    ▼
GenerateRequests() ──── populates ClientID, PrefixGroup, Streaming on sim.Request
    │
    ▼
preGeneratedRequests []*sim.Request
    │
    ▼
ClusterSimulator.Run() ── mutates Request state, timing fields (pointer semantics)
    │                       SessionManager.OnComplete generates follow-ups with ClientID propagated
    │                       onRequestDone wrapper accumulates follow-up requests into allRequests slice
    ▼
allRequests = preGeneratedRequests + followUpRequests
    │
    ▼
RequestsToTraceRecords() ── sim/workload/tracev2.go (NEW)
    │                        Pure function: reads Request fields, produces TraceRecord values
    │                        Uses array index as RequestID (not string ID parsing)
    │                        OutputTokens = len(req.OutputTokens) (pre-determined, for replay fidelity)
    │                        Timing guarded by TTFTSet (avoids ArrivalTime-only for prefill timeouts)
    ▼
ExportTraceV2() ── sim/workload/tracev2.go (EXISTING)
    │                Writes: .yaml + .csv files
    ▼
LoadTraceV2() ── round-trip verification
```

**State ownership:** `sim.Request` gains 3 fields (owned by workload generation). `RequestsToTraceRecords` is a pure function. `allRequests` is a local slice in the CLI handler that accumulates follow-ups via callback.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Issue says "Request-to-TraceRecord Conversion" in cmd/root.go | Conversion function in `sim/workload/tracev2.go` | CORRECTION — library function is reusable and testable; cmd/ only calls it |
| Issue says "Status fields defaulting to ok" | Derives status from Request.State (completed→"ok", timed_out→"timeout", other→"incomplete") | CORRECTION — preserves actual simulation outcome |
| Issue doesn't mention adding fields to sim.Request | Adds ClientID, PrefixGroup, Streaming to Request | ADDITION — needed for round-trip fidelity with replay.go |
| Issue doesn't mention session follow-up propagation | Propagates ClientID in session.go:OnComplete | ADDITION — prevents metadata loss for multi-turn sessions |
| Issue maps OutputTokens directly | Uses pre-determined count (len(OutputTokens)) for ALL requests | CORRECTION — trace records workload input, not simulation results; pre-determined count ensures replayed workloads are identical across A/B runs regardless of timeouts or length-capping |
| Issue doesn't mention session follow-ups | Wraps onRequestDone to collect follow-up requests for export | ADDITION — without this, closed-loop session workloads export only round-0 requests (50%+ data loss) |
| Issue doesn't mention reasoning.go | Propagates ClientID in reasoning.go construction site | ADDITION — reasoning function receives clientID parameter but didn't set it on Request |
| Issue implies export before simulation (timing=0, status="ok") | Exports after simulation with real timing and derived status | SCOPE_CHANGE — post-simulation export captures timing and status, enabling richer trace analysis alongside A/B replay. Pre-determined OutputTokens still ensures replay fidelity. |
| Issue says parse RequestID from "request_N" | Uses array index as RequestID | CORRECTION — session follow-up IDs are non-numeric ("session_X_round_Y_Z"); array index is universal |

### E) Review Guide

**Scrutinize:** The `RequestsToTraceRecords` conversion function — timing field computation (relative→absolute), actual output token count for length-capped requests, array-index RequestID assignment. The session.go `ClientID` propagation.

**Safe to skim:** CLI flag registration (follows existing `--results-path` pattern). The 3 new Request fields (zero-value safe strings/bool, same pattern as TenantID).

**Known debt:**
- `ServerInputTokens` always 0 in simulation mode
- `NumChunks` always 0 (no streaming chunk tracking in DES)
- `SendTimeUs` set to `ArrivalTimeUs` (no real network send)
- `LastChunkTimeUs` excludes `PostDecodeFixedOverhead` — records client-observable timing, not server E2E. See BC-3 MECHANISM.
- `PrefixGroup` cleared in trace records: `InputTokens` already includes prefix tokens baked in during generation. Setting `PrefixGroup` would cause `LoadTraceV2Requests` to double-prepend prefix (50 tokens per request). `PrefixGroup` on `sim.Request` is still populated for metadata purposes but not exported.
- `Streaming` and `PrefixGroup` NOT propagated to session follow-ups (only `ClientID` via `SessionBlueprint`). Follow-ups get zero-value. Propagating requires SessionBlueprint changes (separate PR).
- `StateRunning` requests with `TTFTSet=true` produce `LastChunkTimeUs` from partial ITL (last token generated so far, not final token). Status "incomplete" distinguishes these from completed requests.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | What |
|---|---|---|
| `sim/request.go` | Modify | Add `ClientID`, `PrefixGroup`, `Streaming` fields |
| `sim/workload/generator.go` | Modify | Populate new fields from `ClientSpec` during generation |
| `sim/workload/replay.go` | Modify | Map `ClientID`, `PrefixGroup`, `Streaming` from TraceRecord |
| `sim/workload/session.go` | Modify | Propagate `ClientID` from `SessionBlueprint` to follow-up requests |
| `sim/workload/reasoning.go` | Modify | Set `ClientID` from `clientID` parameter in construction site |
| `sim/workload/tracev2.go` | Modify | Add `RequestsToTraceRecords` conversion function |
| `sim/workload/tracev2_test.go` | Modify | Add unit + invariant tests for conversion |
| `cmd/root.go` | Modify | Add `--trace-output` flag + post-simulation export logic |
| `CLAUDE.md` | Modify | Add `--trace-output` to CLI flags documentation |

No dead code. No new interfaces. No new packages.

### G) Task Breakdown

**Task 1: Add metadata fields + test** (BC-5)

_Test first:_
1. Write `TestRequestMetadataFields` in `sim/workload/tracev2_test.go` — construct a `*sim.Request` with `ClientID`, `PrefixGroup`, `Streaming` set, pass through `RequestsToTraceRecords`, assert fields preserved on `TraceRecord`. This will fail to compile (fields don't exist yet, function doesn't exist yet).
2. Add `ClientID string`, `PrefixGroup string`, `Streaming bool` to `sim/request.go` in the "Workload metadata" section (after `ReasonRatio`)
3. Add stub `RequestsToTraceRecords` that returns empty slice — test still fails (wrong values)
4. Implement `RequestsToTraceRecords` field mapping for metadata — test passes
5. Populate in `sim/workload/generator.go:269`: `ClientID: client.ID`, `PrefixGroup: client.PrefixGroup`, `Streaming: client.Streaming`
6. Map in `sim/workload/replay.go:44`: `ClientID: rec.ClientID`, `PrefixGroup: rec.PrefixGroup`, `Streaming: rec.Streaming`
7. Propagate in `sim/workload/session.go:168`: add `ClientID: bp.ClientID` to follow-up request construction
8. Set in `sim/workload/reasoning.go:68`: add `ClientID: clientID` (parameter already available at line 19)
9. `go test ./sim/workload/... && golangci-lint run ./sim/... ./cmd/...`

**Task 2: Conversion function — token counts + status + timing** (BC-2, BC-3, BC-9)

_Test first:_
1. Write `TestRequestsToTraceRecords_FieldMapping` — construct requests with known InputTokens, OutputTokens, State, FirstTokenTime, ITL, ArrivalTime. Assert: OutputTokens in trace = len(req.OutputTokens) (pre-determined, NOT actual), correct status mapping, correct absolute timing, correct record count (BC-9).
2. Write `TestRequestsToTraceRecords_StatusMapping` — table-driven: `StateCompleted`→"ok", `StateTimedOut`→"timeout", `StateQueued`→"incomplete", `StateRunning`→"incomplete".
3. Write `TestRequestsToTraceRecords_TimingCausality` — for completed requests, assert `FirstChunkTimeUs >= ArrivalTimeUs` and `LastChunkTimeUs >= FirstChunkTimeUs`. For requests with `TTFTSet=false`, assert both chunk timestamps are 0 (INV-5 projection).
4. Write `TestRequestsToTraceRecords_PrefillTimeout` — request with `State=StateTimedOut`, `TTFTSet=false`. Assert `FirstChunkTimeUs == 0` and `LastChunkTimeUs == 0` (not `ArrivalTime`).
5. Implement full `RequestsToTraceRecords` in `sim/workload/tracev2.go`:
   - Uses array index as `RequestID` (not string parsing)
   - `OutputTokens`: `len(req.OutputTokens)` for ALL requests (pre-determined count = workload input for replay fidelity)
   - Timing: guarded by `TTFTSet` — only compute `FirstChunkTimeUs` and `LastChunkTimeUs` when TTFT was set. Comment explaining exclusion of PostDecodeFixedOverhead.
   - Status: switch on `req.State`
6. All tests pass. `golangci-lint run ./sim/workload/...`

**Task 3: Round-trip test** (BC-4)

_Test first:_
1. Write `TestRequestsToTraceRecords_RoundTrip` — construct requests, convert to TraceRecords, export to temp files via `ExportTraceV2`, load back via `LoadTraceV2`, assert: token counts match, metadata matches, arrival times match. Also assert token sequences do NOT match (defensive — LoadTraceV2Requests generates new random IDs).
2. Test passes (uses implementation from Tasks 1-2). `golangci-lint run ./sim/workload/...`

**Task 4: CLI flag + export logic + session follow-up collection** (BC-1, BC-6, BC-7, BC-8, BC-9)

1. Add `traceOutput string` var in `cmd/root.go` globals (~line 101)
2. Register flag in `init()`: `runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")`
3. Wrap `onRequestDone` to collect follow-up requests when trace export is enabled (~line 995):

```go
var followUpRequests []*sim.Request
var onRequestDone func(*sim.Request, int64) []*sim.Request
if sessionMgr != nil {
    baseCb := sessionMgr.OnComplete
    if traceOutput != "" {
        // Wrap callback to accumulate follow-up requests for trace export
        onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
            followUps := baseCb(req, clock)
            followUpRequests = append(followUpRequests, followUps...)
            return followUps
        }
    } else {
        onRequestDone = baseCb
    }
}
```

4. After `cs.Run()` completes (~line 1002), merge and export:

```go
if traceOutput != "" {
    allRequests := make([]*sim.Request, 0, len(preGeneratedRequests)+len(followUpRequests))
    allRequests = append(allRequests, preGeneratedRequests...)
    allRequests = append(allRequests, followUpRequests...)
    // Sort by arrival time so RequestIDs (array indices) are arrival-ordered
    sort.SliceStable(allRequests, func(i, j int) bool {
        return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
    })
    records := workload.RequestsToTraceRecords(allRequests)
    header := &workload.TraceHeader{
        Version:  2,
        TimeUnit: "microseconds",
        Mode:     "generated",
    }
    if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
        logrus.Fatalf("Trace export failed: %v", err)
    }
    logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
}
```

5. `go build ./... && go test ./... && golangci-lint run ./...`

**Task 5: Update CLAUDE.md** — add `--trace-output` to CLI flags list in `cmd/root.go` comment and File Organization tree.

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|---|---|---|---|
| BC-2 | Task 2 | Unit | `TestRequestsToTraceRecords_FieldMapping` |
| BC-3 | Task 2 | Unit | `TestRequestsToTraceRecords_FieldMapping` (timing) |
| BC-3 | Task 2 | Invariant | `TestRequestsToTraceRecords_TimingCausality` |
| BC-4 | Task 3 | Invariant | `TestRequestsToTraceRecords_RoundTrip` |
| BC-5 | Task 1 | Unit | `TestRequestMetadataFields` |
| BC-3 | Task 2 | Unit | `TestRequestsToTraceRecords_PrefillTimeout` |
| BC-6 | N/A | By design | Export writes to file + logrus (stderr), no stdout |
| BC-7 | N/A | By design | Guarded by `if traceOutput != ""` |
| BC-8 | N/A | By construction | `ExportTraceV2` returns error → `logrus.Fatalf` |
| BC-9 | Task 2 | Invariant | `TestRequestsToTraceRecords_FieldMapping` (count assertion) |

**Invariant tests (R7):**
- **Record count conservation (BC-9):** `len(records) == len(requests)` — asserted in FieldMapping test
- **Timing causality (INV-5 projection):** `FirstChunkTimeUs >= ArrivalTimeUs && LastChunkTimeUs >= FirstChunkTimeUs` — dedicated test
- **Round-trip preservation (BC-4):** export→load preserves counts and metadata — dedicated test

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LastChunkTimeUs differs from E2E in metrics output | High | Low | Documented semantic: trace records client-observable timing (excludes PostDecodeFixedOverhead). Code comment prevents incorrect "optimization." |
| Session follow-up PrefixGroup/Streaming empty | Low | Low | Documented as known debt. ClientID propagated; PrefixGroup/Streaming require SessionBlueprint changes (separate PR). |
| OutputTokens is pre-determined, not actual generated | Medium | Low | Deliberate for A/B replay fidelity — trace is workload INPUT, not simulation RESULT. Documented in BC-2 MECHANISM. |
| Prefill-timeout timing produces 0 instead of ArrivalTime | Medium | Medium | Guarded by TTFTSet check. Tested in `TestRequestsToTraceRecords_PrefillTimeout`. |
| New Request fields break determinism (INV-6) | Low | High | Fields are zero-value safe, not printed to stdout. Verified by `go test ./...`. |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific:**
- [x] No unnecessary abstractions (single conversion function, no new interfaces)
- [x] No feature creep (just export, no new analysis)
- [x] No unexercised flags (`--trace-output` used in export)
- [x] No partial implementations (full pipeline: generate → populate → convert → export)
- [x] No breaking changes (3 new fields are zero-value safe)
- [x] Shared test helpers: uses existing `ExportTraceV2`/`LoadTraceV2` for round-trip
- [x] CLAUDE.md update: Task 5
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed: 9 deviations, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies ordered: 1 → 2 → 3 → 4 → 5
- [x] All contracts mapped to tasks (H table)
- [x] Golden dataset: unaffected (no simulation output changes)
- [x] Construction site audit: 5 sites listed, 4 updated explicitly (generator, replay, session, reasoning)

**Antipattern rules:**
- [x] R1: No silent data loss (export errors → `logrus.Fatalf`)
- [x] R2: N/A (no map iteration in new code)
- [x] R3: N/A (no new numeric parameters)
- [x] R4: Construction sites audited (generator.go, replay.go, session.go, reasoning.go updated; test files zero-value safe)
- [x] R5: N/A (no resource-allocating loops)
- [x] R6: No Fatalf in `sim/` (conversion is pure function)
- [x] R7: Invariant tests alongside unit tests (record count, timing causality, round-trip)
- [x] R8: N/A (no new maps)
- [x] R9: N/A (no new YAML config fields)
- [x] R10: N/A (no new YAML parsing)
- [x] R11: N/A (no division)
- [x] R12: N/A (golden dataset unaffected)
- [x] R13: N/A (no new interfaces)
- [x] R14: N/A (no multi-concern methods)
- [x] R15: N/A (no PR references to update)
- [x] R16: N/A (no new config types)
- [x] R17: N/A (no routing signals)
- [x] R18: N/A (no defaults.yaml interaction for this flag)
- [x] R19: N/A (no retry loops)
- [x] R20: N/A (no anomaly detectors)
- [x] R21: N/A (no range over mutable slices)
- [x] R22: N/A (no pre-checks)
- [x] R23: N/A (no parallel code paths)

---

## Appendix: File-Level Implementation Details

### `sim/request.go`

Add after `ReasonRatio` field (line 61):

```go
ClientID    string  // Client identifier from workload spec (empty for legacy/test workloads)
PrefixGroup string  // Shared prefix group name (empty for no prefix)
Streaming   bool    // Whether client expects streaming response
```

### `sim/workload/tracev2.go`

Add `RequestsToTraceRecords` function:

```go
// RequestsToTraceRecords converts simulation requests to trace v2 records.
// Uses array index as RequestID (request IDs may be non-numeric for session follow-ups).
// LastChunkTimeUs is computed as ArrivalTime + FirstTokenTime + sum(ITL), which
// represents the client-observable last-token delivery time. This deliberately
// excludes PostDecodeFixedOverhead (server-side processing after final token)
// and therefore differs from the E2E value stored in Metrics.RequestE2Es.
func RequestsToTraceRecords(requests []*sim.Request) []TraceRecord {
    records := make([]TraceRecord, 0, len(requests))
    for i, req := range requests {
        status := "incomplete"
        switch req.State {
        case sim.StateCompleted:
            status = "ok"
        case sim.StateTimedOut:
            status = "timeout"
        }

        // Absolute timing (ticks = microseconds)
        // Both chunk timestamps guarded by TTFTSet to avoid producing
        // LastChunkTimeUs = ArrivalTime for prefill-timeout requests.
        // For StateRunning requests with TTFTSet=true, LastChunkTimeUs
        // represents the last token generated so far (partial execution),
        // not the final token. Status "incomplete" distinguishes these.
        var firstChunkUs, lastChunkUs int64
        if req.TTFTSet {
            firstChunkUs = req.ArrivalTime + req.FirstTokenTime
            // LastChunkTimeUs = ArrivalTime + FirstTokenTime + sum(ITL)
            // This deliberately excludes PostDecodeFixedOverhead (server-side
            // processing after final token) and therefore differs from the
            // E2E value stored in Metrics.RequestE2Es.
            e2e := req.FirstTokenTime
            for _, itl := range req.ITL {
                e2e += itl
            }
            lastChunkUs = req.ArrivalTime + e2e
        }

        records = append(records, TraceRecord{
            RequestID:        i, // array index, not parsed from string ID
            ClientID:         req.ClientID,
            TenantID:         req.TenantID,
            SLOClass:         req.SLOClass,
            SessionID:        req.SessionID,
            RoundIndex:       req.RoundIndex,
            PrefixGroup:      "", // intentionally empty: InputTokens already includes prefix; setting PrefixGroup would cause LoadTraceV2Requests to double-prepend
            Streaming:        req.Streaming,
            InputTokens:      len(req.InputTokens),
            OutputTokens:     len(req.OutputTokens), // pre-determined count for replay fidelity
            TextTokens:       req.TextTokenCount,
            ImageTokens:      req.ImageTokenCount,
            AudioTokens:      req.AudioTokenCount,
            VideoTokens:      req.VideoTokenCount,
            ReasonRatio:      req.ReasonRatio,
            Model:            req.Model,
            DeadlineUs:       req.Deadline,
            ArrivalTimeUs:    req.ArrivalTime,
            SendTimeUs:       req.ArrivalTime, // no real network send in simulation
            FirstChunkTimeUs: firstChunkUs,
            LastChunkTimeUs:  lastChunkUs,
            NumChunks:        0, // not tracked in simulation
            Status:           status,
        })
    }
    return records
}
```

### `sim/workload/generator.go`

At the request construction site (~line 269), add to the `sim.Request{}` literal:

```go
ClientID:    client.ID,
PrefixGroup: client.PrefixGroup,
Streaming:   client.Streaming,
```

### `sim/workload/replay.go`

At the `LoadTraceV2Requests` construction site (~line 44), add:

```go
ClientID:    rec.ClientID,
PrefixGroup: rec.PrefixGroup,
Streaming:   rec.Streaming,
```

Note: `PrefixGroup` on Request is for metadata round-trip only. `LoadTraceV2Requests` already handles prefix token prepend via `rec.PrefixGroup` directly (lines 22-40).

### `sim/workload/session.go`

At the follow-up request construction site (~line 168), add:

```go
ClientID: bp.ClientID,
```

`PrefixGroup` and `Streaming` are NOT propagated — `SessionBlueprint` doesn't carry them. Documented as known debt.

### `sim/workload/reasoning.go`

At the request construction site (~line 68), add:

```go
ClientID: clientID,
```

The `clientID` parameter is already available at line 19 of `GenerateReasoningRequests`. `PrefixGroup` and `Streaming` are not passed to this function and default to zero-value.

### `cmd/root.go`

Add global var (~line 101):
```go
traceOutput string
```

Register flag in `init()`:
```go
runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")
```

Wrap `onRequestDone` to collect follow-ups (~line 995):
```go
var followUpRequests []*sim.Request
var onRequestDone func(*sim.Request, int64) []*sim.Request
if sessionMgr != nil {
    baseCb := sessionMgr.OnComplete
    if traceOutput != "" {
        onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
            followUps := baseCb(req, clock)
            followUpRequests = append(followUpRequests, followUps...)
            return followUps
        }
    } else {
        onRequestDone = baseCb
    }
}
```

Export logic after `cs.Run()` (~line 1002):
```go
if traceOutput != "" {
    allRequests := make([]*sim.Request, 0, len(preGeneratedRequests)+len(followUpRequests))
    allRequests = append(allRequests, preGeneratedRequests...)
    allRequests = append(allRequests, followUpRequests...)
    // Sort by arrival time so RequestIDs (array indices) are arrival-ordered
    sort.SliceStable(allRequests, func(i, j int) bool {
        return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
    })
    records := workload.RequestsToTraceRecords(allRequests)
    header := &workload.TraceHeader{
        Version:  2,
        TimeUnit: "microseconds",
        Mode:     "generated",
    }
    if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
        logrus.Fatalf("Trace export failed: %v", err)
    }
    logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
}
```

---

## Verification

```bash
go build ./...
go test ./...
golangci-lint run ./...

# Manual verification
./blis run --model qwen/qwen3-14b --num-requests 10 --trace-output /tmp/test-trace
cat /tmp/test-trace.yaml
head -5 /tmp/test-trace.csv
```
