# Fix Trace V2 Prefix-Group Preservation — Implementation Plan

**Goal:** Preserve prefix-group structure through the trace v2 export/import cycle so that `blis replay` and `blis calibrate` produce correct prefix cache hit rates for prefix-heavy workloads.

**The problem today:** The trace v2 write path (`RequestsToTraceRecords`) intentionally clears `PrefixGroup` and records `InputTokens` as total (prefix + suffix). On replay, `LoadTraceV2Requests` never enters the prefix-building branch, so all requests get unique random token sequences with zero shared prefix. Additionally, the replay path hardcodes a 50-token prefix length regardless of the original workload specification.

**What this PR adds:**
1. A new `prefix_length` column in the trace v2 CSV schema, placed after `prefix_group`
2. `PrefixGroup` preservation through the write path (no longer cleared)
3. `InputTokens` semantics change to suffix-only count in the CSV (prefix_length + input_tokens = original total)
4. A `PrefixLength` field on `sim.Request` to carry prefix metadata through the pipeline
5. A `WorkloadSeed` field on `TraceHeader` for reproducible prefix string generation during re-observation
6. Backward compatibility for old 26-column traces (detected by column count, treated as legacy)

**Why this matters:** Without this fix, `blis replay` and `blis calibrate` silently produce wrong results for any workload using prefix groups — the most common production workload pattern.

**Architecture:** Changes span `sim/request.go` (new field), `sim/workload/tracev2.go` (schema + write path), `sim/workload/replay.go` (read path), `cmd/observe_cmd.go` (seed in header), and `cmd/root.go` (seed in header). All changes are within the existing trace v2 subsystem with no new interfaces or modules.

**Source:** [Issue #829](https://github.com/inference-sim/inference-sim/issues/829)

**Closes:** Fixes #829

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Trace v2 schema (write path: `RequestsToTraceRecords`, read path: `LoadTraceV2Requests`, CSV format: `traceV2Columns`)
2. **Adjacent blocks:** Workload generator (`generatePrefixTokens`, `GenerateRequests`), observe command (`buildPrefixStrings`), replay command, calibrate command
3. **Invariants touched:** INV-6 (Determinism) — same seed must produce identical replay. R23 (parallel code path parity) — run/observe/replay must produce equivalent prefix structure.
4. **Construction site audit:**
   - `TraceRecord{}`: `tracev2.go:395` (RequestsToTraceRecords), `tracev2.go:327` (parseTraceRecord), `tracev2_test.go` (multiple test sites), `replay_test.go` (multiple test sites), `calibrate_test.go`, `cmd/observe.go`
   - `TraceHeader{}`: `cmd/root.go:1372`, `cmd/observe_cmd.go:270`, `tracev2_test.go` (multiple), `replay_test.go` (multiple), `cmd/observe_cmd_test.go:414`
   - `sim.Request{}`: `replay.go:44` (LoadTraceV2Requests), `generator.go` (multiple), plus test sites — only `replay.go` needs the new `PrefixLength` field wired

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a data loss bug in the trace v2 export/import pipeline where prefix-group information is silently discarded. The fix adds a `prefix_length` column to the CSV schema, changes `input_tokens` to record suffix-only token counts, preserves `PrefixGroup` through the write path, adds `PrefixLength` to `sim.Request`, and stores `WorkloadSeed` in the trace header. Old traces are detected by column count and parsed in legacy mode. The changes touch 5 Go source files and their corresponding tests.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Prefix group round-trip preservation
- GIVEN a set of requests with non-empty `PrefixGroup` and known `PrefixLength`
- WHEN exported via `RequestsToTraceRecords` and loaded via `LoadTraceV2`
- THEN the loaded `TraceRecord` has the original `PrefixGroup` value (non-empty) and correct `PrefixLength`
- MECHANISM: Write path copies `req.PrefixGroup` and `req.PrefixLength` to the record; CSV includes new `prefix_length` column

BC-2: Input tokens suffix-only semantics
- GIVEN a request with `PrefixLength=P` and `len(InputTokens)=T`
- WHEN exported via `RequestsToTraceRecords`
- THEN the `TraceRecord.InputTokens` equals `T - P` (suffix-only count)
- MECHANISM: Write path subtracts `PrefixLength` from total input token count

BC-3: Replay prefix sharing
- GIVEN a trace with two requests in the same prefix group with `PrefixLength=128` and `InputTokens=100` (suffix)
- WHEN loaded via `LoadTraceV2Requests`
- THEN both requests have identical first 128 tokens, and total input length is 128 + 100 = 228 per request
- MECHANISM: Read path generates shared prefix of `rec.PrefixLength` tokens per group, prepends to suffix-length random tokens

BC-4: Workload seed header preservation
- GIVEN a `TraceHeader` with `WorkloadSeed=42`
- WHEN exported and loaded via `ExportTraceV2` / `LoadTraceV2`
- THEN the loaded header has `WorkloadSeed=42`

BC-5: Backward compatibility with 26-column traces
- GIVEN a legacy 26-column CSV (no `prefix_length` column)
- WHEN loaded via `LoadTraceV2`
- THEN all records parse successfully with `PrefixLength=0`, and `InputTokens` is treated as total (current behavior)

**Negative contracts:**

BC-6: No double-prepend on replay
- GIVEN a trace exported with the new schema (prefix_length + suffix-only input_tokens)
- WHEN loaded via `LoadTraceV2Requests`
- THEN total input tokens per request = prefix_length + input_tokens (no double counting)

BC-7: Zero prefix length for non-prefix requests
- GIVEN a request with empty `PrefixGroup`
- WHEN exported via `RequestsToTraceRecords`
- THEN `PrefixLength=0` and `InputTokens` equals `len(req.InputTokens)` (unchanged)

### C) Component Interaction

```
Workload Generator                    Observe Command
  GenerateRequests()                    buildPrefixStrings()
  → sets req.PrefixGroup               → uses spec.Seed
  → sets req.PrefixLength (NEW)        → builds prefix strings
  → prepends prefix to InputTokens     → sends to server
        │                                     │
        ▼                                     ▼
  RequestsToTraceRecords()             Recorder.Export()
  → writes PrefixGroup (RESTORED)     → writes PrefixGroup from server response
  → writes PrefixLength (NEW)         → writes PrefixLength (NEW from req)
  → writes InputTokens as suffix      → header.WorkloadSeed = spec.Seed (NEW)
        │                                     │
        ▼                                     ▼
    TraceV2 CSV (27 columns)           TraceV2 YAML + CSV
        │                                     │
        ▼                                     ▼
  LoadTraceV2() → parseTraceRecord()
  → reads prefix_length column (NEW)
  → detects 26-col legacy → PrefixLength=0
        │
        ▼
  LoadTraceV2Requests()
  → uses rec.PrefixLength (not hardcoded 50)
  → generates prefix of correct length
  → prepends to suffix-only input tokens
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue proposes storing `WorkloadSeed` for re-observation path | Plan adds `WorkloadSeed` to header but does not modify the re-observation code path | DEFERRAL: Re-observation from saved trace is a separate feature; this PR focuses on the replay/calibrate path. The seed is stored for future use. |
| Issue proposes `sim.Request` carry `PrefixLength int` | Plan adds the field but only wires it through the generator and trace paths | CLARIFICATION: The field is only needed by the trace write path; other pipeline stages don't need it |

### E) Review Guide

**Tricky part:** The `input_tokens` semantics change from total to suffix-only. Every code path that reads `TraceRecord.InputTokens` or writes it must be audited. The backward compat detection (26 vs 27 columns) must be correct.

**Scrutinize:** The `parseTraceRecord` column index shifts — adding `prefix_length` after column 7 (`prefix_group`) shifts all subsequent column indices by 1. This is the highest-risk change.

**Safe to skim:** `TraceHeader.WorkloadSeed` addition (simple YAML field), `sim.Request.PrefixLength` addition (simple int field).

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `sim/request.go` — Add `PrefixLength int` field
- `sim/workload/tracev2.go` — Add `PrefixLength` to `TraceRecord`, `WorkloadSeed` to `TraceHeader`, add `prefix_length` to CSV columns, update write/parse paths, backward compat
- `sim/workload/replay.go` — Use `rec.PrefixLength` instead of hardcoded 50; suffix-only InputTokens semantics
- `sim/workload/generator.go` — Set `req.PrefixLength` during generation
- `cmd/observe_cmd.go` — Set `header.WorkloadSeed` from spec seed
- `cmd/root.go` — Set `header.WorkloadSeed` from spec seed (run command trace export)

Key decisions:
- Column insertion: `prefix_length` goes after `prefix_group` (column index 7), shifting subsequent columns by +1
- Backward compat: Detect by column count (26 = legacy, 27+ = new schema). Legacy traces get `PrefixLength=0`
- No dead code: All new fields are exercised by both write and read paths

### G) Task Breakdown

#### Task 1: Add PrefixLength to sim.Request and TraceRecord (BC-1, BC-7)

**Files:** modify `sim/request.go`, modify `sim/workload/tracev2.go`

**Test:** (in Task 2 — this task is schema-only, tested through the round-trip in Task 2)

**Impl:**

1. In `sim/request.go`, add after `PrefixGroup` field:
```go
PrefixLength    int    // Shared prefix token count; 0 = no prefix. Set during workload generation.
```

2. In `sim/workload/tracev2.go`, add to `TraceRecord` struct after `PrefixGroup`:
```go
PrefixLength int
```

3. In `sim/workload/tracev2.go`, add to `TraceHeader` struct:
```go
WorkloadSeed int64 `yaml:"workload_seed,omitempty"`
```

4. Update `traceV2Columns` — insert `"prefix_length"` after `"prefix_group"`:
```go
var traceV2Columns = []string{
    "request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
    "prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
    "text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
    "model", "deadline_us", "server_input_tokens",
    "arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
    "num_chunks", "status", "error_message", "finish_reason",
}
```

5. Update CSV write in `ExportTraceV2` — insert `PrefixLength` after `PrefixGroup` in the row:
```go
row := []string{
    strconv.Itoa(r.RequestID),
    r.ClientID,
    r.TenantID,
    r.SLOClass,
    r.SessionID,
    strconv.Itoa(r.RoundIndex),
    r.PrefixGroup,
    strconv.Itoa(r.PrefixLength), // NEW
    strconv.FormatBool(r.Streaming),
    strconv.Itoa(r.InputTokens),
    // ... rest unchanged
```

6. Update `parseTraceRecord` — shift all column indices after column 7 by +1, add `prefix_length` parsing at column 7:
```go
// Column 7: prefix_length (new)
prefixLength, err := strconv.Atoi(row[7])
if err != nil {
    return nil, fmt.Errorf("parsing prefix_length %q: %w", row[7], err)
}
if prefixLength < 0 {
    return nil, fmt.Errorf("parsing prefix_length: negative value %d not allowed", prefixLength)
}
// Column 8: streaming (was 7)
streaming, err := strconv.ParseBool(row[8])
// Column 9: input_tokens (was 8)
inputTokens, err := strconv.Atoi(row[9])
// ... shift all remaining columns by +1
```

7. Update backward compat in `LoadTraceV2`: detect 26-column (old) vs 27-column (new) traces. For 26-column traces, use the old `parseTraceRecord` logic (call a `parseTraceRecordLegacy` function that keeps old indices).

8. Update `RequestsToTraceRecords` — restore `PrefixGroup` and add `PrefixLength`, change `InputTokens` to suffix-only:
```go
prefixLen := req.PrefixLength
inputTokens := len(req.InputTokens) - prefixLen
if inputTokens < 0 {
    // Safety: PrefixLength exceeds InputTokens (should not happen with well-formed data).
    // Treat as no prefix. Detectable in output: PrefixLength=0 with non-empty PrefixGroup.
    // R6: no logrus in sim/ — caller is responsible for detecting this via the record.
    inputTokens = len(req.InputTokens)
    prefixLen = 0
}
// ...
PrefixGroup:  req.PrefixGroup,  // RESTORED
PrefixLength: prefixLen,        // NEW
InputTokens:  inputTokens,      // CHANGED: suffix-only
```

**Verify:** `go test ./sim/workload/... -run TestTraceV2`
**Lint:** `golangci-lint run ./sim/... ./cmd/...`
**Commit:** `fix(trace): add prefix_length column and preserve prefix group in trace v2 schema (BC-1, BC-7)`

#### Task 2: Test prefix group round-trip (BC-1, BC-2, BC-7)

**Files:** modify `sim/workload/tracev2_test.go`

**Test:**
```go
func TestTraceV2_PrefixGroup_RoundTrip(t *testing.T) {
    // BC-1: PrefixGroup and PrefixLength survive export → load
    header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated", WorkloadSeed: 42}
    records := []TraceRecord{
        {RequestID: 0, PrefixGroup: "group-a", PrefixLength: 128,
            InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, Status: "ok"},
        {RequestID: 1, PrefixGroup: "group-a", PrefixLength: 128,
            InputTokens: 200, OutputTokens: 75, ArrivalTimeUs: 1000, Status: "ok"},
        {RequestID: 2, PrefixGroup: "", PrefixLength: 0,
            InputTokens: 300, OutputTokens: 100, ArrivalTimeUs: 2000, Status: "ok"},
    }

    dir := t.TempDir()
    headerPath := filepath.Join(dir, "header.yaml")
    dataPath := filepath.Join(dir, "data.csv")
    if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
        t.Fatal(err)
    }

    loaded, err := LoadTraceV2(headerPath, dataPath)
    if err != nil {
        t.Fatal(err)
    }

    // BC-1: prefix group preserved
    if loaded.Records[0].PrefixGroup != "group-a" {
        t.Errorf("record 0 PrefixGroup = %q, want %q", loaded.Records[0].PrefixGroup, "group-a")
    }
    if loaded.Records[0].PrefixLength != 128 {
        t.Errorf("record 0 PrefixLength = %d, want 128", loaded.Records[0].PrefixLength)
    }
    // BC-2: input_tokens is suffix-only
    if loaded.Records[0].InputTokens != 100 {
        t.Errorf("record 0 InputTokens = %d, want 100", loaded.Records[0].InputTokens)
    }
    // BC-7: non-prefix request unchanged
    if loaded.Records[2].PrefixGroup != "" {
        t.Errorf("record 2 PrefixGroup = %q, want empty", loaded.Records[2].PrefixGroup)
    }
    if loaded.Records[2].PrefixLength != 0 {
        t.Errorf("record 2 PrefixLength = %d, want 0", loaded.Records[2].PrefixLength)
    }
    if loaded.Records[2].InputTokens != 300 {
        t.Errorf("record 2 InputTokens = %d, want 300", loaded.Records[2].InputTokens)
    }
    // BC-4: WorkloadSeed preserved
    if loaded.Header.WorkloadSeed != 42 {
        t.Errorf("WorkloadSeed = %d, want 42", loaded.Header.WorkloadSeed)
    }
}
```

**Impl:** Already implemented in Task 1.

**Verify:** `go test ./sim/workload/... -run TestTraceV2_PrefixGroup_RoundTrip`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `test(trace): add prefix group round-trip test (BC-1, BC-2, BC-7)`

#### Task 3: Update replay path to use PrefixLength (BC-3, BC-6)

**Files:** modify `sim/workload/replay.go`

**Test:** Update `TestLoadTraceV2Requests_PrefixGroup_SharedTokens` in `sim/workload/replay_test.go`:
```go
func TestLoadTraceV2Requests_PrefixGroup_SharedTokens(t *testing.T) {
    header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
    records := []TraceRecord{
        {RequestID: 0, InputTokens: 100, OutputTokens: 50,
            PrefixGroup: "shared", PrefixLength: 128, ArrivalTimeUs: 0, Status: "ok"},
        {RequestID: 1, InputTokens: 100, OutputTokens: 50,
            PrefixGroup: "shared", PrefixLength: 128, ArrivalTimeUs: 100000, Status: "ok"},
    }

    dir := t.TempDir()
    headerPath := filepath.Join(dir, "header.yaml")
    dataPath := filepath.Join(dir, "data.csv")
    if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
        t.Fatal(err)
    }

    trace, err := LoadTraceV2(headerPath, dataPath)
    if err != nil {
        t.Fatal(err)
    }

    requests, err := LoadTraceV2Requests(trace, 42)
    if err != nil {
        t.Fatal(err)
    }

    // BC-3: Both requests share identical first 128 tokens
    if len(requests[0].InputTokens) < 128 || len(requests[1].InputTokens) < 128 {
        t.Fatal("input tokens too short for prefix check")
    }
    for i := 0; i < 128; i++ {
        if requests[0].InputTokens[i] != requests[1].InputTokens[i] {
            t.Errorf("prefix token %d differs: %d vs %d", i,
                requests[0].InputTokens[i], requests[1].InputTokens[i])
            break
        }
    }
    // BC-6: Total input length = prefix(128) + suffix(100) = 228
    if len(requests[0].InputTokens) != 228 {
        t.Errorf("input length = %d, want 228 (128 prefix + 100 suffix)", len(requests[0].InputTokens))
    }
    // BC-3: PrefixGroup propagated to Request
    if requests[0].PrefixGroup != "shared" {
        t.Errorf("PrefixGroup = %q, want %q", requests[0].PrefixGroup, "shared")
    }
}
```

**Impl:** Update `LoadTraceV2Requests` in `replay.go`:
```go
// Generate shared prefix tokens per prefix group using trace-specified length
prefixTokens := make(map[string][]int)
for _, rec := range trace.Records {
    if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
        if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
            prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
        }
    }
}

// ...per request:
// InputTokens is now suffix-only; generate suffix tokens
inputTokens := sim.GenerateRandomTokenIDs(rng, rec.InputTokens)

// Prepend prefix if in a group
if rec.PrefixGroup != "" {
    if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
        inputTokens = append(append([]int{}, prefix...), inputTokens...)
    }
}
```

Also add `PrefixLength` to the Request construction in `LoadTraceV2Requests` (line 65, after `PrefixGroup`):
```go
PrefixGroup:  rec.PrefixGroup,
PrefixLength: rec.PrefixLength, // NEW: propagate for downstream trace re-export
```

**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2Requests`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(replay): use trace prefix_length instead of hardcoded 50 (BC-3, BC-6)`

#### Task 4: Backward compatibility for 26-column traces (BC-5)

**Files:** modify `sim/workload/tracev2.go`, add/update test in `sim/workload/tracev2_test.go`

**Test:**
```go
func TestTraceV2_BackwardCompat_26Columns(t *testing.T) {
    // BC-5: 26-column CSV (pre-prefix_length) loads with PrefixLength=0
    header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
    headerPath := filepath.Join(t.TempDir(), "h.yaml")
    dataPath := filepath.Join(t.TempDir(), "d.csv")

    headerData, err := yaml.Marshal(header)
    if err != nil {
        t.Fatal(err)
    }
    if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
        t.Fatal(err)
    }

    // Write 26-column CSV (old schema without prefix_length)
    csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index," +
        "prefix_group,streaming,input_tokens,output_tokens," +
        "text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio," +
        "model,deadline_us,server_input_tokens," +
        "arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us," +
        "num_chunks,status,error_message,finish_reason\n" +
        "0,c1,t1,batch,,0,group-a,false,500,100,0,0,0,0,0,model1,0,0,1000,1000,0,0,0,ok,,stop\n"
    if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
        t.Fatal(err)
    }

    trace, err := LoadTraceV2(headerPath, dataPath)
    if err != nil {
        t.Fatalf("LoadTraceV2: %v", err)
    }

    // BC-5: PrefixLength defaults to 0 for legacy traces
    if trace.Records[0].PrefixLength != 0 {
        t.Errorf("PrefixLength = %d, want 0 for legacy trace", trace.Records[0].PrefixLength)
    }
    // PrefixGroup still preserved from CSV
    if trace.Records[0].PrefixGroup != "group-a" {
        t.Errorf("PrefixGroup = %q, want %q", trace.Records[0].PrefixGroup, "group-a")
    }
    // InputTokens is total for legacy (no prefix subtraction)
    if trace.Records[0].InputTokens != 500 {
        t.Errorf("InputTokens = %d, want 500", trace.Records[0].InputTokens)
    }
}
```

**Impl:** In `LoadTraceV2`, detect column count and dispatch to appropriate parser:
- 27+ columns: new schema with `prefix_length`
- 26 columns: legacy schema, use `parseTraceRecordLegacy` (current column indices, `PrefixLength=0`)
- 25 columns: existing backward compat (pre-finish_reason)

**Verify:** `go test ./sim/workload/... -run TestTraceV2_BackwardCompat`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(trace): backward compatibility for 26-column legacy traces (BC-5)`

#### Task 5: Wire PrefixLength through workload generator (BC-1, BC-2)

**Files:** modify `sim/workload/generator.go`, update test

**Test:** Add assertion in existing generator tests that `req.PrefixLength` is set correctly.

**Impl:** In `GenerateRequests`, set `req.PrefixLength` at ALL prefix prepend sites:

Site 1 — `generator.go:168` (reasoning requests, single-session):
```go
req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
req.PrefixLength = len(prefix) // NEW
```

Site 2 — `generator.go:216` (reasoning requests, multi-session):
```go
req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
req.PrefixLength = len(prefix) // NEW
```

Both sites are inside the `if prefix, ok := prefixes[client.PrefixGroup]; ok` block.

**Verify:** `go test ./sim/workload/... -run TestGenerate`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(generator): set PrefixLength on generated requests (BC-1, BC-2)`

#### Task 6: Wire WorkloadSeed through observe and run commands (BC-4)

**Files:** modify `cmd/observe_cmd.go`, modify `cmd/root.go`

**Impl:**
1. In `cmd/observe_cmd.go`, add `WorkloadSeed` to the header construction (guarded by nil check):
```go
// After existing header construction at line 270:
if spec != nil {
    header.WorkloadSeed = spec.Seed
}
```

2. In `cmd/root.go`, add `WorkloadSeed` to the trace export header at line 1372. The `spec` variable (declared at line ~910 in the same `Run` function) is in scope:
```go
header := &workload.TraceHeader{
    Version:      2,
    TimeUnit:     "microseconds",
    Mode:         "generated",
    WorkloadSeed: spec.Seed,
}
```

**Verify:** `go test ./cmd/... -run TestObserve`
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `fix(cmd): store workload seed in trace header (BC-4)`

#### Task 7: Update existing tests for new schema (BC-1 through BC-7)

**Files:** update `sim/workload/tracev2_test.go`, `sim/workload/replay_test.go`, `sim/workload/calibrate_test.go`, `cmd/observe_cmd_test.go`

**Impl:** Update all `TraceRecord{}` construction sites to include `PrefixLength: 0` where appropriate. Update `TraceHeader{}` construction sites where `WorkloadSeed` is relevant. Fix the existing round-trip test `TestRequestsToTraceRecords_RoundTrip` to expect non-empty `PrefixGroup` when set. Update `TestTraceV2_BackwardCompat_25Columns` to account for 27-column schema.

**Verify:** `go test ./sim/workload/... ./cmd/...`
**Lint:** `golangci-lint run ./sim/workload/... ./cmd/...`
**Commit:** `test(trace): update existing tests for 27-column schema (BC-1 through BC-7)`

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestTraceV2_PrefixGroup_RoundTrip |
| BC-2 | Task 2 | Unit | TestTraceV2_PrefixGroup_RoundTrip |
| BC-3 | Task 3 | Unit | TestLoadTraceV2Requests_PrefixGroup_SharedTokens |
| BC-4 | Task 2, 6 | Unit | TestTraceV2_PrefixGroup_RoundTrip (header), cmd tests |
| BC-5 | Task 4 | Unit | TestTraceV2_BackwardCompat_26Columns |
| BC-6 | Task 3 | Unit | TestLoadTraceV2Requests_PrefixGroup_SharedTokens |
| BC-7 | Task 2 | Unit | TestTraceV2_PrefixGroup_RoundTrip (record 2) |

Invariant tests:
- Token count conservation: `prefix_length + input_tokens = original total` (BC-2, BC-6)
- Prefix identity: requests in same group share identical first N tokens (BC-3)
- Backward compat: legacy traces load without error (BC-5)

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Column index shift breaks existing parsing | Medium | High | Backward compat detection + legacy parser | Task 4 |
| Existing tests break due to schema change | High | Medium | Task 7 updates all test construction sites | Task 7 |
| TraceHeader strict YAML parsing rejects `workload_seed` in old headers | Low | Medium | Field is `omitempty` — absent in old headers | Task 1 |
| PrefixLength not set on requests from non-generator paths | Low | Low | Default 0 = no prefix (safe fallback) | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (re-observation path deferred)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package
- [x] CLAUDE.md does not need updating (no new CLI flags, packages, or file organization changes)
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — re-observation deferred, justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered
- [x] All contracts mapped to tasks
- [x] Construction site audit completed

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data
- [x] R2: No map iteration for ordered output
- [x] R3: PrefixLength validated (negative check in parser)
- [x] R4: All construction sites audited (TraceRecord, TraceHeader, sim.Request)
- [x] R5: No resource allocation loops
- [x] R6: No logrus.Fatalf in sim/ packages
- [x] R7: Invariant tests alongside golden tests
- [x] R8: No exported mutable maps
- [x] R9: Not applicable (no new float64 YAML fields)
- [x] R10: TraceHeader uses strict YAML parsing (KnownFields(true))
- [x] R11: No division by runtime-derived denominators
- [x] R12: Golden datasets will be regenerated in Task 7
- [x] R13: No new interfaces
- [x] R14: No multi-module methods
- [x] R15: No stale PR references
- [x] R16: Not applicable (no new config params)
- [x] R17: Not applicable (no routing signals)
- [x] R18: Not applicable (no CLI flags)
- [x] R19: No unbounded retry loops
- [x] R20: Not applicable (no detectors/analyzers)
- [x] R21: No range over shrinking slices
- [x] R22: Not applicable
- [x] R23: This PR fixes the R23 violation — run/observe/replay now apply equivalent prefix transformations

---

## Appendix: File-Level Implementation Details

### File: `sim/request.go`

**Purpose:** Add `PrefixLength` field to `Request` struct.

Add after line 63 (`PrefixGroup` field):
```go
PrefixLength    int    // Shared prefix token count; 0 = no prefix. Set during workload generation.
```

### File: `sim/workload/tracev2.go`

**Purpose:** Schema changes (TraceRecord, TraceHeader, CSV columns), write path fix, parse path update, backward compat.

Changes:
1. `TraceHeader` — add `WorkloadSeed int64 \`yaml:"workload_seed,omitempty"\``
2. `TraceRecord` — add `PrefixLength int` after `PrefixGroup`
3. `traceV2Columns` — insert `"prefix_length"` after `"prefix_group"`
4. CSV write loop — insert `strconv.Itoa(r.PrefixLength)` after `r.PrefixGroup`
5. `parseTraceRecord` — shift all indices after column 7 by +1, parse `prefix_length` at column 7
6. `RequestsToTraceRecords` — restore `PrefixGroup`, add `PrefixLength`, change `InputTokens` to suffix-only
7. `LoadTraceV2` — detect column count for backward compat

### File: `sim/workload/replay.go`

**Purpose:** Use `rec.PrefixLength` instead of hardcoded 50 for prefix token generation.

Changes:
1. First loop: change `sim.GenerateRandomTokenIDs(rng, 50)` to `sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)`
2. Add `rec.PrefixLength > 0` guard to the condition

### File: `sim/workload/generator.go`

**Purpose:** Set `req.PrefixLength` after prefix prepend.

Changes:
1. After the prefix tokens are prepended to `req.InputTokens`, set `req.PrefixLength = len(prefix)`

### File: `cmd/observe_cmd.go`

**Purpose:** Store workload seed in trace header.

Changes:
1. Add `WorkloadSeed: spec.Seed` to the `TraceHeader{}` construction (guarded by `spec != nil`)

### File: `cmd/root.go`

**Purpose:** Store workload seed in trace header for `blis run` command.

Changes:
1. Add `WorkloadSeed` to the `TraceHeader{}` construction at the trace export point
