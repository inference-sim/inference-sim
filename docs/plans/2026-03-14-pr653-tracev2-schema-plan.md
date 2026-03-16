# TraceV2 Schema Extension: Model, DeadlineUs, ServerInputTokens

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three missing fields to the TraceV2 CSV schema so that replayed traces correctly carry model identity, per-request timeouts, and server-reported token counts.

**The problem today:** When `blis observe` records a real serving run and you later replay the trace, each request loses its model name, its deadline (so timeouts never fire during replay), and the server's actual tokenization count (making calibration mismatches invisible). These three gaps break the observe/replay/calibrate loop that is the foundation of PR #652.

**What this PR adds:**
1. **`Model` column in TraceV2 CSV** — the model name string (e.g., `meta-llama/Llama-3.1-8B-Instruct`) stored per record; replayed requests carry this model identity into the simulator.
2. **`DeadlineUs` column in TraceV2 CSV** — per-request absolute timeout in microseconds; replayed requests have `sim.Request.Deadline` set so session cancellation fires correctly.
3. **`ServerInputTokens` column in TraceV2 CSV** — server-reported `prompt_tokens` from the real inference response; stored in the trace for calibration analysis (not propagated to `sim.Request`).

**Why this matters:** Without these fields the observe/replay/calibrate loop (PR #652) produces replays that lose model routing, ignore client timeouts, and silently miss tokenization drift. This PR is the schema foundation all other sub-issues in #652 depend on.

**Architecture:** All changes are confined to `sim/workload/tracev2.go` (struct + CSV export/import) and `sim/workload/replay.go` (sim.Request assignment). The three new columns are inserted after `reason_ratio` and before `arrival_time_us`, shifting all timing column indices by +3. This is an intentional breaking schema change — there are no production traces in the wild.

**Source:** GitHub issue #653

**Closes:** Fixes #653

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR extends `TraceRecord` (the in-memory representation of one row in a TraceV2 CSV file) with three fields, and extends `traceV2Columns`, `ExportTraceV2`, and `parseTraceRecord` to serialize/deserialize them. `LoadTraceV2Requests` (the replay path) is updated to propagate `Model` and `Deadline` onto `sim.Request`; `ServerInputTokens` stays in `TraceRecord` only (consumed by calibration, not simulation).

**System position:** `TraceV2` sits between `blis observe` (which writes traces) and `blis replay`/`blis calibrate` (which read them). This PR changes the on-disk schema and the in-memory struct. Downstream consumers: `LoadTraceV2Requests` (replay) and `PrepareCalibrationPairs` (calibration). Upstream producer: `cmd/observe.go:RecordRequest`. The new fields in `cmd/observe.go` will be zero-valued until a follow-up PR wires them to the HTTP response — acceptable because zero-value semantics are documented ("no timeout", "default model", "not recorded").

**Adjacent blocks:** `sim.Request` (gains `Model` and `Deadline` from replay), `calibrate.go` `PrepareCalibrationPairs` (gains `ServerInputTokens` for future use), `cmd/observe.go` `RecordRequest` (construction site; not changed in this PR — new fields zero-valued).

**DEVIATION flags:** None — issue description matches current codebase exactly.

### B) Behavioral Contracts

**BC-1: Schema round-trip — new fields survive export→load**
- GIVEN a `TraceRecord` with `Model="test-model"`, `DeadlineUs=5000000`, `ServerInputTokens=512`
- WHEN `ExportTraceV2` writes it and `LoadTraceV2` reads it back
- THEN the loaded record has identical values for all three fields
- MECHANISM: fields added to `traceV2Columns`, `ExportTraceV2` row, and `parseTraceRecord` in matching positions

**BC-2: Existing fields unaffected by new columns**
- GIVEN a `TraceRecord` with all 22 original fields populated with non-zero values
- WHEN exported and loaded with the new 25-column format
- THEN all 22 original fields round-trip with identical values
- MECHANISM: new columns inserted between position 14 and 15 (0-indexed); all original field indices ≥15 shift by +3 in `parseTraceRecord` and `ExportTraceV2`

**BC-3: Replay propagates Model to sim.Request**
- GIVEN a trace record with `Model="meta-llama/Llama-3.1-8B-Instruct"`
- WHEN `LoadTraceV2Requests` processes it
- THEN the resulting `sim.Request.Model` equals `"meta-llama/Llama-3.1-8B-Instruct"`
- MECHANISM: `req.Model = rec.Model` added; comment about omission removed

**BC-4: Replay propagates DeadlineUs to sim.Request.Deadline**
- GIVEN a trace record with `DeadlineUs=7500000`
- WHEN `LoadTraceV2Requests` processes it
- THEN the resulting `sim.Request.Deadline` equals `7500000`
- MECHANISM: `req.Deadline = rec.DeadlineUs` added
- NOTE: `DeadlineUs` stores an **absolute microsecond timestamp** using the same time origin as `ArrivalTimeUs` (both relative to trace start, not wall-clock epoch). This PR only defines the schema field and passes whatever value is in the trace file directly to `req.Deadline`. The observe-side normalization (converting wall-clock deadline to trace-relative deadline by subtracting the trace start time) is the responsibility of the follow-up PR that wires `cmd/observe.go:RecordRequest` — not this PR. For generated/synthetic traces, `DeadlineUs=0` means no timeout (BC-5).

**BC-5: Zero DeadlineUs → no timeout**
- GIVEN a trace record with `DeadlineUs=0`
- WHEN `LoadTraceV2Requests` processes it
- THEN `sim.Request.Deadline` is `0` (simulator interprets as "no timeout")
- MECHANISM: zero-value pass-through; no special handling needed

**BC-6: Zero Model → default model (empty string)**
- GIVEN a trace record with `Model=""` (zero value)
- WHEN `LoadTraceV2Requests` processes it
- THEN `sim.Request.Model` is `""` (simulator interprets as default model)
- MECHANISM: zero-value pass-through; no special handling needed

**BC-7: ServerInputTokens not set on sim.Request**
- GIVEN a trace record with `ServerInputTokens=600`
- WHEN `LoadTraceV2Requests` processes it
- THEN no field on the returned `sim.Request` holds the value 600 from `ServerInputTokens`
- MECHANISM: `ServerInputTokens` is a calibration-only field; intentionally not assigned
- NOTE: `ServerInputTokens` enables comparing the server's actual tokenization count against the client-specified count (`rec.ServerInputTokens != rec.InputTokens`), detecting real tokenization drift (e.g., chat template overhead, special tokens, multimodal expansion). The existing `calibrate.go` comparison (`rec.InputTokens != sr.InputTokens`) is a different kind of check: it compares client-specified token count against what BLIS actually simulated, which can differ when prefix groups add synthetic tokens (`sr.InputTokens = prefix + body > rec.InputTokens`). Both comparisons are useful; `ServerInputTokens` specifically enables server-vs-client drift detection. Wiring `PrepareCalibrationPairs` to use `ServerInputTokens` when non-zero is deferred to a follow-up PR (#652).

**BC-8: Old-format CSV (22 columns) rejected with clear error**
- GIVEN a CSV file with 22 columns (old format, missing the 3 new columns)
- WHEN `LoadTraceV2` attempts to parse it
- THEN an error is returned that states the column count mismatch
- MECHANISM: `len(row) < len(traceV2Columns)` check (now expects 25); error message shows actual vs expected

**BC-9: Invalid DeadlineUs parses as error**
- GIVEN a CSV row where the `deadline_us` column contains a non-integer value
- WHEN `parseTraceRecord` processes that row
- THEN it returns an error mentioning `deadline_us`
- MECHANISM: `strconv.ParseInt` with error check that wraps the field name

**BC-10: Invalid ServerInputTokens parses as error**
- GIVEN a CSV row where `server_input_tokens` contains a non-integer value
- WHEN `parseTraceRecord` processes that row
- THEN it returns an error mentioning `server_input_tokens`
- MECHANISM: `strconv.Atoi` with error check that wraps the field name

### C) Component Interaction

```
cmd/observe.go (RecordRequest)
    │  produces TraceRecord{} literals
    ▼
sim/workload/tracev2.go (ExportTraceV2)
    │  writes 25-column CSV
    ▼
[disk: header.yaml + data.csv]
    │
    ▼
sim/workload/tracev2.go (LoadTraceV2 → parseTraceRecord)
    │  reconstructs TraceRecord with all 25 fields
    ├──▶ sim/workload/replay.go (LoadTraceV2Requests)
    │        sets sim.Request.Model, .Deadline
    │        does NOT set sim.Request.ServerInputTokens (no such field)
    └──▶ sim/workload/calibrate.go (PrepareCalibrationPairs)
             has rec.ServerInputTokens available for token mismatch detection
```

**API change:** `TraceRecord` struct gains 3 fields. `traceV2Columns` slice grows from 22 to 25 entries. This is the only public API surface change.

**State changes:** None — `TraceRecord` is a pure data struct; `traceV2Columns` is a package-level var (unexported by convention but used within package). No global state mutation.

**Extension friction:** Adding another field to `TraceRecord` requires touching 4 locations: struct definition, `traceV2Columns`, `ExportTraceV2` row builder, `parseTraceRecord` (including shifting subsequent indices). Plus any construction sites. This is inherent to the CSV schema design — acceptable for the trace format's relatively low change frequency.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Add fields after `reason_ratio` and before `arrival_time_us`" | Same | No deviation |
| "`parseTraceRecord`: update column index offsets for fields that follow" | Updates indices 15→18, 16→19, 17→20, 18→21, 19→22, 20→23, 21→24 | Direct implementation of stated intent |
| "Update minimum column count check at line 182: `len(traceV2Columns)` will automatically reflect the new count" | Check uses `len(traceV2Columns)` already — no code change needed, just the slice grows | Verification finding: implementation already uses len() indirectly |
| "`status`/`error_message` defensive `len(row) > 20/21` checks" | These shift to direct access (row[23], row[24]) since the column count check above guarantees the row is long enough | CORRECTION: defensive checks are no longer needed once we enforce exact column count; simplifying improves clarity. **This intentionally removes backward-compatible tolerance for rows with fewer than 25 columns** — all rows now require exactly 25 columns (enforced by `len(row) < len(traceV2Columns)` guard). Old 22-column CSV files are rejected via BC-8. |
| "Update `sim/workload/tracev2_test.go`: Update the existing round-trip test" | New test `TestTraceV2_RoundTrip_NewFields` added alongside the existing test | ADDITION: keeping existing test untouched preserves regression coverage; adding a focused test for the new fields is cleaner |
| "`TestParseTraceRecord_InvalidInteger_ReturnsError` uses `row := make([]string, 22)`" | Row size updated to 25 | CORRECTION: must match new column count or the column-count check fires first |

### E) Review Guide

**The tricky part:** The index shift in `parseTraceRecord`. Columns 15–21 (0-indexed) all shift to 18–24. A single off-by-one in any of the 7 shifted indices silently misassigns fields. The round-trip test BC-1/BC-2 catches this — if any index is wrong, the loaded value will be mismatched.

**What to scrutinize:** The `parseTraceRecord` function — verify each index 15 through 24 maps to the correct column name by cross-checking against `traceV2Columns`. Also verify `ExportTraceV2` row slice order matches `traceV2Columns` order exactly.

**What's safe to skim:** `LoadTraceV2Requests` changes (2 lines + comment removal). The new test methods (mechanical struct setup + round-trip or assertion).

**Known debt:** `cmd/observe.go:RecordRequest` construction site does not yet populate `Model`, `DeadlineUs`, or `ServerInputTokens` — these will be zero-valued until PR #652 sub-issues wire the HTTP response fields. This is intentional and documented.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Change |
|------|--------|
| `sim/workload/tracev2.go` | Add 3 fields to `TraceRecord`; add 3 columns to `traceV2Columns`; add 3 rows to `ExportTraceV2`; rewrite `parseTraceRecord` with new indices + 5 validation guards (DeadlineUs<0, InputTokens<0, OutputTokens<0, ServerInputTokens<0, DeadlineUs<ArrivalTimeUs) |
| `sim/workload/replay.go` | Set `req.Model` and `req.Deadline` from record; remove stale omission comment |
| `sim/workload/tracev2_test.go` | Add `TestTraceV2_RoundTrip_NewFields`; update `TestParseTraceRecord_InvalidInteger_ReturnsError` row size; add 6 new error tests |
| `sim/workload/replay_test.go` | Add `TestLoadTraceV2Requests_ModelAndDeadline` |
| `cmd/observe.go` | Add TODO comment at `RecordRequest` construction site noting 3 zero-valued fields deferred to follow-up PR (#652) |

**Key restructuring:** Tasks 1 and 2 from the original plan are **merged into a single Task 1**. The reason: `TraceRecord` struct fields, `traceV2Columns`, `ExportTraceV2`, and `parseTraceRecord` form an atomically-consistent serialization contract. Committing struct+export changes without the matching parser update breaks ALL round-trip tests simultaneously (export writes 25 columns; parser reads row[15] as `arrival_time_us` which is now the `model` string → `strconv.ParseInt` error). Splitting was a TDD anti-pattern here.

**Key decisions:**
- New columns inserted at positions 15–17 (after `reason_ratio`, before `arrival_time_us`) — matches issue spec exactly
- `parseTraceRecord` defensive `len(row) > N` checks for `status`/`error_message` simplified to direct index access since the column-count guard above ensures exact length
- `cmd/observe.go` construction site NOT updated in this PR (new fields zero-valued is correct behavior for the observe path until #652 wires the response fields)

**No dead code:** All 3 new `TraceRecord` fields are exercised: `Model` and `DeadlineUs` by `LoadTraceV2Requests`; `ServerInputTokens` by the round-trip test (export→load verifies it survives the CSV).

### G) Task Breakdown

---

### Task 1: Extend TraceRecord struct, CSV schema, and parser (atomic)

**Contracts Implemented:** BC-1, BC-2, BC-8, BC-9, BC-10

**Files:**
- Modify: `sim/workload/tracev2.go` (struct + columns + export + parseTraceRecord)
- Modify: `sim/workload/tracev2_test.go` (new tests + fix row size)

**Why all in one task:** `TraceRecord` struct, `traceV2Columns`, `ExportTraceV2`, and `parseTraceRecord` form an atomically-consistent serialization contract. Committing struct+export without a matching parser update breaks ALL round-trip tests simultaneously: the exporter writes 25 columns but the parser reads row[15] as `arrival_time_us` when it is now the `model` string, causing `strconv.ParseInt` errors on every load.

**Step 1: Write failing tests for new fields and new error paths**

In `sim/workload/tracev2_test.go`, add after `TestTraceV2_RoundTrip_WithServerConfig`:

```go
// TestTraceV2_RoundTrip_NewFields verifies BC-1 and BC-2: all three new
// schema fields survive export → load with correct values.
func TestTraceV2_RoundTrip_NewFields(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "real"}
	records := []TraceRecord{
		{
			RequestID:         0,
			Model:             "meta-llama/Llama-3.1-8B-Instruct",
			DeadlineUs:        5000000,
			ServerInputTokens: 512,
			InputTokens:       256,
			OutputTokens:      64,
			ArrivalTimeUs:     1000,
			SendTimeUs:        1010,
			FirstChunkTimeUs:  1800,
			LastChunkTimeUs:   2500,
			Status:            "ok",
		},
		{
			RequestID:         1,
			Model:             "",   // zero value: default model
			DeadlineUs:        0,    // zero value: no timeout
			ServerInputTokens: 0,   // zero value: not recorded
			InputTokens:       128,
			OutputTokens:      32,
			ArrivalTimeUs:     5000,
			Status:            "ok",
		},
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
	if len(loaded.Records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(loaded.Records))
	}

	r0 := loaded.Records[0]
	// BC-1: new fields round-trip with non-zero values
	if r0.Model != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("record 0 model = %q, want %q", r0.Model, "meta-llama/Llama-3.1-8B-Instruct")
	}
	if r0.DeadlineUs != 5000000 {
		t.Errorf("record 0 deadline_us = %d, want 5000000", r0.DeadlineUs)
	}
	if r0.ServerInputTokens != 512 {
		t.Errorf("record 0 server_input_tokens = %d, want 512", r0.ServerInputTokens)
	}
	// BC-2: existing timing fields unaffected by index shift
	if r0.ArrivalTimeUs != 1000 {
		t.Errorf("record 0 arrival_time_us = %d, want 1000", r0.ArrivalTimeUs)
	}
	if r0.SendTimeUs != 1010 {
		t.Errorf("record 0 send_time_us = %d, want 1010", r0.SendTimeUs)
	}
	if r0.FirstChunkTimeUs != 1800 {
		t.Errorf("record 0 first_chunk_time_us = %d, want 1800", r0.FirstChunkTimeUs)
	}
	if r0.InputTokens != 256 {
		t.Errorf("record 0 input_tokens = %d, want 256", r0.InputTokens)
	}

	r1 := loaded.Records[1]
	// BC-5, BC-6: zero values round-trip correctly
	if r1.Model != "" {
		t.Errorf("record 1 model = %q, want empty", r1.Model)
	}
	if r1.DeadlineUs != 0 {
		t.Errorf("record 1 deadline_us = %d, want 0", r1.DeadlineUs)
	}
	if r1.ServerInputTokens != 0 {
		t.Errorf("record 1 server_input_tokens = %d, want 0", r1.ServerInputTokens)
	}
}
```

Also add these three tests after `TestParseTraceRecord_InvalidInteger_ReturnsError`:

```go
// TestParseTraceRecord_InvalidDeadlineUs_ReturnsError verifies BC-9.
func TestParseTraceRecord_InvalidDeadlineUs_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[16] = "not_a_number" // deadline_us column

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for non-numeric deadline_us, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}

// TestParseTraceRecord_InvalidServerInputTokens_ReturnsError verifies BC-10.
func TestParseTraceRecord_InvalidServerInputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "not_a_number" // server_input_tokens column

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for non-numeric server_input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "server_input_tokens") {
		t.Errorf("error should mention 'server_input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeDeadlineUs_ReturnsError verifies that negative
// deadline_us is rejected (R3: validate numeric parameters).
func TestParseTraceRecord_NegativeDeadlineUs_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[16] = "-1" // negative deadline_us

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for negative deadline_us, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeInputTokens_ReturnsError verifies that negative
// input_tokens is rejected (R3: prevents make([]int, negative) panic in replay).
func TestParseTraceRecord_NegativeInputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[8] = "-1" // input_tokens column

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for negative input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "input_tokens") {
		t.Errorf("error should mention 'input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeOutputTokens_ReturnsError verifies that negative
// output_tokens is rejected (R3: prevents make([]int, negative) panic in replay).
func TestParseTraceRecord_NegativeOutputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[9] = "-1" // output_tokens column

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for negative output_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "output_tokens") {
		t.Errorf("error should mention 'output_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeServerInputTokens_ReturnsError verifies R3 for
// server_input_tokens (consistent with InputTokens/OutputTokens/DeadlineUs validation).
func TestParseTraceRecord_NegativeServerInputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "-1" // server_input_tokens column

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for negative server_input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "server_input_tokens") {
		t.Errorf("error should mention 'server_input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError verifies cross-field
// validation: deadline_us must not precede arrival_time_us when both are nonzero.
func TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError(t *testing.T) {
	row := make([]string, 25)
	for i := range row {
		row[i] = "0"
	}
	row[16] = "1000"   // deadline_us = 1000
	row[18] = "5000"   // arrival_time_us = 5000 (deadline < arrival)

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for deadline before arrival, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}
```

Also update `TestParseTraceRecord_InvalidInteger_ReturnsError` — change `make([]string, 22)` to `make([]string, 25)` and fill with `"0"`:

```go
func TestParseTraceRecord_InvalidInteger_ReturnsError(t *testing.T) {
	// GIVEN a row with a non-numeric request_id
	row := make([]string, 25) // must match new column count
	row[0] = "abc"             // request_id should be integer
	for i := 1; i < len(row); i++ {
		row[i] = "0"
	}

	// WHEN parsing
	_, err := parseTraceRecord(row)

	// THEN error about invalid value
	if err == nil {
		t.Fatal("expected error for non-numeric request_id, got nil")
	}
	if !strings.Contains(err.Error(), "request_id") {
		t.Errorf("error should mention 'request_id', got: %s", err.Error())
	}
}
```

**Step 2: Run tests to verify they fail (compile error)**

```bash
cd .worktrees/pr653-tracev2-schema && go test ./sim/workload/... -run "TestTraceV2_RoundTrip_NewFields|TestParseTraceRecord" -v
```
Expected: FAIL with compile error — `TraceRecord` has no `Model`, `DeadlineUs`, `ServerInputTokens` fields.

**Step 3: Implement all schema changes in tracev2.go**

**3a. Add fields to TraceRecord struct** (after `ReasonRatio`, before `ArrivalTimeUs`):

```go
// TraceRecord represents one row in a trace v2 CSV.
type TraceRecord struct {
	RequestID         int
	ClientID          string
	TenantID          string
	SLOClass          string
	SessionID         string
	RoundIndex        int
	PrefixGroup       string
	Streaming         bool
	InputTokens       int
	OutputTokens      int
	TextTokens        int
	ImageTokens       int
	AudioTokens       int
	VideoTokens       int
	ReasonRatio       float64
	Model             string // model name (e.g., "meta-llama/Llama-3.1-8B-Instruct"); empty = default model
	DeadlineUs        int64  // absolute timeout in microseconds (same time origin as ArrivalTimeUs); 0 = no timeout
	ServerInputTokens int    // server-reported prompt_tokens; 0 = not recorded (e.g., generated traces)
	ArrivalTimeUs     int64
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	Status            string // "ok", "error", "timeout"
	ErrorMessage      string
}
```

**3b. Update traceV2Columns** (add 3 entries after `"reason_ratio"`, before `"arrival_time_us"`):

```go
var traceV2Columns = []string{
	"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
	"prefix_group", "streaming", "input_tokens", "output_tokens",
	"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
	"model", "deadline_us", "server_input_tokens",
	"arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
	"num_chunks", "status", "error_message",
}
```

**3c. Update ExportTraceV2 row builder** (add 3 entries after `FormatFloat(r.ReasonRatio...)`, before `FormatInt(r.ArrivalTimeUs...)`):

```go
for _, r := range records {
	row := []string{
		strconv.Itoa(r.RequestID),
		r.ClientID,
		r.TenantID,
		r.SLOClass,
		r.SessionID,
		strconv.Itoa(r.RoundIndex),
		r.PrefixGroup,
		strconv.FormatBool(r.Streaming),
		strconv.Itoa(r.InputTokens),
		strconv.Itoa(r.OutputTokens),
		strconv.Itoa(r.TextTokens),
		strconv.Itoa(r.ImageTokens),
		strconv.Itoa(r.AudioTokens),
		strconv.Itoa(r.VideoTokens),
		strconv.FormatFloat(r.ReasonRatio, 'f', -1, 64),
		r.Model,
		strconv.FormatInt(r.DeadlineUs, 10),
		strconv.Itoa(r.ServerInputTokens),
		strconv.FormatInt(r.ArrivalTimeUs, 10),   // integer format
		strconv.FormatInt(r.SendTimeUs, 10),       // integer format
		strconv.FormatInt(r.FirstChunkTimeUs, 10), // integer format
		strconv.FormatInt(r.LastChunkTimeUs, 10),  // integer format
		strconv.Itoa(r.NumChunks),
		r.Status,
		r.ErrorMessage,
	}
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("writing CSV row %d: %w", r.RequestID, err)
	}
}
```

**3d. Rewrite parseTraceRecord with correct indices and DeadlineUs validation**

Replace the entire `parseTraceRecord` function:

```go
func parseTraceRecord(row []string) (*TraceRecord, error) {
	requestID, err := strconv.Atoi(row[0])
	if err != nil {
		return nil, fmt.Errorf("parsing request_id %q: %w", row[0], err)
	}
	roundIndex, err := strconv.Atoi(row[5])
	if err != nil {
		return nil, fmt.Errorf("parsing round_index %q: %w", row[5], err)
	}
	streaming, err := strconv.ParseBool(row[7])
	if err != nil {
		return nil, fmt.Errorf("parsing streaming %q: %w", row[7], err)
	}
	inputTokens, err := strconv.Atoi(row[8])
	if err != nil {
		return nil, fmt.Errorf("parsing input_tokens %q: %w", row[8], err)
	}
	// NOTE: This negative-value check is NEW — the pre-existing code had no validation here.
	// Negative input_tokens would cause make([]int, negative) panic in LoadTraceV2Requests.
	if inputTokens < 0 {
		return nil, fmt.Errorf("parsing input_tokens: negative value %d not allowed", inputTokens)
	}
	outputTokens, err := strconv.Atoi(row[9])
	if err != nil {
		return nil, fmt.Errorf("parsing output_tokens %q: %w", row[9], err)
	}
	// NOTE: Same as inputTokens — new negative-value check closing a pre-existing panic vector.
	if outputTokens < 0 {
		return nil, fmt.Errorf("parsing output_tokens: negative value %d not allowed", outputTokens)
	}
	textTokens, err := strconv.Atoi(row[10])
	if err != nil {
		return nil, fmt.Errorf("parsing text_tokens %q: %w", row[10], err)
	}
	imageTokens, err := strconv.Atoi(row[11])
	if err != nil {
		return nil, fmt.Errorf("parsing image_tokens %q: %w", row[11], err)
	}
	audioTokens, err := strconv.Atoi(row[12])
	if err != nil {
		return nil, fmt.Errorf("parsing audio_tokens %q: %w", row[12], err)
	}
	videoTokens, err := strconv.Atoi(row[13])
	if err != nil {
		return nil, fmt.Errorf("parsing video_tokens %q: %w", row[13], err)
	}
	reasonRatio, err := strconv.ParseFloat(row[14], 64)
	if err != nil {
		return nil, fmt.Errorf("parsing reason_ratio %q: %w", row[14], err)
	}
	// row[15] = model (string, no parsing needed)
	deadlineUs, err := strconv.ParseInt(row[16], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing deadline_us %q: %w", row[16], err)
	}
	if deadlineUs < 0 {
		return nil, fmt.Errorf("parsing deadline_us: negative value %d not allowed (use 0 for no timeout)", deadlineUs)
	}
	serverInputTokens, err := strconv.Atoi(row[17])
	if err != nil {
		return nil, fmt.Errorf("parsing server_input_tokens %q: %w", row[17], err)
	}
	if serverInputTokens < 0 {
		return nil, fmt.Errorf("parsing server_input_tokens: negative value %d not allowed", serverInputTokens)
	}
	// Timing columns shifted +3 from original positions (were 15-21, now 18-24)
	arrivalTimeUs, err := strconv.ParseInt(row[18], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing arrival_time_us %q: %w", row[18], err)
	}
	sendTimeUs, err := strconv.ParseInt(row[19], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing send_time_us %q: %w", row[19], err)
	}
	firstChunkTimeUs, err := strconv.ParseInt(row[20], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing first_chunk_time_us %q: %w", row[20], err)
	}
	lastChunkTimeUs, err := strconv.ParseInt(row[21], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing last_chunk_time_us %q: %w", row[21], err)
	}
	numChunks, err := strconv.Atoi(row[22])
	if err != nil {
		return nil, fmt.Errorf("parsing num_chunks %q: %w", row[22], err)
	}
	// Cross-field invariant: deadline must not precede arrival (would cause immediate
	// silent timeout in simulator). Zero deadline means "no timeout" — exempt from check.
	if deadlineUs > 0 && arrivalTimeUs > 0 && deadlineUs < arrivalTimeUs {
		return nil, fmt.Errorf("parsing deadline_us: value %d precedes arrival_time_us %d (corrupt trace?)", deadlineUs, arrivalTimeUs)
	}
	// row[23] = status, row[24] = error_message
	// (column-count guard in LoadTraceV2 ensures these indices always exist)
	return &TraceRecord{
		RequestID:         requestID,
		ClientID:          row[1],
		TenantID:          row[2],
		SLOClass:          row[3],
		SessionID:         row[4],
		RoundIndex:        roundIndex,
		PrefixGroup:       row[6],
		Streaming:         streaming,
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		TextTokens:        textTokens,
		ImageTokens:       imageTokens,
		AudioTokens:       audioTokens,
		VideoTokens:       videoTokens,
		ReasonRatio:       reasonRatio,
		Model:             row[15],
		DeadlineUs:        deadlineUs,
		ServerInputTokens: serverInputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        sendTimeUs,
		FirstChunkTimeUs:  firstChunkTimeUs,
		LastChunkTimeUs:   lastChunkTimeUs,
		NumChunks:         numChunks,
		Status:            row[23],
		ErrorMessage:      strings.TrimSpace(row[24]),
	}, nil
}
```

**Step 3e: Add TODO comment to cmd/observe.go:RecordRequest**

In `cmd/observe.go`, find the `RecordRequest` function's `workload.TraceRecord{}` struct literal. Add the following comment at the TOP of the struct literal, before `RequestID`:

```go
// TODO(#652): populate Model, DeadlineUs, ServerInputTokens from HTTP response
// fields once the observe-side wiring lands. Until then these are zero-valued:
//   Model=""      → simulator treats as default model
//   DeadlineUs=0  → no timeout
//   ServerInputTokens=0 → not recorded
```

Do NOT change any other fields or logic in this function.

**Step 4: Run all workload tests to verify they pass**

```bash
cd .worktrees/pr653-tracev2-schema && go test ./sim/workload/... -v 2>&1 | tail -40
```
Expected: All tests PASS — `TestTraceV2_RoundTrip_NewFields`, all seven new `TestParseTraceRecord_*` tests (`InvalidDeadlineUs`, `InvalidServerInputTokens`, `NegativeDeadlineUs`, `NegativeInputTokens`, `NegativeOutputTokens`, `NegativeServerInputTokens`, `DeadlineBeforeArrival`), and all pre-existing tests.

**Step 5: Run lint**

```bash
cd .worktrees/pr653-tracev2-schema && golangci-lint run ./sim/workload/... && golangci-lint run ./cmd/...
```
Expected: No new issues.

**Step 6: Commit**

```bash
cd .worktrees/pr653-tracev2-schema
git add sim/workload/tracev2.go sim/workload/tracev2_test.go
git commit -m "feat(workload): extend TraceRecord with Model, DeadlineUs, ServerInputTokens (BC-1–10)

- Add Model (string), DeadlineUs (int64), ServerInputTokens (int) to TraceRecord
- Add model, deadline_us, server_input_tokens to traceV2Columns (positions 15-17)
  shifting timing columns to positions 18-24
- Update ExportTraceV2 row builder (25 columns)
- Rewrite parseTraceRecord with new indices and validations (R3):
  * deadline_us < 0 rejected
  * server_input_tokens < 0 rejected (consistent R3 validation for all token counts)
  * input_tokens < 0 rejected (prevents make([]int, negative) panic in replay)
  * output_tokens < 0 rejected (same reason)
  * deadline_us < arrival_time_us rejected when both nonzero (cross-field invariant)
- Add TODO(#652) comment to cmd/observe.go:RecordRequest construction site
- Add TestTraceV2_RoundTrip_NewFields (BC-1, BC-2, BC-5, BC-6)
- Add TestParseTraceRecord_Invalid{DeadlineUs,ServerInputTokens} (BC-9, BC-10)
- Add TestParseTraceRecord_Negative{DeadlineUs,ServerInputTokens,InputTokens} (R3)
- Add TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError (cross-field invariant)
- Fix TestParseTraceRecord_InvalidInteger_ReturnsError: row size 22 → 25

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Update replay to propagate Model and Deadline

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-6, BC-7

**Files:**
- Modify: `sim/workload/replay.go`
- Modify: `sim/workload/replay_test.go`

**Step 1: Write failing test for Model and Deadline propagation**

In `sim/workload/replay_test.go`, add after `TestLoadTraceV2Requests_PrefixGroup_SharedTokens`:

```go
// TestLoadTraceV2Requests_ModelAndDeadline verifies BC-3, BC-4, BC-5, BC-6, BC-7.
func TestLoadTraceV2Requests_ModelAndDeadline(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "real"}
	records := []TraceRecord{
		{
			RequestID:         0,
			Model:             "meta-llama/Llama-3.1-8B-Instruct",
			DeadlineUs:        7500000,
			ServerInputTokens: 300, // must NOT appear on sim.Request
			InputTokens:       100,
			OutputTokens:      50,
			ArrivalTimeUs:     0,
			Status:            "ok",
		},
		{
			RequestID:         1,
			Model:             "",  // BC-6: empty = default model
			DeadlineUs:        0,   // BC-5: no timeout
			ServerInputTokens: 0,
			InputTokens:       50,
			OutputTokens:      25,
			ArrivalTimeUs:     1000,
			Status:            "ok",
		},
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
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	// BC-3: Model propagated
	if requests[0].Model != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("request 0 Model = %q, want %q", requests[0].Model, "meta-llama/Llama-3.1-8B-Instruct")
	}
	// BC-4: Deadline propagated
	if requests[0].Deadline != 7500000 {
		t.Errorf("request 0 Deadline = %d, want 7500000", requests[0].Deadline)
	}
	// BC-6: empty Model propagated as-is
	if requests[1].Model != "" {
		t.Errorf("request 1 Model = %q, want empty", requests[1].Model)
	}
	// BC-5: zero Deadline propagated as-is (no timeout)
	if requests[1].Deadline != 0 {
		t.Errorf("request 1 Deadline = %d, want 0", requests[1].Deadline)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/pr653-tracev2-schema && go test ./sim/workload/... -run TestLoadTraceV2Requests_ModelAndDeadline -v
```
Expected: FAIL — `requests[0].Model` is `""` and `requests[0].Deadline` is `0` (not yet set in replay.go).

**Step 3: Update LoadTraceV2Requests in replay.go**

Replace the `req := &sim.Request{...}` literal in `sim/workload/replay.go`. Change the comment and add the two new field assignments:

```go
req := &sim.Request{
    ID:               fmt.Sprintf("request_%d", rec.RequestID),
    ArrivalTime:      rec.ArrivalTimeUs,
    InputTokens:      inputTokens,
    OutputTokens:     outputTokens,
    MaxOutputLen:     len(outputTokens),
    State:            sim.StateQueued,
    ScheduledStepIdx: 0,
    FinishedStepIdx:  0,
    TenantID:         rec.TenantID,
    SLOClass:         rec.SLOClass,
    SessionID:        rec.SessionID,
    RoundIndex:       rec.RoundIndex,
    TextTokenCount:   rec.TextTokens,
    ImageTokenCount:  rec.ImageTokens,
    AudioTokenCount:  rec.AudioTokens,
    VideoTokenCount:  rec.VideoTokens,
    ReasonRatio:      rec.ReasonRatio,
    Model:            rec.Model,    // BC-3, BC-6: model identity from trace; empty = default model
    Deadline:         rec.DeadlineUs, // BC-4, BC-5: client timeout; 0 = no timeout
    // ServerInputTokens: not propagated to sim.Request (calibration-only field, BC-7)
}
```

**Step 4: Run all workload tests to verify they pass**

```bash
cd .worktrees/pr653-tracev2-schema && go test ./sim/workload/... -v 2>&1 | tail -20
```
Expected: All tests PASS.

**Step 5: Run full test suite and lint**

```bash
cd .worktrees/pr653-tracev2-schema && go test ./... 2>&1 | tail -10
cd .worktrees/pr653-tracev2-schema && golangci-lint run ./... 2>&1 | tail -10
```
Expected: All tests pass; zero new lint issues.

**Step 6: Commit**

```bash
cd .worktrees/pr653-tracev2-schema
git add sim/workload/replay.go sim/workload/replay_test.go
git commit -m "feat(workload): propagate Model and Deadline from trace to sim.Request (BC-3, BC-4, BC-5, BC-6, BC-7)

- Set req.Model = rec.Model (replaces 'omitted' comment)
- Set req.Deadline = rec.DeadlineUs (was unset; zero = no timeout)
- ServerInputTokens intentionally not propagated (calibration-only, BC-7)
- Add TestLoadTraceV2Requests_ModelAndDeadline covering BC-3 through BC-7

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | 1 | Unit/Round-trip | `TestTraceV2_RoundTrip_NewFields` |
| BC-2 | 1 | Unit/Round-trip | `TestTraceV2_RoundTrip_NewFields` (timing field assertions) |
| BC-3 | 2 | Unit | `TestLoadTraceV2Requests_ModelAndDeadline` |
| BC-4 | 2 | Unit | `TestLoadTraceV2Requests_ModelAndDeadline` |
| BC-5 | 2 | Unit | `TestLoadTraceV2Requests_ModelAndDeadline` (r1.Deadline == 0) |
| BC-6 | 2 | Unit | `TestLoadTraceV2Requests_ModelAndDeadline` (r1.Model == "") |
| BC-7 | 2 | Unit | Implicit: test confirms Model/Deadline set; ServerInputTokens is not a `sim.Request` field |
| BC-8 | 1 | Failure | `TestLoadTraceV2_UnknownYAMLField_ReturnsError` (pre-existing); `TestTraceV2_RoundTrip_NewFields` (loads successfully with 25 cols) |
| BC-9 | 1 | Failure | `TestParseTraceRecord_InvalidDeadlineUs_ReturnsError` |
| BC-10 | 1 | Failure | `TestParseTraceRecord_InvalidServerInputTokens_ReturnsError` |
| R3 (DeadlineUs≥0) | 1 | Failure | `TestParseTraceRecord_NegativeDeadlineUs_ReturnsError` |
| R3 (InputTokens≥0) | 1 | Failure | `TestParseTraceRecord_NegativeInputTokens_ReturnsError` |
| R3 (OutputTokens≥0) | 1 | Failure | `TestParseTraceRecord_NegativeOutputTokens_ReturnsError` |
| R3 (ServerInputTokens≥0) | 1 | Failure | `TestParseTraceRecord_NegativeServerInputTokens_ReturnsError` |
| Cross-field (DeadlineUs≥ArrivalTimeUs) | 1 | Failure | `TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError` |

**Golden dataset:** This PR does not change `blis run` simulation output or the JSON results schema. No golden dataset update needed.

**No invariant tests needed:** This PR does not touch request lifecycle, KV cache, or metrics pipeline — the rule in Phase 6 about invariant tests alongside golden tests does not apply. The round-trip test directly verifies the serialization law (data in = data out).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Index off-by-one in `parseTraceRecord` silently misassigns a field | Medium | High | Round-trip test BC-1/BC-2 asserts specific values for every shifted field | Task 1 |
| `cmd/observe.go` construction site produces 0-valued fields, confusing future consumers | Low | Low | Documented in Section E (Review Guide) as known debt; zero-value semantics are defined in struct comments | Not in scope |
| Breaking change silently accepts old-format traces with wrong column count | Low | Medium | `len(row) < len(traceV2Columns)` check triggers error (BC-8) | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — 3 fields added, 2 lines in replay.go, 1 test per new behavior
- [x] No feature creep — `cmd/observe.go` and `calibrate.go` behavior unchanged
- [x] No unexercised code — all 3 fields exercised by round-trip test + replay test
- [x] No partial implementations — all 4 touch points updated together (struct, columns, export, parse)
- [x] No breaking changes without explicit contract updates — BC-8 explicitly contracts the old-format rejection
- [x] No hidden global state impact — `traceV2Columns` is a package-level var used only within `sim/workload/`
- [x] All new code will pass golangci-lint — no new external imports; standard strconv patterns
- [x] Shared test helpers used — no new shared infrastructure needed; tests use `t.TempDir()` and existing ExportTraceV2/LoadTraceV2
- [x] CLAUDE.md updated if needed — no new files/packages; file organization unchanged; no new CLI flags
- [x] No stale references — no "planned for PR N" comments to add
- [x] Documentation DRY — no canonical standards files modified
- [x] Deviation log reviewed — all deviations documented in Section D
- [x] Each task produces working, testable code — Task 1 (struct+schema+parser, all tests pass atomically) → Task 2 (replay, all tests pass)
- [x] Task dependencies correctly ordered — Task 1 must complete before Task 2 (Task 2 references `TraceRecord.Model` and `.DeadlineUs` created in Task 1)
- [x] All contracts mapped to tasks — confirmed in test strategy table
- [x] Golden dataset regeneration not needed — no simulation output changed
- [x] Construction site audit — see R4 item above; `parseTraceRecord` updated in Task 1; `cmd/observe.go:RecordRequest` documented known debt

**Antipattern rules:**
- [x] R1: No silent data loss — all new parse errors returned with field name context
- [x] R2: No map iteration → not applicable
- [x] R3: All new and affected numeric fields validated in `parseTraceRecord`. Note: `InputTokens` and `OutputTokens` are **pre-existing fields** that had NO negative-value validation before this PR — these are **new validations being added** here because (a) this PR rewrites `parseTraceRecord` in full, and (b) negative values would cause `make([]int, negative)` panics in `LoadTraceV2Requests`. Reviewers should NOT assume these checks already exist in the current code:
  - `DeadlineUs < 0` → error (new field, new validation)
  - `DeadlineUs > 0 && DeadlineUs < ArrivalTimeUs` → error (cross-field invariant, prevents silent immediate-timeout on corrupt traces)
  - `InputTokens < 0` → error (**pre-existing field, NEW validation** — closes panic vector: negative → `make([]int, negative)` in replay)
  - `OutputTokens < 0` → error (**pre-existing field, NEW validation** — same reason)
  - `ServerInputTokens < 0` → error (new field, new validation — consistent R3 for all token count fields)
- [x] R4: Construction sites fully audited — all 5 sites enumerated:
  1. `sim/workload/tracev2.go:parseTraceRecord` — updated (Task 1)
  2. `cmd/observe.go:RecordRequest` — NOT updated; new fields zero-valued until follow-up PR (#652); documented as known debt
  3. `sim/workload/calibrate_test.go` — ~5 literal sites; use named fields; zero values for new fields are correct for test data; no code change needed
  4. `sim/workload/tracev2_test.go` — existing literal sites; same rationale
  5. `sim/workload/replay_test.go` — existing literal sites; same rationale
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in sim/ — not added
- [x] R7: No golden test added — round-trip test is a law test (data in = data out)
- [x] R8: No exported maps
- [x] R9: No new YAML fields — struct fields are CSV columns, not YAML
- [x] R10: YAML parsing unchanged — TraceHeader uses `KnownFields(true)` already
- [x] R11: No new divisions
- [x] R12: Golden dataset not changed
- [x] R13–R23: Not applicable to this schema-extension change

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/tracev2.go`

**Purpose:** Defines TraceRecord struct, CSV schema, export, and import for trace v2 format.

**Changes:**

1. **TraceRecord struct** — add 3 fields after `ReasonRatio`, before `ArrivalTimeUs`:
```go
Model             string // model name (e.g., "meta-llama/Llama-3.1-8B-Instruct"); empty = default model
DeadlineUs        int64  // per-request timeout deadline in microseconds; 0 = no timeout
ServerInputTokens int    // server-reported prompt_tokens; 0 = not recorded (e.g., generated traces)
```

2. **traceV2Columns** — add `"model"`, `"deadline_us"`, `"server_input_tokens"` after `"reason_ratio"` and before `"arrival_time_us"`. Slice grows from 22 to 25 entries.

3. **ExportTraceV2** — add 3 entries to the row slice after `FormatFloat(r.ReasonRatio...)` and before `FormatInt(r.ArrivalTimeUs...)`:
```go
r.Model,
strconv.FormatInt(r.DeadlineUs, 10),
strconv.Itoa(r.ServerInputTokens),
```

4. **parseTraceRecord** — complete rewrite with new indices (see Task 1 Step 3d above). Key index mapping:
   - row[15] → `Model` (string, direct)
   - row[16] → `DeadlineUs` (ParseInt + `>= 0` validation)
   - row[17] → `ServerInputTokens` (Atoi)
   - row[18] → `ArrivalTimeUs` (was 15)
   - row[19] → `SendTimeUs` (was 16)
   - row[20] → `FirstChunkTimeUs` (was 17)
   - row[21] → `LastChunkTimeUs` (was 18)
   - row[22] → `NumChunks` (was 19)
   - row[23] → `Status` (was 20, defensive check removed)
   - row[24] → `ErrorMessage` (was 21, defensive check removed)

**Error handling:** All parse errors return `fmt.Errorf("parsing %s %q: %w", fieldName, value, err)` — consistent with pre-existing pattern.

---

### File: `cmd/observe.go`

**Purpose:** Real-mode HTTP client that records requests into TraceRecord.

**Changes:** In `RecordRequest`, add a TODO comment at the top of the `workload.TraceRecord{}` struct literal body, before the first field:
```go
// TODO(#652): populate Model, DeadlineUs, ServerInputTokens from HTTP response
// fields once the observe-side wiring lands. Until then these are zero-valued:
//   Model=""      → simulator treats as default model
//   DeadlineUs=0  → no timeout
//   ServerInputTokens=0 → not recorded
```
No other changes to this file.

---

### File: `sim/workload/replay.go`

**Purpose:** Converts `TraceV2` records into `sim.Request` objects for simulation replay.

**Changes:** In `LoadTraceV2Requests`, inside the `for _, rec := range trace.Records` loop, update the `req := &sim.Request{...}` literal:
- Replace `// Model: omitted — TraceRecord predates Model field; zero-value = default model (BC-5)` comment with: `Model: rec.Model,`
- Add: `Deadline: rec.DeadlineUs,`
- Add comment after Deadline: `// ServerInputTokens: not propagated to sim.Request (calibration-only field, BC-7)`

---

### File: `sim/workload/tracev2_test.go`

**Changes:**
1. Add `TestTraceV2_RoundTrip_NewFields` (complete code in Task 1 Step 1)
2. Update `TestParseTraceRecord_InvalidInteger_ReturnsError`: change `make([]string, 22)` to `make([]string, 25)` (complete replacement in Task 1 Step 1)
3. Add `TestParseTraceRecord_InvalidDeadlineUs_ReturnsError` (complete code in Task 1 Step 1)
4. Add `TestParseTraceRecord_InvalidServerInputTokens_ReturnsError` (complete code in Task 1 Step 1)
5. Add `TestParseTraceRecord_NegativeDeadlineUs_ReturnsError` (complete code in Task 1 Step 1)
6. Add `TestParseTraceRecord_NegativeInputTokens_ReturnsError` (complete code in Task 1 Step 1)
7. Add `TestParseTraceRecord_NegativeOutputTokens_ReturnsError` (complete code in Task 1 Step 1)
8. Add `TestParseTraceRecord_NegativeServerInputTokens_ReturnsError` (complete code in Task 1 Step 1)
9. Add `TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError` (complete code in Task 1 Step 1)

---

### File: `sim/workload/replay_test.go`

**Changes:**
1. Add `TestLoadTraceV2Requests_ModelAndDeadline` (complete code in Task 2 Step 1)
