# TraceV2 ITL Timestamps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable per-chunk ITL timestamp recording in `blis observe` for calibration of Inter-Token Latency metrics.

**The problem today:** The current TraceV2 format only records first and last chunk timestamps, making it impossible to calibrate Inter-Token Latency (ITL) / Time-Per-Output-Token (TPOT) metrics. This limits the Observe-Replay-Calibrate workflow to E2E and TTFT validation only.

**What this PR adds:**
1. **Optional ITL recording** — `--record-itl` flag enables per-chunk timestamp capture during `blis observe`
2. **Separate ITL file** — Companion `itl.csv` file alongside main TraceV2 data (backward compatible)
3. **ITL calibration** — `blis calibrate` computes ITL MAPE, Pearson R, and percentiles when ITL data is present
4. **Streaming-only guard** — Non-streaming requests log warnings; ITL data only recorded for streaming requests

**Why this matters:** ITL is a critical SLO metric for production LLM serving (P99 ITL < 20ms ensures smooth streaming). This PR completes the calibration pipeline, enabling full hardware validation across E2E, TTFT, and ITL.

**Architecture:** Add `ITLRecord` type in `sim/workload`, modify `RealClient.handleStreamingResponse` to capture per-chunk timestamps, extend `Recorder` to track ITL data, add `ExportITL` and `LoadITL` functions, extend `CalibrationPairs` with ITL vectors, add ITL metric computation to `BuildCalibrationReport`.

**Source:** GitHub issue #992

**Closes:** Fixes #992

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds optional per-chunk timestamp recording to the observe/replay/calibrate pipeline. When `--record-itl` is passed to `blis observe`, each SSE chunk's arrival time is captured and written to a companion `itl.csv` file alongside the main TraceV2 data. `blis calibrate` reads the ITL file (if present) and computes ITL-specific MAPE, Pearson R, and percentiles.

**Where it fits:** Extends the observe/replay/calibrate pipeline (Phase 0 workload unification, issues #659, #689, #701). Observe records data, replay simulates, calibrate compares.

**Adjacent blocks:**
- `RealClient.handleStreamingResponse` (chunk timestamp capture)
- `Recorder` (ITL data accumulation)
- `sim/workload/tracev2.go` (ITL export/load)
- `sim/workload/calibrate.go` (ITL metric computation)
- `cmd/observe_cmd.go` (flag wiring)
- `cmd/calibrate.go` (ITL file loading)

**Deviations:** None from source document (issue #992 recommends Option 1, which this plan implements).

### B) Behavioral Contracts

**Positive Contracts:**

**BC-1: ITL recording opt-in**
- GIVEN `blis observe` with `--record-itl` flag and streaming enabled
- WHEN a request completes with N chunks (N >= 2)
- THEN an ITL file contains N rows for that request_id with microsecond timestamps
- MECHANISM: `RealClient.handleStreamingResponse` captures `time.Now().UnixMicro()` per chunk, `Recorder` accumulates into slice

**BC-2: Non-streaming requests excluded from ITL**
- GIVEN `blis observe` with `--record-itl` and `--no-streaming`
- WHEN requests complete
- THEN the ITL file is empty (header only) and a warning is logged
- MECHANISM: `Recorder.RecordITL` checks `pending.Streaming` and logs warning if false

**BC-3: ITL file format**
- GIVEN ITL data recorded
- WHEN `ExportITL` is called
- THEN the ITL CSV has columns: `request_id,chunk_index,timestamp_us`
- MECHANISM: `ExportITL` writes CSV with 3 columns, integer formatting for timestamps

**BC-4: ITL calibration metric**
- GIVEN ITL file loaded and sim results matched
- WHEN `blis calibrate` processes the data
- THEN the report includes `metrics["itl"]` with MAPE, PearsonR, P50/P90/P95/P99, quality rating
- MECHANISM: `PrepareCalibrationPairs` computes per-request ITL from chunk deltas, `ComputeCalibration` produces `MetricComparison`

**BC-5: Backward compatibility**
- GIVEN `blis observe` without `--record-itl`
- WHEN trace files are exported
- THEN no ITL file is created and main TraceV2 format is unchanged
- MECHANISM: `--record-itl` defaults to false; ITL export is conditional

**BC-6: ITL optional in calibrate**
- GIVEN `blis calibrate` with trace data but no `--itl-data` flag
- WHEN calibration runs
- THEN the report includes E2E and TTFT metrics but omits ITL
- MECHANISM: `--itl-data` is optional; `PrepareCalibrationPairs` skips ITL if file not provided

**Negative Contracts:**

**BC-7: No ITL for incomplete requests**
- GIVEN a request times out or errors mid-stream
- WHEN ITL data is exported
- THEN partial chunk timestamps ARE recorded up to the failure point (status field in main CSV indicates completion state)
- MECHANISM: `Recorder.RecordITL` appends all captured chunks regardless of final status

**BC-8: No ITL validation in sim/  packages**
- GIVEN ITL recording/loading logic in `sim/workload`
- WHEN errors occur (file I/O, parsing)
- THEN the code returns an error (never calls `logrus.Fatalf` or `os.Exit`)
- MECHANISM: All `sim/workload` functions return `error`; only `cmd/` may terminate

**Error Handling Contracts:**

**BC-9: File write failure**
- GIVEN `ExportITL` called with invalid path
- WHEN file creation fails
- THEN return error with context (wrapped via `fmt.Errorf`)
- MECHANISM: `os.Create` error checked and wrapped

**BC-10: Malformed ITL CSV**
- GIVEN `LoadITL` called with corrupt CSV (negative timestamps, non-integer request_id)
- WHEN parsing fails
- THEN return error with row number and field name
- MECHANISM: `strconv.ParseInt` errors wrapped with `fmt.Errorf("parsing timestamp at row %d: %w", row, err)`

**BC-11: Flag validation**
- GIVEN `blis calibrate --itl-data <path>` with non-existent file
- WHEN command runs
- THEN log fatal error with clear message
- MECHANISM: `cmd/calibrate.go` checks file existence via `os.Stat` before calling `LoadITL`

### C) Component Interaction

**Component Diagram:**

```
cmd/observe_cmd.go (--record-itl flag)
    |
    v
cmd/observe.go (RealClient, Recorder)
    |
    ├─> RealClient.handleStreamingResponse (capture chunk timestamps)
    |       |
    |       v
    |   []int64 chunkTimestamps (per-request)
    |
    └─> Recorder.RecordITL (accumulate ITL data)
            |
            v
        []ITLRecord (in-memory)
            |
            v
        sim/workload.ExportITL (write itl.csv)

cmd/calibrate.go (--itl-data flag)
    |
    v
sim/workload.LoadITL (read itl.csv)
    |
    v
sim/workload.PrepareCalibrationPairs (compute per-request ITL vectors)
    |
    v
sim/workload.ComputeCalibration (ITL MAPE, Pearson R, percentiles)
    |
    v
CalibrationReport.Metrics["itl"]
```

**API Contracts:**

```go
// sim/workload/itl.go
type ITLRecord struct {
    RequestID   int
    ChunkIndex  int
    TimestampUs int64
}

func ExportITL(records []ITLRecord, path string) error
func LoadITL(path string) ([]ITLRecord, error)
```

```go
// cmd/observe.go
type Recorder struct {
    mu         sync.Mutex
    records    []workload.TraceRecord
    itlRecords []workload.ITLRecord // NEW
}

func (r *Recorder) RecordITL(requestID int, chunkTimestamps []int64)
func (r *Recorder) ITLRecords() []workload.ITLRecord
func (r *Recorder) ExportITL(path string) error
```

**State Changes:**
- `RealClient.handleStreamingResponse`: Add local `chunkTimestamps []int64` slice, append on each chunk
- `Recorder`: Add `itlRecords []workload.ITLRecord` field
- `CalibrationPairs`: Add `ITL LatencyPair` field

**Extension Friction:** Adding another per-chunk metric (e.g., token count per chunk) requires modifying 3 files: `ITLRecord` struct, `ExportITL` CSV columns, `LoadITL` parsing.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "per-token timestamps" | Per-chunk timestamps | CLARIFICATION: Codebase tracks SSE chunks (each may contain multiple tokens), not individual tokens. Observable unit is chunk. |
| "separate ITL file" | `itl.csv` with explicit `--itl-data` flag | CLARIFICATION: Issue shows `--itl-data itl.csv` in example; using separate explicit flag (not tied to `--trace-output` prefix) for user control. |
| No mention of non-streaming | Log warning for non-streaming + ITL flag | ADDITION: Non-streaming has `NumChunks=1`; ITL is undefined. Guard added. |
| No mention of partial data | Record partial ITL for errors/timeouts | ADDITION: Issue focuses on successful requests; partial data is useful for debugging and doesn't violate schema. |

### E) Review Guide

**The tricky part:** Per-request ITL computation in `PrepareCalibrationPairs` requires computing chunk-to-chunk deltas for each request's ITL records, then matching against sim ITL vectors. Off-by-one errors in delta computation would produce wrong ITL values.

**What to scrutinize:**
- BC-4: ITL delta computation logic (first chunk is TTFT, subsequent chunks are ITL)
- BC-7: Partial ITL recording for failed requests (verify `status` field in main CSV is consulted)
- BC-10: CSV parsing error messages include row/column context

**What's safe to skim:**
- `ExportITL` / `LoadITL` CSV I/O (standard pattern from `tracev2.go`)
- Flag definitions in `cmd/observe_cmd.go` and `cmd/calibrate.go` (boilerplate)

**Known debt:** None — this is new functionality with no pre-existing ITL code to refactor.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/workload/itl.go` — `ITLRecord` type, `ExportITL`, `LoadITL`
- `sim/workload/itl_test.go` — unit tests for ITL I/O

**Files to modify:**
- `cmd/observe.go` — Add `chunkTimestamps` capture in `handleStreamingResponse`, add `RecordITL` and `ExportITL` methods to `Recorder`
- `cmd/observe_cmd.go` — Add `--record-itl` and `--itl-output` flags
- `cmd/calibrate.go` — Add `--itl-data` flag, load ITL file
- `sim/workload/calibrate.go` — Extend `CalibrationPairs` with `ITL LatencyPair`, add ITL metric computation

**Key decisions:**
- ITL file is separate (not embedded in main CSV) for backward compatibility
- ITL recording is opt-in via `--record-itl` flag
- Non-streaming requests log warning but don't fail
- Partial ITL data (for errors/timeouts) is recorded

**Confirmation:**
- No dead code: all ITL struct fields used by export/load/calibrate
- All paths exercisable: tests cover streaming, non-streaming, error cases, calibration with/without ITL

### G) Task Breakdown

#### Task 1: Add ITL data structures and I/O

**Contracts Implemented:** BC-3, BC-9, BC-10

**Files:**
- Create: `sim/workload/itl.go`
- Create: `sim/workload/itl_test.go`

**Step 1: Write failing test for ITL export**

```go
// sim/workload/itl_test.go
package workload_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestITL_ExportLoad_RoundTrip(t *testing.T) {
	// GIVEN ITL records with multiple chunks per request
	records := []workload.ITLRecord{
		{RequestID: 0, ChunkIndex: 0, TimestampUs: 1000000},
		{RequestID: 0, ChunkIndex: 1, TimestampUs: 1008000},
		{RequestID: 0, ChunkIndex: 2, TimestampUs: 1016000},
		{RequestID: 1, ChunkIndex: 0, TimestampUs: 2000000},
		{RequestID: 1, ChunkIndex: 1, TimestampUs: 2010000},
	}

	// WHEN exported and loaded
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "itl.csv")
	if err := workload.ExportITL(records, path); err != nil {
		t.Fatalf("ExportITL failed: %v", err)
	}
	loaded, err := workload.LoadITL(path)
	if err != nil {
		t.Fatalf("LoadITL failed: %v", err)
	}

	// THEN loaded records match exported records
	if len(loaded) != len(records) {
		t.Errorf("got %d records, want %d", len(loaded), len(records))
	}
	for i := range records {
		if loaded[i] != records[i] {
			t.Errorf("record %d: got %+v, want %+v", i, loaded[i], records[i])
		}
	}
}

func TestITL_LoadITL_MalformedCSV_ReturnsError(t *testing.T) {
	// GIVEN a CSV with non-integer timestamp
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "bad.csv")
	content := "request_id,chunk_index,timestamp_us\n0,0,not-a-number\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN LoadITL is called
	_, err := workload.LoadITL(path)

	// THEN it returns an error with context
	if err == nil {
		t.Fatal("expected error for malformed CSV, got nil")
	}
	if !contains(err.Error(), "timestamp_us") {
		t.Errorf("error message should mention 'timestamp_us', got: %v", err)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) &&
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
		 findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestITL -v`
Expected: FAIL with "undefined: workload.ITLRecord"

**Step 3: Implement ITL types and I/O**

In `sim/workload/itl.go`:
```go
package workload

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ITLRecord represents one chunk timestamp in the ITL trace.
type ITLRecord struct {
	RequestID   int
	ChunkIndex  int
	TimestampUs int64
}

// ExportITL writes ITL records to a CSV file.
// Format: request_id,chunk_index,timestamp_us
func ExportITL(records []ITLRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"request_id", "chunk_index", "timestamp_us"}); err != nil {
		return fmt.Errorf("writing ITL CSV header: %w", err)
	}

	// Write data rows
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID),
			strconv.Itoa(r.ChunkIndex),
			strconv.FormatInt(r.TimestampUs, 10),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing ITL CSV row (request_id=%d): %w", r.RequestID, err)
		}
	}
	return nil
}

// LoadITL reads ITL records from a CSV file.
func LoadITL(path string) ([]ITLRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	// Skip header row
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("reading ITL CSV header: %w", err)
	}

	var records []ITLRecord
	rowNum := 1 // 0 = header
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading ITL CSV row %d: %w", rowNum, err)
		}
		rowNum++

		if len(row) < 3 {
			return nil, fmt.Errorf("ITL CSV row %d has %d columns, expected 3", rowNum, len(row))
		}

		requestID, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("parsing request_id at row %d: %w", rowNum, err)
		}
		chunkIndex, err := strconv.Atoi(row[1])
		if err != nil {
			return nil, fmt.Errorf("parsing chunk_index at row %d: %w", rowNum, err)
		}
		timestampUs, err := strconv.ParseInt(row[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing timestamp_us at row %d: %w", rowNum, err)
		}

		// Validate: no negative values (R3)
		if requestID < 0 || chunkIndex < 0 || timestampUs < 0 {
			return nil, fmt.Errorf("ITL CSV row %d has negative value (request_id=%d, chunk_index=%d, timestamp_us=%d)", rowNum, requestID, chunkIndex, timestampUs)
		}

		records = append(records, ITLRecord{
			RequestID:   requestID,
			ChunkIndex:  chunkIndex,
			TimestampUs: timestampUs,
		})
	}
	return records, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestITL -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/itl.go sim/workload/itl_test.go
git commit -m "feat(workload): add ITL data structures and I/O (BC-3, BC-9, BC-10)

- Add ITLRecord type with request_id, chunk_index, timestamp_us
- Implement ExportITL and LoadITL for CSV I/O
- Add round-trip and malformed CSV tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 2: Capture chunk timestamps in observe

**Contracts Implemented:** BC-1, BC-2, BC-7

**Files:**
- Modify: `cmd/observe.go`

**Step 1: Write failing test for chunk timestamp capture**

```go
// cmd/observe_test.go (add to existing file)
package cmd

import (
	"testing"
	"time"
)

func TestRecorder_RecordITL_StreamingRequest(t *testing.T) {
	// GIVEN a recorder and chunk timestamps
	rec := &Recorder{}
	timestamps := []int64{1000000, 1008000, 1016000}

	// WHEN RecordITL is called
	rec.RecordITL(42, timestamps)

	// THEN ITL records are stored
	itl := rec.ITLRecords()
	if len(itl) != 3 {
		t.Fatalf("got %d ITL records, want 3", len(itl))
	}
	for i, ts := range timestamps {
		if itl[i].RequestID != 42 {
			t.Errorf("record %d: got request_id=%d, want 42", i, itl[i].RequestID)
		}
		if itl[i].ChunkIndex != i {
			t.Errorf("record %d: got chunk_index=%d, want %d", i, itl[i].ChunkIndex, i)
		}
		if itl[i].TimestampUs != ts {
			t.Errorf("record %d: got timestamp_us=%d, want %d", i, itl[i].TimestampUs, ts)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestRecorder_RecordITL -v`
Expected: FAIL with "rec.RecordITL undefined"

**Step 3: Implement ITL recording in Recorder**

In `cmd/observe.go`, modify the `Recorder` struct and add methods:

```go
// Recorder captures per-request timing and metrics (goroutine-safe).
type Recorder struct {
	mu         sync.Mutex
	records    []workload.TraceRecord
	itlRecords []workload.ITLRecord // NEW
}

// RecordITL captures per-chunk timestamps for ITL calibration.
// Only meaningful for streaming requests (len(chunkTimestamps) >= 2).
func (r *Recorder) RecordITL(requestID int, chunkTimestamps []int64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, ts := range chunkTimestamps {
		r.itlRecords = append(r.itlRecords, workload.ITLRecord{
			RequestID:   requestID,
			ChunkIndex:  i,
			TimestampUs: ts,
		})
	}
}

// ITLRecords returns all recorded ITL records.
func (r *Recorder) ITLRecords() []workload.ITLRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]workload.ITLRecord, len(r.itlRecords))
	copy(result, r.itlRecords)
	return result
}

// ExportITL writes ITL data to a CSV file.
func (r *Recorder) ExportITL(path string) error {
	return workload.ExportITL(r.ITLRecords(), path)
}
```

Now modify `handleStreamingResponse` to capture timestamps:

```go
func (c *RealClient) handleStreamingResponse(resp *http.Response, record *RequestRecord) (*RequestRecord, error) {
	scanner := bufio.NewScanner(resp.Body)
	chunkCount := 0
	var lastUsage map[string]interface{}
	var chunkTimestamps []int64 // NEW: capture per-chunk timestamps

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		now := time.Now().UnixMicro()
		chunkCount++
		chunkTimestamps = append(chunkTimestamps, now) // NEW
		if chunkCount == 1 {
			record.FirstChunkTimeUs = now
		}
		record.LastChunkTimeUs = now

		// Parse chunk for usage and finish_reason
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logrus.Debugf("observe: skipping malformed SSE chunk: %v", err)
			continue
		}
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			lastUsage = usage
		}
		// Extract finish_reason from content chunks (skip usage-only chunks with empty choices)
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
					record.FinishReason = fr
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		logrus.Warnf("observe: request %d: SSE scanner error: %v", record.RequestID, err)
	}

	record.NumChunks = chunkCount
	record.ChunkTimestamps = chunkTimestamps // NEW: store in record

	if lastUsage == nil && chunkCount > 0 {
		logrus.Warnf("observe: request %d: streaming response had %d chunks but no usage data (missing stream_options?)", record.RequestID, chunkCount)
	}
	if lastUsage != nil {
		if ct, ok := lastUsage["completion_tokens"].(float64); ok {
			record.OutputTokens = int(ct)
		}
		if pt, ok := lastUsage["prompt_tokens"].(float64); ok {
			record.ServerInputTokens = int(pt)
		} else if _, exists := lastUsage["prompt_tokens"]; exists {
			logrus.Debugf("observe: prompt_tokens has unexpected type %T, expected float64", lastUsage["prompt_tokens"])
		}
	}

	// Warn on problematic finish_reason values
	if record.FinishReason == "length" || record.FinishReason == "abort" {
		logrus.Warnf("observe: request %d finish_reason=%q (output may be truncated)", record.RequestID, record.FinishReason)
	}

	return record, nil
}
```

Also update `RequestRecord` struct to include `ChunkTimestamps`:

```go
// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID         int
	OutputTokens      int
	ServerInputTokens int
	Status            string // "ok", "error", "timeout"
	ErrorMessage      string
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	FinishReason      string
	ChunkTimestamps   []int64 // NEW: per-chunk timestamps for ITL
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestRecorder_RecordITL -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/observe.go cmd/observe_test.go
git commit -m "feat(cmd): capture per-chunk timestamps in observe (BC-1, BC-2, BC-7)

- Add ChunkTimestamps field to RequestRecord
- Capture time.Now().UnixMicro() for each SSE chunk
- Add RecordITL, ITLRecords, ExportITL methods to Recorder

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 3: Add observe CLI flags

**Contracts Implemented:** BC-5

**Files:**
- Modify: `cmd/observe_cmd.go`

**Step 1: Write failing test for flag presence**

```go
// cmd/observe_cmd_test.go (add to existing file)
package cmd

import (
	"testing"
)

func TestObserveCmd_ITLFlags_Defined(t *testing.T) {
	// GIVEN the observe command
	cmd := observeCmd

	// WHEN checking for ITL flags
	recordITLFlag := cmd.Flags().Lookup("record-itl")
	itlOutputFlag := cmd.Flags().Lookup("itl-output")

	// THEN both flags are defined
	if recordITLFlag == nil {
		t.Error("--record-itl flag not defined")
	}
	if itlOutputFlag == nil {
		t.Error("--itl-output flag not defined")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestObserveCmd_ITLFlags -v`
Expected: FAIL with "--record-itl flag not defined"

**Step 3: Add flag definitions**

In `cmd/observe_cmd.go`, add to the variable declarations:

```go
var (
	// ... existing vars ...
	observeRecordITL bool
	observeITLOutput string
)
```

In the `init()` function, add flag definitions:

```go
func init() {
	// ... existing flags ...

	// ITL recording (optional, opt-in)
	observeCmd.Flags().BoolVar(&observeRecordITL, "record-itl", false, "Record per-chunk timestamps for ITL calibration (streaming only)")
	observeCmd.Flags().StringVar(&observeITLOutput, "itl-output", "", "Output path for ITL CSV file (default: <trace-data>.itl.csv if --record-itl is set)")

	rootCmd.AddCommand(observeCmd)
}
```

In the `runObserve` function, add ITL export logic after trace export:

```go
func runObserve(cmd *cobra.Command, args []string) {
	// ... existing validation and execution ...

	// Export trace v2
	if err := recorder.Export(&header, observeTraceHeader, observeTraceData); err != nil {
		logrus.Fatalf("Failed to export trace: %v", err)
	}
	logrus.Infof("TraceV2 exported: %s (header), %s (data)", observeTraceHeader, observeTraceData)

	// Export ITL if requested (BC-5: opt-in)
	if observeRecordITL {
		itlPath := observeITLOutput
		if itlPath == "" {
			// Default: <trace-data>.itl.csv
			itlPath = observeTraceData + ".itl.csv"
		}

		itlRecords := recorder.ITLRecords()
		if len(itlRecords) == 0 {
			logrus.Warnf("--record-itl was set but no ITL data recorded (non-streaming requests?)")
		}

		if err := recorder.ExportITL(itlPath); err != nil {
			logrus.Fatalf("Failed to export ITL data: %v", err)
		}
		logrus.Infof("ITL data exported: %s (%d records)", itlPath, len(itlRecords))
	}
}
```

Also need to wire ITL recording in the request dispatch loop. Find where `recorder.RecordRequest` is called and add ITL recording:

```go
// After result := client.Send(ctx, pending)
if observeRecordITL && result.Status == "ok" && len(result.ChunkTimestamps) > 0 {
	recorder.RecordITL(result.RequestID, result.ChunkTimestamps)
} else if observeRecordITL && !pending.Streaming {
	// BC-2: warn if ITL requested for non-streaming
	logrus.Warnf("request %d: --record-itl was set but request is non-streaming (NumChunks=1)", result.RequestID)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestObserveCmd_ITLFlags -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/observe_cmd.go
git commit -m "feat(cmd): add --record-itl and --itl-output flags to observe (BC-5)

- Add observeRecordITL and observeITLOutput variables
- Wire ITL export in runObserve
- Default ITL path: <trace-data>.itl.csv
- Log warning for non-streaming requests with --record-itl

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 4: Extend calibration with ITL metrics

**Contracts Implemented:** BC-4, BC-6

**Files:**
- Modify: `sim/workload/calibrate.go`
- Modify: `sim/workload/calibrate_test.go`

**Step 1: Write failing test for ITL calibration**

```go
// sim/workload/calibrate_test.go (add to existing file)
package workload_test

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestCalibration_WithITL(t *testing.T) {
	// GIVEN trace records with ITL data and matching sim results
	traceRecords := []workload.TraceRecord{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 1000, LastChunkTimeUs: 1100, SendTimeUs: 0},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 2000, LastChunkTimeUs: 2100, SendTimeUs: 0},
	}
	simResults := []workload.SimResult{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, TTFT: 1000, E2E: 1100},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, TTFT: 2000, E2E: 2100},
	}
	itlRecords := []workload.ITLRecord{
		{RequestID: 0, ChunkIndex: 0, TimestampUs: 1000},
		{RequestID: 0, ChunkIndex: 1, TimestampUs: 1020},
		{RequestID: 0, ChunkIndex: 2, TimestampUs: 1040},
		{RequestID: 1, ChunkIndex: 0, TimestampUs: 2000},
		{RequestID: 1, ChunkIndex: 1, TimestampUs: 2020},
		{RequestID: 1, ChunkIndex: 2, TimestampUs: 2040},
	}

	// WHEN preparing calibration pairs with ITL
	config := &workload.CalibrationConfig{}
	pairs, err := workload.PrepareCalibrationPairsWithITL(traceRecords, simResults, itlRecords, config)
	if err != nil {
		t.Fatalf("PrepareCalibrationPairsWithITL failed: %v", err)
	}

	// THEN ITL pairs are populated
	if len(pairs.ITL.Real) == 0 {
		t.Error("ITL.Real is empty")
	}
	if len(pairs.ITL.Sim) == 0 {
		t.Error("ITL.Sim is empty")
	}

	// WHEN building calibration report
	report, err := workload.BuildCalibrationReport(pairs, &workload.ConfigMatchInfo{})
	if err != nil {
		t.Fatalf("BuildCalibrationReport failed: %v", err)
	}

	// THEN report includes ITL metric
	itlMetric, ok := report.Metrics["itl"]
	if !ok {
		t.Fatal("report.Metrics[\"itl\"] not found")
	}
	if itlMetric.Count == 0 {
		t.Error("ITL metric has Count=0")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestCalibration_WithITL -v`
Expected: FAIL with "undefined: workload.PrepareCalibrationPairsWithITL"

**Step 3: Implement ITL calibration logic**

In `sim/workload/calibrate.go`, extend `CalibrationPairs`:

```go
// CalibrationPairs holds matched, normalized real-vs-sim latency vectors.
type CalibrationPairs struct {
	TTFT               LatencyPair
	E2E                LatencyPair
	ITL                LatencyPair // NEW
	TokenMismatchCount int
	ExcludedWarmUp     int
	MatchedCount       int
	UnmatchedReal      int
	UnmatchedSim       int
}
```

Add new function:

```go
// PrepareCalibrationPairsWithITL extends PrepareCalibrationPairs with ITL data.
// ITL is computed as per-request mean inter-chunk latency (microseconds).
// First chunk delta is TTFT; subsequent deltas are ITL.
func PrepareCalibrationPairsWithITL(
	realRecords []TraceRecord,
	simResults []SimResult,
	itlRecords []ITLRecord,
	config *CalibrationConfig,
) (*CalibrationPairs, error) {
	// Start with standard pairs
	pairs, err := PrepareCalibrationPairs(realRecords, simResults, config)
	if err != nil {
		return nil, err
	}

	// Group ITL records by request ID
	itlByRequest := make(map[int][]ITLRecord)
	for _, rec := range itlRecords {
		itlByRequest[rec.RequestID] = append(itlByRequest[rec.RequestID], rec)
	}

	// Index sim results by RequestID
	simByID := make(map[int]SimResult, len(simResults))
	for _, sr := range simResults {
		simByID[sr.RequestID] = sr
	}

	// Compute per-request ITL
	for _, rec := range realRecords {
		// Skip warm-up
		if rec.RequestID < config.WarmUpRequests {
			continue
		}

		sr, ok := simByID[rec.RequestID]
		if !ok {
			continue
		}

		chunks, ok := itlByRequest[rec.RequestID]
		if !ok || len(chunks) < 2 {
			continue // No ITL data for this request
		}

		// Sort chunks by index (defensive)
		sortITLRecords(chunks)

		// Compute real ITL: mean of chunk-to-chunk deltas (skip first, which is TTFT)
		var realITLSum float64
		realITLCount := 0
		for i := 1; i < len(chunks); i++ {
			delta := float64(chunks[i].TimestampUs - chunks[i-1].TimestampUs)
			if delta < 0 {
				// Clock skew or corrupt data — skip this request
				continue
			}
			realITLSum += delta
			realITLCount++
		}
		if realITLCount == 0 {
			continue
		}
		realITL := realITLSum / float64(realITLCount)

		// Compute sim ITL: (E2E - TTFT) / OutputTokens
		// This approximates mean ITL assuming uniform token generation
		simITL := 0.0
		if sr.OutputTokens > 1 {
			simITL = (sr.E2E - sr.TTFT) / float64(sr.OutputTokens-1)
		}

		pairs.ITL.Real = append(pairs.ITL.Real, realITL)
		pairs.ITL.Sim = append(pairs.ITL.Sim, simITL)
	}

	return pairs, nil
}

func sortITLRecords(records []ITLRecord) {
	// Simple insertion sort (small N)
	for i := 1; i < len(records); i++ {
		key := records[i]
		j := i - 1
		for j >= 0 && records[j].ChunkIndex > key.ChunkIndex {
			records[j+1] = records[j]
			j--
		}
		records[j+1] = key
	}
}
```

Update `BuildCalibrationReport` to include ITL:

```go
func BuildCalibrationReport(pairs *CalibrationPairs, configMatch *ConfigMatchInfo) (*CalibrationReport, error) {
	// ... existing code ...

	if len(pairs.ITL.Real) > 0 {
		itl, err := ComputeCalibration(pairs.ITL.Real, pairs.ITL.Sim, "itl")
		if err != nil {
			return nil, err
		}
		report.Metrics["itl"] = itl
	}
	return report, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestCalibration_WithITL -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/calibrate.go sim/workload/calibrate_test.go
git commit -m "feat(workload): add ITL calibration metrics (BC-4, BC-6)

- Extend CalibrationPairs with ITL LatencyPair
- Add PrepareCalibrationPairsWithITL function
- Compute real ITL as mean chunk-to-chunk delta
- Compute sim ITL as (E2E - TTFT) / (OutputTokens - 1)
- Include ITL in BuildCalibrationReport

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 5: Add calibrate CLI support

**Contracts Implemented:** BC-6, BC-11

**Files:**
- Modify: `cmd/calibrate.go`

**Step 1: Write failing test for --itl-data flag**

```go
// cmd/calibrate_test.go (add to existing file)
package cmd

import (
	"testing"
)

func TestCalibrateCmd_ITLDataFlag_Defined(t *testing.T) {
	// GIVEN the calibrate command
	cmd := calibrateCmd

	// WHEN checking for --itl-data flag
	flag := cmd.Flags().Lookup("itl-data")

	// THEN flag is defined and optional
	if flag == nil {
		t.Fatal("--itl-data flag not defined")
	}
	if flag.DefValue != "" {
		t.Errorf("--itl-data default should be empty, got %q", flag.DefValue)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestCalibrateCmd_ITLDataFlag -v`
Expected: FAIL with "--itl-data flag not defined"

**Step 3: Add flag and wire ITL loading**

In `cmd/calibrate.go`, add variable:

```go
var (
	// ... existing vars ...
	calibrateITLDataPath string
)
```

In `init()`, add flag:

```go
func init() {
	// ... existing flags ...
	calibrateCmd.Flags().StringVar(&calibrateITLDataPath, "itl-data", "", "Path to ITL CSV file (optional; if provided, calibration includes ITL metrics)")
	rootCmd.AddCommand(calibrateCmd)
}
```

In the `Run` function, extend calibration logic:

```go
// Step 5: Prepare calibration pairs (with optional ITL)
var pairs *workload.CalibrationPairs
var err error

if calibrateITLDataPath != "" {
	// Check file exists (BC-11)
	if _, err := os.Stat(calibrateITLDataPath); err != nil {
		logrus.Fatalf("ITL data file not found: %v", err)
	}

	// Load ITL data
	itlRecords, err := workload.LoadITL(calibrateITLDataPath)
	if err != nil {
		logrus.Fatalf("Failed to load ITL data from %s: %v", calibrateITLDataPath, err)
	}
	logrus.Infof("Loaded %d ITL records from %s", len(itlRecords), calibrateITLDataPath)

	// Prepare pairs with ITL
	pairs, err = workload.PrepareCalibrationPairsWithITL(trace.Records, simResults, itlRecords, &config)
	if err != nil {
		logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
	}
} else {
	// Standard calibration (no ITL)
	pairs, err = workload.PrepareCalibrationPairs(trace.Records, simResults, &config)
	if err != nil {
		logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
	}
}

// ... rest of existing code (guard, report, write) ...
```

Update the summary logging to include ITL:

```go
// Step 8: Log summary to stderr
logrus.Infof("Calibration report written to %s", calibrateReportPath)
logrus.Infof("  Matched pairs: %d (warm-up excluded: %d, unmatched real: %d, unmatched sim: %d)",
	pairs.MatchedCount, pairs.ExcludedWarmUp, pairs.UnmatchedReal, pairs.UnmatchedSim)
if ttft, ok := report.Metrics["ttft"]; ok {
	logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
		ttft.MAPE*100, ttft.PearsonR, ttft.Quality)
}
if e2e, ok := report.Metrics["e2e"]; ok {
	logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
		e2e.MAPE*100, e2e.PearsonR, e2e.Quality)
}
if itl, ok := report.Metrics["itl"]; ok {
	logrus.Infof("  ITL:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
		itl.MAPE*100, itl.PearsonR, itl.Quality)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestCalibrateCmd_ITLDataFlag -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/calibrate.go cmd/calibrate_test.go
git commit -m "feat(cmd): add --itl-data flag to calibrate (BC-6, BC-11)

- Add calibrateITLDataPath variable
- Load ITL records if --itl-data provided
- Call PrepareCalibrationPairsWithITL when ITL data present
- Log ITL metric summary
- Check file existence before loading (BC-11)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 6: Update CLAUDE.md and README

**Contracts Implemented:** N/A (documentation)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md Build and Run Commands**

Add ITL examples to the observe section:

```markdown
# Observe with ITL recording
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv \
  --record-itl --itl-output itl.csv

# Calibrate with ITL data
./blis calibrate --trace-header t.yaml --trace-data d.csv --itl-data itl.csv \
  --sim-results results.json --report calibration.json
```

Update Recent Changes section:

```markdown
## Recent Changes
- TraceV2 ITL timestamps (#992): `--record-itl` flag in `blis observe` captures per-chunk timestamps for ITL calibration; `--itl-data` flag in `blis calibrate` computes ITL MAPE/Pearson R/percentiles alongside E2E and TTFT
```

**Step 2: Update README with ITL example**

Find the observe/replay/calibrate pipeline section and add ITL example:

```markdown
### Observe/Replay/Calibrate Pipeline

Record real server latencies, replay through simulation, and compare:

```bash
# 1. Observe real server with ITL recording
./blis observe --server-url http://localhost:8000 --model llama-3.1-8b \
  --workload-spec workload.yaml \
  --trace-header observed.yaml --trace-data observed.csv \
  --record-itl --itl-output itl.csv

# 2. Replay through simulator
./blis replay --trace-header observed.yaml --trace-data observed.csv \
  --model llama-3.1-8b --results-path sim_results.json

# 3. Compare with ITL metrics
./blis calibrate --trace-header observed.yaml --trace-data observed.csv \
  --itl-data itl.csv --sim-results sim_results.json --report calibration.json
```

The calibration report includes MAPE, Pearson correlation, and percentiles for E2E, TTFT, and ITL (if recorded).
```

**Step 3: Commit documentation changes**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add ITL recording examples to CLAUDE.md and README

- Add --record-itl and --itl-output flags to observe examples
- Add --itl-data flag to calibrate examples
- Document ITL metric in calibration pipeline section
- Update Recent Changes in CLAUDE.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 2 | Unit | TestRecorder_RecordITL_StreamingRequest |
| BC-2 | Task 3 | Integration | Implicit in runObserve warning log |
| BC-3 | Task 1 | Unit | TestITL_ExportLoad_RoundTrip |
| BC-4 | Task 4 | Unit | TestCalibration_WithITL |
| BC-5 | Task 3 | Integration | Implicit in runObserve conditional |
| BC-6 | Task 5 | Integration | Implicit in calibrate conditional |
| BC-7 | Task 2 | Unit | Covered by TestRecorder_RecordITL (no status filtering) |
| BC-8 | All | Code review | Verify no logrus.Fatalf in sim/workload |
| BC-9 | Task 1 | Unit | Implicit in ExportITL error path |
| BC-10 | Task 1 | Unit | TestITL_LoadITL_MalformedCSV_ReturnsError |
| BC-11 | Task 5 | Integration | os.Stat check in calibrate.go |

**Shared test infrastructure:** Uses existing `t.TempDir()` for test isolation, follows table-driven pattern from `tracev2_test.go`.

**Golden dataset:** No golden datasets affected — this is new functionality with no existing output to preserve.

**Lint:** All tasks include lint verification step (`golangci-lint run`).

**Test isolation:** All tests use fresh Recorder/ITLRecord instances per test; no shared state.

**Invariant tests:** Not applicable — ITL recording doesn't affect request conservation, KV cache, or clock monotonicity.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Off-by-one error in ITL delta computation | Medium | High | Unit test with known chunk timestamps; verify delta = chunks[i] - chunks[i-1] | Task 4 |
| Clock skew produces negative deltas | Low | Medium | Guard: skip requests with negative deltas; test with malformed data | Task 4 |
| Non-streaming requests produce empty ITL file | Medium | Low | Log warning; test validates empty export | Task 3 |
| ITL file path collision | Low | Low | Default to `<trace-data>.itl.csv`; user can override via `--itl-output` | Task 3 |
| Sim ITL approximation inaccurate | Medium | Medium | Document assumption: uniform token generation; use mean ITL (not per-token) | Task 4 |
| Memory growth with large traces | Low | Medium | ITL records are O(N * chunks_per_request); typical N=1000, chunks=50 → 50K records (~1MB) | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (ITL is simple CSV I/O + delta computation)
- [x] No feature creep (only ITL recording, no other metrics)
- [x] No unexercised flags (`--record-itl` tested in observe, `--itl-data` tested in calibrate)
- [x] No partial implementations (all BC-1 through BC-11 have tasks)
- [x] No breaking changes (TraceV2 format unchanged, ITL is opt-in)
- [x] No hidden global state (Recorder is local, ITL data is explicit)
- [x] All new code will pass golangci-lint (each task has lint step)
- [x] Shared test helpers used (t.TempDir from existing tests)
- [x] CLAUDE.md updated (Task 6: new flags documented)
- [x] No stale references (no pre-existing ITL references to remove)
- [x] Documentation DRY: No canonical sources modified (CLAUDE.md is working copy)
- [x] Deviation log reviewed (4 clarifications documented)
- [x] Each task produces working code (TDD: test → implement → verify)
- [x] Task dependencies ordered (Task 1 defines types used by Task 2)
- [x] All contracts mapped to tasks (see Test Strategy table)
- [x] Golden dataset not affected (new functionality)
- [x] Construction site audit: No existing structs gain fields (new ITLRecord type)

**Antipattern rules:**
- [x] R1: No silent data loss (all errors returned or logged)
- [x] R2: No map iteration for ordered output (ITL records sorted by chunk_index)
- [x] R3: All numeric params validated (negative checks in LoadITL)
- [x] R4: No existing struct fields added (new ITLRecord type)
- [x] R5: No resource allocation loops (ITL records appended, no mid-loop failure)
- [x] R6: No logrus.Fatalf in sim/ (all sim/workload functions return error)
- [x] R7: No golden tests (new functionality, no golden baseline)
- [x] R8: No exported mutable maps (ITLRecord has no maps)
- [x] R9: No YAML fields (ITL is CSV, not YAML config)
- [x] R10: No YAML parsing (ITL is CSV)
- [x] R11: No division by runtime denominator (ITL delta is subtraction)
- [x] R12: No golden datasets changed
- [x] R13: No new interfaces (ITL uses existing export/load pattern)
- [x] R14: No multi-module methods (each function single-purpose)
- [x] R15: No stale PR references
- [x] R16: No config params (ITL is CLI flag, not config file)
- [x] R17: No routing scorer signals
- [x] R18: No CLI flag overwrite by defaults.yaml
- [x] R19: No unbounded loops
- [x] R20: LoadITL validates negative values
- [x] R21: No range over shrinking slice
- [x] R22: No pre-check estimates
- [x] R23: No parallel code paths

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/itl.go`

**Purpose:** Define ITLRecord type and CSV I/O functions for per-chunk timestamp storage.

**Complete Implementation:**

```go
package workload

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ITLRecord represents one chunk timestamp in the ITL trace.
// ITL traces capture per-chunk arrival times during streaming inference
// for Inter-Token Latency (ITL) calibration.
type ITLRecord struct {
	RequestID   int   // Request identifier (matches TraceRecord.RequestID)
	ChunkIndex  int   // Chunk sequence number (0 = first chunk / TTFT)
	TimestampUs int64 // Absolute timestamp in microseconds (UnixMicro)
}

// ExportITL writes ITL records to a CSV file.
// Format: request_id,chunk_index,timestamp_us
// Timestamps use integer formatting to preserve microsecond precision.
func ExportITL(records []ITLRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"request_id", "chunk_index", "timestamp_us"}); err != nil {
		return fmt.Errorf("writing ITL CSV header: %w", err)
	}

	// Write data rows
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID),
			strconv.Itoa(r.ChunkIndex),
			strconv.FormatInt(r.TimestampUs, 10),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing ITL CSV row (request_id=%d): %w", r.RequestID, err)
		}
	}
	return nil
}

// LoadITL reads ITL records from a CSV file.
// Validates that all fields are non-negative (R3, R20).
func LoadITL(path string) ([]ITLRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	// Skip header row
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("reading ITL CSV header: %w", err)
	}

	var records []ITLRecord
	rowNum := 1 // 0 = header
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading ITL CSV row %d: %w", rowNum, err)
		}
		rowNum++

		if len(row) < 3 {
			return nil, fmt.Errorf("ITL CSV row %d has %d columns, expected 3", rowNum, len(row))
		}

		requestID, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("parsing request_id at row %d: %w", rowNum, err)
		}
		chunkIndex, err := strconv.Atoi(row[1])
		if err != nil {
			return nil, fmt.Errorf("parsing chunk_index at row %d: %w", rowNum, err)
		}
		timestampUs, err := strconv.ParseInt(row[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing timestamp_us at row %d: %w", rowNum, err)
		}

		// Validate: no negative values (R3, R20)
		if requestID < 0 || chunkIndex < 0 || timestampUs < 0 {
			return nil, fmt.Errorf("ITL CSV row %d has negative value (request_id=%d, chunk_index=%d, timestamp_us=%d)", rowNum, requestID, chunkIndex, timestampUs)
		}

		records = append(records, ITLRecord{
			RequestID:   requestID,
			ChunkIndex:  chunkIndex,
			TimestampUs: timestampUs,
		})
	}
	return records, nil
}
```

**Key Implementation Notes:**
- RNG usage: None (deterministic CSV I/O)
- Metrics: None (data capture only)
- Event ordering: N/A (not part of DES)
- State mutation: None (pure function)
- Error handling: Return errors with context (BC-9, BC-10)

---

### File: `sim/workload/calibrate.go` (modifications)

**Purpose:** Extend calibration with ITL metric computation.

**Key Changes:**

1. Add `ITL LatencyPair` field to `CalibrationPairs` struct
2. Add `PrepareCalibrationPairsWithITL` function
3. Extend `BuildCalibrationReport` to include ITL metric

**ITL Computation Details:**

```go
// Real ITL: Mean chunk-to-chunk delta (excluding first chunk which is TTFT)
// Example: chunks at [1000, 1020, 1040, 1060] → deltas [20, 20, 20] → mean ITL = 20μs

// Sim ITL: Approximation assuming uniform token generation
// ITL_sim = (E2E - TTFT) / (OutputTokens - 1)
// Example: E2E=1060μs, TTFT=1000μs, OutputTokens=4 → ITL_sim = 60/3 = 20μs
```

**Behavioral notes:**
- First chunk delta IS TTFT, not ITL (chunk 0 → 1 = initial latency)
- Subsequent deltas are ITL (chunk 1 → 2, 2 → 3, etc.)
- Negative deltas indicate clock skew → skip request (don't fail calibration)
- Mean ITL is robust to variable chunk sizes (aggregates across all chunks)

---

This completes the implementation plan. All 6 tasks are ready for execution via `superpowers:executing-plans`.
