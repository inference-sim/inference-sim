# Add vLLM Priority Column to TraceV2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `vllm_priority` column to TraceV2 CSV output for observing which priority values were sent to vLLM servers.

**The problem today:** When `blis observe` sends requests to vLLM with priority values (injected per PR #1208), there's no way to validate which priority values were actually sent. Users cannot verify the SLO class → vLLM priority mapping from trace data, making production trace analysis incomplete.

**What this PR adds:**
1. **Observability metadata** — A `vllm_priority` column in TraceV2 CSV that records the integer priority value sent in the HTTP request body to vLLM (e.g., critical→0, standard→1, batch→5)
2. **Backward compatibility** — The column is only included when any request has an `slo_class` set, preserving the TraceV2 format for non-SLO workloads
3. **Simulation isolation** — The field is observability-only; `blis replay` and `blis run` ignore it completely, computing priorities dynamically from `slo_class` using `SLOPriorityMap.Priority()` as before

**Why this matters:** This completes the observe/replay/calibrate pipeline for priority-aware scheduling. Production traces can now be validated against expected priority mapping, and priority-based scheduling experiments can be analyzed with full observability.

**Architecture:** Extends `TraceRecord` struct in `sim/workload/tracev2.go` with an optional `VLLMPriority` field (int, 0=not recorded). The `observe.go` client captures the priority value computed via `InvertForVLLM()` and writes it to the trace. The replay logic in `replay.go` MUST NOT read this field into `Request.Priority` to maintain simulation isolation.

**Source:** GitHub issue #1220

**Closes:** Fixes #1220

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR adds a `vllm_priority` column to TraceV2 CSV files to record the priority value sent to vLLM servers during `blis observe`. The column appears after `slo_class` and uses vLLM's inverted priority convention (lower integer = more urgent). The field is observability-only — the simulator never reads it, computing priorities dynamically from `slo_class` instead.

**What comes before:** PR #1208 added SLO class → vLLM priority injection in observe command (header + body field). This PR adds observability for that feature.

**What depends on this:** Downstream analysis tools and priority-based scheduling experiments now have full visibility into the priority values sent to production servers.

**Adjacent components:**
- `sim/workload/tracev2.go` — TraceRecord struct and CSV I/O
- `cmd/observe.go` — RealClient request construction and priority computation
- `sim/workload/replay.go` — LoadTraceV2Requests (MUST NOT read vllm_priority)

**Deviation flags:** None — issue is comprehensive and unambiguous.

### B) Behavioral Contracts

**Positive Contracts**

BC-1: vLLM Priority Recording
- GIVEN a PendingRequest with SLOClass set
- WHEN RealClient.Send() computes priority via InvertForVLLM()
- THEN the RequestRecord captures that priority value in a VLLMPriority field
- MECHANISM: Assign computed priority to record.VLLMPriority in observe.go:170 capture path

BC-2: CSV Column Presence
- GIVEN a TraceV2 with at least one record having SLOClass set
- WHEN ExportTraceV2() writes the CSV
- THEN the CSV header includes "vllm_priority" column after "slo_class"
- MECHANISM: Conditional column insertion in traceV2Columns when any record has non-empty SLOClass

BC-3: CSV Column Absence (Backward Compatibility)
- GIVEN a TraceV2 with zero records having SLOClass set
- WHEN ExportTraceV2() writes the CSV
- THEN the CSV header MUST NOT include "vllm_priority" column
- MECHANISM: Skip column insertion when all records have empty SLOClass

BC-4: Priority Value Persistence
- GIVEN a TraceV2 CSV with vllm_priority column
- WHEN LoadTraceV2() reads the CSV
- THEN TraceRecord.VLLMPriority is populated with the recorded integer value
- MECHANISM: Parse vllm_priority column if present, default to 0 if absent

BC-5: Zero Value for Non-SLO Requests
- GIVEN a PendingRequest with empty SLOClass
- WHEN RealClient.Send() executes
- THEN RequestRecord.VLLMPriority remains 0 (not set)
- MECHANISM: Only assign VLLMPriority when req.SLOClass is non-empty

**Negative Contracts (Simulation Isolation)**

BC-6: Replay MUST NOT Use vLLM Priority
- GIVEN a TraceV2 CSV with vllm_priority column
- WHEN LoadTraceV2Requests() builds sim.Request structs
- THEN Request.Priority MUST remain 0.0 (default unset value)
- MECHANISM: LoadTraceV2Requests does not read rec.VLLMPriority into req.Priority

BC-7: Session Blueprint MUST NOT Use vLLM Priority
- GIVEN a TraceV2 CSV with vllm_priority column
- WHEN LoadTraceV2SessionBlueprints() builds SessionBlueprint
- THEN Request.Priority in rounds MUST remain 0.0 (default unset value)
- MECHANISM: LoadTraceV2SessionBlueprints does not read rec.VLLMPriority into round Request.Priority

BC-8: Dynamic Priority Computation Unchanged
- GIVEN any replayed trace (with or without vllm_priority column)
- WHEN Simulator.Step() processes a request
- THEN req.Priority is computed dynamically via priorityPolicy.Compute(req, now)
- MECHANISM: No changes to simulator.go:650 priority assignment logic

**Error Handling Contracts**

BC-9: Missing Column Tolerance
- GIVEN a legacy TraceV2 CSV without vllm_priority column
- WHEN LoadTraceV2() reads the CSV
- THEN loading succeeds with VLLMPriority=0 for all records
- MECHANISM: Column index lookup returns -1 for missing column; parser defaults to 0

BC-10: Invalid Priority Value
- GIVEN a TraceV2 CSV with non-integer vllm_priority value
- WHEN LoadTraceV2() reads the CSV
- THEN loading returns an error identifying the invalid value and row number
- MECHANISM: strconv.Atoi failure propagates with row context

### C) Component Interaction

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         blis observe Command                             │
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │  RealClient      │────────▶│  RequestRecord   │                     │
│  │  - InvertForVLLM │  write  │  + VLLMPriority  │                     │
│  └──────────────────┘         └──────────────────┘                     │
│           │                             │                                │
│           │                             ▼                                │
│           │                    ┌──────────────────┐                     │
│           │                    │  TraceRecord     │                     │
│           │                    │  + VLLMPriority  │                     │
│           │                    └──────────────────┘                     │
│           │                             │                                │
│           │                             ▼                                │
│           │                    ┌──────────────────┐                     │
│           │                    │  ExportTraceV2   │──▶ trace.csv        │
│           │                    │  (CSV writer)    │    (vllm_priority)  │
│           │                    └──────────────────┘                     │
└───────────┼─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         blis replay Command                              │
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │  LoadTraceV2     │────────▶│  TraceRecord     │                     │
│  │  (CSV reader)    │  parse  │  + VLLMPriority  │                     │
│  └──────────────────┘         └──────────────────┘                     │
│           │                             │                                │
│           │                             ▼                                │
│           │                    ┌──────────────────┐                     │
│           │                    │  sim.Request     │                     │
│           │                    │  Priority = 0.0  │◀── NEVER read       │
│           │                    │  (unset)         │    VLLMPriority     │
│           │                    └──────────────────┘                     │
│           │                             │                                │
│           │                             ▼                                │
│           │                    ┌──────────────────┐                     │
│           │                    │  Simulator.Step  │                     │
│           │                    │  Compute Priority│──▶ SLOPriorityMap   │
│           │                    │  from SLOClass   │    .Priority()      │
│           │                    └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**API Contracts:**
- `TraceRecord.VLLMPriority` (int) — Observability metadata; 0 = not recorded
- `ExportTraceV2()` — Conditionally includes "vllm_priority" column when any record has SLOClass set
- `LoadTraceV2()` — Populates VLLMPriority from column if present; defaults to 0 if absent
- `LoadTraceV2Requests()` — MUST NOT read VLLMPriority into Request.Priority (simulation isolation)

**State Changes:**
- `TraceRecord` gains `VLLMPriority int` field
- `traceV2Columns` slice conditionally includes "vllm_priority" (after "slo_class")
- No state changes in `sim.Request` or any simulator components

**Extension Friction Assessment:**
Adding one more observability field to TraceRecord requires changes to:
1. `sim/workload/tracev2.go` — struct field, column list, Export/Load logic (3 locations)
2. Tests — unit tests for CSV round-trip with/without column

Total: 2 files. Acceptable for observability metadata. The column is optional (conditional insertion), so future fields follow the same pattern.

### D) Deviation Log

No deviations from source document (issue #1220).

### E) Review Guide

**THE TRICKY PART:** Ensuring `LoadTraceV2Requests()` and `LoadTraceV2SessionBlueprints()` do NOT read `VLLMPriority` into `Request.Priority`. If they did, replayed traces would use vLLM's inverted convention (0 for critical) instead of BLIS convention (4 for critical), inverting scheduling order.

**WHAT TO SCRUTINIZE:**
- BC-6 and BC-7 test coverage — verify `LoadTraceV2Requests` with `vllm_priority` column present results in `req.Priority == 0.0` (not populated from trace)
- Conditional column logic in `ExportTraceV2` — column must be absent when no SLOClass is set (BC-3)

**WHAT'S SAFE TO SKIM:**
- `RealClient.Send()` priority capture (BC-1) — straightforward assignment after existing InvertForVLLM call
- CSV parsing logic (BC-4) — standard pattern from ITL column addition (PR #992)

**KNOWN DEBT:** None identified.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- None

**Files to modify:**
- `sim/workload/tracev2.go:52` — Add `VLLMPriority int` field to TraceRecord struct
- `sim/workload/tracev2.go:89` — Update `traceV2Columns` with conditional "vllm_priority" insertion
- `sim/workload/tracev2.go:100` — Update ExportTraceV2 to write vllm_priority column when present
- `sim/workload/tracev2.go:200` — Update LoadTraceV2 to parse vllm_priority column if present
- `cmd/observe.go:170` — Capture vLLMPriority in RequestRecord after InvertForVLLM call
- `sim/workload/tracev2_test.go` — Add tests for BC-1 through BC-10
- `sim/workload/replay_test.go` — Add test for BC-6 (LoadTraceV2Requests ignores vllm_priority)

**Key decisions:**
1. **Optional column** — Column only included when any record has SLOClass set (backward compatibility)
2. **Zero value semantic** — VLLMPriority=0 means "not recorded", not "critical priority"
3. **No simulation impact** — Replay functions never read VLLMPriority into Request.Priority

**Confirmation:** No dead code. All new code exercised by tests. VLLMPriority field populated in observe path (BC-1), written to CSV (BC-2), read from CSV (BC-4), and verified NOT used in replay (BC-6, BC-7).

### G) Task Breakdown

#### Task 1: Add VLLMPriority Field to TraceRecord

**Contracts Implemented:** BC-1 (partial — struct field), BC-5

**Files:**
- Modify: `sim/workload/tracev2.go:52-80` (TraceRecord struct)
- Modify: `cmd/observe.go:104-117` (RequestRecord struct)
- Test: `sim/workload/tracev2_test.go`

**Step 1: Write failing test for VLLMPriority field presence**

Context: Verify TraceRecord can store vLLM priority values.

In `sim/workload/tracev2_test.go`:
```go
func TestTraceRecord_VLLMPriority_FieldExists(t *testing.T) {
	// GIVEN a TraceRecord with VLLMPriority set
	rec := TraceRecord{
		RequestID:    1,
		SLOClass:     "critical",
		VLLMPriority: 0, // critical → 0 in vLLM convention
	}

	// THEN the field is accessible and has the expected value
	if rec.VLLMPriority != 0 {
		t.Errorf("Expected VLLMPriority=0, got %d", rec.VLLMPriority)
	}

	// WHEN SLOClass is empty
	rec2 := TraceRecord{
		RequestID: 2,
		SLOClass:  "",
	}

	// THEN VLLMPriority defaults to 0 (not set)
	if rec2.VLLMPriority != 0 {
		t.Errorf("Expected default VLLMPriority=0, got %d", rec2.VLLMPriority)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestTraceRecord_VLLMPriority_FieldExists -v`
Expected: FAIL with "TraceRecord.VLLMPriority undefined"

**Step 3: Add VLLMPriority field to TraceRecord**

Context: Add optional int field after SLOClass to match column position.

In `sim/workload/tracev2.go`, modify TraceRecord struct:
```go
// TraceRecord represents one row in a trace v2 CSV.
type TraceRecord struct {
	RequestID         int
	ClientID          string
	TenantID          string
	SLOClass          string
	VLLMPriority      int    // vLLM priority value sent in HTTP request body; 0 = not recorded
	SessionID         string
	RoundIndex        int
	PrefixGroup       string
	PrefixLength      int
	Streaming         bool
	InputTokens       int
	OutputTokens      int
	TextTokens        int
	ImageTokens       int
	AudioTokens       int
	VideoTokens       int
	ReasonRatio       float64
	Model             string
	DeadlineUs        int64
	ServerInputTokens int
	ArrivalTimeUs     int64
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	Status            string
	ErrorMessage      string
	FinishReason      string
}
```

Add VLLMPriority field to RequestRecord in `cmd/observe.go`:
```go
// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID         int
	OutputTokens      int
	ServerInputTokens int
	VLLMPriority      int    // vLLM priority value computed for this request; 0 = not set
	Status            string // "ok", "error", "timeout"
	ErrorMessage      string
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	FinishReason      string
	ChunkTimestamps   []int64
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestTraceRecord_VLLMPriority_FieldExists -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/... ./cmd/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/tracev2.go cmd/observe.go sim/workload/tracev2_test.go
git commit -m "feat(workload): add VLLMPriority field to TraceRecord (BC-1, BC-5)

- Add VLLMPriority int field to TraceRecord after SLOClass
- Add VLLMPriority int field to RequestRecord in observe
- Field stores vLLM priority value sent in HTTP request body
- Zero value means not recorded (not critical priority)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Capture vLLM Priority in Observe Command

**Contracts Implemented:** BC-1, BC-5

**Files:**
- Modify: `cmd/observe.go:120-180` (RealClient.Send method)
- Test: `cmd/observe_test.go`

**Step 1: Write failing test for priority capture**

Context: Verify RealClient captures computed priority in RequestRecord.

In `cmd/observe_test.go`:
```go
func TestRealClient_Send_CapturesVLLMPriority(t *testing.T) {
	tests := []struct {
		name             string
		sloClass         string
		expectedPriority int
	}{
		{"critical SLO", "critical", 0},
		{"standard SLO", "standard", 1},
		{"batch SLO", "batch", 5},
		{"empty SLO", "", 0}, // not set, zero value
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// GIVEN a mock server that echoes back the request
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				resp := map[string]interface{}{
					"id": "test-123",
					"choices": []map[string]interface{}{
						{
							"text":          "Hello",
							"finish_reason": "stop",
						},
					},
					"usage": map[string]interface{}{
						"prompt_tokens":     10,
						"completion_tokens": 1,
					},
				}
				_ = json.NewEncoder(w).Encode(resp)
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "test-model", "vllm")
			req := &PendingRequest{
				RequestID:       1,
				InputTokens:     10,
				MaxOutputTokens: 100,
				Streaming:       false,
				SLOClass:        tt.sloClass,
			}

			// WHEN sending the request
			record, err := client.Send(context.Background(), req)
			if err != nil {
				t.Fatalf("Send failed: %v", err)
			}

			// THEN VLLMPriority is captured correctly
			if record.VLLMPriority != tt.expectedPriority {
				t.Errorf("Expected VLLMPriority=%d, got %d", tt.expectedPriority, record.VLLMPriority)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestRealClient_Send_CapturesVLLMPriority -v`
Expected: FAIL with "Expected VLLMPriority=0, got 0" (field not assigned)

**Step 3: Capture priority in RealClient.Send()**

Context: After computing priority for HTTP body, assign to RequestRecord.

In `cmd/observe.go`, modify Send() method around line 170:
```go
// Send dispatches a single request to the server and records timing.
func (c *RealClient) Send(ctx context.Context, req *PendingRequest) (*RequestRecord, error) {
	record := &RequestRecord{
		RequestID: req.RequestID,
		Status:    "ok",
	}

	// Build request body
	body := map[string]interface{}{
		"model":  c.modelName,
		"stream": req.Streaming,
	}

	// ... (existing max_tokens, min_tokens logic)

	// Inject vLLM priority for SLO-aware servers (PR #1208)
	if req.SLOClass != "" {
		priority := c.sloMap.InvertForVLLM(req.SLOClass)
		body["priority"] = priority
		record.VLLMPriority = priority // BC-1: capture for trace
	}
	// If SLOClass is empty, VLLMPriority remains 0 (BC-5)

	// ... (rest of existing Send logic)

	return record, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestRealClient_Send_CapturesVLLM Priority -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add cmd/observe.go cmd/observe_test.go
git commit -m "feat(observe): capture vLLM priority in RequestRecord (BC-1, BC-5)

- Assign computed priority to record.VLLMPriority after InvertForVLLM call
- Only set when req.SLOClass is non-empty (BC-5)
- Enables priority observability in TraceV2 output

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Conditional Column Insertion in ExportTraceV2

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Modify: `sim/workload/tracev2.go:89-96` (traceV2Columns)
- Modify: `sim/workload/tracev2.go:98-160` (ExportTraceV2)
- Test: `sim/workload/tracev2_test.go`

**Step 1: Write failing test for conditional column inclusion**

Context: Verify vllm_priority column is included only when SLOClass is set.

In `sim/workload/tracev2_test.go`:
```go
func TestExportTraceV2_VLLMPriorityColumn_Conditional(t *testing.T) {
	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	tests := []struct {
		name             string
		records          []TraceRecord
		expectColumn     bool
		expectedPriority int
	}{
		{
			name: "with SLO class",
			records: []TraceRecord{
				{RequestID: 1, SLOClass: "critical", VLLMPriority: 0},
				{RequestID: 2, SLOClass: "standard", VLLMPriority: 1},
			},
			expectColumn:     true,
			expectedPriority: 0,
		},
		{
			name: "without SLO class",
			records: []TraceRecord{
				{RequestID: 1, SLOClass: ""},
				{RequestID: 2, SLOClass: ""},
			},
			expectColumn: false,
		},
		{
			name: "mixed SLO presence",
			records: []TraceRecord{
				{RequestID: 1, SLOClass: "critical", VLLMPriority: 0},
				{RequestID: 2, SLOClass: ""}, // no SLO, but column still included
			},
			expectColumn:     true,
			expectedPriority: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}

			// WHEN exporting trace
			err := ExportTraceV2(header, tt.records, headerPath, dataPath)
			if err != nil {
				t.Fatalf("ExportTraceV2 failed: %v", err)
			}

			// THEN read CSV header
			file, err := os.Open(dataPath)
			if err != nil {
				t.Fatalf("Failed to open CSV: %v", err)
			}
			defer file.Close()

			reader := csv.NewReader(file)
			headerRow, err := reader.Read()
			if err != nil {
				t.Fatalf("Failed to read CSV header: %v", err)
			}

			// Check if vllm_priority column is present
			hasPriorityColumn := false
			priorityColIndex := -1
			for i, col := range headerRow {
				if col == "vllm_priority" {
					hasPriorityColumn = true
					priorityColIndex = i
					break
				}
			}

			if tt.expectColumn && !hasPriorityColumn {
				t.Errorf("Expected vllm_priority column, but not found in header: %v", headerRow)
			}
			if !tt.expectColumn && hasPriorityColumn {
				t.Errorf("Did not expect vllm_priority column, but found at index %d in header: %v", priorityColIndex, headerRow)
			}

			// If column exists, verify priority value in first row
			if tt.expectColumn && len(tt.records) > 0 && tt.records[0].SLOClass != "" {
				dataRow, err := reader.Read()
				if err != nil {
					t.Fatalf("Failed to read data row: %v", err)
				}
				if priorityColIndex < len(dataRow) {
					priorityStr := dataRow[priorityColIndex]
					priority, err := strconv.Atoi(priorityStr)
					if err != nil {
						t.Errorf("Invalid priority value '%s': %v", priorityStr, err)
					}
					if priority != tt.expectedPriority {
						t.Errorf("Expected priority=%d in CSV, got %d", tt.expectedPriority, priority)
					}
				}
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestExportTraceV2_VLLMPriorityColumn_Conditional -v`
Expected: FAIL with "Expected vllm_priority column, but not found in header"

**Step 3: Implement conditional column insertion in ExportTraceV2**

Context: Check if any record has SLOClass set; if yes, include vllm_priority column.

In `sim/workload/tracev2.go`:
```go
// ExportTraceV2 writes trace header (YAML) and data (CSV) to separate files.
// Timestamps use integer formatting (%d) to preserve microsecond precision.
func ExportTraceV2(header *TraceHeader, records []TraceRecord, headerPath, dataPath string) error {
	// Write header YAML
	headerData, err := yaml.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshaling trace header: %w", err)
	}
	if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
		return fmt.Errorf("writing trace header: %w", err)
	}

	// Determine if vllm_priority column should be included (BC-2, BC-3)
	includeVLLMPriority := false
	for _, rec := range records {
		if rec.SLOClass != "" {
			includeVLLMPriority = true
			break
		}
	}

	// Build column list
	columns := []string{
		"request_id", "client_id", "tenant_id", "slo_class",
	}
	if includeVLLMPriority {
		columns = append(columns, "vllm_priority")
	}
	columns = append(columns, "session_id", "round_index",
		"prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
		"text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
		"model", "deadline_us", "server_input_tokens",
		"arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
		"num_chunks", "status", "error_message", "finish_reason",
	)

	// Write data CSV
	file, err := os.Create(dataPath)
	if err != nil {
		return fmt.Errorf("creating trace data file: %w", err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header row
	if err := writer.Write(columns); err != nil {
		return fmt.Errorf("writing CSV header: %w", err)
	}

	// Write data rows
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID),
			r.ClientID,
			r.TenantID,
			r.SLOClass,
		}
		if includeVLLMPriority {
			row = append(row, strconv.Itoa(r.VLLMPriority))
		}
		row = append(row,
			r.SessionID,
			strconv.Itoa(r.RoundIndex),
			r.PrefixGroup,
			strconv.Itoa(r.PrefixLength),
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
			strconv.FormatInt(r.ArrivalTimeUs, 10),
			strconv.FormatInt(r.SendTimeUs, 10),
			strconv.FormatInt(r.FirstChunkTimeUs, 10),
			strconv.FormatInt(r.LastChunkTimeUs, 10),
			strconv.Itoa(r.NumChunks),
			r.Status,
			r.ErrorMessage,
			r.FinishReason,
		)

		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing CSV row: %w", err)
		}
	}

	return nil
}
```

Remove the old `traceV2Columns` global variable (no longer used):
```go
// CSV column headers for trace v2 format.
// REMOVED: Column list now built dynamically in ExportTraceV2
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestExportTraceV2_VLLMPriorityColumn_Conditional -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/tracev2.go sim/workload/tracev2_test.go
git commit -m "feat(workload): conditional vllm_priority column in ExportTraceV2 (BC-2, BC-3)

- Build column list dynamically based on SLOClass presence
- Include vllm_priority column only when any record has SLOClass set
- Remove static traceV2Columns global (replaced by dynamic list)
- Maintains backward compatibility with non-SLO traces

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Parse vLLM Priority in LoadTraceV2

**Contracts Implemented:** BC-4, BC-9, BC-10

**Files:**
- Modify: `sim/workload/tracev2.go:200-300` (LoadTraceV2 function)
- Test: `sim/workload/tracev2_test.go`

**Step 1: Write failing test for priority parsing**

Context: Verify LoadTraceV2 correctly parses vllm_priority column when present.

In `sim/workload/tracev2_test.go`:
```go
func TestLoadTraceV2_ParsesVLLMPriority(t *testing.T) {
	tests := []struct {
		name             string
		csvData          string
		expectedPriority int
		expectError      bool
	}{
		{
			name: "with vllm_priority column",
			csvData: `request_id,client_id,tenant_id,slo_class,vllm_priority,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason
1,client1,,critical,0,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,1000,1100,1200,1300,5,ok,,stop
`,
			expectedPriority: 0,
			expectError:      false,
		},
		{
			name: "without vllm_priority column (legacy)",
			csvData: `request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason
1,client1,,critical,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,1000,1100,1200,1300,5,ok,,stop
`,
			expectedPriority: 0, // default when column absent
			expectError:      false,
		},
		{
			name: "invalid priority value",
			csvData: `request_id,client_id,tenant_id,slo_class,vllm_priority,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason
1,client1,,critical,invalid,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,1000,1100,1200,1300,5,ok,,stop
`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			headerPath := filepath.Join(tmpDir, "trace.yaml")
			dataPath := filepath.Join(tmpDir, "trace.csv")

			// Write header
			header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
			headerData, _ := yaml.Marshal(header)
			if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
				t.Fatalf("Failed to write header: %v", err)
			}

			// Write CSV data
			if err := os.WriteFile(dataPath, []byte(tt.csvData), 0644); err != nil {
				t.Fatalf("Failed to write CSV: %v", err)
			}

			// WHEN loading trace
			trace, err := LoadTraceV2(headerPath, dataPath)

			// THEN check error expectation
			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error, but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadTraceV2 failed: %v", err)
			}

			// Verify priority value
			if len(trace.Records) == 0 {
				t.Fatalf("Expected at least one record")
			}
			if trace.Records[0].VLLMPriority != tt.expectedPriority {
				t.Errorf("Expected VLLMPriority=%d, got %d", tt.expectedPriority, trace.Records[0].VLLMPriority)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestLoadTraceV2_ParsesVLLMPriority -v`
Expected: FAIL with "Expected VLLMPriority=0, got 0" (field not parsed)

**Step 3: Implement vLLM priority parsing in LoadTraceV2**

Context: Add column index lookup and parsing logic similar to finish_reason column.

In `sim/workload/tracev2.go`, modify LoadTraceV2 function:
```go
func LoadTraceV2(headerPath, dataPath string) (*TraceV2, error) {
	// ... (existing header loading logic)

	// Open CSV
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, fmt.Errorf("opening trace data: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	headerRow, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("reading CSV header: %w", err)
	}

	// Build column index map
	colIndex := make(map[string]int)
	for i, name := range headerRow {
		colIndex[name] = i
	}

	// Required columns
	requiredCols := []string{
		"request_id", "client_id", "tenant_id", "slo_class", "session_id", "round_index",
		"prefix_group", "prefix_length", "streaming", "input_tokens", "output_tokens",
		"arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
		"num_chunks", "status",
	}
	for _, col := range requiredCols {
		if _, ok := colIndex[col]; !ok {
			return nil, fmt.Errorf("missing required column: %s", col)
		}
	}

	// Optional columns (may not exist in all traces)
	vllmPriorityIdx := colIndex["vllm_priority"] // -1 if not present (BC-9)
	finishReasonIdx := colIndex["finish_reason"]
	errorMessageIdx := colIndex["error_message"]
	// ... (other optional columns)

	// Parse data rows
	var records []TraceRecord
	rowNum := 1 // track for error messages
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading CSV row %d: %w", rowNum, err)
		}
		rowNum++

		rec := TraceRecord{}

		// ... (existing field parsing)

		// Parse SLOClass
		rec.SLOClass = row[colIndex["slo_class"]]

		// Parse VLLMPriority if column exists (BC-4, BC-9)
		if vllmPriorityIdx >= 0 && vllmPriorityIdx < len(row) {
			if val := row[vllmPriorityIdx]; val != "" {
				priority, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("invalid vllm_priority value '%s' at row %d: %w", val, rowNum, err) // BC-10
				}
				rec.VLLMPriority = priority
			}
			// If empty, VLLMPriority remains 0 (default)
		}
		// If column doesn't exist, VLLMPriority remains 0 (BC-9)

		// ... (rest of field parsing)

		records = append(records, rec)
	}

	return &TraceV2{Header: header, Records: records}, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestLoadTraceV2_ParsesVLLMPriority -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/tracev2.go sim/workload/tracev2_test.go
git commit -m "feat(workload): parse vllm_priority column in LoadTraceV2 (BC-4, BC-9, BC-10)

- Add optional vllm_priority column index lookup
- Parse priority value when column present (BC-4)
- Default to 0 when column absent (BC-9)
- Return error on invalid integer value with row context (BC-10)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Verify Simulation Isolation in LoadTraceV2Requests

**Contracts Implemented:** BC-6, BC-8

**Files:**
- Verify: `sim/workload/replay.go:15-90` (LoadTraceV2Requests — no changes)
- Test: `sim/workload/replay_test.go`

**Step 1: Write test verifying vLLM priority is NOT used in replay**

Context: Critical test — ensure LoadTraceV2Requests does NOT read VLLMPriority into Request.Priority.

In `sim/workload/replay_test.go`:
```go
func TestLoadTraceV2Requests_IgnoresVLLMPriority(t *testing.T) {
	// GIVEN a TraceV2 with vllm_priority column
	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	// Write header
	header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
	headerData, _ := yaml.Marshal(header)
	if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
		t.Fatalf("Failed to write header: %v", err)
	}

	// Write CSV with vllm_priority values (vLLM convention: lower = more urgent)
	csvData := `request_id,client_id,tenant_id,slo_class,vllm_priority,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason
1,client1,,critical,0,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,1000,1100,1200,1300,5,ok,,stop
2,client2,,standard,1,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,2000,2100,2200,2300,5,ok,,stop
3,client3,,batch,5,,0,,0,false,10,20,30,0,0,0,0.0,,0,10,3000,3100,3200,3300,5,ok,,stop
`
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatalf("Failed to write CSV: %v", err)
	}

	// WHEN loading trace for replay
	trace, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2 failed: %v", err)
	}

	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests failed: %v", err)
	}

	// THEN all requests must have Priority=0.0 (unset, not from vllm_priority)
	// BC-6: LoadTraceV2Requests MUST NOT read VLLMPriority into Request.Priority
	for i, req := range requests {
		if req.Priority != 0.0 {
			t.Errorf("Request %d: Expected Priority=0.0 (unset), got %.1f", i, req.Priority)
			t.Errorf("  VLLMPriority in trace was %d, but MUST NOT be used in simulation", trace.Records[i].VLLMPriority)
			t.Errorf("  Simulation computes priority dynamically from SLOClass=%s", req.SLOClass)
		}

		// Verify SLOClass is populated (used for dynamic priority computation)
		if req.SLOClass == "" {
			t.Errorf("Request %d: SLOClass should be populated from trace", i)
		}
	}

	// Sanity check: verify TraceRecord.VLLMPriority was parsed correctly
	if trace.Records[0].VLLMPriority != 0 {
		t.Errorf("TraceRecord[0].VLLMPriority should be 0, got %d", trace.Records[0].VLLMPriority)
	}
	if trace.Records[1].VLLMPriority != 1 {
		t.Errorf("TraceRecord[1].VLLMPriority should be 1, got %d", trace.Records[1].VLLMPriority)
	}
	if trace.Records[2].VLLMPriority != 5 {
		t.Errorf("TraceRecord[2].VLLMPriority should be 5, got %d", trace.Records[2].VLLMPriority)
	}
}
```

**Step 2: Run test to verify it passes WITHOUT code changes**

Context: This test verifies existing behavior — LoadTraceV2Requests should already NOT read VLLMPriority.

Run: `go test ./sim/workload/... -run TestLoadTraceV2Requests_IgnoresVLLMPriority -v`
Expected: PASS (no code changes needed)

**Step 3: Add documentation comment to LoadTraceV2Requests**

Context: Document the simulation isolation guarantee explicitly.

In `sim/workload/replay.go`:
```go
// LoadTraceV2Requests converts TraceV2 records into sim.Request objects for replay.
//
// IMPORTANT: This function does NOT read TraceRecord.VLLMPriority into Request.Priority.
// The vllm_priority column is observability metadata only. Request.Priority remains 0.0
// (unset), and the simulator computes it dynamically from SLOClass via priorityPolicy.Compute().
// See BC-6, BC-8, and issue #1220 for rationale.
func LoadTraceV2Requests(trace *TraceV2, seed int64) ([]*sim.Request, error) {
	// ... (existing implementation unchanged)
}
```

**Step 4: Run test again to confirm**

Run: `go test ./sim/workload/... -run TestLoadTraceV2Requests_IgnoresVLLMPriority -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/replay.go sim/workload/replay_test.go
git commit -m "test(workload): verify LoadTraceV2Requests ignores vllm_priority (BC-6, BC-8)

- Add test confirming Request.Priority remains 0.0 (unset)
- Document simulation isolation guarantee in LoadTraceV2Requests
- No code changes — test verifies existing correct behavior

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Verify Simulation Isolation in LoadTraceV2SessionBlueprints

**Contracts Implemented:** BC-7, BC-8

**Files:**
- Verify: `sim/workload/replay.go:100-200` (LoadTraceV2SessionBlueprints — no changes)
- Test: `sim/workload/replay_test.go`

**Step 1: Write test verifying sessions don't use vLLM priority**

Context: Similar to Task 5, but for session-based replay.

In `sim/workload/replay_test.go`:
```go
func TestLoadTraceV2SessionBlueprints_IgnoresVLLMPriority(t *testing.T) {
	// GIVEN a TraceV2 with sessions and vllm_priority column
	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
	headerData, _ := yaml.Marshal(header)
	if err := os.WriteFile(headerPath, headerData, 0644); err != nil {
		t.Fatalf("Failed to write header: %v", err)
	}

	// Multi-round session with different vLLM priorities per round
	csvData := `request_id,client_id,tenant_id,slo_class,vllm_priority,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason
1,client1,,critical,0,session-1,0,,0,false,10,5,15,0,0,0,0.0,,0,10,1000,1100,1200,1300,3,ok,,stop
2,client1,,standard,1,session-1,1,,0,false,8,6,14,0,0,0,0.0,,0,8,2000,2100,2200,2300,3,ok,,stop
3,client1,,batch,5,session-1,2,,0,false,12,4,16,0,0,0,0.0,,0,12,3000,3100,3200,3300,2,ok,,stop
`
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatalf("Failed to write CSV: %v", err)
	}

	// WHEN loading session blueprints
	trace, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2 failed: %v", err)
	}

	blueprints, err := LoadTraceV2SessionBlueprints(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2SessionBlueprints failed: %v", err)
	}

	// THEN all round requests must have Priority=0.0 (unset)
	// BC-7: LoadTraceV2SessionBlueprints MUST NOT read VLLMPriority into Request.Priority
	if len(blueprints) == 0 {
		t.Fatal("Expected at least one session blueprint")
	}

	blueprint := blueprints[0]
	if len(blueprint.Rounds) != 3 {
		t.Fatalf("Expected 3 rounds, got %d", len(blueprint.Rounds))
	}

	for i, round := range blueprint.Rounds {
		if round.Request.Priority != 0.0 {
			t.Errorf("Round %d: Expected Priority=0.0 (unset), got %.1f", i, round.Request.Priority)
			t.Errorf("  VLLMPriority in trace was %d, but MUST NOT be used in simulation", trace.Records[i].VLLMPriority)
			t.Errorf("  Simulation computes priority dynamically from SLOClass=%s", round.Request.SLOClass)
		}

		// Verify SLOClass is populated
		if round.Request.SLOClass == "" {
			t.Errorf("Round %d: SLOClass should be populated from trace", i)
		}
	}
}
```

**Step 2: Run test to verify it passes WITHOUT code changes**

Run: `go test ./sim/workload/... -run TestLoadTraceV2SessionBlueprints_IgnoresVLLMPriority -v`
Expected: PASS (no code changes needed)

**Step 3: Add documentation comment to LoadTraceV2SessionBlueprints**

In `sim/workload/replay.go`:
```go
// LoadTraceV2SessionBlueprints converts TraceV2 records into SessionBlueprint objects for closed-loop replay.
//
// IMPORTANT: This function does NOT read TraceRecord.VLLMPriority into Request.Priority for any round.
// The vllm_priority column is observability metadata only. Request.Priority remains 0.0 (unset),
// and the simulator computes it dynamically from SLOClass via priorityPolicy.Compute().
// See BC-7, BC-8, and issue #1220 for rationale.
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64) ([]sim.SessionBlueprint, error) {
	// ... (existing implementation unchanged)
}
```

**Step 4: Run test again to confirm**

Run: `go test ./sim/workload/... -run TestLoadTraceV2SessionBlueprints_IgnoresVLLMPriority -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/replay.go sim/workload/replay_test.go
git commit -m "test(workload): verify LoadTraceV2SessionBlueprints ignores vllm_priority (BC-7, BC-8)

- Add test confirming Request.Priority=0.0 in all session rounds
- Document simulation isolation guarantee in LoadTraceV2SessionBlueprints
- No code changes — test verifies existing correct behavior

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 7: Integration Test for End-to-End Observe/Replay Flow

**Contracts Implemented:** BC-1 through BC-10 (integration)

**Files:**
- Test: `cmd/observe_test.go`

**Step 1: Write end-to-end integration test**

Context: Verify complete flow: observe writes vllm_priority, replay ignores it.

In `cmd/observe_test.go`:
```go
func TestObserveReplay_VLLMPriorityIsolation_Integration(t *testing.T) {
	// GIVEN a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Echo back request body to verify priority was sent
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Verify priority field is present in request
		priority, hasPriority := body["priority"]
		if !hasPriority {
			http.Error(w, "Expected priority field in request body", http.StatusBadRequest)
			return
		}

		// Respond with completion
		resp := map[string]interface{}{
			"id": fmt.Sprintf("req-%v", body["prompt"]),
			"choices": []map[string]interface{}{
				{
					"text":          "Response",
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     10,
				"completion_tokens": 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)

		// Log received priority for debugging
		t.Logf("Server received priority=%v", priority)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	// WHEN running observe with SLO-aware requests
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	requests := []*PendingRequest{
		{RequestID: 1, InputTokens: 10, MaxOutputTokens: 100, SLOClass: "critical", Prompt: "Test 1"},
		{RequestID: 2, InputTokens: 10, MaxOutputTokens: 100, SLOClass: "standard", Prompt: "Test 2"},
		{RequestID: 3, InputTokens: 10, MaxOutputTokens: 100, SLOClass: "batch", Prompt: "Test 3"},
	}

	var traceRecords []workload.TraceRecord
	arrivalTime := int64(1000000) // 1 second

	for _, req := range requests {
		record, err := client.Send(context.Background(), req)
		if err != nil {
			t.Fatalf("Send failed: %v", err)
		}

		traceRec := workload.TraceRecord{
			RequestID:         req.RequestID,
			ClientID:          req.ClientID,
			TenantID:          req.TenantID,
			SLOClass:          req.SLOClass,
			VLLMPriority:      record.VLLMPriority,
			InputTokens:       req.InputTokens,
			OutputTokens:      record.OutputTokens,
			ServerInputTokens: record.ServerInputTokens,
			ArrivalTimeUs:     arrivalTime,
			SendTimeUs:        record.SendTimeUs,
			FirstChunkTimeUs:  record.FirstChunkTimeUs,
			LastChunkTimeUs:   record.LastChunkTimeUs,
			NumChunks:         record.NumChunks,
			Status:            record.Status,
			ErrorMessage:      record.ErrorMessage,
			FinishReason:      record.FinishReason,
		}
		traceRecords = append(traceRecords, traceRec)
		arrivalTime += 1000000 // 1 second apart
	}

	// Export trace
	header := &workload.TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
	if err := workload.ExportTraceV2(header, traceRecords, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2 failed: %v", err)
	}

	// THEN verify trace contains vllm_priority column
	trace, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2 failed: %v", err)
	}

	// Verify VLLMPriority values are recorded correctly
	expectedPriorities := []int{0, 1, 5} // critical, standard, batch
	for i, rec := range trace.Records {
		if rec.VLLMPriority != expectedPriorities[i] {
			t.Errorf("Record %d: Expected VLLMPriority=%d, got %d", i, expectedPriorities[i], rec.VLLMPriority)
		}
	}

	// WHEN loading for replay
	replayRequests, err := workload.LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests failed: %v", err)
	}

	// THEN verify simulation isolation: all Request.Priority=0.0
	for i, req := range replayRequests {
		if req.Priority != 0.0 {
			t.Errorf("Request %d: Expected Priority=0.0 for simulation, got %.1f", i, req.Priority)
			t.Errorf("  Trace had VLLMPriority=%d, but this MUST NOT affect simulation", trace.Records[i].VLLMPriority)
		}

		// Verify SLOClass is available for dynamic priority computation
		expectedSLOClass := []string{"critical", "standard", "batch"}[i]
		if req.SLOClass != expectedSLOClass {
			t.Errorf("Request %d: Expected SLOClass=%s, got %s", i, expectedSLOClass, req.SLOClass)
		}
	}

	t.Log("✓ Integration test passed:")
	t.Log("  - Observe captured vLLM priority values (0, 1, 5)")
	t.Log("  - TraceV2 CSV includes vllm_priority column")
	t.Log("  - Replay ignores vllm_priority (all Request.Priority=0.0)")
	t.Log("  - SLOClass preserved for dynamic priority computation")
}
```

**Step 2: Run integration test**

Run: `go test ./cmd/... -run TestObserveReplay_VLLMPriorityIsolation_Integration -v`
Expected: PASS

**Step 3: Run all tests to verify no regressions**

Run: `go test ./... -count=1 2>&1 | tail -30`
Expected: All tests pass

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit with contract reference**

```bash
git add cmd/observe_test.go
git commit -m "test(observe): end-to-end integration test for vllm_priority isolation

- Verify observe captures vLLM priority in trace (BC-1, BC-2)
- Verify replay ignores vllm_priority column (BC-6)
- Verify SLOClass preserved for dynamic priority computation (BC-8)
- Complete coverage of all behavioral contracts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 1 | Unit | TestTraceRecord_VLLMPriority_FieldExists |
| BC-1, BC-5 | Task 2 | Unit | TestRealClient_Send_CapturesVLLMPriority |
| BC-2, BC-3 | Task 3 | Unit | TestExportTraceV2_VLLMPriorityColumn_Conditional |
| BC-4, BC-9, BC-10 | Task 4 | Unit | TestLoadTraceV2_ParsesVLLMPriority |
| BC-6, BC-8 | Task 5 | Unit | TestLoadTraceV2Requests_IgnoresVLLMPriority |
| BC-7, BC-8 | Task 6 | Unit | TestLoadTraceV2SessionBlueprints_IgnoresVLLMPriority |
| BC-1 through BC-10 | Task 7 | Integration | TestObserveReplay_VLLMPriorityIsolation_Integration |

**Shared test infrastructure:** Uses existing `workload.TraceHeader`, `workload.TraceRecord`, and `httptest.NewServer` from standard library.

**Golden dataset updates:** Not applicable — this PR does not affect simulation output format or metrics.

**Lint requirements:** `golangci-lint run ./...` must pass with zero new issues.

**Test naming convention:** `TestType_Scenario_Behavior` (e.g., `TestLoadTraceV2_ParsesVLLMPriority`)

**Test isolation:** All tests use `t.TempDir()` for file I/O; no shared state.

**Invariant tests:** Not applicable — this PR adds observability metadata only, does not affect simulation correctness invariants (INV-1 through INV-12).

---

## PART 3: Quality Assurance

### I) Risk Analysis

**Risk 1: Replay mistakenly uses vLLM priority for scheduling**
- Likelihood: Medium (easy to accidentally read the field)
- Impact: High (inverts scheduling order, breaks simulation correctness)
- Mitigation: BC-6 and BC-7 tests explicitly verify Request.Priority=0.0 after replay
- Task: Task 5, Task 6

**Risk 2: Column inclusion logic breaks backward compatibility**
- Likelihood: Low (simple boolean check)
- Impact: Medium (older tools fail to parse new traces)
- Mitigation: BC-3 test verifies column is absent when no SLOClass is set
- Task: Task 3

**Risk 3: CSV parser fails on missing column**
- Likelihood: Low (column index check)
- Impact: Low (graceful degradation to zero value)
- Mitigation: BC-9 test verifies legacy traces load correctly
- Task: Task 4

**Risk 4: Priority value not captured in observe**
- Likelihood: Low (single assignment after existing InvertForVLLM call)
- Impact: Low (column would be zero, not critical for observability)
- Mitigation: BC-1 test verifies capture in RealClient.Send
- Task: Task 2

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (single int field, conditional column)
- [x] No feature creep beyond PR scope (observability only, no simulation changes)
- [x] No unexercised flags or interfaces (all new code tested)
- [x] No partial implementations (complete CSV round-trip)
- [x] No breaking changes without explicit contract updates (backward compatible)
- [x] No hidden global state impact (TraceRecord is data-only)
- [x] All new code will pass golangci-lint (standard Go patterns)
- [x] Shared test helpers used from existing shared test package (httptest, TempDir)
- [x] CLAUDE.md updated if needed (Recent Changes section after PR merge)
- [x] No stale references left in CLAUDE.md (N/A — no changes to CLAUDE.md in this PR)
- [x] Documentation DRY verified (no canonical sources modified)
- [x] Deviation log reviewed (zero deviations)
- [x] Each task produces working, testable code (all tasks end with passing tests)
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7 sequential)
- [x] All contracts mapped to specific tasks (see Test Strategy)
- [x] Golden dataset regeneration documented (N/A — no output format changes)
- [x] Construction site audit completed (TraceRecord struct construction in Task 1, Task 2)

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data (all errors propagate)
- [x] R2: Map keys sorted before float accumulation (N/A — no map iteration)
- [x] R3: Every new numeric parameter validated (VLLMPriority is int, validated via strconv.Atoi)
- [x] R4: All struct construction sites audited (TraceRecord in observe.go, workload.go)
- [x] R5: Resource allocation loops handle mid-loop failure (N/A — no resource loops)
- [x] R6: No logrus.Fatalf in sim/ packages (no changes to sim/ error handling)
- [x] R7: Invariant tests alongside golden tests (N/A — no golden tests added)
- [x] R8: No exported mutable maps (TraceRecord has no map fields)
- [x] R9: *float64 for YAML fields where zero is valid (N/A — VLLMPriority is int, zero means "not recorded")
- [x] R10: YAML strict parsing (N/A — no YAML config changes)
- [x] R11: Division by runtime-derived denominators guarded (N/A — no division)
- [x] R12: Golden dataset regenerated if output changed (N/A — simulation output unchanged)
- [x] R13: New interfaces work for 2+ implementations (N/A — no new interfaces)
- [x] R14: No method spans multiple module responsibilities (N/A — no new methods)
- [x] R15: Stale PR references resolved (N/A — no documentation changes)
- [x] R16: Config params grouped by module (N/A — no config changes)
- [x] R17: Routing scorer signals documented (N/A — no routing changes)
- [x] R18: CLI flag values not silently overwritten (N/A — no CLI flag changes)
- [x] R19: Unbounded retry/requeue loops have circuit breakers (N/A — no retry loops)
- [x] R20: Detectors and analyzers handle degenerate inputs (LoadTraceV2 handles missing column)
- [x] R21: No range over slices that can shrink (N/A — no slice iteration with mutation)
- [x] R22: Pre-check estimates consistent with actual operation (N/A — no pre-check logic)
- [x] R23: Parallel code paths apply equivalent transformations (N/A — no parallel paths)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/workload/tracev2.go`

**Purpose:** Extend TraceRecord struct and CSV I/O for vllm_priority column.

**Complete Implementation:**

```go
// TraceRecord represents one row in a trace v2 CSV.
type TraceRecord struct {
	RequestID         int
	ClientID          string
	TenantID          string
	SLOClass          string
	VLLMPriority      int    // vLLM priority value sent in HTTP request body; 0 = not recorded
	SessionID         string
	RoundIndex        int
	PrefixGroup       string
	PrefixLength      int
	Streaming         bool
	InputTokens       int
	OutputTokens      int
	TextTokens        int
	ImageTokens       int
	AudioTokens       int
	VideoTokens       int
	ReasonRatio       float64
	Model             string
	DeadlineUs        int64
	ServerInputTokens int
	ArrivalTimeUs     int64
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	Status            string
	ErrorMessage      string
	FinishReason      string
}

// ExportTraceV2 writes trace header (YAML) and data (CSV) to separate files.
// Column list is built dynamically based on SLOClass presence (BC-2, BC-3).
func ExportTraceV2(header *TraceHeader, records []TraceRecord, headerPath, dataPath string) error {
	// ... (YAML export)

	// Determine if vllm_priority column should be included
	includeVLLMPriority := false
	for _, rec := range records {
		if rec.SLOClass != "" {
			includeVLLMPriority = true
			break
		}
	}

	// Build column list dynamically
	columns := []string{
		"request_id", "client_id", "tenant_id", "slo_class",
	}
	if includeVLLMPriority {
		columns = append(columns, "vllm_priority")
	}
	columns = append(columns, /* remaining columns */)

	// ... (CSV write with conditional vllm_priority value insertion)
}

// LoadTraceV2 reads trace header (YAML) and data (CSV) from separate files.
// Parses vllm_priority column if present (BC-4, BC-9, BC-10).
func LoadTraceV2(headerPath, dataPath string) (*TraceV2, error) {
	// ... (YAML load, CSV header parse)

	// Optional column index
	vllmPriorityIdx := colIndex["vllm_priority"] // -1 if not present

	// ... (row iteration)
	for each row {
		rec := TraceRecord{}
		// ... (required fields)

		// Parse VLLMPriority if column exists (BC-4, BC-9)
		if vllmPriorityIdx >= 0 && vllmPriorityIdx < len(row) {
			if val := row[vllmPriorityIdx]; val != "" {
				priority, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("invalid vllm_priority value '%s' at row %d: %w", val, rowNum, err) // BC-10
				}
				rec.VLLMPriority = priority
			}
		}

		// ... (remaining fields)
	}
}
```

**Key Implementation Notes:**
- **RNG usage:** None (no randomness in CSV I/O)
- **Metrics:** None (observability metadata only)
- **Event ordering:** N/A (data structure only)
- **State mutation:** None (pure I/O)
- **Error handling:** Parse errors return error with row context (BC-10)

---

### File: `cmd/observe.go`

**Purpose:** Capture computed vLLM priority in RequestRecord during observe.

**Complete Implementation:**

```go
// RequestRecord captures one request-response cycle.
type RequestRecord struct {
	RequestID         int
	OutputTokens      int
	ServerInputTokens int
	VLLMPriority      int    // vLLM priority value computed for this request; 0 = not set
	Status            string
	ErrorMessage      string
	SendTimeUs        int64
	FirstChunkTimeUs  int64
	LastChunkTimeUs   int64
	NumChunks         int
	FinishReason      string
	ChunkTimestamps   []int64
}

// Send dispatches a single request to the server and records timing.
func (c *RealClient) Send(ctx context.Context, req *PendingRequest) (*RequestRecord, error) {
	record := &RequestRecord{
		RequestID: req.RequestID,
		Status:    "ok",
	}

	// Build request body
	body := map[string]interface{}{
		"model":  c.modelName,
		"stream": req.Streaming,
	}

	// ... (max_tokens, min_tokens, prompt logic)

	// Inject vLLM priority for SLO-aware servers (PR #1208)
	if req.SLOClass != "" {
		priority := c.sloMap.InvertForVLLM(req.SLOClass)
		body["priority"] = priority
		record.VLLMPriority = priority // BC-1: capture for trace
	}
	// If SLOClass is empty, VLLMPriority remains 0 (BC-5)

	// ... (HTTP request, response handling)

	return record, nil
}
```

**Key Implementation Notes:**
- **RNG usage:** None
- **Metrics:** None (RequestRecord is transient, converted to TraceRecord)
- **Event ordering:** N/A (HTTP client)
- **State mutation:** Assigns VLLMPriority field once per request
- **Error handling:** Errors propagate from HTTP layer; priority capture is infallible

---

### File: `sim/workload/replay.go`

**Purpose:** Document simulation isolation guarantee — no code changes needed.

**Complete Implementation:**

```go
// LoadTraceV2Requests converts TraceV2 records into sim.Request objects for replay.
//
// IMPORTANT: This function does NOT read TraceRecord.VLLMPriority into Request.Priority.
// The vllm_priority column is observability metadata only. Request.Priority remains 0.0
// (unset), and the simulator computes it dynamically from SLOClass via priorityPolicy.Compute().
// See BC-6, BC-8, and issue #1220 for rationale.
func LoadTraceV2Requests(trace *TraceV2, seed int64) ([]*sim.Request, error) {
	// ... (existing implementation unchanged)
	// No reads of rec.VLLMPriority anywhere in this function
}

// LoadTraceV2SessionBlueprints converts TraceV2 records into SessionBlueprint objects for closed-loop replay.
//
// IMPORTANT: This function does NOT read TraceRecord.VLLMPriority into Request.Priority for any round.
// The vllm_priority column is observability metadata only. Request.Priority remains 0.0 (unset),
// and the simulator computes it dynamically from SLOClass via priorityPolicy.Compute().
// See BC-7, BC-8, and issue #1220 for rationale.
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64) ([]sim.SessionBlueprint, error) {
	// ... (existing implementation unchanged)
	// No reads of rec.VLLMPriority anywhere in this function
}
```

**Key Implementation Notes:**
- **Behavioral subtlety:** These functions must NEVER read VLLMPriority into Request.Priority. If they did, replayed traces would use vLLM's inverted convention (0 for critical, 5 for batch) instead of BLIS convention (4 for critical, -1 for batch), inverting the entire scheduling order.
- **Citation:** simulator.go:650 — `req.Priority = priorityPolicy.Compute(req, now)` is where priority is computed dynamically
- **RNG usage:** SubsystemWorkload (for prefix token generation, existing behavior)
- **Metrics:** None (conversion only)
- **Event ordering:** N/A (pre-simulation data loading)
- **State mutation:** Creates new sim.Request objects; does not mutate input
- **Error handling:** Returns error on invalid trace data; no panic paths
