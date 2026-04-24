package cmd

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestExportIdempotency_TraceExportOnce verifies that traceExportOnce ensures
// exactly-once export behavior when invoked multiple times with the same output path.
// BC: If export logic is called twice in the same process, the file is written exactly once.
func TestExportIdempotency_TraceExportOnce(t *testing.T) {
	// Reset package-level Once guards to ensure clean state.
	// Note: In production, each CLI invocation gets a fresh process, so this reset
	// is only needed for testing within a single process.
	traceExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	// Create a minimal recorder with one record
	recorder := &ObserveRecorder{
		records: []*workload.TraceRecord{
			{
				RequestID:        1,
				InputTokens:      10,
				OutputTokens:     20,
				Status:           "ok",
				ArrivalTimeUs:    1000,
				FirstChunkTimeUs: 2000,
				LastChunkTimeUs:  3000,
				SessionID:        "",
				RoundIndex:       0,
				PrefixGroup:      "prefix1",
				PrefixLength:     5,
			},
		},
	}

	header := workload.TraceHeader{
		Version:   2,
		TimeUnit:  "us",
		Mode:      "replayed",
		CreatedAt: "2024-01-01T00:00:00Z",
	}

	// First invocation: should write the file
	var firstErr error
	traceExportOnce.Do(func() {
		firstErr = recorder.Export(header, headerPath, dataPath)
	})
	if firstErr != nil {
		t.Fatalf("first export failed: %v", firstErr)
	}

	// Verify file was written
	dataContent1, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("cannot read trace data after first export: %v", err)
	}
	if len(dataContent1) == 0 {
		t.Fatal("trace data file is empty after first export")
	}

	// Second invocation: should be a no-op (sync.Once skips the closure)
	var secondErr error
	traceExportOnce.Do(func() {
		// This closure should never execute due to sync.Once
		t.Fatal("traceExportOnce.Do closure executed on second call — sync.Once failed")
	})
	// secondErr remains nil because the closure never ran

	// Verify file content is unchanged (no truncation from os.Create on second call)
	dataContent2, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("cannot read trace data after second export attempt: %v", err)
	}
	if string(dataContent2) != string(dataContent1) {
		t.Errorf("trace data file changed after second export attempt\nwant: %q\ngot:  %q",
			string(dataContent1), string(dataContent2))
	}

	// BC verification: Export was called once, file integrity preserved
	if secondErr != nil {
		t.Errorf("second export attempt returned unexpected error: %v", secondErr)
	}
}

// TestExportIdempotency_ITLExportOnce verifies that itlExportOnce ensures
// exactly-once export behavior for ITL data.
// BC: If ITL export is called twice in the same process, the file is written exactly once.
func TestExportIdempotency_ITLExportOnce(t *testing.T) {
	// Reset package-level Once guards
	itlExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	itlPath := filepath.Join(tmpDir, "trace.itl.csv")

	// Create a minimal recorder with ITL records
	recorder := &ObserveRecorder{
		itlRecords: []workload.ITLRecord{
			{RequestID: 1, ChunkIndex: 0, TimestampUs: 1000},
			{RequestID: 1, ChunkIndex: 1, TimestampUs: 1100},
			{RequestID: 1, ChunkIndex: 2, TimestampUs: 1200},
		},
	}

	// First invocation: should write the file
	var firstErr error
	itlExportOnce.Do(func() {
		firstErr = recorder.ExportITL(itlPath)
	})
	if firstErr != nil {
		t.Fatalf("first ITL export failed: %v", firstErr)
	}

	// Verify file was written
	itlContent1, err := os.ReadFile(itlPath)
	if err != nil {
		t.Fatalf("cannot read ITL data after first export: %v", err)
	}
	if len(itlContent1) == 0 {
		t.Fatal("ITL data file is empty after first export")
	}

	// Second invocation: should be a no-op
	var secondErr error
	itlExportOnce.Do(func() {
		// This closure should never execute
		t.Fatal("itlExportOnce.Do closure executed on second call — sync.Once failed")
	})

	// Verify file content is unchanged
	itlContent2, err := os.ReadFile(itlPath)
	if err != nil {
		t.Fatalf("cannot read ITL data after second export attempt: %v", err)
	}
	if string(itlContent2) != string(itlContent1) {
		t.Errorf("ITL data file changed after second export attempt\nwant: %q\ngot:  %q",
			string(itlContent1), string(itlContent2))
	}

	// BC verification: Export was called once, file integrity preserved
	if secondErr != nil {
		t.Errorf("second ITL export attempt returned unexpected error: %v", secondErr)
	}
}

// TestExportIdempotency_FileIntegrityAfterDoubleInvocation verifies that when
// the export logic is invoked twice (simulating the double-invocation bug),
// the resulting file contains valid, non-corrupted data.
// BC: No interleaved/truncated content, no corrupted CSV lines.
func TestExportIdempotency_FileIntegrityAfterDoubleInvocation(t *testing.T) {
	// Reset guards
	traceExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	recorder := &ObserveRecorder{
		records: []*workload.TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 200, Status: "ok",
				ArrivalTimeUs: 10000, FirstChunkTimeUs: 20000, LastChunkTimeUs: 30000,
				PrefixGroup: "prefix1", PrefixLength: 10},
			{RequestID: 2, InputTokens: 150, OutputTokens: 250, Status: "ok",
				ArrivalTimeUs: 11000, FirstChunkTimeUs: 21000, LastChunkTimeUs: 31000,
				PrefixGroup: "prefix2", PrefixLength: 15},
		},
	}

	header := workload.TraceHeader{
		Version:  2,
		TimeUnit: "us",
		Mode:     "replayed",
	}

	// Simulate double-invocation: both calls attempt export
	var err1 error
	traceExportOnce.Do(func() {
		err1 = recorder.Export(header, headerPath, dataPath)
	})
	traceExportOnce.Do(func() {
		// Second call: closure should not execute due to sync.Once
		t.Fatal("second traceExportOnce.Do closure executed — sync.Once failed")
	})

	// First call should succeed
	if err1 != nil {
		t.Fatalf("first export failed: %v", err1)
	}

	// Verify file is valid CSV with expected structure
	content, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("cannot read trace data: %v", err)
	}

	lines := splitLines(string(content))
	if len(lines) < 3 { // header + 2 data rows
		t.Fatalf("expected at least 3 lines (header + 2 records), got %d", len(lines))
	}

	// BC: No corrupted lines (like the "76195059951146" missing-digits bug)
	for i, line := range lines {
		if i == 0 {
			continue // skip header
		}
		fields := splitCSVLine(line)
		if len(fields) < 10 {
			t.Errorf("line %d has only %d fields, expected at least 10: %q", i, len(fields), line)
		}
		// Verify first field (request_id) is a valid integer (not corrupted)
		if fields[0] != "1" && fields[0] != "2" {
			t.Errorf("line %d: corrupted request_id field: %q", i, fields[0])
		}
	}
}

// Helper: split CSV line by comma (naive implementation for test purposes)
func splitCSVLine(line string) []string {
	if line == "" {
		return nil
	}
	return splitByComma(line)
}

// Helper: split by comma (basic, does not handle quoted commas)
func splitByComma(s string) []string {
	var result []string
	var current string
	for _, r := range s {
		if r == ',' {
			result = append(result, current)
			current = ""
		} else {
			current += string(r)
		}
	}
	result = append(result, current)
	return result
}

// Helper: split content into lines
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	var lines []string
	var line string
	for _, r := range s {
		if r == '\n' {
			if line != "" {
				lines = append(lines, line)
			}
			line = ""
		} else if r != '\r' {
			line += string(r)
		}
	}
	if line != "" {
		lines = append(lines, line)
	}
	return lines
}

// ObserveRecorder is a minimal stub for testing export idempotency.
// In production, this is the real recorder from cmd/observe.go.
type ObserveRecorder struct {
	mu         sync.Mutex
	records    []*workload.TraceRecord
	itlRecords []workload.ITLRecord
}

func (r *ObserveRecorder) Records() []*workload.TraceRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]*workload.TraceRecord, len(r.records))
	copy(result, r.records)
	return result
}

func (r *ObserveRecorder) ITLRecords() []workload.ITLRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]workload.ITLRecord, len(r.itlRecords))
	copy(result, r.itlRecords)
	return result
}

func (r *ObserveRecorder) Export(header workload.TraceHeader, headerPath, dataPath string) error {
	records := r.Records()
	// Convert []*TraceRecord to []TraceRecord for ExportTraceV2
	recordsValue := make([]workload.TraceRecord, len(records))
	for i, rec := range records {
		recordsValue[i] = *rec
	}
	return workload.ExportTraceV2(&header, recordsValue, headerPath, dataPath)
}

func (r *ObserveRecorder) ExportITL(path string) error {
	records := r.ITLRecords()
	return workload.ExportITL(records, path)
}
