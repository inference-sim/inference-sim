package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestTraceExportOnce_PreventsDuplicateExport verifies that traceExportOnce (package-level)
// ensures exactly-once export behavior when the export logic is invoked multiple times.
// BC: If runObserve is called twice in the same process (the bug scenario from #1047),
// the trace file is written exactly once, preventing file corruption from os.Create truncation.
func TestTraceExportOnce_PreventsDuplicateExport(t *testing.T) {
	// Reset package-level Once guard to simulate a fresh process.
	// In production, each CLI invocation gets a fresh process with fresh Once guards.
	traceExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	// Create a real Recorder with minimal data (production type from cmd/observe.go)
	recorder := &Recorder{}
	recorder.RecordRequest(
		&PendingRequest{RequestID: 1, InputTokens: 10, MaxOutputTokens: 20, Streaming: true},
		&RequestRecord{RequestID: 1, OutputTokens: 15, Status: "ok", SendTimeUs: 1000, FirstChunkTimeUs: 2000, LastChunkTimeUs: 3000, NumChunks: 15},
		1000, "", 0,
	)

	header := &workload.TraceHeader{
		Version:  2,
		TimeUnit: "us",
		Mode:     "real",
	}

	// Track how many times the export closure executes
	var exportCallCount int

	// First invocation: should execute the closure and write the file
	var firstErr error
	traceExportOnce.Do(func() {
		exportCallCount++
		firstErr = recorder.Export(header, headerPath, dataPath)
	})
	if firstErr != nil {
		t.Fatalf("first export failed: %v", firstErr)
	}
	if exportCallCount != 1 {
		t.Fatalf("expected export closure to run once, ran %d times", exportCallCount)
	}

	// Verify file was written
	dataContent1, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("cannot read trace data after first export: %v", err)
	}
	if len(dataContent1) == 0 {
		t.Fatal("trace data file is empty after first export")
	}

	// Second invocation: should skip the closure (sync.Once guarantees this)
	traceExportOnce.Do(func() {
		exportCallCount++
		t.Fatal("traceExportOnce.Do closure executed on second call — sync.Once failed")
	})
	if exportCallCount != 1 {
		t.Errorf("export closure ran %d times after second Do(), expected 1", exportCallCount)
	}

	// Verify file content is unchanged (no truncation from os.Create on second call)
	dataContent2, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("cannot read trace data after second export attempt: %v", err)
	}
	if string(dataContent2) != string(dataContent1) {
		t.Errorf("trace data file changed after second export attempt\nwant length: %d\ngot length:  %d",
			len(dataContent1), len(dataContent2))
	}
}

// TestITLExportOnce_PreventsDuplicateExport verifies that itlExportOnce (package-level)
// ensures exactly-once export behavior for ITL data.
// BC: If runObserve is called twice, ITL CSV is written exactly once, preventing corruption.
func TestITLExportOnce_PreventsDuplicateExport(t *testing.T) {
	// Reset package-level Once guard
	itlExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	itlPath := filepath.Join(tmpDir, "trace.itl.csv")

	// Create a real Recorder with ITL records (production type from cmd/observe.go)
	recorder := &Recorder{}
	recorder.RecordITL(1, []int64{1000, 1100, 1200})

	// Track closure execution
	var exportCallCount int

	// First invocation: should execute
	var firstErr error
	itlExportOnce.Do(func() {
		exportCallCount++
		firstErr = recorder.ExportITL(itlPath)
	})
	if firstErr != nil {
		t.Fatalf("first ITL export failed: %v", firstErr)
	}
	if exportCallCount != 1 {
		t.Fatalf("expected ITL export closure to run once, ran %d times", exportCallCount)
	}

	// Verify file was written
	itlContent1, err := os.ReadFile(itlPath)
	if err != nil {
		t.Fatalf("cannot read ITL data after first export: %v", err)
	}
	if len(itlContent1) == 0 {
		t.Fatal("ITL data file is empty after first export")
	}

	// Second invocation: should be skipped
	itlExportOnce.Do(func() {
		exportCallCount++
		t.Fatal("itlExportOnce.Do closure executed on second call — sync.Once failed")
	})
	if exportCallCount != 1 {
		t.Errorf("ITL export closure ran %d times after second Do(), expected 1", exportCallCount)
	}

	// Verify file content is unchanged
	itlContent2, err := os.ReadFile(itlPath)
	if err != nil {
		t.Fatalf("cannot read ITL data after second export attempt: %v", err)
	}
	if string(itlContent2) != string(itlContent1) {
		t.Errorf("ITL data file changed after second export attempt\nwant length: %d\ngot length:  %d",
			len(itlContent1), len(itlContent2))
	}
}

// TestTraceExport_FileIntegrityAfterDoubleInvocation verifies that when the export
// logic is invoked twice (simulating the double-invocation bug from #1047), the resulting
// file contains valid, non-corrupted data with no interleaved writes or corrupted CSV lines.
// BC: No corrupted lines like "76195059951146" (missing leading digits) from the original bug.
func TestTraceExport_FileIntegrityAfterDoubleInvocation(t *testing.T) {
	// Reset guard
	traceExportOnce = sync.Once{}

	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "trace.yaml")
	dataPath := filepath.Join(tmpDir, "trace.csv")

	// Create recorder with multiple records (production type)
	recorder := &Recorder{}
	recorder.RecordRequest(
		&PendingRequest{RequestID: 1, InputTokens: 100, MaxOutputTokens: 200, Streaming: true},
		&RequestRecord{RequestID: 1, OutputTokens: 150, Status: "ok", SendTimeUs: 10000, FirstChunkTimeUs: 20000, LastChunkTimeUs: 30000, NumChunks: 150},
		10000, "", 0,
	)
	recorder.RecordRequest(
		&PendingRequest{RequestID: 2, InputTokens: 150, MaxOutputTokens: 250, Streaming: true},
		&RequestRecord{RequestID: 2, OutputTokens: 200, Status: "ok", SendTimeUs: 11000, FirstChunkTimeUs: 21000, LastChunkTimeUs: 31000, NumChunks: 200},
		11000, "", 0,
	)

	header := &workload.TraceHeader{
		Version:  2,
		TimeUnit: "us",
		Mode:     "real",
	}

	// Simulate double-invocation: both calls attempt export
	var err1 error
	traceExportOnce.Do(func() {
		err1 = recorder.Export(header, headerPath, dataPath)
	})
	// Second call: closure should not execute
	traceExportOnce.Do(func() {
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

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) < 3 { // header + 2 data rows
		t.Fatalf("expected at least 3 lines (header + 2 records), got %d", len(lines))
	}

	// BC: No corrupted lines (like the "76195059951146" missing-digits bug)
	for i, line := range lines {
		if i == 0 {
			continue // skip header
		}
		fields := strings.Split(line, ",")
		if len(fields) < 10 {
			t.Errorf("line %d has only %d fields, expected at least 10: %q", i, len(fields), line)
		}
		// Verify first field (request_id) is a valid integer (not corrupted)
		requestIDField := fields[0]
		if requestIDField != "1" && requestIDField != "2" {
			t.Errorf("line %d: corrupted or unexpected request_id field: %q (full line: %q)",
				i, requestIDField, line)
		}
	}
}

// TestExportErrorHandling_FirstInvocationFailure verifies that errors from the export
// function are correctly captured and propagated when the export fails on first invocation.
func TestExportErrorHandling_FirstInvocationFailure(t *testing.T) {
	// Reset guard
	traceExportOnce = sync.Once{}

	// Use an invalid path to trigger an export error
	recorder := &Recorder{}
	recorder.RecordRequest(
		&PendingRequest{RequestID: 1, InputTokens: 10, MaxOutputTokens: 20, Streaming: true},
		&RequestRecord{RequestID: 1, OutputTokens: 15, Status: "ok", SendTimeUs: 1000, FirstChunkTimeUs: 2000, LastChunkTimeUs: 3000, NumChunks: 15},
		1000, "", 0,
	)

	header := &workload.TraceHeader{
		Version:  2,
		TimeUnit: "us",
		Mode:     "real",
	}

	// Invalid path: directory doesn't exist
	invalidPath := "/nonexistent/directory/trace.csv"

	var exportErr error
	traceExportOnce.Do(func() {
		exportErr = recorder.Export(header, "/nonexistent/header.yaml", invalidPath)
	})

	// Error should be captured
	if exportErr == nil {
		t.Fatal("expected export to fail with invalid path, but got nil error")
	}
	if !strings.Contains(exportErr.Error(), "no such file or directory") &&
		!strings.Contains(exportErr.Error(), "cannot find the path") {
		t.Errorf("unexpected error message: %v", exportErr)
	}
}
