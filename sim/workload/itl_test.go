package workload_test

import (
	"os"
	"path/filepath"
	"strings"
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
	if !strings.Contains(err.Error(), "timestamp_us") {
		t.Errorf("error message should mention 'timestamp_us', got: %v", err)
	}
}
