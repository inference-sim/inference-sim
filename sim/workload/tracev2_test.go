package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTraceV2_RoundTrip_PreservesAllFields(t *testing.T) {
	header := &TraceHeader{
		Version: 2, TimeUnit: "microseconds", Mode: "generated",
		WarmUpRequests: 5, WorkloadSpec: "test.yaml",
	}
	records := []TraceRecord{
		{RequestID: 0, ClientID: "c1", TenantID: "t1", SLOClass: "batch",
			InputTokens: 512, OutputTokens: 128, ArrivalTimeUs: 0,
			Status: "ok"},
		{RequestID: 1, ClientID: "c2", TenantID: "t2", SLOClass: "realtime",
			InputTokens: 256, OutputTokens: 64, ArrivalTimeUs: 100000,
			SendTimeUs: 100010, FirstChunkTimeUs: 100800, LastChunkTimeUs: 101500,
			NumChunks: 5, Streaming: true, Status: "ok"},
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

	if loaded.Header.Version != 2 {
		t.Errorf("version = %d, want 2", loaded.Header.Version)
	}
	if loaded.Header.WarmUpRequests != 5 {
		t.Errorf("warm_up = %d, want 5", loaded.Header.WarmUpRequests)
	}
	if len(loaded.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(loaded.Records))
	}

	r0 := loaded.Records[0]
	if r0.RequestID != 0 || r0.ClientID != "c1" || r0.TenantID != "t1" || r0.SLOClass != "batch" {
		t.Errorf("record 0 metadata mismatch")
	}
	if r0.InputTokens != 512 || r0.OutputTokens != 128 {
		t.Errorf("record 0 tokens: input=%d output=%d, want 512/128", r0.InputTokens, r0.OutputTokens)
	}

	r1 := loaded.Records[1]
	if r1.ArrivalTimeUs != 100000 {
		t.Errorf("record 1 arrival = %d, want 100000", r1.ArrivalTimeUs)
	}
	if r1.SendTimeUs != 100010 {
		t.Errorf("record 1 send = %d, want 100010", r1.SendTimeUs)
	}
	if r1.FirstChunkTimeUs != 100800 {
		t.Errorf("record 1 first_chunk = %d, want 100800", r1.FirstChunkTimeUs)
	}
	if !r1.Streaming {
		t.Error("record 1 should be streaming")
	}
	if r1.NumChunks != 5 {
		t.Errorf("record 1 chunks = %d, want 5", r1.NumChunks)
	}
}

func TestTraceV2_RoundTrip_WithServerConfig(t *testing.T) {
	header := &TraceHeader{
		Version: 2, TimeUnit: "microseconds", Mode: "real",
		Server: &TraceServerConfig{
			Type: "vllm", Model: "llama-8b", TensorParallel: 1,
			MaxNumSeqs: 256, BlockSize: 16,
		},
		Network: &TraceNetworkConfig{MeasuredRTTMs: 2.5},
	}

	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	if err := ExportTraceV2(header, nil, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}

	if loaded.Header.Server == nil {
		t.Fatal("server config should be present")
	}
	if loaded.Header.Server.MaxNumSeqs != 256 {
		t.Errorf("max_num_seqs = %d, want 256", loaded.Header.Server.MaxNumSeqs)
	}
	if loaded.Header.Network == nil || loaded.Header.Network.MeasuredRTTMs != 2.5 {
		t.Error("network config mismatch")
	}
}

func TestTraceV2_IntegerTimestamps_Preserved(t *testing.T) {
	// Large timestamp values should not lose precision
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "real"}
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 1708100000000000, // epoch-like value
			SendTimeUs: 1708100000000010, FirstChunkTimeUs: 1708100000045200,
			LastChunkTimeUs: 1708100001590300, Status: "ok"},
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

	r := loaded.Records[0]
	if r.ArrivalTimeUs != 1708100000000000 {
		t.Errorf("arrival = %d, want 1708100000000000 (precision lost)", r.ArrivalTimeUs)
	}
	if r.SendTimeUs != 1708100000000010 {
		t.Errorf("send = %d, want 1708100000000010", r.SendTimeUs)
	}
}

// TestLoadTraceV2_UnknownYAMLField_ReturnsError verifies BC-11: strict YAML parsing.
func TestLoadTraceV2_UnknownYAMLField_ReturnsError(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")

	// GIVEN a header with a typo
	if err := os.WriteFile(headerPath, []byte("tme_unit: microseconds\ntrace_version: 2\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dataPath, []byte("request_id\n1\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN loading
	_, err := LoadTraceV2(headerPath, dataPath)

	// THEN error about unknown field
	if err == nil {
		t.Fatal("expected error for unknown YAML field, got nil")
	}
	if !strings.Contains(err.Error(), "tme_unit") {
		t.Errorf("error should mention unknown field 'tme_unit', got: %s", err.Error())
	}
}

// TestParseTraceRecord_InvalidInteger_ReturnsError verifies BC-12: CSV error propagation.
func TestParseTraceRecord_InvalidInteger_ReturnsError(t *testing.T) {
	// GIVEN a row with a non-numeric request_id
	row := make([]string, 22)
	row[0] = "abc" // request_id should be integer
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
