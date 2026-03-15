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
		{RequestID: 1, ClientID: "c2", TenantID: "t2", SLOClass: "critical",
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
			Model:             "",  // zero value: default model
			DeadlineUs:        0,   // zero value: no timeout
			ServerInputTokens: 0,  // zero value: not recorded
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

// TestParseTraceRecord_NegativeDeadlineUs_ReturnsError verifies R3 validation.
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

// TestParseTraceRecord_NegativeInputTokens_ReturnsError verifies R3 for
// input_tokens (prevents make([]int, negative) panic in replay).
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

// TestParseTraceRecord_NegativeOutputTokens_ReturnsError verifies R3 for
// output_tokens (prevents make([]int, negative) panic in replay).
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

// TestParseTraceRecord_NegativeServerInputTokens_ReturnsError verifies R3
// for server_input_tokens (consistent validation for all token count fields).
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
	row[16] = "1000" // deadline_us = 1000
	row[18] = "5000" // arrival_time_us = 5000 (deadline < arrival)

	_, err := parseTraceRecord(row)

	if err == nil {
		t.Fatal("expected error for deadline before arrival, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}
