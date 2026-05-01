package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
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
		{
			RequestID:         2,
			Model:             "org/model,with-comma", // CSV-special: encoding/csv quotes automatically
			DeadlineUs:        9000, // > ArrivalTimeUs (6000) — valid per cross-field invariant
			ServerInputTokens: 0,
			InputTokens:       64,
			OutputTokens:      16,
			ArrivalTimeUs:     6000,
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
	if len(loaded.Records) != 3 {
		t.Fatalf("expected 3 records, got %d", len(loaded.Records))
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

	r2 := loaded.Records[2]
	// CSV-special chars: encoding/csv quotes fields containing commas automatically
	if r2.Model != "org/model,with-comma" {
		t.Errorf("record 2 model = %q, want %q (CSV quoting must preserve commas)", r2.Model, "org/model,with-comma")
	}
	if r2.DeadlineUs != 9000 {
		t.Errorf("record 2 deadline_us = %d, want 9000", r2.DeadlineUs)
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
	row := make([]string, 27) // must match current column count
	row[0] = "abc"             // request_id should be integer
	for i := 1; i < len(row); i++ {
		row[i] = "0"
	}

	// WHEN parsing
	_, err := parseTraceRecord(row, false)

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
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "not_a_number" // deadline_us column (shifted +1 by prefix_length)

	_, err := parseTraceRecord(row, false)

	if err == nil {
		t.Fatal("expected error for non-numeric deadline_us, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}

// TestParseTraceRecord_InvalidServerInputTokens_ReturnsError verifies BC-10.
func TestParseTraceRecord_InvalidServerInputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[18] = "not_a_number" // server_input_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false)

	if err == nil {
		t.Fatal("expected error for non-numeric server_input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "server_input_tokens") {
		t.Errorf("error should mention 'server_input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_InvalidVLLMPriority_ReturnsError verifies that non-numeric vllm_priority values are rejected.
func TestParseTraceRecord_InvalidVLLMPriority_ReturnsError(t *testing.T) {
	// GIVEN a row with vllm_priority column present and set to non-numeric value
	row := make([]string, 28) // 28 columns when vllm_priority is present
	for i := range row {
		row[i] = "0"
	}
	row[4] = "not_a_number" // vllm_priority column (index 4)

	// WHEN parsing with hasVLLMPriority=true
	_, err := parseTraceRecord(row, true)

	// THEN error about invalid value
	if err == nil {
		t.Fatal("expected error for non-numeric vllm_priority, got nil")
	}
	if !strings.Contains(err.Error(), "vllm_priority") {
		t.Errorf("error should mention 'vllm_priority', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeVLLMPriority_ReturnsError verifies that negative vllm_priority values are rejected.
func TestParseTraceRecord_NegativeVLLMPriority_ReturnsError(t *testing.T) {
	// GIVEN a row with vllm_priority column present and set to negative value
	row := make([]string, 28) // 28 columns when vllm_priority is present
	for i := range row {
		row[i] = "0"
	}
	row[4] = "-1" // negative vllm_priority

	// WHEN parsing with hasVLLMPriority=true
	_, err := parseTraceRecord(row, true)

	// THEN error about negative value
	if err == nil {
		t.Fatal("expected error for negative vllm_priority, got nil")
	}
	if !strings.Contains(err.Error(), "vllm_priority") {
		t.Errorf("error should mention 'vllm_priority', got: %s", err.Error())
	}
	if !strings.Contains(err.Error(), "negative") {
		t.Errorf("error should mention 'negative', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeDeadlineUs_ReturnsError verifies R3 validation.
func TestParseTraceRecord_NegativeDeadlineUs_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "-1" // negative deadline_us (shifted +1)

	_, err := parseTraceRecord(row, false)

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
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[9] = "-1" // input_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false)

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
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[10] = "-1" // output_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false)

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
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[18] = "-1" // server_input_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false)

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
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "1000" // deadline_us = 1000 (shifted +1)
	row[19] = "5000" // arrival_time_us = 5000 (shifted +1)

	_, err := parseTraceRecord(row, false)

	if err == nil {
		t.Fatal("expected error for deadline before arrival, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}

// TestParseTraceRecord_InvalidReasonRatio_ReturnsError verifies R3 for
// reason_ratio: NaN, Inf, negative, and > 1.0 are all rejected.
func TestParseTraceRecord_InvalidReasonRatio_ReturnsError(t *testing.T) {
	cases := []struct {
		value string
	}{
		{"NaN"},
		{"+Inf"},
		{"-Inf"},
		{"-0.1"},
		{"1.5"},
	}
	for _, tc := range cases {
		row := make([]string, 27)
		for i := range row {
			row[i] = "0"
		}
		row[15] = tc.value // reason_ratio column (shifted +1)

		_, err := parseTraceRecord(row, false)

		if err == nil {
			t.Errorf("reason_ratio=%q: expected error, got nil", tc.value)
			continue
		}
		if !strings.Contains(err.Error(), "reason_ratio") {
			t.Errorf("reason_ratio=%q: error should mention 'reason_ratio', got: %s", tc.value, err.Error())
		}
	}
}

func TestTraceV2_FinishReason_RoundTrip(t *testing.T) {
	header := &TraceHeader{Version: 1, TimeUnit: "us", Mode: "real"}
	records := []TraceRecord{{
		RequestID:    1,
		InputTokens:  10,
		OutputTokens: 5,
		ArrivalTimeUs: 1000,
		SendTimeUs:    2000,
		Status:        "ok",
		FinishReason:  "stop",
	}}

	headerPath := filepath.Join(t.TempDir(), "h.yaml")
	dataPath := filepath.Join(t.TempDir(), "d.csv")
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}

	tv2, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(tv2.Records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(tv2.Records))
	}
	if tv2.Records[0].FinishReason != "stop" {
		t.Errorf("FinishReason: got %q, want %q", tv2.Records[0].FinishReason, "stop")
	}
}

func TestTraceV2_PrefixGroup_RoundTrip(t *testing.T) {
	// BC-1: PrefixGroup and PrefixLength survive export → load
	seed := int64(42)
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated", WorkloadSeed: &seed}
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
	// BC-4: WorkloadSeed preserved (pointer type per R9 — seed=0 is valid)
	if loaded.Header.WorkloadSeed == nil || *loaded.Header.WorkloadSeed != 42 {
		t.Errorf("WorkloadSeed = %v, want 42", loaded.Header.WorkloadSeed)
	}
}

// --- RequestsToTraceRecords tests ---

func TestRequestMetadataFields(t *testing.T) {
	req := &sim.Request{
		ID:             "request_0",
		State:          sim.StateCompleted,
		InputTokens:    []int{1, 2, 3},
		OutputTokens:   []int{4, 5},
		ArrivalTime:    1000,
		TTFTSet:        true,
		FirstTokenTime: 500,
		ClientID:       "client-alpha",
		PrefixGroup:    "shared-prefix",
		Streaming:      true,
		TenantID:       "tenant-1",
		SLOClass:       "critical",
		SessionID:      "sess-42",
		RoundIndex:     2,
		Model:          "llama-8b",
	}
	records := RequestsToTraceRecords([]*sim.Request{req})
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]

	// BC-5: ClientID preserved, PrefixGroup preserved, Streaming preserved
	if r.ClientID != "client-alpha" {
		t.Errorf("ClientID: got %q, want %q", r.ClientID, "client-alpha")
	}
	if r.PrefixGroup != "shared-prefix" {
		t.Errorf("PrefixGroup: got %q, want %q", r.PrefixGroup, "shared-prefix")
	}
	if !r.Streaming {
		t.Errorf("Streaming: got %v, want true", r.Streaming)
	}
	if r.TenantID != "tenant-1" {
		t.Errorf("TenantID: got %q, want %q", r.TenantID, "tenant-1")
	}
	if r.SLOClass != "critical" {
		t.Errorf("SLOClass: got %q, want %q", r.SLOClass, "critical")
	}
	if r.SessionID != "sess-42" {
		t.Errorf("SessionID: got %q, want %q", r.SessionID, "sess-42")
	}
	if r.RoundIndex != 2 {
		t.Errorf("RoundIndex: got %d, want 2", r.RoundIndex)
	}
	if r.Model != "llama-8b" {
		t.Errorf("Model: got %q, want %q", r.Model, "llama-8b")
	}
}

func TestRequestsToTraceRecords_FieldMapping(t *testing.T) {
	req := &sim.Request{
		ID:              "request_0",
		State:           sim.StateCompleted,
		InputTokens:     make([]int, 512),
		OutputTokens:    make([]int, 256),
		ProgressIndex:   512 + 256,
		ArrivalTime:     10000,
		TTFTSet:         true,
		FirstTokenTime:  5000,
		ITL:             []int64{100, 200, 300},
		TextTokenCount:  400,
		ImageTokenCount: 50,
		AudioTokenCount: 30,
		VideoTokenCount: 32,
		ReasonRatio:     0.25,
		Deadline:        50000,
	}
	records := RequestsToTraceRecords([]*sim.Request{req})

	// BC-9: record count conservation
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]

	// BC-2: token counts use pre-determined (len of slices)
	if r.InputTokens != 512 {
		t.Errorf("InputTokens: got %d, want 512", r.InputTokens)
	}
	if r.OutputTokens != 256 {
		t.Errorf("OutputTokens: got %d, want 256 (pre-determined)", r.OutputTokens)
	}

	// Multimodal breakdown
	if r.TextTokens != 400 {
		t.Errorf("TextTokens: got %d, want 400", r.TextTokens)
	}
	if r.ImageTokens != 50 {
		t.Errorf("ImageTokens: got %d, want 50", r.ImageTokens)
	}
	if r.AudioTokens != 30 {
		t.Errorf("AudioTokens: got %d, want 30", r.AudioTokens)
	}
	if r.VideoTokens != 32 {
		t.Errorf("VideoTokens: got %d, want 32", r.VideoTokens)
	}
	if r.ReasonRatio != 0.25 {
		t.Errorf("ReasonRatio: got %f, want 0.25", r.ReasonRatio)
	}
	if r.DeadlineUs != 50000 {
		t.Errorf("DeadlineUs: got %d, want 50000", r.DeadlineUs)
	}

	// BC-3: timing (absolute)
	if r.ArrivalTimeUs != 10000 {
		t.Errorf("ArrivalTimeUs: got %d, want 10000", r.ArrivalTimeUs)
	}
	if r.SendTimeUs != 10000 {
		t.Errorf("SendTimeUs: got %d, want 10000 (= ArrivalTime)", r.SendTimeUs)
	}
	// FirstChunkTimeUs = ArrivalTime + FirstTokenTime = 10000 + 5000 = 15000
	if r.FirstChunkTimeUs != 15000 {
		t.Errorf("FirstChunkTimeUs: got %d, want 15000", r.FirstChunkTimeUs)
	}
	// LastChunkTimeUs = ArrivalTime + FirstTokenTime + sum(ITL) = 10000 + 5000 + 600 = 15600
	if r.LastChunkTimeUs != 15600 {
		t.Errorf("LastChunkTimeUs: got %d, want 15600", r.LastChunkTimeUs)
	}

	// Status
	if r.Status != "ok" {
		t.Errorf("Status: got %q, want %q", r.Status, "ok")
	}

	// RequestID = array index
	if r.RequestID != 0 {
		t.Errorf("RequestID: got %d, want 0", r.RequestID)
	}
}

func TestRequestsToTraceRecords_StatusMapping(t *testing.T) {
	cases := []struct {
		state sim.RequestState
		want  string
	}{
		{sim.StateCompleted, "ok"},
		{sim.StateTimedOut, "timeout"},
		{sim.StateQueued, "incomplete"},
		{sim.StateRunning, "incomplete"},
	}
	for _, tc := range cases {
		req := &sim.Request{
			ID:           "request_0",
			State:        tc.state,
			InputTokens:  []int{1},
			OutputTokens: []int{2},
		}
		records := RequestsToTraceRecords([]*sim.Request{req})
		if records[0].Status != tc.want {
			t.Errorf("State=%q: got status %q, want %q", tc.state, records[0].Status, tc.want)
		}
	}
}

func TestRequestsToTraceRecords_TimingCausality(t *testing.T) {
	req := &sim.Request{
		ID:             "request_0",
		State:          sim.StateCompleted,
		InputTokens:    make([]int, 100),
		OutputTokens:   make([]int, 50),
		ArrivalTime:    5000,
		TTFTSet:        true,
		FirstTokenTime: 3000,
		ITL:            []int64{100, 200},
	}
	records := RequestsToTraceRecords([]*sim.Request{req})
	r := records[0]

	// INV-5 projection: FirstChunkTimeUs >= ArrivalTimeUs
	if r.FirstChunkTimeUs < r.ArrivalTimeUs {
		t.Errorf("INV-5 violation: FirstChunkTimeUs (%d) < ArrivalTimeUs (%d)",
			r.FirstChunkTimeUs, r.ArrivalTimeUs)
	}
	// INV-5 projection: LastChunkTimeUs >= FirstChunkTimeUs
	if r.LastChunkTimeUs < r.FirstChunkTimeUs {
		t.Errorf("INV-5 violation: LastChunkTimeUs (%d) < FirstChunkTimeUs (%d)",
			r.LastChunkTimeUs, r.FirstChunkTimeUs)
	}
}

func TestRequestsToTraceRecords_PrefillTimeout(t *testing.T) {
	req := &sim.Request{
		ID:           "request_0",
		State:        sim.StateTimedOut,
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 50),
		ArrivalTime:  5000,
		TTFTSet:      false,
	}
	records := RequestsToTraceRecords([]*sim.Request{req})
	r := records[0]

	if r.FirstChunkTimeUs != 0 {
		t.Errorf("Prefill-timeout FirstChunkTimeUs: got %d, want 0", r.FirstChunkTimeUs)
	}
	if r.LastChunkTimeUs != 0 {
		t.Errorf("Prefill-timeout LastChunkTimeUs: got %d, want 0", r.LastChunkTimeUs)
	}
	if r.Status != "timeout" {
		t.Errorf("Status: got %q, want %q", r.Status, "timeout")
	}
}

func TestRequestsToTraceRecords_RecordCount(t *testing.T) {
	reqs := make([]*sim.Request, 100)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           "request_0",
			State:        sim.StateCompleted,
			InputTokens:  []int{1},
			OutputTokens: []int{2},
		}
	}
	records := RequestsToTraceRecords(reqs)
	if len(records) != 100 {
		t.Errorf("BC-9: got %d records, want 100", len(records))
	}
}

func TestRequestsToTraceRecords_RoundTrip(t *testing.T) {
	reqs := []*sim.Request{
		{
			ID:             "request_0",
			State:          sim.StateCompleted,
			InputTokens:    make([]int, 512),
			OutputTokens:   make([]int, 128),
			ArrivalTime:    1000,
			TTFTSet:        true,
			FirstTokenTime: 500,
			TenantID:       "t1",
			SLOClass:       "batch",
			ClientID:       "c1",
			Streaming:      true,
			Model:          "test-model",
			Deadline:        50000,
		},
		{
			ID:           "request_1",
			State:        sim.StateTimedOut,
			InputTokens:  make([]int, 256),
			OutputTokens: make([]int, 64),
			ArrivalTime:  2000,
			TTFTSet:      false,
			TenantID:     "t2",
			SLOClass:     "critical",
		},
	}

	records := RequestsToTraceRecords(reqs)

	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")

	header := &TraceHeader{
		Version:  2,
		TimeUnit: "microseconds",
		Mode:     "generated",
	}
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}

	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}

	if len(loaded.Records) != len(reqs) {
		t.Fatalf("record count: got %d, want %d", len(loaded.Records), len(reqs))
	}

	lr := loaded.Records[0]
	if lr.InputTokens != 512 {
		t.Errorf("InputTokens: got %d, want 512", lr.InputTokens)
	}
	if lr.OutputTokens != 128 {
		t.Errorf("OutputTokens: got %d, want 128", lr.OutputTokens)
	}
	if lr.ArrivalTimeUs != 1000 {
		t.Errorf("ArrivalTimeUs: got %d, want 1000", lr.ArrivalTimeUs)
	}
	if lr.TenantID != "t1" {
		t.Errorf("TenantID: got %q, want %q", lr.TenantID, "t1")
	}
	if lr.ClientID != "c1" {
		t.Errorf("ClientID: got %q, want %q", lr.ClientID, "c1")
	}
	if !lr.Streaming {
		t.Errorf("Streaming: got %v, want true", lr.Streaming)
	}
	if lr.Model != "test-model" {
		t.Errorf("Model: got %q, want %q", lr.Model, "test-model")
	}
	if lr.DeadlineUs != 50000 {
		t.Errorf("DeadlineUs: got %d, want 50000", lr.DeadlineUs)
	}
	// PrefixGroup preserved (no longer cleared)
	if lr.PrefixGroup != "" {
		t.Errorf("PrefixGroup: got %q, want empty (no prefix set)", lr.PrefixGroup)
	}

	// Verify second record (timed out during prefill)
	lr2 := loaded.Records[1]
	if lr2.Status != "timeout" {
		t.Errorf("Status: got %q, want %q", lr2.Status, "timeout")
	}
	if lr2.FirstChunkTimeUs != 0 {
		t.Errorf("Prefill-timeout FirstChunkTimeUs: got %d, want 0", lr2.FirstChunkTimeUs)
	}
}

// TestRequestsToTraceRecords_VLLMPriority_AlwaysZero verifies that VLLMPriority
// is always 0 in simulation-generated records (simulation isolation boundary).
func TestRequestsToTraceRecords_VLLMPriority_AlwaysZero(t *testing.T) {
	// GIVEN sim.Request instances (simulation-generated, no vLLM priority computed)
	reqs := []*sim.Request{
		{
			ID:            "req1",
			State:         sim.StateCompleted,
			SLOClass:      "critical",
			InputTokens:   make([]int, 10),
			OutputTokens:  make([]int, 5),
			ProgressIndex: 15,
		},
		{
			ID:            "req2",
			State:         sim.StateCompleted,
			SLOClass:      "batch",
			InputTokens:   make([]int, 20),
			OutputTokens:  make([]int, 10),
			ProgressIndex: 30,
		},
		{
			ID:            "req3",
			State:         sim.StateCompleted,
			SLOClass:      "", // no SLO class
			InputTokens:   make([]int, 5),
			OutputTokens:  make([]int, 2),
			ProgressIndex: 7,
		},
	}

	// WHEN converting to TraceRecords
	records := RequestsToTraceRecords(reqs)

	// THEN all VLLMPriority values should be 0 (never computed during simulation)
	for i, rec := range records {
		if rec.VLLMPriority != 0 {
			t.Errorf("record[%d].VLLMPriority: got %d, want 0 (simulation isolation)",
				i, rec.VLLMPriority)
		}
	}
}

func TestTraceRecord_VLLMPriority_FieldExists(t *testing.T) {
	// GIVEN a TraceRecord with VLLMPriority set to a non-zero value
	rec := TraceRecord{
		RequestID:    1,
		SLOClass:     "batch",
		VLLMPriority: 5, // batch → 5 in vLLM convention
	}

	// THEN the field is accessible and has the expected value
	if rec.VLLMPriority != 5 {
		t.Errorf("Expected VLLMPriority=5, got %d", rec.VLLMPriority)
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

func TestExportTraceV2_VLLMPriority_ConditionalColumn(t *testing.T) {
	// BC-3: vllm_priority column included only when priority was actually computed.
	// Distinguishes between:
	// - observe with critical (SLOClass="critical", Mode="real", VLLMPriority=0) → column present
	// - simulation with SLOClass (Mode="synthetic", VLLMPriority=0) → column absent

	dir := t.TempDir()

	t.Run("simulation with SLOClass → no column (Mode=synthetic, all priority=0)", func(t *testing.T) {
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "synthetic"}
		records := []TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000, SLOClass: "critical", VLLMPriority: 0, Status: "ok"},
			{RequestID: 2, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000, SLOClass: "batch", VLLMPriority: 0, Status: "ok"},
		}

		headerPath := filepath.Join(dir, "sim_header.yaml")
		dataPath := filepath.Join(dir, "sim_data.csv")
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatal(err)
		}

		// Read CSV header row
		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		if len(lines) < 1 {
			t.Fatal("CSV file is empty")
		}
		headerRow := lines[0]

		// THEN vllm_priority should NOT be in the header
		if strings.Contains(headerRow, "vllm_priority") {
			t.Errorf("vllm_priority column should not be present in simulation (Mode=synthetic, all priority=0)")
		}
	})

	t.Run("observe all-critical → column present (Mode=real, SLOClass set, priority=0)", func(t *testing.T) {
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
		records := []TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000, SLOClass: "critical", VLLMPriority: 0, Status: "ok"},
			{RequestID: 2, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000, SLOClass: "critical", VLLMPriority: 0, Status: "ok"},
		}

		headerPath := filepath.Join(dir, "critical_header.yaml")
		dataPath := filepath.Join(dir, "critical_data.csv")
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatal(err)
		}

		// Read CSV header row
		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		if len(lines) < 1 {
			t.Fatal("CSV file is empty")
		}
		headerRow := lines[0]

		// THEN vllm_priority SHOULD be in the header (critical=0 is a real priority)
		if !strings.Contains(headerRow, "vllm_priority") {
			t.Errorf("vllm_priority column should be present for observe with critical (Mode=real, SLOClass set)")
		}

		// Verify all values are 0 (critical priority)
		if len(lines) < 3 {
			t.Fatal("Expected at least 2 data rows")
		}
		row1 := strings.Split(lines[1], ",")
		row2 := strings.Split(lines[2], ",")

		// Find vllm_priority column index
		cols := strings.Split(headerRow, ",")
		vllmIdx := -1
		for i, col := range cols {
			if col == "vllm_priority" {
				vllmIdx = i
				break
			}
		}
		if vllmIdx == -1 {
			t.Fatal("vllm_priority column not found")
		}

		if row1[vllmIdx] != "0" {
			t.Errorf("Row 1 vllm_priority: got %q, want \"0\" (critical)", row1[vllmIdx])
		}
		if row2[vllmIdx] != "0" {
			t.Errorf("Row 2 vllm_priority: got %q, want \"0\" (critical)", row2[vllmIdx])
		}
	})

	t.Run("any VLLMPriority!=0 → column included (observe mixed)", func(t *testing.T) {
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
		records := []TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000, SLOClass: "critical", VLLMPriority: 0, Status: "ok"},
			{RequestID: 2, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000, SLOClass: "batch", VLLMPriority: 5, Status: "ok"},
		}

		headerPath := filepath.Join(dir, "observe_header.yaml")
		dataPath := filepath.Join(dir, "observe_data.csv")
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatal(err)
		}

		// Read CSV header row
		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		if len(lines) < 1 {
			t.Fatal("CSV file is empty")
		}
		headerRow := lines[0]

		// THEN vllm_priority should be in the header
		if !strings.Contains(headerRow, "vllm_priority") {
			t.Errorf("vllm_priority column should be present when any record has VLLMPriority!=0")
		}

		// AND it should appear after slo_class
		columns := strings.Split(headerRow, ",")
		sloIdx := -1
		vllmIdx := -1
		for i, col := range columns {
			if col == "slo_class" {
				sloIdx = i
			}
			if col == "vllm_priority" {
				vllmIdx = i
			}
		}
		if sloIdx == -1 {
			t.Fatal("slo_class column not found")
		}
		if vllmIdx == -1 {
			t.Fatal("vllm_priority column not found")
		}
		if vllmIdx != sloIdx+1 {
			t.Errorf("vllm_priority should appear immediately after slo_class, got indices slo=%d vllm=%d", sloIdx, vllmIdx)
		}

		// BC-4: Priority values preserved in round-trip
		// Verify data rows contain correct priority values
		if len(lines) < 3 {
			t.Fatal("Expected at least 3 lines (header + 2 data rows)")
		}
		row1 := strings.Split(lines[1], ",")
		row2 := strings.Split(lines[2], ",")
		
		if row1[vllmIdx] != "0" {
			t.Errorf("Row 1 vllm_priority: got %q, want \"0\"", row1[vllmIdx])
		}
		if row2[vllmIdx] != "5" {
			t.Errorf("Row 2 vllm_priority: got %q, want \"5\"", row2[vllmIdx])
		}
	})

	t.Run("observe without SLOClass → no column (Mode=real, no SLOClass, priority=0)", func(t *testing.T) {
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
		records := []TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000, SLOClass: "", VLLMPriority: 0, Status: "ok"},
			{RequestID: 2, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000, SLOClass: "", VLLMPriority: 0, Status: "ok"},
		}

		headerPath := filepath.Join(dir, "no_slo_header.yaml")
		dataPath := filepath.Join(dir, "no_slo_data.csv")
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatal(err)
		}

		// Read CSV header row
		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		headerRow := lines[0]

		// THEN vllm_priority should NOT be in the header
		if strings.Contains(headerRow, "vllm_priority") {
			t.Errorf("vllm_priority column should not be present when observe has no SLOClass")
		}
	})

	t.Run("mixed: observe with some non-zero priorities", func(t *testing.T) {
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
		records := []TraceRecord{
			{RequestID: 1, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000, SLOClass: "", VLLMPriority: 0, Status: "ok"},
			{RequestID: 2, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000, SLOClass: "standard", VLLMPriority: 1, Status: "ok"},
		}

		headerPath := filepath.Join(dir, "mixed_header.yaml")
		dataPath := filepath.Join(dir, "mixed_data.csv")
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatal(err)
		}

		// Read CSV header row
		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		headerRow := lines[0]

		// THEN vllm_priority should be in the header (at least one record has VLLMPriority!=0)
		if !strings.Contains(headerRow, "vllm_priority") {
			t.Errorf("vllm_priority column should be present when at least one record has VLLMPriority!=0")
		}
	})
}

func TestLoadTraceV2Requests_IgnoresVLLMPriority_SimulationIsolation(t *testing.T) {
	// BC-5: LoadTraceV2Requests must NOT read VLLMPriority into Request.Priority
	// This is a simulation isolation requirement — observability metadata must
	// not affect simulation behavior.
	
	trace := &TraceV2{
		Header: TraceHeader{Version: 2, TimeUnit: "us"},
		Records: []TraceRecord{
			{RequestID: 1, SLOClass: "critical", VLLMPriority: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000},
			{RequestID: 2, SLOClass: "batch", VLLMPriority: 5, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 2000},
		},
	}

	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 2 {
		t.Fatalf("len(requests)=%d, want 2", len(requests))
	}

	// THEN Request.Priority must remain 0 (default) regardless of VLLMPriority
	if requests[0].Priority != 0 {
		t.Errorf("Request 0 Priority=%f, want 0 (must not read VLLMPriority)", requests[0].Priority)
	}
	if requests[1].Priority != 0 {
		t.Errorf("Request 1 Priority=%f, want 0 (must not read VLLMPriority)", requests[1].Priority)
	}

	// AND SLOClass should be preserved for admission/routing decisions
	if requests[0].SLOClass != "critical" {
		t.Errorf("Request 0 SLOClass=%q, want %q", requests[0].SLOClass, "critical")
	}
	if requests[1].SLOClass != "batch" {
		t.Errorf("Request 1 SLOClass=%q, want %q", requests[1].SLOClass, "batch")
	}
}

func TestLoadTraceV2SessionBlueprints_IgnoresVLLMPriority_SimulationIsolation(t *testing.T) {
	// BC-6: LoadTraceV2SessionBlueprints must NOT read VLLMPriority
	// Same simulation isolation requirement as BC-5, but for session blueprints.
	
	trace := &TraceV2{
		Header: TraceHeader{Version: 2, TimeUnit: "us"},
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "s1", RoundIndex: 0, SLOClass: "critical", VLLMPriority: 0,
				InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 1000,
				FirstChunkTimeUs: 1500, LastChunkTimeUs: 2000},
			{RequestID: 2, SessionID: "s1", RoundIndex: 1, SLOClass: "batch", VLLMPriority: 5,
				InputTokens: 50, OutputTokens: 25, ArrivalTimeUs: 3000,
				FirstChunkTimeUs: 3500, LastChunkTimeUs: 4000},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Session rounds are grouped into one blueprint, not separate requests
	if len(requests) != 1 {
		t.Fatalf("len(requests)=%d, want 1 (session grouped)", len(requests))
	}
	if len(blueprints) != 1 {
		t.Fatalf("len(blueprints)=%d, want 1", len(blueprints))
	}

	// THEN returned request must NOT have Priority set from VLLMPriority
	if requests[0].Priority != 0 {
		t.Errorf("Request 0 Priority=%f, want 0 (must not read VLLMPriority)", requests[0].Priority)
	}

	// AND SLOClass should be preserved in blueprint for admission decisions
	// (uses first round's SLOClass)
	if blueprints[0].SLOClass != "critical" {
		t.Errorf("Blueprint SLOClass=%q, want %q", blueprints[0].SLOClass, "critical")
	}

	// AND request should have SLOClass preserved
	if requests[0].SLOClass != "critical" {
		t.Errorf("Request 0 SLOClass=%q, want %q", requests[0].SLOClass, "critical")
	}
}
