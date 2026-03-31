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
	row := make([]string, 28) // must match current column count
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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[18] = "not_a_number" // deadline_us column

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[19] = "not_a_number" // server_input_tokens column

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[18] = "-1" // negative deadline_us

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[10] = "-1" // input_tokens column

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[11] = "-1" // output_tokens column

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[19] = "-1" // server_input_tokens column

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
	row := make([]string, 28)
	for i := range row {
		row[i] = "0"
	}
	row[18] = "1000" // deadline_us
	row[20] = "5000" // arrival_time_us

	_, err := parseTraceRecord(row)

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
		row := make([]string, 28)
		for i := range row {
			row[i] = "0"
		}
		row[16] = tc.value // reason_ratio column

		_, err := parseTraceRecord(row)

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

// TestTraceV2_GIEPriority_RoundTrip verifies BC-4: priority field survives
// export → load round-trip.
func TestTraceV2_GIEPriority_RoundTrip(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "real"}
	records := []TraceRecord{
		{RequestID: 0, ClientID: "c1", TenantID: "t1", SLOClass: "critical",
			GIEPriority: 7, InputTokens: 10, OutputTokens: 5,
			ArrivalTimeUs: 1000, Status: "ok"},
		{RequestID: 1, ClientID: "c2", TenantID: "t2", SLOClass: "standard",
			GIEPriority: 0, InputTokens: 20, OutputTokens: 10,
			ArrivalTimeUs: 2000, Status: "ok"},
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
		t.Fatalf("records = %d, want 2", len(loaded.Records))
	}
	if loaded.Records[0].GIEPriority != 7 {
		t.Errorf("record 0 GIEPriority = %d, want 7", loaded.Records[0].GIEPriority)
	}
	if loaded.Records[1].GIEPriority != 0 {
		t.Errorf("record 1 GIEPriority = %d, want 0", loaded.Records[1].GIEPriority)
	}
}

// TestRequestsToTraceRecords_GIEPriority verifies BC-1: GIEPriority from
// Request propagates to TraceRecord.
func TestRequestsToTraceRecords_GIEPriority(t *testing.T) {
	reqs := []*sim.Request{
		{
			ID: "r1", InputTokens: []int{1, 2}, OutputTokens: []int{3},
			State: sim.StateCompleted, GIEPriority: 5,
		},
		{
			ID: "r2", InputTokens: []int{4, 5, 6}, OutputTokens: []int{7, 8},
			State: sim.StateCompleted, GIEPriority: 0,
		},
	}
	records := RequestsToTraceRecords(reqs)
	if records[0].GIEPriority != 5 {
		t.Errorf("record 0 GIEPriority = %d, want 5", records[0].GIEPriority)
	}
	if records[1].GIEPriority != 0 {
		t.Errorf("record 1 GIEPriority = %d, want 0", records[1].GIEPriority)
	}
}
