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
		Version: 3, TimeUnit: "microseconds", Mode: "generated",
		WarmUpRequests: 5, WorkloadSpec: "test.yaml",
		GoodputSLOTargets: map[string]SLODimTargets{
			"critical": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
			"batch":    {E2EMs: 120000},
		},
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

	if loaded.Header.Version != 3 {
		t.Errorf("version = %d, want 3", loaded.Header.Version)
	}
	if loaded.Header.WarmUpRequests != 5 {
		t.Errorf("warm_up = %d, want 5", loaded.Header.WarmUpRequests)
	}
	// BC-8: GoodputSLOTargets round-trips per (class, dimension).
	if got := loaded.Header.GoodputSLOTargets["critical"].TTFTMs; got != 100 {
		t.Errorf("critical.TTFTMs = %f, want 100", got)
	}
	if got := loaded.Header.GoodputSLOTargets["critical"].ITLMs; got != 50 {
		t.Errorf("critical.ITLMs = %f, want 50", got)
	}
	if got := loaded.Header.GoodputSLOTargets["critical"].E2EMs; got != 5000 {
		t.Errorf("critical.E2EMs = %f, want 5000", got)
	}
	if got := loaded.Header.GoodputSLOTargets["batch"].E2EMs; got != 120000 {
		t.Errorf("batch.E2EMs = %f, want 120000", got)
	}
	// BC-3 surface: zero is "not gated" — survives round-trip.
	if got := loaded.Header.GoodputSLOTargets["batch"].ITLMs; got != 0 {
		t.Errorf("batch.ITLMs = %f, want 0 (not gated)", got)
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

// TestTraceV2_LoadHeader_UnknownField_ReturnsError verifies BC-N3: KnownFields(true)
// causes an old binary reading a header with an unknown field (e.g. a future
// schema addition past v3) to hard-fail rather than silently dropping the field.
// Strict parsing is enforced at sim/workload/tracev2.go:223; the trace Version
// field is the schema-change signal.
func TestTraceV2_LoadHeader_UnknownField_ReturnsError(t *testing.T) {
	yamlText := []byte("trace_version: 99\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\nbogus_unknown_field: 42\n")
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	if err := os.WriteFile(headerPath, yamlText, 0o644); err != nil {
		t.Fatal(err)
	}
	dataPath := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(dataPath, []byte("request_id\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTraceV2(headerPath, dataPath); err == nil ||
		!strings.Contains(err.Error(), "field") {
		t.Errorf("expected unknown-field error, got %v", err)
	}
}

// TestTraceV2_RoundTrip_NewFields verifies BC-1 and BC-2: all three new
// schema fields survive export → load with correct values.
func TestTraceV2_RoundTrip_NewFields(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "converted"}
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
			Model:             "", // zero value: default model
			DeadlineUs:        0,  // zero value: no timeout
			ServerInputTokens: 0,  // zero value: not recorded
			InputTokens:       128,
			OutputTokens:      32,
			ArrivalTimeUs:     5000,
			Status:            "ok",
		},
		{
			RequestID:         2,
			Model:             "org/model,with-comma", // CSV-special: encoding/csv quotes automatically
			DeadlineUs:        9000,                   // > ArrivalTimeUs (6000) — valid per cross-field invariant
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
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "converted"}
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
	row[0] = "abc"            // request_id should be integer
	for i := 1; i < len(row); i++ {
		row[i] = "0"
	}

	// WHEN parsing
	_, err := parseTraceRecord(row, false, -1, -1, -1)

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

	_, err := parseTraceRecord(row, false, -1, -1, -1)

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

	_, err := parseTraceRecord(row, false, -1, -1, -1)

	if err == nil {
		t.Fatal("expected error for non-numeric server_input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "server_input_tokens") {
		t.Errorf("error should mention 'server_input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeDeadlineUs_ReturnsError verifies R3 validation.
func TestParseTraceRecord_NegativeDeadlineUs_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[17] = "-1" // negative deadline_us (shifted +1)

	_, err := parseTraceRecord(row, false, -1, -1, -1)

	if err == nil {
		t.Fatal("expected error for negative deadline_us, got nil")
	}
	if !strings.Contains(err.Error(), "deadline_us") {
		t.Errorf("error should mention 'deadline_us', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeArrivalTimeUs_ReturnsError verifies the #1606
// hardening: arrival_time_us is the injection-origin anchor, so a negative value
// (which would yield a negative DES injection tick, violating INV-3) is rejected
// loudly, matching the other time/count fields (R1). send_time_us stays exempt
// (injectionTime falls back to arrival when send <= 0).
func TestParseTraceRecord_NegativeArrivalTimeUs_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[19] = "-1" // arrival_time_us column (offset 0 without slo_target_us)

	_, err := parseTraceRecord(row, false, -1, -1, -1)

	if err == nil {
		t.Fatal("expected error for negative arrival_time_us, got nil")
	}
	if !strings.Contains(err.Error(), "arrival_time_us") {
		t.Errorf("error should mention 'arrival_time_us', got: %s", err.Error())
	}
	if !strings.Contains(err.Error(), "negative") {
		t.Errorf("error should mention 'negative', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeSendTimeUs_Tolerated verifies the deliberate
// asymmetry (#1606): a negative send_time_us is NOT rejected — injectionTime
// falls back to arrival_time_us for send <= 0, so a corrupt/absent send column
// must not fail trace loading.
func TestParseTraceRecord_NegativeSendTimeUs_Tolerated(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[20] = "-100" // send_time_us column; must be tolerated

	rec, err := parseTraceRecord(row, false, -1, -1, -1)
	if err != nil {
		t.Fatalf("negative send_time_us must be tolerated (falls back to arrival), got error: %v", err)
	}
	if rec.SendTimeUs != -100 {
		t.Errorf("send_time_us = %d, want -100 (preserved verbatim; fallback happens at injection)", rec.SendTimeUs)
	}
}

// TestParseTraceRecord_NegativeInputTokens_ReturnsError verifies R3 for
// input_tokens (prevents make([]sim.TokenID, negative) panic in replay).
func TestParseTraceRecord_NegativeInputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[9] = "-1" // input_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false, -1, -1, -1)

	if err == nil {
		t.Fatal("expected error for negative input_tokens, got nil")
	}
	if !strings.Contains(err.Error(), "input_tokens") {
		t.Errorf("error should mention 'input_tokens', got: %s", err.Error())
	}
}

// TestParseTraceRecord_NegativeOutputTokens_ReturnsError verifies R3 for
// output_tokens (prevents make([]sim.TokenID, negative) panic in replay).
func TestParseTraceRecord_NegativeOutputTokens_ReturnsError(t *testing.T) {
	row := make([]string, 27)
	for i := range row {
		row[i] = "0"
	}
	row[10] = "-1" // output_tokens column (shifted +1)

	_, err := parseTraceRecord(row, false, -1, -1, -1)

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

	_, err := parseTraceRecord(row, false, -1, -1, -1)

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

	_, err := parseTraceRecord(row, false, -1, -1, -1)

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

		_, err := parseTraceRecord(row, false, -1, -1, -1)

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
	header := &TraceHeader{Version: 1, TimeUnit: "us", Mode: "converted"}
	records := []TraceRecord{{
		RequestID:     1,
		InputTokens:   10,
		OutputTokens:  5,
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
		InputTokens:    []sim.TokenID{1, 2, 3},
		OutputTokens:   []sim.TokenID{4, 5},
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
		InputTokens:     make([]sim.TokenID, 512),
		OutputTokens:    make([]sim.TokenID, 256),
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
			InputTokens:  []sim.TokenID{1},
			OutputTokens: []sim.TokenID{2},
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
		InputTokens:    make([]sim.TokenID, 100),
		OutputTokens:   make([]sim.TokenID, 50),
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
		InputTokens:  make([]sim.TokenID, 100),
		OutputTokens: make([]sim.TokenID, 50),
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
			InputTokens:  []sim.TokenID{1},
			OutputTokens: []sim.TokenID{2},
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
			InputTokens:    make([]sim.TokenID, 512),
			OutputTokens:   make([]sim.TokenID, 128),
			ArrivalTime:    1000,
			TTFTSet:        true,
			FirstTokenTime: 500,
			TenantID:       "t1",
			SLOClass:       "batch",
			ClientID:       "c1",
			Streaming:      true,
			Model:          "test-model",
			Deadline:       50000,
		},
		{
			ID:           "request_1",
			State:        sim.StateTimedOut,
			InputTokens:  make([]sim.TokenID, 256),
			OutputTokens: make([]sim.TokenID, 64),
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

// TestTraceV2_Adapter_RoundTrip verifies that LoRA adapter identity survives the
// full run→trace→replay round-trip (#1464 F1). A request carrying a non-empty
// Adapter must be reconstructed by LoadTraceV2Requests with the same Adapter, and
// the trace must omit the adapter column entirely for adapter-blind workloads so
// the no-op default adds no columns (INV-6).
func TestTraceV2_Adapter_RoundTrip(t *testing.T) {
	t.Run("adapter preserved through export and replay reconstruction", func(t *testing.T) {
		reqs := []*sim.Request{
			{
				ID:           "request_0",
				State:        sim.StateCompleted,
				InputTokens:  make([]sim.TokenID, 64),
				OutputTokens: make([]sim.TokenID, 32),
				ArrivalTime:  1000,
				Adapter:      "sql-lora",
			},
			{
				ID:           "request_1",
				State:        sim.StateCompleted,
				InputTokens:  make([]sim.TokenID, 48),
				OutputTokens: make([]sim.TokenID, 16),
				ArrivalTime:  2000,
				Adapter:      "", // base-model-only
			},
		}

		records := RequestsToTraceRecords(reqs)
		if records[0].Adapter != "sql-lora" {
			t.Fatalf("RequestsToTraceRecords dropped adapter: got %q, want %q", records[0].Adapter, "sql-lora")
		}

		dir := t.TempDir()
		headerPath := filepath.Join(dir, "header.yaml")
		dataPath := filepath.Join(dir, "data.csv")
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "generated"}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}

		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[0].Adapter != "sql-lora" {
			t.Errorf("TraceRecord round-trip: got adapter %q, want %q", loaded.Records[0].Adapter, "sql-lora")
		}
		if loaded.Records[1].Adapter != "" {
			t.Errorf("base-model record: got adapter %q, want empty", loaded.Records[1].Adapter)
		}

		// The replay reconstruction path must restore Adapter onto the request.
		replayed, err := LoadTraceV2Requests(loaded, 42)
		if err != nil {
			t.Fatalf("LoadTraceV2Requests: %v", err)
		}
		if len(replayed) != len(reqs) {
			t.Fatalf("replayed request count: got %d, want %d", len(replayed), len(reqs))
		}
		if replayed[0].Adapter != "sql-lora" {
			t.Errorf("replayed request adapter: got %q, want %q (INV-13 identity lost)", replayed[0].Adapter, "sql-lora")
		}
		if replayed[1].Adapter != "" {
			t.Errorf("replayed base-model request adapter: got %q, want empty", replayed[1].Adapter)
		}
	})

	t.Run("adapter threads through session round-0 request and blueprint", func(t *testing.T) {
		// Two-round session; both rounds carry the same adapter. The blueprint
		// drives follow-up rounds, so it must also carry the adapter (session.go
		// applies bp.Adapter to each follow-up round request).
		records := []TraceRecord{
			{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 32, OutputTokens: 16, ArrivalTimeUs: 1000, Adapter: "sql-lora", Status: "ok"},
			{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 24, OutputTokens: 8, ArrivalTimeUs: 5000, Adapter: "sql-lora", Status: "ok"},
		}
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "header.yaml")
		dataPath := filepath.Join(dir, "data.csv")
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "generated"}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}

		round0, blueprints, err := LoadTraceV2SessionBlueprints(loaded, 42, nil, 0)
		if err != nil {
			t.Fatalf("LoadTraceV2SessionBlueprints: %v", err)
		}
		if len(round0) != 1 || len(blueprints) != 1 {
			t.Fatalf("got %d round-0 requests and %d blueprints, want 1 and 1", len(round0), len(blueprints))
		}
		if round0[0].Adapter != "sql-lora" {
			t.Errorf("session round-0 request adapter: got %q, want %q", round0[0].Adapter, "sql-lora")
		}
		if blueprints[0].Adapter != "sql-lora" {
			t.Errorf("session blueprint adapter: got %q, want %q (follow-up rounds would lose adapter)", blueprints[0].Adapter, "sql-lora")
		}
	})

	t.Run("adapter-blind workload omits the adapter column (INV-6 no-op)", func(t *testing.T) {
		reqs := []*sim.Request{
			{
				ID:           "request_0",
				State:        sim.StateCompleted,
				InputTokens:  make([]sim.TokenID, 64),
				OutputTokens: make([]sim.TokenID, 32),
				ArrivalTime:  1000,
			},
		}
		records := RequestsToTraceRecords(reqs)

		dir := t.TempDir()
		headerPath := filepath.Join(dir, "header.yaml")
		dataPath := filepath.Join(dir, "data.csv")
		header := &TraceHeader{Version: 2, TimeUnit: "us", Mode: "generated"}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}

		data, err := os.ReadFile(dataPath)
		if err != nil {
			t.Fatal(err)
		}
		headerRow := strings.Split(string(data), "\n")[0]
		if strings.Contains(headerRow, "adapter") {
			t.Errorf("adapter column must be absent for adapter-blind workloads; header: %q", headerRow)
		}
	})
}

func TestTraceV2_ThinkTimeUs_RoundTrip(t *testing.T) {
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}

	t.Run("present: value round-trips and column is emitted", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ThinkTimeUs: i64p(0)},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ThinkTimeUs: i64p(5000)},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		data, _ := os.ReadFile(dataPath)
		if !strings.Contains(string(data), "think_time_us") {
			t.Error("think_time_us column missing from CSV when a record carries it")
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		// A recorded &0 round-trips as non-nil &0 (NOT nil) — the core #1608 distinction.
		if loaded.Records[0].ThinkTimeUs == nil || *loaded.Records[0].ThinkTimeUs != 0 {
			t.Errorf("recorded &0 round-trip = %v, want &0 (non-nil)", loaded.Records[0].ThinkTimeUs)
		}
		if loaded.Records[1].ThinkTimeUs == nil || *loaded.Records[1].ThinkTimeUs != 5000 {
			t.Errorf("recorded &5000 round-trip = %v, want &5000", loaded.Records[1].ThinkTimeUs)
		}
	})

	t.Run("mixed presence: nil / &0 / &N round-trip distinctly (#1608)", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		// nil = not recorded (empty cell); &0 = recorded zero ("0"); &N = recorded N.
		// think_time_us is the last trailing column, so a nil is an empty trailing field.
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ThinkTimeUs: nil},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ThinkTimeUs: i64p(0)},
			{RequestID: 2, SessionID: "A", RoundIndex: 2, InputTokens: 70, OutputTokens: 30, ThinkTimeUs: i64p(9000)},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[0].ThinkTimeUs != nil {
			t.Errorf("record 0 (nil) round-trip = %v, want nil (empty cell)", loaded.Records[0].ThinkTimeUs)
		}
		if loaded.Records[1].ThinkTimeUs == nil || *loaded.Records[1].ThinkTimeUs != 0 {
			t.Errorf("record 1 (&0) round-trip = %v, want &0 (distinct from nil)", loaded.Records[1].ThinkTimeUs)
		}
		if loaded.Records[2].ThinkTimeUs == nil || *loaded.Records[2].ThinkTimeUs != 9000 {
			t.Errorf("record 2 (&9000) round-trip = %v, want &9000", loaded.Records[2].ThinkTimeUs)
		}
	})

	t.Run("absent: no column when no record carries it (INV-6 byte-identity)", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		data, _ := os.ReadFile(dataPath)
		if strings.Contains(string(data), "think_time_us") {
			t.Error("think_time_us column emitted when no record carries it (must be omitted, INV-6)")
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[1].ThinkTimeUs != nil {
			t.Errorf("absent column: ThinkTimeUs = %v, want nil", loaded.Records[1].ThinkTimeUs)
		}
	})

	t.Run("lone recorded &0 emits column (BC-5, was omitted pre-#1608)", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		// A single recorded &0 (with all other records nil) now emits the column —
		// pre-#1608 the != 0 gate would have omitted it, losing the recorded zero.
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ThinkTimeUs: nil},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ThinkTimeUs: i64p(0)},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		data, _ := os.ReadFile(dataPath)
		if !strings.Contains(string(data), "think_time_us") {
			t.Error("think_time_us column omitted when a record recorded &0 (must be present, BC-5)")
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[1].ThinkTimeUs == nil || *loaded.Records[1].ThinkTimeUs != 0 {
			t.Errorf("lone &0 round-trip = %v, want &0", loaded.Records[1].ThinkTimeUs)
		}
	})
}

func TestTraceV2_InputTokensReset_RoundTrip(t *testing.T) {
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	i64 := func(v int64) *int64 { return &v }

	t.Run("present: marker round-trips and column is emitted", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},                           // no reset
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 0, OutputTokens: 40, InputTokensReset: i64(120)}, // compaction
			{RequestID: 2, SessionID: "A", RoundIndex: 2, InputTokens: 30, OutputTokens: 10},                            // monotone
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		data, _ := os.ReadFile(dataPath)
		if !strings.Contains(string(data), "input_tokens_reset") {
			t.Error("input_tokens_reset column missing from CSV when a record carries it")
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[0].InputTokensReset != nil {
			t.Errorf("record0 InputTokensReset = %d, want nil (empty cell)", *loaded.Records[0].InputTokensReset)
		}
		if loaded.Records[1].InputTokensReset == nil || *loaded.Records[1].InputTokensReset != 120 {
			t.Errorf("record1 InputTokensReset round-trip = %v, want &120", loaded.Records[1].InputTokensReset)
		}
		if loaded.Records[2].InputTokensReset != nil {
			t.Errorf("record2 InputTokensReset = %d, want nil (monotone)", *loaded.Records[2].InputTokensReset)
		}
	})

	t.Run("absent: no column when no record carries it (INV-6 byte-identity)", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		data, _ := os.ReadFile(dataPath)
		if strings.Contains(string(data), "input_tokens_reset") {
			t.Error("input_tokens_reset column emitted when no record carries it (must be omitted, INV-6)")
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Records[1].InputTokensReset != nil {
			t.Errorf("absent column: InputTokensReset = %d, want nil", *loaded.Records[1].InputTokensReset)
		}
	})

	t.Run("zero is a valid recorded absolute (distinct from absent)", func(t *testing.T) {
		dir := t.TempDir()
		headerPath := filepath.Join(dir, "h.yaml")
		dataPath := filepath.Join(dir, "d.csv")
		records := []TraceRecord{
			{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},
			{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 0, OutputTokens: 40, InputTokensReset: i64(0)},
		}
		if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		loaded, err := LoadTraceV2(headerPath, dataPath)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		// Presence encoding (#1612 precedent): recorded &0 survives as non-nil,
		// distinct from an absent (nil) cell.
		if loaded.Records[1].InputTokensReset == nil || *loaded.Records[1].InputTokensReset != 0 {
			t.Errorf("recorded &0 round-trip = %v, want &0 (presence, not absence)", loaded.Records[1].InputTokensReset)
		}
	})
}

func TestParseTraceRecord_NegativeInputTokensReset_Errors(t *testing.T) {
	// input_tokens_reset carrying a negative value must be rejected (an input
	// length cannot be negative).
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	i64 := func(v int64) *int64 { return &v }
	records := []TraceRecord{
		{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},
		{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 0, OutputTokens: 40, InputTokensReset: i64(120)},
	}
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	data, _ := os.ReadFile(dataPath)
	corrupt := strings.Replace(string(data), "120", "-120", 1)
	if err := os.WriteFile(dataPath, []byte(corrupt), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadTraceV2(headerPath, dataPath); err == nil || !strings.Contains(err.Error(), "input_tokens_reset") {
		t.Errorf("expected negative input_tokens_reset to error, got %v", err)
	}
}

func TestParseTraceRecord_NegativeThinkTime_Errors(t *testing.T) {
	// A think_time_us column carrying a negative value must be rejected (INV-3:
	// recorded client think time cannot be negative).
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ThinkTimeUs: i64p(5000)},
	}
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	data, _ := os.ReadFile(dataPath)
	corrupt := strings.Replace(string(data), "5000", "-5000", 1)
	if err := os.WriteFile(dataPath, []byte(corrupt), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadTraceV2(headerPath, dataPath); err == nil || !strings.Contains(err.Error(), "think_time_us") {
		t.Errorf("expected negative think_time_us to error, got %v", err)
	}
}

func TestTraceV2_SessionContextGrowth_HeaderRoundTrip(t *testing.T) {
	t.Run("accumulate survives export→load", func(t *testing.T) {
		dir := t.TempDir()
		hp := filepath.Join(dir, "h.yaml")
		dp := filepath.Join(dir, "d.csv")
		h := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
		if err := ExportTraceV2(h, nil, hp, dp); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		loaded, err := LoadTraceV2(hp, dp)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Header.SessionContextGrowth != "accumulate" {
			t.Errorf("SessionContextGrowth = %q, want %q", loaded.Header.SessionContextGrowth, "accumulate")
		}
	})
	t.Run("absent → empty (omitempty)", func(t *testing.T) {
		dir := t.TempDir()
		hp := filepath.Join(dir, "h.yaml")
		dp := filepath.Join(dir, "d.csv")
		h := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
		if err := ExportTraceV2(h, nil, hp, dp); err != nil {
			t.Fatalf("ExportTraceV2: %v", err)
		}
		loaded, err := LoadTraceV2(hp, dp)
		if err != nil {
			t.Fatalf("LoadTraceV2: %v", err)
		}
		if loaded.Header.SessionContextGrowth != "" {
			t.Errorf("SessionContextGrowth = %q, want empty", loaded.Header.SessionContextGrowth)
		}
	})
}

func TestExportTraceV2_ThinkTimeUs_CoexistsWithAdapter(t *testing.T) {
	// think_time_us is appended after the adapter trailing column; verify both optional
	// trailing columns round-trip together with correct values (guards the
	// column-ordering / index math when they coexist).
	dir := t.TempDir()
	hp := filepath.Join(dir, "h.yaml")
	dp := filepath.Join(dir, "d.csv")
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "converted"}
	records := []TraceRecord{
		{RequestID: 0, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, Adapter: "sql-lora", ThinkTimeUs: i64p(0)},
		{RequestID: 1, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, Adapter: "sql-lora", ThinkTimeUs: i64p(7000)},
	}
	if err := ExportTraceV2(header, records, hp, dp); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	loaded, err := LoadTraceV2(hp, dp)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	r := loaded.Records[1]
	if r.Adapter != "sql-lora" || r.ThinkTimeUs == nil || *r.ThinkTimeUs != 7000 {
		t.Errorf("coexistence round-trip: adapter=%q think=%v; want sql-lora/&7000", r.Adapter, r.ThinkTimeUs)
	}
	if loaded.Records[0].ThinkTimeUs == nil || *loaded.Records[0].ThinkTimeUs != 0 {
		t.Errorf("record 0 think_time_us = %v, want &0 (recorded zero, non-nil)", loaded.Records[0].ThinkTimeUs)
	}
}

func TestTraceV2_SLOTargetUs_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")

	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, SLOTargetUs: 200000, DeadlineUs: 500000, ArrivalTimeUs: 100, Status: "ok"},
		{RequestID: 1, SLOTargetUs: 0, DeadlineUs: 300000, ArrivalTimeUs: 200, Status: "ok"},
	}

	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}

	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(loaded.Records))
	}
	if loaded.Records[0].SLOTargetUs != 200000 {
		t.Errorf("record 0 SLOTargetUs=%d, want 200000", loaded.Records[0].SLOTargetUs)
	}
	if loaded.Records[1].SLOTargetUs != 0 {
		t.Errorf("record 1 SLOTargetUs=%d, want 0", loaded.Records[1].SLOTargetUs)
	}
	// Verify other fields survived the offset shift
	if loaded.Records[0].DeadlineUs != 500000 {
		t.Errorf("record 0 DeadlineUs=%d, want 500000", loaded.Records[0].DeadlineUs)
	}
	if loaded.Records[0].ArrivalTimeUs != 100 {
		t.Errorf("record 0 ArrivalTimeUs=%d, want 100", loaded.Records[0].ArrivalTimeUs)
	}
}

func TestTraceV2_SLOTargetUs_AllZero_ColumnOmitted(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")

	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, SLOTargetUs: 0, ArrivalTimeUs: 100, Status: "ok"},
	}

	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}

	// Read raw CSV to verify column is absent
	data, _ := os.ReadFile(dataPath)
	if strings.Contains(string(data), "slo_target_us") {
		t.Error("slo_target_us column should be omitted when all values are 0")
	}

	// Verify it still loads correctly
	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Records[0].SLOTargetUs != 0 {
		t.Errorf("SLOTargetUs=%d, want 0", loaded.Records[0].SLOTargetUs)
	}
}

// TestTraceV2_SLOTargetUs_ShiftsLaterColumns verifies that slo_target_us, inserted after
// deadline_us, shifts every later column by one on load.
func TestTraceV2_SLOTargetUs_ShiftsLaterColumns(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")

	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{
			RequestID: 0, SLOClass: "critical",
			SLOTargetUs: 150000, DeadlineUs: 400000,
			ArrivalTimeUs: 1000, SendTimeUs: 1100,
			FirstChunkTimeUs: 2000, LastChunkTimeUs: 3000,
			NumChunks: 5, Status: "ok",
		},
	}

	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}

	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r := loaded.Records[0]
	if r.SLOTargetUs != 150000 {
		t.Errorf("SLOTargetUs=%d, want 150000", r.SLOTargetUs)
	}
	if r.DeadlineUs != 400000 {
		t.Errorf("DeadlineUs=%d, want 400000", r.DeadlineUs)
	}
	if r.ArrivalTimeUs != 1000 {
		t.Errorf("ArrivalTimeUs=%d, want 1000 (offset corruption)", r.ArrivalTimeUs)
	}
	if r.SendTimeUs != 1100 {
		t.Errorf("SendTimeUs=%d, want 1100 (offset corruption)", r.SendTimeUs)
	}
	if r.NumChunks != 5 {
		t.Errorf("NumChunks=%d, want 5 (offset corruption)", r.NumChunks)
	}
}

// TestTraceV2_RoundTrip_WithKVOffload verifies the resolved KV-offload config
// (H5, #1587, BC-G6) survives export→load with all fields intact, and that a header
// without offload omits the key (BC-G5).
func TestTraceV2_RoundTrip_WithKVOffload(t *testing.T) {
	header := &TraceHeader{
		Version: 3, TimeUnit: "microseconds", Mode: "generated",
		KVOffload: &TraceKVOffloadConfig{
			CPUBytesToUse:          17179869184,
			BlockSize:              16,
			BlocksPerChunk:         1,
			TokensPerHash:          16,
			EvictionPolicy:         "lru",
			OffloadPromptOnly:      true,
			SelfDescribingKVEvents: false,
			Tiers: []TraceKVOffloadTier{
				{
					Type: "fs", RootDir: "/mnt/kv-cache",
					NReadThreads: 16, NWriteThreads: 16,
					Locality: "LOCAL", EnableKVEvents: false, DirectIO: true,
					DeviceClass:   "nvme_gen4",
					ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
				},
			},
		},
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
	got := loaded.Header.KVOffload
	if got == nil {
		t.Fatal("kv_offload config should round-trip, got nil")
	}
	if got.CPUBytesToUse != 17179869184 || got.EvictionPolicy != "lru" || !got.OffloadPromptOnly {
		t.Errorf("scalar mismatch after round-trip: %+v", got)
	}
	if len(got.Tiers) != 1 {
		t.Fatalf("tiers: got %d, want 1", len(got.Tiers))
	}
	tr := got.Tiers[0]
	if tr.Type != "fs" || tr.RootDir != "/mnt/kv-cache" || !tr.DirectIO ||
		tr.ReadBandwidth != 7000 || tr.WriteBandwidth != 5000 || tr.BaseLatency != 80 ||
		tr.NReadThreads != 16 || tr.Locality != "LOCAL" || tr.DeviceClass != "nvme_gen4" {
		t.Errorf("tier mismatch after round-trip: %+v", tr)
	}
}

// TestTraceV2_NoKVOffload_OmitsKey verifies a header without offload writes no
// kv_offload key (BC-G5 byte-identity for the disabled case).
func TestTraceV2_NoKVOffload_OmitsKey(t *testing.T) {
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	if err := ExportTraceV2(header, nil, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(headerPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "kv_offload") {
		t.Errorf("header without offload must not contain a kv_offload key:\n%s", data)
	}
	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Header.KVOffload != nil {
		t.Error("KVOffload must be nil when absent")
	}
}

// TestTraceV2_KVOffload_UnknownSubKey_Errors verifies strict parsing rejects an
// unknown offload sub-key (BC-G6: replay must fail loudly on a config it cannot
// reconstruct, e.g. one written by a future binary).
func TestTraceV2_KVOffload_UnknownSubKey_Errors(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	yaml := "trace_version: 3\ntime_unit: microseconds\nmode: generated\n" +
		"warm_up_requests: 0\nkv_offload:\n  cpu_bytes_to_use: 1\n  future_knob: 42\n"
	if err := os.WriteFile(headerPath, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dataPath, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTraceV2(headerPath, dataPath); err == nil {
		t.Fatal("expected strict-parse error for unknown kv_offload sub-key, got nil")
	}
}

// TestTraceV2_SLOTargetUs_KeepsTheColumnsBeforeIt: slo_target_us sits after deadline_us, so the
// columns before it are read where they are. A record with a target keeps its session_id,
// prefix_group and model through a round trip, and so does a record without one in the same
// file (the column is in every row once one row carries it). Before this was fixed, the loader
// read those three one column to the right: session_id took round_index's value, prefix_group
// prefix_length's, and model deadline_us's. The writer was correct.
func TestTraceV2_SLOTargetUs_KeepsTheColumnsBeforeIt(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath := filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, ClientID: "c", TenantID: "t", SLOClass: "critical", SessionID: "s0", PrefixGroup: "g",
			PrefixLength: 16, Model: "m", DeadlineUs: 900000, SLOTargetUs: 150000, InputTokens: 30, OutputTokens: 5,
			ArrivalTimeUs: 1000, Status: "ok"},
		{RequestID: 1, ClientID: "c", TenantID: "t", SLOClass: "standard", SessionID: "s1", PrefixGroup: "",
			Model: "", InputTokens: 31, OutputTokens: 6, ArrivalTimeUs: 2000, Status: "ok"},
	}
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	for i, want := range records {
		got := loaded.Records[i]
		if got.SessionID != want.SessionID || got.PrefixGroup != want.PrefixGroup || got.Model != want.Model ||
			got.SLOTargetUs != want.SLOTargetUs || got.DeadlineUs != want.DeadlineUs || got.Status != want.Status {
			t.Errorf("record %d: session %q prefix %q model %q slo %d deadline %d status %q; want %q %q %q %d %d %q", i,
				got.SessionID, got.PrefixGroup, got.Model, got.SLOTargetUs, got.DeadlineUs, got.Status,
				want.SessionID, want.PrefixGroup, want.Model, want.SLOTargetUs, want.DeadlineUs, want.Status)
		}
	}
}
