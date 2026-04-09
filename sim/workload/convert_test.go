package workload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestConvertPreset_ValidParams_ProducesV2Spec(t *testing.T) {
	// GIVEN a valid preset config
	preset := PresetConfig{
		PrefixTokens:      0,
		PromptTokensMean:  512,
		PromptTokensStdev: 256,
		PromptTokensMin:   2,
		PromptTokensMax:   7000,
		OutputTokensMean:  512,
		OutputTokensStdev: 256,
		OutputTokensMin:   2,
		OutputTokensMax:   7000,
	}

	// WHEN converting
	spec, err := ConvertPreset("chatbot", 10.0, 100, preset)

	// THEN a valid v2 spec is produced
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Version != "2" {
		t.Errorf("version = %q, want %q", spec.Version, "2")
	}
	if spec.AggregateRate != 10.0 {
		t.Errorf("aggregate_rate = %f, want 10.0", spec.AggregateRate)
	}
	if spec.NumRequests != 100 {
		t.Errorf("num_requests = %d, want 100", spec.NumRequests)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("clients count = %d, want 1", len(spec.Clients))
	}
	if spec.Clients[0].InputDist.Type != "gaussian" {
		t.Errorf("input dist type = %q, want %q", spec.Clients[0].InputDist.Type, "gaussian")
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("converted spec fails validation: %v", err)
	}
}

func TestConvertPreset_InvalidRate_ReturnsError(t *testing.T) {
	preset := PresetConfig{}
	_, err := ConvertPreset("test", 0, 100, preset)
	if err == nil {
		t.Fatal("expected error for zero rate")
	}
}

func TestComposeSpecs_TwoSpecs_MergesClients(t *testing.T) {
	// GIVEN two specs with one client each
	specA := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Clients: []ClientSpec{
			{ID: "a", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}}},
		},
	}
	specB := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 5.0,
		Clients: []ClientSpec{
			{ID: "b", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 20, "min": 1, "max": 400}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}}},
		},
	}

	// WHEN composing
	merged, err := ComposeSpecs([]*WorkloadSpec{specA, specB})

	// THEN merged spec has both clients, summed rate, renormalized fractions
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(merged.Clients) != 2 {
		t.Fatalf("clients count = %d, want 2", len(merged.Clients))
	}
	if merged.AggregateRate != 15.0 {
		t.Errorf("aggregate_rate = %f, want 15.0", merged.AggregateRate)
	}
	// Rate-weighted fractions: A (10/15 ≈ 0.6667), B (5/15 ≈ 0.3333)
	const eps = 1e-9
	expectedA := 10.0 / 15.0
	expectedB := 5.0 / 15.0
	if diff := merged.Clients[0].RateFraction - expectedA; diff > eps || diff < -eps {
		t.Errorf("client %s rate_fraction = %f, want %f", merged.Clients[0].ID, merged.Clients[0].RateFraction, expectedA)
	}
	if diff := merged.Clients[1].RateFraction - expectedB; diff > eps || diff < -eps {
		t.Errorf("client %s rate_fraction = %f, want %f", merged.Clients[1].ID, merged.Clients[1].RateFraction, expectedB)
	}
}

func TestComposeSpecs_AllConcurrency_MergesClients(t *testing.T) {
	dist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	spec1 := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{
			{ID: "c1", Concurrency: 5, Arrival: ArrivalSpec{Process: "constant"}, InputDist: dist, OutputDist: dist},
		},
	}
	spec2 := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{
			{ID: "c2", Concurrency: 10, Arrival: ArrivalSpec{Process: "constant"}, InputDist: dist, OutputDist: dist},
		},
	}
	merged, err := ComposeSpecs([]*WorkloadSpec{spec1, spec2})
	if err != nil {
		t.Fatalf("ComposeSpecs failed for all-concurrency specs: %v", err)
	}
	if len(merged.Clients) != 2 {
		t.Errorf("expected 2 merged clients, got %d", len(merged.Clients))
	}
	if merged.AggregateRate != 0 {
		t.Errorf("expected AggregateRate=0 for all-concurrency, got %f", merged.AggregateRate)
	}
	if err := merged.Validate(); err != nil {
		t.Errorf("merged spec fails validation: %v", err)
	}
}

func TestComposeSpecs_NegativeTotalRate_ReturnsError(t *testing.T) {
	dist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: -5.0,
		Clients: []ClientSpec{
			{ID: "c1", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "poisson"}, InputDist: dist, OutputDist: dist},
		},
	}
	_, err := ComposeSpecs([]*WorkloadSpec{spec})
	if err == nil {
		t.Error("expected error for negative aggregate rate")
	}
}

func TestComposeSpecs_EmptyList_ReturnsError(t *testing.T) {
	_, err := ComposeSpecs(nil)
	if err == nil {
		t.Fatal("expected error for empty spec list")
	}
}

func TestConvertServeGen_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertServeGen("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

// ---------- ConvertSWESmith ----------

func writeSWESmithFile(t *testing.T, items []sweSmithItem) string {
	t.Helper()
	data, err := json.Marshal(items)
	if err != nil {
		t.Fatalf("marshal swe-smith: %v", err)
	}
	path := filepath.Join(t.TempDir(), "swe_smith.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write swe-smith: %v", err)
	}
	return path
}

// TestConvertSWESmith_TrajIDAsSessionKey verifies that TrajID — not InstanceID — is
// used as the session key. Two items sharing the same InstanceID but with different
// TrajIDs must produce two distinct sessions, each with consecutive round indices
// starting at 0 (not duplicates that would fail LoadTraceV2SessionBlueprints).
func TestConvertSWESmith_TrajIDAsSessionKey(t *testing.T) {
	msgs := func(n int) []sweSmithMessage {
		var out []sweSmithMessage
		for i := 0; i < n; i++ {
			out = append(out, sweSmithMessage{Role: "user", Content: "hello world"})
			out = append(out, sweSmithMessage{Role: "assistant", Content: "ok"})
		}
		return out
	}

	// GIVEN: two items with the same InstanceID but different TrajIDs
	items := []sweSmithItem{
		{Messages: msgs(2), Metadata: sweSmithMetadata{InstanceID: "issue-42", TrajID: "traj-A"}},
		{Messages: msgs(3), Metadata: sweSmithMetadata{InstanceID: "issue-42", TrajID: "traj-B"}},
	}
	path := writeSWESmithFile(t, items)

	// WHEN converting
	_, records, err := ConvertSWESmith(path, 0, 30_000_000)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// THEN: session IDs should be "traj-A" and "traj-B" (not both "issue-42")
	sessionRounds := make(map[string][]int)
	for _, r := range records {
		sessionRounds[r.SessionID] = append(sessionRounds[r.SessionID], r.RoundIndex)
	}

	if len(sessionRounds) != 2 {
		t.Fatalf("got %d unique sessions, want 2; sessions: %v", len(sessionRounds), sessionRounds)
	}
	if _, ok := sessionRounds["traj-A"]; !ok {
		t.Error("session traj-A missing")
	}
	if _, ok := sessionRounds["traj-B"]; !ok {
		t.Error("session traj-B missing")
	}

	// Round indices within each session must be consecutive from 0.
	for sid, rounds := range sessionRounds {
		for i, ri := range rounds {
			if ri != i {
				t.Errorf("session %q: round_index[%d] = %d, want %d (non-consecutive)", sid, i, ri, i)
			}
		}
	}
}

// TestConvertSWESmith_EmptyPath_ReturnsError verifies path validation.
func TestConvertSWESmith_EmptyPath_ReturnsError(t *testing.T) {
	_, _, err := ConvertSWESmith("", 0, 30_000_000)
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

// TestConvertSWESmith_ZeroTurnGap_ReturnsError verifies turn-gap-us validation.
func TestConvertSWESmith_ZeroTurnGap_ReturnsError(t *testing.T) {
	_, _, err := ConvertSWESmith("some/path", 0, 0)
	if err == nil {
		t.Fatal("expected error for zero turn-gap-us")
	}
}

// ---------- ConvertChatTrace ----------

func writeChatTraceFile(t *testing.T, records []chatTraceRecord) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create chat trace file: %v", err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatalf("encode chat trace record: %v", err)
		}
	}
	return path
}

// TestConvertChatTrace_MultiTurnSession_ConsecutiveRounds verifies that a two-turn
// session produces round indices 0 and 1, with prefix length derived from hash_id
// overlap.
func TestConvertChatTrace_MultiTurnSession_ConsecutiveRounds(t *testing.T) {
	// GIVEN: two turns sharing 2 common leading hash blocks
	turn1 := chatTraceRecord{
		ChatID: 1, ParentChatID: -1, Timestamp: 0.0,
		InputLength: 64, OutputLength: 16, Turn: 1,
		HashIDs: []int{10, 20, 30, 40},
	}
	turn2 := chatTraceRecord{
		ChatID: 2, ParentChatID: 1, Timestamp: 1.0,
		InputLength: 80, OutputLength: 20, Turn: 2,
		HashIDs: []int{10, 20, 50, 60},
	}
	path := writeChatTraceFile(t, []chatTraceRecord{turn1, turn2})

	// WHEN converting
	_, records, err := ConvertChatTrace(path, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(records) != 2 {
		t.Fatalf("got %d records, want 2", len(records))
	}

	// THEN: rounds are consecutive
	if records[0].RoundIndex != 0 {
		t.Errorf("round 0: RoundIndex = %d, want 0", records[0].RoundIndex)
	}
	if records[1].RoundIndex != 1 {
		t.Errorf("round 1: RoundIndex = %d, want 1", records[1].RoundIndex)
	}

	// Prefix length for turn2 = 2 common blocks × 16 tokens/block = 32
	if records[1].PrefixLength != 32 {
		t.Errorf("turn2 prefix_length = %d, want 32", records[1].PrefixLength)
	}
}

// TestConvertChatTrace_EmptyPath_ReturnsError verifies path validation.
func TestConvertChatTrace_EmptyPath_ReturnsError(t *testing.T) {
	_, _, err := ConvertChatTrace("", 0)
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestConvertInferencePerf_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertInferencePerf("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}
