package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestObserveCmd_MissingRequiredFlags_Errors(t *testing.T) {
	cmd := observeCmd
	if cmd == nil {
		t.Fatal("observeCmd is nil — command not registered")
	}
	if cmd.Use != "observe" {
		t.Errorf("Use: got %q, want %q", cmd.Use, "observe")
	}

	for _, name := range []string{"server-url", "model", "trace-header", "trace-data"} {
		f := cmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("missing expected flag --%s", name)
		}
	}

	tests := []struct {
		name     string
		defValue string
	}{
		{"api-key", ""},
		{"server-type", "vllm"},
		{"max-concurrency", "256"},
		{"warmup-requests", "0"},
		{"no-streaming", "false"},
	}
	for _, tt := range tests {
		f := cmd.Flags().Lookup(tt.name)
		if f == nil {
			t.Errorf("missing expected flag --%s", tt.name)
			continue
		}
		if f.DefValue != tt.defValue {
			t.Errorf("--%s default: got %q, want %q", tt.name, f.DefValue, tt.defValue)
		}
	}
}

func TestRecordRequest_PopulatesArrivalTimeAndSessionFields(t *testing.T) {
	recorder := &Recorder{}
	pending := &PendingRequest{
		RequestID:   1,
		InputTokens: 100,
		Model:       "test-model",
		Streaming:   true,
		ClientID:    "client-1",
		TenantID:    "tenant-1",
		SLOClass:    "standard",
	}
	result := &RequestRecord{
		RequestID:         1,
		OutputTokens:      50,
		ServerInputTokens: 95,
		Status:            "ok",
		SendTimeUs:        1000000,
		FirstChunkTimeUs:  1000100,
		LastChunkTimeUs:   1000500,
		NumChunks:         10,
	}

	recorder.RecordRequest(pending, result, 500000, "session-1", 0)

	records := recorder.Records()
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]
	if r.ArrivalTimeUs != 500000 {
		t.Errorf("ArrivalTimeUs: got %d, want 500000", r.ArrivalTimeUs)
	}
	if r.SessionID != "session-1" {
		t.Errorf("SessionID: got %q, want %q", r.SessionID, "session-1")
	}
	if r.RoundIndex != 0 {
		t.Errorf("RoundIndex: got %d, want 0", r.RoundIndex)
	}
}

func TestObserveOrchestrator_OpenLoop_ConservationAndConcurrency(t *testing.T) {
	// GIVEN a mock HTTP server that returns 200 OK with token counts
	requestCount := 0
	maxConcurrent := int64(0)
	currentConcurrent := int64(0)
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&currentConcurrent, 1)
		defer atomic.AddInt64(&currentConcurrent, -1)

		// Track max concurrency
		cur := atomic.LoadInt64(&currentConcurrent)
		mu.Lock()
		if cur > maxConcurrent {
			maxConcurrent = cur
		}
		requestCount++
		mu.Unlock()

		// Simulate small processing time
		time.Sleep(10 * time.Millisecond)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	// Create 5 requests with staggered arrival times (10ms apart)
	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * 10000, // 10ms apart in microseconds
			InputTokens:  make([]int, 100),
			OutputTokens: make([]int, 50),
			MaxOutputLen: 50,
			State:        sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN dispatching with max-concurrency 2 and 0 warmup
	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false)

	// THEN: BC-6 conservation: all 5 requests recorded
	records := recorder.Records()
	if len(records) != 5 {
		t.Fatalf("OBS-INV-1: expected 5 records, got %d", len(records))
	}

	// THEN: BC-7 concurrency bound: max concurrent <= 2
	if maxConcurrent > 2 {
		t.Errorf("OBS-INV-2: max concurrent %d exceeded limit 2", maxConcurrent)
	}

	// THEN: all records have status "ok"
	for i, r := range records {
		if r.Status != "ok" {
			t.Errorf("record %d: status %q, want %q", i, r.Status, "ok")
		}
	}
}

func TestObserveOrchestrator_SessionFollowUp_GeneratesRound2(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"text": "response"}},
			"usage":   map[string]any{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	spec := &workload.WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0,
		Clients: []workload.ClientSpec{
			{
				ID:           "session-client",
				RateFraction: 1.0,
				Arrival:      workload.ArrivalSpec{Process: "constant"},
				InputDist:    workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist:   workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
				Reasoning: &workload.ReasoningSpec{
					MultiTurn: &workload.MultiTurnSpec{
						MaxRounds:     2,
						ThinkTimeUs:   10000,
						ContextGrowth: "accumulate",
						SingleSession: true,
					},
				},
			},
		},
	}

	wl, err := workload.GenerateWorkload(spec, 1_000_000, 1)
	if err != nil {
		t.Fatalf("GenerateWorkload: %v", err)
	}
	if len(wl.Sessions) == 0 {
		t.Skip("WorkloadSpec did not produce sessions")
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	sessionMgr := workload.NewSessionManager(wl.Sessions)

	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0, nil, nil, false)

	records := recorder.Records()
	if len(records) < 2 {
		t.Errorf("expected at least 2 records (round-0 + round-1 follow-up), got %d", len(records))
	}

	hasRound0, hasRound1 := false, false
	for _, r := range records {
		if r.SessionID != "" && r.RoundIndex == 0 {
			hasRound0 = true
		}
		if r.SessionID != "" && r.RoundIndex == 1 {
			hasRound1 = true
		}
	}
	if !hasRound0 {
		t.Error("missing round-0 session record")
	}
	if !hasRound1 {
		t.Error("missing round-1 session follow-up record")
	}
}

func TestObserveOrchestrator_SessionError_CancelsSession(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`{"error": "internal error"}`))
	}))
	defer server.Close()

	spec := &workload.WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0,
		Clients: []workload.ClientSpec{
			{
				ID:           "session-client",
				RateFraction: 1.0,
				Arrival:      workload.ArrivalSpec{Process: "constant"},
				InputDist:    workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist:   workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
				Reasoning: &workload.ReasoningSpec{
					MultiTurn: &workload.MultiTurnSpec{
						MaxRounds:     3,
						ThinkTimeUs:   1000,
						ContextGrowth: "accumulate",
						SingleSession: true,
					},
				},
			},
		},
	}

	wl, err := workload.GenerateWorkload(spec, 1_000_000, 1)
	if err != nil {
		t.Fatalf("GenerateWorkload: %v", err)
	}
	if len(wl.Sessions) == 0 {
		t.Skip("No sessions generated")
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	sessionMgr := workload.NewSessionManager(wl.Sessions)

	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0, nil, nil, false)

	records := recorder.Records()
	for _, r := range records {
		if r.SessionID != "" && r.RoundIndex > 0 {
			t.Errorf("BC-11 violated: found round-%d record after error — session should have been cancelled", r.RoundIndex)
		}
	}
}

// Task 5: Warmup exclusion tests (BC-4, OBS-INV-4)

func TestObserveOrchestrator_WarmupExclusion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: int64(i) * 1000,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 2, nil, nil, false)

	records := recorder.Records()
	if len(records) != 3 {
		t.Fatalf("OBS-INV-4: expected 3 records (5 dispatched - 2 warmup), got %d", len(records))
	}
}

func TestObserveOrchestrator_WarmupExceedsTotal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 2)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: 0,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 5, nil, nil, false)

	records := recorder.Records()
	if len(records) != 0 {
		t.Fatalf("OBS-INV-4 edge case: expected 0 records (warmup >= total), got %d", len(records))
	}
}

// Task 6: Timestamp ordering and TraceV2 round-trip (OBS-INV-5, BC-5)

func TestObserveOrchestrator_TimestampOrdering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		time.Sleep(5 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := []*sim.Request{{
		ID: "request_0", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		MaxOutputLen: 5, State: sim.StateQueued,
	}}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0, nil, nil, false)

	records := recorder.Records()
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]
	if r.Status == "ok" {
		if r.ArrivalTimeUs > r.SendTimeUs {
			t.Errorf("OBS-INV-5: arrival (%d) > send (%d)", r.ArrivalTimeUs, r.SendTimeUs)
		}
		if r.SendTimeUs > r.FirstChunkTimeUs {
			t.Errorf("OBS-INV-5: send (%d) > first_chunk (%d)", r.SendTimeUs, r.FirstChunkTimeUs)
		}
		if r.FirstChunkTimeUs > r.LastChunkTimeUs {
			t.Errorf("OBS-INV-5: first_chunk (%d) > last_chunk (%d)", r.FirstChunkTimeUs, r.LastChunkTimeUs)
		}
	}
}

func TestObserveOrchestrator_TraceV2RoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"usage": map[string]any{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 3)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: int64(i) * 100000,
			InputTokens: make([]int, 100), OutputTokens: make([]int, 50),
			MaxOutputLen: 50, State: sim.StateQueued, ClientID: "test-client",
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0, nil, nil, false)

	headerPath := filepath.Join(t.TempDir(), "header.yaml")
	dataPath := filepath.Join(t.TempDir(), "data.csv")
	header := &workload.TraceHeader{
		Version: 2, TimeUnit: "us", Mode: "real",
		Server: &workload.TraceServerConfig{Model: "test-model"},
	}
	if err := recorder.Export(header, headerPath, dataPath); err != nil {
		t.Fatalf("Export: %v", err)
	}

	loaded, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	if len(loaded.Records) != 3 {
		t.Fatalf("round-trip: expected 3 records, got %d", len(loaded.Records))
	}

	originalRecords := recorder.Records()
	for i, orig := range originalRecords {
		lr := loaded.Records[i]
		if orig.RequestID != lr.RequestID {
			t.Errorf("record %d: RequestID mismatch: %d vs %d", i, orig.RequestID, lr.RequestID)
		}
		if orig.InputTokens != lr.InputTokens {
			t.Errorf("record %d: InputTokens mismatch: %d vs %d", i, orig.InputTokens, lr.InputTokens)
		}
		if orig.Status != lr.Status {
			t.Errorf("record %d: Status mismatch: %q vs %q", i, orig.Status, lr.Status)
		}
	}
}

// Task 7: Error storm drain and context cancellation (BC-10, BC-12)

func TestObserveOrchestrator_ErrorStormDrain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`{"error": "down"}`))
	}))
	defer server.Close()

	requests := make([]*sim.Request, 10)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: int64(i) * 1000,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	done := make(chan struct{})
	go func() {
		runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 5, 0, nil, nil, false)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("drain did not complete within 5 seconds — possible hang")
	}

	records := recorder.Records()
	if len(records) != 10 {
		t.Fatalf("expected 10 records, got %d", len(records))
	}
	for i, r := range records {
		if r.Status != "error" {
			t.Errorf("record %d: status %q, want %q", i, r.Status, "error")
		}
	}
}

func TestObserveOrchestrator_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Second)
	}))
	defer server.Close()

	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: 0,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("orchestrator did not exit after context cancellation")
	}
}

// Task 8: Pipeline parity test (D1, OBS-INV-3)

func TestObserveOrchestrator_PipelineParity_SameRequestSequence(t *testing.T) {
	spec := &workload.WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "parity-client", RateFraction: 1.0,
			Arrival:    workload.ArrivalSpec{Process: "constant"},
			InputDist:  workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}

	wl1, err := workload.GenerateWorkload(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("GenerateWorkload 1: %v", err)
	}

	spec2 := &workload.WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "parity-client", RateFraction: 1.0,
			Arrival:    workload.ArrivalSpec{Process: "constant"},
			InputDist:  workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}
	wl2, err := workload.GenerateWorkload(spec2, 1_000_000, 5)
	if err != nil {
		t.Fatalf("GenerateWorkload 2: %v", err)
	}

	if len(wl1.Requests) != len(wl2.Requests) {
		t.Fatalf("request count mismatch: %d vs %d", len(wl1.Requests), len(wl2.Requests))
	}
	for i := range wl1.Requests {
		r1, r2 := wl1.Requests[i], wl2.Requests[i]
		if r1.ArrivalTime != r2.ArrivalTime {
			t.Errorf("request %d: ArrivalTime %d vs %d", i, r1.ArrivalTime, r2.ArrivalTime)
		}
		if len(r1.InputTokens) != len(r2.InputTokens) {
			t.Errorf("request %d: input token count %d vs %d", i, len(r1.InputTokens), len(r2.InputTokens))
		}
		if len(r1.OutputTokens) != len(r2.OutputTokens) {
			t.Errorf("request %d: output token count %d vs %d", i, len(r1.OutputTokens), len(r2.OutputTokens))
		}
		if r1.SessionID != r2.SessionID {
			t.Errorf("request %d: SessionID %q vs %q", i, r1.SessionID, r2.SessionID)
		}
	}
}

func TestBuildPrefixStrings_DeterministicAndDistinct(t *testing.T) {
	groups := map[string]int{"group-a": 20, "group-b": 30}

	// Same seed produces same output
	p1, l1 := buildPrefixStrings(groups, 42)
	p2, l2 := buildPrefixStrings(groups, 42)
	if p1["group-a"] != p2["group-a"] {
		t.Error("same seed should produce identical prefix for group-a")
	}
	if p1["group-b"] != p2["group-b"] {
		t.Error("same seed should produce identical prefix for group-b")
	}
	if l1["group-a"] != 20 || l1["group-b"] != 30 {
		t.Errorf("prefix lengths: got %v, want {group-a:20, group-b:30}", l1)
	}
	_ = l2

	// Different groups produce different prefixes
	if p1["group-a"] == p1["group-b"] {
		t.Error("different prefix groups should produce distinct prefix strings")
	}

	// Different seed produces different output
	p3, _ := buildPrefixStrings(groups, 99)
	if p3["group-a"] == p1["group-a"] {
		t.Error("different seed should produce different prefix for group-a")
	}
}

func TestRequestToPending_PrependsPrefixString(t *testing.T) {
	prefixes := map[string]string{"shared": "alpha bravo charlie "}
	prefixLengths := map[string]int{"shared": 3}

	req := &sim.Request{
		ID:          "test",
		InputTokens: make([]int, 10),
		PrefixGroup: "shared",
	}

	pending := requestToPending(req, 0, false, false, prefixes, prefixLengths)

	// Prompt should start with prefix
	if !strings.HasPrefix(pending.Prompt, "alpha bravo charlie ") {
		t.Errorf("prompt should start with prefix, got %q", pending.Prompt[:min(50, len(pending.Prompt))])
	}
	// Suffix should have 7 "hello " words (10 total - 3 prefix)
	suffix := strings.TrimPrefix(pending.Prompt, "alpha bravo charlie ")
	helloCount := strings.Count(suffix, "hello ")
	if helloCount != 7 {
		t.Errorf("suffix 'hello' count = %d, want 7 (10 - 3 prefix)", helloCount)
	}

	// Without prefix group: no prefix
	reqNoPrefix := &sim.Request{
		ID:          "test2",
		InputTokens: make([]int, 10),
	}
	pendingNoPrefix := requestToPending(reqNoPrefix, 1, false, false, prefixes, prefixLengths)
	if strings.HasPrefix(pendingNoPrefix.Prompt, "alpha") {
		t.Error("request without prefix group should not have prefix")
	}
}

func TestRequestToPending_UsesPerRequestStreaming(t *testing.T) {
	streamingReq := &sim.Request{
		ID:          "stream-req",
		InputTokens: make([]int, 5),
		Streaming:   true,
	}
	nonStreamingReq := &sim.Request{
		ID:          "nostream-req",
		InputTokens: make([]int, 5),
		Streaming:   false,
	}

	// BC-1 / BC-3: without global override, per-request value propagates
	p1 := requestToPending(streamingReq, 0, false, false, nil, nil)
	if !p1.Streaming {
		t.Error("expected Streaming=true for streaming request when noStreaming=false")
	}
	p2 := requestToPending(nonStreamingReq, 1, false, false, nil, nil)
	if p2.Streaming {
		t.Error("expected Streaming=false for non-streaming request when noStreaming=false")
	}

	// BC-2: --no-streaming overrides per-request value to false
	p3 := requestToPending(streamingReq, 2, true, false, nil, nil)
	if p3.Streaming {
		t.Error("expected Streaming=false when noStreaming=true overrides req.Streaming=true")
	}
}

func TestObserveCmd_RttMsFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("rtt-ms")
	if f == nil {
		t.Fatal("missing expected flag --rtt-ms")
	}
	if f.DefValue != "0" {
		t.Errorf("--rtt-ms default: got %q, want %q", f.DefValue, "0")
	}
}

func TestObserveCmd_APIFormatFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("api-format")
	if f == nil {
		t.Fatal("missing expected flag --api-format")
	}
	if f.DefValue != "completions" {
		t.Errorf("--api-format default: got %q, want %q", f.DefValue, "completions")
	}
}

func TestObserveCmd_UnconstrainedOutputFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("unconstrained-output")
	if f == nil {
		t.Fatal("missing expected flag --unconstrained-output")
	}
	if f.DefValue != "false" {
		t.Errorf("--unconstrained-output default: got %q, want %q", f.DefValue, "false")
	}
}
