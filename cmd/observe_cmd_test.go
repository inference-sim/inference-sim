package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
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
	runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false, 0, false, 1.0)

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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0, nil, nil, false, 0, false, 1.0)

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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0, nil, nil, false, 0, false, 1.0)

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
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 2, nil, nil, false, 0, false, 1.0)

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
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 5, nil, nil, false, 0, false, 1.0)

	records := recorder.Records()
	if len(records) != 0 {
		t.Fatalf("OBS-INV-4 edge case: expected 0 records (warmup >= total), got %d", len(records))
	}
}

func TestObserveOrchestrator_RecordITL_CapturesChunkTimestamps(t *testing.T) {
	// GIVEN a streaming server that returns 3 SSE chunks
	// WHEN observeOrchestrator is called with recordITL=true
	// THEN ITL records are captured with per-chunk timestamps
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		// Chunk 0
		_, _ = fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"a"}}]}`)
		flusher.Flush()
		// Chunk 1
		_, _ = fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"b"}}]}`)
		flusher.Flush()
		// Chunk 2 with usage
		_, _ = fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"c"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}`)
		flusher.Flush()
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	requests := []*sim.Request{
		{
			ID: "request_0", ArrivalTime: 0,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 3),
			MaxOutputLen: 3, State: sim.StateQueued, Streaming: true,
		},
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0, nil, nil, false, 0, true, 1.0)

	// THEN ITL records are captured
	itlRecords := recorder.ITLRecords()
	if len(itlRecords) != 3 {
		t.Fatalf("expected 3 ITL records (3 chunks), got %d", len(itlRecords))
	}
	// Verify structure: request 0, chunk indices 0,1,2
	for i, rec := range itlRecords {
		if rec.RequestID != 0 {
			t.Errorf("ITL record %d: RequestID = %d, want 0", i, rec.RequestID)
		}
		if rec.ChunkIndex != i {
			t.Errorf("ITL record %d: ChunkIndex = %d, want %d", i, rec.ChunkIndex, i)
		}
		if rec.TimestampUs <= 0 {
			t.Errorf("ITL record %d: TimestampUs = %d, want > 0", i, rec.TimestampUs)
		}
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
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0, nil, nil, false, 0, false, 1.0)

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
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0, nil, nil, false, 0, false, 1.0)

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
		runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 5, 0, nil, nil, false, 0, false, 1.0)
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
		runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0, nil, nil, false, 0, false, 1.0)
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
	p1, l1 := buildPrefixStrings(groups, 42, 1.0)
	p2, l2 := buildPrefixStrings(groups, 42, 1.0)
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
	p3, _ := buildPrefixStrings(groups, 99, 1.0)
	if p3["group-a"] == p1["group-a"] {
		t.Error("different seed should produce different prefix for group-a")
	}
}

func TestRequestToPending_PrependsPrefixString(t *testing.T) {
	prefixes := map[string]string{"shared": "alpha bravo charlie "}
	prefixLengths := map[string]int{"shared": 3}

	req := &sim.Request{
		ID:           "test",
		InputTokens:  make([]int, 10),
		PrefixGroup:  "shared",
		PrefixLength: 64,
	}

	pending := requestToPending(req, 0, false, false, 0, prefixes, prefixLengths, 1.0)

	// PrefixGroup and PrefixLength propagated to PendingRequest
	if pending.PrefixGroup != "shared" {
		t.Errorf("PrefixGroup = %q, want %q", pending.PrefixGroup, "shared")
	}
	if pending.PrefixLength != 64 {
		t.Errorf("PrefixLength = %d, want 64", pending.PrefixLength)
	}

	// Prompt should start with prefix
	if !strings.HasPrefix(pending.Prompt, "alpha bravo charlie ") {
		t.Errorf("prompt should start with prefix, got %q", pending.Prompt[:min(50, len(pending.Prompt))])
	}
	// Suffix should have 7 words (10 total - 3 prefix), drawn from vocabulary
	suffix := strings.TrimPrefix(pending.Prompt, "alpha bravo charlie ")
	suffixWords := strings.Fields(suffix)
	if len(suffixWords) != 7 {
		t.Errorf("suffix word count = %d, want 7 (10 - 3 prefix)", len(suffixWords))
	}

	// Without prefix group: no prefix
	reqNoPrefix := &sim.Request{
		ID:          "test2",
		InputTokens: []int{5, 15, 25, 35, 45, 55, 65, 75, 85, 95},
	}
	pendingNoPrefix := requestToPending(reqNoPrefix, 1, false, false, 0, prefixes, prefixLengths, 1.0)
	// Without prefix group, prompt should not start with the group prefix string
	if strings.HasPrefix(pendingNoPrefix.Prompt, "alpha bravo charlie ") {
		t.Error("request without prefix group should not have prefix group's prefix string")
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
	p1 := requestToPending(streamingReq, 0, false, false, 0, nil, nil, 1.0)
	if !p1.Streaming {
		t.Error("expected Streaming=true for streaming request when noStreaming=false")
	}
	p2 := requestToPending(nonStreamingReq, 1, false, false, 0, nil, nil, 1.0)
	if p2.Streaming {
		t.Error("expected Streaming=false for non-streaming request when noStreaming=false")
	}

	// BC-2: --no-streaming overrides per-request value to false
	p3 := requestToPending(streamingReq, 2, true, false, 0, nil, nil, 1.0)
	if p3.Streaming {
		t.Error("expected Streaming=false when noStreaming=true overrides req.Streaming=true")
	}
}

func TestCalibratePrefixTokenRatio_ReturnsRatio(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "length"},
			},
			"usage": map[string]interface{}{
				"prompt_tokens": 167.0, "completion_tokens": 1.0,
			},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	expected := 167.0 / float64(calibrationWordCount)
	if math.Abs(ratio-expected) > 0.01 {
		t.Errorf("ratio = %.4f, want %.4f", ratio, expected)
	}
}

func TestCalibratePrefixTokenRatio_FallbackOnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	if ratio != 1.0 {
		t.Errorf("ratio = %.4f, want 1.0 (fallback)", ratio)
	}
}

func TestBuildPrefixStrings_ScalesWordCount(t *testing.T) {
	groups := map[string]int{"test-group": 1000}

	// With ratio 1.0 (no scaling): 1000 words
	prefixes1, lengths1 := buildPrefixStrings(groups, 42, 1.0)
	words1 := strings.Fields(prefixes1["test-group"])

	// With ratio 1.67: round(1000/1.67) = 599 words
	prefixes2, lengths2 := buildPrefixStrings(groups, 42, 1.67)
	words2 := strings.Fields(prefixes2["test-group"])

	if len(words1) != 1000 {
		t.Errorf("ratio=1.0: word count = %d, want 1000", len(words1))
	}
	if lengths1["test-group"] != 1000 {
		t.Errorf("ratio=1.0: prefixLengths = %d, want 1000 (target tokens)", lengths1["test-group"])
	}

	expectedWords := int(math.Round(1000.0 / 1.67))
	if len(words2) != expectedWords {
		t.Errorf("ratio=1.67: word count = %d, want %d", len(words2), expectedWords)
	}
	if lengths2["test-group"] != 1000 {
		t.Errorf("ratio=1.67: prefixLengths = %d, want 1000 (target tokens)", lengths2["test-group"])
	}
}

func TestCalibratePrefixTokenRatio_FallbackOnOutOfBounds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "length"},
			},
			"usage": map[string]interface{}{
				"prompt_tokens": 50000.0, "completion_tokens": 1.0,
			},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	if ratio != 1.0 {
		t.Errorf("ratio = %.4f, want 1.0 (fallback for out-of-bounds)", ratio)
	}
}

func TestCalibratePrefixTokenRatio_FallbackOnLowRatio(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "length"},
			},
			"usage": map[string]interface{}{
				"prompt_tokens": 50.0, "completion_tokens": 1.0,
			},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	if ratio != 1.0 {
		t.Errorf("ratio = %.4f, want 1.0 (fallback for ratio < 1.0)", ratio)
	}
}

func TestRequestToPending_SuffixUsesTokenCountNotWordCount(t *testing.T) {
	// Build prefix with tokensPerWord=2.0: 100 target tokens → 50 words in prefix string
	groups := map[string]int{"scaled": 100}
	prefixes, prefixLengths := buildPrefixStrings(groups, 42, 2.0)

	// prefixLengths should store target token count (100), not word count (50)
	if prefixLengths["scaled"] != 100 {
		t.Fatalf("prefixLengths = %d, want 100 (target tokens)", prefixLengths["scaled"])
	}
	prefixWords := strings.Fields(prefixes["scaled"])
	if len(prefixWords) != 50 {
		t.Fatalf("prefix word count = %d, want 50", len(prefixWords))
	}

	suffixTokens := make([]int, 200)
	for i := range suffixTokens {
		suffixTokens[i] = i * 3
	}
	req := &sim.Request{
		ID:          "test",
		InputTokens: suffixTokens,
		PrefixGroup: "scaled",
	}
	pending := requestToPending(req, 0, false, false, 0, prefixes, prefixLengths, 1.0)

	// Suffix should have 100 words (200 total tokens - 100 prefix tokens at ratio 1.0),
	// NOT 150 (which would happen if word count leaked into prefixLengths)
	suffix := pending.Prompt[len(prefixes["scaled"]):]
	suffixWords := strings.Fields(suffix)
	if len(suffixWords) != 100 {
		t.Errorf("suffix word count = %d, want 100 (200 total tokens - 100 prefix tokens)", len(suffixWords))
	}
}

func TestRequestToPending_NoPrefixDiversePrompt(t *testing.T) {
	// Two requests with different random token IDs and no prefix group
	req1 := &sim.Request{
		ID:          "r1",
		InputTokens: []int{10, 20, 30, 40, 50},
	}
	req2 := &sim.Request{
		ID:          "r2",
		InputTokens: []int{60, 70, 80, 90, 100},
	}

	// tokensPerWord=1.0 for direct word-to-token mapping
	p1 := requestToPending(req1, 0, false, false, 0, nil, nil, 1.0)
	p2 := requestToPending(req2, 1, false, false, 0, nil, nil, 1.0)

	// BC-2: different token IDs -> different prompts
	if p1.Prompt == p2.Prompt {
		t.Error("requests with different token IDs should produce different prompts")
	}

	// BC-1: prompt should NOT be "hello hello hello..."
	if strings.Contains(p1.Prompt, "hello") {
		t.Error("prompt should use vocabulary words, not 'hello'")
	}

	// Prompt should have 5 words (matching InputTokens length at ratio 1.0)
	words := strings.Fields(p1.Prompt)
	if len(words) != 5 {
		t.Errorf("word count = %d, want 5", len(words))
	}
}

func TestRequestToPending_WordCountScaledByTokensPerWord(t *testing.T) {
	// BC-5: with tokensPerWord=2.0, 100 tokens should produce 50 words
	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i * 7 // deterministic, diverse values
	}
	req := &sim.Request{
		ID:          "scaled",
		InputTokens: tokens,
	}

	pending := requestToPending(req, 0, false, false, 0, nil, nil, 2.0)
	words := strings.Fields(pending.Prompt)
	if len(words) != 50 {
		t.Errorf("word count = %d, want 50 (100 tokens / 2.0 tokensPerWord)", len(words))
	}
}

func TestRequestToPending_UnknownPrefixGroupFallback(t *testing.T) {
	// When PrefixGroup is set but not found in prefixes map, the prompt
	// should fall back to tokensToPrompt (diverse vocabulary), not "hello".
	prefixes := map[string]string{"groupA": "some prefix "}
	prefixLengths := map[string]int{"groupA": 5}

	req := &sim.Request{
		ID:          "fallback",
		InputTokens: []int{3, 17, 42, 88, 61},
		PrefixGroup: "unknownGroup",
	}
	pending := requestToPending(req, 0, false, false, 0, prefixes, prefixLengths, 1.0)

	if strings.Contains(pending.Prompt, "hello") {
		t.Error("unknown prefix group fallback should use vocabulary words, not 'hello'")
	}
	words := strings.Fields(pending.Prompt)
	if len(words) != 5 {
		t.Errorf("word count = %d, want 5", len(words))
	}
	// Should NOT start with groupA's prefix
	if strings.HasPrefix(pending.Prompt, "some prefix") {
		t.Error("unknown group should not use another group's prefix")
	}
}

func TestTokensToPrompt_DiverseWords(t *testing.T) {
	tokens := []int{0, 1, 50, 99, 100, 200}
	result := tokensToPrompt(tokens, 6)
	words := strings.Fields(result)

	if len(words) != 6 {
		t.Fatalf("word count = %d, want 6", len(words))
	}

	// Each word must be from prefixVocabulary
	vocabSet := make(map[string]bool)
	for _, w := range prefixVocabulary {
		vocabSet[w] = true
	}
	for i, w := range words {
		if !vocabSet[w] {
			t.Errorf("word[%d] = %q, not in prefixVocabulary", i, w)
		}
	}

	// Words should not all be the same (unlike the old "hello hello hello" bug)
	uniqueWords := make(map[string]bool)
	for _, w := range words {
		uniqueWords[w] = true
	}
	if len(uniqueWords) < 2 {
		t.Errorf("expected diverse words, got %d unique out of %d", len(uniqueWords), len(words))
	}
}

func TestTokensToPrompt_EmptyTokens(t *testing.T) {
	// Edge case: empty token array with wordCount=1 should not panic
	result := tokensToPrompt(nil, 1)
	words := strings.Fields(result)
	if len(words) != 1 {
		t.Fatalf("word count = %d, want 1", len(words))
	}
	// Word must be from vocabulary (fallback path uses i % vocabLen)
	vocabSet := make(map[string]bool)
	for _, w := range prefixVocabulary {
		vocabSet[w] = true
	}
	if !vocabSet[words[0]] {
		t.Errorf("word = %q, not in prefixVocabulary", words[0])
	}
}

func TestTokensToPrompt_NegativeTokenIDs(t *testing.T) {
	// Negative token IDs should not panic (Go % preserves sign).
	tokens := []int{-1, -100, -50, 7}
	result := tokensToPrompt(tokens, 4)
	words := strings.Fields(result)
	if len(words) != 4 {
		t.Fatalf("word count = %d, want 4", len(words))
	}
	vocabSet := make(map[string]bool)
	for _, w := range prefixVocabulary {
		vocabSet[w] = true
	}
	for i, w := range words {
		if !vocabSet[w] {
			t.Errorf("word[%d] = %q, not in prefixVocabulary", i, w)
		}
	}
}

func TestRequestToPending_WordCountClampedToOne(t *testing.T) {
	// 1 token with tokensPerWord=2.0 → round(0.5)=0 → clamped to 1
	req := &sim.Request{
		ID:          "tiny",
		InputTokens: []int{42},
	}
	pending := requestToPending(req, 0, false, false, 0, nil, nil, 2.0)
	words := strings.Fields(pending.Prompt)
	if len(words) != 1 {
		t.Errorf("word count = %d, want 1 (clamped minimum)", len(words))
	}
}

func TestBuildPrefixStrings_EmptyGroupsNoWork(t *testing.T) {
	// BC-5: no prefix groups → no prefix strings, no calibration needed
	groups := map[string]int{}
	prefixes, prefixLengths := buildPrefixStrings(groups, 42, 1.0)
	if len(prefixes) != 0 {
		t.Errorf("expected empty prefixes, got %d", len(prefixes))
	}
	if len(prefixLengths) != 0 {
		t.Errorf("expected empty prefixLengths, got %d", len(prefixLengths))
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

func TestObserveCmd_MinTokensFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("min-tokens")
	if f == nil {
		t.Fatal("missing expected flag --min-tokens")
	}
	if f.DefValue != "0" {
		t.Errorf("--min-tokens default: got %q, want %q", f.DefValue, "0")
	}
}

func TestRequestToPending_MinTokensPropagated(t *testing.T) {
	req := &sim.Request{
		ID:          "min-tok",
		InputTokens: make([]int, 5),
	}
	p := requestToPending(req, 0, false, false, 128, nil, nil, 1.0)
	if p.MinTokens != 128 {
		t.Errorf("MinTokens = %d, want 128", p.MinTokens)
	}
}

func TestSend_MinTokensInBody(t *testing.T) {
	var receivedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"text": "ok", "finish_reason": "length"},
			},
			"usage": map[string]interface{}{"completion_tokens": 10, "prompt_tokens": 5},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")

	// min_tokens > 0: included in body
	req := &PendingRequest{Prompt: "hello", MaxOutputTokens: 256, MinTokens: 128}
	_, _ = client.Send(context.Background(), req)
	if v, ok := receivedBody["min_tokens"]; !ok {
		t.Error("min_tokens not found in request body")
	} else if int(v.(float64)) != 128 {
		t.Errorf("min_tokens = %v, want 128", v)
	}

	// min_tokens = 0: omitted from body
	receivedBody = nil
	req2 := &PendingRequest{Prompt: "hello", MaxOutputTokens: 256, MinTokens: 0}
	_, _ = client.Send(context.Background(), req2)
	if _, ok := receivedBody["min_tokens"]; ok {
		t.Error("min_tokens should be omitted when 0")
	}
}

// TestObserveDistributionDefaults_MatchRunDefaults verifies that observeCmd's eight
// distribution flag defaults are identical to runCmd's defaults (BC-1).
//
// Why this test matters: the observe→calibrate loop compares simulator output against
// real-server output. If the two commands generate different workload shapes by default,
// calibration conflates model error with distribution skew.
//
// If someone changes a constant in root.go, both commands change together and this
// test still passes. If someone accidentally re-introduces a hardcoded literal with a
// different value, this test fails.
func TestObserveDistributionDefaults_MatchRunDefaults(t *testing.T) {
	tests := []struct {
		flag    string
		wantInt int
	}{
		{"prompt-tokens", defaultPromptMean},
		{"prompt-tokens-stdev", defaultPromptStdev},
		{"prompt-tokens-min", defaultPromptMin},
		{"prompt-tokens-max", defaultPromptMax},
		{"output-tokens", defaultOutputMean},
		{"output-tokens-stdev", defaultOutputStdev},
		{"output-tokens-min", defaultOutputMin},
		{"output-tokens-max", defaultOutputMax},
	}
	for _, tt := range tests {
		f := observeCmd.Flags().Lookup(tt.flag)
		if f == nil {
			t.Fatalf("flag --%s not found on observeCmd", tt.flag)
		}
		got, err := strconv.Atoi(f.DefValue)
		if err != nil {
			t.Fatalf("--%s DefValue %q is not an int: %v", tt.flag, f.DefValue, err)
		}
		if got != tt.wantInt {
			t.Errorf("--%s default: got %d, want %d (must match runCmd default)",
				tt.flag, got, tt.wantInt)
		}
	}

	// BC-3: num-requests intentional asymmetry — observe must default to 0, not 100.
	f := observeCmd.Flags().Lookup("num-requests")
	if f == nil {
		t.Fatal("flag --num-requests not found on observeCmd")
	}
	if f.DefValue != "0" {
		t.Errorf("--num-requests default: got %q, want \"0\" (intentional asymmetry with run's 100)", f.DefValue)
	}
}

// TestObserveDistributionDefaults_NoHardcodedLiterals verifies that none of the old
// literal defaults (50, 1, 2048) appear in observe_cmd.go's distribution IntVar calls
// (BC-2: single source of truth).
//
// This is a source-level scan test, following the pattern in simconfig_shared_test.go.
// It catches the scenario where someone re-introduces a literal that coincidentally has
// the same value as a constant — which the value-equality test above would miss.
func TestObserveDistributionDefaults_NoHardcodedLiterals(t *testing.T) {
	data, err := os.ReadFile("observe_cmd.go")
	if err != nil {
		t.Fatalf("cannot read observe_cmd.go: %v", err)
	}
	content := string(data)

	// These patterns are the hardcoded literals that must no longer appear as
	// the default argument in distribution flag IntVar calls.
	// Format: the flag name string followed by the old literal default.
	// Covers all 8 distribution flags:
	//   - 6 formerly-divergent flags (old values: stdev=50, min=1, max=2048)
	//   - 2 mean flags (old values: 512 — already matched run, but must use the constant)
	forbidden := []string{
		`"prompt-tokens", 512`,
		`"prompt-tokens-stdev", 50`,
		`"prompt-tokens-min", 1`,
		`"prompt-tokens-max", 2048`,
		`"output-tokens", 512`,
		`"output-tokens-stdev", 50`,
		`"output-tokens-min", 1`,
		`"output-tokens-max", 2048`,
	}
	for _, pattern := range forbidden {
		if strings.Contains(content, pattern) {
			t.Errorf("old hardcoded literal found in observe_cmd.go: %q\n"+
				"Use the distDefaults constants from root.go instead (BC-2).", pattern)
		}
	}
}

func TestValidateObserveWorkloadFlags_PresetRequiresRate(t *testing.T) {
	// BC-2: preset without --rate is rejected
	msg := validateObserveWorkloadFlags("chatbot", "", false, 0)
	if msg == "" {
		t.Fatal("expected validation error for preset without --rate, got none")
	}
	if !strings.Contains(msg, "--rate") {
		t.Errorf("error should mention --rate, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_PresetExclusiveWithSpec(t *testing.T) {
	// BC-3: preset + --workload-spec is rejected
	msg := validateObserveWorkloadFlags("chatbot", "spec.yaml", true, 0)
	if msg == "" {
		t.Fatal("expected validation error for --workload + --workload-spec, got none")
	}
	if !strings.Contains(msg, "--workload-spec") {
		t.Errorf("error should mention --workload-spec, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_PresetExclusiveWithConcurrency(t *testing.T) {
	// BC-4: preset + --concurrency is rejected
	msg := validateObserveWorkloadFlags("chatbot", "", true, 50)
	if msg == "" {
		t.Fatal("expected validation error for --workload + --concurrency, got none")
	}
	if !strings.Contains(msg, "--concurrency") {
		t.Errorf("error should mention --concurrency, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_ValidPreset_Accepted(t *testing.T) {
	// Precondition for BC-1: valid preset + rate is accepted by validator
	msg := validateObserveWorkloadFlags("chatbot", "", true, 0)
	if msg != "" {
		t.Errorf("expected no error for valid preset+rate combination, got: %q", msg)
	}
}

func TestValidateObserveWorkloadFlags_EmptyPreset_NoOp(t *testing.T) {
	// Non-preset modes must not be affected by the validator
	msg := validateObserveWorkloadFlags("", "", false, 0)
	if msg != "" {
		t.Errorf("expected no error when --workload is not set, got: %q", msg)
	}
}

// TestObserveCmd_WorkloadInputGuard_IncludesPreset verifies BC-7:
// the required-input error message lists --workload as an option.
// Uses source-level scan (acceptable here: tests the exact error message text
// that users see — a behavioral property of the CLI contract).
func TestObserveCmd_WorkloadInputGuard_IncludesPreset(t *testing.T) {
	data, err := os.ReadFile("observe_cmd.go")
	if err != nil {
		t.Fatalf("cannot read observe_cmd.go: %v", err)
	}
	content := string(data)
	// The required-input guard message must list all four input modes.
	// We check for the specific error string text, not just any occurrence of "--workload".
	wantText := "Either --workload, --workload-spec, --rate, or --concurrency is required"
	if !strings.Contains(content, wantText) {
		t.Errorf("required-input guard in observe_cmd.go must contain:\n  %q\nnot found in file", wantText)
	}
}

func TestObserveCmd_WorkloadFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("workload")
	if f == nil {
		t.Fatal("missing expected flag --workload on observeCmd")
	}
	if f.DefValue != "" {
		t.Errorf("--workload default: got %q, want %q (empty — no default preset)", f.DefValue, "")
	}
}

func TestObserveCmd_DefaultsFilepathFlag_Exists(t *testing.T) {
	f := observeCmd.Flags().Lookup("defaults-filepath")
	if f == nil {
		t.Fatal("missing expected flag --defaults-filepath on observeCmd")
	}
	if f.DefValue != "defaults.yaml" {
		t.Errorf("--defaults-filepath default: got %q, want %q", f.DefValue, "defaults.yaml")
	}
}

// TestBuildPresetSpec_MatchesPresetDefinition verifies BC-1:
// buildPresetSpec loads token distribution from defaults.yaml and passes it
// to SynthesizeFromPreset, producing a spec with correct rate and token means.
//
// This is the wiring test: it exercises the actual buildPresetSpec code path
// in observe_cmd.go, not just the workload package independently.
func TestBuildPresetSpec_MatchesPresetDefinition(t *testing.T) {
	dir := t.TempDir()
	defaultsPath := filepath.Join(dir, "defaults.yaml")
	// YAML keys must match Workload struct tags exactly (R10: KnownFields(true)).
	// prompt_tokens → PromptTokensMean, output_tokens → OutputTokensMean (see default_config.go:15,19)
	defaultsContent := `workloads:
  chatbot:
    prefix_tokens: 0
    prompt_tokens: 512
    prompt_tokens_stdev: 100
    prompt_tokens_min: 50
    prompt_tokens_max: 1024
    output_tokens: 256
    output_tokens_stdev: 50
    output_tokens_min: 10
    output_tokens_max: 512
`
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0600); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	const testRate = 5.0
	const testNumRequests = 10
	spec, errMsg := buildPresetSpec("chatbot", defaultsPath, testRate, testNumRequests)
	if errMsg != "" {
		t.Fatalf("buildPresetSpec returned error: %q", errMsg)
	}
	if spec == nil {
		t.Fatal("buildPresetSpec returned nil spec")
	}
	if len(spec.Clients) == 0 {
		t.Fatal("spec has no clients")
	}
	client := spec.Clients[0]
	// Invariant: aggregate rate matches requested rate (set on spec, not per-client)
	if spec.AggregateRate != testRate {
		t.Errorf("AggregateRate: got %v, want %v", spec.AggregateRate, testRate)
	}
	// Invariant: token means come from preset YAML (via InputDist/OutputDist params),
	// not distribution synthesis defaults
	gotPromptMean := client.InputDist.Params["mean"]
	if gotPromptMean != 512 {
		t.Errorf("InputDist.Params[mean]: got %v, want 512 (from chatbot preset)", gotPromptMean)
	}
	gotOutputMean := client.OutputDist.Params["mean"]
	if gotOutputMean != 256 {
		t.Errorf("OutputDist.Params[mean]: got %v, want 256 (from chatbot preset)", gotOutputMean)
	}
	// Invariant: num-requests bound propagated into spec
	if spec.NumRequests != int64(testNumRequests) {
		t.Errorf("spec.NumRequests: got %d, want %d", spec.NumRequests, testNumRequests)
	}
}

// TestBuildPresetSpec_ParityWithRunPresetPath verifies BC-1 cross-command parity:
// buildPresetSpec (observe path, observe_cmd.go:177-188) must produce a WorkloadSpec
// identical to the inline synthesis in root.go:1167-1173 (run path) for the same input.
//
// This is a law test: it exercises both code paths against the same Workload input and
// asserts they produce identical WorkloadSpec output. If a future change adds a field
// to Workload but updates only one code path's PresetConfig construction, this test fails.
func TestBuildPresetSpec_ParityWithRunPresetPath(t *testing.T) {
	// Write a temp defaults.yaml so buildPresetSpec (observe path) can load the preset.
	dir := t.TempDir()
	defaultsPath := filepath.Join(dir, "defaults.yaml")
	defaultsContent := `workloads:
  chatbot:
    prefix_tokens: 32
    prompt_tokens: 512
    prompt_tokens_stdev: 100
    prompt_tokens_min: 50
    prompt_tokens_max: 1024
    output_tokens: 256
    output_tokens_stdev: 50
    output_tokens_min: 10
    output_tokens_max: 512
`
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0600); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	const presetName = "chatbot"
	const testRate = 7.0
	const testNumRequests = 20

	// blis observe path: goes through buildPresetSpec -> loadPresetWorkload -> SynthesizeFromPreset
	observeSpec, errMsg := buildPresetSpec(presetName, defaultsPath, testRate, testNumRequests)
	if errMsg != "" {
		t.Fatalf("buildPresetSpec error: %q", errMsg)
	}

	// blis run path (root.go:1163-1173): loadPresetWorkload + inline PresetConfig construction.
	// Mirror the exact field mapping from root.go to catch any future divergence.
	wl := loadPresetWorkload(defaultsPath, presetName)
	if wl == nil {
		t.Fatal("loadPresetWorkload returned nil for chatbot preset")
	}
	runSpec := workload.SynthesizeFromPreset(presetName, workload.PresetConfig{
		PrefixTokens:     wl.PrefixTokens,
		PromptTokensMean: wl.PromptTokensMean, PromptTokensStdev: wl.PromptTokensStdev,
		PromptTokensMin: wl.PromptTokensMin, PromptTokensMax: wl.PromptTokensMax,
		OutputTokensMean: wl.OutputTokensMean, OutputTokensStdev: wl.OutputTokensStdev,
		OutputTokensMin: wl.OutputTokensMin, OutputTokensMax: wl.OutputTokensMax,
	}, testRate, testNumRequests)

	if runSpec == nil || observeSpec == nil {
		t.Fatal("SynthesizeFromPreset returned nil spec")
	}
	if len(runSpec.Clients) == 0 || len(observeSpec.Clients) == 0 {
		t.Fatal("spec has no clients")
	}

	// Parity invariants: both paths must produce identical WorkloadSpec
	if runSpec.AggregateRate != observeSpec.AggregateRate {
		t.Errorf("AggregateRate parity: run=%v, observe=%v", runSpec.AggregateRate, observeSpec.AggregateRate)
	}
	if runSpec.NumRequests != observeSpec.NumRequests {
		t.Errorf("NumRequests parity: run=%d, observe=%d", runSpec.NumRequests, observeSpec.NumRequests)
	}
	rc, oc := runSpec.Clients[0], observeSpec.Clients[0]
	if rc.InputDist.Params["mean"] != oc.InputDist.Params["mean"] {
		t.Errorf("PromptMean parity: run=%v, observe=%v", rc.InputDist.Params["mean"], oc.InputDist.Params["mean"])
	}
	if rc.InputDist.Params["std_dev"] != oc.InputDist.Params["std_dev"] {
		t.Errorf("PromptStdev parity: run=%v, observe=%v", rc.InputDist.Params["std_dev"], oc.InputDist.Params["std_dev"])
	}
	if rc.OutputDist.Params["mean"] != oc.OutputDist.Params["mean"] {
		t.Errorf("OutputMean parity: run=%v, observe=%v", rc.OutputDist.Params["mean"], oc.OutputDist.Params["mean"])
	}
	if rc.OutputDist.Params["std_dev"] != oc.OutputDist.Params["std_dev"] {
		t.Errorf("OutputStdev parity: run=%v, observe=%v", rc.OutputDist.Params["std_dev"], oc.OutputDist.Params["std_dev"])
	}
	if rc.InputDist.Params["min"] != oc.InputDist.Params["min"] {
		t.Errorf("PromptMin parity: run=%v, observe=%v", rc.InputDist.Params["min"], oc.InputDist.Params["min"])
	}
	if rc.InputDist.Params["max"] != oc.InputDist.Params["max"] {
		t.Errorf("PromptMax parity: run=%v, observe=%v", rc.InputDist.Params["max"], oc.InputDist.Params["max"])
	}
	if rc.OutputDist.Params["min"] != oc.OutputDist.Params["min"] {
		t.Errorf("OutputMin parity: run=%v, observe=%v", rc.OutputDist.Params["min"], oc.OutputDist.Params["min"])
	}
	if rc.OutputDist.Params["max"] != oc.OutputDist.Params["max"] {
		t.Errorf("OutputMax parity: run=%v, observe=%v", rc.OutputDist.Params["max"], oc.OutputDist.Params["max"])
	}
	if rc.PrefixLength != oc.PrefixLength {
		t.Errorf("PrefixLength parity: run=%d, observe=%d", rc.PrefixLength, oc.PrefixLength)
	}
}

// TestBuildPresetSpec_MissingDefaultsFile verifies that buildPresetSpec returns a
// user-friendly error (mentioning --workload and --defaults-filepath) when the
// defaults.yaml file does not exist, rather than a generic fatal from loadDefaultsConfig.
func TestBuildPresetSpec_MissingDefaultsFile(t *testing.T) {
	spec, errMsg := buildPresetSpec("chatbot", "/nonexistent/path/defaults.yaml", 5.0, 10)
	if spec != nil {
		t.Error("expected nil spec for missing defaults file, got non-nil")
	}
	if errMsg == "" {
		t.Fatal("expected error for missing defaults file, got empty message")
	}
	// Error must guide user toward --defaults-filepath flag
	if !strings.Contains(errMsg, "--defaults-filepath") {
		t.Errorf("error should mention --defaults-filepath, got: %q", errMsg)
	}
	if !strings.Contains(errMsg, "--workload") {
		t.Errorf("error should mention --workload, got: %q", errMsg)
	}
}

// TestBuildPresetSpec_UnknownPreset_ReturnsError verifies BC-5:
// buildPresetSpec returns a non-empty error for an undefined preset name.
// Error message must list valid preset names.
func TestBuildPresetSpec_UnknownPreset_ReturnsError(t *testing.T) {
	dir := t.TempDir()
	defaultsPath := filepath.Join(dir, "defaults.yaml")
	// Minimal valid defaults.yaml — no workloads section means all presets are undefined
	if err := os.WriteFile(defaultsPath, []byte("version: test\n"), 0600); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	spec, errMsg := buildPresetSpec("unknown-preset", defaultsPath, 5.0, 10)
	if spec != nil {
		t.Error("expected nil spec for unknown preset, got non-nil")
	}
	if errMsg == "" {
		t.Fatal("expected error for unknown preset, got empty message")
	}
	// Invariant: error lists valid preset names so users know what to pass
	for _, name := range []string{"chatbot", "summarization", "contentgen", "multidoc"} {
		if !strings.Contains(errMsg, name) {
			t.Errorf("error message should list valid preset %q, got: %q", name, errMsg)
		}
	}
}

func TestObserveCmd_ITLFlags_Defined(t *testing.T) {
	// GIVEN the observe command
	cmd := observeCmd

	// WHEN checking for ITL flags
	recordITLFlag := cmd.Flags().Lookup("record-itl")
	itlOutputFlag := cmd.Flags().Lookup("itl-output")

	// THEN both flags are defined
	if recordITLFlag == nil {
		t.Error("--record-itl flag not defined")
	}
	if itlOutputFlag == nil {
		t.Error("--itl-output flag not defined")
	}
}

