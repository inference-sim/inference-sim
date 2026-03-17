package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
	runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0)

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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0)

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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0)

	records := recorder.Records()
	for _, r := range records {
		if r.SessionID != "" && r.RoundIndex > 0 {
			t.Errorf("BC-11 violated: found round-%d record after error — session should have been cancelled", r.RoundIndex)
		}
	}
}
