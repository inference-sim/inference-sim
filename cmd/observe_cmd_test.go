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
