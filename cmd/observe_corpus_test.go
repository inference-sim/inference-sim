package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestValidateObserveCorpusFlags(t *testing.T) {
	cases := []struct {
		name                          string
		concurrentSessions, totalSess int
		corpusHeader, corpusData      string
		workload, workloadSpec        string
		rateChanged                   bool
		concurrency                   int
		wantErrSubstr                 string // "" = expect valid
	}{
		{name: "spec-mode untouched", workload: "chatbot", rateChanged: true, wantErrSubstr: ""},
		{name: "valid corpus", concurrentSessions: 4, totalSess: 20, corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: ""},
		{name: "corpus needs both files", concurrentSessions: 4, corpusHeader: "h.yaml", wantErrSubstr: "--corpus-data"},
		{name: "corpus + concurrency conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", concurrency: 8, wantErrSubstr: "--concurrency"},
		{name: "corpus + workload conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workload: "chatbot", wantErrSubstr: "--workload"},
		{name: "corpus + workload-spec conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workloadSpec: "w.yaml", wantErrSubstr: "--workload-spec"},
		{name: "corpus + rate conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", rateChanged: true, wantErrSubstr: "--rate"},
		{name: "corpus files without concurrent-sessions", corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: "--concurrent-sessions"},
		{name: "total-sessions without concurrent-sessions", totalSess: 10, wantErrSubstr: "--concurrent-sessions"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := validateObserveCorpusFlags(tc.concurrentSessions, tc.totalSess, tc.corpusHeader, tc.corpusData, tc.workload, tc.workloadSpec, tc.rateChanged, tc.concurrency)
			if tc.wantErrSubstr == "" && got != "" {
				t.Errorf("expected valid, got error %q", got)
			}
			if tc.wantErrSubstr != "" && !strings.Contains(got, tc.wantErrSubstr) {
				t.Errorf("error %q does not mention %q", got, tc.wantErrSubstr)
			}
		})
	}
}

func TestBuildObserveCorpusPool_DuplicatesToTarget(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")

	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 0, InputTokens: 120, OutputTokens: 12, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 5, 42)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}
	if driver.TotalSessions() != 5 {
		t.Errorf("TotalSessions = %d, want 5 (duplicate-to-fill)", driver.TotalSessions())
	}
	if len(initial) != 2 {
		t.Errorf("initial requests = %d, want 2 (concurrent-sessions)", len(initial))
	}
}

func TestBuildObserveCorpusPool_EmptyCorpusErrors(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "empty.yaml")
	dataPath := filepath.Join(dir, "empty.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	if err := workload.ExportTraceV2(header, []workload.TraceRecord{}, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	_, _, err := buildObserveCorpusPool(headerPath, dataPath, 2, 4, 42)
	if err == nil {
		t.Fatal("expected error for empty corpus, got nil")
	}
}

// TestObserveCorpusMode_DrainsAllSessions is the load-bearing corpus-mode test:
// a 2-session corpus scaled to --total-sessions 6 at --concurrent-sessions 2
// must dispatch and complete exactly 6 sessions against the (mock) server, with
// the dispatch loop draining to completion (not hanging). Single-round sessions
// ⇒ sessions == distinct recorded SessionIDs. This proves refill-on-terminate:
// the initial 2 are counted via takePreGen, each terminating session's refill
// replaces it (serializer does not decrement while a follow-up is returned), and
// only the final 2 decrement to 0 — so all 6 run and the loop exits.
func TestObserveCorpusMode_DrainsAllSessions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10},
		})
	}))
	defer server.Close()

	// 2 single-round sessions → pool duplicates to 6.
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 0, InputTokens: 120, OutputTokens: 12, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 6, 42)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// Guard against a hang (the failure mode a broken active-session count would
	// cause): run the orchestrator in a goroutine and fail if it does not return.
	done := make(chan struct{})
	go func() {
		defer close(done)
		runObserveOrchestrator(context.Background(), client, recorder, driver,
			cluster.NewSliceRequestSource(initial), true, 2, 0, nil, nil, false, false, 1.0)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("orchestrator did not drain within 30s — pool likely stalled (active-session accounting)")
	}

	// Exactly 6 distinct sessions must have completed.
	sessions := make(map[string]bool)
	for _, rec := range recorder.Records() {
		if rec.SessionID != "" {
			sessions[rec.SessionID] = true
		}
	}
	if len(sessions) != 6 {
		t.Errorf("distinct completed sessions = %d, want 6 (duplicate-to-fill + refill drain)", len(sessions))
	}
}
