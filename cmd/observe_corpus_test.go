package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
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
		thinkTimeMs                   int
		thinkTimeDist                 string
		lazyGeneration                bool
		horizonChanged                bool
		numRequestsChanged            bool
		shuffleCorpus                 bool
		wantErrSubstr                 string // "" = expect valid
	}{
		{name: "spec-mode untouched", workload: "chatbot", rateChanged: true, wantErrSubstr: ""},
		{name: "valid corpus", concurrentSessions: 4, totalSess: 20, corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: ""},
		{name: "valid corpus with shuffle", concurrentSessions: 4, totalSess: 20, corpusHeader: "h.yaml", corpusData: "d.csv", shuffleCorpus: true, wantErrSubstr: ""},
		{name: "corpus needs both files", concurrentSessions: 4, corpusHeader: "h.yaml", wantErrSubstr: "--corpus-data"},
		{name: "corpus + concurrency conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", concurrency: 8, wantErrSubstr: "--concurrency"},
		{name: "corpus + workload conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workload: "chatbot", wantErrSubstr: "--workload"},
		{name: "corpus + workload-spec conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workloadSpec: "w.yaml", wantErrSubstr: "--workload-spec"},
		{name: "corpus + rate conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", rateChanged: true, wantErrSubstr: "--rate"},
		{name: "corpus files without concurrent-sessions", corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: "--concurrent-sessions"},
		{name: "total-sessions without concurrent-sessions", totalSess: 10, wantErrSubstr: "--concurrent-sessions"},
		{name: "corpus + think-time-ms conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", thinkTimeMs: 200, wantErrSubstr: "--think-time-ms"},
		{name: "corpus + think-time-dist conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", thinkTimeDist: "constant:value=500ms", wantErrSubstr: "--think-time-dist"},
		{name: "corpus + lazy-generation conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", lazyGeneration: true, wantErrSubstr: "--lazy-generation"},
		{name: "corpus + horizon conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", horizonChanged: true, wantErrSubstr: "--horizon"},
		{name: "corpus + num-requests conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", numRequestsChanged: true, wantErrSubstr: "--num-requests"},
		{name: "shuffle-corpus without concurrent-sessions", shuffleCorpus: true, wantErrSubstr: "--shuffle-corpus"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := validateObserveCorpusFlags(tc.concurrentSessions, tc.totalSess, tc.corpusHeader, tc.corpusData, tc.workload, tc.workloadSpec, tc.rateChanged, tc.concurrency, tc.thinkTimeMs, tc.thinkTimeDist, tc.lazyGeneration, tc.horizonChanged, tc.numRequestsChanged, tc.shuffleCorpus)
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

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 5, false, 42)
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
	_, _, err := buildObserveCorpusPool(headerPath, dataPath, 2, 4, false, 42)
	if err == nil {
		t.Fatal("expected error for empty corpus, got nil")
	}
}

// TestBuildObserveCorpusPool_MixedNonSessionErrors is the observe-side parity of
// the replay non-session guard (PR-C review follow-up: "same behavior in observe
// needs to be addressed"). A corpus that mixes session records with non-session
// (empty session_id) records cannot be pooled 1:1; buildObserveCorpusPool must
// return an actionable "no session_id" error, pre-empting BuildSessionPool's
// internal "count mismatch" wording (R1).
func TestBuildObserveCorpusPool_MixedNonSessionErrors(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "mixed.yaml")
	dataPath := filepath.Join(dir, "mixed.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	// One session record + one non-session (empty SessionID) record.
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 10, OutputTokens: 5, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "", RoundIndex: 0, InputTokens: 8, OutputTokens: 4, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	_, _, err := buildObserveCorpusPool(headerPath, dataPath, 1, 0, false, 42)
	if err == nil {
		t.Fatal("expected error for a corpus mixing session and non-session records, got nil")
	}
	if !strings.Contains(err.Error(), "no session_id") {
		t.Errorf("error should name the non-session records ('no session_id'), got: %v", err)
	}
	if strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("guard should pre-empt BuildSessionPool's internal 'count mismatch', but it surfaced: %v", err)
	}
}

// TestBuildObserveCorpusPool_ShuffleReproducibleAndReorders verifies observe's
// --shuffle-corpus (PR-C parity, #1480): the seeded permutation is reproducible
// from the same --seed, actually reorders the corpus, and preserves the set;
// shuffle=false keeps file order. observe draws from the SAME salted stream as
// `blis replay --shuffle-corpus`, so one --seed selects the same subset on both.
func TestBuildObserveCorpusPool_ShuffleReproducibleAndReorders(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	var records []workload.TraceRecord
	for i := 0; i < 6; i++ {
		records = append(records, workload.TraceRecord{
			RequestID: i, SessionID: fmt.Sprintf("s%d", i), RoundIndex: 0,
			InputTokens: 100 + i, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok",
		})
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}
	// joinIDs returns the comma-joined SessionID sequence of the initial injection
	// (concurrent == total == 6 ⇒ all injected, in queued/admission order).
	joinIDs := func(reqs []*sim.Request) string {
		s := ""
		for _, r := range reqs {
			s += r.SessionID + ","
		}
		return s
	}
	const fileOrder = "s0,s1,s2,s3,s4,s5,"

	// shuffle=false → file order.
	_, initNoShuf, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, false, 42)
	if err != nil {
		t.Fatalf("no-shuffle: %v", err)
	}
	if got := joinIDs(initNoShuf); got != fileOrder {
		t.Errorf("shuffle=false order = %q, want file order %q", got, fileOrder)
	}

	// shuffle=true, same seed twice → identical (reproducible).
	_, initA, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, true, 7)
	if err != nil {
		t.Fatalf("shuffle A: %v", err)
	}
	_, initB, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, true, 7)
	if err != nil {
		t.Fatalf("shuffle B: %v", err)
	}
	ordA, ordB := joinIDs(initA), joinIDs(initB)
	if ordA != ordB {
		t.Errorf("same seed → different order:\n %q\n %q", ordA, ordB)
	}
	// Actually reorders (seed 7 permutes 6 elements away from identity).
	if ordA == fileOrder {
		t.Errorf("shuffle=true seed=7 did not reorder; got file order %q", ordA)
	}
	// Set preserved: the same 6 sessions, none dropped or duplicated.
	seen := map[string]bool{}
	for _, r := range initA {
		seen[r.SessionID] = true
	}
	if len(seen) != 6 {
		t.Errorf("shuffle changed the session set: %d unique, want 6", len(seen))
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

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 6, false, 42)
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

// TestObserveCorpusMode_MultiRoundAccumulate drives a multi-round accumulate-mode
// corpus end-to-end through the observe orchestrator (#1487 review, recommended):
// a single 3-round session must dispatch round 0 plus both follow-ups, confirming
// the SessionManager follow-up path works over the real-server dispatch loop.
func TestObserveCorpusMode_MultiRoundAccumulate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10},
		})
	}))
	defer server.Close()

	// One 3-round accumulate session (per-round input deltas). Small recorded think
	// times keep the closed-loop wall-clock pacing fast.
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s0", RoundIndex: 1, InputTokens: 40, OutputTokens: 10, ArrivalTimeUs: 1000, ThinkTimeUs: i64p(1000), Status: "ok"},
		{RequestID: 2, SessionID: "s0", RoundIndex: 2, InputTokens: 25, OutputTokens: 10, ArrivalTimeUs: 2000, ThinkTimeUs: i64p(1000), Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 1, 1, false, 42)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	done := make(chan struct{})
	go func() {
		defer close(done)
		runObserveOrchestrator(context.Background(), client, recorder, driver,
			cluster.NewSliceRequestSource(initial), true, 1, 0, nil, nil, false, false, 1.0)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("orchestrator did not drain within 30s — multi-round follow-up dispatch likely stalled")
	}

	// All three rounds (round 0 + two accumulate follow-ups) must have dispatched.
	rounds := map[int]bool{}
	for _, rec := range recorder.Records() {
		if rec.SessionID == "s0" {
			rounds[rec.RoundIndex] = true
		}
	}
	if !rounds[0] || !rounds[1] || !rounds[2] {
		t.Errorf("session s0 dispatched rounds %v, want {0,1,2} (multi-round accumulate follow-ups)", rounds)
	}
}
