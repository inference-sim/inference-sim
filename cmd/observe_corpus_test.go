package cmd

import (
	"path/filepath"
	"strings"
	"testing"

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
