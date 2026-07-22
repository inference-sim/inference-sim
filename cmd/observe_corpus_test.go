package cmd

import (
	"strings"
	"testing"
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
