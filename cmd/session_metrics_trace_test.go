package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// traceEntry is a compact helper for building TraceRecord test fixtures.
type traceEntry struct {
	id      int
	sid     string
	round   int
	status  string
	sendUs  int64
	firstUs int64
	lastUs  int64
}

func buildTraceRecords(entries []traceEntry) []workload.TraceRecord {
	records := make([]workload.TraceRecord, len(entries))
	for i, e := range entries {
		records[i] = workload.TraceRecord{
			RequestID:        e.id,
			SessionID:        e.sid,
			RoundIndex:       e.round,
			Status:           e.status,
			SendTimeUs:       e.sendUs,
			FirstChunkTimeUs: e.firstUs,
			LastChunkTimeUs:  e.lastUs,
		}
	}
	return records
}

// BC-1 (trace): returns nil when no records have SessionID
func TestComputeSessionMetricsFromTrace_NilWhenNoSessions(t *testing.T) {
	records := buildTraceRecords([]traceEntry{
		{id: 0, status: "ok", sendUs: 1000, firstUs: 1050, lastUs: 2000},
	})
	if got := computeSessionMetricsFromTrace(records); got != nil {
		t.Errorf("expected nil for non-session trace, got %+v", got)
	}
}

// BC-2 (trace): cold/warm partitioned by RoundIndex
func TestComputeSessionMetricsFromTrace_ColdWarmPartition(t *testing.T) {
	records := buildTraceRecords([]traceEntry{
		{id: 0, sid: "s1", round: 0, status: "ok", sendUs: 1_000_000, firstUs: 1_010_000, lastUs: 1_100_000},
		{id: 1, sid: "s1", round: 1, status: "ok", sendUs: 1_200_000, firstUs: 1_205_000, lastUs: 1_300_000},
	})
	got := computeSessionMetricsFromTrace(records)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1", got.TTFTCold.Count)
	}
	wantCold := float64(1_010_000-1_000_000) / 1000.0 // 10ms
	if got.TTFTCold.Mean != wantCold {
		t.Errorf("TTFTCold.Mean: got %.3f, want %.3f", got.TTFTCold.Mean, wantCold)
	}
	if got.TTFTWarm.Count != 1 {
		t.Errorf("TTFTWarm.Count: got %d, want 1", got.TTFTWarm.Count)
	}
	wantWarm := float64(1_205_000-1_200_000) / 1000.0 // 5ms
	if got.TTFTWarm.Mean != wantWarm {
		t.Errorf("TTFTWarm.Mean: got %.3f, want %.3f", got.TTFTWarm.Mean, wantWarm)
	}
}

// BC-3 (trace): session duration = max LastChunkTimeUs − round-0 SendTimeUs (ms)
func TestComputeSessionMetricsFromTrace_SessionDuration(t *testing.T) {
	records := buildTraceRecords([]traceEntry{
		{id: 0, sid: "s1", round: 0, status: "ok", sendUs: 1_000_000, firstUs: 1_010_000, lastUs: 1_100_000},
		{id: 1, sid: "s1", round: 1, status: "ok", sendUs: 1_200_000, firstUs: 1_205_000, lastUs: 1_400_000},
	})
	got := computeSessionMetricsFromTrace(records)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	// duration = (1_400_000 - 1_000_000) / 1000 = 400ms
	wantDur := 400.0
	if got.SessionDuration.Mean != wantDur {
		t.Errorf("SessionDuration.Mean: got %.3f, want %.3f", got.SessionDuration.Mean, wantDur)
	}
}

// BC-4 (trace): non-session records excluded
func TestComputeSessionMetricsFromTrace_NonSessionExcluded(t *testing.T) {
	records := buildTraceRecords([]traceEntry{
		{id: 0, sid: "s1", round: 0, status: "ok", sendUs: 1_000_000, firstUs: 1_010_000, lastUs: 1_100_000},
		{id: 1, status: "ok", sendUs: 500_000, firstUs: 510_000, lastUs: 600_000}, // no SessionID
	})
	got := computeSessionMetricsFromTrace(records)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.SessionCount != 1 {
		t.Errorf("SessionCount: got %d, want 1", got.SessionCount)
	}
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1 (non-session must be excluded)", got.TTFTCold.Count)
	}
}

// BC-5 (trace): error records excluded from TTFT (status != "ok")
func TestComputeSessionMetricsFromTrace_ErrorRecordsExcludedFromTTFT(t *testing.T) {
	records := buildTraceRecords([]traceEntry{
		{id: 0, sid: "s1", round: 0, status: "error", sendUs: 1_000_000, firstUs: 1_010_000, lastUs: 1_100_000},
		{id: 1, sid: "s1", round: 1, status: "ok", sendUs: 1_200_000, firstUs: 1_205_000, lastUs: 1_300_000},
	})
	got := computeSessionMetricsFromTrace(records)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.TTFTCold.Count != 0 {
		t.Errorf("TTFTCold.Count: got %d, want 0 (error round-0 must not contribute TTFT)", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 1 {
		t.Errorf("TTFTWarm.Count: got %d, want 1", got.TTFTWarm.Count)
	}
}
