package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildSessionTestMetrics is a test helper that builds a minimal *sim.Metrics
// with the given RequestMetrics entries and completion times.
// TTFT values on RequestMetrics are treated as ms and mirrored into m.RequestTTFTs
// as µs (× 1000), matching how ComputeSessionMetrics reads them in production.
// ArrivedAt values are passed as-is (seconds in production; tests use 0 or small
// integers which represent seconds — duration arithmetic is in ms, so ArrivedAt
// is multiplied by 1000.0 inside ComputeSessionMetrics).
func buildSessionTestMetrics(reqs []sim.RequestMetrics, completionTimes map[string]float64) *sim.Metrics {
	m := sim.NewMetrics()
	for _, r := range reqs {
		m.Requests[r.ID] = r
		if r.TTFT > 0 {
			m.RequestTTFTs[r.ID] = r.TTFT * 1000.0 // ms → µs
		}
	}
	for id, t := range completionTimes {
		m.RequestCompletionTimes[id] = t
	}
	return m
}

// BC-1: returns nil when no requests have SessionID
func TestComputeSessionMetrics_NoSessions_ReturnsNil(t *testing.T) {
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "r1", TTFT: 10.0},
		{ID: "r2", TTFT: 20.0},
	}, map[string]float64{"r1": 100e3, "r2": 200e3})

	got := ComputeSessionMetrics(m)
	if got != nil {
		t.Errorf("expected nil for non-session workload, got %+v", got)
	}
}

// BC-1: returns nil for empty metrics
func TestComputeSessionMetrics_EmptyMetrics_ReturnsNil(t *testing.T) {
	m := sim.NewMetrics()
	got := ComputeSessionMetrics(m)
	if got != nil {
		t.Errorf("expected nil for empty metrics, got %+v", got)
	}
}

// BC-2: TTFT partitioned correctly by RoundIndex
func TestComputeSessionMetrics_TTFTPartition(t *testing.T) {
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		{ID: "r2", SessionID: "s1", RoundIndex: 2, TTFT: 18.0, ArrivedAt: 200},
	}, map[string]float64{
		"r0": 500e3,
		"r1": 600e3,
		"r2": 700e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 2 {
		t.Errorf("TTFTWarm.Count: got %d, want 2", got.TTFTWarm.Count)
	}
	if got.TTFTCold.Mean != 50.0 {
		t.Errorf("TTFTCold.Mean: got %.2f, want 50.0", got.TTFTCold.Mean)
	}
}

// BC-3: SessionDuration = max_completion_ms - round0_arrival_ms
func TestComputeSessionMetrics_SessionDuration(t *testing.T) {
	// Round-0 arrives at ArrivedAt=0s (0s × 1000 = 0ms); final round completes at 700,000 µs = 700ms.
	// Session duration = 700ms - 0ms = 700ms.
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
	}, map[string]float64{
		"r0": 300e3, // 300ms in µs
		"r1": 700e3, // 700ms in µs
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	if got.SessionDuration.Count != 1 {
		t.Errorf("SessionDuration.Count: got %d, want 1", got.SessionDuration.Count)
	}
	const wantDuration = 700.0 // ms
	if got.SessionDuration.Mean != wantDuration {
		t.Errorf("SessionDuration.Mean: got %.2f, want %.2f", got.SessionDuration.Mean, wantDuration)
	}
}

// BC-4: non-session requests do not affect metrics
func TestComputeSessionMetrics_MixedWorkload(t *testing.T) {
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		// Non-session requests (SessionID == "")
		{ID: "ns1", TTFT: 999.0},
		{ID: "ns2", TTFT: 888.0},
	}, map[string]float64{
		"r0": 300e3, "r1": 700e3,
		"ns1": 400e3, "ns2": 500e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1 (non-session must be excluded)", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 1 {
		t.Errorf("TTFTWarm.Count: got %d, want 1 (non-session must be excluded)", got.TTFTWarm.Count)
	}
	if got.TTFTCold.Mean != 50.0 {
		t.Errorf("TTFTCold.Mean: got %.2f, want 50.0 (non-session contamination?)", got.TTFTCold.Mean)
	}
}

// BC-6: session without round-0 in completed requests is excluded from SessionDuration, no panic
func TestComputeSessionMetrics_MissingRound0(t *testing.T) {
	// Only round-1 and round-2 appear (round-0 timed out / not in Requests map)
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		{ID: "r2", SessionID: "s1", RoundIndex: 2, TTFT: 18.0, ArrivedAt: 200},
	}, map[string]float64{
		"r1": 600e3, "r2": 700e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics (warm requests exist)")
	}
	if got.SessionDuration.Count != 0 {
		t.Errorf("SessionDuration.Count: got %d, want 0 (no round-0 entry)", got.SessionDuration.Count)
	}
	if got.TTFTCold.Count != 0 {
		t.Errorf("TTFTCold.Count: got %d, want 0 (no round-0 request present)", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 2 {
		t.Errorf("TTFTWarm.Count: got %d, want 2", got.TTFTWarm.Count)
	}
}

// BC-3 multi-session: SessionDuration distribution is computed per-session and averaged
func TestComputeSessionMetrics_MultiSessionDuration(t *testing.T) {
	// s1: round0 arrives at 0s, completes at 300ms → duration 300ms
	// s2: round0 arrives at 0s, completes at 700ms → duration 700ms
	// mean = 500ms
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "s1r0", SessionID: "s1", RoundIndex: 0, TTFT: 10.0, ArrivedAt: 0},
		{ID: "s1r1", SessionID: "s1", RoundIndex: 1, TTFT: 8.0, ArrivedAt: 50},
		{ID: "s2r0", SessionID: "s2", RoundIndex: 0, TTFT: 12.0, ArrivedAt: 0},
		{ID: "s2r1", SessionID: "s2", RoundIndex: 1, TTFT: 9.0, ArrivedAt: 100},
	}, map[string]float64{
		"s1r0": 100e3, "s1r1": 300e3, // s1 max completion = 300ms
		"s2r0": 200e3, "s2r1": 700e3, // s2 max completion = 700ms
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.SessionDuration.Count != 2 {
		t.Errorf("SessionDuration.Count: got %d, want 2", got.SessionDuration.Count)
	}
	const wantMean = 500.0
	if got.SessionDuration.Mean != wantMean {
		t.Errorf("SessionDuration.Mean: got %.2f, want %.2f", got.SessionDuration.Mean, wantMean)
	}
}

// Invariant: SessionCount equals number of distinct SessionIDs
// TestComputeSessionMetrics_TimedOutBeforeFirstToken_NotCountedInTTFT verifies the R1 guard:
// a session request that is admitted but times out before producing its first token is absent
// from RequestTTFTs. It must not contribute a spurious 0.0ms entry to cold/warm distributions.
func TestComputeSessionMetrics_TimedOutBeforeFirstToken_NotCountedInTTFT(t *testing.T) {
	m := sim.NewMetrics()
	// round-0 request admitted (in Requests) but no TTFT entry (timed out before first token)
	m.Requests["r0"] = sim.RequestMetrics{ID: "r0", SessionID: "s1", RoundIndex: 0, ArrivedAt: 0}
	// round-1 request with a valid TTFT
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", SessionID: "s1", RoundIndex: 1, ArrivedAt: 100_000}
	m.RequestTTFTs["r1"] = 5_000.0 // 5ms

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics (session request exists)")
	}
	if got.TTFTCold.Count != 0 {
		t.Errorf("TTFTCold.Count: got %d, want 0 (timed-out round-0 must not inject 0ms entry)", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 1 {
		t.Errorf("TTFTWarm.Count: got %d, want 1 (round-1 with valid TTFT must be counted)", got.TTFTWarm.Count)
	}
}

func TestComputeSessionMetrics_SessionCount(t *testing.T) {
	m := buildSessionTestMetrics([]sim.RequestMetrics{
		{ID: "a0", SessionID: "s1", RoundIndex: 0, TTFT: 10.0, ArrivedAt: 0},
		{ID: "a1", SessionID: "s1", RoundIndex: 1, TTFT: 8.0, ArrivedAt: 50},
		{ID: "b0", SessionID: "s2", RoundIndex: 0, TTFT: 12.0, ArrivedAt: 0},
	}, map[string]float64{
		"a0": 100e3, "a1": 200e3, "b0": 150e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.SessionCount != 2 {
		t.Errorf("SessionCount: got %d, want 2", got.SessionCount)
	}
}

