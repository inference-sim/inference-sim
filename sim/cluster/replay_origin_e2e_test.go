package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// epochSendTrace builds a TraceV2 that mimics a real `blis observe` trace: every
// record carries an epoch-scale send_time_us (time.Now().UnixMicro()) alongside a
// run-relative arrival_time_us. deadlineOffsetUs > 0 sets deadline_us = arrival +
// offset (also on the relative clock); deadlineOffsetUs == 0 means "no timeout".
func epochSendTrace(n int, deadlineOffsetUs int64) *workload.TraceV2 {
	const epoch = int64(1_787_274_995_712_218) // ~2026 in Unix µs
	recs := make([]workload.TraceRecord, n)
	for i := 0; i < n; i++ {
		arrival := int64(i) * 40_000 // relative arrivals, 40ms apart
		var deadline int64
		if deadlineOffsetUs > 0 {
			deadline = arrival + deadlineOffsetUs
		}
		recs[i] = workload.TraceRecord{
			RequestID:     i,
			ArrivalTimeUs: arrival,
			SendTimeUs:    epoch + arrival, // epoch clock, same spacing as arrival
			DeadlineUs:    deadline,
			InputTokens:   40,
			OutputTokens:  8,
			Status:        "ok",
		}
	}
	return &workload.TraceV2{Records: recs}
}

// TestReplay_EpochSendDeadline_AllComplete verifies BC-2 (fixes #1606): a real
// observe-style trace whose send_time_us is epoch-scale and whose deadline_us is
// run-relative and positive replays to completed == injected with zero timeouts.
// Before the fix, injection happened at the epoch tick (~1.79e15) while the
// relative deadline (~3.4e8) was already past, so EnqueueRequest's past-due guard
// (sim/simulator.go) instantly timed out every request (0 completions, empty
// sim_result.json). INV-1 (conservation) + INV-5 (causality).
func TestReplay_EpochSendDeadline_AllComplete(t *testing.T) {
	const n = 5
	trace := epochSendTrace(n, 300_000_000) // 300s relative deadline (never exceeded)

	reqs, err := workload.LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}
	if len(reqs) != n {
		t.Fatalf("built %d requests, want %d", len(reqs), n)
	}

	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	m := cs.AggregatedMetrics()

	if m.TimedOutRequests != 0 {
		t.Errorf("BC-2 (#1606): %d requests timed out, want 0 — injection origin must match the deadline clock", m.TimedOutRequests)
	}
	if m.CompletedRequests != n {
		t.Errorf("BC-2 (#1606): completed %d of %d requests, want %d (all servable)", m.CompletedRequests, n, n)
	}
	if len(m.RequestE2Es) != n {
		t.Errorf("BC-2 (#1606): %d requests have E2E recorded, want %d (non-empty sim_result)", len(m.RequestE2Es), n)
	}
}

// TestReplay_EpochSendDeadlineZero_RelativeOrigin verifies BC-7 (issue follow-up):
// even when deadline_us == 0 (no timeout, so nothing is killed), injection must be
// re-based onto the relative/arrival origin rather than left on the epoch clock —
// otherwise TTFT/E2E are silently computed on a ~1.79e15-tick clock. We assert the
// observable, deterministic property: the injected ArrivalTimes are on the relative
// origin (orders of magnitude below the epoch send), and the requests complete.
func TestReplay_EpochSendDeadlineZero_RelativeOrigin(t *testing.T) {
	const n = 5
	trace := epochSendTrace(n, 0) // deadline_us == 0 (no timeout)

	reqs, err := workload.LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}

	// The earliest injection is on the arrival origin (0); the latest is bounded
	// by the send window (n*40ms), FAR below any epoch-scale tick. A value in the
	// 1e12+ range would mean injection was left on the epoch clock (the bug).
	const epochScaleFloor = int64(1_000_000_000_000) // 1e12: any real relative tick is well under this
	var maxArrival int64
	for _, r := range reqs {
		if r.ArrivalTime < 0 {
			t.Errorf("BC-5/INV-3: negative injection tick %d for %s", r.ArrivalTime, r.ID)
		}
		if r.ArrivalTime > maxArrival {
			maxArrival = r.ArrivalTime
		}
	}
	if maxArrival >= epochScaleFloor {
		t.Errorf("BC-7 (#1606): max injected ArrivalTime = %d is epoch-scale; injection was not re-based to the relative origin", maxArrival)
	}

	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != n {
		t.Errorf("BC-7 (#1606): completed %d of %d requests, want %d", m.CompletedRequests, n, n)
	}
}
