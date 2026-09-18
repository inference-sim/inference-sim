package cluster

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestArrivalHook_FiresOncePerInitialRequest verifies the hook is invoked
// exactly once per request supplied through the RequestSource, in arrival
// order (issue #1440 — trace-export hook contract).
func TestArrivalHook_FiresOncePerInitialRequest(t *testing.T) {
	requests := newTestRequests(20)
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(requests), nil)

	seen := make([]*sim.Request, 0, len(requests))
	cs.SetArrivalHook(func(req *sim.Request) {
		seen = append(seen, req)
	})
	mustRun(t, cs)

	if got, want := len(seen), len(requests); got != want {
		t.Fatalf("arrival hook fired %d times, want %d (one per request)", got, want)
	}

	// Hook must observe the SAME pointers the cluster received — the trace
	// exporter relies on this identity to read final per-request state at
	// export time.
	for i, req := range seen {
		if req != requests[i] {
			t.Errorf("arrival hook[%d]: got request %q (ptr %p), want %q (ptr %p)",
				i, req.ID, req, requests[i].ID, requests[i])
		}
	}

	// Arrival-order invariant (INV-3): timestamps must be non-decreasing.
	var prev int64
	for i, req := range seen {
		if req.ArrivalTime < prev {
			t.Fatalf("arrival hook[%d] saw out-of-order ArrivalTime %d (prev=%d)", i, req.ArrivalTime, prev)
		}
		prev = req.ArrivalTime
	}
}

// TestArrivalHook_NoDuplicateFiresForRedirectedRequests verifies that
// REDIRECT re-injections (drain policy) do NOT fire the hook a second
// time — a redirected request is the same logical request already
// recorded on its original arrival.
func TestArrivalHook_NoDuplicateFiresForRedirectedRequests(t *testing.T) {
	requests := newTestRequests(5)
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(requests), nil)

	calls := 0
	cs.SetArrivalHook(func(req *sim.Request) {
		calls++
	})

	// Simulate a REDIRECT re-injection by pushing the same logical
	// request a second time with Redirected=true. The hook must NOT fire.
	cs.fireArrivalHook(requests[0], requests[0].ArrivalTime)
	if calls != 1 {
		t.Fatalf("after one fresh fireArrivalHook, want calls=1, got %d", calls)
	}
	redirected := *requests[0]
	redirected.Redirected = true
	cs.fireArrivalHook(&redirected, requests[0].ArrivalTime)
	if calls != 1 {
		t.Fatalf("hook must not fire for redirected request, got calls=%d", calls)
	}
}

// TestArrivalHook_DisabledWhenNotSet verifies zero overhead path (BC-1):
// when no hook is installed, fireArrivalHook is a no-op and the cluster
// completes normally.
func TestArrivalHook_DisabledWhenNotSet(t *testing.T) {
	requests := newTestRequests(10)
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(requests), nil)
	mustRun(t, cs) // no hook installed; should not panic, should not record anything
}

// TestArrivalHook_PanicsOnNonMonotonicArrival verifies the INV-3/INV-6
// guard: if the cluster ever fires the hook with a timestamp behind the
// previous one, we fail loudly rather than silently sort. The hook is
// the sole trace-emission point, so order regressions would corrupt the
// downstream parity contract.
func TestArrivalHook_PanicsOnNonMonotonicArrival(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on non-monotonic arrival, got none")
		}
	}()
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)
	cs.SetArrivalHook(func(req *sim.Request) {})

	req1 := &sim.Request{ID: "r1", ArrivalTime: 100}
	req2 := &sim.Request{ID: "r2", ArrivalTime: 50} // earlier than req1
	cs.fireArrivalHook(req1, req1.ArrivalTime)
	cs.fireArrivalHook(req2, req2.ArrivalTime) // expected: panic
}

// TestSetArrivalHook_AfterRun_Panics verifies the hook cannot be installed
// after Run() has executed (avoids silent contract violations where a
// trace exporter is wired up post-hoc and sees zero records).
func TestSetArrivalHook_AfterRun_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when SetArrivalHook is called after Run")
		}
	}()
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)
	mustRun(t, cs)
	cs.SetArrivalHook(func(req *sim.Request) {})
}

// TestArrivalHook_FiresForSessionFollowUps verifies that requests injected
// via the onRequestDone callback (closed-loop session follow-ups) reach
// the hook through the same path as initial arrivals. This is the
// contract the trace exporter depends on: every request — initial or
// follow-up — emits a single arrival-hook event.
func TestArrivalHook_FiresForSessionFollowUps(t *testing.T) {
	initial := newTestRequests(3)

	// onRequestDone emits one follow-up for each terminal completion, up
	// to a small budget — small enough not to slow the test, large enough
	// to exercise the follow-up code path. ArrivalTime is monotonically
	// later than the completion clock so the INV-3 guard inside the hook
	// stays happy.
	remaining := 3
	onRequestDone := func(req *sim.Request, clock int64) []*sim.Request {
		if remaining == 0 {
			return nil
		}
		remaining--
		return []*sim.Request{{
			ID:           req.ID + "-followup",
			ArrivalTime:  clock,
			InputTokens:  []sim.TokenID{1, 2, 3},
			OutputTokens: []sim.TokenID{4, 5},
			Model:        req.Model,
		}}
	}

	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(initial), onRequestDone)

	calls := 0
	cs.SetArrivalHook(func(req *sim.Request) {
		calls++
	})
	mustRun(t, cs)

	// The follow-up budget is fully consumed (each of the 3 initial
	// requests yields one follow-up before remaining hits 0), so the
	// exact expected count is initial + budget.
	const followUpBudget = 3
	if want := len(initial) + followUpBudget; calls != want {
		t.Fatalf("arrival hook fired %d times, want %d (initial=%d, follow-up budget=%d consumed)",
			calls, want, len(initial), followUpBudget)
	}
}

// TestArrivalHook_BeyondHorizonFollowUpsExcluded asserts the documented
// behavior shift introduced by issue #1440: a closed-loop follow-up whose
// scheduled ArrivalTime exceeds config.Horizon never executes in the DES
// event loop, so the hook does not see it. This is intentional — the
// old eager `preGeneratedRequests + followUpRequests` assembly emitted
// trace records for these never-arrived requests; the hook does not.
// Excluding them strengthens INV-13 (replay reads the same trace the
// run produced).
func TestArrivalHook_BeyondHorizonFollowUpsExcluded(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 100_000 // 100ms — small enough that follow-ups can land past it
	initial := newTestRequests(2)

	// Always emit a follow-up at clock+1 hour. With Horizon=100ms the
	// resulting ClusterArrivalEvent never executes and the hook never
	// sees this request.
	const beyondHorizonOffsetUs = int64(60 * 60 * 1_000_000) // 1 hour
	beyondHorizonID := "fu-beyond-horizon"
	emitted := false
	onRequestDone := func(req *sim.Request, clock int64) []*sim.Request {
		if emitted {
			return nil
		}
		emitted = true
		return []*sim.Request{{
			ID:           beyondHorizonID,
			ArrivalTime:  clock + beyondHorizonOffsetUs,
			InputTokens:  []sim.TokenID{1, 2, 3},
			OutputTokens: []sim.TokenID{4, 5},
			Model:        req.Model,
		}}
	}

	cs := NewClusterSimulator(cfg, NewSliceRequestSource(initial), onRequestDone)

	seenIDs := make(map[string]int)
	cs.SetArrivalHook(func(req *sim.Request) {
		seenIDs[req.ID]++
	})
	mustRun(t, cs)

	if got := seenIDs[beyondHorizonID]; got != 0 {
		t.Fatalf("beyond-horizon follow-up %q must not reach the arrival hook; got %d fire(s)", beyondHorizonID, got)
	}
	// Sanity: at least the initial requests still fire.
	for _, req := range initial {
		if seenIDs[req.ID] != 1 {
			t.Fatalf("initial request %q: hook fired %d times, want 1", req.ID, seenIDs[req.ID])
		}
	}
}

// TestINV13_ArrivalHookRunReplayParity asserts the central INV-13 claim of
// issue #1440: trace records captured via the arrival hook on a run, when
// exported and replayed through a fresh ClusterSimulator with the same
// config, produce identical per-request TTFT and E2E metrics. This closes
// the gap noted in the round-3 review — previously only record count +
// ArrivalTimeUs monotonicity were checked, not the full run→replay loop
// for the hook-driven path.
func TestINV13_ArrivalHookRunReplayParity(t *testing.T) {
	const fixedSeed int64 = 1234
	initial := newTestRequests(20)

	cfg := newTestDeploymentConfig(1)
	cfg.Seed = fixedSeed

	// WHEN: direct run captures records via the arrival hook (the new path).
	csRun := NewClusterSimulator(cfg, NewSliceRequestSource(initial), nil)
	var traceArrivals []*sim.Request
	csRun.SetArrivalHook(func(req *sim.Request) {
		traceArrivals = append(traceArrivals, req)
	})
	mustRun(t, csRun)
	runTTFTs := csRun.AggregatedMetrics().RequestTTFTs
	runE2Es := csRun.AggregatedMetrics().RequestE2Es

	if len(runTTFTs) == 0 {
		t.Fatal("INV-13: direct run produced no completed requests — cannot verify parity")
	}
	if len(traceArrivals) != len(initial) {
		t.Fatalf("hook captured %d requests, want %d (one per initial)", len(traceArrivals), len(initial))
	}

	// AND: export the hook-captured trace and reload it as replay would.
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	records := workload.RequestsToTraceRecords(traceArrivals)
	hdr := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	if err := workload.ExportTraceV2(hdr, records, headerPath, dataPath); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	loaded, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	replayReqs, err := workload.LoadTraceV2Requests(loaded, fixedSeed)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}

	// WHEN: replay the trace through a fresh ClusterSimulator with the same config.
	csReplay := NewClusterSimulator(cfg, NewSliceRequestSource(replayReqs), nil)
	mustRun(t, csReplay)
	replayTTFTs := csReplay.AggregatedMetrics().RequestTTFTs
	replayE2Es := csReplay.AggregatedMetrics().RequestE2Es

	// THEN: per-request TTFT and E2E are identical (INV-13).
	if len(runTTFTs) != len(replayTTFTs) {
		t.Fatalf("INV-13: completed-request count diverged: run=%d replay=%d", len(runTTFTs), len(replayTTFTs))
	}
	for id, ttft := range runTTFTs {
		got, ok := replayTTFTs[id]
		if !ok {
			t.Errorf("INV-13: request %q present in run, missing from replay TTFTs", id)
			continue
		}
		if got != ttft {
			t.Errorf("INV-13: request %q TTFT mismatch: run=%v replay=%v", id, ttft, got)
		}
	}
	for id, e2e := range runE2Es {
		got, ok := replayE2Es[id]
		if !ok {
			t.Errorf("INV-13: request %q present in run, missing from replay E2Es", id)
			continue
		}
		if got != e2e {
			t.Errorf("INV-13: request %q E2E mismatch: run=%v replay=%v", id, e2e, got)
		}
	}
}

// TestSetArrivalHook_NilClearResetsMonotonicityFloor verifies the reset
// behavior of SetArrivalHook(nil): a subsequently installed hook must
// not inherit the previous hook's last-seen ArrivalTime, otherwise
// callers re-using a ClusterSimulator instance to set up scenarios
// would see spurious panics from the monotonicity guard.
func TestSetArrivalHook_NilClearResetsMonotonicityFloor(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)

	// First hook sees a high timestamp, then is cleared.
	cs.SetArrivalHook(func(req *sim.Request) {})
	cs.fireArrivalHook(&sim.Request{ID: "r1", ArrivalTime: 1_000_000}, 1_000_000)
	cs.SetArrivalHook(nil)

	// Install a new hook and fire an EARLIER timestamp. Without the
	// reset, this would panic — the prior watermark of 1_000_000 would
	// still be in place.
	calls := 0
	cs.SetArrivalHook(func(req *sim.Request) { calls++ })
	cs.fireArrivalHook(&sim.Request{ID: "r2", ArrivalTime: 10}, 10)

	if calls != 1 {
		t.Fatalf("after nil-clear + reinstall, hook fired %d times, want 1 (monotonicity floor was not reset)", calls)
	}
}
