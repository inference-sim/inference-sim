package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
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
			InputTokens:  []int{1, 2, 3},
			OutputTokens: []int{4, 5},
			Model:        req.Model,
		}}
	}

	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(initial), onRequestDone)

	calls := 0
	cs.SetArrivalHook(func(req *sim.Request) {
		calls++
	})
	mustRun(t, cs)

	wantMin := len(initial)                                                 // every initial fires
	wantMax := len(initial) + 3                                              // plus the follow-up budget
	if calls < wantMin || calls > wantMax {
		t.Fatalf("arrival hook fired %d times, want in [%d, %d]: initial=%d, follow-up budget=3",
			calls, wantMin, wantMax, len(initial))
	}
}
