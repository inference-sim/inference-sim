package cluster

import (
	"sync"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// fakeObserverDecider wraps any DisaggregationDecider and records every
// OnOutcome call for assertion. Thread-safe (the simulator is single-threaded
// but future parallel runs should remain safe).
type fakeObserverDecider struct {
	sim.DisaggregationDecider
	mu    sync.Mutex
	Calls []fakeObserverCall
}

type fakeObserverCall struct {
	ReqID    string
	Decision sim.DisaggregationDecision
	Outcome  sim.Outcome
}

func (f *fakeObserverDecider) OnOutcome(req *sim.Request, d sim.DisaggregationDecision, o sim.Outcome) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Calls = append(f.Calls, fakeObserverCall{ReqID: req.ID, Decision: d, Outcome: o})
}

// Compile-time: fakeObserverDecider implements both interfaces.
var (
	_ sim.DisaggregationDecider = (*fakeObserverDecider)(nil)
	_ sim.DisaggregationObserver = (*fakeObserverDecider)(nil)
)

// TestObserver_FakeDecider_ImplementsObserver is a plumbing sanity check.
func TestObserver_FakeDecider_ImplementsObserver(t *testing.T) {
	f := &fakeObserverDecider{DisaggregationDecider: &sim.AlwaysDisaggregate{}}
	if _, ok := interface{}(f).(sim.DisaggregationObserver); !ok {
		t.Fatal("fakeObserverDecider must implement sim.DisaggregationObserver")
	}
	if d := f.Decide(&sim.Request{}, nil); !d.Disaggregate {
		t.Fatal("delegated Decide should return AlwaysDisaggregate semantics")
	}
}

// newPDClusterWithObserver builds a PD-enabled cluster and injects the supplied
// observer-aware decider in place of the factory-built one, priming the
// tracking map so the non-disagg path records observations.
//
// Reuses newTestDisaggDeploymentConfig(numInstances=4, prefill=2, decode=2) to
// match the conventions used by sibling disagg integration tests.
func newPDClusterWithObserver(t *testing.T, obs sim.DisaggregationDecider, numRequests int) (*ClusterSimulator, []*sim.Request) {
	t.Helper()
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	// Factory default PDDecider is "always"; we replace it below.
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)
	cs.disaggregationDecider = obs
	// Mirror the constructor's observer-aware path: if the swapped-in decider
	// implements DisaggregationObserver, allocate the tracking map so the
	// non-disagg branch records observations (also exercised via lazy init in
	// executeDisaggregatedRouting, but we initialize here for clarity).
	cs.maybeInitNonDisaggObservations()
	return cs, requests
}

// TestObserver_DisaggPath_FiresOncePerCompletedParent (BC-2) verifies that
// when a DisaggregationObserver-implementing decider returns Disaggregate=true,
// OnOutcome fires exactly once per successfully completed parent request with
// Disaggregated=true, non-empty decode and prefill instance IDs, a non-negative
// transfer duration, and TTFT matching the projected parent TTFT written to
// RequestTTFTs[pid].
func TestObserver_DisaggPath_FiresOncePerCompletedParent(t *testing.T) {
	obs := &fakeObserverDecider{DisaggregationDecider: &sim.AlwaysDisaggregate{}}
	cs, requests := newPDClusterWithObserver(t, obs, 5)
	mustRun(t, cs)

	// All 5 parents should complete (small workload, generous horizon) — each
	// fires OnOutcome exactly once.
	if got, want := len(obs.Calls), len(requests); got != want {
		t.Fatalf("OnOutcome call count = %d, want %d (one per completed parent); calls=%+v",
			got, want, obs.Calls)
	}

	m := cs.AggregatedMetrics()
	for i, c := range obs.Calls {
		if !c.Decision.Disaggregate {
			t.Errorf("call[%d] decision.Disaggregate = false, want true", i)
		}
		if !c.Outcome.Disaggregated {
			t.Errorf("call[%d] outcome.Disaggregated = false, want true", i)
		}
		if c.Outcome.DecodeInstanceID == "" {
			t.Errorf("call[%d] outcome.DecodeInstanceID empty", i)
		}
		if c.Outcome.PrefillInstanceID == "" {
			t.Errorf("call[%d] outcome.PrefillInstanceID empty", i)
		}
		if c.Outcome.TransferDurationUs < 0 {
			t.Errorf("call[%d] outcome.TransferDurationUs = %d, want >= 0",
				i, c.Outcome.TransferDurationUs)
		}
		if c.Outcome.CompletionTime <= 0 {
			t.Errorf("call[%d] outcome.CompletionTime = %d, want > 0",
				i, c.Outcome.CompletionTime)
		}
		// TTFT reported to observer must match projected parent TTFT.
		wantTTFT, ok := m.RequestTTFTs[c.ReqID]
		if !ok {
			t.Errorf("call[%d] req %q: no projected TTFT in metrics", i, c.ReqID)
			continue
		}
		if c.Outcome.TTFT != wantTTFT {
			t.Errorf("call[%d] req %q: outcome.TTFT = %f, projected RequestTTFTs = %f",
				i, c.ReqID, c.Outcome.TTFT, wantTTFT)
		}
	}

	// Determinism (INV-6): calls must be ordered by request ID.
	for i := 1; i < len(obs.Calls); i++ {
		if obs.Calls[i-1].ReqID >= obs.Calls[i].ReqID {
			t.Errorf("disagg observer call sequence not sorted by request ID: [%d]=%q [%d]=%q",
				i-1, obs.Calls[i-1].ReqID, i, obs.Calls[i].ReqID)
		}
	}
}

// TestObserver_NonDisaggPath_FiresOncePerCompletion (BC-3) verifies that when
// the decider returns Disaggregate=false for requests flowing through the PD
// routing entry point, OnOutcome fires exactly once on terminal success with
// Disaggregated=false, PrefillInstanceID="", TransferDurationUs=0, and a
// DecodeInstanceID matching the pod that handled the request.
func TestObserver_NonDisaggPath_FiresOncePerCompletion(t *testing.T) {
	obs := &fakeObserverDecider{DisaggregationDecider: &sim.NeverDisaggregate{}}
	cs, requests := newPDClusterWithObserver(t, obs, 5)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()

	// Every successfully completed request (non-disagg path) must fire once.
	expected := 0
	for _, req := range requests {
		if _, ok := m.RequestCompletionTimes[req.ID]; ok {
			expected++
		}
	}
	if got := len(obs.Calls); got != expected {
		t.Fatalf("OnOutcome call count = %d, want %d (matching completed requests); calls=%+v",
			got, expected, obs.Calls)
	}

	for i, c := range obs.Calls {
		if c.Decision.Disaggregate {
			t.Errorf("call[%d] decision.Disaggregate = true, want false (NeverDisaggregate)", i)
		}
		if c.Outcome.Disaggregated {
			t.Errorf("call[%d] outcome.Disaggregated = true, want false", i)
		}
		if c.Outcome.PrefillInstanceID != "" {
			t.Errorf("call[%d] outcome.PrefillInstanceID = %q, want empty (non-disagg)",
				i, c.Outcome.PrefillInstanceID)
		}
		if c.Outcome.TransferDurationUs != 0 {
			t.Errorf("call[%d] outcome.TransferDurationUs = %d, want 0 (non-disagg)",
				i, c.Outcome.TransferDurationUs)
		}
		if c.Outcome.DecodeInstanceID == "" {
			t.Errorf("call[%d] outcome.DecodeInstanceID empty", i)
		}
		// DecodeInstanceID must be a decode-pool pod.
		role, ok := cs.poolMembership[c.Outcome.DecodeInstanceID]
		if !ok {
			t.Errorf("call[%d] DecodeInstanceID %q not in pool membership",
				i, c.Outcome.DecodeInstanceID)
		} else if !role.Has(PoolRoleDecode) {
			t.Errorf("call[%d] DecodeInstanceID %q has role %v, expected decode",
				i, c.Outcome.DecodeInstanceID, role)
		}
		if c.Outcome.CompletionTime <= 0 {
			t.Errorf("call[%d] outcome.CompletionTime = %d, want > 0",
				i, c.Outcome.CompletionTime)
		}
	}
}

// TestObserver_NonObserverDecider_NoCallbacks (BC-5) verifies that a decider
// not implementing DisaggregationObserver triggers zero callbacks, and no
// tracking map is allocated.
func TestObserver_NonObserverDecider_NoCallbacks(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	// Built-in "always" decider does NOT implement DisaggregationObserver.
	config.PDDecider = "always"
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if cs.nonDisaggObservations != nil {
		t.Errorf("nonDisaggObservations should be nil for non-observer decider, got %v",
			cs.nonDisaggObservations)
	}
	// No way to "detect" observer calls without a hook — the strong guarantee
	// is the nil-map check above plus the type-assertion unit test in sim/.
}

// TestObserver_BuiltinsDoNotTriggerObserverPath (BC-5) verifies parity: runs
// with each built-in decider succeed without error and produce a positive
// CompletedRequests count (not zero — regression guard for a hook that might
// have inadvertently discarded requests).
func TestObserver_BuiltinsDoNotTriggerObserverPath(t *testing.T) {
	cases := []string{"never", "always"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			config := newTestDisaggDeploymentConfig(4, 2, 2)
			config.PDDecider = name
			cs := NewClusterSimulator(config, newTestRequests(5), nil)
			mustRun(t, cs)
			if cs.nonDisaggObservations != nil {
				t.Errorf("PDDecider=%q: nonDisaggObservations = %v, want nil",
					name, cs.nonDisaggObservations)
			}
			if cs.AggregatedMetrics().CompletedRequests == 0 {
				t.Errorf("PDDecider=%q: CompletedRequests = 0 (regression)", name)
			}
		})
	}
}

// TestObserver_Determinism (BC-6) verifies the observer sees the exact same
// sequence of (ReqID, Outcome) pairs across repeated runs with the same seed.
// Disagg-path variant: all requests flow through the end-of-run
// projectPDMetrics dispatch (AlwaysDisaggregate).
func TestObserver_Determinism(t *testing.T) {
	run := func() []fakeObserverCall {
		obs := &fakeObserverDecider{DisaggregationDecider: &sim.AlwaysDisaggregate{}}
		cs, _ := newPDClusterWithObserver(t, obs, 5)
		mustRun(t, cs)
		return obs.Calls
	}
	got1 := run()
	got2 := run()
	if len(got1) != len(got2) {
		t.Fatalf("call count differs across runs: run1=%d run2=%d", len(got1), len(got2))
	}
	for i := range got1 {
		if got1[i] != got2[i] {
			t.Errorf("call[%d] differs:\n  run1=%+v\n  run2=%+v", i, got1[i], got2[i])
		}
	}
}

// TestObserver_NonDisagg_Determinism (BC-6) is the symmetric determinism check
// for the non-disagg path: all requests are routed locally (NeverDisaggregate),
// so callbacks fire from detectNonDisaggObserverCompletions across per-instance
// event-loop ticks. Pins the deterministic-dispatch guarantee across runs.
func TestObserver_NonDisagg_Determinism(t *testing.T) {
	run := func() []fakeObserverCall {
		obs := &fakeObserverDecider{DisaggregationDecider: &sim.NeverDisaggregate{}}
		cs, _ := newPDClusterWithObserver(t, obs, 5)
		mustRun(t, cs)
		return obs.Calls
	}
	got1 := run()
	got2 := run()
	if len(got1) != len(got2) {
		t.Fatalf("call count differs across runs: run1=%d run2=%d", len(got1), len(got2))
	}
	for i := range got1 {
		if got1[i] != got2[i] {
			t.Errorf("call[%d] differs:\n  run1=%+v\n  run2=%+v", i, got1[i], got2[i])
		}
	}
}

// TestObserver_DisaggPath_PreservesOriginalDecision (addresses PR #1346 review,
// finding MODERATE): the DisaggregationDecision delivered to OnOutcome on the
// disagg path must equal the decision returned by Decide() at routing time —
// including any DecodePodOverride or PrefillPodHint fields set by a joint D+P
// decider. Prior to persisting parent.DisaggDecision, the observer dispatch
// synthesized {Disaggregate: true} and stripped these fields.
func TestObserver_DisaggPath_PreservesOriginalDecision(t *testing.T) {
	// A decider that always returns Disaggregate=true AND sets a
	// PrefillPodHint so we can verify the hint round-trips through
	// ParentRequest into OnOutcome.
	obs := &hintedObserverDecider{prefillHint: "instance_0"}
	cs, _ := newPDClusterWithObserver(t, obs, 3)
	mustRun(t, cs)

	if len(obs.Calls) == 0 {
		t.Fatal("no observer callbacks fired; expected at least one")
	}
	for i, c := range obs.Calls {
		if !c.Decision.Disaggregate {
			t.Errorf("call[%d] decision.Disaggregate = false, want true", i)
		}
		if c.Decision.PrefillPodHint != "instance_0" {
			t.Errorf("call[%d] decision.PrefillPodHint = %q, want %q (decision stripped across projection)",
				i, c.Decision.PrefillPodHint, "instance_0")
		}
	}
}

// hintedObserverDecider is a test-only decider that records every OnOutcome
// call and returns Disaggregate=true with a non-empty PrefillPodHint to
// exercise the decision-round-trip guarantee.
type hintedObserverDecider struct {
	prefillHint string
	mu          sync.Mutex
	Calls       []fakeObserverCall
}

func (h *hintedObserverDecider) Decide(_ *sim.Request, _ *sim.RouterState) sim.DisaggregationDecision {
	return sim.DisaggregationDecision{
		Disaggregate:   true,
		PrefillPodHint: h.prefillHint,
	}
}

func (h *hintedObserverDecider) OnOutcome(req *sim.Request, d sim.DisaggregationDecision, o sim.Outcome) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.Calls = append(h.Calls, fakeObserverCall{ReqID: req.ID, Decision: d, Outcome: o})
}

// TestObserver_DisaggAndNonDisaggMixed verifies that a decider that flips
// disaggregate based on request ID fires OnOutcome once per request with
// the correct path-specific outcome shape.
func TestObserver_DisaggAndNonDisaggMixed(t *testing.T) {
	obs := &mixedObserverDecider{}
	cs, requests := newPDClusterWithObserver(t, obs, 6)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()

	// Every completed request (either path) fires once.
	expected := 0
	for _, req := range requests {
		if _, ok := m.RequestCompletionTimes[req.ID]; ok {
			expected++
		}
	}
	if got := len(obs.Calls); got != expected {
		t.Fatalf("OnOutcome count = %d, want %d", got, expected)
	}

	// Each call's Disaggregated flag must agree with its decision.
	for _, c := range obs.Calls {
		if c.Decision.Disaggregate != c.Outcome.Disaggregated {
			t.Errorf("req %q: decision.Disaggregate=%v outcome.Disaggregated=%v (mismatch)",
				c.ReqID, c.Decision.Disaggregate, c.Outcome.Disaggregated)
		}
		if c.Outcome.Disaggregated {
			if c.Outcome.PrefillInstanceID == "" {
				t.Errorf("req %q: disagg outcome has empty PrefillInstanceID", c.ReqID)
			}
		} else {
			if c.Outcome.PrefillInstanceID != "" {
				t.Errorf("req %q: non-disagg outcome has PrefillInstanceID=%q",
					c.ReqID, c.Outcome.PrefillInstanceID)
			}
			if c.Outcome.TransferDurationUs != 0 {
				t.Errorf("req %q: non-disagg outcome has TransferDurationUs=%d",
					c.ReqID, c.Outcome.TransferDurationUs)
			}
		}
	}
}

// mixedObserverDecider returns Disaggregate=true for even-numbered request IDs
// and Disaggregate=false otherwise. Deterministic (keyed on request ID only).
type mixedObserverDecider struct {
	mu    sync.Mutex
	Calls []fakeObserverCall
}

func (m *mixedObserverDecider) Decide(req *sim.Request, _ *sim.RouterState) sim.DisaggregationDecision {
	// request IDs are "request_N" with N in [0, numRequests).
	// Parse the suffix and return Disaggregate=(N%2==0).
	n := parseTrailingIndex(req.ID)
	return sim.DisaggregationDecision{Disaggregate: n%2 == 0}
}

func (m *mixedObserverDecider) OnOutcome(req *sim.Request, d sim.DisaggregationDecision, o sim.Outcome) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Calls = append(m.Calls, fakeObserverCall{ReqID: req.ID, Decision: d, Outcome: o})
}

// parseTrailingIndex returns the integer suffix of a request ID like
// "request_7" → 7. Returns 0 for any ID that does not parse (treat as
// non-disagg for test stability).
func parseTrailingIndex(id string) int {
	// Find the last '_' and parse the suffix manually (avoiding strconv import
	// for a trivial helper).
	n := 0
	started := false
	for i := len(id) - 1; i >= 0; i-- {
		c := id[i]
		if c >= '0' && c <= '9' {
			started = true
			continue
		}
		if started && c == '_' {
			for j := i + 1; j < len(id); j++ {
				n = n*10 + int(id[j]-'0')
			}
			return n
		}
		break
	}
	return 0
}
