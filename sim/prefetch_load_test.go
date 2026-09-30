package sim

import "testing"

// These are the Spec 3 Task 3 contract tests for StartPrefetch, the t>0 sibling of
// the demand-driven cold-load gate (maybeStartAdapterLoad). They observe behavior
// through resident state and the exported adapter metrics, mirroring the fixture
// idiom in sim/creation_seam_test.go and sim/cold_load_gate_test.go.
//
// gateTestConfig / mustNewSimulator / newTestRequest are shared helpers from
// cold_load_gate_test.go and progress_hook_test.go. LoRA is wired into package
// sim's tests via the blank import in lora_import_test.go, which registers the
// resident-set, registry, cost, eviction- and creation-policy construction funcs.

// newPrefetchTestSim builds a LoRA-enabled Simulator with the given per-instance
// adapter capacity and a pre-declared registry of rank-8 adapters at the given
// ids, exactly the gateTestConfig/mustNewSimulator construction used throughout
// the package's other seam tests (no new fixture idiom).
func newPrefetchTestSim(t *testing.T, capacity int, adapterIDs []string) *Simulator {
	t.Helper()
	specs := make([]AdapterSpec, len(adapterIDs))
	for i, id := range adapterIDs {
		specs[i] = AdapterSpec{ID: id, Rank: 8}
	}
	cfg := gateTestConfig(capacity, specs...)
	return mustNewSimulator(t, cfg)
}

// enqueuePrefillForAdapter builds a new prefill *Request for adapter id and pushes
// it onto sim.WaitQ, mirroring how TestOnResidentMissFalse_HoldsRequestInert
// (sim/creation_seam_test.go:162) drives the gate — except here we enqueue
// directly rather than via InjectArrival+Run, since the test wants to call
// maybeStartAdapterLoad itself rather than run the full simulation.
func enqueuePrefillForAdapter(t *testing.T, sim *Simulator, id string) *Request {
	t.Helper()
	req := newTestRequest("req-"+id, 0, 8, 4)
	req.Adapter = id
	sim.WaitQ.Enqueue(req)
	return req
}

// TestStartPrefetch_ChargesBothCounters is obligation 7: a prefetch is charged exactly
// like a demand load (INV-L3, once at completion) and additionally recorded as a
// prefetch, so demand = load - prefetch is recoverable.
func TestStartPrefetch_ChargesBothCounters(t *testing.T) {
	sim := newPrefetchTestSim(t, 2 /* capacity */, []string{"a1", "a2"})

	if !sim.StartPrefetch(0, "a1") {
		t.Fatal("StartPrefetch returned false on a free channel with a free slot")
	}
	// Nothing is charged at START — residency and charge both land at completion.
	if got := sim.Metrics.AdapterLoadCounts["a1"]; got != 0 {
		t.Errorf("load charged at start: AdapterLoadCounts[a1] = %d, want 0", got)
	}
	sim.completeAdapterLoad(9000, "a1")

	if got := sim.Metrics.AdapterLoadCounts["a1"]; got != 1 {
		t.Errorf("AdapterLoadCounts[a1] = %d, want 1 (INV-L3: charged once)", got)
	}
	if got := sim.Metrics.AdapterPrefetchCounts["a1"]; got != 1 {
		t.Errorf("AdapterPrefetchCounts[a1] = %d, want 1", got)
	}
	if !sim.residentAdapters.IsResident("a1") {
		t.Error("a1 must be resident after load completion")
	}
}

// TestDemandLoad_NotCountedAsPrefetch is the other half of obligation 7: the classifier
// must not leak. A load the gate started is a demand load even though it completes
// through the same path.
func TestDemandLoad_NotCountedAsPrefetch(t *testing.T) {
	sim := newPrefetchTestSim(t, 2, []string{"a1", "a2"})
	sim.loadingAdapter = "a1" // as maybeStartAdapterLoad would set it
	sim.loadIsPrefetch = false

	sim.completeAdapterLoad(9000, "a1")

	if got := sim.Metrics.AdapterLoadCounts["a1"]; got != 1 {
		t.Errorf("AdapterLoadCounts[a1] = %d, want 1", got)
	}
	if got, ok := sim.Metrics.AdapterPrefetchCounts["a1"]; ok && got != 0 {
		t.Errorf("demand load counted as prefetch: AdapterPrefetchCounts[a1] = %d, want absent/0", got)
	}
}

// TestStartPrefetch_RefusesWhenChannelBusy is half of obligation 6: loads are serialized
// per instance, so a prefetch must never start on top of an in-flight load.
func TestStartPrefetch_RefusesWhenChannelBusy(t *testing.T) {
	sim := newPrefetchTestSim(t, 2, []string{"a1", "a2"})
	sim.loadingAdapter = "a2"

	if sim.StartPrefetch(0, "a1") {
		t.Error("StartPrefetch must refuse while another load is in flight")
	}
	if sim.loadingAdapter != "a2" {
		t.Errorf("refused prefetch corrupted the channel: loadingAdapter = %q, want a2", sim.loadingAdapter)
	}
}

// TestStartPrefetch_RefusesWhenAlreadyResident guards against a wasted charged load for
// an adapter that is already there.
func TestStartPrefetch_RefusesWhenAlreadyResident(t *testing.T) {
	sim := newPrefetchTestSim(t, 2, []string{"a1", "a2"})
	sim.residentAdapters.Store("a1")

	if sim.StartPrefetch(0, "a1") {
		t.Error("StartPrefetch must refuse an already-resident adapter")
	}
}

// TestStartPrefetch_AtCapacityDelegatesToEvictionSeam is obligation 9. The tick policy
// must never name a victim; victim choice stays the eviction seam's decision, which is
// what keeps the three decisions factored.
func TestStartPrefetch_AtCapacityDelegatesToEvictionSeam(t *testing.T) {
	sim := newPrefetchTestSim(t, 1 /* capacity: full after one Store */, []string{"a1", "a2"})
	sim.residentAdapters.Store("a2")

	calls := 0
	sim.evictionPolicy = recordingEviction{inner: sim.evictionPolicy, calls: &calls}

	if !sim.StartPrefetch(0, "a1") {
		t.Fatal("StartPrefetch returned false when a victim was available")
	}
	if calls != 1 {
		t.Errorf("eviction seam called %d times, want exactly 1", calls)
	}
	if got := sim.Metrics.AdapterEvictionCounts["a2"]; got != 1 {
		t.Errorf("AdapterEvictionCounts[a2] = %d, want 1 (the reserved slot's victim)", got)
	}
}

type recordingEviction struct {
	inner EvictionPolicy
	calls *int
}

func (r recordingEviction) SelectVictim(ctx EvictionContext) (string, bool) {
	*r.calls++
	return r.inner.SelectVictim(ctx)
}

// TestRequestArrivingMidPrefetch_ChargedOnce is obligation 8. A request for an adapter a
// prefetch is already loading must wait at the gate and NOT start a second load. The
// existing loadingAdapter guard in maybeStartAdapterLoad gives this; the test pins it so
// a future refactor of that guard cannot silently double-charge.
func TestRequestArrivingMidPrefetch_ChargedOnce(t *testing.T) {
	sim := newPrefetchTestSim(t, 2, []string{"a1", "a2"})
	if !sim.StartPrefetch(0, "a1") {
		t.Fatal("prefetch did not start")
	}
	// A request for the very adapter being prefetched arrives and reaches the gate.
	enqueuePrefillForAdapter(t, sim, "a1")
	sim.maybeStartAdapterLoad(100)

	if sim.loadingAdapter != "a1" || !sim.loadIsPrefetch {
		t.Errorf("gate disturbed the in-flight prefetch: loadingAdapter=%q isPrefetch=%v", sim.loadingAdapter, sim.loadIsPrefetch)
	}
	sim.completeAdapterLoad(9000, "a1")

	if got := sim.Metrics.AdapterLoadCounts["a1"]; got != 1 {
		t.Errorf("AdapterLoadCounts[a1] = %d, want 1 — the arriving request must not add a load", got)
	}
	if got := sim.Metrics.AdapterPrefetchCounts["a1"]; got != 1 {
		t.Errorf("AdapterPrefetchCounts[a1] = %d, want 1 — classification is by initiation, not consumption", got)
	}
}
