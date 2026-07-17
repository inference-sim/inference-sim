package sim

import (
	"testing"
)

// The cold-load pre-admission gate charges a one-time adapter load latency when a
// request's adapter is not yet resident on the instance (FR-007, US3 scenario 1,
// SC-004). A cold request is held out of the running batch until its adapter load
// completes; a warm request is admitted immediately with no load latency. Adapter
// loads on an instance are serialized. These are the T027 contract tests and the
// T029 companion invariant tests; they observe behavior through per-request TTFT
// and the exported adapter metrics, independent of the gate's internal mechanism.

// gateTestConfig returns a LoRA-enabled SimConfig with a single ranked adapter
// and simple cost coefficients so the load latency is an exact, predictable value.
// base=1000µs, bandwidth=2e6 B/µs, footprint=2e6 B/rank ⇒ LoadLatency(rank 8) =
// 1000 + ceil(1.6e7/2e6) = 1008µs.
func gateTestConfig(capacity int, adapters ...AdapterSpec) SimConfig {
	cfg := newTestSimConfig()
	c := capacity
	cfg.LoRAConfig = LoRAConfig{
		AdapterCapacity:       &c,
		LoadBaseLatencyUs:     fptrGate(1000.0),
		LoadBandwidthBytesUs:  fptrGate(2.0e6),
		FootprintBytesPerRank: fptrGate(2.0e6),
		Adapters:              adapters,
	}
	return cfg
}

func fptrGate(v float64) *float64 { return &v }

const gateLoadLatencyRank8 = 1008 // base 1000 + ceil(footprint/bw) 8

// runSingleTTFT drives a one-request simulation and returns the request's TTFT
// (FirstTokenTime, measured relative to arrival). The request pointer is mutated
// in place by the run, so its FirstTokenTime is readable afterward.
func runSingleTTFT(t *testing.T, cfg SimConfig, adapter string) int64 {
	t.Helper()
	req := newTestRequest("solo", 0, 8, 4)
	req.Adapter = adapter
	s := mustNewSimulator(t, cfg)
	s.InjectArrival(req)
	s.Run()
	if !req.TTFTSet {
		t.Fatalf("request never produced a first token (adapter=%q)", adapter)
	}
	return req.FirstTokenTime
}

// TestColdLoadGate_ColdAddsLoadLatencyToTTFT is the core T027 contract: a cold
// adapter request's TTFT exceeds an otherwise-identical base-model request's TTFT
// by exactly the modeled load latency (US3 scenario 1, SC-004).
func TestColdLoadGate_ColdAddsLoadLatencyToTTFT(t *testing.T) {
	adapter := AdapterSpec{ID: "a8", Rank: 8}

	// Base-model request on a LoRA-enabled instance (empty adapter never gates).
	base := runSingleTTFT(t, gateTestConfig(2, adapter), "")
	// Cold adapter request: identical shape, adapter not yet resident.
	cold := runSingleTTFT(t, gateTestConfig(2, adapter), "a8")

	if got := cold - base; got != gateLoadLatencyRank8 {
		t.Errorf("cold TTFT excess over base = %d, want %d (one load latency); base=%d cold=%d",
			got, gateLoadLatencyRank8, base, cold)
	}
}

// TestColdLoadGate_WarmIncursNoLoadLatency is the T027 warm case: once an adapter
// is resident, a subsequent request for it incurs zero load latency — its TTFT
// equals the base-model TTFT (US3 scenario 1). The second request arrives long
// after the first completes so it does not queue behind it.
func TestColdLoadGate_WarmIncursNoLoadLatency(t *testing.T) {
	adapter := AdapterSpec{ID: "a8", Rank: 8}
	cfg := gateTestConfig(2, adapter)

	baseTTFT := runSingleTTFT(t, gateTestConfig(2, adapter), "")

	r1 := newTestRequest("cold", 0, 8, 4)
	r1.Adapter = "a8"
	r2 := newTestRequest("warm", 5_000_000, 8, 4) // arrives well after r1 completes
	r2.Adapter = "a8"

	s := mustNewSimulator(t, cfg)
	s.InjectArrival(r1)
	s.InjectArrival(r2)
	s.Run()

	if !r1.TTFTSet || !r2.TTFTSet {
		t.Fatalf("requests did not both complete: r1.set=%v r2.set=%v", r1.TTFTSet, r2.TTFTSet)
	}
	// r1 cold: pays the load; r2 warm: pays nothing (TTFT == base).
	if got := r1.FirstTokenTime - baseTTFT; got != gateLoadLatencyRank8 {
		t.Errorf("cold r1 TTFT excess = %d, want %d", got, gateLoadLatencyRank8)
	}
	if r2.FirstTokenTime != baseTTFT {
		t.Errorf("warm r2 TTFT = %d, want %d (base, no load latency)", r2.FirstTokenTime, baseTTFT)
	}

	// The adapter is cold-loaded exactly once across both requests (INV-L3).
	out := s.Metrics.BuildOutput("test-instance", nil)
	if lc := out.Adapters["a8"].LoadCount; lc != 1 {
		t.Errorf("adapter a8 LoadCount = %d, want 1 (charged once; second use warm)", lc)
	}
}

// TestColdLoadGate_LoadsSerializePerInstance is the T027 serialization contract:
// two distinct cold adapters arriving together cannot load concurrently — the
// second request is gated behind the first load (blocking model), so its TTFT is
// strictly greater, and each adapter loads exactly once.
func TestColdLoadGate_LoadsSerializePerInstance(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "a8", Rank: 8}, AdapterSpec{ID: "b8", Rank: 8})

	r1 := newTestRequest("r1", 0, 8, 4)
	r1.Adapter = "a8"
	r2 := newTestRequest("r2", 0, 8, 4)
	r2.Adapter = "b8"

	s := mustNewSimulator(t, cfg)
	s.InjectArrival(r1)
	s.InjectArrival(r2)
	s.Run()

	if !r1.TTFTSet || !r2.TTFTSet {
		t.Fatalf("requests did not both complete")
	}
	// Serialized + blocking: the second-admitted request waited behind the first
	// load, so the two TTFTs cannot be equal.
	if r1.FirstTokenTime == r2.FirstTokenTime {
		t.Errorf("both requests share TTFT %d: loads did not serialize (blocking model)", r1.FirstTokenTime)
	}

	out := s.Metrics.BuildOutput("test-instance", nil)
	for _, id := range []string{"a8", "b8"} {
		if lc := out.Adapters[id].LoadCount; lc != 1 {
			t.Errorf("adapter %s LoadCount = %d, want 1", id, lc)
		}
	}
}

// TestColdLoadGate_SameAdapterCoalesces verifies INV-L3: two requests for the
// SAME cold adapter arriving together share a single load — the adapter is loaded
// exactly once and no second load is charged (same-adapter coalescing, §7). The
// second request finds the adapter resident after the first load completes.
func TestColdLoadGate_SameAdapterCoalesces(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "a8", Rank: 8})

	r1 := newTestRequest("r1", 0, 8, 4)
	r1.Adapter = "a8"
	r2 := newTestRequest("r2", 0, 8, 4) // same adapter, arrives together
	r2.Adapter = "a8"
	s := mustNewSimulator(t, cfg)
	s.InjectArrival(r1)
	s.InjectArrival(r2)
	s.Run()

	if !r1.TTFTSet || !r2.TTFTSet {
		t.Fatalf("requests did not both complete")
	}
	if s.Metrics.CompletedRequests != 2 {
		t.Errorf("CompletedRequests = %d, want 2", s.Metrics.CompletedRequests)
	}
	out := s.Metrics.BuildOutput("test-instance", nil)
	if lc := out.Adapters["a8"].LoadCount; lc != 1 {
		t.Errorf("adapter a8 LoadCount = %d, want 1 (both requests coalesce onto one load, INV-L3)", lc)
	}
	if ev := out.Adapters["a8"].EvictionCount; ev != 0 {
		t.Errorf("adapter a8 EvictionCount = %d, want 0 (single adapter, capacity 2)", ev)
	}
}

// TestAdapterLoadEventPriorityOrdering pins the §7 ordering contract as an
// explicit invariant: an adapter-load completion MUST fire ahead of a co-timed
// step (so the newly-resident adapter's request is batch-eligible that tick) and
// after queueing. A future renumbering that breaks this would silently delay
// gated admissions; this guards against it.
func TestAdapterLoadEventPriorityOrdering(t *testing.T) {
	if PriorityQueued >= PriorityAdapterLoad || PriorityAdapterLoad >= PriorityStep {
		t.Fatalf("priority ordering broken: want Queued(%d) < AdapterLoad(%d) < Step(%d)",
			PriorityQueued, PriorityAdapterLoad, PriorityStep)
	}
}

// --- T029: companion invariant tests ---

// TestColdLoadGate_INV6_NoOpByteIdentity verifies that a run with no adapters
// configured is unaffected by the gate machinery: an adapter-blind workload on a
// non-LoRA config produces the same per-request TTFT as before (INV-6, SC-001).
// (Cross-checked against the golden baseline in the package's no-op test; here we
// assert the gate never perturbs a base-model request even when LoRA is enabled.)
func TestColdLoadGate_INV6_BaseRequestUnaffectedByLoRAConfig(t *testing.T) {
	adapter := AdapterSpec{ID: "a8", Rank: 8}
	// Same base-model request, once on a plain config and once on a LoRA-enabled
	// config: the empty adapter must never gate, so TTFT is identical.
	plain := runSingleTTFT(t, newTestSimConfig(), "")
	loraEnabled := runSingleTTFT(t, gateTestConfig(2, adapter), "")
	if plain != loraEnabled {
		t.Errorf("base-model TTFT differs with LoRA enabled: plain=%d lora=%d (INV-6)", plain, loraEnabled)
	}
}

// TestColdLoadGate_INV5_Causality verifies the gate delays scheduling, not
// arrival/enqueue: for a cold request, the scheduling delay (queued→running)
// absorbs the full load latency, so schedule_time > enqueue_time ≥ arrival_time
// (INV-5). Observed via RequestSchedulingDelays, which is (schedule − arrival).
func TestColdLoadGate_INV5_GateDelaysSchedule(t *testing.T) {
	cfg := gateTestConfig(2, AdapterSpec{ID: "a8", Rank: 8})
	req := newTestRequest("cold", 0, 8, 4)
	req.Adapter = "a8"
	s := mustNewSimulator(t, cfg)
	s.InjectArrival(req)
	s.Run()

	delay, ok := s.Metrics.RequestSchedulingDelays["cold"]
	if !ok {
		t.Fatal("no scheduling delay recorded for cold request")
	}
	// The cold request cannot be scheduled before its adapter finishes loading, so
	// its schedule delay is at least the load latency (INV-5: arrival ≤ schedule).
	if delay < gateLoadLatencyRank8 {
		t.Errorf("cold scheduling delay = %d, want >= %d (adapter load precedes schedule)", delay, gateLoadLatencyRank8)
	}
}

// TestColdLoadGate_INV8_NoDeadlockUnderCapacityPressure verifies the gate never
// strands a request (INV-8 work-conserving, INV-11 completeness): with capacity 1
// and two distinct cold adapters, the second load cannot start until the first
// request completes and unpins its slot (its adapter is the only eviction victim
// and is pinned while in use, INV-L5). The run must still fully drain — both
// requests complete and each adapter loads exactly once.
func TestColdLoadGate_INV8_NoDeadlockUnderCapacityPressure(t *testing.T) {
	cfg := gateTestConfig(1, AdapterSpec{ID: "a8", Rank: 8}, AdapterSpec{ID: "b8", Rank: 8})

	r1 := newTestRequest("r1", 0, 8, 4)
	r1.Adapter = "a8"
	r2 := newTestRequest("r2", 0, 8, 4)
	r2.Adapter = "b8"
	s := mustNewSimulator(t, cfg)
	s.InjectArrival(r1)
	s.InjectArrival(r2)
	s.Run()

	if !r1.TTFTSet || !r2.TTFTSet {
		t.Fatalf("gate stranded a request under capacity pressure: r1.set=%v r2.set=%v", r1.TTFTSet, r2.TTFTSet)
	}
	if s.Metrics.CompletedRequests != 2 {
		t.Errorf("CompletedRequests = %d, want 2 (both drain, INV-11)", s.Metrics.CompletedRequests)
	}
	out := s.Metrics.BuildOutput("test-instance", nil)
	for _, id := range []string{"a8", "b8"} {
		if lc := out.Adapters[id].LoadCount; lc != 1 {
			t.Errorf("adapter %s LoadCount = %d, want 1", id, lc)
		}
	}
	// Capacity 1 with two distinct adapters forces exactly one eviction (the first
	// adapter is evicted to make room for the second once its request completes).
	if ev := out.Adapters["a8"].EvictionCount; ev != 1 {
		t.Errorf("adapter a8 EvictionCount = %d, want 1 (evicted for b8 after r1 completes)", ev)
	}
}

// TestColdLoadGate_INV3_LoadCompletionInFuture verifies determinism and that the
// gate never stalls the simulator (INV-8: a gated request's load is scheduled
// work, so the run always drains). Two identical seeds produce identical TTFT
// (INV-6 determinism, no RNG in the cost path — R7).
func TestColdLoadGate_Determinism(t *testing.T) {
	run := func() int64 {
		return runSingleTTFT(t, gateTestConfig(2, AdapterSpec{ID: "a8", Rank: 8}), "a8")
	}
	if a, b := run(), run(); a != b {
		t.Errorf("cold-load gate not deterministic: run1=%d run2=%d", a, b)
	}
}
