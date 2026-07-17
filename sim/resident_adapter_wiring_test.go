package sim

import (
	"encoding/json"
	"fmt"
	"testing"
)

// TestResidentAdapterSet_CapacityBoundAcrossRun is the T022/T024 integration
// invariant test (SC-003, new invariant resident ≤ capacity). It drives a real
// resident-adapter set (from sim/lora, wired via the blank import in
// lora_import_test.go) through a full simulation where more distinct adapters (M)
// are scheduled than the per-instance slot capacity (N < M), and asserts:
//
//   - the resident set never exceeds capacity (checked at end; guaranteed across
//     all steps by the set's structural bound, unit-proven in sim/lora),
//   - every distinct adapter is cold-loaded exactly once (M loads total),
//   - the overflow is evicted (M − N evictions total),
//
// observed through the exported per-adapter metrics (LoadCount/EvictionCount),
// which is how downstream tooling sees residency behavior.
func TestResidentAdapterSet_CapacityBoundAcrossRun(t *testing.T) {
	const (
		capacity    = 2
		numAdapters = 5 // M > N
	)

	cfg := newTestSimConfig()
	capVal := capacity
	adapters := make([]AdapterSpec, numAdapters)
	for i := range adapters {
		adapters[i] = AdapterSpec{ID: fmt.Sprintf("adapter_%d", i), Rank: 8}
	}
	cfg.LoRAConfig = LoRAConfig{
		AdapterCapacity:       &capVal,
		LoadBaseLatencyUs:     fptrGate(1000.0),
		LoadBandwidthBytesUs:  fptrGate(2.0e6),
		FootprintBytesPerRank: fptrGate(2.0e6),
		Adapters:              adapters,
	}

	sim := mustNewSimulator(t, cfg)
	if sim.residentAdapters == nil {
		t.Fatal("residentAdapters is nil: LoRA subsystem not wired (is sim/lora blank-imported?)")
	}

	// One request per distinct adapter, staggered so scheduling order is unambiguous.
	for i := 0; i < numAdapters; i++ {
		req := newTestRequest(fmt.Sprintf("r%d", i), int64(i), 8, 4)
		req.Adapter = fmt.Sprintf("adapter_%d", i)
		sim.InjectArrival(req)
	}

	sim.Run()

	// The capacity bound holds: with M > N distinct adapters, the set ends full.
	if got := sim.residentAdapters.Len(); got > capacity {
		t.Fatalf("resident set size %d exceeds capacity %d", got, capacity)
	}
	if got := sim.residentAdapters.Len(); got != capacity {
		t.Fatalf("resident set size = %d, want %d (full after M>N distinct adapters)", got, capacity)
	}

	out := sim.Metrics.BuildOutput("test-instance", nil)
	if len(out.Adapters) != numAdapters {
		t.Fatalf("expected %d adapters in metrics, got %d", numAdapters, len(out.Adapters))
	}

	var totalLoads, totalEvictions int64
	for id, am := range out.Adapters {
		if am.LoadCount != 1 {
			t.Errorf("adapter %s: LoadCount = %d, want 1 (each distinct adapter cold-loaded once)", id, am.LoadCount)
		}
		totalLoads += am.LoadCount
		totalEvictions += am.EvictionCount
	}
	if totalLoads != numAdapters {
		t.Errorf("total loads = %d, want %d (M)", totalLoads, numAdapters)
	}
	if want := int64(numAdapters - capacity); totalEvictions != want {
		t.Errorf("total evictions = %d, want %d (M − N)", totalEvictions, want)
	}

	// Eviction attribution: with staggered arrivals the schedule order is r0..r4, so
	// the LRU victims are the first-scheduled adapters (adapter_0/1/2), each evicted
	// once; the two still-resident adapters (adapter_3/4) are never evicted. This
	// pins that evictions are counted against the EVICTED id, not the incoming one.
	for _, id := range []string{"adapter_0", "adapter_1", "adapter_2"} {
		if got := out.Adapters[id].EvictionCount; got != 1 {
			t.Errorf("%s EvictionCount = %d, want 1 (LRU victim)", id, got)
		}
	}
	for _, id := range []string{"adapter_3", "adapter_4"} {
		if got := out.Adapters[id].EvictionCount; got != 0 {
			t.Errorf("%s EvictionCount = %d, want 0 (still resident)", id, got)
		}
	}
}

// TestResidentAdapterSet_MixedBaseAndAdapterTraffic verifies that, on a LoRA-enabled
// instance, base-model requests (empty Adapter) interleaved with adapter requests
// produce no resident-set events: they never enter the resident set, never appear in
// the adapters block, and never perturb the adapter load/eviction counts.
func TestResidentAdapterSet_MixedBaseAndAdapterTraffic(t *testing.T) {
	capVal := 4
	cfg := newTestSimConfig()
	cfg.LoRAConfig = LoRAConfig{
		AdapterCapacity:       &capVal,
		LoadBaseLatencyUs:     fptrGate(1000.0),
		LoadBandwidthBytesUs:  fptrGate(2.0e6),
		FootprintBytesPerRank: fptrGate(2.0e6),
		Adapters:              []AdapterSpec{{ID: "adapter_0", Rank: 8}},
	}
	sim := mustNewSimulator(t, cfg)

	// Interleave: base, adapter_0, base, adapter_0, base.
	adapters := []string{"", "adapter_0", "", "adapter_0", ""}
	for i, a := range adapters {
		req := newTestRequest(fmt.Sprintf("r%d", i), int64(i), 8, 4)
		req.Adapter = a
		sim.InjectArrival(req)
	}
	sim.Run()

	out := sim.Metrics.BuildOutput("test-instance", nil)
	// Only adapter_0 surfaces; the base ("") requests form no adapter entry.
	if _, ok := out.Adapters[""]; ok {
		t.Fatal("base-model (empty adapter) requests must not form an adapter entry")
	}
	if len(out.Adapters) != 1 {
		t.Fatalf("expected exactly 1 adapter entry (adapter_0), got %d: %v", len(out.Adapters), out.Adapters)
	}
	a0 := out.Adapters["adapter_0"]
	if a0.LoadCount != 1 {
		t.Errorf("adapter_0 LoadCount = %d, want 1 (cold-loaded once; second use is warm)", a0.LoadCount)
	}
	if a0.EvictionCount != 0 {
		t.Errorf("adapter_0 EvictionCount = %d, want 0 (capacity 4, single adapter)", a0.EvictionCount)
	}
	if got := sim.residentAdapters.Len(); got != 1 {
		t.Errorf("resident set size = %d, want 1 (only adapter_0; base traffic adds nothing)", got)
	}
}

// TestNewSimulator_InvalidAdapterCapacity verifies NewSimulator returns an error
// (not a panic) when adapters are declared with a non-positive capacity — the
// library-boundary defense-in-depth for a caller that bypasses the CLI's
// LoRAConfig.Validate (R6: library code must not terminate on user input).
func TestNewSimulator_InvalidAdapterCapacity(t *testing.T) {
	for _, capVal := range []int{0, -1} {
		cfg := newTestSimConfig()
		c := capVal
		cfg.LoRAConfig = LoRAConfig{
			AdapterCapacity: &c,
			Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: 8}},
		}
		kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
		latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
		if err != nil {
			t.Fatalf("MustNewLatencyModel: %v", err)
		}
		if _, err := NewSimulator(cfg, kvStore, latencyModel); err == nil {
			t.Fatalf("capacity %d: expected error, got nil", capVal)
		}
	}
}

// TestResidentAdapterSet_InertWhenNoLoRA verifies the no-op default: with no
// LoRA config, the resident set is not constructed and no adapter metrics are
// emitted (INV-6, SC-001). Requests without an adapter run exactly as before.
// TestResidentAdapterSet_ActiveRun_Deterministic (#1470, T045, INV-6/SC-006)
// asserts byte-identical output across two runs of an *active* adapter workload
// (adapters loading and evicting, M > N) with the same seed — deliberately
// including co-arriving requests (same arrival tick) and a repeated adapter to
// exercise simultaneous-touch and same-timestamp tie conditions. LRU ties are
// broken by insertion order; any map-iteration-order nondeterminism leaking into
// output would diverge the two JSON blobs.
func TestResidentAdapterSet_ActiveRun_Deterministic(t *testing.T) {
	const (
		capacity    = 2
		numAdapters = 5 // M > N → active load/evict churn
	)
	build := func() string {
		cfg := newTestSimConfig()
		capVal := capacity
		adapters := make([]AdapterSpec, numAdapters)
		for i := range adapters {
			adapters[i] = AdapterSpec{ID: fmt.Sprintf("adapter_%d", i), Rank: 8}
		}
		cfg.LoRAConfig = LoRAConfig{
			AdapterCapacity:       &capVal,
			LoadBaseLatencyUs:     fptrGate(1000.0),
			LoadBandwidthBytesUs:  fptrGate(2.0e6),
			FootprintBytesPerRank: fptrGate(2.0e6),
			Adapters:              adapters,
		}
		s := mustNewSimulator(t, cfg)
		// Co-arriving requests (same tick 0) + a repeated adapter to hit tie paths.
		arrivals := []struct {
			id      string
			t       int64
			adapter string
		}{
			{"r0", 0, "adapter_0"},
			{"r1", 0, "adapter_1"}, // co-arrival tie with r0
			{"r2", 0, "adapter_0"}, // same adapter, same tick (coalesce/touch tie)
			{"r3", 10, "adapter_2"},
			{"r4", 10, "adapter_3"}, // co-arrival tie
			{"r5", 20, "adapter_4"},
		}
		for _, a := range arrivals {
			req := newTestRequest(a.id, a.t, 8, 4)
			req.Adapter = a.adapter
			s.InjectArrival(req)
		}
		s.Run()
		out := s.Metrics.BuildOutput("test-instance", nil)
		b, err := json.Marshal(out)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}

	first := build()
	second := build()
	if first != second {
		t.Errorf("active-adapter run output not byte-identical across two runs (INV-6/SC-006 determinism)\nfirst:  %s\nsecond: %s", first, second)
	}
}

func TestResidentAdapterSet_InertWhenNoLoRA(t *testing.T) {
	cfg := newTestSimConfig() // no LoRAConfig
	sim := mustNewSimulator(t, cfg)
	if sim.residentAdapters != nil {
		t.Fatal("residentAdapters must be nil when no adapters are configured")
	}

	for i := 0; i < 3; i++ {
		sim.InjectArrival(newTestRequest(fmt.Sprintf("r%d", i), int64(i), 8, 4))
	}
	sim.Run()

	out := sim.Metrics.BuildOutput("test-instance", nil)
	if out.Adapters != nil {
		t.Fatalf("adapter-blind run emitted adapters block: %v", out.Adapters)
	}
}
