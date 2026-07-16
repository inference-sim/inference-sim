package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterSimulator_AdapterMetrics_AggregatedAcrossInstances_E2E drives a real
// 2-instance cluster end to end: every request targets the same adapter, round-robin
// spreads them across both instances, each instance cold-loads the adapter once (its
// first request; the rest are warm), and the cluster aggregate sums to one load per
// instance. This exercises the full pipeline — scheduling hook populates each
// instance's counts, aggregateMetrics sums them cluster-wide — complementing the
// direct-input unit test below.
func TestClusterSimulator_AdapterMetrics_AggregatedAcrossInstances_E2E(t *testing.T) {
	config := newTestDeploymentConfig(2)
	config.RoutingPolicy = "round-robin"
	capVal := 8 // capacity >> 1 adapter, so no evictions
	config.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity: &capVal,
		Adapters:        []sim.AdapterSpec{{ID: "adapter_shared", Rank: 8}},
	}

	requests := newTestRequests(6)
	for _, r := range requests {
		r.Adapter = "adapter_shared"
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	// One cold load per instance (round-robin gives each instance ≥1 request); the
	// shared adapter's counts must sum across the two instances.
	if got := agg.AdapterLoadCounts["adapter_shared"]; got != 2 {
		t.Errorf("cluster LoadCount[adapter_shared] = %d, want 2 (one cold load per instance)", got)
	}
	if got := agg.AdapterEvictionCounts["adapter_shared"]; got != 0 {
		t.Errorf("cluster EvictionCount[adapter_shared] = %d, want 0 (capacity 8, single adapter)", got)
	}
}

// TestAggregateMetrics_AdapterCountsSummedAcrossInstances verifies that per-adapter
// resident-set load/eviction counts (#1465) are summed across instances during
// cluster aggregation. The same adapter can be resident on multiple instances, so
// its counts must add up cluster-wide rather than being dropped or overwritten
// (unlike globally-unique request ids). This guards the aggregateMetrics gap where
// the new adapter maps were initially not merged.
func TestAggregateMetrics_AdapterCountsSummedAcrossInstances(t *testing.T) {
	config := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)

	instances := cs.Instances()
	if len(instances) != 2 {
		t.Fatalf("expected 2 instances, got %d", len(instances))
	}

	// Simulate per-instance residency events. adapter_shared is resident on both
	// instances (its counts must sum); adapter_a / adapter_b are instance-local.
	m0 := instances[0].Metrics()
	m0.AdapterLoadCounts["adapter_shared"] = 3
	m0.AdapterLoadCounts["adapter_a"] = 1
	m0.AdapterEvictionCounts["adapter_shared"] = 1

	m1 := instances[1].Metrics()
	m1.AdapterLoadCounts["adapter_shared"] = 2
	m1.AdapterLoadCounts["adapter_b"] = 5
	m1.AdapterEvictionCounts["adapter_b"] = 2

	agg := cs.aggregateMetrics()

	wantLoads := map[string]int64{"adapter_shared": 5, "adapter_a": 1, "adapter_b": 5}
	for id, want := range wantLoads {
		if got := agg.AdapterLoadCounts[id]; got != want {
			t.Errorf("aggregated LoadCount[%s] = %d, want %d", id, got, want)
		}
	}
	wantEvictions := map[string]int64{"adapter_shared": 1, "adapter_b": 2}
	for id, want := range wantEvictions {
		if got := agg.AdapterEvictionCounts[id]; got != want {
			t.Errorf("aggregated EvictionCount[%s] = %d, want %d", id, got, want)
		}
	}
	if got := agg.AdapterEvictionCounts["adapter_a"]; got != 0 {
		t.Errorf("adapter_a had no evictions, aggregated EvictionCount = %d, want 0", got)
	}
}
