package cluster

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// loraAffinityTestConfig builds a LoRA-enabled multi-instance deployment with a
// small per-instance adapter capacity, so distinct adapters compete for slots and
// routing choices drive the total cold-load count. cost coefficients are required
// once the cold-load gate consumes them (#1466).
func loraAffinityTestConfig(numInstances, capacity int, adapters []string) DeploymentConfig {
	config := newTestDeploymentConfig(numInstances)
	capVal := capacity
	base, bw, fp := 1000.0, 2.0e6, 2.0e6
	specs := make([]sim.AdapterSpec, len(adapters))
	for i, id := range adapters {
		specs[i] = sim.AdapterSpec{ID: id, Rank: 8}
	}
	config.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &capVal,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters:              specs,
	}
	return config
}

// zipfianAdapterRequests builds n deterministic test requests and assigns each a
// LoRA adapter drawn from a fixed Zipfian distribution over adapters (a few
// adapters dominate the traffic — the popularity skew LoRA serving exhibits in
// practice, spec SC-005). The request sequence and the adapter draw are both
// seeded, so two calls with the same arguments produce byte-identical workloads —
// essential for an apples-to-apples routing comparison.
func zipfianAdapterRequests(n int, adapters []string) []*sim.Request {
	reqs := newTestRequests(n)
	z := rand.NewZipf(rand.New(rand.NewSource(7)), 1.3, 1.0, uint64(len(adapters)-1))
	for _, r := range reqs {
		r.Adapter = adapters[z.Uint64()]
	}
	return reqs
}

// totalAdapterLoads sums cold-load counts across all adapters — the metric SC-005
// targets. Fewer total loads means the router kept more requests on instances that
// already held their adapter.
func totalAdapterLoads(m *sim.Metrics) int64 {
	var sum int64
	for _, c := range m.AdapterLoadCounts {
		sum += c
	}
	return sum
}

// TestLoRAAffinity_SkewedPopularity_FewerLoadsThanLeastLoaded is the SC-005
// integration payoff (spec US4 scenario 2, T040): under a skewed adapter-popularity
// workload, weighted routing with the lora-affinity scorer keeps repeat requests on
// the instance that already holds their adapter, so the cluster performs
// substantially fewer total cold loads than adapter-oblivious least-loaded routing.
//
// The two runs share an identical workload (same requests, same adapter draw) and
// differ only in routing policy, isolating the scorer as the cause of the delta.
func TestLoRAAffinity_SkewedPopularity_FewerLoadsThanLeastLoaded(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SC-005 skewed-popularity e2e test in short mode (-short flag)")
	}
	const (
		numInstances = 4
		capacity     = 2 // << number of adapters ⇒ eviction pressure
		numRequests  = 200
	)
	adapters := []string{"a0", "a1", "a2", "a3", "a4", "a5"}

	// Baseline: adapter-oblivious least-loaded routing.
	baseCfg := loraAffinityTestConfig(numInstances, capacity, adapters)
	baseCfg.RoutingPolicy = "least-loaded"
	baseCS := NewClusterSimulator(baseCfg, NewSliceRequestSource(zipfianAdapterRequests(numRequests, adapters)), nil)
	mustRun(t, baseCS)
	baselineLoads := totalAdapterLoads(baseCS.AggregatedMetrics())

	// Treatment: weighted routing with the lora-affinity scorer.
	loraCfg := loraAffinityTestConfig(numInstances, capacity, adapters)
	loraCfg.RoutingPolicy = "weighted"
	loraCfg.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "lora-affinity", Weight: 1.0}}
	loraCS := NewClusterSimulator(loraCfg, NewSliceRequestSource(zipfianAdapterRequests(numRequests, adapters)), nil)
	mustRun(t, loraCS)
	loraLoads := totalAdapterLoads(loraCS.AggregatedMetrics())

	if baselineLoads == 0 || loraLoads == 0 {
		t.Fatalf("expected non-zero cold loads in both runs, got baseline=%d lora=%d", baselineLoads, loraLoads)
	}
	// SC-005 target: at least 30% fewer total adapter loads with the scorer.
	maxAllowed := 0.70 * float64(baselineLoads)
	reduction := 1.0 - float64(loraLoads)/float64(baselineLoads)
	t.Logf("total adapter loads: least-loaded=%d, lora-affinity=%d (%.1f%% fewer)",
		baselineLoads, loraLoads, reduction*100)
	if float64(loraLoads) > maxAllowed {
		t.Errorf("lora-affinity loads=%d exceed SC-005 target (≤%.0f, ≥30%% fewer than baseline=%d); reduction=%.1f%%",
			loraLoads, maxAllowed, baselineLoads, reduction*100)
	}
}
