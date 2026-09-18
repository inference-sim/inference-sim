package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterSimulator_AdapterStepOverhead_InflatesLatency_E2E is the PR4 (#1467)
// end-to-end law test: the per-step adapter compute-overhead factor, wired into
// the latency backend at NewInstanceSimulator, must actually lengthen the running
// step and therefore observable request latency through a full cluster run.
//
// It runs the SAME deterministic cluster twice — identical adapters, requests,
// routing, and cold-load coefficients — varying only the step_overhead_tiers K6.
// K6=0 yields factor 1.0 (no-op short-circuit); K6>0 yields factor 1+(K6/K7)·A_B.
// Because step times can only grow under the factor and the cold-load latency is
// identical across both runs, the total completed-request E2E must be STRICTLY
// larger with overhead than without. This isolates the step-overhead effect from
// the (unchanged) cold-load charge and proves the accessor reaches production.
func TestClusterSimulator_AdapterStepOverhead_InflatesLatency_E2E(t *testing.T) {
	sumE2E := func(k6 float64) float64 {
		config := newTestDeploymentConfig(2)
		config.RoutingPolicy = "round-robin"
		capVal := 8
		base, bw, fp := 1000.0, 2.0e6, 2.0e6
		k7 := 1.0
		config.LoRAConfig = sim.LoRAConfig{
			AdapterCapacity:       &capVal,
			LoadBaseLatencyUs:     &base,
			LoadBandwidthBytesUs:  &bw,
			FootprintBytesPerRank: &fp,
			StepOverheadTiers:     map[int]sim.StepOverheadTier{8: {K6: &k6, K7: &k7}},
			Adapters:              []sim.AdapterSpec{{ID: "adapter_shared", Rank: 8}},
		}

		requests := newTestRequests(6)
		for _, r := range requests {
			r.Adapter = "adapter_shared"
		}

		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		mustRun(t, cs)

		var total float64
		for _, e2e := range cs.AggregatedMetrics().RequestE2Es {
			total += e2e
		}
		return total
	}

	noOverhead := sumE2E(0.0)   // factor == 1.0 (short-circuits to base)
	withOverhead := sumE2E(0.5) // factor == 1.5 per step for the single shared adapter

	if noOverhead <= 0 {
		t.Fatalf("baseline run produced no completed-request E2E (%v); test cannot discriminate", noOverhead)
	}
	// Assert a MEANINGFUL increase, not merely withOverhead > noOverhead: a
	// one-tick rounding artifact would satisfy strict-greater without the factor
	// actually reaching the steps. A 50% per-step factor over many decode steps
	// must move total E2E well beyond a rounding boundary; require >= 10% so the
	// test cannot pass trivially.
	if rel := (withOverhead - noOverhead) / noOverhead; rel < 0.10 {
		t.Errorf("step-overhead run total E2E %v vs no-overhead %v = %.1f%% increase, want >= 10%% — factor did not meaningfully reach the running step (#1467)",
			withOverhead, noOverhead, rel*100)
	}
}
