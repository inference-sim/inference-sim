package cluster

import (
	"fmt"
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
)

// INV-M1 over whole cluster runs: every measured request a colocated cluster was sent ends
// in exactly one of measured_completed_requests, cancelled_requests (timed out, or still in
// flight at the horizon), context_overflow_requests or failed_requests, whatever mix of
// phases, deadlines, oversized prompts, admission rejections and horizons the run has.
func TestMeasuredAccountingClosesOverClusterRuns(t *testing.T) {
	rejectedSeen := false
	defer func() {
		if !rejectedSeen {
			t.Error("no draw rejected a request at admission; the failed_requests path went unexercised")
		}
	}()
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 30).Draw(rt, "n")
		var requests []*sim.Request
		measured := 0
		for i := 0; i < n; i++ {
			in := rapid.IntRange(16, 600).Draw(rt, "input") // > 400 overflows MaxModelLen
			out := rapid.IntRange(1, 40).Draw(rt, "output")
			r := &sim.Request{
				ID:           fmt.Sprintf("request_%d", i),
				InputTokens:  make([]sim.TokenID, in),
				OutputTokens: make([]sim.TokenID, out),
				State:        sim.StateQueued,
				ArrivalTime:  int64(i) * rapid.Int64Range(100, 20_000).Draw(rt, "gap"),
				Phase:        rapid.SampledFrom([]sim.Phase{sim.PhaseUnset, sim.PhaseWarmup, sim.PhaseProfile, sim.PhaseGrace}).Draw(rt, "phase"),
			}
			for j := range r.InputTokens {
				r.InputTokens[j] = sim.TokenID(i*1000 + j)
			}
			if rapid.Bool().Draw(rt, "deadline") {
				r.Deadline = r.ArrivalTime + rapid.Int64Range(1, 50_000).Draw(rt, "timeout")
			}
			requests = append(requests, r)
		}
		config := baseDeploymentConfig(rapid.IntRange(1, 3).Draw(rt, "instances"))
		config.Horizon = rapid.Int64Range(10_000, 2_000_000).Draw(rt, "horizon")
		for _, r := range requests {
			if r.Phase.Measured() && r.ArrivalTime <= config.Horizon {
				measured++ // a request due after the horizon is never sent
			}
		}
		config.ModelHardwareConfig = sim.NewModelHardwareConfig(testModelConfig(), "test-model", "", 1, 1, false, 400)
		if rapid.Bool().Draw(rt, "tokenBucket") { // a small bucket rejects at admission
			config.AdmissionPolicy = "token-bucket"
			config.TokenBucketCapacity = float64(rapid.IntRange(100, 3000).Draw(rt, "bucket"))
			config.TokenBucketRefillRate = 1000
		}
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		if err := cs.Run(); err != nil {
			rt.Fatal(err)
		}
		mm := cs.AggregatedMetrics().Measured()
		if cs.RejectedRequests() > 0 {
			rejectedSeen = true
		}
		if got := mm.MeasuredCompletedRequests + mm.CancelledRequests + mm.ContextOverflowRequests +
			mm.FailedRequests; got != measured {
			rt.Fatalf("INV-M1: completed+cancelled+context_overflow+failed = %d, %d measured requests sent", got, measured)
		}
	})
}

// INV-M3 over cluster runs: the cluster hit rate lies within the per-instance rates of the
// instances that looked anything up, and is their summed hits over summed lookups.
func TestAggregateCacheHitRateLiesWithinTheInstanceRates(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		requests := makeSharedPrefixRequests(rapid.IntRange(2, 40).Draw(rt, "n"),
			float64(rapid.IntRange(0, 10).Draw(rt, "shared"))/10, 16*rapid.IntRange(1, 16).Draw(rt, "prefixBlocks"),
			rapid.IntRange(1, 64).Draw(rt, "suffix"), 4, rapid.Int64Range(100, 50_000).Draw(rt, "gap"))
		config := baseDeploymentConfig(rapid.IntRange(1, 4).Draw(rt, "instances"))
		config.RoutingPolicy = rapid.SampledFrom([]string{"round-robin", "weighted"}).Draw(rt, "routing")
		if config.RoutingPolicy == "weighted" {
			config.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "prefix-affinity", Weight: 1}, {Name: "queue-depth", Weight: 1}}
		}
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		if err := cs.Run(); err != nil {
			rt.Fatal(err)
		}
		lo, hi, seen := 1.0, 0.0, false
		for _, inst := range cs.Instances() {
			m := inst.Metrics()
			if m.PrefixCacheLookupTokens == 0 {
				continue
			}
			rate := float64(m.PrefixCacheHitTokens) / float64(m.PrefixCacheLookupTokens)
			lo, hi, seen = min(lo, rate), max(hi, rate), true
		}
		agg := cs.AggregatedMetrics().CacheHitRate
		if seen && (agg < lo-1e-12 || agg > hi+1e-12) {
			rt.Fatalf("INV-M3: cluster hit rate %v outside the instance rates [%v, %v]", agg, lo, hi)
		}
	})
}

// Metamorphic (#1904): adding idle instances leaves the cluster hit rate unchanged. Prefix
// affinity alone keeps a fully shared-prefix workload on the instance holding the prefix,
// so every added instance is idle.
func TestAggregateCacheHitRate_AddingIdleInstancesChangesNothing(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		requests := makeSharedPrefixRequests(rapid.IntRange(2, 30).Draw(rt, "n"), 1.0,
			16*rapid.IntRange(1, 16).Draw(rt, "prefixBlocks"), rapid.IntRange(1, 64).Draw(rt, "suffix"),
			8, rapid.Int64Range(50_000, 200_000).Draw(rt, "gap"))
		run := func(n int) (*ClusterSimulator, float64) {
			config := baseDeploymentConfig(n)
			config.RoutingPolicy = "weighted"
			config.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "prefix-affinity", Weight: 1}}
			cs := NewClusterSimulator(config, NewSliceRequestSource(copyRequests(requests)), nil)
			if err := cs.Run(); err != nil {
				rt.Fatal(err)
			}
			return cs, cs.AggregatedMetrics().CacheHitRate
		}
		_, one := run(1)
		cs, more := run(1 + rapid.IntRange(1, 4).Draw(rt, "idle"))
		if busiest := maxValue(getRoutingDistribution(cs)); busiest != len(requests) {
			rt.Fatalf("premise: prefix affinity spread the workload (%d of %d on the busiest)", busiest, len(requests))
		}
		if more != one {
			rt.Fatalf("#1904: idle instances moved the cluster hit rate from %v to %v", one, more)
		}
	})
}
