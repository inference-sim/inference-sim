package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestCollectRawMetrics_IncludesPreemptionRate(t *testing.T) {
	// GIVEN per-instance metrics with preemption counts
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	m1.PreemptionCount = 3
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 10
	m2.PreemptionCount = 1

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 20
	aggregated.SimEndedTime = 1000000

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m1, m2}, 0, "", 0, 0, nil)

	// THEN PreemptionRate = 4/20 = 0.2
	expected := 4.0 / 20.0
	if raw.PreemptionRate != expected {
		t.Errorf("PreemptionRate = %f, want %f", raw.PreemptionRate, expected)
	}
}

func TestCollectRawMetrics_IncludesCacheHitRate(t *testing.T) {
	// GIVEN per-instance rates of 0.8 and 0.6, and an aggregate whose lookup-weighted rate
	// is 0.75 (the busier instance dominates)
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	m1.CacheHitRate = 0.8
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 10
	m2.CacheHitRate = 0.6

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 20
	aggregated.SimEndedTime = 1000000
	aggregated.CacheHitRate = 0.75

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m1, m2}, 0, "", 0, 0, nil)

	// THEN the aggregate's weighted rate is reported, not the unweighted instance mean (#1904)
	if raw.CacheHitRate != 0.75 {
		t.Errorf("CacheHitRate = %f, want the aggregate's 0.75 (not the instance mean 0.7)", raw.CacheHitRate)
	}
}

// The law #1904 states: replicating a one-instance run across N instances, with N-1 of them
// idle, leaves the cluster hit rate unchanged. Prefix affinity alone keeps every request on
// the instance that holds the shared prefix (pinnedGateway stands in for it), so the busy instance sees the one-instance
// run's exact lookups; an unweighted mean would report a quarter of it.
func TestAggregateCacheHitRate_IdleInstancesDoNotDiluteIt(t *testing.T) {
	requests := makeSharedPrefixRequests(30, 1.0, 256, 64, 16, 100_000)
	run := func(n int) (*ClusterSimulator, *sim.Metrics) {
		cs := newTestCluster(baseDeploymentConfig(n), NewSliceRequestSource(copyRequests(requests)), nil)
		cs.SetGateway(pinnedGateway())
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		return cs, cs.AggregatedMetrics()
	}
	_, one := run(1)
	cs, four := run(4)
	if busiest := maxValue(getRoutingDistribution(cs)); busiest != len(requests) {
		t.Fatalf("prefix affinity spread the requests (busiest instance served %d of %d); "+
			"the law's premise does not hold", busiest, len(requests))
	}
	if one.CacheHitRate <= 0 {
		t.Fatal("the one-instance run hit no prefix; the comparison would be vacuous")
	}
	if four.CacheHitRate != one.CacheHitRate {
		t.Errorf("cluster cache_hit_rate with 3 idle instances = %v, want the one-instance %v",
			four.CacheHitRate, one.CacheHitRate)
	}
	if four.PrefixCacheQueryTokens != one.PrefixCacheQueryTokens {
		t.Errorf("lookup tokens %d, want %d: idle instances must add none",
			four.PrefixCacheQueryTokens, one.PrefixCacheQueryTokens)
	}
}

// The cluster rate is the instances' summed hit tokens over their summed lookup tokens.
func TestAggregateCacheHitRate_IsTokenWeighted(t *testing.T) {
	requests := makeSharedPrefixRequests(30, 1.0, 256, 64, 16, 100_000)
	cs := newTestCluster(baseDeploymentConfig(2), NewSliceRequestSource(copyRequests(requests)), nil)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	var hits, lookups int64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		hits += m.PrefixCacheHitTokens
		lookups += m.PrefixCacheQueryTokens
	}
	if lookups == 0 {
		t.Fatal("no lookups recorded")
	}
	agg := cs.AggregatedMetrics()
	if want := float64(hits) / float64(lookups); agg.CacheHitRate != want {
		t.Errorf("cluster cache_hit_rate = %v, want summed hits/lookups %v", agg.CacheHitRate, want)
	}
}

// #1904's P/D case: decode instances never look a prefix up (a decode sub-request arrives
// with its KV), so they add no queries, the cluster rate is the prefill pool's, and adding
// decode instances leaves it unchanged. The store's own tally would count each decode
// block allocation as a miss and dilute it.
func TestAggregateCacheHitRate_TheDecodePoolDoesNotDiluteIt(t *testing.T) {
	requests := makeSharedPrefixRequests(30, 1.0, 256, 64, 16, 20_000)
	run := func(decode int) (*ClusterSimulator, float64) {
		cs := newTestCluster(newTestDisaggDeploymentConfig(2+decode, 2, decode),
			NewSliceRequestSource(copyRequests(requests)), nil)
		mustRun(t, cs)
		return cs, cs.AggregatedMetrics().CacheHitRate
	}
	cs, one := run(1)
	var hits, queries int64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		if cs.PoolMembership()[string(inst.ID())].Has(PoolRolePrefill) {
			hits += m.PrefixCacheHitTokens
			queries += m.PrefixCacheQueryTokens
		} else if m.PrefixCacheQueryTokens != 0 {
			t.Errorf("decode instance %s recorded %d query tokens", inst.ID(), m.PrefixCacheQueryTokens)
		}
	}
	if queries == 0 || hits == 0 {
		t.Fatal("the prefill pool hit nothing; the comparison would be vacuous")
	}
	if want := float64(hits) / float64(queries); one != want {
		t.Errorf("cluster rate %v, want the prefill pool's %v", one, want)
	}
	if _, three := run(3); three != one {
		t.Errorf("three decode instances moved the cluster rate from %v to %v", one, three)
	}
}
