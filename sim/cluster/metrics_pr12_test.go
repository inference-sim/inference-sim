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
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m1, m2}, 0)

	// THEN PreemptionRate = 4/20 = 0.2
	expected := 4.0 / 20.0
	if raw.PreemptionRate != expected {
		t.Errorf("PreemptionRate = %f, want %f", raw.PreemptionRate, expected)
	}
}

func TestCollectRawMetrics_IncludesCacheHitRate(t *testing.T) {
	// GIVEN per-instance metrics with cache hit rates
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	m1.CacheHitRate = 0.8
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 10
	m2.CacheHitRate = 0.6

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 20
	aggregated.SimEndedTime = 1000000

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m1, m2}, 0)

	// THEN CacheHitRate = average(0.8, 0.6) = 0.7
	expected := 0.7
	if raw.CacheHitRate != expected {
		t.Errorf("CacheHitRate = %f, want %f", raw.CacheHitRate, expected)
	}
}
