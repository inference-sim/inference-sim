package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestGenerateRequestsFromDistribution_ZeroRate_Panics(t *testing.T) {
	// GIVEN a ClusterSimulator with workload Rate = 0
	// WHEN generateRequestsFromDistribution is called
	// THEN it panics instead of entering an infinite loop (#202)
	cs := &ClusterSimulator{
		rng:     sim.NewPartitionedRNG(42),
		workload: &sim.GuideLLMConfig{Rate: 0, NumRequests: 10},
		config:  DeploymentConfig{SimConfig: sim.SimConfig{Horizon: 1000}},
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for zero Rate, but did not panic")
		}
	}()
	cs.generateRequestsFromDistribution()
}
