package workload

// Built-in scenario presets for common workload patterns.
// Each returns a valid WorkloadSpec ready for use with GenerateRequests.

// ScenarioBurstyTraffic creates a spec with Gamma-distributed bursty arrivals.
func ScenarioBurstyTraffic(seed int64, rate float64) *WorkloadSpec {
	cv := 3.5
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: rate,
		Clients: []ClientSpec{{
			ID: "bursty-client", TenantID: "tenant-A", SLOClass: "batch",
			RateFraction: 1.0, Arrival: ArrivalSpec{Process: "gamma", CV: &cv},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 512}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 256}},
		}},
	}
}

// ScenarioUnfairTenants creates a spec with 90% low-priority / 10% high-priority traffic.
func ScenarioUnfairTenants(seed int64, rate float64) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: rate,
		Clients: []ClientSpec{
			{ID: "low-priority-bulk", TenantID: "tenant-bulk", SLOClass: "batch",
				RateFraction: 0.9, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 1024}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 512}},
			},
			{ID: "high-priority-realtime", TenantID: "tenant-realtime", SLOClass: "critical",
				RateFraction: 0.1, Streaming: true, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "std_dev": 50, "min": 10, "max": 2048}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 64}},
			},
		},
	}
}

// ScenarioPrefixHeavy creates a spec where 80% of requests share a common prefix.
func ScenarioPrefixHeavy(seed int64, rate float64) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: rate,
		Clients: []ClientSpec{
			{ID: "shared-prefix", TenantID: "tenant-A", SLOClass: "batch",
				RateFraction: 0.8, PrefixGroup: "system-prompt",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 256}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 128}},
			},
			{ID: "unique-prefix", TenantID: "tenant-B", SLOClass: "standard",
				RateFraction: 0.2, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 512}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 256}},
			},
		},
	}
}

// ScenarioMixedSLO creates a spec with equal mix of realtime/interactive/batch.
func ScenarioMixedSLO(seed int64, rate float64) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: rate,
		Clients: []ClientSpec{
			{ID: "realtime", TenantID: "tenant-rt", SLOClass: "critical",
				RateFraction: 0.33, Streaming: true, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 64, "std_dev": 20, "min": 10, "max": 256}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 32}},
			},
			{ID: "interactive", TenantID: "tenant-int", SLOClass: "standard",
				RateFraction: 0.34, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 256, "std_dev": 100, "min": 32, "max": 2048}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 128}},
			},
			{ID: "batch", TenantID: "tenant-batch", SLOClass: "batch",
				RateFraction: 0.33, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 1024}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 512}},
			},
		},
	}
}
